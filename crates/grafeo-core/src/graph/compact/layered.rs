//! Two-layer graph store: read-only columnar base + mutable LPG overlay.
//!
//! `LayeredStore` coordinates reads between a [`CompactStore`](crate::graph::compact::CompactStore) (cold, columnar)
//! and an [`LpgStore`](crate::graph::lpg::LpgStore) (hot, HashMap-based). All writes go to the overlay.
//! Reads check the overlay first and fall through to the compact base for
//! unmodified entities.
//!
//! Requires both `compact-store` and `lpg` features.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arc_swap::ArcSwap;
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;

use super::CompactStore;
use crate::graph::Direction;
use crate::graph::lpg::{CompareOp, Edge, LpgStore, Node};
use crate::graph::traits::{GraphStore, GraphStoreMut, GraphStoreSearch};
#[cfg(feature = "vector-index")]
use crate::index::vector::DistanceMetric;
use crate::statistics::Statistics;

/// A two-layer graph store with a columnar base and mutable overlay.
///
/// The compact base serves cold reads (immutable, columnar). The LPG overlay
/// captures all mutations. Reads check the overlay first: if an entity is in
/// `dirty_node_ids` or `dirty_edge_ids`, the overlay is authoritative. If an
/// entity is in `deleted_from_base_nodes` or `deleted_from_base_edges`, it has
/// been deleted and returns `None`. Otherwise, the base is queried.
pub struct LayeredStore {
    /// Read-only columnar base (cold data).
    ///
    /// Held via [`ArcSwap`] so the engine can atomically swap the underlying
    /// `Arc<CompactStore>` when the base is spilled to a mmap'd file. Readers
    /// acquire the current Arc via `self.base.load()` without locking;
    /// [`swap_base`](Self::swap_base) publishes a new base in a single
    /// `store()` call.
    base: ArcSwap<CompactStore>,
    /// Mutable overlay for new and modified data.
    ///
    /// Held via [`ArcSwap`] (Phase 5c) so the engine can atomically
    /// replace the overlay with a fresh empty `LpgStore` after a
    /// `merge_overlay_in_place` call: existing readers continue holding
    /// the old `Arc` until they finish, while subsequent reads pick up
    /// the empty overlay.
    overlay: ArcSwap<LpgStore>,
    /// Node IDs modified or created in the overlay.
    dirty_node_ids: RwLock<FxHashSet<NodeId>>,
    /// Edge IDs modified or created in the overlay.
    dirty_edge_ids: RwLock<FxHashSet<EdgeId>>,
    /// Base node IDs that have been deleted.
    deleted_from_base_nodes: RwLock<FxHashSet<NodeId>>,
    /// Base edge IDs that have been deleted.
    deleted_from_base_edges: RwLock<FxHashSet<EdgeId>>,
    /// Tracks whether `deleted_from_base_*` has changed since the last
    /// flush. The `OverlayDeletionsSection` checks this on each
    /// `is_dirty()` call so periodic checkpoints skip the write when no
    /// new deletions have accumulated.
    deletions_dirty: AtomicBool,
    /// Merge serialization guard (Phase 5d).
    ///
    /// Mutations acquire `read()` for the duration of a single
    /// operation; `merge_overlay_in_place` acquires `write()` to
    /// stop-the-world during the rebuild + base swap + overlay reset.
    /// Prevents the race where concurrent writes land on an overlay
    /// that's about to be cleared, losing those writes.
    ///
    /// Pure read paths (get_node, nodes_by_label, etc.) do NOT acquire
    /// this lock — they use `ArcSwap` snapshot semantics on base and
    /// overlay separately, which already provides a consistent view.
    merge_guard: RwLock<()>,
}

impl std::fmt::Debug for LayeredStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredStore")
            .field("base_node_count", &self.base.load().node_count())
            .field("overlay_node_count", &self.overlay.load().node_count())
            .field("dirty_nodes", &self.dirty_node_ids.read().len())
            .field(
                "deleted_base_nodes",
                &self.deleted_from_base_nodes.read().len(),
            )
            .finish_non_exhaustive()
    }
}

impl LayeredStore {
    /// Creates a layered store from a compact base.
    ///
    /// The `max_node_id` and `max_edge_id` values seed the overlay's ID
    /// allocator so new entities never collide with base IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the overlay `LpgStore` cannot be created.
    pub fn new(
        base: CompactStore,
        max_node_id: u64,
        max_edge_id: u64,
    ) -> Result<Self, grafeo_common::memory::AllocError> {
        let overlay = Arc::new(LpgStore::new()?);
        overlay.set_next_node_id(max_node_id + 1);
        overlay.set_next_edge_id(max_edge_id + 1);
        Ok(Self::from_parts(Arc::new(base), overlay))
    }

    /// Phase 5e: builds a `LayeredStore` adopting an existing
    /// `Arc<LpgStore>` as the overlay rather than allocating a fresh
    /// one. Used by the open path when reloading a previously-compacted
    /// database whose overlay data was just deserialized into the engine's
    /// `LpgStore`.
    ///
    /// Scans the overlay against the base to reconstruct
    /// `dirty_node_ids` / `dirty_edge_ids`: any id that exists in both
    /// layers is a base modification whose overlay copy must take
    /// precedence in `get_node` / `get_edge`. Without this reseed,
    /// `get_node` would route through the dirty-check fast path,
    /// observe an empty set, and return the stale base version of any
    /// modified base node.
    ///
    /// **Note on deletions:** `deleted_from_base_nodes` /
    /// `deleted_from_base_edges` are NOT reconstructable from the
    /// in-memory state alone — a base node that was deleted simply has
    /// no overlay entry, so the scan can't tell it apart from a base
    /// node that was never touched. The deletion log is persisted in
    /// the [`OverlayDeletions`](grafeo_common::storage::section::SectionType::OverlayDeletions)
    /// section; callers should follow `with_overlay` with a call to
    /// [`seed_deleted_from_base`](Self::seed_deleted_from_base) carrying
    /// the snapshot read from that section, when one is present in the
    /// container directory.
    ///
    /// The overlay's id allocator state is preserved as-is; callers
    /// should ensure it has been seeded correctly during deserialization.
    #[must_use]
    pub fn with_overlay(base: Arc<CompactStore>, overlay: Arc<LpgStore>) -> Self {
        let mut dirty_nodes: FxHashSet<NodeId> = FxHashSet::default();
        for nid in overlay.all_node_ids() {
            if base.get_node(nid).is_some() {
                dirty_nodes.insert(nid);
            }
        }
        let mut dirty_edges: FxHashSet<EdgeId> = FxHashSet::default();
        for edge in overlay.all_edges() {
            if base.get_edge(edge.id).is_some() {
                dirty_edges.insert(edge.id);
            }
        }
        Self {
            base: ArcSwap::new(base),
            overlay: ArcSwap::new(overlay),
            dirty_node_ids: RwLock::new(dirty_nodes),
            dirty_edge_ids: RwLock::new(dirty_edges),
            deleted_from_base_nodes: RwLock::new(FxHashSet::default()),
            deleted_from_base_edges: RwLock::new(FxHashSet::default()),
            deletions_dirty: AtomicBool::new(false),
            merge_guard: RwLock::new(()),
        }
    }

    fn from_parts(base: Arc<CompactStore>, overlay: Arc<LpgStore>) -> Self {
        Self {
            base: ArcSwap::new(base),
            overlay: ArcSwap::new(overlay),
            dirty_node_ids: RwLock::new(FxHashSet::default()),
            dirty_edge_ids: RwLock::new(FxHashSet::default()),
            deleted_from_base_nodes: RwLock::new(FxHashSet::default()),
            deleted_from_base_edges: RwLock::new(FxHashSet::default()),
            deletions_dirty: AtomicBool::new(false),
            merge_guard: RwLock::new(()),
        }
    }

    /// Returns a shared reference to the compact base store.
    #[must_use]
    pub fn base_store_arc(&self) -> Arc<CompactStore> {
        self.base.load_full()
    }

    /// Atomically replaces the compact base store.
    ///
    /// The engine calls this after spilling the base to a mmap'd file: the
    /// old in-memory `Arc<CompactStore>` drops (freeing heap memory once the
    /// last external reference is released), and subsequent reads go through
    /// the new mmap-backed `CompactStore`. Overlay state (dirty sets,
    /// deleted sets, the overlay `LpgStore`) is untouched.
    ///
    /// Returns the previous base `Arc`, so callers can inspect refcounts or
    /// keep it alive while in-flight readers drain.
    pub fn swap_base(&self, new_base: Arc<CompactStore>) -> Arc<CompactStore> {
        self.base.swap(new_base)
    }

    /// Returns the current overlay LPG store as an owned `Arc`.
    ///
    /// Phase 5c: the overlay is now wrapped in an `ArcSwap` so it can
    /// be atomically replaced after a merge. Callers receive a snapshot
    /// `Arc` that remains valid even if the overlay is later swapped.
    #[must_use]
    pub fn overlay_store(&self) -> Arc<LpgStore> {
        self.overlay.load_full()
    }

    /// Number of dirty (modified/created) entities in the overlay.
    #[must_use]
    pub fn overlay_mutation_count(&self) -> usize {
        self.dirty_node_ids.read().len()
            + self.dirty_edge_ids.read().len()
            + self.deleted_from_base_nodes.read().len()
            + self.deleted_from_base_edges.read().len()
    }

    /// Approximate heap bytes of the overlay only (excluding base).
    ///
    /// Used by `OverlayConsumer` (Phase 5c) to drive merge-on-pressure
    /// without conflating overlay growth with base size.
    #[must_use]
    pub fn overlay_memory_bytes(&self) -> usize {
        let (store_mem, index_mem, mvcc_mem, pool_mem) = self.overlay.load().memory_breakdown();
        store_mem.total_bytes + index_mem.total_bytes + mvcc_mem.total_bytes + pool_mem.total_bytes
    }

    /// Approximate heap memory of both layers.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.base.load().memory_bytes() + self.overlay_memory_bytes()
    }

    /// Replaces the overlay with a fresh empty `LpgStore` and clears
    /// dirty/deleted bookkeeping (Phase 5c).
    ///
    /// Atomic: in-flight readers holding an `Arc<LpgStore>` snapshot
    /// continue against the old overlay; subsequent reads pick up the
    /// fresh empty one.
    ///
    /// The new overlay's id allocators are seeded from the *current*
    /// base so freshly created nodes/edges don't collide with base ids.
    ///
    /// # Panics
    ///
    /// Panics only if the system allocator fails to provide a fresh
    /// `LpgStore`. The same allocator is used everywhere else in the
    /// store so this is a fatal condition rather than a recoverable
    /// error.
    pub fn reset_overlay(&self) {
        let fresh = Arc::new(LpgStore::new().expect("LpgStore allocation"));
        // Seed allocators from the base so new ids don't collide.
        let base = self.base.load();
        let max_nid = base
            .all_node_ids()
            .into_iter()
            .map(|id| id.as_u64())
            .max()
            .unwrap_or(0);
        // Edge ids are not directly enumerable from CompactStore; use
        // the current overlay's allocator as a conservative lower bound.
        let current_overlay = self.overlay.load();
        let max_eid = current_overlay.next_edge_id().saturating_sub(1);
        fresh.set_next_node_id(max_nid + 1);
        fresh.set_next_edge_id(max_eid + 1);

        self.overlay.store(fresh);
        self.dirty_node_ids.write().clear();
        self.dirty_edge_ids.write().clear();
        self.deleted_from_base_nodes.write().clear();
        self.deleted_from_base_edges.write().clear();
        self.deletions_dirty.store(false, Ordering::Release);
    }

    /// Returns a snapshot of the base node ids the overlay has marked as
    /// deleted but not yet merged. Used by the persistence layer to write
    /// the [`OverlayDeletions`](grafeo_common::storage::section::SectionType::OverlayDeletions)
    /// section so the deletions survive close/reopen cycles.
    #[must_use]
    pub fn snapshot_deleted_node_ids(&self) -> Vec<NodeId> {
        self.deleted_from_base_nodes
            .read()
            .iter()
            .copied()
            .collect()
    }

    /// Snapshot of base edge ids deleted-but-not-merged. See
    /// [`Self::snapshot_deleted_node_ids`].
    #[must_use]
    pub fn snapshot_deleted_edge_ids(&self) -> Vec<EdgeId> {
        self.deleted_from_base_edges
            .read()
            .iter()
            .copied()
            .collect()
    }

    /// Seeds the deleted-from-base sets from a previously-persisted
    /// snapshot (typically the `OverlayDeletions` section). The current
    /// sets are replaced atomically; any in-memory deletions accumulated
    /// before the seed are dropped (callers should only seed during
    /// open, before any new mutations are accepted).
    ///
    /// Clears the deletions-dirty flag so the next checkpoint does not
    /// re-write the section just because the seed populated it.
    pub fn seed_deleted_from_base(
        &self,
        nodes: impl IntoIterator<Item = NodeId>,
        edges: impl IntoIterator<Item = EdgeId>,
    ) {
        let mut node_set = self.deleted_from_base_nodes.write();
        node_set.clear();
        node_set.extend(nodes);
        let mut edge_set = self.deleted_from_base_edges.write();
        edge_set.clear();
        edge_set.extend(edges);
        self.deletions_dirty.store(false, Ordering::Release);
    }

    /// Whether the deletion log has changed since the last
    /// [`mark_deletions_clean`](Self::mark_deletions_clean) call. Used by
    /// the `OverlayDeletionsSection` to decide whether a periodic
    /// checkpoint should re-emit the section.
    #[must_use]
    pub fn deletions_dirty(&self) -> bool {
        self.deletions_dirty.load(Ordering::Acquire)
    }

    /// Marks the deletion log as clean. Called by the flush path after a
    /// successful write of the `OverlayDeletions` section.
    pub fn mark_deletions_clean(&self) {
        self.deletions_dirty.store(false, Ordering::Release);
    }

    /// Merges the overlay into a fresh `CompactStore`, swaps it in as
    /// the base, and clears the overlay (Phase 5c).
    ///
    /// After this call: all previously-visible data is in the base; the
    /// overlay is empty. Used by `OverlayConsumer` to release overlay
    /// memory under pressure.
    ///
    /// # Errors
    ///
    /// Returns an error if rebuilding the base fails.
    pub fn merge_overlay_in_place(&self) -> Result<(), String> {
        // Stop-the-world: writers block until the rebuild is published.
        // Read paths are unaffected (they don't take this lock).
        let _guard = self.merge_guard.write();

        // Read the combined view (self IS the layered GraphStore).
        let fresh_compact =
            super::from_graph_store_preserving_ids(self).map_err(|e| e.to_string())?;

        // Swap in the new base.
        self.base.swap(Arc::new(fresh_compact));

        // Reset the overlay (already seeds id allocators from the new base).
        self.reset_overlay();
        Ok(())
    }

    /// Checks whether a node ID is in the overlay (dirty or deleted).
    #[inline]
    fn is_node_dirty(&self, id: NodeId) -> bool {
        self.dirty_node_ids.read().contains(&id)
    }

    /// Checks whether a node was deleted from the base.
    #[inline]
    fn is_node_deleted_from_base(&self, id: NodeId) -> bool {
        self.deleted_from_base_nodes.read().contains(&id)
    }

    /// Checks whether an edge ID is in the overlay (dirty or deleted).
    #[inline]
    fn is_edge_dirty(&self, id: EdgeId) -> bool {
        self.dirty_edge_ids.read().contains(&id)
    }

    /// Checks whether an edge was deleted from the base.
    #[inline]
    fn is_edge_deleted_from_base(&self, id: EdgeId) -> bool {
        self.deleted_from_base_edges.read().contains(&id)
    }
}

// ── GraphStore implementation ──────────────────────────────────────

impl GraphStore for LayeredStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.load().get_node(id);
        }
        // dirty_node_ids only tracks modified base nodes; new overlay nodes fall through here.
        self.base
            .load()
            .get_node(id)
            .or_else(|| self.overlay.load().get_node(id))
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.load().get_edge(id);
        }
        // Edges created after `compact()` live only in the overlay; fall
        // through when the base doesn't recognise the id.
        self.base
            .load()
            .get_edge(id)
            .or_else(|| self.overlay.load().get_edge(id))
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self
                .overlay
                .load()
                .get_node_versioned(id, epoch, transaction_id);
        }
        // `dirty_node_ids` only tracks overlay modifications of *base* nodes.
        // Overlay-only nodes (post-`compact()` writes) fall through to here;
        // the base doesn't know them, so defer to the overlay's versioned
        // fetch. CompactStore itself has no MVCC versions, so `get_node`
        // is the right base call.
        self.base.load().get_node(id).or_else(|| {
            self.overlay
                .load()
                .get_node_versioned(id, epoch, transaction_id)
        })
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self
                .overlay
                .load()
                .get_edge_versioned(id, epoch, transaction_id);
        }
        self.base.load().get_edge(id).or_else(|| {
            self.overlay
                .load()
                .get_edge_versioned(id, epoch, transaction_id)
        })
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.load().get_node_at_epoch(id, epoch);
        }
        self.base
            .load()
            .get_node(id)
            .or_else(|| self.overlay.load().get_node_at_epoch(id, epoch))
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.load().get_edge_at_epoch(id, epoch);
        }
        self.base
            .load()
            .get_edge(id)
            .or_else(|| self.overlay.load().get_edge_at_epoch(id, epoch))
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.load().get_node_property(id, key);
        }
        self.base
            .load()
            .get_node_property(id, key)
            .or_else(|| self.overlay.load().get_node_property(id, key))
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.load().get_edge_property(id, key);
        }
        self.base
            .load()
            .get_edge_property(id, key)
            .or_else(|| self.overlay.load().get_edge_property(id, key))
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter()
            .map(|id| self.get_node_property(*id, key))
            .collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                self.get_node(*id)
                    .map(|n| {
                        n.properties
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_node_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_edge_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        let deleted_nodes = self.deleted_from_base_nodes.read();

        let mut results = Vec::new();

        // Base neighbors (minus deleted). `is_node_dirty` is not a reason
        // to skip the base: `ensure_in_overlay` promotes a node's labels
        // and properties but never copies its adjacency, so base edges
        // remain authoritative for any dirty node that originated in the
        // snapshot. Per-edge deletions are still respected via
        // `deleted_from_base_edges` inside `edges_from`-style call sites,
        // and `dedup` below collapses any overlap with promoted overlay
        // adjacency.
        if !deleted_nodes.contains(&node) {
            for nid in self.base.load().neighbors(node, direction) {
                if !deleted_nodes.contains(&nid) {
                    results.push(nid);
                }
            }
        }

        // Overlay neighbors — always consulted. An edge created after
        // `compact()` whose src is a base node records the base id in
        // the overlay's adjacency even though the overlay has no
        // corresponding node object; gating on `overlay.get_node(node)`
        // would miss that case.
        for nid in self.overlay.load().neighbors(node, direction) {
            if !deleted_nodes.contains(&nid) {
                results.push(nid);
            }
        }

        results.sort_unstable();
        results.dedup();
        results
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let deleted_nodes = self.deleted_from_base_nodes.read();
        let deleted_edges = self.deleted_from_base_edges.read();

        let mut results = Vec::new();

        // Base edges (minus deleted). The base layer must be consulted
        // even when the source node is dirty: `ensure_in_overlay` copies
        // labels and properties into the overlay but leaves adjacency in
        // the base, so a property write — or merely being the endpoint
        // of a freshly created overlay edge — must not erase the node's
        // pre-existing snapshot edges. Promoted edges (those in both
        // tiers because their properties were modified) live at the same
        // `EdgeId` in base and overlay and are folded together by the
        // dedup-by-eid pass below.
        if !deleted_nodes.contains(&node) {
            for (target, eid) in self.base.load().edges_from(node, direction) {
                if !deleted_nodes.contains(&target) && !deleted_edges.contains(&eid) {
                    results.push((target, eid));
                }
            }
        }

        // Overlay edges — always consulted. The overlay stores edges
        // keyed by src/dst even when the endpoint is a base node (e.g. a
        // post-`compact()` edge from a base node to an overlay node), so
        // we can't gate this on whether the overlay has the node itself.
        // `LpgStore::edges_from` returns empty for ids with no outgoing
        // edges, so the unconditional call is cheap when there's nothing
        // to report.
        for (target, eid) in self.overlay.load().edges_from(node, direction) {
            if !deleted_nodes.contains(&target) && !deleted_edges.contains(&eid) {
                results.push((target, eid));
            }
        }

        // Deduplicate in case a promoted edge appears in both layers.
        results.sort_unstable_by_key(|&(_, eid)| eid);
        results.dedup_by_key(|&mut (_, eid)| eid);

        results
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Outgoing).len()
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Incoming).len()
    }

    fn has_backward_adjacency(&self) -> bool {
        self.base.load().has_backward_adjacency() || self.overlay.load().has_backward_adjacency()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();

        let mut ids: Vec<NodeId> = self
            .base
            .load()
            .node_ids()
            .into_iter()
            .filter(|id| !deleted.contains(id))
            .collect();
        ids.extend(self.overlay.load().node_ids());
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut ids: Vec<NodeId> = self
            .base
            .load()
            .nodes_by_label(label)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();
        ids.extend(
            self.overlay
                .load()
                .nodes_by_label(label)
                .into_iter()
                .filter(|id| !deleted.contains(id)),
        );
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn node_count(&self) -> usize {
        let base_count = self.base.load().node_count();
        let deleted = self.deleted_from_base_nodes.read().len();
        let overlay_count = self.overlay.load().node_count();
        // Dirty nodes that came from the base are counted once in the overlay.
        // We subtract them from the base total to avoid double counting.
        let promoted = self
            .dirty_node_ids
            .read()
            .iter()
            .filter(|id| self.base.load().get_node(**id).is_some())
            .count();
        base_count - deleted - promoted + overlay_count
    }

    fn edge_count(&self) -> usize {
        let base_count = self.base.load().edge_count();
        let deleted = self.deleted_from_base_edges.read().len();
        let overlay_count = self.overlay.load().edge_count();
        let promoted = self
            .dirty_edge_ids
            .read()
            .iter()
            .filter(|id| self.base.load().get_edge(**id).is_some())
            .count();
        base_count - deleted - promoted + overlay_count
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.load().edge_type(id);
        }
        self.base
            .load()
            .edge_type(id)
            .or_else(|| self.overlay.load().edge_type(id))
    }

    fn has_property_index(&self, property: &str) -> bool {
        // Property indexes only live on the overlay LpgStore (the columnar
        // base has no index store). Without this delegate the trait default
        // returns false, and the planner's property-index fast path silently
        // disables itself after `compact()`.
        self.overlay.load().has_property_index(property)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .load()
            .find_nodes_by_property(property, value)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.load().find_nodes_by_property(property, value));
        results
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        if conditions.is_empty() {
            return self.node_ids();
        }
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .load()
            .find_nodes_by_properties(conditions)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.load().find_nodes_by_properties(conditions));
        results
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .load()
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.load().find_nodes_in_range(
            property,
            min,
            max,
            min_inclusive,
            max_inclusive,
        ));
        results
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.base
            .load()
            .node_property_might_match(property, op, value)
            || self
                .overlay
                .load()
                .node_property_might_match(property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.base
            .load()
            .edge_property_might_match(property, op, value)
            || self
                .overlay
                .load()
                .edge_property_might_match(property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        // Combine base + overlay statistics. Snapshot the overlay once
        // so the labels we enumerate and the per-label counts we read
        // observe the same `LpgStore` revision — otherwise a concurrent
        // `merge_overlay_in_place` (which swaps the overlay) could let
        // us see a label and then read its count from the post-swap
        // empty overlay.
        let base_stats = self.base.load().statistics();
        let overlay = self.overlay.load();

        let mut combined = (*base_stats).clone();
        combined.total_nodes = self.node_count() as u64;
        combined.total_edges = self.edge_count() as u64;

        // Merge label stats from the snapshotted overlay.
        for label in overlay.all_labels() {
            let count = overlay.nodes_by_label(&label).len() as u64;
            if let Some(existing) = combined.get_label(&label) {
                combined.update_label(
                    &label,
                    crate::statistics::LabelStatistics::new(existing.node_count + count),
                );
            } else {
                combined.update_label(&label, crate::statistics::LabelStatistics::new(count));
            }
        }

        Arc::new(combined)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.base.load().estimate_label_cardinality(label)
            + self.overlay.load().estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        // Rough approximation: weighted average.
        let base_est = self.base.load().estimate_avg_degree(edge_type, outgoing);
        let overlay_est = self.overlay.load().estimate_avg_degree(edge_type, outgoing);
        let base_edges = self.base.load().edge_count() as f64;
        let overlay_edges = self.overlay.load().edge_count() as f64;
        let total = base_edges + overlay_edges;
        if total == 0.0 {
            return 0.0;
        }
        (base_est * base_edges + overlay_est * overlay_edges) / total
    }

    fn current_epoch(&self) -> EpochId {
        self.overlay.load().current_epoch()
    }

    fn all_labels(&self) -> Vec<String> {
        let mut labels: FxHashSet<String> = self.base.load().all_labels().into_iter().collect();
        labels.extend(self.overlay.load().all_labels());
        labels.into_iter().collect()
    }

    fn all_edge_types(&self) -> Vec<String> {
        let mut types: FxHashSet<String> = self.base.load().all_edge_types().into_iter().collect();
        types.extend(self.overlay.load().all_edge_types());
        types.into_iter().collect()
    }

    fn all_property_keys(&self) -> Vec<String> {
        let mut keys: FxHashSet<String> =
            self.base.load().all_property_keys().into_iter().collect();
        keys.extend(self.overlay.load().all_property_keys());
        keys.into_iter().collect()
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        if self.is_node_deleted_from_base(id) {
            return false;
        }
        if self.is_node_dirty(id) {
            return self.overlay.load().is_node_visible_at_epoch(id, epoch);
        }
        // `dirty_node_ids` only tracks overlay *modifications of base nodes*
        // — overlay-only nodes (e.g. post-`compact()` writes) fall through
        // here and must be dispatched to the overlay's MVCC check. The base
        // doesn't know the id, so it would otherwise report them invisible.
        //
        // Snapshot the base once: a concurrent `swap_base` between the
        // presence check and the visibility call would otherwise dispatch
        // through a different `CompactStore` than the one we tested.
        let base = self.base.load();
        if base.get_node(id).is_some() {
            base.is_node_visible_at_epoch(id, epoch)
        } else {
            self.overlay.load().is_node_visible_at_epoch(id, epoch)
        }
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_node_deleted_from_base(id) {
            return false;
        }
        if self.is_node_dirty(id) {
            return self
                .overlay
                .load()
                .is_node_visible_versioned(id, epoch, transaction_id);
        }
        let base = self.base.load();
        if base.get_node(id).is_some() {
            base.is_node_visible_versioned(id, epoch, transaction_id)
        } else {
            self.overlay
                .load()
                .is_node_visible_versioned(id, epoch, transaction_id)
        }
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        if self.is_edge_deleted_from_base(id) {
            return false;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.load().is_edge_visible_at_epoch(id, epoch);
        }
        let base = self.base.load();
        if base.get_edge(id).is_some() {
            base.is_edge_visible_at_epoch(id, epoch)
        } else {
            self.overlay.load().is_edge_visible_at_epoch(id, epoch)
        }
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_edge_deleted_from_base(id) {
            return false;
        }
        if self.is_edge_dirty(id) {
            return self
                .overlay
                .load()
                .is_edge_visible_versioned(id, epoch, transaction_id);
        }
        let base = self.base.load();
        if base.get_edge(id).is_some() {
            base.is_edge_visible_versioned(id, epoch, transaction_id)
        } else {
            self.overlay
                .load()
                .is_edge_visible_versioned(id, epoch, transaction_id)
        }
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_at_epoch(*id, epoch))
            .collect()
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_versioned(*id, epoch, transaction_id))
            .collect()
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        if self.is_node_dirty(id) {
            return self.overlay.load().get_node_history(id);
        }
        Vec::new()
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        if self.is_edge_dirty(id) {
            return self.overlay.load().get_edge_history(id);
        }
        Vec::new()
    }
}

impl GraphStoreSearch for LayeredStore {
    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.overlay.load().has_text_index(label, property)
    }

    #[cfg(feature = "text-index")]
    fn score_text(&self, node_id: NodeId, label: &str, property: &str, query: &str) -> Option<f64> {
        if self.is_node_deleted_from_base(node_id) {
            return None;
        }
        self.overlay
            .load()
            .score_text(node_id, label, property, query)
    }

    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Vec<(NodeId, f64)> {
        let deleted = self.deleted_from_base_nodes.read();
        let mut results =
            self.overlay
                .load()
                .text_search(label, property, query, k + deleted.len());
        results.retain(|(id, _)| !deleted.contains(id));
        results.truncate(k);
        results
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
    ) -> Vec<(NodeId, f64)> {
        let deleted = self.deleted_from_base_nodes.read();
        let mut results = self
            .overlay
            .load()
            .text_search_with_threshold(label, property, query, threshold);
        results.retain(|(id, _)| !deleted.contains(id));
        results
    }

    #[cfg(feature = "vector-index")]
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        self.overlay.load().has_vector_index(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_index_metric(&self, label: &str, property: &str) -> Option<DistanceMetric> {
        self.overlay.load().vector_index_metric(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        // Forward to overlay, then filter nodes deleted from base so stale hits
        // from the underlying index do not leak through the layered view.
        let deleted = self.deleted_from_base_nodes.read();
        let mut results =
            self.overlay
                .load()
                .vector_search(label, property, query, k + deleted.len(), metric);
        results.retain(|(id, _)| !deleted.contains(id));
        results.truncate(k);
        results
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        let deleted = self.deleted_from_base_nodes.read();
        let mut results = self
            .overlay
            .load()
            .vector_search_with_threshold(label, property, query, threshold, metric);
        results.retain(|(id, _)| !deleted.contains(id));
        results
    }
}

// ── GraphStoreMut implementation ───────────────────────────────────

impl GraphStoreMut for LayeredStore {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        let _guard = self.merge_guard.read();
        let id = self.overlay.load().create_node(labels);
        self.dirty_node_ids.write().insert(id);
        id
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let _guard = self.merge_guard.read();
        let id = self
            .overlay
            .load()
            .create_node_versioned(labels, epoch, transaction_id);
        self.dirty_node_ids.write().insert(id);
        id
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let _guard = self.merge_guard.read();
        // Promote base-only endpoints into the overlay.
        self.ensure_in_overlay(src);
        self.ensure_in_overlay(dst);
        let id = self.overlay.load().create_edge(src, dst, edge_type);
        self.dirty_edge_ids.write().insert(id);
        id
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(src);
        self.ensure_in_overlay(dst);
        let id =
            self.overlay
                .load()
                .create_edge_versioned(src, dst, edge_type, epoch, transaction_id);
        self.dirty_edge_ids.write().insert(id);
        id
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        let _guard = self.merge_guard.read();
        for &(src, dst, _) in edges {
            self.ensure_in_overlay(src);
            self.ensure_in_overlay(dst);
        }
        let ids = self.overlay.load().batch_create_edges(edges);
        let mut dirty = self.dirty_edge_ids.write();
        for &id in &ids {
            dirty.insert(id);
        }
        ids
    }

    fn delete_node(&self, id: NodeId) -> bool {
        let _guard = self.merge_guard.read();
        if self.is_node_dirty(id) {
            // Node is in the overlay: delete from overlay.
            return self.overlay.load().delete_node(id);
        }
        if self.base.load().get_node(id).is_some() {
            if self.deleted_from_base_nodes.write().insert(id) {
                self.deletions_dirty.store(true, Ordering::Release);
            }
            return true;
        }
        false
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let _guard = self.merge_guard.read();
        if self.is_node_dirty(id) {
            return self
                .overlay
                .load()
                .delete_node_versioned(id, epoch, transaction_id);
        }
        if self.base.load().get_node(id).is_some() {
            if self.deleted_from_base_nodes.write().insert(id) {
                self.deletions_dirty.store(true, Ordering::Release);
            }
            return true;
        }
        false
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        let _guard = self.merge_guard.read();
        // Delete overlay edges.
        if self.is_node_dirty(node_id) {
            self.overlay.load().delete_node_edges(node_id);
        }
        // Mark base edges as deleted.
        let mut deleted_any = false;
        let mut edges = self.deleted_from_base_edges.write();
        for (_, eid) in self.base.load().edges_from(node_id, Direction::Both) {
            if edges.insert(eid) {
                deleted_any = true;
            }
        }
        drop(edges);
        if deleted_any {
            self.deletions_dirty.store(true, Ordering::Release);
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        let _guard = self.merge_guard.read();
        if self.is_edge_dirty(id) {
            return self.overlay.load().delete_edge(id);
        }
        if self.base.load().get_edge(id).is_some() {
            if self.deleted_from_base_edges.write().insert(id) {
                self.deletions_dirty.store(true, Ordering::Release);
            }
            return true;
        }
        false
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let _guard = self.merge_guard.read();
        if self.is_edge_dirty(id) {
            return self
                .overlay
                .load()
                .delete_edge_versioned(id, epoch, transaction_id);
        }
        if self.base.load().get_edge(id).is_some() {
            if self.deleted_from_base_edges.write().insert(id) {
                self.deletions_dirty.store(true, Ordering::Release);
            }
            return true;
        }
        false
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(id);
        self.overlay.load().set_node_property(id, key, value);
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(id);
        self.overlay
            .load()
            .set_node_property_versioned(id, key, value, transaction_id);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        let _guard = self.merge_guard.read();
        self.ensure_edge_in_overlay(id);
        self.overlay.load().set_edge_property(id, key, value);
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let _guard = self.merge_guard.read();
        self.ensure_edge_in_overlay(id);
        self.overlay
            .load()
            .set_edge_property_versioned(id, key, value, transaction_id);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(id);
        self.overlay.load().remove_node_property(id, key)
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(id);
        self.overlay
            .load()
            .remove_node_property_versioned(id, key, transaction_id)
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        let _guard = self.merge_guard.read();
        self.ensure_edge_in_overlay(id);
        self.overlay.load().remove_edge_property(id, key)
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let _guard = self.merge_guard.read();
        self.ensure_edge_in_overlay(id);
        self.overlay
            .load()
            .remove_edge_property_versioned(id, key, transaction_id)
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(node_id);
        self.overlay.load().add_label(node_id, label)
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(node_id);
        self.overlay
            .load()
            .add_label_versioned(node_id, label, transaction_id)
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(node_id);
        self.overlay.load().remove_label(node_id, label)
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let _guard = self.merge_guard.read();
        self.ensure_in_overlay(node_id);
        self.overlay
            .load()
            .remove_label_versioned(node_id, label, transaction_id)
    }
}

// ── Private helpers ────────────────────────────────────────────────

impl LayeredStore {
    /// Ensures a node exists in the overlay. If the node is base-only,
    /// copies its labels and properties into the overlay and marks it dirty.
    fn ensure_in_overlay(&self, id: NodeId) {
        if self.is_node_dirty(id) {
            return; // already in overlay
        }
        let Some(base_node) = self.base.load().get_node(id) else {
            return; // not in base either (new node case handled by caller)
        };

        // Copy the node into the overlay at the same ID.
        // We temporarily lower the ID counter, create the node, then restore it.
        let saved_next = self.overlay.load().next_node_id();
        self.overlay.load().set_next_node_id(id.as_u64());
        let labels: Vec<&str> = base_node.labels.iter().map(|l| l.as_str()).collect();
        let promoted_id = self.overlay.load().create_node(&labels);
        debug_assert_eq!(
            promoted_id, id,
            "promoted node should reuse the original ID"
        );
        self.overlay.load().set_next_node_id(saved_next);

        // Copy properties.
        for (key, value) in base_node.properties.iter() {
            self.overlay
                .load()
                .set_node_property(id, key.as_str(), value.clone());
        }

        self.dirty_node_ids.write().insert(id);
    }

    /// Ensures an edge exists in the overlay.
    fn ensure_edge_in_overlay(&self, id: EdgeId) {
        if self.is_edge_dirty(id) {
            return;
        }
        let Some(base_edge) = self.base.load().get_edge(id) else {
            return;
        };

        // Ensure endpoints are in the overlay first.
        self.ensure_in_overlay(base_edge.src);
        self.ensure_in_overlay(base_edge.dst);

        // Create the edge at the same ID.
        let saved_next = self.overlay.load().next_edge_id();
        self.overlay.load().set_next_edge_id(id.as_u64());
        let promoted_id = self.overlay.load().create_edge(
            base_edge.src,
            base_edge.dst,
            base_edge.edge_type.as_str(),
        );
        debug_assert_eq!(
            promoted_id, id,
            "promoted edge should reuse the original ID"
        );
        self.overlay.load().set_next_edge_id(saved_next);

        // Copy properties.
        for (key, value) in base_edge.properties.iter() {
            self.overlay
                .load()
                .set_edge_property(id, key.as_str(), value.clone());
        }

        self.dirty_edge_ids.write().insert(id);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compact::from_graph_store_preserving_ids;

    fn build_test_layered() -> LayeredStore {
        let store = LpgStore::new().unwrap();

        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

        let e1 = store.create_edge(alix, amsterdam, "LIVES_IN");
        store.set_edge_property(e1, "since", Value::Int64(2020));

        let e2 = store.create_edge(gus, amsterdam, "LIVES_IN");
        store.set_edge_property(e2, "since", Value::Int64(2022));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let max_nid = store
            .node_ids()
            .into_iter()
            .map(|id| id.as_u64())
            .max()
            .unwrap_or(0);
        let max_eid = 10u64; // edges start at 0 in LpgStore
        LayeredStore::new(compact, max_nid, max_eid).unwrap()
    }

    #[test]
    fn test_read_through_base() {
        let layered = build_test_layered();
        assert_eq!(layered.node_count(), 3);
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);
    }

    #[test]
    fn test_create_node_in_overlay() {
        let layered = build_test_layered();
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        assert_eq!(layered.node_count(), 4);
        let node = layered.get_node(vincent).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );
    }

    #[test]
    fn test_delete_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);

        let deleted = layered.delete_node(persons[0]);
        assert!(deleted);
        assert!(layered.get_node(persons[0]).is_none());

        let remaining_persons = layered.nodes_by_label("Person");
        assert_eq!(remaining_persons.len(), 1);
        assert_eq!(layered.node_count(), 2);
    }

    #[test]
    fn test_modify_base_node_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Original value.
        let original_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();
        assert!(matches!(original_age, Value::Int64(_)));

        // Modify: this should promote the node to the overlay.
        layered.set_node_property(first, "age", Value::Int64(99));

        let new_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();
        assert_eq!(new_age, Value::Int64(99));
    }

    #[test]
    fn test_create_edge_between_base_and_overlay() {
        let layered = build_test_layered();
        let paris = layered.create_node(&["City"]);
        layered.set_node_property(paris, "name", Value::from("Paris"));

        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // Create cross-layer edge.
        let eid = layered.create_edge(first_person, paris, "VISITS");
        assert!(layered.get_edge(eid).is_some());

        let edge = layered.get_edge(eid).unwrap();
        assert_eq!(edge.src, first_person);
        assert_eq!(edge.dst, paris);
    }

    #[test]
    fn test_traversal_merges_layers() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        // Base has 2 incoming LIVES_IN edges.
        let incoming = layered.edges_from(amsterdam, Direction::Incoming);
        assert_eq!(incoming.len(), 2);
    }

    #[test]
    fn test_node_ids_combines_layers() {
        let layered = build_test_layered();
        let initial = layered.node_ids();
        assert_eq!(initial.len(), 3);

        layered.create_node(&["New"]);
        let after = layered.node_ids();
        assert_eq!(after.len(), 4);
    }

    #[test]
    fn test_delete_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let (_, eid) = edges[0];
        let deleted = layered.delete_edge(eid);
        assert!(deleted);

        let after = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_all_labels_combines() {
        let layered = build_test_layered();
        layered.create_node(&["NewLabel"]);

        let labels = layered.all_labels();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"City".to_string()));
        assert!(labels.contains(&"NewLabel".to_string()));
    }

    // ── A. Read-through operations ────────────────────────────────

    #[test]
    fn test_get_edge_from_base() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let (_, eid) = edges[0];
        let edge = layered.get_edge(eid);
        assert!(edge.is_some(), "edge should be readable from base");
        let edge = edge.unwrap();
        assert_eq!(edge.edge_type.as_str(), "LIVES_IN");

        // edge_type() accessor should agree
        assert_eq!(layered.edge_type(eid).as_deref(), Some("LIVES_IN"));

        // Edge property from base should be readable
        let since = layered.get_edge_property(eid, &PropertyKey::new("since"));
        assert!(
            since.is_some(),
            "edge property should be readable from base"
        );
    }

    #[test]
    fn test_get_node_property_batch_across_layers() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");

        // Create an overlay-only node.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        let all_ids: Vec<NodeId> = persons
            .iter()
            .copied()
            .chain(std::iter::once(vincent))
            .collect();
        let names = layered.get_node_property_batch(&all_ids, &PropertyKey::new("name"));

        // All should have a name.
        for name in &names {
            assert!(name.is_some(), "every node should have a name property");
        }
        // The overlay node should return "Vincent".
        assert_eq!(
            names.last().unwrap().as_ref().unwrap(),
            &Value::String(ArcStr::from("Vincent"))
        );
    }

    #[test]
    fn test_out_degree_both_layers() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // Base has 1 outgoing edge (LIVES_IN) for an unmodified node.
        assert_eq!(layered.out_degree(first_person), 1);

        // Add an edge purely in the overlay (between overlay-only nodes).
        let vincent = layered.create_node(&["Person"]);
        let berlin = layered.create_node(&["City"]);
        layered.create_edge(vincent, berlin, "VISITS");

        // Overlay-only node should have 1 outgoing edge.
        assert_eq!(layered.out_degree(vincent), 1);

        // Base node remains unmodified, still sees its base edge.
        assert_eq!(layered.out_degree(first_person), 1);
    }

    #[test]
    fn test_in_degree_both_layers() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        // Base has 2 incoming LIVES_IN edges.
        assert_eq!(layered.in_degree(amsterdam), 2);

        // Create overlay-only edges between overlay-only nodes.
        let jules = layered.create_node(&["Person"]);
        let berlin = layered.create_node(&["City"]);
        layered.create_edge(jules, berlin, "LIVES_IN");

        // Berlin (overlay-only) should have 1 incoming edge.
        assert_eq!(layered.in_degree(berlin), 1);

        // Amsterdam (base, not dirty) should still have 2 incoming edges.
        assert_eq!(layered.in_degree(amsterdam), 2);
    }

    // ── B. Mutation operations ────────────────────────────────────

    #[test]
    fn test_set_node_property_promotes_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        assert_eq!(layered.overlay_mutation_count(), 0);

        // Setting a property on a base node should promote it.
        layered.set_node_property(first, "city", Value::from("Amsterdam"));

        // Node should now be dirty (in overlay).
        assert!(layered.overlay_mutation_count() > 0);

        // Property should be readable.
        let city = layered
            .get_node_property(first, &PropertyKey::new("city"))
            .unwrap();
        assert_eq!(city, Value::String(ArcStr::from("Amsterdam")));
    }

    #[test]
    fn test_set_edge_property_promotes_base_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        assert_eq!(layered.overlay_mutation_count(), 0);

        // Setting a property on a base edge should promote it and its endpoints.
        layered.set_edge_property(eid, "weight", Value::Float64(1.5));

        assert!(layered.overlay_mutation_count() > 0);

        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(1.5));
    }

    #[test]
    fn test_remove_node_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Node has "age" property in the base.
        assert!(
            layered
                .get_node_property(first, &PropertyKey::new("age"))
                .is_some()
        );

        // Remove it (promotes to overlay first).
        let removed = layered.remove_node_property(first, "age");
        assert!(removed.is_some());

        // Should be gone now.
        assert!(
            layered
                .get_node_property(first, &PropertyKey::new("age"))
                .is_none()
        );
    }

    #[test]
    fn test_remove_edge_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Remove edge property (promotes edge and endpoints).
        let removed = layered.remove_edge_property(eid, "since");
        assert!(removed.is_some());

        // Should be gone now.
        assert!(
            layered
                .get_edge_property(eid, &PropertyKey::new("since"))
                .is_none()
        );
    }

    #[test]
    fn test_add_label_to_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Add a new label (promotes the node).
        let added = layered.add_label(first, "Employee");
        assert!(added);

        // Node should now have both labels.
        let node = layered.get_node(first).unwrap();
        let label_strs: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
        assert!(label_strs.contains(&"Person"));
        assert!(label_strs.contains(&"Employee"));
    }

    #[test]
    fn test_remove_label_from_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Remove the "Person" label (promotes first).
        let removed = layered.remove_label(first, "Person");
        assert!(removed);

        // Should no longer appear in nodes_by_label("Person").
        let after_persons = layered.nodes_by_label("Person");
        assert!(!after_persons.contains(&first));

        // But node should still exist.
        assert!(layered.get_node(first).is_some());
    }

    #[test]
    fn test_delete_node_edges_cascade() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // First person has 1 outgoing edge.
        let edges_before = layered.edges_from(first_person, Direction::Outgoing);
        assert_eq!(edges_before.len(), 1);

        // Delete all edges connected to this node.
        layered.delete_node_edges(first_person);

        // The base edges should now be marked as deleted.
        let edges_after = layered.edges_from(first_person, Direction::Outgoing);
        assert_eq!(edges_after.len(), 0);
    }

    #[test]
    fn test_batch_create_edges_cross_layer() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_person = persons[0];

        // Create overlay-only cities.
        let berlin = layered.create_node(&["City"]);
        let paris = layered.create_node(&["City"]);

        // Batch create edges with a mix of base and overlay endpoints.
        let edge_specs: Vec<(NodeId, NodeId, &str)> = vec![
            (base_person, berlin, "VISITS"),
            (base_person, paris, "VISITS"),
        ];
        let eids = layered.batch_create_edges(&edge_specs);
        assert_eq!(eids.len(), 2);

        for eid in &eids {
            let edge = layered.get_edge(*eid);
            assert!(edge.is_some());
            assert_eq!(edge.unwrap().edge_type.as_str(), "VISITS");
        }
    }

    // ── C. Promotion logic ────────────────────────────────────────

    #[test]
    fn test_promotion_copies_all_properties() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Before promotion, read the base node properties.
        let original_name = layered
            .get_node_property(first, &PropertyKey::new("name"))
            .unwrap();
        let original_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();

        // Promote by setting a new property.
        layered.set_node_property(first, "city", Value::from("Berlin"));

        // All original properties should still be present.
        let after_name = layered
            .get_node_property(first, &PropertyKey::new("name"))
            .unwrap();
        let after_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();

        assert_eq!(original_name, after_name);
        assert_eq!(original_age, after_age);

        // New property also present.
        let city = layered
            .get_node_property(first, &PropertyKey::new("city"))
            .unwrap();
        assert_eq!(city, Value::String(ArcStr::from("Berlin")));
    }

    #[test]
    fn test_promotion_is_idempotent() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // First promotion.
        layered.set_node_property(first, "x", Value::Int64(1));
        let count_after_first = layered.node_count();

        // Second promotion attempt (node already in overlay).
        layered.set_node_property(first, "y", Value::Int64(2));
        let count_after_second = layered.node_count();

        // Node count should not change.
        assert_eq!(count_after_first, count_after_second);

        // Both properties should exist.
        assert_eq!(
            layered
                .get_node_property(first, &PropertyKey::new("x"))
                .unwrap(),
            Value::Int64(1)
        );
        assert_eq!(
            layered
                .get_node_property(first, &PropertyKey::new("y"))
                .unwrap(),
            Value::Int64(2)
        );
    }

    #[test]
    fn test_edge_promotion_promotes_endpoints() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Promote the edge by setting a property on it.
        layered.set_edge_property(eid, "weight", Value::Float64(0.5));

        // The edge's source and destination nodes should now be in the overlay.
        let edge = layered.get_edge(eid).unwrap();
        let src_node = layered.get_node(edge.src);
        let dst_node = layered.get_node(edge.dst);
        assert!(
            src_node.is_some(),
            "source node should be accessible after edge promotion"
        );
        assert!(
            dst_node.is_some(),
            "destination node should be accessible after edge promotion"
        );
    }

    // ── D. Deleted entity tracking ────────────────────────────────

    #[test]
    fn test_deleted_base_node_invisible() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        let deleted = layered.delete_node(target);
        assert!(deleted);

        // get_node should return None.
        assert!(layered.get_node(target).is_none());

        // get_node_property should also return None.
        assert!(
            layered
                .get_node_property(target, &PropertyKey::new("name"))
                .is_none()
        );
    }

    #[test]
    fn test_deleted_node_excluded_from_nodes_by_label() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);

        let target = persons[0];
        layered.delete_node(target);

        let after = layered.nodes_by_label("Person");
        assert_eq!(after.len(), 1);
        assert!(!after.contains(&target));
    }

    #[test]
    fn test_deleted_node_excluded_from_node_ids() {
        let layered = build_test_layered();
        let all_before = layered.node_ids();
        assert_eq!(all_before.len(), 3);

        let persons = layered.nodes_by_label("Person");
        let target = persons[0];
        layered.delete_node(target);

        let all_after = layered.node_ids();
        assert_eq!(all_after.len(), 2);
        assert!(!all_after.contains(&target));
    }

    #[test]
    fn test_deleted_edge_excluded_from_edges_from() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        let edges = layered.edges_from(first, Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let (_, eid) = edges[0];

        layered.delete_edge(eid);

        let after = layered.edges_from(first, Direction::Outgoing);
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_deleted_edge_excluded_from_neighbors() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Deleting the target NODE should remove it from neighbors.
        let neighbors_before = layered.neighbors(first, Direction::Outgoing);
        assert_eq!(neighbors_before.len(), 1);

        let target_node = neighbors_before[0];
        layered.delete_node(target_node);

        let neighbors_after = layered.neighbors(first, Direction::Outgoing);
        assert_eq!(
            neighbors_after.len(),
            0,
            "deleted node should not appear in neighbors"
        );
    }

    #[test]
    fn test_node_count_reflects_deletions() {
        let layered = build_test_layered();
        assert_eq!(layered.node_count(), 3);

        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);
        assert_eq!(layered.node_count(), 2);

        layered.delete_node(persons[1]);
        assert_eq!(layered.node_count(), 1);
    }

    #[test]
    fn test_edge_count_reflects_deletions() {
        let layered = build_test_layered();
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        layered.delete_edge(eid);
        assert_eq!(layered.edge_count(), 1);
    }

    // ── E. Search & statistics ────────────────────────────────────

    #[test]
    fn test_find_nodes_by_property_across_layers() {
        let layered = build_test_layered();

        // Base has Alix (age=30) and Gus (age=25).
        let age_30 = layered.find_nodes_by_property("age", &Value::Int64(30));
        assert_eq!(age_30.len(), 1);

        // Add an overlay node with the same property value.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "age", Value::Int64(30));

        let age_30_after = layered.find_nodes_by_property("age", &Value::Int64(30));
        assert_eq!(age_30_after.len(), 2);
        assert!(age_30_after.contains(&vincent));
    }

    #[test]
    fn test_find_nodes_in_range_across_layers() {
        let layered = build_test_layered();

        // Add an overlay node with age=35.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "age", Value::Int64(35));

        // Range query: age in [25, 35].
        let in_range = layered.find_nodes_in_range(
            "age",
            Some(&Value::Int64(25)),
            Some(&Value::Int64(35)),
            true,
            true,
        );

        // Should find Gus (25), Alix (30), and Mia (35).
        assert!(
            in_range.len() >= 3,
            "expected at least 3 nodes in range, got {}",
            in_range.len()
        );
        assert!(in_range.contains(&mia));
    }

    #[test]
    fn test_statistics_reflects_overlay() {
        let layered = build_test_layered();
        let stats_before = layered.statistics();
        let nodes_before = stats_before.total_nodes;

        // Add overlay nodes.
        layered.create_node(&["Person"]);
        layered.create_node(&["City"]);

        let stats_after = layered.statistics();
        assert_eq!(stats_after.total_nodes, nodes_before + 2);
    }

    #[test]
    fn test_all_edge_types_combines_layers() {
        let layered = build_test_layered();
        let types_before = layered.all_edge_types();
        assert!(types_before.contains(&"LIVES_IN".to_string()));

        // Add a new edge type in the overlay.
        let persons = layered.nodes_by_label("Person");
        let butch = layered.create_node(&["Person"]);
        layered.create_edge(persons[0], butch, "KNOWS");

        let types_after = layered.all_edge_types();
        assert!(types_after.contains(&"LIVES_IN".to_string()));
        assert!(types_after.contains(&"KNOWS".to_string()));
    }

    #[test]
    fn test_all_property_keys_combines_layers() {
        let layered = build_test_layered();
        let keys_before = layered.all_property_keys();
        assert!(keys_before.contains(&"name".to_string()));
        assert!(keys_before.contains(&"age".to_string()));

        // Add a new property key in the overlay.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "email", Value::from("mia@example.com"));

        let keys_after = layered.all_property_keys();
        assert!(keys_after.contains(&"email".to_string()));
        assert!(keys_after.contains(&"name".to_string()));
    }

    // ── F. Visibility ─────────────────────────────────────────────

    #[test]
    fn test_overlay_mutation_count() {
        let layered = build_test_layered();
        assert_eq!(layered.overlay_mutation_count(), 0);

        // Create a node: 1 dirty node.
        layered.create_node(&["Person"]);
        assert_eq!(layered.overlay_mutation_count(), 1);

        // Delete a base node: 1 dirty node + 1 deleted base node.
        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);
        assert_eq!(layered.overlay_mutation_count(), 2);
    }

    #[test]
    fn test_memory_bytes_nonzero() {
        let layered = build_test_layered();
        assert!(
            layered.memory_bytes() > 0,
            "memory_bytes should be positive for a non-empty store"
        );
    }

    // ── G. Versioned mutation methods ────────────────────────────────

    // ── G. Versioned read methods ────────────────────────────────────
    // Note: versioned mutation tests are omitted because the layered store's
    // epoch ordering (base at MAX, overlay at 0) prevents versioned writes
    // from appending to the version log. The non-versioned mutation tests
    // above already exercise the ensure_in_overlay promotion logic.

    #[test]
    fn test_versioned_node_reads() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Base node falls through
        assert!(
            layered.get_node_versioned(first, epoch, txn_id).is_some(),
            "versioned read should fall through to base"
        );
        assert!(
            layered.get_node_at_epoch(first, epoch).is_some(),
            "base node should be visible at epoch 0"
        );

        // Overlay node is readable
        let hans = layered.create_node_versioned(&["Person"], epoch, txn_id);
        layered.set_node_property(hans, "name", Value::from("Hans"));
        let node = layered.get_node_versioned(hans, epoch, txn_id).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Hans")))
        );
        assert!(layered.get_node_at_epoch(hans, epoch).is_some());

        // Deleted base node returns None
        layered.delete_node(first);
        assert!(
            layered.get_node_versioned(first, epoch, txn_id).is_none(),
            "versioned read should return None for deleted base node"
        );
        assert!(
            layered.get_node_at_epoch(first, epoch).is_none(),
            "deleted base node should not be visible at epoch"
        );
    }

    #[test]
    fn test_versioned_edge_reads() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let base_edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = base_edges[0];

        // Base edge falls through
        assert!(
            layered
                .get_edge_versioned(base_eid, epoch, txn_id)
                .is_some(),
            "versioned read should fall through to base edge"
        );
        assert!(
            layered.get_edge_at_epoch(base_eid, epoch).is_some(),
            "base edge should be visible at epoch 0"
        );

        // Overlay edge is readable
        let barcelona = layered.create_node(&["City"]);
        let overlay_eid =
            layered.create_edge_versioned(persons[0], barcelona, "VISITS", epoch, txn_id);
        let edge = layered
            .get_edge_versioned(overlay_eid, epoch, txn_id)
            .unwrap();
        assert_eq!(edge.edge_type.as_str(), "VISITS");

        // Deleted base edge returns None
        layered.delete_edge(base_eid);
        assert!(
            layered
                .get_edge_versioned(base_eid, epoch, txn_id)
                .is_none(),
            "versioned read should return None for deleted base edge"
        );
        assert!(
            layered.get_edge_at_epoch(base_eid, epoch).is_none(),
            "deleted base edge should not be visible at epoch"
        );
    }

    // ── I. Visibility methods ────────────────────────────────────────

    #[test]
    fn test_node_visibility() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        // Base node visible (epoch and versioned)
        assert!(layered.is_node_visible_at_epoch(target, epoch));
        assert!(layered.is_node_visible_versioned(target, epoch, txn_id));

        // Overlay node visible
        let beatrix = layered.create_node(&["Person"]);
        assert!(layered.is_node_visible_at_epoch(beatrix, epoch));

        // Versioned overlay node visible
        let butch = layered.create_node_versioned(&["Person"], epoch, txn_id);
        assert!(layered.is_node_visible_versioned(butch, epoch, txn_id));

        // Deleted base node invisible
        layered.delete_node(target);
        assert!(!layered.is_node_visible_at_epoch(target, epoch));
        assert!(!layered.is_node_visible_versioned(target, epoch, txn_id));
    }

    #[test]
    fn test_edge_visibility() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Base edge visible (epoch and versioned)
        assert!(layered.is_edge_visible_at_epoch(eid, epoch));
        assert!(layered.is_edge_visible_versioned(eid, epoch, txn_id));

        // Deleted base edge invisible
        layered.delete_edge(eid);
        assert!(!layered.is_edge_visible_at_epoch(eid, epoch));
        assert!(!layered.is_edge_visible_versioned(eid, epoch, txn_id));
    }

    #[test]
    fn test_filter_visible_node_ids() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);

        let all_ids = layered.node_ids();
        assert_eq!(all_ids.len(), 3);

        // All nodes visible (epoch and versioned)
        assert_eq!(layered.filter_visible_node_ids(&all_ids, epoch).len(), 3);
        assert_eq!(
            layered
                .filter_visible_node_ids_versioned(&all_ids, epoch, txn_id)
                .len(),
            3
        );

        // Delete one, both filters should exclude it
        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);

        let visible_epoch = layered.filter_visible_node_ids(&all_ids, epoch);
        assert_eq!(visible_epoch.len(), 2);
        assert!(!visible_epoch.contains(&persons[0]));

        let visible_versioned = layered.filter_visible_node_ids_versioned(&all_ids, epoch, txn_id);
        assert_eq!(visible_versioned.len(), 2);
        assert!(!visible_versioned.contains(&persons[0]));
    }

    // ── J. History methods ───────────────────────────────────────────

    #[test]
    fn test_history_base_only_and_dirty() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];
        let edges = layered.edges_from(first, Direction::Outgoing);
        let (_, eid) = edges[0];

        // Base-only entities have empty history
        assert!(
            layered.get_node_history(first).is_empty(),
            "base-only node should have no history entries"
        );
        assert!(
            layered.get_edge_history(eid).is_empty(),
            "base-only edge should have no history entries"
        );

        // Promote both to overlay by modifying them
        layered.set_node_property(first, "age", Value::Int64(42));
        layered.set_edge_property(eid, "weight", Value::Float64(2.0));

        // Dirty entities delegate to overlay history (should not panic)
        let _ = layered.get_node_history(first);
        let _ = layered.get_edge_history(eid);
    }

    // ── K. Multi-condition search and batch reads ────────────────────

    #[test]
    fn test_find_nodes_by_properties_across_layers() {
        let layered = build_test_layered();

        // Base has Alix (name="Alix", age=30) and Gus (name="Gus", age=25).
        let results = layered
            .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::Int64(30))]);
        assert_eq!(results.len(), 1);

        // Add overlay node matching the same conditions.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "name", Value::from("Alix"));
        layered.set_node_property(mia, "age", Value::Int64(30));

        let results_after = layered
            .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::Int64(30))]);
        assert_eq!(results_after.len(), 2);
        assert!(results_after.contains(&mia));
    }

    #[test]
    fn test_find_nodes_by_properties_empty_conditions() {
        let layered = build_test_layered();

        // Empty conditions should return all node IDs.
        let results = layered.find_nodes_by_properties(&[]);
        assert_eq!(results.len(), layered.node_ids().len());
    }

    #[test]
    fn test_get_nodes_properties_selective_batch() {
        let layered = build_test_layered();

        let persons = layered.nodes_by_label("Person");
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));
        layered.set_node_property(vincent, "age", Value::Int64(38));

        let all_ids: Vec<NodeId> = persons
            .iter()
            .copied()
            .chain(std::iter::once(vincent))
            .collect();

        let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
        let batch = layered.get_nodes_properties_selective_batch(&all_ids, &keys);

        assert_eq!(batch.len(), all_ids.len());
        // Each map should contain only requested keys.
        for map in &batch {
            for key in map.keys() {
                assert!(
                    keys.contains(key),
                    "unexpected key {:?} in selective batch",
                    key
                );
            }
        }

        // Vincent's map should have both keys.
        let vincent_map = &batch[batch.len() - 1];
        assert_eq!(
            vincent_map.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );
        assert_eq!(
            vincent_map.get(&PropertyKey::new("age")),
            Some(&Value::Int64(38))
        );
    }

    #[test]
    fn test_get_edges_properties_selective_batch() {
        let layered = build_test_layered();

        let persons = layered.nodes_by_label("Person");
        let edges_a = layered.edges_from(persons[0], Direction::Outgoing);
        let edges_b = layered.edges_from(persons[1], Direction::Outgoing);

        let edge_ids: Vec<EdgeId> = edges_a
            .iter()
            .chain(edges_b.iter())
            .map(|(_, eid)| *eid)
            .collect();

        let keys = vec![PropertyKey::new("since")];
        let batch = layered.get_edges_properties_selective_batch(&edge_ids, &keys);

        assert_eq!(batch.len(), edge_ids.len());
        for map in &batch {
            assert!(
                map.contains_key(&PropertyKey::new("since")),
                "each edge should have the 'since' property"
            );
        }
    }

    // ── L. Other uncovered methods ───────────────────────────────────

    #[test]
    fn test_estimate_label_cardinality() {
        let layered = build_test_layered();

        let person_card = layered.estimate_label_cardinality("Person");
        assert!(
            person_card >= 2.0,
            "should estimate at least 2 Person nodes, got {}",
            person_card
        );

        let city_card = layered.estimate_label_cardinality("City");
        assert!(
            city_card >= 1.0,
            "should estimate at least 1 City node, got {}",
            city_card
        );

        // Non-existent labels: the overlay's statistics may return a non-zero
        // default estimate, so we only check the call does not panic.
        let missing_card = layered.estimate_label_cardinality("NonExistent");
        assert!(
            missing_card >= 0.0,
            "cardinality for unknown label should be non-negative"
        );
    }

    #[test]
    fn test_estimate_avg_degree() {
        let layered = build_test_layered();

        let avg_out = layered.estimate_avg_degree("LIVES_IN", true);
        assert!(
            avg_out > 0.0,
            "average out-degree for LIVES_IN should be positive"
        );

        let avg_in = layered.estimate_avg_degree("LIVES_IN", false);
        assert!(
            avg_in > 0.0,
            "average in-degree for LIVES_IN should be positive"
        );
    }

    #[test]
    fn test_estimate_avg_degree_empty() {
        let store = LpgStore::new().unwrap();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let layered = LayeredStore::new(compact, 0, 0).unwrap();

        let avg = layered.estimate_avg_degree("NONEXISTENT", true);
        assert_eq!(avg, 0.0, "empty store should have avg degree 0");
    }

    #[test]
    fn test_node_property_might_match() {
        let layered = build_test_layered();

        // "age" exists in the base, so might_match for Eq with an Int64 should be true
        // (zone maps allow Int64 values).
        let might = layered.node_property_might_match(
            &PropertyKey::new("age"),
            CompareOp::Eq,
            &Value::Int64(30),
        );
        assert!(might, "zone map should indicate age might match 30");
    }

    #[test]
    fn test_edge_property_might_match() {
        let layered = build_test_layered();

        let might = layered.edge_property_might_match(
            &PropertyKey::new("since"),
            CompareOp::Eq,
            &Value::Int64(2020),
        );
        assert!(might, "zone map should indicate since might match 2020");
    }

    // ── M. Focused coverage for promotion and delete-then-recreate ────

    /// Deleting a base node and then creating a fresh overlay node with the
    /// same label must not double-count. The overlay allocator seeded by
    /// `max_node_id + 1` guarantees the new node gets a distinct ID, and
    /// the tombstone on the deleted base ID prevents it from reappearing.
    /// Covers the deletion bookkeeping in `neighbors`, `node_ids`, and
    /// `node_count`.
    #[test]
    fn test_layered_delete_and_recreate_node() {
        let layered = build_test_layered();

        let persons_before = layered.nodes_by_label("Person");
        assert_eq!(persons_before.len(), 2);
        let target = persons_before[0];

        // Record neighbors of the other person (baseline). Alix -> Amsterdam,
        // Gus -> Amsterdam both exist in the base; choose the non-target.
        let other = persons_before[1];
        let other_neighbors_before = layered.neighbors(other, Direction::Outgoing);

        // Delete target (base node), then create a fresh Person in the overlay.
        assert!(layered.delete_node(target));
        let replacement = layered.create_node(&["Person"]);
        layered.set_node_property(replacement, "name", Value::from("Shosanna"));

        // Counts should reflect: 3 base - 1 deleted + 1 overlay node = 3.
        assert_eq!(layered.node_count(), 3);

        // nodes_by_label should see exactly 2 Persons again: the non-deleted
        // base person and the new overlay person.
        let persons_after = layered.nodes_by_label("Person");
        assert_eq!(persons_after.len(), 2);
        assert!(persons_after.contains(&other));
        assert!(persons_after.contains(&replacement));
        assert!(
            !persons_after.contains(&target),
            "deleted base node must not reappear"
        );

        // Neighbors of the non-target person should be unchanged.
        let other_neighbors_after = layered.neighbors(other, Direction::Outgoing);
        assert_eq!(other_neighbors_before, other_neighbors_after);

        // node_ids must not contain the tombstoned id.
        let all_ids = layered.node_ids();
        assert!(!all_ids.contains(&target));
        assert!(all_ids.contains(&replacement));
    }

    /// Mutating a base-only node promotes it into the overlay with all its
    /// labels and properties copied over, and the overlay's node ID counter
    /// is restored so subsequent `create_node` calls still get fresh IDs.
    /// Exercises `ensure_in_overlay` end to end.
    #[test]
    fn test_layered_promote_node_on_mutation() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        // Record the overlay's next-id allocator before promotion so we can
        // verify it is restored afterwards.
        let next_id_before = layered.overlay.load().next_node_id();

        // Snapshot the base node to compare after promotion.
        let base_node = layered.base.load().get_node(target).unwrap();
        let base_labels: Vec<String> = base_node
            .labels
            .iter()
            .map(|l| l.as_str().to_string())
            .collect();
        let base_name = layered
            .base
            .load()
            .get_node_property(target, &PropertyKey::new("name"))
            .unwrap();

        // Mutate: set a new property to trigger promotion.
        layered.set_node_property(target, "city", Value::from("Amsterdam"));

        // Overlay now owns the node; labels survived.
        let promoted = layered.overlay.load().get_node(target).unwrap();
        let promoted_labels: Vec<String> = promoted
            .labels
            .iter()
            .map(|l| l.as_str().to_string())
            .collect();
        assert_eq!(promoted_labels, base_labels);

        // Existing properties survived (read through the layered store).
        let name_after = layered
            .get_node_property(target, &PropertyKey::new("name"))
            .unwrap();
        assert_eq!(name_after, base_name);

        // New property is set.
        assert_eq!(
            layered.get_node_property(target, &PropertyKey::new("city")),
            Some(Value::String(ArcStr::from("Amsterdam")))
        );

        // ID counter was restored: allocating a new node must not collide
        // with the promoted id or any existing base id.
        let next_id_after = layered.overlay.load().next_node_id();
        assert_eq!(
            next_id_before, next_id_after,
            "overlay next_node_id should be restored after promotion"
        );
        let fresh = layered.create_node(&["Person"]);
        assert_ne!(fresh, target);
        for &p in &persons {
            assert_ne!(fresh, p);
        }
    }

    /// Mutating a base-only edge promotes the edge into the overlay together
    /// with both its endpoints, and its properties are preserved. Covers
    /// `ensure_edge_in_overlay` including its cascade into
    /// `ensure_in_overlay` for src and dst.
    #[test]
    fn test_layered_promote_edge_on_mutation() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let (target_dst, target_eid) = edges[0];

        // Capture the base edge's metadata for later comparison.
        let base_edge = layered.base.load().get_edge(target_eid).unwrap();
        let base_since = base_edge
            .properties
            .get(&PropertyKey::new("since"))
            .cloned()
            .unwrap();

        // Mutate: this promotes the edge and its endpoints.
        layered.set_edge_property(target_eid, "weight", Value::Float64(0.75));

        // The edge is now owned by the overlay.
        assert!(layered.overlay.load().get_edge(target_eid).is_some());
        // Original property still readable via the layered store.
        let since_after = layered
            .get_edge_property(target_eid, &PropertyKey::new("since"))
            .unwrap();
        assert_eq!(since_after, base_since);
        // New property is readable.
        assert_eq!(
            layered.get_edge_property(target_eid, &PropertyKey::new("weight")),
            Some(Value::Float64(0.75))
        );

        // Both endpoints are promoted and reachable from the overlay.
        assert!(
            layered.overlay.load().get_node(persons[0]).is_some(),
            "edge source must be in the overlay after promotion"
        );
        assert!(
            layered.overlay.load().get_node(target_dst).is_some(),
            "edge destination must be in the overlay after promotion"
        );

        // Endpoints' existing properties are intact through the layered view.
        assert!(
            layered
                .get_node_property(persons[0], &PropertyKey::new("name"))
                .is_some()
        );
        assert!(
            layered
                .get_node_property(target_dst, &PropertyKey::new("name"))
                .is_some()
        );
    }

    /// Setting a property on a base-only node marks the node dirty. Directly
    /// exercises the private `is_node_dirty` accessor used by the promotion
    /// machinery.
    #[test]
    fn test_layered_is_node_dirty_after_mutation() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        assert!(
            !layered.is_node_dirty(target),
            "base-only node should start clean"
        );

        layered.set_node_property(target, "city", Value::from("Berlin"));

        assert!(
            layered.is_node_dirty(target),
            "node must be dirty after a mutating set_node_property"
        );
    }

    // ── N. Accessor / debug coverage ─────────────────────────────────

    #[test]
    fn test_base_store_and_overlay_store_accessors() {
        let layered = build_test_layered();

        // base_store() returns a reference whose counts match the original.
        assert_eq!(layered.base_store_arc().node_count(), 3);
        assert_eq!(layered.base_store_arc().edge_count(), 2);

        // base_store_arc() returns an owned Arc that aliases the base.
        let arc = layered.base_store_arc();
        assert_eq!(arc.node_count(), 3);

        // overlay_store() returns the Arc<LpgStore> reference.
        assert_eq!(layered.overlay_store().node_count(), 0);
    }

    #[test]
    fn test_has_backward_adjacency() {
        let layered = build_test_layered();
        // Base is built via from_graph_store_preserving_ids which enables
        // backward CSR for every rel table.
        assert!(layered.has_backward_adjacency());
    }

    #[test]
    fn test_debug_format_does_not_panic() {
        let layered = build_test_layered();
        let s = format!("{layered:?}");
        assert!(s.contains("LayeredStore"));
    }

    #[test]
    fn test_current_epoch_delegates_to_overlay() {
        let layered = build_test_layered();
        let epoch = layered.current_epoch();
        // Just verify that the delegation does not panic, and that overlay
        // agrees.
        assert_eq!(epoch, layered.overlay_store().current_epoch());
    }

    // ── N. Overlay-only mutation and delete scenarios ────────────────

    #[test]
    fn test_delete_overlay_only_node() {
        let layered = build_test_layered();

        // Create a fresh overlay node, then delete it. This hits the
        // `is_node_dirty` branch of delete_node.
        let beatrix = layered.create_node(&["Person"]);
        assert!(layered.get_node(beatrix).is_some());

        let deleted = layered.delete_node(beatrix);
        assert!(deleted);
        assert!(
            layered.get_node(beatrix).is_none(),
            "overlay-only node should be unreadable after delete"
        );

        // delete on a non-existent ID returns false.
        let missing = NodeId::from(9_999_999u64);
        assert!(!layered.delete_node(missing));
    }

    #[test]
    fn test_delete_overlay_only_edge() {
        let layered = build_test_layered();

        // Overlay-only edge between two overlay-only nodes.
        let django = layered.create_node(&["Person"]);
        let shosanna = layered.create_node(&["Person"]);
        let eid = layered.create_edge(django, shosanna, "KNOWS");
        assert!(layered.get_edge(eid).is_some());

        let deleted = layered.delete_edge(eid);
        assert!(deleted);
        assert!(
            layered.get_edge(eid).is_none(),
            "overlay-only edge should be unreadable after delete"
        );

        // delete on an unknown edge id returns false.
        let missing = EdgeId::from(9_999_999u64);
        assert!(!layered.delete_edge(missing));
    }

    #[test]
    fn test_delete_then_recreate_node_with_same_label() {
        let layered = build_test_layered();
        let persons_before = layered.nodes_by_label("Person");
        assert_eq!(persons_before.len(), 2);

        // Delete one base Person, then add a new overlay Person.
        layered.delete_node(persons_before[0]);
        let hans = layered.create_node(&["Person"]);
        layered.set_node_property(hans, "name", Value::from("Hans"));

        let persons_after = layered.nodes_by_label("Person");
        // 1 remaining base Person + 1 new overlay Person = 2.
        assert_eq!(persons_after.len(), 2);
        assert!(persons_after.contains(&hans));
        assert!(!persons_after.contains(&persons_before[0]));
    }

    #[test]
    fn test_neighbors_from_promoted_node_with_new_overlay_edges() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Promote the base node, then add a new outgoing edge on it.
        layered.set_node_property(first, "touched", Value::Bool(true));
        let paris = layered.create_node(&["City"]);
        layered.set_node_property(paris, "name", Value::from("Paris"));
        let _ = layered.create_edge(first, paris, "VISITS");

        // Neighbors should include BOTH the original Amsterdam (a base edge
        // that pre-dates promotion) and the new Paris (an overlay edge added
        // after promotion). `ensure_in_overlay` only copies the node's labels
        // and properties — never its adjacency — so the base layer remains
        // authoritative for pre-promotion edges and must be merged with the
        // overlay's post-promotion edges.
        let outgoing = layered.neighbors(first, Direction::Outgoing);
        assert!(
            outgoing.contains(&paris),
            "overlay-created edge target should appear in neighbors"
        );
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities
            .iter()
            .copied()
            .find(|&c| c != paris)
            .expect("base City Amsterdam should still be present");
        assert!(
            outgoing.contains(&amsterdam),
            "base edge target should still appear in neighbors after promotion"
        );
    }

    #[test]
    fn test_edges_from_promoted_node_has_overlay_edges() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Promote and add overlay-only edges.
        let berlin = layered.create_node(&["City"]);
        layered.set_node_property(first, "touched", Value::Bool(true));
        let new_eid = layered.create_edge(first, berlin, "VISITS");

        let edges = layered.edges_from(first, Direction::Outgoing);
        let found_ids: Vec<EdgeId> = edges.iter().map(|(_, e)| *e).collect();
        assert!(
            found_ids.contains(&new_eid),
            "new overlay edge should be reachable via edges_from after promotion"
        );
    }

    #[test]
    fn test_edge_count_with_overlay_adds() {
        let layered = build_test_layered();
        // Base: 2 edges.
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        let vincent = layered.create_node(&["Person"]);
        let _ = layered.create_edge(persons[0], vincent, "KNOWS");
        let _ = layered.create_edge(persons[1], vincent, "KNOWS");

        // Base (2) - deleted (0) - promoted (0) + overlay (2 new) = 4.
        assert_eq!(layered.edge_count(), 4);
    }

    #[test]
    fn test_edge_count_with_base_edge_promoted_is_not_double_counted() {
        let layered = build_test_layered();
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        let base_edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = base_edges[0];

        // Promote base edge to overlay (by setting a new property on it).
        layered.set_edge_property(base_eid, "weight", Value::Float64(1.0));

        // Total must remain 2 (promoted, not duplicated).
        assert_eq!(layered.edge_count(), 2);
    }

    #[test]
    fn test_delete_edge_then_recreate() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];
        let edges = layered.edges_from(first, Direction::Outgoing);
        let (target, base_eid) = edges[0];

        // Delete the base edge.
        assert!(layered.delete_edge(base_eid));
        assert_eq!(layered.edges_from(first, Direction::Outgoing).len(), 0);

        // Recreate a fresh overlay edge between the same endpoints.
        let new_eid = layered.create_edge(first, target, "LIVES_IN");
        assert_ne!(new_eid, base_eid);

        let edges_after = layered.edges_from(first, Direction::Outgoing);
        assert_eq!(edges_after.len(), 1);
        assert_eq!(edges_after[0].1, new_eid);
    }

    #[test]
    fn test_overlay_mutation_count_tracks_all_four_kinds() {
        let layered = build_test_layered();
        assert_eq!(layered.overlay_mutation_count(), 0);

        // Kind 1: dirty node (new overlay node).
        layered.create_node(&["Person"]);
        assert_eq!(
            layered.overlay_mutation_count(),
            1,
            "kind 1 (dirty node) must increment by exactly 1"
        );

        // Kind 2: dirty edge (new overlay edge between new overlay nodes).
        // Two more dirty nodes + one dirty edge = +3. Running total 1 + 3 = 4.
        let a = layered.create_node(&["Person"]);
        let b = layered.create_node(&["Person"]);
        let _ = layered.create_edge(a, b, "KNOWS");
        assert_eq!(
            layered.overlay_mutation_count(),
            4,
            "kind 2 (2 nodes + 1 edge) must add exactly 3, for total 4"
        );

        // Kind 3: deleted base node.
        let persons = layered.nodes_by_label("Person");
        let base_person = *persons
            .iter()
            .find(|id| layered.base.load().get_node(**id).is_some())
            .expect("fixture must have at least one base node");
        let before_delete_node = layered.overlay_mutation_count();
        layered.delete_node(base_person);
        assert_eq!(
            layered.overlay_mutation_count(),
            before_delete_node + 1,
            "kind 3 (deleted base node) must increment by exactly 1"
        );

        // Kind 4: deleted base edge. The fixture is required to have one so
        // this branch always executes; a conditional would let the kind-4
        // tracker silently regress.
        let persons2 = layered.nodes_by_label("Person");
        let (other_base, base_eid) = persons2
            .iter()
            .find_map(|id| {
                layered.base.load().get_node(*id)?;
                let edges = layered.edges_from(*id, Direction::Outgoing);
                edges.first().map(|(_, eid)| (*id, *eid))
            })
            .expect("fixture must have at least one base edge to delete");
        let _ = other_base;
        let before_delete_edge = layered.overlay_mutation_count();
        layered.delete_edge(base_eid);
        assert_eq!(
            layered.overlay_mutation_count(),
            before_delete_edge + 1,
            "kind 4 (deleted base edge) must increment by exactly 1"
        );
    }

    #[test]
    fn test_get_node_history_base_edge_returns_empty() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = edges[0];

        // A pristine base edge has no history entries.
        assert!(layered.get_edge_history(base_eid).is_empty());

        // A pristine base node also has empty history.
        assert!(layered.get_node_history(persons[0]).is_empty());
    }

    // ── M. Accessors and Debug ───────────────────────────────────────

    #[test]
    fn test_base_store_accessors() {
        let layered = build_test_layered();

        // base_store returns a reference with 3 base nodes
        assert_eq!(layered.base_store_arc().node_count(), 3);

        // base_store_arc returns a cloned Arc that sees the same data
        let arc_clone = layered.base_store_arc();
        assert_eq!(arc_clone.node_count(), 3);
        assert!(Arc::strong_count(&arc_clone) >= 2);
    }

    #[test]
    fn test_overlay_store_accessor() {
        let layered = build_test_layered();
        assert_eq!(layered.overlay_store().node_count(), 0);

        layered.create_node(&["Person"]);
        assert_eq!(layered.overlay_store().node_count(), 1);
    }

    #[test]
    fn test_debug_impl_renders() {
        let layered = build_test_layered();
        layered.create_node(&["Person"]);
        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);

        let rendered = format!("{layered:?}");
        assert!(rendered.contains("LayeredStore"));
        assert!(rendered.contains("base_node_count"));
        assert!(rendered.contains("overlay_node_count"));
        assert!(rendered.contains("dirty_nodes"));
        assert!(rendered.contains("deleted_base_nodes"));
    }

    // ── N. Miscellaneous read paths ──────────────────────────────────

    #[test]
    fn test_get_nodes_properties_batch_full() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        let ids: Vec<NodeId> = persons
            .iter()
            .copied()
            .chain(std::iter::once(vincent))
            .collect();
        let batch = layered.get_nodes_properties_batch(&ids);
        assert_eq!(batch.len(), ids.len());

        // Base nodes have name and age.
        for map in batch.iter().take(persons.len()) {
            assert!(map.contains_key(&PropertyKey::new("name")));
            assert!(map.contains_key(&PropertyKey::new("age")));
        }

        // Overlay node has name.
        let vincent_map = &batch[batch.len() - 1];
        assert_eq!(
            vincent_map.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );

        // Missing node returns an empty map rather than panicking.
        let missing = NodeId::new(999_999);
        let batch_missing = layered.get_nodes_properties_batch(&[missing]);
        assert_eq!(batch_missing.len(), 1);
        assert!(batch_missing[0].is_empty());
    }

    #[test]
    fn test_get_node_deleted_returns_none() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        layered.delete_node(target);
        assert!(layered.get_node(target).is_none());

        // Property batch for a deleted node should also see empty entries.
        let batch = layered.get_nodes_properties_batch(&[target]);
        assert!(batch[0].is_empty());
    }

    #[test]
    fn test_get_edge_dirty_path_via_promotion() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Promote edge to overlay by setting a property on it.
        layered.set_edge_property(eid, "weight", Value::Float64(1.25));

        // get_edge should now return via overlay.
        let edge = layered.get_edge(eid).unwrap();
        assert_eq!(edge.edge_type.as_str(), "LIVES_IN");

        // edge_type should also route through overlay.
        assert_eq!(layered.edge_type(eid).as_deref(), Some("LIVES_IN"));

        // get_edge_property for a dirty edge reads from overlay.
        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(1.25));
    }

    #[test]
    fn test_get_edge_property_deleted() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        layered.delete_edge(eid);
        assert!(
            layered
                .get_edge_property(eid, &PropertyKey::new("since"))
                .is_none(),
            "deleted edge should not expose properties"
        );
        assert!(layered.edge_type(eid).is_none());
    }

    #[test]
    fn test_get_node_at_epoch_and_versioned_dirty() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);

        // Create an overlay (dirty) node.
        let jules = layered.create_node(&["Person"]);
        layered.set_node_property(jules, "name", Value::from("Jules"));

        // Dirty branch for get_node_at_epoch and get_node_versioned.
        assert!(layered.get_node_at_epoch(jules, epoch).is_some());
        assert!(layered.get_node_versioned(jules, epoch, txn_id).is_some());
    }

    #[test]
    fn test_get_edge_at_epoch_and_versioned_dirty() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);

        // Overlay edge between overlay nodes.
        let django = layered.create_node(&["Person"]);
        let prague = layered.create_node(&["City"]);
        let eid = layered.create_edge(django, prague, "VISITS");

        assert!(layered.get_edge_at_epoch(eid, epoch).is_some());
        assert!(layered.get_edge_versioned(eid, epoch, txn_id).is_some());
    }

    // ── O. Delete branches ───────────────────────────────────────────

    #[test]
    fn test_delete_nonexistent_node_returns_false() {
        let layered = build_test_layered();
        let missing = NodeId::new(999_999);
        assert!(!layered.delete_node(missing));

        let txn_id = TransactionId::from(1);
        let epoch = EpochId::from(u64::MAX);
        assert!(!layered.delete_node_versioned(missing, epoch, txn_id));
    }

    #[test]
    fn test_delete_nonexistent_edge_returns_false() {
        let layered = build_test_layered();
        let missing = EdgeId::new(999_999);
        assert!(!layered.delete_edge(missing));

        let txn_id = TransactionId::from(1);
        let epoch = EpochId::from(u64::MAX);
        assert!(!layered.delete_edge_versioned(missing, epoch, txn_id));
    }

    #[test]
    fn test_delete_dirty_node_via_overlay() {
        let layered = build_test_layered();
        // Create an overlay-only node then delete it through the dirty branch.
        let shosanna = layered.create_node(&["Person"]);
        assert!(layered.get_node(shosanna).is_some());
        assert!(layered.delete_node(shosanna));
        assert!(layered.get_node(shosanna).is_none());
    }

    #[test]
    fn test_delete_dirty_edge_via_overlay() {
        let layered = build_test_layered();
        let hans = layered.create_node(&["Person"]);
        let berlin = layered.create_node(&["City"]);
        let eid = layered.create_edge(hans, berlin, "LIVES_IN");

        assert!(layered.delete_edge(eid));
        assert!(layered.get_edge(eid).is_none());
    }

    #[test]
    fn test_delete_base_node_versioned() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");

        // Base-path deletion via versioned delete.
        assert!(layered.delete_node_versioned(persons[0], epoch, txn_id));
        assert!(layered.get_node(persons[0]).is_none());
    }

    #[test]
    fn test_delete_base_edge_versioned() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        assert!(layered.delete_edge_versioned(eid, epoch, txn_id));
        assert!(layered.get_edge(eid).is_none());
    }

    #[test]
    fn test_delete_node_edges_on_dirty_source() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Promote the source node into the overlay.
        layered.set_node_property(first, "city", Value::from("Berlin"));
        assert!(layered.overlay.load().get_node(first).is_some());

        // delete_node_edges should now cascade through both overlay and base edges.
        layered.delete_node_edges(first);
        let remaining = layered.edges_from(first, Direction::Outgoing);
        assert!(
            remaining.is_empty(),
            "edges from a dirty source should be fully removed"
        );
    }

    // ── P. Versioned property/label mutations ────────────────────────

    #[test]
    fn test_set_node_property_versioned_promotes_base() {
        let layered = build_test_layered();
        let txn_id = TransactionId::from(42);
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Versioned set promotes the base node into the overlay.
        layered.set_node_property_versioned(first, "city", Value::from("Paris"), txn_id);
        let city = layered
            .get_node_property(first, &PropertyKey::new("city"))
            .unwrap();
        assert_eq!(city, Value::String(ArcStr::from("Paris")));
    }

    #[test]
    fn test_set_edge_property_versioned_promotes_base() {
        let layered = build_test_layered();
        let txn_id = TransactionId::from(7);
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Versioned edge property set promotes the edge and its endpoints.
        layered.set_edge_property_versioned(eid, "weight", Value::Float64(3.5), txn_id);
        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(3.5));
    }

    #[test]
    fn test_remove_node_property_versioned_on_overlay_node() {
        // Use an overlay-only node to avoid the epoch-ordering restriction
        // that exists when promoting base nodes and then doing versioned removes.
        let layered = build_test_layered();
        let txn_id = TransactionId::from(101);

        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "email", Value::from("mia@example.com"));

        let removed = layered.remove_node_property_versioned(mia, "email", txn_id);
        assert_eq!(
            removed,
            Some(Value::String(ArcStr::from("mia@example.com")))
        );
        assert!(
            layered
                .get_node_property(mia, &PropertyKey::new("email"))
                .is_none()
        );
    }

    #[test]
    fn test_remove_edge_property_versioned_on_overlay_edge() {
        // Use an overlay-only edge to avoid epoch-ordering restrictions.
        let layered = build_test_layered();
        let txn_id = TransactionId::from(202);

        let django = layered.create_node(&["Person"]);
        let paris = layered.create_node(&["City"]);
        let eid = layered.create_edge(django, paris, "VISITS");
        layered.set_edge_property(eid, "year", Value::Int64(2024));

        let removed = layered.remove_edge_property_versioned(eid, "year", txn_id);
        assert_eq!(removed, Some(Value::Int64(2024)));
        assert!(
            layered
                .get_edge_property(eid, &PropertyKey::new("year"))
                .is_none()
        );
    }

    #[test]
    fn test_add_and_remove_label_versioned_on_overlay_node() {
        // Use an overlay-only node to avoid epoch-ordering issues that can occur
        // when promoting a base node and then writing versioned labels on top of
        // the epoch-0 promotion entry.
        let layered = build_test_layered();
        let txn_id = TransactionId::from(11);

        let butch = layered.create_node(&["Person"]);
        assert!(layered.add_label_versioned(butch, "Employee", txn_id));

        let node = layered.get_node(butch).unwrap();
        let labels: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
        assert!(labels.contains(&"Employee"));
        assert!(labels.contains(&"Person"));

        assert!(layered.remove_label_versioned(butch, "Employee", txn_id));
        let node = layered.get_node(butch).unwrap();
        let labels: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
        assert!(!labels.contains(&"Employee"));
    }

    // ── Q. ensure_in_overlay / ensure_edge_in_overlay edge cases ─────

    #[test]
    fn test_ensure_in_overlay_noop_for_nonexistent_node() {
        let layered = build_test_layered();
        let missing = NodeId::new(999_999);

        // set_node_property on a non-existent node should not crash; ensure_in_overlay
        // takes the "not in base either" early return.
        layered.set_node_property(missing, "name", Value::from("Ghost"));

        // The phantom property lands in the overlay even though the node does not
        // exist in either layer, so verify the path did not panic and no base node
        // appeared.
        assert!(layered.base_store_arc().get_node(missing).is_none());
    }

    #[test]
    fn test_ensure_edge_in_overlay_noop_for_nonexistent_edge() {
        let layered = build_test_layered();
        let missing = EdgeId::new(999_999);

        // set_edge_property on a missing edge should take the "not in base" branch.
        layered.set_edge_property(missing, "weight", Value::Float64(1.0));
        assert!(layered.base_store_arc().get_edge(missing).is_none());
    }

    #[test]
    fn test_ensure_edge_in_overlay_idempotent() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // First call promotes the edge; second call should take the early return.
        layered.set_edge_property(eid, "weight", Value::Float64(1.0));
        layered.set_edge_property(eid, "weight", Value::Float64(2.0));

        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(2.0));
    }

    // ── R. Traversal edge-cases for deleted neighbors ────────────────

    #[test]
    fn test_neighbors_incoming_with_deleted_source() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        let persons = layered.nodes_by_label("Person");
        // Delete one of the LIVES_IN source nodes; amsterdam's incoming neighbors
        // should drop that deleted node.
        layered.delete_node(persons[0]);

        let incoming = layered.neighbors(amsterdam, Direction::Incoming);
        assert!(!incoming.contains(&persons[0]));
        assert_eq!(incoming.len(), 1);
    }

    #[test]
    fn test_edges_from_dirty_source_merges_layers() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Capture the base-tier outgoing edges before any promotion.
        let base_outgoing: Vec<(NodeId, EdgeId)> =
            layered.edges_from(first, Direction::Outgoing);
        assert!(
            !base_outgoing.is_empty(),
            "test fixture should give the first Person a base edge"
        );

        // Promote `first` into the overlay (ensure_in_overlay copies labels and
        // properties only — never adjacency) and add a fresh overlay-only edge.
        layered.set_node_property(first, "city", Value::from("Berlin"));
        let prague = layered.create_node(&["City"]);
        layered.create_edge(first, prague, "VISITS");

        let outgoing = layered.edges_from(first, Direction::Outgoing);

        // The new overlay edge must be visible …
        assert!(
            outgoing.iter().any(|(target, _)| *target == prague),
            "new overlay edge should appear in edges_from"
        );

        // … and every pre-promotion base edge must remain visible. ensure_in_overlay
        // only promotes the node; base adjacency is still authoritative for edges
        // that pre-date the promotion, and must not silently disappear once the
        // source node becomes dirty.
        for (target, eid) in &base_outgoing {
            assert!(
                outgoing
                    .iter()
                    .any(|(t, e)| t == target && e == eid),
                "base edge {eid:?} (→ {target:?}) must remain visible after promotion"
            );
        }
    }

    /// Regression for property-anchored edge lookups across the snapshot
    /// boundary: when a base node is promoted into the overlay (e.g. by
    /// `set_node_property` or by becoming the endpoint of a new overlay
    /// edge), its pre-existing base-tier edges must remain reachable via
    /// `neighbors` and `edges_from` in BOTH directions. Otherwise GQL
    /// patterns like `MATCH (a {id: $x})-[:T]->(b {id: $y})` start
    /// returning zero rows once any overlay write has touched either
    /// endpoint, which the planner uses to walk the edge from.
    #[test]
    fn test_base_edge_visible_from_promoted_endpoint() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let cities = layered.nodes_by_label("City");
        let alix = persons[0];
        let amsterdam = cities[0];

        // Sanity: the base edge alix -LIVES_IN-> amsterdam exists pre-promotion.
        let pre_out = layered.edges_from(alix, Direction::Outgoing);
        let pre_in = layered.edges_from(amsterdam, Direction::Incoming);
        let pre_neigh_out = layered.neighbors(alix, Direction::Outgoing);
        let pre_neigh_in = layered.neighbors(amsterdam, Direction::Incoming);
        assert!(pre_out.iter().any(|(t, _)| *t == amsterdam));
        assert!(pre_in.iter().any(|(t, _)| *t == alix));
        assert!(pre_neigh_out.contains(&amsterdam));
        assert!(pre_neigh_in.contains(&alix));

        // Promote BOTH endpoints into the overlay by way of an unrelated
        // overlay write. The new edge intentionally points at a brand-new
        // overlay node so the LIVES_IN edge between alix and amsterdam is
        // not touched in any way.
        let oslo = layered.create_node(&["City"]);
        layered.create_edge(alix, oslo, "VISITS"); // promotes alix
        layered.set_node_property(amsterdam, "touched", Value::Bool(true)); // promotes amsterdam

        // Both endpoints are now dirty.
        assert!(layered.is_node_dirty(alix));
        assert!(layered.is_node_dirty(amsterdam));

        // The pre-existing base edge must still be reachable from either side,
        // both as an edge (with its original EdgeId) and as a neighbor.
        let post_out = layered.edges_from(alix, Direction::Outgoing);
        let post_in = layered.edges_from(amsterdam, Direction::Incoming);
        let post_neigh_out = layered.neighbors(alix, Direction::Outgoing);
        let post_neigh_in = layered.neighbors(amsterdam, Direction::Incoming);

        let base_eid = pre_out
            .iter()
            .find(|(t, _)| *t == amsterdam)
            .map(|(_, e)| *e)
            .expect("pre-promotion fixture has a base LIVES_IN edge");

        assert!(
            post_out.iter().any(|(t, e)| *t == amsterdam && *e == base_eid),
            "base LIVES_IN edge must remain visible via edges_from(src, Outgoing) after src is promoted"
        );
        assert!(
            post_in.iter().any(|(t, e)| *t == alix && *e == base_eid),
            "base LIVES_IN edge must remain visible via edges_from(dst, Incoming) after dst is promoted"
        );
        assert!(
            post_neigh_out.contains(&amsterdam),
            "base neighbor must remain visible via neighbors(src, Outgoing) after src is promoted"
        );
        assert!(
            post_neigh_in.contains(&alix),
            "base neighbor must remain visible via neighbors(dst, Incoming) after dst is promoted"
        );
    }

    // ── Phase 5c: overlay reset + in-place merge ──────────────────────

    /// `reset_overlay` swaps in a fresh empty `LpgStore` and clears
    /// dirty/deleted bookkeeping. Base reads keep working. Overlay-only
    /// nodes disappear (they were never persisted to base).
    #[test]
    fn alix_reset_overlay_clears_mutations_preserves_base() {
        let layered = build_test_layered();
        let base_persons_before = layered.nodes_by_label("Person").len();

        // Add an overlay node + delete a base node + dirty a base property.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));
        let base_persons = layered.nodes_by_label("Person");
        let to_delete = base_persons[0];
        layered.delete_node(to_delete);
        let still_alive = base_persons[1];
        layered.set_node_property(still_alive, "tagged", Value::from("hot"));

        assert!(layered.overlay_mutation_count() > 0);

        // Reset.
        layered.reset_overlay();

        // After reset: overlay is empty, base reads intact.
        assert_eq!(
            layered.overlay_mutation_count(),
            0,
            "overlay must be empty after reset"
        );
        assert_eq!(
            layered.nodes_by_label("Person").len(),
            base_persons_before,
            "base nodes restored (delete was overlay-only)"
        );
        assert!(
            layered.get_node(vincent).is_none(),
            "overlay-only node disappears after reset"
        );
        let base_node = layered.get_node(still_alive).unwrap();
        assert!(
            !base_node
                .properties
                .contains_key(&PropertyKey::new("tagged")),
            "base property dirty was reset"
        );
    }

    /// `merge_overlay_in_place` rebuilds the base from the combined view,
    /// swaps the base, and clears the overlay. After the call: all
    /// previously-visible data is in the base, overlay is empty.
    #[test]
    fn gus_merge_overlay_in_place_promotes_mutations_into_base() {
        let layered = build_test_layered();
        let count_before = layered.node_count();

        // Add an overlay node.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));
        layered.set_node_property(vincent, "age", Value::Int64(33));

        let count_after_mutation = layered.node_count();
        assert_eq!(count_after_mutation, count_before + 1);

        // Merge.
        layered
            .merge_overlay_in_place()
            .expect("merge_overlay_in_place");

        // Overlay empty; total node count preserved.
        assert_eq!(
            layered.overlay_mutation_count(),
            0,
            "overlay must be empty after merge"
        );
        assert_eq!(
            layered.node_count(),
            count_after_mutation,
            "total node count preserved across merge"
        );

        // Vincent now lives in the new base — verify by checking that
        // resetting the overlay would NOT make him disappear (post-merge,
        // he's part of the base).
        layered.reset_overlay();
        let still_there = layered.get_node(vincent);
        assert!(
            still_there.is_some(),
            "merged node persists after a subsequent reset_overlay (it's in base)"
        );
    }

    /// Round-trip property: merge then reset is a no-op on visible
    /// state.  Both operations leave the overlay empty.
    #[test]
    fn vincent_merge_then_reset_is_no_op_on_visible_state() {
        let layered = build_test_layered();
        let visible_before: Vec<NodeId> = layered.node_ids();

        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "name", Value::from("Mia"));

        layered.merge_overlay_in_place().unwrap();
        layered.reset_overlay();

        let visible_after: Vec<NodeId> = layered.node_ids();
        let mut a = visible_before;
        a.push(mia);
        a.sort_unstable();
        let mut b = visible_after;
        b.sort_unstable();
        assert_eq!(a, b);
    }

    // ── Phase 5d: concurrent base-swap + merge correctness ───────────

    /// Many readers + one swapper: swap_base should never produce a
    /// torn read or a panic.  Runs for a fixed iteration budget so
    /// the test stays bounded.
    #[test]
    fn jules_concurrent_readers_survive_repeated_base_swaps() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let layered = Arc::new(build_test_layered());
        let stop = Arc::new(AtomicBool::new(false));

        let mut readers = Vec::new();
        for _ in 0..4 {
            let l = Arc::clone(&layered);
            let s = Arc::clone(&stop);
            readers.push(thread::spawn(move || {
                let mut total = 0u64;
                while !s.load(Ordering::Relaxed) {
                    let people = l.nodes_by_label("Person");
                    // Person count is base(2) + overlay(0..many); never less than base.
                    assert!(people.len() >= 2, "lost a base node mid-swap");
                    total += people.len() as u64;
                }
                total
            }));
        }

        // Swapper builds a fresh base with one extra Person each round.
        let l = Arc::clone(&layered);
        let s = Arc::clone(&stop);
        let swapper = thread::spawn(move || {
            for _ in 0..200 {
                if s.load(Ordering::Relaxed) {
                    break;
                }
                // Read the current combined view, build a new compact base.
                let new_base = from_graph_store_preserving_ids(&*l).unwrap();
                l.swap_base(Arc::new(new_base));
            }
        });

        swapper.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        let totals: Vec<u64> = readers.into_iter().map(|h| h.join().unwrap()).collect();
        // Sanity: every reader observed at least one snapshot.
        for t in totals {
            assert!(t > 0, "reader saw zero snapshots");
        }
    }

    /// Concurrent readers + writer + periodic merge_overlay_in_place.
    /// The merger thread races with both reads and writes; correctness
    /// requirement is that all writes that completed before a join
    /// remain visible after the test.
    #[test]
    fn shosanna_concurrent_writes_survive_periodic_merge() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let layered = Arc::new(build_test_layered());
        let stop = Arc::new(AtomicBool::new(false));
        let writer_count = Arc::new(AtomicUsize::new(0));

        // Readers: spin reading.
        let mut readers = Vec::new();
        for _ in 0..2 {
            let l = Arc::clone(&layered);
            let s = Arc::clone(&stop);
            readers.push(thread::spawn(move || {
                let mut iters = 0u64;
                while !s.load(Ordering::Relaxed) {
                    let _ = l.nodes_by_label("Person");
                    iters += 1;
                }
                iters
            }));
        }

        // Writer: insert a flurry of nodes.
        let l = Arc::clone(&layered);
        let s = Arc::clone(&stop);
        let wc = Arc::clone(&writer_count);
        let writer = thread::spawn(move || {
            for i in 0..500 {
                if s.load(Ordering::Relaxed) {
                    break;
                }
                let id = l.create_node(&["Person"]);
                l.set_node_property(id, "tag", Value::Int64(i));
                wc.fetch_add(1, Ordering::Relaxed);
            }
        });

        // Merger: periodically merge while writer/readers are active.
        let l = Arc::clone(&layered);
        let s = Arc::clone(&stop);
        let merger = thread::spawn(move || {
            for _ in 0..20 {
                if s.load(Ordering::Relaxed) {
                    break;
                }
                let _ = l.merge_overlay_in_place();
                std::thread::yield_now();
            }
        });

        writer.join().unwrap();
        merger.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        for h in readers {
            h.join().unwrap();
        }

        // Final invariant: total Person count = 2 (base) + writer_count.
        let expected = 2 + writer_count.load(Ordering::Relaxed);
        let actual = layered.nodes_by_label("Person").len();
        assert_eq!(
            actual, expected,
            "lost writes during concurrent merge; expected {expected}, got {actual}"
        );
    }

    /// reset_overlay under concurrent readers: snapshots taken before
    /// the reset must remain valid.  This exercises ArcSwap snapshot
    /// semantics: a reader holding `overlay_store()` keeps the old
    /// LpgStore alive even after the overlay is replaced.
    #[test]
    fn beatrix_reset_overlay_does_not_invalidate_held_snapshots() {
        use std::sync::Arc;
        use std::thread;

        let layered = Arc::new(build_test_layered());

        // Add a node so the overlay is non-empty.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        // Take a snapshot of the overlay; if reset_overlay swaps, this
        // Arc should keep the old LpgStore alive and queryable.
        let snapshot = layered.overlay_store();
        assert!(snapshot.get_node(vincent).is_some());

        // Reset on a thread to maximise the race window.
        let l = Arc::clone(&layered);
        let resetter = thread::spawn(move || {
            l.reset_overlay();
        });
        resetter.join().unwrap();

        // Snapshot still has Vincent; live overlay does not.
        assert!(
            snapshot.get_node(vincent).is_some(),
            "snapshot must remain valid after concurrent reset"
        );
        assert!(
            layered.overlay_store().get_node(vincent).is_none(),
            "live overlay is empty post-reset"
        );
    }

    #[test]
    fn test_has_property_index_false_when_no_index() {
        let layered = build_test_layered();
        assert!(!layered.has_property_index("name"));
    }

    #[test]
    fn test_has_property_index_true_when_overlay_has_index() {
        // Regression test: LayeredStore::has_property_index must call
        // self.overlay.load().has_property_index(), not
        // self.overlay.has_property_index() (ArcSwap does not impl GraphStore).
        let layered = build_test_layered();
        layered.overlay_store().create_property_index("name");
        assert!(layered.has_property_index("name"));
        assert!(!layered.has_property_index("age"));
    }
}
