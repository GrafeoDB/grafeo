//! HybridStore: columnar base with mutable MVCC overlay.

pub mod id;
mod reader;
mod writer;

#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, TransactionId};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_common::utils::hash::FxHashSet;
use parking_lot::RwLock;

use grafeo_common::types::PropertyKey;

use crate::graph::compact::from_graph_store_incremental;
use crate::graph::compact::id::{decode_edge_id, decode_node_id};
use crate::graph::compact::CompactStore;
use crate::graph::lpg::LpgStore;
use crate::graph::traits::GraphStore;

use self::id::compute_id_offset;

/// Entity ID for tracking pending deletions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EntityId {
    Node(NodeId),
    Edge(EdgeId),
}

/// Per-entity dirty bitmap for skipping overlay lookups on clean compact entities.
///
/// Organized as per-table bitsets: `bits[table_id][word_index]` where each u64
/// word holds 64 entity flags. Checking dirtiness is two array lookups + one
/// bit test (~1-2ns, L1-friendly) vs a hash lookup (~5-10ns, cache-hostile).
///
/// Also tracks which property columns have any overlay modifications, so scans
/// on untouched columns skip the overlay entirely.
pub(crate) struct DirtySet {
    /// Per-table entity bitmaps. `entity_bits[table_id][offset / 64] & (1 << offset % 64)`.
    entity_bits: Vec<Vec<u64>>,
    /// Property columns with at least one overlay modification on a compact entity.
    dirty_columns: FxHashSet<PropertyKey>,
}

impl DirtySet {
    fn new(table_sizes: &[usize]) -> Self {
        let entity_bits = table_sizes
            .iter()
            .map(|&size| vec![0u64; (size + 63) / 64])
            .collect();
        Self {
            entity_bits,
            dirty_columns: FxHashSet::default(),
        }
    }

    /// Marks a compact entity as dirty (has overlay modifications).
    #[inline]
    fn mark_entity(&mut self, table_id: u16, offset: u64) {
        if let Some(words) = self.entity_bits.get_mut(table_id as usize) {
            let word_idx = (offset / 64) as usize;
            if word_idx < words.len() {
                words[word_idx] |= 1 << (offset % 64);
            }
        }
    }

    /// Marks a property column as having overlay modifications.
    #[inline]
    fn mark_column(&mut self, key: PropertyKey) {
        self.dirty_columns.insert(key);
    }

    /// Returns `true` if this compact entity has overlay modifications.
    #[inline]
    fn is_dirty(&self, table_id: u16, offset: u64) -> bool {
        self.entity_bits
            .get(table_id as usize)
            .and_then(|words| words.get((offset / 64) as usize))
            .is_some_and(|&word| (word >> (offset % 64)) & 1 == 1)
    }

    /// Returns `true` if this property column has any overlay modifications
    /// on compact entities.
    #[inline]
    fn is_column_dirty(&self, key: &PropertyKey) -> bool {
        self.dirty_columns.contains(key)
    }

    /// Returns `true` if any entity in the given table has overlay modifications.
    fn has_dirty_in_table(&self, table_id: u16) -> bool {
        self.entity_bits
            .get(table_id as usize)
            .is_some_and(|words| words.iter().any(|&w| w != 0))
    }

    /// Check dirtiness given a pre-decoded (table_id, offset). No lock needed
    /// when the caller already holds the `RwLockReadGuard`.
    #[inline]
    pub(crate) fn id_is_dirty_node(dirty: &DirtySet, id: NodeId) -> bool {
        let (table_id, offset) = decode_node_id(id);
        dirty.is_dirty(table_id, offset)
    }

    fn clear_and_resize(&mut self, table_sizes: &[usize]) {
        self.entity_bits = table_sizes
            .iter()
            .map(|&size| vec![0u64; (size + 63) / 64])
            .collect();
        self.dirty_columns.clear();
    }
}

/// Hybrid graph store combining a frozen columnar base with a mutable MVCC overlay.
///
/// Reads merge both stores transparently. Writes go to the overlay. The compact
/// store is never mutated after construction.
pub struct HybridStore {
    compact: RwLock<CompactStore>,
    /// Never replaced — cleared in-place during compaction.
    /// No lock needed; LpgStore handles its own internal synchronization.
    overlay: Arc<LpgStore>,
    deleted_nodes: Arc<RwLock<FxHashSet<NodeId>>>,
    deleted_edges: Arc<RwLock<FxHashSet<EdgeId>>>,
    pending_deletions: Arc<RwLock<FxHashMap<TransactionId, Vec<EntityId>>>>,
    /// Per-entity bitmap + per-column flags for skipping overlay lookups.
    dirty_nodes: RwLock<DirtySet>,
    dirty_edges: RwLock<DirtySet>,
    compact_node_count: AtomicUsize,
    compact_edge_count: AtomicUsize,
    id_offset: AtomicU64,
}

impl std::fmt::Debug for HybridStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridStore")
            .field("compact_node_count", &self.compact_node_count.load(Ordering::Relaxed))
            .field("compact_edge_count", &self.compact_edge_count.load(Ordering::Relaxed))
            .field("id_offset", &self.id_offset.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl HybridStore {
    /// Opens a `HybridStore` wrapping the given compact store.
    ///
    /// Computes an ID offset so overlay IDs never collide with compact IDs.
    ///
    /// # Errors
    ///
    /// Returns [`grafeo_common::memory::AllocError`] if the overlay `LpgStore`
    /// cannot be allocated.
    pub fn open(compact: CompactStore) -> Result<Self, grafeo_common::memory::AllocError> {
        let offset = compute_id_offset(&compact);
        let compact_node_count = compact.node_count();
        let compact_edge_count = compact.edge_count();

        let overlay = Arc::new(LpgStore::new()?);
        // Set overlay IDs to start above compact range so IDs never collide.
        overlay.set_next_node_id(offset);
        overlay.set_next_edge_id(offset);

        let deleted_nodes = Arc::new(RwLock::new(FxHashSet::default()));
        let deleted_edges = Arc::new(RwLock::new(FxHashSet::default()));
        let pending_deletions = Arc::new(RwLock::new(FxHashMap::default()));

        Self::register_hooks(&overlay, &deleted_nodes, &deleted_edges, &pending_deletions);

        let node_table_sizes: Vec<usize> = compact
            .table_id_to_label()
            .iter()
            .filter_map(|label| compact.node_table(label))
            .map(|nt| nt.len())
            .collect();
        let edge_table_sizes: Vec<usize> = compact
            .rel_tables_by_id()
            .iter()
            .map(|rt| rt.num_edges())
            .collect();

        Ok(Self {
            compact: RwLock::new(compact),
            overlay,
            deleted_nodes,
            deleted_edges,
            pending_deletions,
            dirty_nodes: RwLock::new(DirtySet::new(&node_table_sizes)),
            dirty_edges: RwLock::new(DirtySet::new(&edge_table_sizes)),
            compact_node_count: AtomicUsize::new(compact_node_count),
            compact_edge_count: AtomicUsize::new(compact_edge_count),
            id_offset: AtomicU64::new(offset),
        })
    }

    /// Returns a reference to the overlay `LpgStore`.
    pub fn overlay(&self) -> &Arc<LpgStore> {
        &self.overlay
    }

    /// Returns the ID boundary value.
    ///
    /// Compact IDs are strictly below this value; overlay IDs are at or above it.
    pub fn id_offset(&self) -> u64 {
        self.id_offset.load(Ordering::Relaxed)
    }

    /// Returns `true` if `id` belongs to the compact store (below the offset).
    pub fn is_compact_node_id(&self, id: NodeId) -> bool {
        id.as_u64() < self.id_offset.load(Ordering::Relaxed)
    }

    /// Returns `true` if `id` belongs to the compact store (below the offset).
    pub fn is_compact_edge_id(&self, id: EdgeId) -> bool {
        id.as_u64() < self.id_offset.load(Ordering::Relaxed)
    }

    /// Marks a compact node as dirty (has overlay modifications).
    pub(crate) fn mark_node_dirty(&self, id: NodeId, column: Option<&str>) {
        let (table_id, offset) = decode_node_id(id);
        let mut dirty = self.dirty_nodes.write();
        dirty.mark_entity(table_id, offset);
        if let Some(col) = column {
            dirty.mark_column(PropertyKey::new(col));
        }
    }

    /// Marks a compact edge as dirty.
    pub(crate) fn mark_edge_dirty(&self, id: EdgeId, column: Option<&str>) {
        let (table_id, offset) = decode_edge_id(id);
        let mut dirty = self.dirty_edges.write();
        dirty.mark_entity(table_id, offset);
        if let Some(col) = column {
            dirty.mark_column(PropertyKey::new(col));
        }
    }

    /// Returns `true` if a compact node has overlay modifications.
    #[inline]
    pub(crate) fn is_node_dirty(&self, id: NodeId) -> bool {
        let (table_id, offset) = decode_node_id(id);
        self.dirty_nodes.read().is_dirty(table_id, offset)
    }

    /// Returns `true` if a compact edge has overlay modifications.
    #[inline]
    pub(crate) fn is_edge_dirty(&self, id: EdgeId) -> bool {
        let (table_id, offset) = decode_edge_id(id);
        self.dirty_edges.read().is_dirty(table_id, offset)
    }

    /// Returns `true` if the node has been soft-deleted via the overlay.
    fn is_node_deleted(&self, id: NodeId) -> bool {
        self.deleted_nodes.read().contains(&id)
    }

    /// Returns `true` if the edge has been soft-deleted via the overlay.
    fn is_edge_deleted(&self, id: EdgeId) -> bool {
        self.deleted_edges.read().contains(&id)
    }

    /// Called when a transaction is rolled back.
    /// Removes any compact entity deletions that were part of this transaction.
    pub fn on_rollback(&self, transaction_id: TransactionId) {
        // Remove from pending first, drop the lock, then apply to deletion sets
        let deletions = self.pending_deletions.write().remove(&transaction_id);
        if let Some(deletions) = deletions {
            let mut del_nodes = self.deleted_nodes.write();
            let mut del_edges = self.deleted_edges.write();
            for entity in deletions {
                match entity {
                    EntityId::Node(id) => { del_nodes.remove(&id); }
                    EntityId::Edge(id) => { del_edges.remove(&id); }
                }
            }
        }
    }

    /// Called when a transaction is committed.
    /// Clears the pending deletion tracking (deletions are permanent).
    pub fn on_commit(&self, transaction_id: TransactionId) {
        self.pending_deletions.write().remove(&transaction_id);
    }

    /// Incrementally compacts the hybrid store.
    ///
    /// Unchanged node/rel tables (no dirty entities, no deletions, no new overlay
    /// entities with that label/type) are cloned as-is. Only tables with
    /// modifications are rebuilt from the merged `GraphStore` view.
    ///
    /// Caller must ensure no active transactions.
    ///
    /// # Errors
    ///
    /// Returns a `String` if the rebuild fails.
    pub fn compact(&self) -> Result<(), String> {
        let dirty_n = self.dirty_nodes.read();
        let dirty_e = self.dirty_edges.read();
        let deleted_n = self.deleted_nodes.read();
        let deleted_e = self.deleted_edges.read();

        // Check if anything has changed at all (fast no-op path).
        let compact = self.compact.read();
        let overlay_labels = GraphStore::all_labels(self.overlay.as_ref());
        let has_new_labels = overlay_labels
            .iter()
            .any(|l| !compact.label_to_table_id_map().contains_key(l.as_str()));
        let has_any_overlay_edges = GraphStore::edge_count(self.overlay.as_ref()) > 0;
        let has_overlay_nodes = GraphStore::node_count(self.overlay.as_ref()) > 0;

        let any_node_table_dirty = (0..compact.node_tables_by_id().len())
            .any(|tid| dirty_n.has_dirty_in_table(tid as u16));
        let any_rel_table_dirty = (0..compact.rel_tables_by_id().len())
            .any(|tid| dirty_e.has_dirty_in_table(tid as u16));
        let has_deletions = !deleted_n.is_empty() || !deleted_e.is_empty();

        let needs_rebuild =
            any_node_table_dirty || any_rel_table_dirty || has_deletions
            || has_new_labels || has_any_overlay_edges || has_overlay_nodes;

        if !needs_rebuild {
            return Ok(());
        }

        // Compute which node tables are unchanged:
        //   - no dirty entities in the table
        //   - no deleted nodes in the table
        //   - no overlay nodes with that label
        let overlay_label_set: FxHashSet<String> = overlay_labels.into_iter().collect();
        let mut unchanged_node_tables: FxHashSet<u16> = FxHashSet::default();
        for (tid, label) in compact.table_id_to_label().iter().enumerate() {
            let tid = tid as u16;
            let table_dirty = dirty_n.has_dirty_in_table(tid);
            let has_deleted = deleted_n.iter().any(|&nid| {
                let (t, _) = crate::graph::compact::id::decode_node_id(nid);
                t == tid
            });
            let has_overlay_label = overlay_label_set.contains(label.as_str());
            if !table_dirty && !has_deleted && !has_overlay_label {
                unchanged_node_tables.insert(tid);
            }
        }

        // Compute which rel tables are unchanged:
        //   - no dirty edges in the table
        //   - no deleted edges in the table
        //   - no overlay edges of that edge type
        //   - both src and dst node tables are unchanged
        let overlay_edge_types: FxHashSet<String> =
            GraphStore::all_edge_types(self.overlay.as_ref())
                .into_iter()
                .collect();
        let mut unchanged_rel_tables: FxHashSet<u16> = FxHashSet::default();
        for (rid, rt) in compact.rel_tables_by_id().iter().enumerate() {
            let rid = rid as u16;
            let rel_dirty = dirty_e.has_dirty_in_table(rid);
            let has_deleted_edge = deleted_e.iter().any(|&eid| {
                let (r, _) = crate::graph::compact::id::decode_edge_id(eid);
                r == rid
            });
            let edge_type = compact
                .edge_type_for_rel_table_id(rid)
                .map_or("", |s| s.as_str());
            let has_overlay_type = overlay_edge_types.contains(edge_type);
            let src_unchanged = unchanged_node_tables.contains(&rt.src_table_id());
            let dst_unchanged = unchanged_node_tables.contains(&rt.dst_table_id());
            if !rel_dirty && !has_deleted_edge && !has_overlay_type && src_unchanged && dst_unchanged
            {
                unchanged_rel_tables.insert(rid);
            }
        }

        drop(compact);
        drop(dirty_n);
        drop(dirty_e);
        drop(deleted_n);
        drop(deleted_e);

        let old_compact = self.compact.read();
        let merged = from_graph_store_incremental(
            self as &dyn crate::graph::traits::GraphStore,
            &old_compact,
            &unchanged_node_tables,
            &unchanged_rel_tables,
        )
        .map_err(|e| e.to_string())?;
        drop(old_compact);

        let new_offset = compute_id_offset(&merged);
        let new_node_count = merged.node_count();
        let new_edge_count = merged.edge_count();

        *self.compact.write() = merged;
        self.overlay.clear();
        self.overlay.set_next_node_id(new_offset);
        self.overlay.set_next_edge_id(new_offset);
        self.deleted_nodes.write().clear();
        self.deleted_edges.write().clear();
        self.pending_deletions.write().clear();
        {
            let compact = self.compact.read();
            let node_sizes: Vec<usize> = compact
                .table_id_to_label()
                .iter()
                .filter_map(|label| compact.node_table(label))
                .map(|nt| nt.len())
                .collect();
            let edge_sizes: Vec<usize> = compact
                .rel_tables_by_id()
                .iter()
                .map(|rt| rt.num_edges())
                .collect();
            self.dirty_nodes.write().clear_and_resize(&node_sizes);
            self.dirty_edges.write().clear_and_resize(&edge_sizes);
        }
        self.compact_node_count.store(new_node_count, Ordering::Release);
        self.compact_edge_count.store(new_edge_count, Ordering::Release);
        self.id_offset.store(new_offset, Ordering::Release);

        Ok(())
    }

    /// Registers rollback/commit hooks on an overlay for deletion tracking.
    fn register_hooks(
        overlay: &LpgStore,
        deleted_nodes: &Arc<RwLock<FxHashSet<NodeId>>>,
        deleted_edges: &Arc<RwLock<FxHashSet<EdgeId>>>,
        pending_deletions: &Arc<RwLock<FxHashMap<TransactionId, Vec<EntityId>>>>,
    ) {
        let dn = Arc::clone(deleted_nodes);
        let de = Arc::clone(deleted_edges);
        let pd = Arc::clone(pending_deletions);

        overlay.set_on_rollback_hook(Box::new(move |tx_id| {
            let deletions = pd.write().remove(&tx_id);
            if let Some(deletions) = deletions {
                let mut del_nodes = dn.write();
                let mut del_edges = de.write();
                for entity in deletions {
                    match entity {
                        EntityId::Node(id) => { del_nodes.remove(&id); }
                        EntityId::Edge(id) => { del_edges.remove(&id); }
                    }
                }
            }
        }));

        let pd2 = Arc::clone(pending_deletions);
        overlay.set_on_commit_hook(Box::new(move |tx_id| {
            pd2.write().remove(&tx_id);
        }));
    }
}
