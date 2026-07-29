use super::LpgStore;
use crate::graph::lpg::{EdgeRecord, NodeRecord};
use grafeo_common::memory::AllocError;
use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};
#[cfg(feature = "tiered-storage")]
use grafeo_common::utils::hash::FxHashMap;
use std::sync::atomic::Ordering;

#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;

#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{ColdVersionRef, HotVersionRef, VersionIndex};

impl LpgStore {
    /// Discards all uncommitted versions created by a transaction.
    ///
    /// This is called during transaction rollback to clean up uncommitted changes.
    /// The method removes version chain entries created by the specified transaction,
    /// erases secondary structures published for entities that were created inside
    /// the transaction (label/property indexes, adjacency, type counts, property
    /// columns, live counters), and replays the property undo log to restore
    /// property values on pre-existing entities.
    #[doc(hidden)]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn discard_uncommitted_versions(&self, transaction_id: TransactionId) {
        // Capture create-path entities before version chains are emptied so we
        // still have edge src/dst/type metadata for adjacency cleanup.
        let discarded_nodes = self.collect_solely_created_nodes(transaction_id);
        let discarded_edges = self.collect_solely_created_edges(transaction_id);

        // Remove uncommitted node versions
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.remove_versions_by(transaction_id);
            }
            // Remove completely empty chains (no versions left)
            nodes.retain(|_, chain| !chain.is_empty());
        }

        // Remove uncommitted edge versions
        {
            let mut edges = self.edges.write();
            for chain in edges.values_mut() {
                chain.remove_versions_by(transaction_id);
            }
            // Remove completely empty chains (no versions left)
            edges.retain(|_, chain| !chain.is_empty());
        }

        // Erase non-versioned secondary state for fully discarded creates.
        self.cleanup_discarded_node_secondaries(&discarded_nodes);
        self.cleanup_discarded_edge_secondaries(&discarded_edges);

        // Replay property undo log to restore pre-transaction property values
        self.rollback_transaction_properties(transaction_id);

        // Counters may be out of sync after rollback: force full recompute
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards uncommitted versions for specific entities created by a transaction.
    ///
    /// Used for savepoint rollback: only reverts the entities written after
    /// the savepoint, keeping earlier writes intact. Also erases secondary
    /// structures published for entities that this discard empties. Callers
    /// must run `rollback_transaction_properties_to` first so property undo
    /// for surviving entities is already applied.
    #[doc(hidden)]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn discard_entities_by_id(
        &self,
        transaction_id: TransactionId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        // Capture create-path secondaries before emptying version chains.
        let discarded_nodes: Vec<NodeId> = {
            let nodes = self.nodes.read();
            node_ids
                .iter()
                .copied()
                .filter(|id| {
                    nodes
                        .get(id)
                        .is_some_and(|chain| chain.solely_created_by(transaction_id))
                })
                .collect()
        };
        let discarded_edges: Vec<super::rollback_cleanup::DiscardedEdge> = {
            let edges = self.edges.read();
            edge_ids
                .iter()
                .filter_map(|&id| {
                    let chain = edges.get(&id)?;
                    if !chain.solely_created_by(transaction_id) {
                        return None;
                    }
                    let record = chain.latest()?;
                    Some(super::rollback_cleanup::DiscardedEdge {
                        id,
                        src: record.src,
                        dst: record.dst,
                        type_id: record.type_id,
                    })
                })
                .collect()
        };

        if !node_ids.is_empty() {
            let mut nodes = self.nodes.write();
            for &nid in node_ids {
                if let Some(chain) = nodes.get_mut(&nid) {
                    chain.remove_versions_by(transaction_id);
                    if chain.is_empty() {
                        nodes.remove(&nid);
                    }
                }
            }
        }

        if !edge_ids.is_empty() {
            let mut edges = self.edges.write();
            for &eid in edge_ids {
                if let Some(chain) = edges.get_mut(&eid) {
                    chain.remove_versions_by(transaction_id);
                    if chain.is_empty() {
                        edges.remove(&eid);
                    }
                }
            }
        }

        self.cleanup_discarded_node_secondaries(&discarded_nodes);
        self.cleanup_discarded_edge_secondaries(&discarded_edges);

        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards all uncommitted versions created by a transaction.
    /// (Tiered storage version)
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    pub fn discard_uncommitted_versions(&self, transaction_id: TransactionId) {
        let discarded_nodes = self.collect_solely_created_nodes(transaction_id);
        let discarded_edges = self.collect_solely_created_edges(transaction_id);

        // Remove uncommitted node versions
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.remove_versions_by(transaction_id);
            }
            // Remove completely empty indexes (no versions left)
            versions.retain(|_, index| !index.is_empty());
        }

        // Remove uncommitted edge versions
        {
            let mut versions = self.edge_versions.write();
            for index in versions.values_mut() {
                index.remove_versions_by(transaction_id);
            }
            // Remove completely empty indexes (no versions left)
            versions.retain(|_, index| !index.is_empty());
        }

        self.cleanup_discarded_node_secondaries(&discarded_nodes);
        self.cleanup_discarded_edge_secondaries(&discarded_edges);

        // Replay property undo log to restore pre-transaction property values
        self.rollback_transaction_properties(transaction_id);

        // Counters may be out of sync after rollback: force full recompute
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards uncommitted versions for specific entities (tiered storage version).
    ///
    /// Also erases secondary structures for entities emptied by this scoped
    /// discard. Callers must run `rollback_transaction_properties_to` first.
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    pub fn discard_entities_by_id(
        &self,
        transaction_id: TransactionId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        let discarded_nodes: Vec<NodeId> = {
            let versions = self.node_versions.read();
            node_ids
                .iter()
                .copied()
                .filter(|id| {
                    versions
                        .get(id)
                        .is_some_and(|index| index.solely_created_by(transaction_id))
                })
                .collect()
        };
        let discarded_edges: Vec<super::rollback_cleanup::DiscardedEdge> = {
            let versions = self.edge_versions.read();
            let mut out = Vec::new();
            for &id in edge_ids {
                let Some(index) = versions.get(&id) else {
                    continue;
                };
                if !index.solely_created_by(transaction_id) {
                    continue;
                }
                let Some(vref) = index
                    .visible_to(EpochId::PENDING, transaction_id)
                    .or_else(|| index.latest())
                else {
                    continue;
                };
                let Some(record) = self.read_edge_record(&vref) else {
                    continue;
                };
                out.push(super::rollback_cleanup::DiscardedEdge {
                    id,
                    src: record.src,
                    dst: record.dst,
                    type_id: record.type_id,
                });
            }
            out
        };

        if !node_ids.is_empty() {
            let mut versions = self.node_versions.write();
            for &nid in node_ids {
                if let Some(index) = versions.get_mut(&nid) {
                    index.remove_versions_by(transaction_id);
                    if index.is_empty() {
                        versions.remove(&nid);
                    }
                }
            }
        }

        if !edge_ids.is_empty() {
            let mut versions = self.edge_versions.write();
            for &eid in edge_ids {
                if let Some(index) = versions.get_mut(&eid) {
                    index.remove_versions_by(transaction_id);
                    if index.is_empty() {
                        versions.remove(&eid);
                    }
                }
            }
        }

        self.cleanup_discarded_node_secondaries(&discarded_nodes);
        self.cleanup_discarded_edge_secondaries(&discarded_edges);

        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Finalizes PENDING epochs for all versions created by a transaction.
    ///
    /// Called at commit time: updates `created_epoch` from `EpochId::PENDING`
    /// to the real `commit_epoch`, making the versions visible to other sessions.
    /// Also advances the store's epoch so non-transactional reads can see the
    /// newly committed versions.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn finalize_version_epochs(&self, transaction_id: TransactionId, commit_epoch: EpochId) {
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.finalize_epochs(transaction_id, commit_epoch);
            }
        }
        {
            let mut edges = self.edges.write();
            for chain in edges.values_mut() {
                chain.finalize_epochs(transaction_id, commit_epoch);
            }
        }

        // Finalize PENDING epochs in property and label version logs
        #[cfg(feature = "temporal")]
        {
            self.node_properties.finalize_pending(commit_epoch);
            self.edge_properties.finalize_pending(commit_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.finalize_pending(commit_epoch);
            }
        }

        self.sync_epoch(commit_epoch);
    }

    /// Finalizes PENDING epochs for all versions created by a transaction.
    /// (Tiered storage version, also syncs the store epoch.)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn finalize_version_epochs(&self, transaction_id: TransactionId, commit_epoch: EpochId) {
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.finalize_epochs(transaction_id, commit_epoch);
            }
        }
        {
            let mut versions = self.edge_versions.write();
            for index in versions.values_mut() {
                index.finalize_epochs(transaction_id, commit_epoch);
            }
        }

        // Finalize PENDING epochs in property and label version logs
        #[cfg(feature = "temporal")]
        {
            self.node_properties.finalize_pending(commit_epoch);
            self.edge_properties.finalize_pending(commit_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.finalize_pending(commit_epoch);
            }
        }

        self.sync_epoch(commit_epoch);
    }

    /// Garbage collects old versions that are no longer visible to any transaction.
    ///
    /// Versions older than `min_epoch` are pruned from version chains, keeping
    /// at most one old version per entity as a baseline. Empty chains are removed.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn gc_versions(&self, min_epoch: EpochId) {
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.gc(min_epoch);
            }
            nodes.retain(|_, chain| !chain.is_empty());
        }
        {
            let mut edges = self.edges.write();
            for chain in edges.values_mut() {
                chain.gc(min_epoch);
            }
            edges.retain(|_, chain| !chain.is_empty());
        }

        // GC old property and label versions
        #[cfg(feature = "temporal")]
        {
            self.node_properties.gc(min_epoch);
            self.edge_properties.gc(min_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.gc(min_epoch);
            }
            labels.retain(|_, log| !log.is_empty());
        }
    }

    /// Garbage collects old versions (tiered storage variant).
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn gc_versions(&self, min_epoch: EpochId) {
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.gc(min_epoch);
            }
            versions.retain(|_, index| !index.is_empty());
        }
        {
            let mut versions = self.edge_versions.write();
            for index in versions.values_mut() {
                index.gc(min_epoch);
            }
            versions.retain(|_, index| !index.is_empty());
        }

        // GC old property and label versions
        #[cfg(feature = "temporal")]
        {
            self.node_properties.gc(min_epoch);
            self.edge_properties.gc(min_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.gc(min_epoch);
            }
            labels.retain(|_, log| !log.is_empty());
        }
    }

    /// Freezes an epoch from hot (arena) storage to cold (compressed) storage.
    ///
    /// This is called by the transaction manager when an epoch becomes eligible
    /// for freezing (no active transactions can see it). The freeze process:
    ///
    /// 1. Collects all hot version refs for the epoch
    /// 2. Reads the corresponding records from arena
    /// 3. Compresses them into a `CompressedEpochBlock`
    /// 4. Updates `VersionIndex` entries to point to cold storage
    /// 5. The arena can be deallocated after all epochs in it are frozen
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch to freeze
    ///
    /// # Returns
    ///
    /// The number of records frozen (nodes + edges).
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub fn freeze_epoch(&self, epoch: EpochId) -> usize {
        // Collect node records to freeze
        let mut node_records: Vec<(u64, NodeRecord)> = Vec::new();
        let mut node_hot_refs: Vec<(NodeId, HotVersionRef)> = Vec::new();

        {
            let versions = self.node_versions.read();
            for (node_id, index) in versions.iter() {
                for hot_ref in index.hot_refs_for_epoch(epoch) {
                    let arena = self
                        .arena_allocator
                        .arena(hot_ref.arena_epoch)
                        .expect("arena epoch must exist for hot version ref");
                    // SAFETY: The offset was returned by alloc_value_with_offset for a NodeRecord
                    let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    node_records.push((node_id.as_u64(), *record));
                    node_hot_refs.push((*node_id, *hot_ref));
                }
            }
        }

        // Collect edge records to freeze
        let mut edge_records: Vec<(u64, EdgeRecord)> = Vec::new();
        let mut edge_hot_refs: Vec<(EdgeId, HotVersionRef)> = Vec::new();

        {
            let versions = self.edge_versions.read();
            for (edge_id, index) in versions.iter() {
                for hot_ref in index.hot_refs_for_epoch(epoch) {
                    let arena = self
                        .arena_allocator
                        .arena(hot_ref.arena_epoch)
                        .expect("arena epoch must exist for hot version ref");
                    // SAFETY: The offset was returned by alloc_value_with_offset for an EdgeRecord
                    let record: &EdgeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    edge_records.push((edge_id.as_u64(), *record));
                    edge_hot_refs.push((*edge_id, *hot_ref));
                }
            }
        }

        let total_frozen = node_records.len() + edge_records.len();

        if total_frozen == 0 {
            return 0;
        }

        // Freeze to compressed storage
        let (node_entries, edge_entries) =
            self.epoch_store
                .freeze_epoch(epoch, node_records, edge_records);

        // Build lookup maps for index entries
        let node_entry_map: FxHashMap<u64, _> = node_entries
            .iter()
            .map(|e| (e.entity_id, (e.offset, e.length)))
            .collect();
        let edge_entry_map: FxHashMap<u64, _> = edge_entries
            .iter()
            .map(|e| (e.entity_id, (e.offset, e.length)))
            .collect();

        // Update version indexes to use cold refs
        {
            let mut versions = self.node_versions.write();
            for (node_id, hot_ref) in &node_hot_refs {
                if let Some(index) = versions.get_mut(node_id)
                    && let Some(&(offset, length)) = node_entry_map.get(&node_id.as_u64())
                {
                    let cold_ref = ColdVersionRef {
                        epoch,
                        block_offset: offset,
                        length,
                        created_by: hot_ref.created_by,
                        deleted_epoch: hot_ref.deleted_epoch,
                        deleted_by: hot_ref.deleted_by,
                    };
                    index.freeze_epoch(epoch, std::iter::once(cold_ref));
                }
            }
        }

        {
            let mut versions = self.edge_versions.write();
            for (edge_id, hot_ref) in &edge_hot_refs {
                if let Some(index) = versions.get_mut(edge_id)
                    && let Some(&(offset, length)) = edge_entry_map.get(&edge_id.as_u64())
                {
                    let cold_ref = ColdVersionRef {
                        epoch,
                        block_offset: offset,
                        length,
                        created_by: hot_ref.created_by,
                        deleted_epoch: hot_ref.deleted_epoch,
                        deleted_by: hot_ref.deleted_by,
                    };
                    index.freeze_epoch(epoch, std::iter::once(cold_ref));
                }
            }
        }

        total_frozen
    }

    /// Returns the epoch store for cold storage statistics.
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    #[must_use]
    pub fn epoch_store(&self) -> &crate::codec::EpochStore {
        &self.epoch_store
    }

    // === Recovery Support ===

    /// Creates a node with a specific ID during recovery.
    ///
    /// This is used for WAL recovery to restore nodes with their original IDs.
    /// The caller must ensure IDs don't conflict with existing nodes.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot allocate space
    /// (only possible with the `tiered-storage` feature).
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_node_with_id(&self, id: NodeId, labels: &[&str]) -> Result<(), AllocError> {
        let epoch = self.current_epoch();
        let mut record = NodeRecord::new(id, epoch);
        // reason: label count per node is bounded by practical limits, fits u16
        #[allow(clippy::cast_possible_truncation)]
        record.set_label_count(labels.len() as u16);

        #[cfg(not(feature = "temporal"))]
        self.register_node_labels(id, labels);
        #[cfg(feature = "temporal")]
        self.register_node_labels(id, labels, epoch);

        // Create version chain with initial version (using SYSTEM tx for recovery)
        let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
        self.nodes.write().insert(id, chain);
        self.live_node_count.fetch_add(1, Ordering::Relaxed);

        // Update next_node_id if necessary to avoid future collisions
        let id_val = id.as_u64();
        let _ = self
            .next_node_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        Ok(())
    }

    /// Creates a node with a specific ID during recovery.
    /// (Tiered storage version)
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot create an epoch
    /// or allocate space for the node record.
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_node_with_id(&self, id: NodeId, labels: &[&str]) -> Result<(), AllocError> {
        let epoch = self.current_epoch();
        let mut record = NodeRecord::new(id, epoch);
        // reason: label count per node is bounded by practical limits, fits u16
        #[allow(clippy::cast_possible_truncation)]
        record.set_label_count(labels.len() as u16);

        #[cfg(not(feature = "temporal"))]
        self.register_node_labels(id, labels);
        #[cfg(feature = "temporal")]
        self.register_node_labels(id, labels, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch)?;
        let (offset, _stored) = arena.alloc_value_with_offset(record)?;

        // Create HotVersionRef (using SYSTEM tx for recovery)
        let hot_ref = HotVersionRef::new(epoch, epoch, offset, TransactionId::SYSTEM);
        let mut versions = self.node_versions.write();
        versions.insert(id, VersionIndex::with_initial(hot_ref));
        self.live_node_count.fetch_add(1, Ordering::Relaxed);

        // Update next_node_id if necessary to avoid future collisions
        let id_val = id.as_u64();
        let _ = self
            .next_node_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        Ok(())
    }

    /// Creates an edge with a specific ID during recovery.
    ///
    /// This is used for WAL recovery to restore edges with their original IDs.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot allocate space
    /// (only possible with the `tiered-storage` feature).
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<(), AllocError> {
        let epoch = self.current_epoch();
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);
        let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
        self.edges.write().insert(id, chain);

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);

        // Update next_edge_id if necessary
        let id_val = id.as_u64();
        let _ = self
            .next_edge_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        Ok(())
    }

    /// Creates an edge with a specific ID during recovery.
    /// (Tiered storage version)
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot create an epoch
    /// or allocate space for the edge record.
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<(), AllocError> {
        let epoch = self.current_epoch();
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch)?;
        let (offset, _stored) = arena.alloc_value_with_offset(record)?;

        // Create HotVersionRef (using SYSTEM tx for recovery)
        let hot_ref = HotVersionRef::new(epoch, epoch, offset, TransactionId::SYSTEM);
        let mut versions = self.edge_versions.write();
        versions.insert(id, VersionIndex::with_initial(hot_ref));

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);

        // Update next_edge_id if necessary
        let id_val = id.as_u64();
        let _ = self
            .next_edge_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        Ok(())
    }

    /// Sets the current epoch during recovery.
    #[doc(hidden)]
    pub fn set_epoch(&self, epoch: EpochId) {
        self.current_epoch.store(epoch.as_u64(), Ordering::SeqCst);
    }
}
