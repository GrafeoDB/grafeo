use super::LpgStore;
use crate::graph::lpg::{Edge, EdgeRecord};
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use std::sync::atomic::Ordering;

#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;

#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{HotVersionRef, VersionIndex, VersionRef};

impl LpgStore {
    /// Builds an `Edge` from a record, resolving the type name and loading properties.
    fn build_edge(&self, id: EdgeId, record: &EdgeRecord) -> Option<Edge> {
        let edge_type = {
            let id_to_type = self.id_to_edge_type.read();
            id_to_type.get(record.type_id as usize)?.clone()
        };
        let mut edge = Edge::new(id, record.src, record.dst, edge_type);
        edge.properties = self.edge_properties.get_all(id).into_iter().collect();
        Some(edge)
    }

    /// Builds an `Edge` with properties as they were at a specific epoch.
    #[cfg(feature = "temporal")]
    fn build_edge_at(&self, id: EdgeId, record: &EdgeRecord, epoch: EpochId) -> Option<Edge> {
        let edge_type = {
            let id_to_type = self.id_to_edge_type.read();
            id_to_type.get(record.type_id as usize)?.clone()
        };
        let mut edge = Edge::new(id, record.src, record.dst, edge_type);
        edge.properties = self
            .edge_properties
            .get_all_at(id, epoch)
            .into_iter()
            .collect();
        Some(edge)
    }

    /// Creates a new edge.
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        self.create_edge_versioned(
            src,
            dst,
            edge_type,
            self.current_epoch(),
            TransactionId::SYSTEM,
        )
    }

    /// Creates a new edge within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        let id = EdgeId::new(self.next_edge_id.fetch_add(1, Ordering::Relaxed));
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Uncommitted transactional versions use PENDING epoch so they are
        // invisible to other sessions until the transaction commits.
        let version_epoch = if transaction_id == TransactionId::SYSTEM {
            epoch
        } else {
            EpochId::PENDING
        };
        let chain = VersionChain::with_initial(record, version_epoch, transaction_id);
        self.edges.write().insert(id, chain);

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);
        id
    }

    /// Creates a new edge within a transaction context.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        let id = EdgeId::new(self.next_edge_id.fetch_add(1, Ordering::Relaxed));
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self
            .arena_allocator
            .arena_or_create(epoch)
            .expect("failed to create arena for epoch");
        let (offset, _stored) = arena
            .alloc_value_with_offset(record)
            .expect("arena allocation failed for edge record");

        // Uncommitted transactional versions use PENDING epoch so they are
        // invisible to other sessions until the transaction commits.
        let version_epoch = if transaction_id == TransactionId::SYSTEM {
            epoch
        } else {
            EpochId::PENDING
        };

        // Create HotVersionRef pointing to arena data
        let hot_ref = HotVersionRef::new(version_epoch, epoch, offset, transaction_id);

        // Create or update version index
        let mut versions = self.edge_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            index.add_hot(hot_ref);
        } else {
            versions.insert(id, VersionIndex::with_initial(hot_ref));
        }

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);
        id
    }

    /// Creates a new edge with properties.
    pub fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> EdgeId {
        let id = self.create_edge(src, dst, edge_type);

        for (key, value) in properties {
            #[cfg(not(feature = "temporal"))]
            self.edge_properties.set(id, key.into(), value.into());
            #[cfg(feature = "temporal")]
            self.edge_properties
                .set(id, key.into(), value.into(), self.current_epoch());
        }

        id
    }

    /// Gets an edge by ID (latest visible version).
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.get_edge_at_epoch(id, self.current_epoch())
    }

    /// Gets an edge by ID at a specific epoch.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_at(epoch)?;
        if record.is_deleted() {
            return None;
        }
        let record = *record;
        drop(edges);

        #[cfg(not(feature = "temporal"))]
        {
            self.build_edge(id, &record)
        }
        #[cfg(feature = "temporal")]
        {
            if epoch >= self.current_epoch() {
                self.build_edge(id, &record)
            } else {
                self.build_edge_at(id, &record, epoch)
            }
        }
    }

    /// Gets an edge by ID at a specific epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_at(epoch)?;
        let record = self.read_edge_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);

        #[cfg(not(feature = "temporal"))]
        {
            self.build_edge(id, &record)
        }
        #[cfg(feature = "temporal")]
        {
            if epoch >= self.current_epoch() {
                self.build_edge(id, &record)
            } else {
                self.build_edge_at(id, &record, epoch)
            }
        }
    }

    /// Gets an edge visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_to(epoch, transaction_id)?;
        if record.is_deleted() {
            return None;
        }
        let record = *record;
        drop(edges);
        self.build_edge(id, &record)
    }

    /// Gets an edge visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_to(epoch, transaction_id)?;
        let record = self.read_edge_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);
        self.build_edge(id, &record)
    }

    /// Reads an EdgeRecord from arena using a VersionRef.
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub(super) fn read_edge_record(&self, version_ref: &VersionRef) -> Option<EdgeRecord> {
        match version_ref {
            VersionRef::Hot(hot_ref) => {
                let arena = self
                    .arena_allocator
                    .arena(hot_ref.arena_epoch)
                    .expect("arena epoch must exist for hot version ref");
                // SAFETY: The offset was returned by alloc_value_with_offset for an EdgeRecord
                let record: &EdgeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                Some(*record)
            }
            VersionRef::Cold(cold_ref) => {
                // Read from compressed epoch store
                self.epoch_store
                    .get_edge(cold_ref.epoch, cold_ref.block_offset, cold_ref.length)
            }
            _ => None,
        }
    }

    /// Returns all versions of an edge with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Edge)`.
    /// Without `temporal`: properties reflect the current state.
    /// With `temporal`: each version has correct historical properties.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        let edges = self.edges.read();
        let Some(chain) = edges.get(&id) else {
            return Vec::new();
        };

        let id_to_type = self.id_to_edge_type.read();

        #[cfg(not(feature = "temporal"))]
        {
            let properties: grafeo_common::types::PropertyMap =
                self.edge_properties.get_all(id).into_iter().collect();
            chain
                .history()
                .filter_map(|(info, record)| {
                    let edge_type = id_to_type.get(record.type_id as usize)?.clone();
                    let mut edge = Edge::new(id, record.src, record.dst, edge_type);
                    edge.properties.clone_from(&properties);
                    Some((info.created_epoch, info.deleted_epoch, edge))
                })
                .collect()
        }

        #[cfg(feature = "temporal")]
        {
            chain
                .history()
                .filter_map(|(info, record)| {
                    let edge_type = id_to_type.get(record.type_id as usize)?.clone();
                    let mut edge = Edge::new(id, record.src, record.dst, edge_type);
                    edge.properties = self
                        .edge_properties
                        .get_all_at(id, info.created_epoch)
                        .into_iter()
                        .collect();
                    Some((info.created_epoch, info.deleted_epoch, edge))
                })
                .collect()
        }
    }

    /// Returns all versions of an edge with their creation/deletion epochs, newest first.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        let versions = self.edge_versions.read();
        let Some(index) = versions.get(&id) else {
            return Vec::new();
        };

        let id_to_type = self.id_to_edge_type.read();
        let properties: grafeo_common::types::PropertyMap =
            self.edge_properties.get_all(id).into_iter().collect();

        index
            .version_history()
            .into_iter()
            .filter_map(|(created, deleted, vref)| {
                let record = self.read_edge_record(&vref)?;
                let edge_type = id_to_type.get(record.type_id as usize)?.clone();
                let mut edge = Edge::new(id, record.src, record.dst, edge_type);
                edge.properties.clone_from(&properties);
                Some((created, deleted, edge))
            })
            .collect()
    }

    /// Deletes an edge (using latest epoch).
    pub fn delete_edge(&self, id: EdgeId) -> bool {
        self.delete_edge_at_epoch(id, self.current_epoch())
    }

    /// Deletes an edge at a specific epoch.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let mut edges = self.edges.write();
        if let Some(chain) = edges.get_mut(&id) {
            // Get the visible record to check if deleted and get src/dst/type_id
            let (src, dst, type_id) = {
                match chain.visible_at(epoch) {
                    Some(record) => {
                        if record.is_deleted() {
                            return false;
                        }
                        (record.src, record.dst, record.type_id)
                    }
                    None => return false, // Not visible at this epoch (already deleted)
                }
            };

            // Mark the version chain as deleted
            chain.mark_deleted(epoch, TransactionId::SYSTEM);

            drop(edges); // Release lock

            // Mark as deleted in adjacency (soft delete)
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }

            // Remove properties
            #[cfg(not(feature = "temporal"))]
            self.edge_properties.remove_all(id);
            #[cfg(feature = "temporal")]
            self.edge_properties.remove_all(id, self.current_epoch());

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);

            true
        } else {
            false
        }
    }

    /// Deletes an edge at a specific epoch.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let mut versions = self.edge_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            // Get the visible record to check if deleted and get src/dst/type_id
            let (src, dst, type_id) = {
                match index.visible_at(epoch) {
                    Some(version_ref) => {
                        if let Some(record) = self.read_edge_record(&version_ref) {
                            if record.is_deleted() {
                                return false;
                            }
                            (record.src, record.dst, record.type_id)
                        } else {
                            return false;
                        }
                    }
                    None => return false,
                }
            };

            // Mark as deleted in version index
            index.mark_deleted(epoch, TransactionId::SYSTEM);

            drop(versions); // Release lock

            // Mark as deleted in adjacency (soft delete)
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }

            // Remove properties
            #[cfg(not(feature = "temporal"))]
            self.edge_properties.remove_all(id);
            #[cfg(feature = "temporal")]
            self.edge_properties.remove_all(id, self.current_epoch());

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);

            true
        } else {
            false
        }
    }

    /// Deletes an edge within a transaction, capturing undo information for rollback.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_edge_transactional(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let mut edges = self.edges.write();
        if let Some(chain) = edges.get_mut(&id) {
            let (src, dst, type_id) = {
                match chain.visible_at(epoch) {
                    Some(record) => {
                        if record.is_deleted() {
                            return false;
                        }
                        (record.src, record.dst, record.type_id)
                    }
                    None => return false,
                }
            };

            // Uncommitted deletes use PENDING so rollback matching stays
            // epoch-safe after transaction-id reuse.
            chain.mark_deleted(EpochId::PENDING, transaction_id);
            drop(edges);

            // Get edge type name for undo log
            let edge_type_name = {
                let id_to_type = self.id_to_edge_type.read();
                id_to_type
                    .get(type_id as usize)
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            };

            // Capture properties for undo log
            let properties: Vec<(PropertyKey, Value)> =
                self.edge_properties.get_all(id).into_iter().collect();

            // Mark as deleted in adjacency (soft delete)
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }

            // Remove properties
            #[cfg(not(feature = "temporal"))]
            self.edge_properties.remove_all(id);
            #[cfg(feature = "temporal")]
            self.edge_properties.remove_all(id, self.current_epoch());

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);

            // Record undo entry for rollback
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(super::PropertyUndoEntry::EdgeDeleted {
                    edge_id: id,
                    src,
                    dst,
                    edge_type: edge_type_name,
                    properties,
                });

            true
        } else {
            false
        }
    }

    /// Deletes an edge within a transaction, capturing undo information for rollback.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_edge_transactional(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let mut versions = self.edge_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            let (src, dst, type_id) = {
                match index.visible_at(epoch) {
                    Some(version_ref) => {
                        if let Some(record) = self.read_edge_record(&version_ref) {
                            if record.is_deleted() {
                                return false;
                            }
                            (record.src, record.dst, record.type_id)
                        } else {
                            return false;
                        }
                    }
                    None => return false,
                }
            };

            // Uncommitted deletes use PENDING so rollback matching stays
            // epoch-safe after transaction-id reuse.
            index.mark_deleted(EpochId::PENDING, transaction_id);
            drop(versions);

            // Get edge type name for undo log
            let edge_type_name = {
                let id_to_type = self.id_to_edge_type.read();
                id_to_type
                    .get(type_id as usize)
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            };

            // Capture properties for undo log
            let properties: Vec<(PropertyKey, Value)> =
                self.edge_properties.get_all(id).into_iter().collect();

            // Mark as deleted in adjacency
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }

            // Remove properties
            #[cfg(not(feature = "temporal"))]
            self.edge_properties.remove_all(id);
            #[cfg(feature = "temporal")]
            self.edge_properties.remove_all(id, self.current_epoch());

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);

            // Record undo entry for rollback
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(super::PropertyUndoEntry::EdgeDeleted {
                    edge_id: id,
                    src,
                    dst,
                    edge_type: edge_type_name,
                    properties,
                });

            true
        } else {
            false
        }
    }

    /// Returns the number of edges (non-deleted at current epoch).
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edge_count(&self) -> usize {
        let epoch = self.current_epoch();
        self.edges
            .read()
            .values()
            .filter_map(|chain| chain.visible_at(epoch))
            .filter(|r| !r.is_deleted())
            .count()
    }

    /// Returns the number of edges (non-deleted at current epoch).
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn edge_count(&self) -> usize {
        let epoch = self.current_epoch();
        let versions = self.edge_versions.read();
        versions
            .iter()
            .filter(|(_, index)| {
                index.visible_at(epoch).map_or(false, |vref| {
                    self.read_edge_record(&vref)
                        .map_or(false, |r| !r.is_deleted())
                })
            })
            .count()
    }

    /// Creates multiple edges in batch, significantly faster than calling
    /// `create_edge()` in a loop.
    ///
    /// Each tuple is `(src, dst, edge_type)`. Returns the assigned `EdgeId`s
    /// in the same order. Acquires the adjacency write lock once for all
    /// edges, rather than once per edge.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        if edges.is_empty() {
            return Vec::new();
        }

        let epoch = self.current_epoch();
        let base_id = self
            .next_edge_id
            .fetch_add(edges.len() as u64, Ordering::Relaxed);

        let mut ids = Vec::with_capacity(edges.len());
        let mut forward_batch = Vec::with_capacity(edges.len());
        let mut backward_batch = Vec::with_capacity(edges.len());
        let mut type_increments: grafeo_common::utils::hash::FxHashMap<u32, i64> =
            grafeo_common::utils::hash::FxHashMap::default();

        // Create all edge records under a single edges write lock
        {
            let mut edge_map = self.edges.write();
            for (i, &(src, dst, edge_type)) in edges.iter().enumerate() {
                let id = EdgeId::new(base_id + i as u64);
                let type_id = self.get_or_create_edge_type_id(edge_type);

                let record = EdgeRecord::new(id, src, dst, type_id, epoch);
                let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
                edge_map.insert(id, chain);

                forward_batch.push((src, dst, id));
                if self.backward_adj.is_some() {
                    backward_batch.push((dst, src, id));
                }
                *type_increments.entry(type_id).or_default() += 1;

                ids.push(id);
            }
        }

        // Batch adjacency updates (single lock per direction)
        self.forward_adj.batch_add_edges(&forward_batch);
        if let Some(ref backward) = self.backward_adj {
            backward.batch_add_edges(&backward_batch);
        }

        // Update live counters
        // reason: edge batch size fits i64 for practical sizes
        #[allow(clippy::cast_possible_wrap)]
        let edge_count_i64 = edges.len() as i64;
        self.live_edge_count
            .fetch_add(edge_count_i64, Ordering::Relaxed);
        {
            let mut counts = self.edge_type_live_counts.write();
            for (type_id, increment) in type_increments {
                let idx = type_id as usize;
                if counts.len() <= idx {
                    counts.resize(idx + 1, 0);
                }
                counts[idx] += increment;
            }
        }

        ids
    }

    /// Creates multiple edges in batch, significantly faster than calling
    /// `create_edge()` in a loop.
    /// (Tiered storage version)
    ///
    /// # Panics
    ///
    /// Panics if the arena for the current epoch cannot be created.
    #[cfg(feature = "tiered-storage")]
    pub fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        if edges.is_empty() {
            return Vec::new();
        }

        let epoch = self.current_epoch();
        let base_id = self
            .next_edge_id
            .fetch_add(edges.len() as u64, Ordering::Relaxed);
        let arena = self
            .arena_allocator
            .arena_or_create(epoch)
            .expect("failed to create arena for epoch");

        let mut ids = Vec::with_capacity(edges.len());
        let mut forward_batch = Vec::with_capacity(edges.len());
        let mut backward_batch = Vec::with_capacity(edges.len());
        let mut type_increments: grafeo_common::utils::hash::FxHashMap<u32, i64> =
            grafeo_common::utils::hash::FxHashMap::default();

        // Create all edge records under a single versions write lock
        {
            let mut versions = self.edge_versions.write();
            for (i, &(src, dst, edge_type)) in edges.iter().enumerate() {
                let id = EdgeId::new(base_id + i as u64);
                let type_id = self.get_or_create_edge_type_id(edge_type);

                let record = EdgeRecord::new(id, src, dst, type_id, epoch);
                let (offset, _stored) = arena
                    .alloc_value_with_offset(record)
                    .expect("arena allocation failed for edge record");
                let hot_ref = HotVersionRef::new(epoch, epoch, offset, TransactionId::SYSTEM);
                versions.insert(id, VersionIndex::with_initial(hot_ref));

                forward_batch.push((src, dst, id));
                if self.backward_adj.is_some() {
                    backward_batch.push((dst, src, id));
                }
                *type_increments.entry(type_id).or_default() += 1;

                ids.push(id);
            }
        }

        // Batch adjacency updates (single lock per direction)
        self.forward_adj.batch_add_edges(&forward_batch);
        if let Some(ref backward) = self.backward_adj {
            backward.batch_add_edges(&backward_batch);
        }

        // Update live counters
        // reason: edge batch size fits i64 for practical sizes
        #[allow(clippy::cast_possible_wrap)]
        let edge_count_i64 = edges.len() as i64;
        self.live_edge_count
            .fetch_add(edge_count_i64, Ordering::Relaxed);
        {
            let mut counts = self.edge_type_live_counts.write();
            for (type_id, increment) in type_increments {
                let idx = type_id as usize;
                if counts.len() <= idx {
                    counts.resize(idx + 1, 0);
                }
                counts[idx] += increment;
            }
        }

        ids
    }

    /// Gets the type of an edge by ID.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let epoch = self.current_epoch();
        let record = chain.visible_at(epoch)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Gets the type of an edge by ID.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let epoch = self.current_epoch();
        let vref = index.visible_at(epoch)?;
        let record = self.read_edge_record(&vref)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Gets the type of an edge visible to a specific transaction.
    ///
    /// Used by operators that need edge type info for PENDING (uncommitted) edges.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_to(epoch, transaction_id)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Gets the type of an edge visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let vref = index.visible_to(epoch, transaction_id)?;
        let record = self.read_edge_record(&vref)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    // --- Visibility checks (no type resolution or property loading) ---

    /// Checks if an edge is visible at the given epoch.
    ///
    /// Only checks the version chain, skips type resolution and property loading.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let edges = self.edges.read();
        edges
            .get(&id)
            .is_some_and(|chain| chain.visible_at(epoch).is_some_and(|r| !r.is_deleted()))
    }

    /// Checks if an edge is visible at the given epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let versions = self.edge_versions.read();
        versions.get(&id).is_some_and(|index| {
            index.visible_at(epoch).is_some_and(|vref| {
                self.read_edge_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        })
    }

    /// Checks if an edge is visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let edges = self.edges.read();
        edges.get(&id).is_some_and(|chain| {
            chain
                .visible_to(epoch, transaction_id)
                .is_some_and(|r| !r.is_deleted())
        })
    }

    /// Checks if an edge is visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let versions = self.edge_versions.read();
        versions.get(&id).is_some_and(|index| {
            index.visible_to(epoch, transaction_id).is_some_and(|vref| {
                self.read_edge_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        })
    }
}
