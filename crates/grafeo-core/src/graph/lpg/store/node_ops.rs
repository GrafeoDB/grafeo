use super::LpgStore;
use crate::graph::lpg::{Node, NodeRecord};
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use std::sync::atomic::Ordering;

#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;

#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{HotVersionRef, VersionIndex, VersionRef};

/// The error of a node whose record cannot be read (a cold one that does not
/// decode).
#[cfg(feature = "tiered-storage")]
fn unreadable_node_record(id: NodeId) -> grafeo_common::utils::error::Error {
    grafeo_common::utils::error::Error::Internal(format!(
        "the record of node {} cannot be read",
        id.as_u64()
    ))
}

impl LpgStore {
    /// Creates a new node with the given labels.
    ///
    /// Uses the system transaction for non-transactional operations.
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        self.create_node_versioned(labels, self.current_epoch(), TransactionId::SYSTEM)
    }

    /// Registers labels for a node: builds the label ID set, updates the
    /// label index (single lock acquisition), and stores the node-to-labels
    /// mapping.
    #[cfg(not(feature = "temporal"))]
    pub(super) fn register_node_labels(&self, id: NodeId, labels: &[&str]) {
        let mut node_label_set = FxHashSet::default();
        let mut label_ids = Vec::with_capacity(labels.len());
        for label in labels {
            let label_id = self.get_or_create_label_id(label);
            node_label_set.insert(label_id);
            label_ids.push(label_id);
        }

        // Update label index with a single lock acquisition
        let mut index = self.label_index.write();
        for label_id in label_ids {
            if index.len() <= label_id as usize {
                index.resize_with(label_id as usize + 1, FxHashMap::default);
            }
            index[label_id as usize].insert(id, ());
        }
        drop(index);

        self.node_labels.write().insert(id, node_label_set);
    }

    #[cfg(feature = "temporal")]
    pub(super) fn register_node_labels(&self, id: NodeId, labels: &[&str], epoch: EpochId) {
        use grafeo_common::temporal::VersionLog;

        let mut node_label_set = FxHashSet::default();
        let mut label_ids = Vec::with_capacity(labels.len());
        for label in labels {
            let label_id = self.get_or_create_label_id(label);
            node_label_set.insert(label_id);
            label_ids.push(label_id);
        }

        // Update label index with a single lock acquisition
        let mut index = self.label_index.write();
        for label_id in label_ids {
            if index.len() <= label_id as usize {
                index.resize_with(label_id as usize + 1, FxHashMap::default);
            }
            index[label_id as usize].insert(id, ());
        }
        drop(index);

        self.node_labels
            .write()
            .insert(id, VersionLog::with_value(epoch, node_label_set));
    }

    /// Builds a `Node` populated with labels and properties for the given ID.
    ///
    /// Returns the current (latest) state of the node.
    fn build_node(&self, id: NodeId) -> Node {
        let mut node = self.node_with_labels(id);
        node.properties = self.node_properties.get_all(id).into_iter().collect();
        node
    }

    /// [`build_node`](Self::build_node) reading the properties fallibly: a
    /// spilled value that cannot be read is an error, never left out.
    pub(super) fn try_build_node(&self, id: NodeId) -> grafeo_common::utils::error::Result<Node> {
        let mut node = self.node_with_labels(id);
        node.properties = self.node_properties.try_get_all(id)?.into_iter().collect();
        Ok(node)
    }

    /// A `Node` with the labels `id` has at `epoch` and no properties: with
    /// `temporal`, the label set visible at that epoch, so a set a
    /// transaction still open wrote (at `EpochId::PENDING`) is left out;
    /// without it, the current set (the store keeps no label versions).
    #[cfg(feature = "temporal")]
    pub(crate) fn node_with_labels_at(&self, id: NodeId, epoch: EpochId) -> Node {
        let mut node = Node::new(id);
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        if let Some(log) = node_labels.get(&id)
            && let Some(label_ids) = log.at(epoch)
        {
            for &label_id in label_ids {
                if let Some(label) = registry.get_name(label_id) {
                    node.labels.push(label.clone());
                }
            }
        }
        node
    }

    /// A `Node` with the labels `id` has at `epoch` and no properties: with
    /// `temporal`, the label set visible at that epoch; without it, the
    /// current set (the store keeps no label versions).
    #[cfg(not(feature = "temporal"))]
    pub(crate) fn node_with_labels_at(&self, id: NodeId, _epoch: EpochId) -> Node {
        self.node_with_labels(id)
    }

    /// A `Node` with the current labels of `id` and no properties yet.
    pub(super) fn node_with_labels(&self, id: NodeId) -> Node {
        let mut node = Node::new(id);

        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();

        #[cfg(not(feature = "temporal"))]
        if let Some(label_ids) = node_labels.get(&id) {
            for &label_id in label_ids {
                if let Some(label) = registry.get_name(label_id) {
                    node.labels.push(label.clone());
                }
            }
        }

        #[cfg(feature = "temporal")]
        if let Some(log) = node_labels.get(&id)
            && let Some(label_ids) = log.latest()
        {
            for &label_id in label_ids {
                if let Some(label) = registry.get_name(label_id) {
                    node.labels.push(label.clone());
                }
            }
        }

        node
    }

    /// Builds a `Node` with labels and properties as they were at a specific epoch.
    ///
    /// This is the critical method that makes `get_node_at_epoch()` return
    /// correct historical property values instead of current ones.
    #[cfg(feature = "temporal")]
    fn build_node_at(&self, id: NodeId, epoch: EpochId) -> Node {
        // The labels as every read at `epoch` resolves them (a checkpoint's
        // too), so a historical read and a chunked section cannot disagree.
        let mut node = self.node_with_labels_at(id, epoch);
        node.properties = self
            .node_properties
            .get_all_at(id, epoch)
            .into_iter()
            .collect();
        node
    }

    /// Creates a new node with the given labels within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = NodeId::new(self.next_node_id.fetch_add(1, Ordering::Relaxed));

        let mut record = NodeRecord::new(id, epoch);
        // reason: label count per node is bounded by practical limits, fits u16
        #[allow(clippy::cast_possible_truncation)]
        record.set_label_count(labels.len() as u16);

        // Uncommitted transactional versions use PENDING epoch so they are
        // invisible to other sessions until the transaction commits.
        let version_epoch = if transaction_id == TransactionId::SYSTEM {
            epoch
        } else {
            EpochId::PENDING
        };

        #[cfg(not(feature = "temporal"))]
        self.register_node_labels(id, labels);
        #[cfg(feature = "temporal")]
        self.register_node_labels(id, labels, version_epoch);

        let chain = VersionChain::with_initial(record, version_epoch, transaction_id);
        self.nodes.write().insert(id, chain);
        self.live_node_count.fetch_add(1, Ordering::Relaxed);
        self.record_change(
            transaction_id,
            super::PropertyUndoEntry::NodeCreated { node_id: id },
        );
        id
    }

    /// Creates a new node with the given labels within a transaction context.
    /// (Tiered storage version: stores data in arena, metadata in VersionIndex)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = NodeId::new(self.next_node_id.fetch_add(1, Ordering::Relaxed));

        let mut record = NodeRecord::new(id, epoch);
        // reason: label count per node is bounded by practical limits, fits u16
        #[allow(clippy::cast_possible_truncation)]
        record.set_label_count(labels.len() as u16);

        // Uncommitted transactional versions use PENDING epoch so they are
        // invisible to other sessions until the transaction commits.
        let version_epoch = if transaction_id == TransactionId::SYSTEM {
            epoch
        } else {
            EpochId::PENDING
        };

        #[cfg(not(feature = "temporal"))]
        self.register_node_labels(id, labels);
        #[cfg(feature = "temporal")]
        self.register_node_labels(id, labels, version_epoch);

        // Allocate the record in the epoch's arena (created if needed). The
        // arena's lock is released at the end of this statement, before the
        // version lock below: see the lock order on `arena_allocator`.
        let (offset, _stored) = self
            .arena_allocator
            .arena_or_create(epoch)
            .expect("failed to create arena for epoch")
            .alloc_value_with_offset(record)
            .expect("arena allocation failed for node record");

        // Create HotVersionRef pointing to arena data
        let hot_ref = HotVersionRef::new(version_epoch, epoch, offset, transaction_id);

        // Create or update version index
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            index.add_hot(hot_ref);
            self.gc_candidates.lock().nodes.insert(id);
        } else {
            versions.insert(id, VersionIndex::with_initial(hot_ref));
        }
        drop(versions);

        self.live_node_count.fetch_add(1, Ordering::Relaxed);
        self.record_change(
            transaction_id,
            super::PropertyUndoEntry::NodeCreated { node_id: id },
        );
        id
    }

    /// Creates a new node with labels and properties.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> NodeId {
        self.create_node_with_props_versioned(
            labels,
            properties,
            self.current_epoch(),
            TransactionId::SYSTEM,
        )
    }

    /// Creates a new node with labels and properties within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn create_node_with_props_versioned(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self.create_node_versioned(labels, epoch, transaction_id);

        for (key, value) in properties {
            let prop_key: PropertyKey = key.into();
            let prop_value: Value = value.into();
            // Update property index before setting the property
            self.update_property_index_on_set(id, &prop_key, &prop_value);
            #[cfg(not(feature = "temporal"))]
            self.node_properties.set(id, prop_key, prop_value);
            #[cfg(feature = "temporal")]
            self.node_properties.set(id, prop_key, prop_value, epoch);
        }

        id
    }

    /// Creates a new node with labels and properties within a transaction context.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn create_node_with_props_versioned(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self.create_node_versioned(labels, epoch, transaction_id);

        for (key, value) in properties {
            let prop_key: PropertyKey = key.into();
            let prop_value: Value = value.into();
            // Update property index before setting the property
            self.update_property_index_on_set(id, &prop_key, &prop_value);
            #[cfg(not(feature = "temporal"))]
            self.node_properties.set(id, prop_key, prop_value);
            #[cfg(feature = "temporal")]
            self.node_properties.set(id, prop_key, prop_value, epoch);
        }

        id
    }

    /// Gets a node by ID (latest visible version).
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.get_node_at_epoch(id, self.current_epoch())
    }

    /// Gets a node by ID at a specific epoch.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let nodes = self.nodes.read();
        let chain = nodes.get(&id)?;
        let record = chain.visible_at(epoch)?;
        if record.is_deleted() {
            return None;
        }
        drop(nodes);

        #[cfg(not(feature = "temporal"))]
        {
            Some(self.build_node(id))
        }
        #[cfg(feature = "temporal")]
        {
            // Fast path: current-epoch reads use latest() instead of at(epoch).
            // Safe because non-transactional reads have no PENDING entries.
            if epoch >= self.current_epoch() {
                Some(self.build_node(id))
            } else {
                Some(self.build_node_at(id, epoch))
            }
        }
    }

    /// Gets a node by ID at a specific epoch.
    /// (Tiered storage version: reads from arena via VersionIndex)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let versions = self.node_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_at(epoch)?;
        let record = self.read_node_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);

        #[cfg(not(feature = "temporal"))]
        {
            Some(self.build_node(id))
        }
        #[cfg(feature = "temporal")]
        {
            if epoch >= self.current_epoch() {
                Some(self.build_node(id))
            } else {
                Some(self.build_node_at(id, epoch))
            }
        }
    }

    /// Gets a node visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        let nodes = self.nodes.read();
        let chain = nodes.get(&id)?;
        let record = chain.visible_to(epoch, transaction_id)?;
        if record.is_deleted() {
            return None;
        }
        drop(nodes);
        Some(self.build_node(id))
    }

    /// Gets a node visible to a specific transaction.
    /// (Tiered storage version: reads from arena via VersionIndex)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        let versions = self.node_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_to(epoch, transaction_id)?;
        let record = self.read_node_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);
        Some(self.build_node(id))
    }

    /// Returns all versions of a node with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Node)`.
    /// Without `temporal`: labels and properties reflect the current state.
    /// With `temporal`: each version has correct historical properties and labels.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let nodes = self.nodes.read();
        let Some(chain) = nodes.get(&id) else {
            return Vec::new();
        };

        #[cfg(not(feature = "temporal"))]
        {
            let template = self.build_node(id);
            chain
                .history()
                .map(|(info, _record)| (info.created_epoch, info.deleted_epoch, template.clone()))
                .collect()
        }

        #[cfg(feature = "temporal")]
        {
            chain
                .history()
                .map(|(info, _record)| {
                    let node = self.build_node_at(id, info.created_epoch);
                    (info.created_epoch, info.deleted_epoch, node)
                })
                .collect()
        }
    }

    /// Returns all versions of a node with their creation/deletion epochs, newest first.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let versions = self.node_versions.read();
        let Some(index) = versions.get(&id) else {
            return Vec::new();
        };

        #[cfg(not(feature = "temporal"))]
        {
            let template = self.build_node(id);
            index
                .version_history()
                .into_iter()
                .map(|(created, deleted, _vref)| (created, deleted, template.clone()))
                .collect()
        }

        #[cfg(feature = "temporal")]
        {
            index
                .version_history()
                .into_iter()
                .map(|(created, deleted, _vref)| {
                    let node = self.build_node_at(id, created);
                    (created, deleted, node)
                })
                .collect()
        }
    }

    /// Reads a NodeRecord from arena (hot) or epoch store (cold) using a VersionRef.
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub(super) fn read_node_record(&self, version_ref: &VersionRef) -> Option<NodeRecord> {
        match version_ref {
            VersionRef::Hot(hot_ref) => {
                let arena = self
                    .arena_allocator
                    .arena(hot_ref.arena_epoch)
                    .expect("arena epoch must exist for hot version ref");
                // SAFETY: The offset was returned by alloc_value_with_offset for a NodeRecord
                let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                Some(*record)
            }
            VersionRef::Cold(cold_ref) => {
                // Read from compressed epoch store
                self.epoch_store
                    .get_node(cold_ref.epoch, cold_ref.block_offset, cold_ref.length)
            }
            _ => None,
        }
    }

    /// Deletes a node (using latest epoch). Its edges are not deleted; call
    /// `delete_node_edges` first for a detach delete.
    pub fn delete_node(&self, id: NodeId) -> bool {
        self.delete_node_at_epoch(id, self.current_epoch())
    }

    /// Deletes a node at a specific epoch.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let mut nodes = self.nodes.write();
        if let Some(chain) = nodes.get_mut(&id) {
            // Check if visible at this epoch (not already deleted)
            if let Some(record) = chain.visible_at(epoch) {
                if record.is_deleted() {
                    return false;
                }
            } else {
                // Not visible at this epoch (already deleted or doesn't exist)
                return false;
            }

            // Mark the version chain as deleted at this epoch
            chain.mark_deleted(epoch, TransactionId::SYSTEM);

            // Remove from label index using node_labels map
            let mut index = self.label_index.write();
            let mut node_labels = self.node_labels.write();
            if let Some(removed) = node_labels.remove(&id) {
                #[cfg(not(feature = "temporal"))]
                let label_ids = removed;
                #[cfg(feature = "temporal")]
                let label_ids = removed.latest().cloned().unwrap_or_default();
                for label_id in label_ids {
                    if let Some(set) = index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }

            // Remove from text indexes before removing properties
            #[cfg(feature = "text-index")]
            self.remove_from_all_text_indexes(id);

            // Remove properties
            drop(nodes); // Release lock before removing properties
            drop(index);
            drop(node_labels);
            // Drop the node from property indexes while its values are known
            self.remove_from_all_property_indexes(id);
            #[cfg(feature = "vector-index")]
            self.remove_from_all_vector_indexes(id);
            #[cfg(not(feature = "temporal"))]
            self.node_properties.remove_all(id);
            #[cfg(feature = "temporal")]
            self.node_properties.remove_all(id, self.current_epoch());

            self.live_node_count.fetch_sub(1, Ordering::Relaxed);

            true
        } else {
            false
        }
    }

    /// Deletes a node at a specific epoch.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            // Check if visible at this epoch
            if let Some(version_ref) = index.visible_at(epoch) {
                if let Some(record) = self.read_node_record(&version_ref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }

            // Mark as deleted in version index
            index.mark_deleted(epoch, TransactionId::SYSTEM);

            // Remove from label index using node_labels map
            let mut label_index = self.label_index.write();
            let mut node_labels = self.node_labels.write();
            if let Some(removed) = node_labels.remove(&id) {
                #[cfg(not(feature = "temporal"))]
                let label_ids = removed;
                #[cfg(feature = "temporal")]
                let label_ids = removed.latest().cloned().unwrap_or_default();
                for label_id in label_ids {
                    if let Some(set) = label_index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }

            // Remove from text indexes before removing properties
            #[cfg(feature = "text-index")]
            self.remove_from_all_text_indexes(id);

            // Remove properties
            drop(versions);
            drop(label_index);
            drop(node_labels);
            // Drop the node from property indexes while its values are known
            self.remove_from_all_property_indexes(id);
            #[cfg(feature = "vector-index")]
            self.remove_from_all_vector_indexes(id);
            #[cfg(not(feature = "temporal"))]
            self.node_properties.remove_all(id);
            #[cfg(feature = "temporal")]
            self.node_properties.remove_all(id, self.current_epoch());

            self.live_node_count.fetch_sub(1, Ordering::Relaxed);

            true
        } else {
            false
        }
    }

    /// Deletes a node within a transaction, capturing undo information for rollback.
    ///
    /// Unlike `delete_node_at_epoch`, this method:
    /// 1. Captures labels and properties before deletion (for undo log)
    /// 2. Marks the version with `deleted_by` = transaction_id (for rollback)
    /// 3. Pushes a `NodeDeleted` undo entry so rollback can restore the node
    ///
    /// # Errors
    ///
    /// Returns an error, and deletes nothing, when a property of the node
    /// cannot be read (a spilled value whose file cannot be read).
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_node_transactional(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> grafeo_common::utils::error::Result<bool> {
        // A node the transaction cannot see is not deleted, and none of its
        // values are read: one that cannot be read must not fail the call.
        let visible = self
            .nodes
            .read()
            .get(&id)
            .and_then(|chain| chain.visible_to(epoch, transaction_id))
            .is_some_and(|record| !record.is_deleted());
        if !visible {
            return Ok(false);
        }
        // Capture the properties for the undo log before anything changes: a
        // value that cannot be read (a spilled value whose file cannot be read)
        // refuses the delete, as a rollback could not restore it. The lock is
        // not held for the read; the check below runs again under it.
        let properties: Vec<(PropertyKey, Value)> =
            self.node_properties.try_get_all(id)?.into_iter().collect();
        let mut nodes = self.nodes.write();
        if let Some(chain) = nodes.get_mut(&id) {
            if let Some(record) = chain.visible_to(epoch, transaction_id) {
                if record.is_deleted() {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }

            // Mark deleted with transaction tracking
            chain.mark_deleted(epoch, transaction_id);

            // Capture labels for undo log
            let registry = self.label_registry.read();
            let node_labels_map = self.node_labels.read();

            #[cfg(not(feature = "temporal"))]
            let label_names: Vec<String> = node_labels_map
                .get(&id)
                .map(|label_ids| {
                    label_ids
                        .iter()
                        .filter_map(|&lid| registry.get_name(lid).map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            #[cfg(feature = "temporal")]
            let label_names: Vec<String> = node_labels_map
                .get(&id)
                .and_then(|log| log.latest())
                .map(|label_ids| {
                    label_ids
                        .iter()
                        .filter_map(|&lid| registry.get_name(lid).map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            drop(registry);
            drop(node_labels_map);

            drop(nodes);

            // Remove from label index (will be restored on rollback)
            let mut index = self.label_index.write();
            let mut node_labels_w = self.node_labels.write();
            if let Some(removed) = node_labels_w.remove(&id) {
                #[cfg(not(feature = "temporal"))]
                let label_ids = removed;
                #[cfg(feature = "temporal")]
                let label_ids = removed.latest().cloned().unwrap_or_default();
                for label_id in label_ids {
                    if let Some(set) = index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }
            drop(index);
            drop(node_labels_w);

            // Remove from text indexes
            #[cfg(feature = "text-index")]
            self.remove_from_all_text_indexes(id);

            // Remove properties (will be restored on rollback)
            // Drop the node from property indexes while its values are known
            self.remove_from_all_property_indexes(id);
            #[cfg(feature = "vector-index")]
            self.remove_from_all_vector_indexes(id);
            #[cfg(not(feature = "temporal"))]
            self.node_properties.remove_all(id);
            #[cfg(feature = "temporal")]
            // Tombstones are this transaction's writes: PENDING until it commits.
            self.node_properties
                .remove_all(id, grafeo_common::types::EpochId::PENDING);
            self.live_node_count.fetch_sub(1, Ordering::Relaxed);

            // Record undo entry for rollback
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(super::PropertyUndoEntry::NodeDeleted {
                    node_id: id,
                    labels: label_names,
                    properties,
                });

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Deletes a node within a transaction, capturing undo information for rollback.
    /// (Tiered storage version)
    ///
    /// # Errors
    ///
    /// Returns an error, and deletes nothing, when a property of the node
    /// cannot be read (a spilled value whose file cannot be read), or the
    /// version the transaction sees is a record that cannot be read (a cold
    /// one that does not decode): the node is there, so `false` would be a
    /// wrong answer.
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_node_transactional(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> grafeo_common::utils::error::Result<bool> {
        // A node the transaction cannot see is not deleted, and none of its
        // values are read: one that cannot be read must not fail the call.
        let visible = {
            // The record is read under the versions lock, so a freeze of its
            // epoch cannot move it meanwhile.
            let versions = self.node_versions.read();
            match versions
                .get(&id)
                .and_then(|index| index.visible_to(epoch, transaction_id))
            {
                Some(version_ref) => !self
                    .read_node_record(&version_ref)
                    .ok_or_else(|| unreadable_node_record(id))?
                    .is_deleted(),
                None => false,
            }
        };
        if !visible {
            return Ok(false);
        }
        // Capture the properties for the undo log before anything changes: a
        // value that cannot be read (a spilled value whose file cannot be read)
        // refuses the delete, as a rollback could not restore it. The lock is
        // not held for the read; the check below runs again under it.
        let properties: Vec<(PropertyKey, Value)> =
            self.node_properties.try_get_all(id)?.into_iter().collect();
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            if let Some(version_ref) = index.visible_to(epoch, transaction_id) {
                let record = self
                    .read_node_record(&version_ref)
                    .ok_or_else(|| unreadable_node_record(id))?;
                if record.is_deleted() {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }

            // Mark deleted with transaction tracking
            index.mark_deleted(epoch, transaction_id);

            // Capture labels for undo log
            let registry = self.label_registry.read();
            let node_labels_map = self.node_labels.read();

            #[cfg(not(feature = "temporal"))]
            let label_names: Vec<String> = node_labels_map
                .get(&id)
                .map(|label_ids| {
                    label_ids
                        .iter()
                        .filter_map(|&lid| registry.get_name(lid).map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            #[cfg(feature = "temporal")]
            let label_names: Vec<String> = node_labels_map
                .get(&id)
                .and_then(|log| log.latest())
                .map(|label_ids| {
                    label_ids
                        .iter()
                        .filter_map(|&lid| registry.get_name(lid).map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            drop(registry);
            drop(node_labels_map);

            drop(versions);

            // Remove from label index
            let mut label_index = self.label_index.write();
            let mut node_labels_w = self.node_labels.write();
            if let Some(removed) = node_labels_w.remove(&id) {
                #[cfg(not(feature = "temporal"))]
                let label_ids = removed;
                #[cfg(feature = "temporal")]
                let label_ids = removed.latest().cloned().unwrap_or_default();
                for label_id in label_ids {
                    if let Some(set) = label_index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }
            drop(label_index);
            drop(node_labels_w);

            // Remove from text indexes
            #[cfg(feature = "text-index")]
            self.remove_from_all_text_indexes(id);

            // Remove properties
            // Drop the node from property indexes while its values are known
            self.remove_from_all_property_indexes(id);
            #[cfg(feature = "vector-index")]
            self.remove_from_all_vector_indexes(id);
            #[cfg(not(feature = "temporal"))]
            self.node_properties.remove_all(id);
            #[cfg(feature = "temporal")]
            // Tombstones are this transaction's writes: PENDING until it commits.
            self.node_properties
                .remove_all(id, grafeo_common::types::EpochId::PENDING);
            self.live_node_count.fetch_sub(1, Ordering::Relaxed);

            // Record undo entry for rollback
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(super::PropertyUndoEntry::NodeDeleted {
                    node_id: id,
                    labels: label_names,
                    properties,
                });

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Deletes all edges connected to a node (implements DETACH DELETE).
    ///
    /// Call this before `delete_node()` if you want to remove a node that
    /// has edges. Grafeo doesn't auto-delete edges, you have to be explicit.
    ///
    /// Uses a `HashSet` to dedup self-loops (where src == dst) and holds
    /// the edges write lock for the entire batch to prevent concurrent
    /// readers from observing a partially detached node.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn delete_node_edges(&self, node_id: NodeId) {
        let epoch = self.current_epoch();

        // Collect all edge IDs into a set to dedup self-loops
        let mut edge_ids: FxHashSet<EdgeId> = FxHashSet::default();

        for (_, edge_id) in self.forward_adj.edges_from(node_id) {
            edge_ids.insert(edge_id);
        }

        if let Some(ref backward) = self.backward_adj {
            for (_, edge_id) in backward.edges_from(node_id) {
                edge_ids.insert(edge_id);
            }
        } else {
            // No backward adjacency: scan all edges for incoming
            let edges = self.edges.read();
            for (id, chain) in edges.iter() {
                if let Some(r) = chain.visible_at(epoch)
                    && !r.is_deleted()
                    && r.dst == node_id
                {
                    edge_ids.insert(*id);
                }
            }
        }

        if edge_ids.is_empty() {
            return;
        }

        // Hold the write lock for the entire batch: mark all edges as deleted
        // atomically so concurrent readers never see a partially detached node.
        let deleted: Vec<(EdgeId, NodeId, NodeId, u32)>;
        {
            let mut edges = self.edges.write();
            deleted = edge_ids
                .iter()
                .filter_map(|&edge_id| {
                    let chain = edges.get_mut(&edge_id)?;
                    let record = chain.visible_at(epoch)?;
                    if record.is_deleted() {
                        return None;
                    }
                    let src = record.src;
                    let dst = record.dst;
                    let type_id = record.type_id;
                    chain.mark_deleted(epoch, TransactionId::SYSTEM);
                    Some((edge_id, src, dst, type_id))
                })
                .collect();
        }

        // Batch-update adjacency under a single write lock so concurrent
        // readers of neighbors()/edges_from() never see a partially detached node.
        let fwd_batch: Vec<(NodeId, EdgeId)> =
            deleted.iter().map(|&(eid, src, _, _)| (src, eid)).collect();
        self.forward_adj.batch_mark_deleted(&fwd_batch);

        if let Some(ref backward) = self.backward_adj {
            let bwd_batch: Vec<(NodeId, EdgeId)> =
                deleted.iter().map(|&(eid, _, dst, _)| (dst, eid)).collect();
            backward.batch_mark_deleted(&bwd_batch);
        }

        // Properties and counters (no atomicity requirement with adjacency)
        for &(edge_id, _, _, type_id) in &deleted {
            #[cfg(not(feature = "temporal"))]
            self.edge_properties.remove_all(edge_id);
            #[cfg(feature = "temporal")]
            self.edge_properties.remove_all(edge_id, epoch);

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);
        }
    }

    /// Deletes all edges connected to a node (implements DETACH DELETE).
    /// (Tiered storage version)
    ///
    /// Uses a `HashSet` to dedup self-loops (where src == dst) and holds
    /// the edge_versions write lock for the entire batch to prevent
    /// concurrent readers from observing a partially detached node.
    #[cfg(feature = "tiered-storage")]
    pub fn delete_node_edges(&self, node_id: NodeId) {
        let epoch = self.current_epoch();

        // Collect all edge IDs into a set to dedup self-loops
        let mut edge_ids: FxHashSet<EdgeId> = FxHashSet::default();

        for (_, edge_id) in self.forward_adj.edges_from(node_id) {
            edge_ids.insert(edge_id);
        }

        if let Some(ref backward) = self.backward_adj {
            for (_, edge_id) in backward.edges_from(node_id) {
                edge_ids.insert(edge_id);
            }
        } else {
            // No backward adjacency: scan all edges for incoming
            let versions = self.edge_versions.read();
            for (id, index) in versions.iter() {
                if let Some(vref) = index.visible_at(epoch)
                    && let Some(r) = self.read_edge_record(&vref)
                    && !r.is_deleted()
                    && r.dst == node_id
                {
                    edge_ids.insert(*id);
                }
            }
        }

        if edge_ids.is_empty() {
            return;
        }

        // Hold the write lock for the entire batch: mark all edges as deleted
        // atomically so concurrent readers never see a partially detached node.
        let deleted: Vec<(EdgeId, NodeId, NodeId, u32)>;
        {
            let mut versions = self.edge_versions.write();
            deleted = edge_ids
                .iter()
                .filter_map(|&edge_id| {
                    let index = versions.get_mut(&edge_id)?;
                    let vref = index.visible_at(epoch)?;
                    let record = self.read_edge_record(&vref)?;
                    if record.is_deleted() {
                        return None;
                    }
                    let src = record.src;
                    let dst = record.dst;
                    let type_id = record.type_id;
                    index.mark_deleted(epoch, TransactionId::SYSTEM);
                    Some((edge_id, src, dst, type_id))
                })
                .collect();
        }

        // Batch-update adjacency under a single write lock so concurrent
        // readers of neighbors()/edges_from() never see a partially detached node.
        let fwd_batch: Vec<(NodeId, EdgeId)> =
            deleted.iter().map(|&(eid, src, _, _)| (src, eid)).collect();
        self.forward_adj.batch_mark_deleted(&fwd_batch);

        if let Some(ref backward) = self.backward_adj {
            let bwd_batch: Vec<(NodeId, EdgeId)> =
                deleted.iter().map(|&(eid, _, dst, _)| (dst, eid)).collect();
            backward.batch_mark_deleted(&bwd_batch);
        }

        // Properties and counters (no atomicity requirement with adjacency)
        for &(edge_id, _, _, type_id) in &deleted {
            #[cfg(not(feature = "temporal"))]
            self.edge_properties.remove_all(edge_id);
            #[cfg(feature = "temporal")]
            self.edge_properties.remove_all(edge_id, epoch);

            self.live_edge_count.fetch_sub(1, Ordering::Relaxed);
            self.decrement_edge_type_count(type_id);
        }
    }

    // --- Visibility checks (no label/property loading) ---

    /// Checks if a node is visible at the given epoch.
    ///
    /// Only checks the version chain, skips label and property loading.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let nodes = self.nodes.read();
        nodes
            .get(&id)
            .is_some_and(|chain| chain.visible_at(epoch).is_some_and(|r| !r.is_deleted()))
    }

    /// Checks if a node is visible at the given epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let versions = self.node_versions.read();
        versions.get(&id).is_some_and(|index| {
            index.visible_at(epoch).is_some_and(|vref| {
                self.read_node_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        })
    }

    /// Checks if a node is visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let nodes = self.nodes.read();
        nodes.get(&id).is_some_and(|chain| {
            chain
                .visible_to(epoch, transaction_id)
                .is_some_and(|r| !r.is_deleted())
        })
    }

    /// Checks if a node is visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let versions = self.node_versions.read();
        versions.get(&id).is_some_and(|index| {
            index.visible_to(epoch, transaction_id).is_some_and(|vref| {
                self.read_node_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        })
    }

    /// Filters node IDs to only those visible at the given epoch.
    ///
    /// Holds a single lock for the entire batch instead of per-node locking.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        ids.iter()
            .copied()
            .filter(|id| {
                nodes
                    .get(id)
                    .is_some_and(|chain| chain.visible_at(epoch).is_some_and(|r| !r.is_deleted()))
            })
            .collect()
    }

    /// Filters node IDs to only those visible at the given epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        let versions = self.node_versions.read();
        ids.iter()
            .copied()
            .filter(|id| {
                versions.get(id).is_some_and(|index| {
                    index.visible_at(epoch).is_some_and(|vref| {
                        self.read_node_record(&vref)
                            .is_some_and(|r| !r.is_deleted())
                    })
                })
            })
            .collect()
    }

    /// Filters node IDs to only those visible to a specific transaction.
    ///
    /// Holds a single lock for the entire batch instead of per-node locking.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        ids.iter()
            .copied()
            .filter(|id| {
                nodes.get(id).is_some_and(|chain| {
                    chain
                        .visible_to(epoch, transaction_id)
                        .is_some_and(|r| !r.is_deleted())
                })
            })
            .collect()
    }

    /// Filters node IDs to only those visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let versions = self.node_versions.read();
        ids.iter()
            .copied()
            .filter(|id| {
                versions.get(id).is_some_and(|index| {
                    index.visible_to(epoch, transaction_id).is_some_and(|vref| {
                        self.read_node_record(&vref)
                            .is_some_and(|r| !r.is_deleted())
                    })
                })
            })
            .collect()
    }

    /// Returns the number of nodes (non-deleted at current epoch).
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn node_count(&self) -> usize {
        let epoch = self.current_epoch();
        self.nodes
            .read()
            .values()
            .filter_map(|chain| chain.visible_at(epoch))
            .filter(|r| !r.is_deleted())
            .count()
    }

    /// Returns the number of nodes (non-deleted at current epoch).
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn node_count(&self) -> usize {
        let epoch = self.current_epoch();
        let versions = self.node_versions.read();
        versions
            .iter()
            .filter(|(_, index)| {
                index.visible_at(epoch).map_or(false, |vref| {
                    self.read_node_record(&vref)
                        .map_or(false, |r| !r.is_deleted())
                })
            })
            .count()
    }

    /// Returns all node IDs in the store.
    ///
    /// This returns a snapshot of current node IDs. The returned vector
    /// excludes deleted nodes. Results are sorted by NodeId for deterministic
    /// iteration order.
    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.with_node_ids_held(|ids| ids)
    }

    /// Calls `f` with [`node_ids`](Self::node_ids) while the node lock is
    /// held for a read, so no node is created, adopted or deleted until `f`
    /// returns. `f` may take only the locks that come after the node lock
    /// in the store's lock order.
    pub(super) fn with_node_ids_held<R>(&self, f: impl FnOnce(Vec<NodeId>) -> R) -> R {
        let epoch = self.current_epoch();
        #[cfg(not(feature = "tiered-storage"))]
        let nodes = self.nodes.read();
        #[cfg(not(feature = "tiered-storage"))]
        let mut ids: Vec<NodeId> = nodes
            .iter()
            .filter_map(|(id, chain)| {
                chain
                    .visible_at(epoch)
                    .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
            })
            .collect();
        #[cfg(feature = "tiered-storage")]
        let nodes = self.node_versions.read();
        #[cfg(feature = "tiered-storage")]
        let mut ids: Vec<NodeId> = nodes
            .iter()
            .filter_map(|(id, index)| {
                index.visible_at(epoch).and_then(|vref| {
                    self.read_node_record(&vref)
                        .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
                })
            })
            .collect();
        ids.sort_unstable();
        let result = f(ids);
        drop(nodes);
        result
    }

    /// [`node_ids`](Self::node_ids) for the readers that must not lose a
    /// node (checkpoints, copies). Every record of this build is in memory,
    /// so it never fails.
    ///
    /// # Errors
    ///
    /// None in this build; see the tiered storage version.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn try_node_ids(&self) -> grafeo_common::utils::error::Result<Vec<NodeId>> {
        Ok(self.node_ids())
    }

    /// [`node_ids`](Self::node_ids) for the readers that must not lose a
    /// node (checkpoints, copies): a record that cannot be read (a cold one
    /// that does not decode) is an error, never left out.
    /// (Tiered storage version)
    ///
    /// # Errors
    ///
    /// Returns an error naming the first node whose record cannot be read.
    #[cfg(feature = "tiered-storage")]
    pub fn try_node_ids(&self) -> grafeo_common::utils::error::Result<Vec<NodeId>> {
        let epoch = self.current_epoch();
        let versions = self.node_versions.read();
        let mut ids = Vec::with_capacity(versions.len());
        for (id, index) in versions.iter() {
            let Some(version_ref) = index.visible_at(epoch) else {
                continue;
            };
            let record = self
                .read_node_record(&version_ref)
                .ok_or_else(|| unreadable_node_record(*id))?;
            if !record.is_deleted() {
                ids.push(*id);
            }
        }
        ids.sort_unstable();
        Ok(ids)
    }

    /// Returns all node IDs including uncommitted/PENDING versions.
    ///
    /// Unlike `node_ids()` which pre-filters by current epoch, this returns
    /// every node that has a version chain entry. Used by scan operators that
    /// perform their own MVCC visibility filtering with transaction context.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.nodes.read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Returns all node IDs including uncommitted/PENDING versions.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.node_versions.read().keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}
