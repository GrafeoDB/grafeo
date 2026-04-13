//! Property operations for the LPG store.

use super::LpgStore;
use super::PropertyUndoEntry;
#[cfg(feature = "temporal")]
use grafeo_common::types::EpochId;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
use std::sync::atomic::Ordering;

impl LpgStore {
    /// Sets a property on a node.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let prop_key: PropertyKey = key.into();

        // Update property index before setting the property (needs to read old value)
        self.update_property_index_on_set(id, &prop_key, &value);

        // Sync text index if applicable
        #[cfg(feature = "text-index")]
        self.update_text_index_on_set(id, key, &value);

        #[cfg(not(feature = "temporal"))]
        self.node_properties.set(id, prop_key, value);
        #[cfg(feature = "temporal")]
        self.node_properties
            .set(id, prop_key, value, self.current_epoch());

        // Update props_count in record
        #[cfg(not(feature = "temporal"))]
        {
            let count = self.node_properties.get_all(id).len() as u16;
            if let Some(chain) = self.nodes.write().get_mut(&id)
                && let Some(record) = chain.latest_mut()
            {
                record.props_count = count;
            }
        }
    }

    /// Sets a property on a node.
    /// (Tiered storage version: properties stored separately, record is immutable)
    #[cfg(feature = "tiered-storage")]
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let prop_key: PropertyKey = key.into();

        // Update property index before setting the property (needs to read old value)
        self.update_property_index_on_set(id, &prop_key, &value);

        // Sync text index if applicable
        #[cfg(feature = "text-index")]
        self.update_text_index_on_set(id, key, &value);

        #[cfg(not(feature = "temporal"))]
        self.node_properties.set(id, prop_key, value);
        #[cfg(feature = "temporal")]
        self.node_properties
            .set(id, prop_key, value, self.current_epoch());
    }

    /// Sets a property on an edge.
    pub fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        #[cfg(not(feature = "temporal"))]
        self.edge_properties.set(id, key.into(), value);
        #[cfg(feature = "temporal")]
        self.edge_properties
            .set(id, key.into(), value, self.current_epoch());
    }

    /// Sets a node property at a specific epoch (for snapshot/WAL recovery).
    ///
    /// Unlike [`LpgStore::set_node_property`], this does not update property indexes
    /// or text indexes, and uses the provided epoch instead of `current_epoch()`.
    #[cfg(feature = "temporal")]
    pub fn set_node_property_at_epoch(&self, id: NodeId, key: &str, value: Value, epoch: EpochId) {
        self.node_properties.set(id, key.into(), value, epoch);
    }

    /// Sets an edge property at a specific epoch (for snapshot/WAL recovery).
    #[cfg(feature = "temporal")]
    pub fn set_edge_property_at_epoch(&self, id: EdgeId, key: &str, value: Value, epoch: EpochId) {
        self.edge_properties.set(id, key.into(), value, epoch);
    }

    /// Returns the full version history for all properties of a node.
    ///
    /// Each entry is `(key, Vec<(epoch, value)>)`. Used for temporal
    /// snapshot export.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn node_property_history(&self, id: NodeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        self.node_properties.get_all_history(id)
    }

    /// Returns a property value at a specific epoch.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn get_node_property_at_epoch(
        &self,
        id: NodeId,
        key: &PropertyKey,
        epoch: EpochId,
    ) -> Option<Value> {
        self.node_properties.get_at(id, key, epoch)
    }

    /// Returns the version history for a single property of a node.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn node_property_history_for_key(&self, id: NodeId, key: &str) -> Vec<(EpochId, Value)> {
        self.node_properties.get_history(id, &PropertyKey::new(key))
    }

    /// Returns the full version history for all properties of an edge.
    #[cfg(feature = "temporal")]
    #[must_use]
    pub fn edge_property_history(&self, id: EdgeId) -> Vec<(PropertyKey, Vec<(EpochId, Value)>)> {
        self.edge_properties.get_all_history(id)
    }

    /// Removes a property from a node.
    ///
    /// Returns the previous value if it existed, or None if the property didn't exist.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let prop_key: PropertyKey = key.into();

        // Update property index before removing (needs to read old value)
        self.update_property_index_on_remove(id, &prop_key);

        // Sync text index if applicable
        #[cfg(feature = "text-index")]
        self.update_text_index_on_remove(id, key);

        #[cfg(not(feature = "temporal"))]
        let result = self.node_properties.remove(id, &prop_key);
        #[cfg(feature = "temporal")]
        let result = self
            .node_properties
            .remove(id, &prop_key, self.current_epoch());

        // Update props_count in record
        #[cfg(not(feature = "temporal"))]
        {
            let count = self.node_properties.get_all(id).len() as u16;
            if let Some(chain) = self.nodes.write().get_mut(&id)
                && let Some(record) = chain.latest_mut()
            {
                record.props_count = count;
            }
        }

        result
    }

    /// Removes a property from a node.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let prop_key: PropertyKey = key.into();

        // Update property index before removing (needs to read old value)
        self.update_property_index_on_remove(id, &prop_key);

        // Sync text index if applicable
        #[cfg(feature = "text-index")]
        self.update_text_index_on_remove(id, key);

        #[cfg(not(feature = "temporal"))]
        {
            self.node_properties.remove(id, &prop_key)
        }
        #[cfg(feature = "temporal")]
        {
            self.node_properties
                .remove(id, &prop_key, self.current_epoch())
        }
    }

    /// Removes a property from an edge.
    ///
    /// Returns the previous value if it existed, or None if the property didn't exist.
    pub fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        #[cfg(not(feature = "temporal"))]
        {
            self.edge_properties.remove(id, &key.into())
        }
        #[cfg(feature = "temporal")]
        {
            self.edge_properties
                .remove(id, &key.into(), self.current_epoch())
        }
    }

    /// Gets a single property from a node without loading all properties.
    ///
    /// This is O(1) vs O(properties) for `get_node().get_property()`.
    /// Use this for filter predicates where you only need one property value.
    ///
    /// # Example
    ///
    /// ```
    /// # use grafeo_core::graph::lpg::LpgStore;
    /// # use grafeo_common::types::{PropertyKey, Value};
    /// let store = LpgStore::new().expect("arena allocation");
    /// let node_id = store.create_node(&["Person"]);
    /// store.set_node_property(node_id, "age", Value::from(30i64));
    ///
    /// // Fast: Direct single-property lookup
    /// let age = store.get_node_property(node_id, &PropertyKey::new("age"));
    ///
    /// // Slow: Loads all properties, then extracts one
    /// let age = store.get_node(node_id).and_then(|n| n.get_property("age").cloned());
    /// ```
    #[must_use]
    pub fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.node_properties.get(id, key)
    }

    /// Gets a single property from an edge without loading all properties.
    ///
    /// This is O(1) vs O(properties) for `get_edge().get_property()`.
    #[must_use]
    pub fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.edge_properties.get(id, key)
    }

    // === Batch Property Operations ===

    /// Gets a property for multiple nodes in a single batch operation.
    ///
    /// More efficient than calling [`Self::get_node_property`] in a loop because it
    /// reduces lock overhead and enables better cache utilization.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::{NodeId, PropertyKey, Value};
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    /// let n1 = store.create_node(&["Person"]);
    /// let n2 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "age", Value::from(25i64));
    /// store.set_node_property(n2, "age", Value::from(30i64));
    ///
    /// let ages = store.get_node_property_batch(&[n1, n2], &PropertyKey::new("age"));
    /// assert_eq!(ages, vec![Some(Value::from(25i64)), Some(Value::from(30i64))]);
    /// ```
    #[must_use]
    pub fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        self.node_properties.get_batch(ids, key)
    }

    /// Gets all properties for multiple nodes in a single batch operation.
    ///
    /// Returns a vector of property maps, one per node ID (empty map if no properties).
    /// More efficient than calling [`Self::get_node`] in a loop.
    #[must_use]
    pub fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.node_properties.get_all_batch(ids)
    }

    /// Gets selected properties for multiple nodes (projection pushdown).
    ///
    /// This is more efficient than [`Self::get_nodes_properties_batch`] when you only
    /// need a subset of properties. It only iterates the requested columns instead of
    /// all columns.
    ///
    /// **Use this for**: Queries with explicit projections like `RETURN n.name, n.age`
    /// instead of `RETURN n` (which requires all properties).
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::{PropertyKey, Value};
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    /// let n1 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "name", Value::from("Alix"));
    /// store.set_node_property(n1, "age", Value::from(30i64));
    /// store.set_node_property(n1, "email", Value::from("alix@example.com"));
    ///
    /// // Only fetch name and age (faster than get_nodes_properties_batch)
    /// let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
    /// let props = store.get_nodes_properties_selective_batch(&[n1], &keys);
    ///
    /// assert_eq!(props[0].len(), 2); // Only name and age, not email
    /// ```
    #[must_use]
    pub fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.node_properties.get_selective_batch(ids, keys)
    }

    /// Gets selected properties for multiple edges (projection pushdown).
    ///
    /// Edge-property version of [`Self::get_nodes_properties_selective_batch`].
    #[must_use]
    pub fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.edge_properties.get_selective_batch(ids, keys)
    }

    /// Returns all property overrides for a node ID from PropertyStorage.
    ///
    /// Does not check the nodes map -- works for compact entity IDs that only
    /// have overlay property overrides stored in [`PropertyStorage`].
    pub(crate) fn get_node_properties_all(&self, id: NodeId) -> FxHashMap<PropertyKey, Value> {
        self.node_properties.get_all(id)
    }

    /// Returns all property overrides for an edge ID from PropertyStorage.
    ///
    /// Does not check the edges map -- works for compact entity IDs.
    pub(crate) fn get_edge_properties_all(&self, id: EdgeId) -> FxHashMap<PropertyKey, Value> {
        self.edge_properties.get_all(id)
    }

    /// Returns all node IDs with a property override matching `predicate`.
    ///
    /// Scans the overlay's PropertyStorage column for `key`, returning IDs
    /// whose overridden value satisfies the predicate. Used by HybridStore
    /// to find compact nodes with overlay overrides matching scan criteria.
    pub(crate) fn node_ids_with_property_matching(
        &self,
        key: &PropertyKey,
        predicate: impl Fn(&Value) -> bool,
    ) -> Vec<NodeId> {
        self.node_properties.ids_matching(key, predicate)
    }

    // === Versioned Property Operations (with undo log) ===

    /// Sets a node property within a transaction, recording the previous value
    /// in the undo log so it can be restored on rollback.
    pub fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let prop_key: PropertyKey = key.into();

        // Capture the current value before overwriting
        let old_value = self.node_properties.get(id, &prop_key);

        // Record in undo log
        self.property_undo_log
            .write()
            .entry(transaction_id)
            .or_default()
            .push(PropertyUndoEntry::NodeProperty {
                node_id: id,
                key: prop_key,
                old_value,
            });

        // Delegate to the normal (unversioned) set
        #[cfg(not(feature = "temporal"))]
        self.set_node_property(id, key, value);
        // For temporal: use PENDING epoch directly (finalized on commit)
        #[cfg(feature = "temporal")]
        {
            let prop_key2: PropertyKey = key.into();
            self.update_property_index_on_set(id, &prop_key2, &value);
            #[cfg(feature = "text-index")]
            self.update_text_index_on_set(id, key, &value);
            self.node_properties
                .set(id, prop_key2, value, grafeo_common::types::EpochId::PENDING);
        }
    }

    /// Sets an edge property within a transaction, recording the previous value
    /// in the undo log so it can be restored on rollback.
    pub fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let prop_key: PropertyKey = key.into();

        // Capture the current value before overwriting
        let old_value = self.edge_properties.get(id, &prop_key);

        // Record in undo log
        self.property_undo_log
            .write()
            .entry(transaction_id)
            .or_default()
            .push(PropertyUndoEntry::EdgeProperty {
                edge_id: id,
                key: prop_key,
                old_value,
            });

        // Delegate to the normal (unversioned) set
        #[cfg(not(feature = "temporal"))]
        self.set_edge_property(id, key, value);
        #[cfg(feature = "temporal")]
        self.edge_properties.set(
            id,
            key.into(),
            value,
            grafeo_common::types::EpochId::PENDING,
        );
    }

    /// Removes a node property within a transaction, recording the previous value
    /// in the undo log so it can be restored on rollback.
    pub fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let prop_key: PropertyKey = key.into();

        // Capture the current value before removing
        let old_value = self.node_properties.get(id, &prop_key);

        // Only record if the property actually exists
        if old_value.is_some() {
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(PropertyUndoEntry::NodeProperty {
                    node_id: id,
                    key: prop_key,
                    old_value: old_value.clone(),
                });
        }

        // Delegate to the normal (unversioned) remove
        self.remove_node_property(id, key)
    }

    /// Removes an edge property within a transaction, recording the previous value
    /// in the undo log so it can be restored on rollback.
    pub fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let prop_key: PropertyKey = key.into();

        // Capture the current value before removing
        let old_value = self.edge_properties.get(id, &prop_key);

        // Only record if the property actually exists
        if old_value.is_some() {
            self.property_undo_log
                .write()
                .entry(transaction_id)
                .or_default()
                .push(PropertyUndoEntry::EdgeProperty {
                    edge_id: id,
                    key: prop_key,
                    old_value: old_value.clone(),
                });
        }

        // Delegate to the normal (unversioned) remove
        self.remove_edge_property(id, key)
    }

    /// Replays the undo log for a transaction in reverse order, restoring
    /// all property values to their pre-transaction state.
    ///
    /// Called during rollback.
    #[cfg(not(feature = "temporal"))]
    pub fn rollback_transaction_properties(&self, transaction_id: TransactionId) {
        let entries = self.property_undo_log.write().remove(&transaction_id);
        if let Some(entries) = entries {
            // Replay in reverse order: latest change first
            for entry in entries.into_iter().rev() {
                match entry {
                    PropertyUndoEntry::NodeProperty {
                        node_id,
                        key,
                        old_value,
                    } => {
                        if let Some(value) = old_value {
                            // Restore the old value (bypass undo log, write directly)
                            self.set_node_property(node_id, key.as_str(), value);
                        } else {
                            // Property did not exist before: remove it
                            self.remove_node_property(node_id, key.as_str());
                        }
                    }
                    PropertyUndoEntry::EdgeProperty {
                        edge_id,
                        key,
                        old_value,
                    } => {
                        if let Some(value) = old_value {
                            self.set_edge_property(edge_id, key.as_str(), value);
                        } else {
                            self.remove_edge_property(edge_id, key.as_str());
                        }
                    }
                    PropertyUndoEntry::LabelAdded { node_id, label } => {
                        self.remove_label(node_id, &label);
                    }
                    PropertyUndoEntry::LabelRemoved { node_id, label } => {
                        self.add_label(node_id, &label);
                    }
                    PropertyUndoEntry::NodeDeleted {
                        node_id,
                        labels,
                        properties,
                    } => {
                        self.restore_deleted_node(node_id, transaction_id, &labels, properties);
                    }
                    PropertyUndoEntry::EdgeDeleted {
                        edge_id,
                        src,
                        dst,
                        edge_type,
                        properties,
                    } => {
                        self.restore_deleted_edge(
                            edge_id,
                            src,
                            dst,
                            transaction_id,
                            &edge_type,
                            properties,
                        );
                    }
                }
            }
        }
        if let Some(ref hook) = *self.on_rollback_hook.lock() {
            hook(transaction_id);
        }
    }

    /// Rolls back property/label changes by removing PENDING entries from
    /// version logs, and replays entity deletions from the undo log.
    ///
    /// With temporal properties, there is no need to replay old property
    /// values: `remove_pending()` pops the uncommitted PENDING entries
    /// from the back of each VersionLog, restoring the previous state.
    #[cfg(feature = "temporal")]
    pub fn rollback_transaction_properties(&self, transaction_id: TransactionId) {
        let entries = self.property_undo_log.write().remove(&transaction_id);
        if let Some(entries) = entries {
            // Collect which node/edge properties and labels were touched
            let mut node_props: grafeo_common::utils::hash::FxHashSet<(NodeId, PropertyKey)> =
                grafeo_common::utils::hash::FxHashSet::default();
            let mut edge_props: grafeo_common::utils::hash::FxHashSet<(EdgeId, PropertyKey)> =
                grafeo_common::utils::hash::FxHashSet::default();
            let mut label_nodes: grafeo_common::utils::hash::FxHashSet<NodeId> =
                grafeo_common::utils::hash::FxHashSet::default();

            // First pass: collect touched entries and handle entity deletions
            for entry in entries.into_iter().rev() {
                match entry {
                    PropertyUndoEntry::NodeProperty { node_id, key, .. } => {
                        node_props.insert((node_id, key));
                    }
                    PropertyUndoEntry::EdgeProperty { edge_id, key, .. } => {
                        edge_props.insert((edge_id, key));
                    }
                    PropertyUndoEntry::LabelAdded { node_id, .. }
                    | PropertyUndoEntry::LabelRemoved { node_id, .. } => {
                        label_nodes.insert(node_id);
                    }
                    PropertyUndoEntry::NodeDeleted {
                        node_id,
                        labels,
                        properties,
                    } => {
                        self.restore_deleted_node(node_id, transaction_id, &labels, properties);
                    }
                    PropertyUndoEntry::EdgeDeleted {
                        edge_id,
                        src,
                        dst,
                        edge_type,
                        properties,
                    } => {
                        self.restore_deleted_edge(
                            edge_id,
                            src,
                            dst,
                            transaction_id,
                            &edge_type,
                            properties,
                        );
                    }
                }
            }

            // Remove PENDING entries from affected property version logs
            if !node_props.is_empty() {
                let mut columns = self.node_properties.columns_write();
                for (node_id, key) in &node_props {
                    if let Some(col) = columns.get_mut(key) {
                        col.remove_pending_for(*node_id);
                    }
                }
            }

            if !edge_props.is_empty() {
                let mut columns = self.edge_properties.columns_write();
                for (edge_id, key) in &edge_props {
                    if let Some(col) = columns.get_mut(key) {
                        col.remove_pending_for(*edge_id);
                    }
                }
            }

            // Remove PENDING entries from affected label version logs and
            // reconcile label_index to match the restored state.
            if !label_nodes.is_empty() {
                let mut labels = self.node_labels.write();
                let mut index = self.label_index.write();

                for node_id in &label_nodes {
                    // Get the label set BEFORE removing PENDING (the transactional state)
                    let tx_labels = labels
                        .get(node_id)
                        .and_then(|log| log.latest())
                        .cloned()
                        .unwrap_or_default();

                    // Remove PENDING entries to restore pre-transaction state
                    if let Some(log) = labels.get_mut(node_id) {
                        log.remove_pending();
                    }

                    // Get the restored (pre-transaction) label set
                    let restored_labels = labels
                        .get(node_id)
                        .and_then(|log| log.latest())
                        .cloned()
                        .unwrap_or_default();

                    // Reconcile label_index: remove labels that were added by the
                    // transaction, re-add labels that were removed by the transaction.
                    for label_id in &tx_labels {
                        if !restored_labels.contains(label_id) && (*label_id as usize) < index.len()
                        {
                            index[*label_id as usize].remove(node_id);
                        }
                    }
                    for label_id in &restored_labels {
                        if !tx_labels.contains(label_id) && (*label_id as usize) < index.len() {
                            index[*label_id as usize].insert(*node_id, ());
                        }
                    }
                }
            }
        }
        if let Some(ref hook) = *self.on_rollback_hook.lock() {
            hook(transaction_id);
        }
    }

    /// Discards the undo log entries for a committed transaction.
    ///
    /// Called during commit: properties are already written, so just
    /// clean up the log.
    pub fn commit_transaction_properties(&self, transaction_id: TransactionId) {
        self.property_undo_log.write().remove(&transaction_id);
        if let Some(ref hook) = *self.on_commit_hook.lock() {
            hook(transaction_id);
        }
    }

    /// Returns the current number of undo log entries for a transaction.
    ///
    /// Used by savepoints to record the position so that partial rollback
    /// can replay only entries added after the savepoint.
    #[must_use]
    pub fn property_undo_log_position(&self, transaction_id: TransactionId) -> usize {
        self.property_undo_log
            .read()
            .get(&transaction_id)
            .map_or(0, Vec::len)
    }

    /// Rolls back property mutations recorded after position `since` in the undo log.
    ///
    /// Replays entries from `since..end` in reverse order, then truncates the
    /// log to `since`. Used by savepoint rollback.
    #[cfg(not(feature = "temporal"))]
    pub fn rollback_transaction_properties_to(&self, transaction_id: TransactionId, since: usize) {
        let mut log = self.property_undo_log.write();
        if let Some(entries) = log.get_mut(&transaction_id)
            && since < entries.len()
        {
            // Take entries after the savepoint position
            let to_undo: Vec<PropertyUndoEntry> = entries.drain(since..).collect();
            // Drop the lock before replaying to avoid deadlock
            // (rollback methods need to acquire other locks)
            drop(log);
            // Replay in reverse order
            for entry in to_undo.into_iter().rev() {
                match entry {
                    PropertyUndoEntry::NodeProperty {
                        node_id,
                        key,
                        old_value,
                    } => {
                        if let Some(value) = old_value {
                            self.set_node_property(node_id, key.as_str(), value);
                        } else {
                            self.remove_node_property(node_id, key.as_str());
                        }
                    }
                    PropertyUndoEntry::EdgeProperty {
                        edge_id,
                        key,
                        old_value,
                    } => {
                        if let Some(value) = old_value {
                            self.set_edge_property(edge_id, key.as_str(), value);
                        } else {
                            self.remove_edge_property(edge_id, key.as_str());
                        }
                    }
                    PropertyUndoEntry::LabelAdded { node_id, label } => {
                        self.remove_label(node_id, &label);
                    }
                    PropertyUndoEntry::LabelRemoved { node_id, label } => {
                        self.add_label(node_id, &label);
                    }
                    PropertyUndoEntry::NodeDeleted {
                        node_id,
                        labels,
                        properties,
                    } => {
                        self.restore_deleted_node(node_id, transaction_id, &labels, properties);
                    }
                    PropertyUndoEntry::EdgeDeleted {
                        edge_id,
                        src,
                        dst,
                        edge_type,
                        properties,
                    } => {
                        self.restore_deleted_edge(
                            edge_id,
                            src,
                            dst,
                            transaction_id,
                            &edge_type,
                            properties,
                        );
                    }
                }
            }
        }
    }

    /// Rolls back property mutations recorded after position `since` in the undo log.
    ///
    /// Temporal version: instead of replaying old values (which would create
    /// new VersionLog entries), this pops the PENDING entries that were appended
    /// after the savepoint. Entity deletions are still restored via the normal
    /// `restore_deleted_node`/`restore_deleted_edge` helpers.
    #[cfg(feature = "temporal")]
    pub fn rollback_transaction_properties_to(&self, transaction_id: TransactionId, since: usize) {
        let mut log = self.property_undo_log.write();
        if let Some(entries) = log.get_mut(&transaction_id)
            && since < entries.len()
        {
            let to_undo: Vec<PropertyUndoEntry> = entries.drain(since..).collect();
            drop(log);

            // Count how many PENDING entries to pop per (entity, key) and per label node.
            let mut node_prop_counts: grafeo_common::utils::hash::FxHashMap<
                (NodeId, PropertyKey),
                usize,
            > = grafeo_common::utils::hash::FxHashMap::default();
            let mut edge_prop_counts: grafeo_common::utils::hash::FxHashMap<
                (EdgeId, PropertyKey),
                usize,
            > = grafeo_common::utils::hash::FxHashMap::default();
            let mut label_counts: grafeo_common::utils::hash::FxHashMap<NodeId, usize> =
                grafeo_common::utils::hash::FxHashMap::default();

            for entry in to_undo.into_iter().rev() {
                match entry {
                    PropertyUndoEntry::NodeProperty { node_id, key, .. } => {
                        *node_prop_counts.entry((node_id, key)).or_default() += 1;
                    }
                    PropertyUndoEntry::EdgeProperty { edge_id, key, .. } => {
                        *edge_prop_counts.entry((edge_id, key)).or_default() += 1;
                    }
                    PropertyUndoEntry::LabelAdded { node_id, .. }
                    | PropertyUndoEntry::LabelRemoved { node_id, .. } => {
                        *label_counts.entry(node_id).or_default() += 1;
                    }
                    PropertyUndoEntry::NodeDeleted {
                        node_id,
                        labels,
                        properties,
                    } => {
                        self.restore_deleted_node(node_id, transaction_id, &labels, properties);
                    }
                    PropertyUndoEntry::EdgeDeleted {
                        edge_id,
                        src,
                        dst,
                        edge_type,
                        properties,
                    } => {
                        self.restore_deleted_edge(
                            edge_id,
                            src,
                            dst,
                            transaction_id,
                            &edge_type,
                            properties,
                        );
                    }
                }
            }

            // Pop PENDING entries from node property version logs
            if !node_prop_counts.is_empty() {
                let mut columns = self.node_properties.columns_write();
                for ((node_id, key), count) in &node_prop_counts {
                    if let Some(col) = columns.get_mut(key) {
                        col.pop_n_pending_for(*node_id, *count);
                    }
                }
            }

            // Pop PENDING entries from edge property version logs
            if !edge_prop_counts.is_empty() {
                let mut columns = self.edge_properties.columns_write();
                for ((edge_id, key), count) in &edge_prop_counts {
                    if let Some(col) = columns.get_mut(key) {
                        col.pop_n_pending_for(*edge_id, *count);
                    }
                }
            }

            // Pop PENDING entries from label version logs and reconcile label_index
            if !label_counts.is_empty() {
                let mut labels = self.node_labels.write();
                let mut index = self.label_index.write();

                for (node_id, count) in &label_counts {
                    let tx_labels = labels
                        .get(node_id)
                        .and_then(|log| log.latest())
                        .cloned()
                        .unwrap_or_default();

                    if let Some(version_log) = labels.get_mut(node_id) {
                        version_log.pop_n_pending(*count);
                    }

                    let restored_labels = labels
                        .get(node_id)
                        .and_then(|log| log.latest())
                        .cloned()
                        .unwrap_or_default();

                    // Reconcile label_index
                    for label_id in &tx_labels {
                        if !restored_labels.contains(label_id) && (*label_id as usize) < index.len()
                        {
                            index[*label_id as usize].remove(node_id);
                        }
                    }
                    for label_id in &restored_labels {
                        if !tx_labels.contains(label_id) && (*label_id as usize) < index.len() {
                            index[*label_id as usize].insert(*node_id, ());
                        }
                    }
                }
            }
        }
    }

    // === Deletion Restoration Helpers ===

    /// Restores a node that was deleted within a transaction.
    ///
    /// Called during rollback to undo a transactional node deletion.
    fn restore_deleted_node(
        &self,
        node_id: NodeId,
        transaction_id: TransactionId,
        labels: &[String],
        properties: Vec<(PropertyKey, Value)>,
    ) {
        // Unmark deletion on version chain
        #[cfg(not(feature = "tiered-storage"))]
        {
            let mut nodes = self.nodes.write();
            if let Some(chain) = nodes.get_mut(&node_id) {
                chain.unmark_deleted_by(transaction_id);
            }
        }
        #[cfg(feature = "tiered-storage")]
        {
            let mut versions = self.node_versions.write();
            if let Some(index) = versions.get_mut(&node_id) {
                index.unmark_deleted_by(transaction_id);
            }
        }

        // Restore label index entries
        for label in labels {
            self.add_label(node_id, label);
        }

        // Restore properties
        for (key, value) in properties {
            self.set_node_property(node_id, key.as_str(), value);
        }

        self.live_node_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Restores an edge that was deleted within a transaction.
    ///
    /// Called during rollback to undo a transactional edge deletion.
    fn restore_deleted_edge(
        &self,
        edge_id: EdgeId,
        src: NodeId,
        dst: NodeId,
        transaction_id: TransactionId,
        edge_type: &str,
        properties: Vec<(PropertyKey, Value)>,
    ) {
        // Unmark deletion on version chain
        #[cfg(not(feature = "tiered-storage"))]
        {
            let mut edges = self.edges.write();
            if let Some(chain) = edges.get_mut(&edge_id) {
                chain.unmark_deleted_by(transaction_id);
            }
        }
        #[cfg(feature = "tiered-storage")]
        {
            let mut versions = self.edge_versions.write();
            if let Some(index) = versions.get_mut(&edge_id) {
                index.unmark_deleted_by(transaction_id);
            }
        }

        // Restore adjacency (unmark soft-delete)
        self.forward_adj.unmark_deleted(src, edge_id);
        if let Some(ref backward) = self.backward_adj {
            backward.unmark_deleted(dst, edge_id);
        }

        // Restore properties
        for (key, value) in properties {
            self.set_edge_property(edge_id, key.as_str(), value);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);

        // Restore edge type count
        let type_id = {
            let type_map = self.edge_type_to_id.read();
            type_map.get(edge_type).copied()
        };
        if let Some(type_id) = type_id {
            self.increment_edge_type_count(type_id);
        }
    }

    // === Column-Level Spill / Reload ===

    /// Drains all values from a node property column, returning them for export.
    ///
    /// After this call, `is_node_column_spilled(key)` returns `true` and
    /// `get_node_property(id, key)` returns `None` for all IDs.
    /// Used by the vector spill path to export embeddings to `MmapStorage`.
    #[cfg(not(feature = "temporal"))]
    pub fn drain_node_property_column(&self, key: &PropertyKey) -> Vec<(NodeId, Value)> {
        self.node_properties.drain_column(key)
    }

    /// Restores values into a previously spilled node property column.
    #[cfg(not(feature = "temporal"))]
    pub fn restore_node_property_column(
        &self,
        key: &PropertyKey,
        values: impl Iterator<Item = (NodeId, Value)>,
    ) {
        self.node_properties.restore_column(key, values);
    }

    /// Whether a node property column has been spilled to disk.
    #[cfg(not(feature = "temporal"))]
    #[must_use]
    pub fn is_node_column_spilled(&self, key: &PropertyKey) -> bool {
        self.node_properties.is_column_spilled(key)
    }

    /// Marks a node property column as spilled without draining it.
    ///
    /// Used during startup to re-establish spill state for columns that
    /// were already empty (serialized without values in the previous session).
    #[cfg(not(feature = "temporal"))]
    pub fn node_properties_mark_spilled(&self, key: &PropertyKey) {
        self.node_properties.mark_column_spilled(key);
    }
}
