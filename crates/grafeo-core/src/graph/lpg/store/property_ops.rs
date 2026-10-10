//! Property operations for the LPG store.

use super::LpgStore;
#[cfg(feature = "temporal")]
use grafeo_common::types::EpochId;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
#[cfg(not(feature = "temporal"))]
use std::sync::Arc;

#[cfg(not(feature = "temporal"))]
use crate::graph::lpg::ColumnBacking;

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

        #[cfg(feature = "vector-index")]
        self.sync_vector_indexes_for_property(id, key);
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

        #[cfg(feature = "vector-index")]
        self.sync_vector_indexes_for_property(id, key);
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
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the value cannot be read
    /// (a spilled value whose file cannot be read): hidden unread, it would
    /// read as absent, so the removal would go unlogged and could not be
    /// undone.
    pub fn remove_node_property(
        &self,
        id: NodeId,
        key: &str,
    ) -> grafeo_common::utils::error::Result<Option<Value>> {
        let prop_key: PropertyKey = key.into();

        // The value is read and hidden in one step, before the indexes change.
        #[cfg(not(feature = "temporal"))]
        let removed = self.node_properties.remove(id, &prop_key)?;
        #[cfg(feature = "temporal")]
        let removed = self
            .node_properties
            .remove(id, &prop_key, self.current_epoch());
        self.update_indexes_on_remove(id, key, removed.as_ref());
        Ok(removed)
    }

    /// Brings the indexes in line with the removal of `key` from a node,
    /// whose value was `removed`.
    pub(super) fn update_indexes_on_remove(&self, id: NodeId, key: &str, removed: Option<&Value>) {
        if let Some(old_value) = removed {
            self.update_property_index_on_remove(id, &PropertyKey::new(key), old_value);
        }
        #[cfg(feature = "text-index")]
        self.update_text_index_on_remove(id, key);
        #[cfg(feature = "vector-index")]
        self.sync_vector_indexes_for_property(id, key);
    }

    /// Removes a property from an edge.
    ///
    /// Returns the previous value if it existed, or None if the property didn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the value cannot be read,
    /// as [`remove_node_property`](Self::remove_node_property) does (edge
    /// columns are not spilled today).
    pub fn remove_edge_property(
        &self,
        id: EdgeId,
        key: &str,
    ) -> grafeo_common::utils::error::Result<Option<Value>> {
        #[cfg(not(feature = "temporal"))]
        {
            self.edge_properties.remove(id, &key.into())
        }
        #[cfg(feature = "temporal")]
        {
            Ok(self
                .edge_properties
                .remove(id, &key.into(), self.current_epoch()))
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

    /// [`get_node_property_batch`](Self::get_node_property_batch) as a
    /// fallible read: a spilled value that cannot be read is an error, not
    /// `None`.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value.
    pub fn try_get_node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
    ) -> grafeo_common::utils::error::Result<Vec<Option<Value>>> {
        self.node_properties.try_get_batch(ids, key)
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

    // === Versioned Property Operations ===
    //
    // A transaction the engine runs writes through the store's change target
    // (`ChangeTarget::apply`), which returns what each write replaced for the
    // transaction's change set: its commit stamps and its rollback undoes
    // through that set. These methods write as a transaction without a
    // record: nothing commits or undoes what they wrote.

    /// Sets a node property as `transaction_id` (a PENDING version with
    /// temporal properties). Nothing records the value it replaced.
    ///
    /// # Errors
    ///
    /// Never fails today; the signature is the trait's.
    pub fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) -> grafeo_common::utils::error::Result<()> {
        let _ = transaction_id;
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
            #[cfg(feature = "vector-index")]
            self.sync_vector_indexes_for_property(id, key);
        }
        Ok(())
    }

    /// Sets an edge property as `transaction_id` (a PENDING version with
    /// temporal properties). Nothing records the value it replaced.
    pub fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let _ = transaction_id;
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

    /// Removes a node property as `transaction_id` (a PENDING tombstone with
    /// temporal properties); returns the value it had. Nothing records it.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the current value cannot
    /// be read (a spilled value whose file cannot be read).
    pub fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> grafeo_common::utils::error::Result<Option<Value>> {
        let _ = transaction_id;
        #[cfg(not(feature = "temporal"))]
        let removed = self.remove_node_property(id, key)?;

        // Temporal: the tombstone is this transaction's write, PENDING
        // until it commits.
        #[cfg(feature = "temporal")]
        let removed = {
            let removed = self.node_properties.remove(
                id,
                &key.into(),
                grafeo_common::types::EpochId::PENDING,
            );
            self.update_indexes_on_remove(id, key, removed.as_ref());
            removed
        };
        Ok(removed)
    }

    /// Removes an edge property as `transaction_id`, as
    /// [`remove_node_property_versioned`](Self::remove_node_property_versioned)
    /// does for a node.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the current value cannot
    /// be read.
    pub fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> grafeo_common::utils::error::Result<Option<Value>> {
        let _ = transaction_id;
        #[cfg(not(feature = "temporal"))]
        let removed = self.remove_edge_property(id, key)?;

        // Temporal: the tombstone is this transaction's write, PENDING
        // until it commits.
        #[cfg(feature = "temporal")]
        let removed =
            self.edge_properties
                .remove(id, &key.into(), grafeo_common::types::EpochId::PENDING);
        Ok(removed)
    }

    /// Takes back a node property that a rolled back write set where there
    /// was none. The value is not needed, so one that cannot be read (spilled
    /// meanwhile, to a file that cannot be read) is hidden all the same; a
    /// property index on the key then keeps its entry for the node, as the
    /// value it was filed under is unknown.
    #[cfg(not(feature = "temporal"))]
    pub(super) fn undo_node_property_set(&self, id: NodeId, key: &PropertyKey) {
        if self.remove_node_property(id, key.as_str()).is_err() {
            self.node_properties.discard(id, key);
            self.update_indexes_on_remove(id, key.as_str(), None);
        }
    }

    /// [`undo_node_property_set`](Self::undo_node_property_set) for an edge.
    #[cfg(not(feature = "temporal"))]
    pub(super) fn undo_edge_property_set(&self, id: EdgeId, key: &PropertyKey) {
        if self.remove_edge_property(id, key.as_str()).is_err() {
            self.edge_properties.discard(id, key);
        }
    }

    // === Column reads and spill ===

    /// Fills values of `key` from a spill file an older build left behind
    /// (#594): a load step, like WAL replay, so no undo entry, WAL record,
    /// CDC event or new epoch. A value fills only a node that exists, carries
    /// `label` (the label of the file's vector index, so a node of another
    /// label that took a deleted node's id gets nothing) and has no value for
    /// `key` (a value written later is in the store and wins), so a second run
    /// changes nothing. A vector index on the property that lacks a filled
    /// node (one rebuilt from the data before the fill) gets it. Returns how
    /// many values were filled.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a stored value (a spilled value whose
    /// file cannot be read), which would read as missing and be overwritten;
    /// the values filled before it stay.
    pub fn fill_missing_node_values(
        &self,
        label: &str,
        key: &PropertyKey,
        values: impl IntoIterator<Item = (NodeId, Value)>,
    ) -> grafeo_common::utils::error::Result<usize> {
        let Some(label_id) = self.label_registry.read().get_id(label) else {
            return Ok(0);
        };
        let epoch = self.current_epoch();
        let mut filled = 0;
        for (id, value) in values {
            let has_label = self
                .label_index
                .read()
                .get(label_id as usize)
                .is_some_and(|members| members.contains_key(&id));
            if !has_label
                || !self.is_node_visible_at_epoch(id, epoch)
                || self.node_properties.try_get(id, key)?.is_some()
            {
                continue;
            }
            self.update_property_index_on_set(id, key, &value);
            #[cfg(not(feature = "temporal"))]
            self.node_properties.set(id, key.clone(), value);
            #[cfg(feature = "temporal")]
            self.node_properties.set(id, key.clone(), value, epoch);
            #[cfg(feature = "vector-index")]
            self.index_vector_if_missing(id, key);
            filled += 1;
        }
        Ok(filled)
    }

    /// Returns the nodes with a value for `key`, in id order, spilled values
    /// included.
    #[must_use]
    pub fn node_property_column_ids(&self, key: &PropertyKey) -> Vec<NodeId> {
        self.node_properties.column_ids(key)
    }

    /// Returns the edges with a value for `key`, in id order.
    #[must_use]
    pub fn edge_property_column_ids(&self, key: &PropertyKey) -> Vec<EdgeId> {
        self.edge_properties.column_ids(key)
    }

    /// Returns every `(node, value)` of `key`, in id order, spilled values
    /// included: the snapshot a spill writes to its backing.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value: a snapshot never leaves
    /// one out.
    pub fn node_property_column_entries(
        &self,
        key: &PropertyKey,
    ) -> grafeo_common::utils::error::Result<Vec<(NodeId, Value)>> {
        self.node_properties.try_column_entries(key)
    }

    /// Calls `f` with the vector stored for a node under `key`, without
    /// copying a spilled vector; `None` when there is no vector. `f` runs
    /// without the property storage lock held, so it may read the store
    /// again (pairwise distances do).
    pub fn with_node_vector<R>(
        &self,
        id: NodeId,
        key: &PropertyKey,
        f: impl FnOnce(&[f32]) -> R,
    ) -> Option<R> {
        self.node_properties.with_vector(id, key, f)
    }

    /// Spills a node property column into `backing`, which holds `snapshot`
    /// (from [`node_property_column_entries`](Self::node_property_column_entries)).
    /// See [`PropertyStorage::spill_column`](crate::graph::lpg::PropertyStorage::spill_column).
    #[cfg(not(feature = "temporal"))]
    pub fn spill_node_property_column(
        &self,
        key: &PropertyKey,
        backing: Arc<dyn ColumnBacking<NodeId>>,
        snapshot: &[(NodeId, Value)],
    ) -> bool {
        self.node_properties.spill_column(key, backing, snapshot)
    }

    /// Moves a spilled node property column back onto the heap; `false` when
    /// it is not spilled.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value; the column then stays
    /// spilled.
    #[cfg(not(feature = "temporal"))]
    pub fn reload_node_property_column(
        &self,
        key: &PropertyKey,
    ) -> grafeo_common::utils::error::Result<bool> {
        self.node_properties.reload_column(key)
    }

    /// Returns the keys of the spilled node property columns, in key order.
    #[cfg(not(feature = "temporal"))]
    #[must_use]
    pub fn spilled_node_property_columns(&self) -> Vec<PropertyKey> {
        self.node_properties.spilled_columns()
    }
}
