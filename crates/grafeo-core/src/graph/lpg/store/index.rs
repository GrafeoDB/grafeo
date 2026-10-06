//! Index management methods for [`LpgStore`].

use super::LpgStore;
use dashmap::DashMap;
use grafeo_common::types::{HashableValue, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashSet;
#[cfg(feature = "text-index")]
use parking_lot::RwLock;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use std::sync::Arc;

#[cfg(feature = "vector-index")]
use crate::index::vector::VectorIndexKind;

impl LpgStore {
    /// Creates an index on a node property for O(1) lookups by value.
    ///
    /// After creating an index, calls to [`Self::find_nodes_by_property`] will be
    /// O(1) instead of O(n) for this property. The index is automatically
    /// maintained when properties are set or removed.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::Value;
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    ///
    /// // Create nodes with an 'id' property
    /// let alix = store.create_node(&["Person"]);
    /// store.set_node_property(alix, "id", Value::from("alice_123"));
    ///
    /// // Create an index on the 'id' property
    /// store.create_property_index("id");
    ///
    /// // Now lookups by 'id' are O(1)
    /// let found = store.find_nodes_by_property("id", &Value::from("alice_123"));
    /// assert!(found.contains(&alix));
    /// ```
    pub fn create_property_index(&self, property: &str) {
        let key = PropertyKey::new(property);

        let mut indexes = self.property_indexes.write();
        if indexes.contains_key(&key) {
            return; // Already indexed
        }

        // Create the index and populate it with existing data
        let index: DashMap<HashableValue, FxHashSet<NodeId>> = DashMap::new();

        // Scan all nodes to build the index
        for node_id in self.node_ids() {
            if let Some(value) = self.node_properties.get(node_id, &key) {
                let hv = HashableValue::new(value);
                index.entry(hv).or_default().insert(node_id);
            }
        }

        indexes.insert(key, index);
    }

    /// Drops an index on a node property.
    ///
    /// Returns `true` if the index existed and was removed.
    pub fn drop_property_index(&self, property: &str) -> bool {
        let key = PropertyKey::new(property);
        self.property_indexes.write().remove(&key).is_some()
    }

    /// Returns `true` if the property has an index.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        let key = PropertyKey::new(property);
        self.property_indexes.read().contains_key(&key)
    }

    /// Returns the names of all indexed properties.
    #[must_use]
    pub fn property_index_keys(&self) -> Vec<String> {
        self.property_indexes
            .read()
            .keys()
            .map(|k| k.to_string())
            .collect()
    }

    /// Updates property indexes when a property is set.
    pub(super) fn update_property_index_on_set(
        &self,
        node_id: NodeId,
        key: &PropertyKey,
        new_value: &Value,
    ) {
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            // Get old value to remove from index
            if let Some(old_value) = self.node_properties.get(node_id, key) {
                Self::remove_index_entry(index, &HashableValue::new(old_value), node_id);
            }

            // Add new value to index
            let new_hv = HashableValue::new(new_value.clone());
            index
                .entry(new_hv)
                .or_insert_with(FxHashSet::default)
                .insert(node_id);
        }
    }

    /// Inserts `vector` into `index` when its size fits the index. A vector
    /// of another size cannot be indexed, so the node is taken out instead:
    /// writes through the engine reject such a vector before it gets here
    /// (the schema checks), which leaves replayed and internal writes.
    #[cfg(feature = "vector-index")]
    fn insert_into_vector_index(
        index: &VectorIndexKind,
        node_id: NodeId,
        vector: &[f32],
        accessor: &impl crate::index::vector::VectorAccessor,
    ) {
        let dimensions = index.config().dimensions;
        if vector.len() == dimensions {
            index.insert(node_id, vector, accessor);
        } else {
            index.remove(node_id);
        }
    }

    /// The node's current labels.
    #[cfg(feature = "vector-index")]
    fn node_label_names(&self, node_id: NodeId) -> Vec<String> {
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let label_ids = node_labels.get(&node_id);
        #[cfg(feature = "temporal")]
        let label_ids = node_labels.get(&node_id).and_then(|log| log.latest());
        label_ids
            .into_iter()
            .flatten()
            .filter_map(|&label_id| registry.get_name(label_id).map(|name| name.to_string()))
            .collect()
    }

    /// Brings the vector indexes on `key` of the node's labels in line with
    /// the node's current value: a vector is inserted (or replaces the old
    /// one), anything else, or no value, takes the node out.
    #[cfg(feature = "vector-index")]
    pub(super) fn sync_vector_indexes_for_property(&self, node_id: NodeId, key: &str) {
        let indexes: Vec<Arc<VectorIndexKind>> = {
            let all = self.vector_indexes.read();
            if all.is_empty() {
                return;
            }
            self.node_label_names(node_id)
                .iter()
                .filter_map(|label| all.get(&format!("{label}:{key}")).cloned())
                .collect()
        };
        if indexes.is_empty() {
            return;
        }
        let accessor = crate::index::vector::PropertyVectorAccessor::new(self, key);
        let vector = crate::index::vector::VectorAccessor::get_vector(&accessor, node_id);
        for index in indexes {
            match &vector {
                Some(vector) => Self::insert_into_vector_index(&index, node_id, vector, &accessor),
                None => {
                    index.remove(node_id);
                }
            }
        }
    }

    /// Adds a node to the vector indexes on `key` (of its labels) that do not
    /// hold it yet, with its current vector; one that holds it keeps it,
    /// unless the vector has another size than the index, which takes it out:
    /// an index never points at a vector it cannot measure.
    #[cfg(feature = "vector-index")]
    pub(super) fn index_vector_if_missing(&self, node_id: NodeId, key: &PropertyKey) {
        let indexes: Vec<Arc<VectorIndexKind>> = {
            let all = self.vector_indexes.read();
            if all.is_empty() {
                return;
            }
            self.node_label_names(node_id)
                .iter()
                .filter_map(|label| all.get(&format!("{label}:{}", key.as_str())).cloned())
                .collect()
        };
        if indexes.is_empty() {
            return;
        }
        let accessor = crate::index::vector::PropertyVectorAccessor::new(self, key.as_str());
        if let Some(vector) = crate::index::vector::VectorAccessor::get_vector(&accessor, node_id) {
            for index in indexes {
                if !index.contains(node_id) {
                    Self::insert_into_vector_index(&index, node_id, &vector, &accessor);
                } else if vector.len() != index.config().dimensions {
                    index.remove(node_id);
                }
            }
        }
    }

    /// Adds a node to the text and vector indexes of `label`, which it just
    /// got, with its current values.
    pub(super) fn index_node_under_label(&self, node_id: NodeId, label: &str) {
        let prefix = format!("{label}:");
        #[cfg(feature = "text-index")]
        {
            let indexes: Vec<(String, Arc<RwLock<crate::index::text::InvertedIndex>>)> = self
                .text_indexes
                .read()
                .iter()
                .filter_map(|(key, index)| {
                    let property = key.strip_prefix(&prefix)?;
                    Some((property.to_string(), Arc::clone(index)))
                })
                .collect();
            for (property, index) in indexes {
                if let Some(Value::String(text)) = self
                    .node_properties
                    .get(node_id, &PropertyKey::new(property.as_str()))
                {
                    index.write().insert(node_id, &text);
                }
            }
        }
        #[cfg(feature = "vector-index")]
        {
            let indexes: Vec<(String, Arc<VectorIndexKind>)> = self
                .vector_indexes
                .read()
                .iter()
                .filter_map(|(key, index)| {
                    let property = key.strip_prefix(&prefix)?;
                    Some((property.to_string(), Arc::clone(index)))
                })
                .collect();
            for (property, index) in indexes {
                let accessor =
                    crate::index::vector::PropertyVectorAccessor::new(self, property.as_str());
                if let Some(vector) =
                    crate::index::vector::VectorAccessor::get_vector(&accessor, node_id)
                {
                    Self::insert_into_vector_index(&index, node_id, &vector, &accessor);
                }
            }
        }
        #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
        let _ = (node_id, prefix);
    }

    /// Takes a node out of the text and vector indexes of `label`, which it
    /// just lost.
    pub(super) fn unindex_node_under_label(&self, node_id: NodeId, label: &str) {
        let prefix = format!("{label}:");
        #[cfg(feature = "text-index")]
        for (key, index) in self.text_indexes.read().iter() {
            if key.starts_with(&prefix) {
                index.write().remove(node_id);
            }
        }
        #[cfg(feature = "vector-index")]
        {
            let indexes: Vec<Arc<VectorIndexKind>> = self
                .vector_indexes
                .read()
                .iter()
                .filter(|(key, _)| key.starts_with(&prefix))
                .map(|(_, index)| Arc::clone(index))
                .collect();
            for index in indexes {
                index.remove(node_id);
            }
        }
        #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
        let _ = (node_id, prefix);
    }

    /// Removes a deleted node from every vector index, so it can no longer be
    /// returned by a vector search (quantized indexes keep their own copy of
    /// the vector, so a leftover entry would still score like a live node).
    #[cfg(feature = "vector-index")]
    pub(super) fn remove_from_all_vector_indexes(&self, node_id: NodeId) {
        let indexes: Vec<Arc<VectorIndexKind>> =
            self.vector_indexes.read().values().cloned().collect();
        for index in indexes {
            index.remove(node_id);
        }
    }

    /// Re-inserts a node whose delete was rolled back into the vector indexes
    /// of its labels (keys are `label:property`), reading vectors from the
    /// store's properties.
    #[cfg(feature = "vector-index")]
    pub(super) fn reinsert_into_vector_indexes(&self, node_id: NodeId, labels: &[String]) {
        let indexes: Vec<(String, Arc<VectorIndexKind>)> = self
            .vector_indexes
            .read()
            .iter()
            .map(|(key, index)| (key.clone(), Arc::clone(index)))
            .collect();
        for (key, index) in indexes {
            // Match the key against the node's labels rather than splitting
            // on ':', which labels (`` :`a:b` ``) and properties may contain.
            let Some(property) = labels
                .iter()
                .filter_map(|label| key.strip_prefix(label.as_str())?.strip_prefix(':'))
                .min_by_key(|property| property.len())
            else {
                continue;
            };
            let accessor = crate::index::vector::PropertyVectorAccessor::new(self, property);
            if let Some(vector) =
                crate::index::vector::VectorAccessor::get_vector(&accessor, node_id)
            {
                Self::insert_into_vector_index(&index, node_id, &vector, &accessor);
            }
        }
    }

    /// Removes a node from every property index. Called when the node is
    /// deleted, before its properties are dropped: the index is keyed by value.
    /// A rollback of the delete re-adds the entries through `set_node_property`.
    pub(super) fn remove_from_all_property_indexes(&self, node_id: NodeId) {
        let indexes = self.property_indexes.read();
        for (key, index) in indexes.iter() {
            if let Some(value) = self.node_properties.get(node_id, key) {
                Self::remove_index_entry(index, &HashableValue::new(value), node_id);
            }
        }
    }

    /// Removes `node_id` from the set of `value`, dropping the set when empty.
    fn remove_index_entry(
        index: &DashMap<HashableValue, FxHashSet<NodeId>>,
        value: &HashableValue,
        node_id: NodeId,
    ) {
        if let Some(mut nodes) = index.get_mut(value) {
            nodes.remove(&node_id);
        }
        // Checked again under the shard lock: between releasing the bucket and
        // removing it, another writer may have added a node to it.
        index.remove_if(value, |_, nodes| nodes.is_empty());
    }

    /// The current values of the indexed `(node, key)` pairs, taken before a
    /// temporal rollback pops pending versions (see
    /// [`reconcile_property_indexes`](Self::reconcile_property_indexes)).
    #[cfg(feature = "temporal")]
    pub(super) fn indexed_values<'a>(
        &self,
        pairs: impl Iterator<Item = (NodeId, &'a PropertyKey)>,
    ) -> Vec<(NodeId, PropertyKey, Option<Value>)> {
        let indexes = self.property_indexes.read();
        if indexes.is_empty() {
            return Vec::new();
        }
        pairs
            .filter(|(_, key)| indexes.contains_key(*key))
            .map(|(node_id, key)| (node_id, key.clone(), self.node_properties.get(node_id, key)))
            .collect()
    }

    /// Moves index entries from the values in `before` to the values the
    /// nodes have now. Used after a temporal rollback, which restores property
    /// values by popping versions instead of calling `set_node_property`.
    #[cfg(feature = "temporal")]
    pub(super) fn reconcile_property_indexes(
        &self,
        before: Vec<(NodeId, PropertyKey, Option<Value>)>,
    ) {
        if before.is_empty() {
            return;
        }
        let indexes = self.property_indexes.read();
        for (node_id, key, old_value) in before {
            let Some(index) = indexes.get(&key) else {
                continue;
            };
            let new_value = self.node_properties.get(node_id, &key);
            if new_value == old_value {
                continue;
            }
            if let Some(old_value) = old_value {
                Self::remove_index_entry(index, &HashableValue::new(old_value), node_id);
            }
            if let Some(new_value) = new_value {
                index
                    .entry(HashableValue::new(new_value))
                    .or_insert_with(FxHashSet::default)
                    .insert(node_id);
            }
        }
    }

    /// Updates property indexes when a property is removed.
    pub(super) fn update_property_index_on_remove(&self, node_id: NodeId, key: &PropertyKey) {
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            // Get old value to remove from index
            if let Some(old_value) = self.node_properties.get(node_id, key) {
                Self::remove_index_entry(index, &HashableValue::new(old_value), node_id);
            }
        }
    }

    /// Stores a vector index for a label+property pair.
    #[cfg(feature = "vector-index")]
    pub fn add_vector_index(&self, label: &str, property: &str, index: Arc<VectorIndexKind>) {
        let key = format!("{label}:{property}");
        self.vector_indexes.write().insert(key, index);
    }

    /// Retrieves the vector index for a label+property pair.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn get_vector_index(&self, label: &str, property: &str) -> Option<Arc<VectorIndexKind>> {
        let key = format!("{label}:{property}");
        self.vector_indexes.read().get(&key).cloned()
    }

    /// Removes a vector index for a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "vector-index")]
    pub fn remove_vector_index(&self, label: &str, property: &str) -> bool {
        let key = format!("{label}:{property}");
        self.vector_indexes.write().remove(&key).is_some()
    }

    /// Returns all vector index entries as `(key, index)` pairs.
    ///
    /// Keys are in `"label:property"` format.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn vector_index_entries(&self) -> Vec<(String, Arc<VectorIndexKind>)> {
        self.vector_indexes
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Looks up a vector index by its `"label:property"` key.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn get_vector_index_by_key(&self, key: &str) -> Option<Arc<VectorIndexKind>> {
        self.vector_indexes.read().get(key).cloned()
    }

    /// Stores a text index for a label+property pair.
    #[cfg(feature = "text-index")]
    pub fn add_text_index(
        &self,
        label: &str,
        property: &str,
        index: Arc<RwLock<crate::index::text::InvertedIndex>>,
    ) {
        let key = format!("{label}:{property}");
        self.text_indexes.write().insert(key, index);
    }

    /// Retrieves the text index for a label+property pair.
    #[cfg(feature = "text-index")]
    #[must_use]
    pub fn get_text_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<Arc<RwLock<crate::index::text::InvertedIndex>>> {
        let key = format!("{label}:{property}");
        self.text_indexes.read().get(&key).cloned()
    }

    /// Removes a text index for a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "text-index")]
    pub fn remove_text_index(&self, label: &str, property: &str) -> bool {
        let key = format!("{label}:{property}");
        self.text_indexes.write().remove(&key).is_some()
    }

    /// Returns all text index entries as `(key, index)` pairs.
    ///
    /// The key format is `"label:property"`.
    #[cfg(feature = "text-index")]
    pub fn text_index_entries(
        &self,
    ) -> Vec<(String, Arc<RwLock<crate::index::text::InvertedIndex>>)> {
        self.text_indexes
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Updates text indexes when a node property is set.
    ///
    /// If the node has a label with a text index on this property key,
    /// the index is updated with the new value (if it's a string).
    #[cfg(feature = "text-index")]
    pub(super) fn update_text_index_on_set(&self, id: NodeId, key: &str, value: &Value) {
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let label_set = node_labels.get(&id);
        #[cfg(feature = "temporal")]
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let index_key = format!("{label_name}:{key}");
                    if let Some(index) = text_indexes.get(&index_key) {
                        let mut idx = index.write();
                        // Remove old entry first, then insert new if it's a string
                        idx.remove(id);
                        if let Value::String(text) = value {
                            idx.insert(id, text);
                        }
                    }
                }
            }
        }
    }

    /// Updates text indexes when a node property is removed.
    #[cfg(feature = "text-index")]
    pub(super) fn update_text_index_on_remove(&self, id: NodeId, key: &str) {
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        let registry = self.label_registry.read();
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let label_set = node_labels.get(&id);
        #[cfg(feature = "temporal")]
        let label_set = node_labels.get(&id).and_then(|log| log.latest());
        if let Some(label_ids) = label_set {
            for &label_id in label_ids {
                if let Some(label_name) = registry.get_name(label_id) {
                    let index_key = format!("{label_name}:{key}");
                    if let Some(index) = text_indexes.get(&index_key) {
                        index.write().remove(id);
                    }
                }
            }
        }
    }

    /// Removes a node from all text indexes.
    #[cfg(feature = "text-index")]
    pub(super) fn remove_from_all_text_indexes(&self, id: NodeId) {
        let text_indexes = self.text_indexes.read();
        if text_indexes.is_empty() {
            return;
        }
        for (_, index) in text_indexes.iter() {
            index.write().remove(id);
        }
    }
}
