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
                let old_hv = HashableValue::new(old_value);
                if let Some(mut nodes) = index.get_mut(&old_hv) {
                    nodes.remove(&node_id);
                    if nodes.is_empty() {
                        drop(nodes);
                        index.remove(&old_hv);
                    }
                }
            }

            // Add new value to index
            let new_hv = HashableValue::new(new_value.clone());
            index
                .entry(new_hv)
                .or_insert_with(FxHashSet::default)
                .insert(node_id);
        }
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
            let Some((label, property)) = key.split_once(':') else {
                continue;
            };
            if !labels.iter().any(|l| l == label) {
                continue;
            }
            let accessor = crate::index::vector::PropertyVectorAccessor::new(self, property);
            if let Some(vector) =
                crate::index::vector::VectorAccessor::get_vector(&accessor, node_id)
            {
                index.insert(node_id, &vector, &accessor);
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
            if nodes.is_empty() {
                drop(nodes);
                index.remove(value);
            }
        }
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
                let old_hv = HashableValue::new(old_value);
                if let Some(mut nodes) = index.get_mut(&old_hv) {
                    nodes.remove(&node_id);
                    if nodes.is_empty() {
                        drop(nodes);
                        index.remove(&old_hv);
                    }
                }
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
