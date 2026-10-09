//! Index management methods for [`LpgStore`].

use super::LpgStore;
use super::property_index::PropertyIndex;
use grafeo_common::types::{HashableValue, NodeId, PropertyKey, Value};
#[cfg(feature = "text-index")]
use parking_lot::RwLock;
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use std::sync::Arc;

#[cfg(feature = "vector-index")]
use crate::index::vector::VectorIndexKind;

/// Lends the vectors of the compacted base a store is the overlay of (see
/// [`LpgStore::set_base_vectors`]).
#[cfg(all(feature = "compact-store", feature = "vector-index"))]
pub(crate) trait BaseVectors: Send + Sync {
    /// Calls `f` with the vector in property `key` of base node `id`, and
    /// returns whether there was one.
    fn with_base_vector(&self, id: NodeId, key: &PropertyKey, f: &mut dyn FnMut(&[f32])) -> bool;
}

/// The vectors the vector indexes of a store read: the store's own, and,
/// when the store is the overlay of a compacted base, the base's for a node
/// the store does not hold. An index of an overlay holds base and overlay
/// nodes in one HNSW graph, so the upkeep links a new vector to its nearest
/// neighbors among both, and a search measures both.
///
/// A copy of a base node takes all of the node's values, so the store's
/// values are the node's own from then on: a vector the copy removed, or
/// replaced by a value that is not a vector, is gone, and the base's old one
/// is never lent for it. The base lends only for the nodes the store holds
/// no record of.
///
/// Lock order: the check that the store holds a node takes the store's node
/// lock (level 1) for a read, inside whatever lock the index holds while it
/// reads a vector. No caller holds the node lock while it calls into a vector
/// index (`LpgStore::adopt_node` syncs the vector indexes after it releases
/// it).
#[cfg(feature = "vector-index")]
pub(crate) struct IndexVectors<'a> {
    store: &'a LpgStore,
    key: PropertyKey,
    #[cfg(feature = "compact-store")]
    base: Option<Arc<dyn BaseVectors>>,
}

#[cfg(feature = "vector-index")]
impl crate::index::vector::VectorAccessor for IndexVectors<'_> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        if let Some(Value::Vector(vector)) = self.store.node_properties.get(id, &self.key) {
            return Some(vector);
        }
        let mut copied = None;
        self.with_base_vector(id, &mut |vector| copied = Some(Arc::from(vector)));
        copied
    }

    fn with_vector(&self, id: NodeId, f: &mut dyn FnMut(&[f32])) -> bool {
        self.store
            .node_properties
            .with_vector(id, &self.key, &mut *f)
            .is_some()
            || self.with_base_vector(id, f)
    }
}

#[cfg(feature = "vector-index")]
impl IndexVectors<'_> {
    /// Lends the base's vector of `id`, when the store is an overlay that
    /// holds no record of `id`: a copy of a base node, whatever its values,
    /// never lends the base's.
    fn with_base_vector(&self, id: NodeId, f: &mut dyn FnMut(&[f32])) -> bool {
        #[cfg(feature = "compact-store")]
        {
            self.base.as_ref().is_some_and(|base| {
                !self.store.holds_node_record(id) && base.with_base_vector(id, &self.key, f)
            })
        }
        #[cfg(not(feature = "compact-store"))]
        {
            let _ = (id, f);
            false
        }
    }
}

impl LpgStore {
    /// The vectors of property `key` that this store's vector indexes read
    /// (see [`IndexVectors`]).
    #[cfg(feature = "vector-index")]
    pub(crate) fn index_vectors(&self, key: impl Into<PropertyKey>) -> IndexVectors<'_> {
        IndexVectors {
            store: self,
            key: key.into(),
            #[cfg(feature = "compact-store")]
            base: self.base_vectors.read().clone(),
        }
    }

    /// Makes this store the overlay of a compacted base whose vectors
    /// `base` lends: its vector indexes read them for the nodes it holds no
    /// vector for (see [`IndexVectors`]). A [`successor`](Self::successor)
    /// reads them too.
    #[cfg(all(feature = "compact-store", feature = "vector-index"))]
    pub(crate) fn set_base_vectors(&self, base: Arc<dyn BaseVectors>) {
        *self.base_vectors.write() = Some(base);
    }

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
        if self.property_indexes.read().contains_key(&key) {
            return; // Already indexed
        }

        // The node lock first, then the index lock, as the store's lock order
        // has them: an adoption holds the node lock while it adds its copy to
        // the property indexes, so a build holding the index lock while it
        // waits for the node lock would wait on it forever. Under both, no
        // node is created, adopted or deleted during the scan.
        self.with_node_ids_held(|node_ids| {
            let mut indexes = self.property_indexes.write();
            if indexes.contains_key(&key) {
                return; // Indexed since the check above
            }

            // Create the index and populate it with existing data
            let index = PropertyIndex::default();
            for node_id in node_ids {
                if let Some(value) = self.node_properties.get(node_id, &key) {
                    index.insert(HashableValue::new(value), node_id);
                }
            }

            indexes.insert(key, index);
        });
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
                index.remove(&HashableValue::new(old_value), node_id);
            }

            // Add new value to index
            index.insert(HashableValue::new(new_value.clone()), node_id);
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
        let accessor = self.index_vectors(key);
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
        let accessor = self.index_vectors(key.clone());
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
                let accessor = self.index_vectors(property.as_str());
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
            let accessor = self.index_vectors(property);
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
                index.remove(&HashableValue::new(value), node_id);
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
                index.remove(&HashableValue::new(old_value), node_id);
            }
            if let Some(new_value) = new_value {
                index.insert(HashableValue::new(new_value), node_id);
            }
        }
    }

    /// Updates property indexes when a property whose value was `old_value`
    /// is removed.
    pub(super) fn update_property_index_on_remove(
        &self,
        node_id: NodeId,
        key: &PropertyKey,
        old_value: &Value,
    ) {
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            index.remove(&HashableValue::new(old_value.clone()), node_id);
        }
    }

    /// The nodes whose `property` may be equal to `key` under `=` (see
    /// [`GraphStore::find_nodes_maybe_equal`](crate::graph::GraphStore::find_nodes_maybe_equal)),
    /// found through the property's index: every node `=` finds equal, and
    /// maybe others, for a filter to decide. `None` when the property has no
    /// index.
    #[must_use]
    pub fn find_nodes_maybe_equal(&self, property: &str, key: &Value) -> Option<Vec<NodeId>> {
        let indexes = self.property_indexes.read();
        indexes
            .get(&PropertyKey::new(property))
            .map(|index| index.nodes_maybe_equal(key))
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

/// The searches of the text and vector indexes, which leave out the nodes
/// that are gone.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl LpgStore {
    /// The first `k` hits of a search of one of this store's text or vector
    /// indexes that are still in the graph, best first.
    ///
    /// The indexes of a compacted database's overlay hold the nodes of the
    /// compacted base (see [`successor`](Self::successor)), and a delete of
    /// one leaves its entries there until the next merge of the overlay
    /// drops them: the hits of the base nodes whose delete is committed are
    /// dropped (see [`retain_live_index_hits`](Self::retain_live_index_hits)).
    ///
    /// `search` returns the hits of the index, best first, for the number
    /// of them it is given. It is called with `k`, and again with twice as
    /// many while it returned as many as it was asked for and hits were
    /// dropped, so the result has `k` hits whenever the index finds `k`
    /// nodes that are still there. Every search of these indexes with a
    /// number of hits goes through here: this store's, the layered store's
    /// and the database's.
    pub fn live_index_hits<S>(
        &self,
        k: usize,
        mut search: impl FnMut(usize) -> Vec<(NodeId, S)>,
    ) -> Vec<(NodeId, S)> {
        let mut fetch = k;
        loop {
            let mut hits = search(fetch);
            let found = hits.len();
            self.retain_live_index_hits(&mut hits);
            // At most `found` were dropped, so twice the hits replace them.
            if hits.len() >= k || found < fetch {
                hits.truncate(k);
                return hits;
            }
            fetch = fetch.saturating_mul(2);
        }
    }

    /// Drops from `hits`, of a search of one of this store's text or vector
    /// indexes, the nodes that are gone: the base nodes of the compacted
    /// base this store is the overlay of whose delete is committed. A delete
    /// that is not committed yet hides nothing, as for every current read of
    /// the layered store. A search with a threshold uses this as it is; a
    /// search for the best `k` uses [`live_index_hits`](Self::live_index_hits).
    pub fn retain_live_index_hits<S>(&self, hits: &mut Vec<(NodeId, S)>) {
        #[cfg(feature = "compact-store")]
        {
            let tombstones = self.base_tombstones();
            hits.retain(|(id, _)| !tombstones.node_deleted(*id));
        }
        #[cfg(not(feature = "compact-store"))]
        let _ = hits;
    }

    /// Whether node `id`, which a text index of this store may hold, is gone
    /// (see [`retain_live_index_hits`](Self::retain_live_index_hits)).
    #[cfg(feature = "text-index")]
    pub(crate) fn is_gone_from_indexes(&self, id: NodeId) -> bool {
        #[cfg(feature = "compact-store")]
        {
            self.base_tombstones().node_deleted(id)
        }
        #[cfg(not(feature = "compact-store"))]
        {
            let _ = id;
            false
        }
    }
}
