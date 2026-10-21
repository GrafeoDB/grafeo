//! The row-group store's vector indexes: an HNSW index per label and key
//! that holds the newest vector of each node with the label, and the
//! searches at a view (see the parent module).

use std::sync::Arc;

use grafeo_common::types::{ArcStr, NodeId, PropertyKey, ReadView, Value};

use super::super::{Inner, ROWS_PER_GROUP, Read, RowGroupStore};
use super::{IndexSearch, search_key};
use crate::index::ranking::by_ascending_distance;
use crate::index::vector::{
    DistanceMetric, HnswConfig, VectorAccessor, VectorIndexKind, brute_force_knn, compute_distance,
    is_indexable,
};

/// The search beam width of a search through an index, as `LpgStore`'s.
const EF: usize = 64;

/// A vector index of the store, with the label and key it indexes.
pub(super) struct VectorEntry {
    label: ArcStr,
    property: PropertyKey,
    index: Arc<VectorIndexKind>,
}

/// The vectors of one key as the store holds them now, the newest in
/// place: what its vector indexes hold and measure.
struct NewestVectors<'a> {
    inner: &'a Inner,
    key: Option<u32>,
}

impl VectorAccessor for NewestVectors<'_> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        let (group, row) = self.inner.node(id.as_u64())?;
        match group.columns.get(self.key?, row)? {
            Value::Vector(vector) => Some(vector),
            _ => None,
        }
    }
}

impl Inner {
    /// The newest vectors of `property`.
    fn newest_vectors(&self, property: &str) -> NewestVectors<'_> {
        NewestVectors {
            inner: self,
            key: self.keys.get_id(property),
        }
    }

    /// The vector `entry`'s index holds for node `raw`: the node's newest
    /// value of the key, when it is a vector the index can measure and the
    /// node has the label now.
    fn newest_vector(&self, raw: u64, entry: &VectorEntry) -> Option<Arc<[f32]>> {
        let (group, row) = self.node(raw)?;
        let label = self.label_id(&entry.label)?;
        if !group.labels.get(&label).is_some_and(|rows| rows.get(row)) {
            return None;
        }
        match group
            .columns
            .get(self.keys.get_id(entry.property.as_str())?, row)?
        {
            Value::Vector(vector) if is_indexable(&vector, entry.index.config().dimensions) => {
                Some(vector)
            }
            _ => None,
        }
    }

    /// Brings `entry`'s index in line with node `raw`'s newest state: it
    /// holds the node's vector, taken again when `value_changed` or when it
    /// lacks it, or it does not hold the node.
    fn sync_vector_entry(&self, raw: u64, entry: &VectorEntry, value_changed: bool) {
        let id = NodeId::new(raw);
        let vectors = self.newest_vectors(entry.property.as_str());
        match self.newest_vector(raw, entry) {
            Some(vector) => {
                if value_changed || !entry.index.contains(id) {
                    entry.index.insert(id, &vector, &vectors);
                }
            }
            None => {
                if entry.index.contains(id) {
                    entry.index.remove(id, &vectors);
                }
            }
        }
    }

    /// Brings the vector indexes on key `key` (`None`: any) and of label
    /// `label` (`None`: any) in line with node `raw` (see
    /// [`sync_vector_entry`](Self::sync_vector_entry)).
    pub(in crate::graph::rowgroup) fn sync_vectors(
        &self,
        raw: u64,
        key: Option<&str>,
        label: Option<&str>,
        value_changed: bool,
    ) {
        for entry in self.indexes.vector.values() {
            if key.is_some_and(|key| key != entry.property.as_str())
                || label.is_some_and(|label| label != entry.label.as_str())
            {
                continue;
            }
            self.sync_vector_entry(raw, entry, value_changed);
        }
    }

    /// Takes node `id` out of every vector index, mending the links of the
    /// nodes around it while their vectors are in place.
    pub(in crate::graph::rowgroup) fn unindex_vectors(&self, id: NodeId) {
        for entry in self.indexes.vector.values() {
            if entry.index.contains(id) {
                entry
                    .index
                    .remove(id, &self.newest_vectors(entry.property.as_str()));
            }
        }
    }

    /// Brings a vector index the store takes in line with the newest state
    /// of the nodes whose versions readers may tell apart (see
    /// `reconcile_text`).
    fn reconcile_vectors(&self, entry: &VectorEntry) {
        let key = self.keys.get_id(entry.property.as_str());
        // Only the listed row groups hold stamps or chains (`collect`).
        for (number, group) in self.listed_node_groups() {
            let base = number * ROWS_PER_GROUP;
            for row in Self::unsettled_rows(group, key) {
                let value_changed =
                    key.is_some_and(|key| group.columns.kept(key, row).next().is_some());
                self.sync_vector_entry(base + row as u64, entry, value_changed);
            }
        }
    }

    /// Node `raw`'s vector of key `key` as `read` sees the node, when the
    /// node has label id `label` (`None`: any label) at the view.
    fn vector_at(&self, raw: u64, label: Option<u32>, key: u32, read: Read) -> Option<Arc<[f32]>> {
        let (group, row) = self.node(raw)?;
        if !group.version(row).visible(read)
            || label.is_some_and(|label| !group.has_label_at(row, label, read))
        {
            return None;
        }
        match group.columns.get_at(key, row, read)? {
            Value::Vector(vector) => Some(vector),
            _ => None,
        }
    }

    /// The vectors of `property` of the nodes `read` sees with `label`
    /// (every node it sees without one), for a scan.
    fn scanned_vectors(
        &self,
        label: Option<&str>,
        property: &str,
        read: Read,
    ) -> Vec<(NodeId, Arc<[f32]>)> {
        let Some(key) = self.keys.get_id(property) else {
            return Vec::new();
        };
        let label = match label {
            Some(name) => match self.label_id(name) {
                Some(label) => Some(label),
                None => return Vec::new(),
            },
            None => None,
        };
        let mut found = Vec::new();
        for (number, group) in &self.nodes {
            let base = number * ROWS_PER_GROUP;
            for (row, _) in group.versions.rows() {
                let raw = base + row as u64;
                if let Some(vector) = self.vector_at(raw, label, key, read) {
                    found.push((NodeId::new(raw), vector));
                }
            }
        }
        found
    }

    /// The `k` nodes `read` sees nearest to `query` by `metric`, nearest
    /// first: through the vector index on `label` and `property` when its
    /// metric is `metric`, by a scan of the vectors at the view otherwise
    /// (as `LpgStore` searches).
    pub(in crate::graph::rowgroup) fn vector_search_at(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        read: Read,
    ) -> Vec<(NodeId, f64)> {
        if let Some(name) = label
            && let Some(entry) = self.indexes.vector.get(&search_key(name, property))
            && entry.index.config().metric == metric
        {
            let search = IndexSearch {
                k,
                ef: Some(EF),
                allowed: None,
            };
            return self
                .indexed_nearest(entry, query, search, read)
                .into_iter()
                .map(|(id, distance)| (id, f64::from(distance)))
                .collect();
        }
        let vectors = self.scanned_vectors(label, property, read);
        brute_force_knn(
            vectors.iter().map(|(id, vector)| (*id, vector.as_ref())),
            query,
            k,
            metric,
        )
        .into_iter()
        .map(|(id, distance)| (id, f64::from(distance)))
        .collect()
    }

    /// The `search.k` nodes `read` sees nearest to `query`, through
    /// `entry`'s index with `search`'s beam width (the index's own when
    /// `None`), among `search`'s allowed nodes only when given: the index's
    /// nearest nodes among those every reader sees alike, fetched again with
    /// twice as many until `k` are left after the ones the view does not see
    /// (or the index has no more), and the vectors at the view of the nodes
    /// whose value or labels a chain holds, measured here by the index's
    /// metric.
    fn indexed_nearest(
        &self,
        entry: &VectorEntry,
        query: &[f32],
        search: IndexSearch<'_>,
        read: Read,
    ) -> Vec<(NodeId, f32)> {
        let IndexSearch { k, ef, allowed } = search;
        let key = self.keys.get_id(entry.property.as_str());
        let changed = self.changed_rows(key);
        let vectors = self.newest_vectors(entry.property.as_str());
        let size = entry.index.len();
        let mut fetch = k;
        let mut found = loop {
            let nearest = match (allowed, ef) {
                (Some(allowed), Some(ef)) => entry
                    .index
                    .search_with_ef_and_filter(query, fetch, ef, allowed, &vectors),
                (Some(allowed), None) => entry
                    .index
                    .search_with_filter(query, fetch, allowed, &vectors),
                (None, Some(ef)) => entry.index.search_with_ef(query, fetch, ef, &vectors),
                (None, None) => entry.index.search(query, fetch, &vectors),
            };
            let complete = nearest.len() < fetch || fetch >= size;
            let kept: Vec<(NodeId, f32)> = nearest
                .into_iter()
                .filter(|(id, _)| {
                    !changed.contains(&id.as_u64()) && self.node_visible(id.as_u64(), read)
                })
                .collect();
            if kept.len() >= k || complete {
                break kept;
            }
            fetch = fetch.saturating_mul(2).min(size);
        };
        if changed.is_empty() {
            found.truncate(k);
            return found;
        }
        let metric = entry.index.config().metric;
        let label = self.label_id(&entry.label);
        if let (Some(label), Some(key)) = (label, key) {
            for raw in changed {
                if allowed.is_some_and(|allowed| !allowed.contains(&NodeId::new(raw))) {
                    continue;
                }
                if let Some(vector) = self.vector_at(raw, Some(label), key, read)
                    && vector.len() == query.len()
                {
                    found.push((NodeId::new(raw), compute_distance(query, &vector, metric)));
                }
            }
        }
        found.sort_by(by_ascending_distance);
        found.truncate(k);
        found
    }

    /// The nodes `read` sees within `threshold` of `query` by `metric`,
    /// nearest first, by a scan of the vectors at the view (an HNSW index
    /// has no threshold search; `LpgStore` scans too).
    pub(in crate::graph::rowgroup) fn vector_search_within_at(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
        read: Read,
    ) -> Vec<(NodeId, f64)> {
        let mut found: Vec<(NodeId, f64)> = self
            .scanned_vectors(label, property, read)
            .into_iter()
            .filter(|(_, vector)| vector.len() == query.len())
            .filter_map(|(id, vector)| {
                let distance = f64::from(compute_distance(query, &vector, metric));
                (distance <= threshold).then_some((id, distance))
            })
            .collect();
        found.sort_by(by_ascending_distance);
        found
    }

    /// The configuration of the vector index on `label` and `property`.
    pub(in crate::graph::rowgroup) fn vector_index_config(
        &self,
        label: &str,
        property: &str,
    ) -> Option<HnswConfig> {
        self.indexes
            .vector
            .get(&search_key(label, property))
            .map(|entry| entry.index.config().clone())
    }
}

impl RowGroupStore {
    /// Stores a vector index for a label and a property, in place of one on
    /// the same pair (see `LpgStore::add_vector_index`). The store keeps it
    /// from then on: the writes give it each node's newest vector. It first
    /// brings the index in line with the newest state of the nodes whose
    /// versions readers may tell apart (see
    /// [`add_text_index`](Self::add_text_index)).
    pub fn add_vector_index(&self, label: &str, property: &str, index: Arc<VectorIndexKind>) {
        let mut inner = self.inner.write();
        let entry = VectorEntry {
            label: ArcStr::from(label),
            property: PropertyKey::new(property),
            index,
        };
        inner.reconcile_vectors(&entry);
        inner
            .indexes
            .vector
            .insert(search_key(label, property), entry);
    }

    /// The nodes `view` sees nearest to `query`, nearest first, through the
    /// vector index on `label` and `property`, by the index's metric, with
    /// `search`'s `k`, beam width and allowed nodes: the database's
    /// `vector_search`. The index holds each node's newest vector; the nodes
    /// whose vector or labels a version some reader may still need holds are
    /// measured at the view instead, so the answer is the view's. `None`
    /// without the index.
    ///
    /// The store's lock is held for the search, and no other lock is taken
    /// with it but the index's own, in the order the writes take them.
    #[must_use]
    pub fn vector_index_search_at(
        &self,
        label: &str,
        property: &str,
        query: &[f32],
        search: IndexSearch<'_>,
        view: ReadView,
    ) -> Option<Vec<(NodeId, f32)>> {
        let inner = self.inner.read();
        let entry = inner.indexes.vector.get(&search_key(label, property))?;
        Some(inner.indexed_nearest(entry, query, search, Read::view(view)))
    }

    /// [`vector_index_search_at`](Self::vector_index_search_at) for each of
    /// `queries`, in their order, under one hold of the store's lock (in
    /// parallel with the `parallel` feature). `None` without the index.
    #[must_use]
    pub fn vector_index_batch_search_at(
        &self,
        label: &str,
        property: &str,
        queries: &[Vec<f32>],
        search: IndexSearch<'_>,
        view: ReadView,
    ) -> Option<Vec<Vec<(NodeId, f32)>>> {
        let inner = self.inner.read();
        let entry = inner.indexes.vector.get(&search_key(label, property))?;
        let read = Read::view(view);
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            Some(
                queries
                    .par_iter()
                    .map(|query| inner.indexed_nearest(entry, query, search, read))
                    .collect(),
            )
        }
        #[cfg(not(feature = "parallel"))]
        Some(
            queries
                .iter()
                .map(|query| inner.indexed_nearest(entry, query, search, read))
                .collect(),
        )
    }

    /// Node `id`'s vector of `property` as `view` sees it (any label), for
    /// a reranking of search results at the view.
    #[must_use]
    pub fn vector_at(&self, id: NodeId, property: &str, view: ReadView) -> Option<Arc<[f32]>> {
        let inner = self.inner.read();
        let key = inner.keys.get_id(property)?;
        inner.vector_at(id.as_u64(), None, key, Read::view(view))
    }

    /// Retrieves the vector index for a label and a property.
    #[must_use]
    pub fn get_vector_index(&self, label: &str, property: &str) -> Option<Arc<VectorIndexKind>> {
        self.get_vector_index_by_key(&search_key(label, property))
    }

    /// Looks up a vector index by its `"label:property"` key.
    #[must_use]
    pub fn get_vector_index_by_key(&self, key: &str) -> Option<Arc<VectorIndexKind>> {
        self.inner
            .read()
            .indexes
            .vector
            .get(key)
            .map(|entry| Arc::clone(&entry.index))
    }

    /// Removes the vector index for a label and a property. Returns `true`
    /// if the index existed and was removed.
    pub fn remove_vector_index(&self, label: &str, property: &str) -> bool {
        self.inner
            .write()
            .indexes
            .vector
            .remove(&search_key(label, property))
            .is_some()
    }

    /// Returns all vector index entries as `(key, index)` pairs, keys in
    /// `"label:property"` format.
    #[must_use]
    pub fn vector_index_entries(&self) -> Vec<(String, Arc<VectorIndexKind>)> {
        self.inner
            .read()
            .indexes
            .vector
            .iter()
            .map(|(key, entry)| (key.clone(), Arc::clone(&entry.index)))
            .collect()
    }
}
