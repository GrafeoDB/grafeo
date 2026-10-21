//! The row-group store's indexes (#412 S7, H1d): a property index per key,
//! and text and vector indexes per label and key, each graph's its own, kept
//! by the writes as they apply (decision Q4 A).
//!
//! A property index holds an entry for each value of its key some reader
//! at the horizon or later may see: a write adds the value it writes and
//! keeps the one it replaced while a version holds it, so the index holds
//! old and pending values side by side. An entry goes once no version holds
//! its value: at the write that replaces a value nothing keeps (replay, a
//! transaction's second write), at undo, and at `collect`, which takes out
//! what only versions committed by the horizon held and what a delete
//! committed by then took out of every view. A lookup at a view reads each
//! candidate's value at the view, as a scan does, so it finds what the
//! scan finds; a lookup without a view lists every candidate that exists,
//! for a reader to check at its view, as a label scan without one does.
//!
//! A text or vector index holds one document or vector per node, so it holds
//! the newest state: the text or vector of each node with the label, in
//! place, deleted nodes until `collect` takes them out (`text`, `vector`). A
//! search at a view takes the index's answer for the nodes every reader sees
//! alike and reads the nodes whose value or labels a chain holds (what
//! changed since the horizon) at the view, scoring them itself: membership
//! is exact, and a BM25 score is computed with the statistics of now
//! (decision Q5 A).

#[cfg(feature = "text-index")]
mod text;
#[cfg(feature = "vector-index")]
mod vector;

#[cfg(feature = "text-index")]
pub(super) use text::TextLimit;

use std::fmt;

use grafeo_common::types::{HashableValue, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;

use super::{Inner, NodeGroup, PENDING, ROWS_PER_GROUP, Read, RowGroupStore};
use crate::graph::lpg::{MaybeEqual, PropertyIndex};

/// A store's indexes.
#[derive(Default)]
pub(super) struct Indexes {
    /// The property indexes, by key.
    property: FxHashMap<PropertyKey, PropertyIndex>,
    /// The text indexes, by `label:property`.
    #[cfg(feature = "text-index")]
    text: FxHashMap<String, text::TextEntry>,
    /// The vector indexes, by `label:property`.
    #[cfg(feature = "vector-index")]
    vector: FxHashMap<String, vector::VectorEntry>,
}

impl fmt::Debug for Indexes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("Indexes");
        debug.field("property", &self.property.keys().collect::<Vec<_>>());
        #[cfg(feature = "text-index")]
        debug.field("text", &self.text.keys().collect::<Vec<_>>());
        #[cfg(feature = "vector-index")]
        debug.field("vector", &self.vector.keys().collect::<Vec<_>>());
        debug.finish()
    }
}

impl Indexes {
    /// Whether the store has no index: a write then does no index work.
    pub(super) fn is_empty(&self) -> bool {
        self.property.is_empty() && !self.searches()
    }

    /// Whether the store has a text or a vector index.
    fn searches(&self) -> bool {
        #[cfg(feature = "text-index")]
        if !self.text.is_empty() {
            return true;
        }
        #[cfg(feature = "vector-index")]
        if !self.vector.is_empty() {
            return true;
        }
        false
    }
}

/// How a search through a text or vector index of the store picks its
/// results: the best `k`, with the beam width `ef` of a vector index (the
/// index's own when `None`; a text search has none), among the nodes of
/// `allowed` only when given (the nodes an engine's property filters
/// leave).
#[cfg(any(feature = "text-index", feature = "vector-index"))]
#[derive(Debug, Clone, Copy)]
pub struct IndexSearch<'a> {
    /// How many results, at most.
    pub k: usize,
    /// The beam width of a vector index search.
    pub ef: Option<usize>,
    /// The nodes to search among; `None` searches every node.
    pub allowed: Option<&'a std::collections::HashSet<NodeId>>,
}

/// The key of a text or vector index, `label:property`, as `LpgStore` keys
/// them (a label holding a ':' may give two indexes one key there too).
#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn search_key(label: &str, property: &str) -> String {
    format!("{label}:{property}")
}

/// Whether `value` is the value `wanted` holds, as the property index tells
/// values apart.
fn same(value: &Value, wanted: &HashableValue) -> bool {
    HashableValue::new(value.clone()) == *wanted
}

impl Inner {
    /// Whether a version of row `row`'s value of key `key`, the newest or
    /// one a chain keeps, holds `value`.
    fn holds(group: &NodeGroup, row: usize, key: u32, value: &HashableValue) -> bool {
        group
            .columns
            .get(key, row)
            .is_some_and(|newest| same(&newest, value))
            || group
                .columns
                .kept(key, row)
                .any(|(_, kept)| kept.as_ref().is_some_and(|kept| same(kept, value)))
    }

    /// Brings the indexes in line with a write of node `raw`'s value of key
    /// `key`, also an undo, which replaced `replaced` (`None`: no value):
    /// the property index of the key gets the newest value and loses the
    /// replaced one when no version holds it any more, the text and vector
    /// indexes on the key take the newest value.
    pub(super) fn index_value(&self, raw: u64, key: u32, replaced: Option<&Value>) {
        let (Some(name), Some((group, row))) = (self.keys.get_name(key), self.node(raw)) else {
            return;
        };
        let id = NodeId::new(raw);
        if let Some(index) = self.indexes.property.get(name.as_str()) {
            if let Some(newest) = group.columns.get(key, row) {
                index.insert(HashableValue::new(newest), id);
            }
            if let Some(replaced) = replaced {
                let replaced = HashableValue::new(replaced.clone());
                if !Self::holds(group, row, key, &replaced) {
                    index.remove(&replaced, id);
                }
            }
        }
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        if self.indexes.searches() {
            self.sync_searches(raw, Some(name.as_str()), None, true);
        }
    }

    /// Brings the text and vector indexes in line with a change of node
    /// `raw`'s labels, a write of label `label` (`None`: any, an undo): an
    /// index of a label the node has now holds it, one of a label it lost
    /// does not.
    pub(super) fn index_labels(&self, raw: u64, label: Option<&str>) {
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        if self.indexes.searches() {
            self.sync_searches(raw, None, label, false);
        }
        #[cfg(not(any(feature = "text-index", feature = "vector-index")))]
        let _ = (raw, label);
    }

    /// Indexes node `raw`, which a write just created, with its values and
    /// labels.
    pub(super) fn index_created(&self, raw: u64) {
        let Some((group, row)) = self.node(raw) else {
            return;
        };
        let id = NodeId::new(raw);
        for (name, index) in &self.indexes.property {
            if let Some(value) = self
                .keys
                .get_id(name.as_str())
                .and_then(|key| group.columns.get(key, row))
            {
                index.insert(HashableValue::new(value), id);
            }
        }
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        if self.indexes.searches() {
            self.sync_searches(raw, None, None, true);
        }
    }

    /// Takes node `raw` out of every index: each value it holds or a chain
    /// kept of it, its document and its vector. For a create undone, before
    /// the row goes, and for a node no reader at the horizon sees.
    pub(super) fn unindex(&self, raw: u64) {
        let Some((group, row)) = self.node(raw) else {
            return;
        };
        let id = NodeId::new(raw);
        for (name, index) in &self.indexes.property {
            let Some(key) = self.keys.get_id(name.as_str()) else {
                continue;
            };
            let kept = group
                .columns
                .kept(key, row)
                .filter_map(|(_, kept)| kept.clone());
            for value in group.columns.get(key, row).into_iter().chain(kept) {
                index.remove(&HashableValue::new(value), id);
            }
        }
        #[cfg(feature = "text-index")]
        self.unindex_text(id);
        #[cfg(feature = "vector-index")]
        self.unindex_vectors(id);
    }

    /// Brings the text and vector indexes in line with node `raw`: those on
    /// key `key` (`None`: any) and of label `label` (`None`: any).
    /// `value_changed` makes an index that holds the node take
    /// its newest value again.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    fn sync_searches(&self, raw: u64, key: Option<&str>, label: Option<&str>, value_changed: bool) {
        #[cfg(feature = "text-index")]
        self.sync_text(raw, key, label, value_changed);
        #[cfg(feature = "vector-index")]
        self.sync_vectors(raw, key, label, value_changed);
    }

    /// Takes out what no reader at `horizon` or later can find any more,
    /// before `collect` drops it from the rows: every entry of a node a
    /// delete committed by the horizon took out of every such view, and the
    /// entries of values only versions committed by the horizon held. It
    /// looks at the node row groups `groups` alone: the listed ones, which
    /// hold every stamp and chain (`collect`), or every one.
    pub(super) fn unindex_collected(&self, horizon: u64, groups: &[u64]) {
        if self.indexes.is_empty() {
            return;
        }
        let keys: Vec<(u32, &PropertyIndex)> = self
            .indexes
            .property
            .iter()
            .filter_map(|(name, index)| Some((self.keys.get_id(name.as_str())?, index)))
            .collect();
        for (number, group) in groups
            .iter()
            .filter_map(|number| Some((number, self.nodes.get(number)?)))
        {
            let base = number * ROWS_PER_GROUP;
            for row in group.versions.deleted_by(horizon) {
                self.unindex(base + row as u64);
            }
            for (key, index) in &keys {
                let Some(rows) = group.columns.chained_rows(*key) else {
                    continue;
                };
                for row in rows.rows() {
                    let (mut dropped, mut kept) = (Vec::new(), Vec::new());
                    for (stamp, value) in group.columns.kept(*key, row) {
                        let Some(value) = value else {
                            continue;
                        };
                        let value = HashableValue::new(value.clone());
                        if stamp & PENDING == 0 && stamp <= horizon {
                            dropped.push(value);
                        } else {
                            kept.push(value);
                        }
                    }
                    let newest = group.columns.get(*key, row).map(HashableValue::new);
                    let id = NodeId::new(base + row as u64);
                    for value in dropped {
                        if newest.as_ref() != Some(&value) && !kept.contains(&value) {
                            index.remove(&value, id);
                        }
                    }
                }
            }
        }
    }

    /// The nodes `read` sees whose `property` is `value`, through the
    /// property's index: the candidates the index holds for the value, each
    /// read again at the view. `None` without an index on `property`.
    pub(super) fn indexed_equal_at(
        &self,
        property: &str,
        value: &Value,
        read: Read,
    ) -> Option<Vec<NodeId>> {
        let index = self.indexes.property.get(property)?;
        let wanted = HashableValue::new(value.clone());
        let mut found: Vec<NodeId> = index
            .nodes(&wanted)
            .into_iter()
            .filter(|id| {
                self.node_value_at(id.as_u64(), property, read)
                    .is_some_and(|seen| same(&seen, &wanted))
            })
            .collect();
        found.sort_unstable();
        Some(found)
    }

    /// The nodes `read` sees that meet every equality of `conditions`,
    /// starting from the indexed condition with the fewest candidates.
    /// `None` when no condition's key has an index.
    pub(super) fn indexed_matching_at(
        &self,
        conditions: &[(&str, Value)],
        read: Read,
    ) -> Option<Vec<NodeId>> {
        let candidates = conditions
            .iter()
            .filter_map(|(key, value)| {
                let index = self.indexes.property.get(*key)?;
                Some(index.nodes(&HashableValue::new(value.clone())))
            })
            .min_by_key(Vec::len)?;
        let mut found: Vec<NodeId> = candidates
            .into_iter()
            .filter(|id| {
                conditions.iter().all(|(key, value)| {
                    self.node_value_at(id.as_u64(), key, read).as_ref() == Some(value)
                })
            })
            .collect();
        found.sort_unstable();
        Some(found)
    }

    /// The nodes whose `property` may be equal to `key` under `=`, through
    /// the property's index, in any version: every candidate whose row
    /// exists, ascending, for a reader to check at its view. `None` without
    /// an index on `property`.
    pub(super) fn maybe_equal(&self, property: &str, key: &Value) -> Option<Vec<NodeId>> {
        let index = self.indexes.property.get(property)?;
        let mut found = index.nodes_maybe_equal(key);
        found.retain(|id| self.node(id.as_u64()).is_some());
        Some(found)
    }

    /// [`maybe_equal`](Self::maybe_equal) at `read`: the candidates whose
    /// value as `read` sees it the index finds for `key`, ascending. `None`
    /// without an index on `property`.
    pub(super) fn maybe_equal_at(
        &self,
        property: &str,
        key: &Value,
        read: Read,
    ) -> Option<Vec<NodeId>> {
        let index = self.indexes.property.get(property)?;
        let Some(test) = MaybeEqual::new(key) else {
            return Some(Vec::new()); // NULL equals nothing
        };
        let mut found = index.nodes_maybe_equal(key);
        found.retain(|id| {
            self.node_value_at(id.as_u64(), property, read)
                .is_some_and(|seen| test.finds(&seen))
        });
        Some(found)
    }

    /// The nodes whose value of key `key` (`None`: no key, so none) or
    /// labels a chain holds, by raw id: those a reader may see another text
    /// or vector of than the index holds. Only the listed row groups can
    /// hold a chain (`collect`).
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    fn changed_rows(&self, key: Option<u32>) -> grafeo_common::utils::hash::FxHashSet<u64> {
        let mut changed = grafeo_common::utils::hash::FxHashSet::default();
        for (number, group) in self.listed_node_groups() {
            let base = number * ROWS_PER_GROUP;
            let values = key.and_then(|key| group.columns.chained_rows(key));
            for rows in values.iter().chain(group.label_updates.rows().as_ref()) {
                changed.extend(rows.rows().map(|row| base + row as u64));
            }
        }
        changed
    }

    /// Whether a chain holds node row `row`'s value of key `key` or its
    /// labels (see [`changed_rows`](Self::changed_rows)).
    #[cfg(feature = "text-index")]
    fn changed(group: &NodeGroup, row: usize, key: Option<u32>) -> bool {
        key.is_some_and(|key| group.columns.kept(key, row).next().is_some())
            || group.label_updates.kept(row).next().is_some()
    }

    /// The rows of node group `group` whose state some readers may tell
    /// apart: rows of vectors that hold stamps, and rows a chain of key `key`
    /// or of the labels holds, ascending.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    fn unsettled_rows(group: &NodeGroup, key: Option<u32>) -> Vec<usize> {
        let mut rows: Vec<usize> = group.versions.stamped_rows().collect();
        let values = key.and_then(|key| group.columns.chained_rows(key));
        for chained in values.iter().chain(group.label_updates.rows().as_ref()) {
            rows.extend(chained.rows());
        }
        rows.sort_unstable();
        rows.dedup();
        rows
    }
}

impl RowGroupStore {
    /// Creates an index on a node property, for lookups by value (see
    /// `LpgStore::create_property_index`): it holds every value of the key
    /// any node has in any version, and the writes keep it from then on. An
    /// index on the property already is kept.
    pub fn create_property_index(&self, property: &str) {
        let mut inner = self.inner.write();
        if inner.indexes.property.contains_key(property) {
            return;
        }
        let index = PropertyIndex::default();
        if let Some(key) = inner.keys.get_id(property) {
            for (number, group) in &inner.nodes {
                let base = number * ROWS_PER_GROUP;
                for (row, _) in group.versions.rows() {
                    let id = NodeId::new(base + row as u64);
                    let kept = group
                        .columns
                        .kept(key, row)
                        .filter_map(|(_, kept)| kept.clone());
                    for value in group.columns.get(key, row).into_iter().chain(kept) {
                        index.insert(HashableValue::new(value), id);
                    }
                }
            }
        }
        inner
            .indexes
            .property
            .insert(PropertyKey::new(property), index);
    }

    /// Drops the index on a node property. Returns `true` if the index
    /// existed and was removed.
    pub fn drop_property_index(&self, property: &str) -> bool {
        self.inner
            .write()
            .indexes
            .property
            .remove(property)
            .is_some()
    }

    /// Returns `true` if the property has an index.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        self.inner.read().indexes.property.contains_key(property)
    }

    /// Returns the names of all indexed properties.
    #[must_use]
    pub fn property_index_keys(&self) -> Vec<String> {
        self.inner
            .read()
            .indexes
            .property
            .keys()
            .map(ToString::to_string)
            .collect()
    }

    /// The nodes whose `property` may be equal to `key` under `=` (see
    /// [`GraphStore::find_nodes_maybe_equal`](crate::graph::GraphStore::find_nodes_maybe_equal)),
    /// found through the property's index in any version: every node `=`
    /// finds equal at any view, and maybe others, for a filter to decide at
    /// its view. `None` when the property has no index.
    #[must_use]
    pub fn find_nodes_maybe_equal(&self, property: &str, key: &Value) -> Option<Vec<NodeId>> {
        self.inner.read().maybe_equal(property, key)
    }

    /// How many distinct values the index on `property` holds, `None`
    /// without one: old and pending values count until undo or `collect`
    /// takes them out.
    pub(crate) fn property_index_len(&self, property: &str) -> Option<usize> {
        self.inner
            .read()
            .indexes
            .property
            .get(property)
            .map(PropertyIndex::len)
    }
}

#[cfg(test)]
mod tests;
