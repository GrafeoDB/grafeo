//! The row-group store's indexes (#412 S7, H1d): a property index per key or
//! per label and key, and text and vector indexes per label and key, each
//! graph's its own, kept by the writes as they apply (decision Q4 A).
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
//! An index on a label also holds what a reader may need: a node while its
//! newest labels or a kept version has the label, under every value of the key
//! it holds in any version, so a reader that had it at its view finds it. A
//! write of a value while no version has the label enters nothing, as no reader
//! sees the two together, and a label added enters every version's value. An
//! entry goes when no version has the label: at once for an undone add, and at
//! `collect` for a label removed. A reader checks the label at its view, as it
//! checks the value.
//!
//! A text or vector index holds one document or vector per node, so it holds
//! the newest state: the text or vector of each node with the label, in
//! place, deleted nodes until `collect` takes them out (`text`, `vector`). A
//! search at a view takes the index's answer for the nodes every reader sees
//! alike and reads the nodes whose value or labels a chain holds (what
//! changed since the horizon) at the view, scoring them itself: membership
//! is exact, and a BM25 score is computed with the statistics of now
//! (decision Q5 A).

#[cfg(any(feature = "text-index", feature = "vector-index"))]
mod key;
mod property;
#[cfg(feature = "text-index")]
mod text;
#[cfg(feature = "vector-index")]
mod vector;

#[cfg(feature = "text-index")]
pub(super) use text::TextLimit;

use std::fmt;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::types::{HashableValue, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;

pub(super) use self::property::MaybeEqual;
use self::property::PropertyIndex;
use super::{Inner, NodeGroup, PENDING, ROWS_PER_GROUP, Read, RowGroupStore, Touched};

/// The property index on one label of a key.
#[derive(Debug)]
struct LabelIndex {
    label: String,
    index: PropertyIndex,
    /// The label's id, once it has one: ids are permanent, so a write need not look the name up.
    id: OnceLock<u32>,
    /// How many labels the store had when the name last had no id: until there are more it has
    /// none yet.
    missed_at: AtomicUsize,
}

impl LabelIndex {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            index: PropertyIndex::default(),
            id: OnceLock::new(),
            missed_at: AtomicUsize::new(usize::MAX),
        }
    }
}

/// A store's indexes.
#[derive(Default)]
pub(super) struct Indexes {
    /// The property indexes on every label, by key.
    property: FxHashMap<PropertyKey, PropertyIndex>,
    /// The property indexes on one label, by key, then in the order they were made. By label
    /// name, so an index on a label no node has yet gives the label no id.
    on_label: FxHashMap<PropertyKey, Vec<LabelIndex>>,
    /// The text indexes, by label and property.
    #[cfg(feature = "text-index")]
    text: FxHashMap<key::SearchKey, text::TextEntry>,
    /// The vector indexes, by label and property.
    #[cfg(feature = "vector-index")]
    vector: FxHashMap<key::SearchKey, vector::VectorEntry>,
}

impl fmt::Debug for Indexes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("Indexes");
        debug.field("property", &self.property.keys().collect::<Vec<_>>());
        debug.field("on_label", &self.on_label.keys().collect::<Vec<_>>());
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
        self.property.is_empty() && self.on_label.is_empty() && !self.searches()
    }

    /// The index a lookup of `property` reads whose results all have each of `labels`: the
    /// first of them with an index on the key, in the order given, else the key's own. With no
    /// label, only the key's: an index on a label holds no other label's nodes.
    fn serving(&self, labels: &[&str], property: &str) -> Option<&PropertyIndex> {
        self.on_label
            .get(property)
            .and_then(|on_label| {
                labels.iter().find_map(|label| {
                    on_label
                        .iter()
                        .find(|entry| entry.label == *label)
                        .map(|entry| &entry.index)
                })
            })
            .or_else(|| self.property.get(property))
    }

    /// The heap bytes of its property indexes (on every label and on one), its text indexes
    /// and its vector indexes, as the writes keep each index's count: a load per index.
    pub(super) fn counted_bytes(&self) -> [usize; 3] {
        self.bytes(true)
    }

    /// [`counted_bytes`](Self::counted_bytes) by a walk of every index: the oracle.
    pub(super) fn walk_bytes(&self) -> [usize; 3] {
        self.bytes(false)
    }

    /// The heap bytes of its property, text and vector indexes, as their counts (`counted`)
    /// or their walks give them.
    fn bytes(&self, counted: bool) -> [usize; 3] {
        let of = |index: &PropertyIndex| {
            if counted {
                index.counted_bytes()
            } else {
                index.walk_bytes()
            }
        };
        let mut bytes = [0; 3];
        bytes[0] = self.property.values().map(of).sum::<usize>()
            + self
                .on_label
                .values()
                .flatten()
                .map(|entry| of(&entry.index))
                .sum::<usize>();
        #[cfg(feature = "text-index")]
        {
            bytes[1] = self
                .text
                .values()
                .map(|entry| {
                    if counted {
                        entry.counted_bytes()
                    } else {
                        entry.walk_bytes()
                    }
                })
                .sum();
        }
        #[cfg(feature = "vector-index")]
        {
            bytes[2] = self
                .vector
                .values()
                .map(|entry| {
                    if counted {
                        entry.counted_bytes()
                    } else {
                        entry.walk_bytes()
                    }
                })
                .sum();
        }
        bytes
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
        if let Some(on_label) = self.indexes.on_label.get(name.as_str()) {
            for entry in on_label {
                // No version with the label: no reader sees the two together, and the node
                // has no entry to take out.
                if !self.has_newest_label(group, row, entry)
                    && !self.had_label(group, row, entry, |_| true)
                {
                    continue;
                }
                if let Some(newest) = group.columns.get(key, row) {
                    entry.index.insert(HashableValue::new(newest), id);
                }
                if let Some(replaced) = replaced {
                    let replaced = HashableValue::new(replaced.clone());
                    if !Self::holds(group, row, key, &replaced) {
                        entry.index.remove(&replaced, id);
                    }
                }
            }
        }
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        if self.indexes.searches() {
            self.sync_searches(raw, Some(name.as_str()), None, true);
        }
    }

    /// The id of the label of `entry`, `None` while no node has had it.
    fn id_of_label(&self, entry: &LabelIndex) -> Option<u32> {
        if let Some(id) = entry.id.get() {
            return Some(*id);
        }
        let names = self.labels.len();
        if entry.missed_at.load(Ordering::Relaxed) == names {
            return None;
        }
        match self.label_id(&entry.label) {
            Some(id) => Some(*entry.id.get_or_init(|| id)),
            None => {
                entry.missed_at.store(names, Ordering::Relaxed);
                None
            }
        }
    }

    /// Whether row `row`'s newest labels have the label of `entry`.
    fn has_newest_label(&self, group: &NodeGroup, row: usize, entry: &LabelIndex) -> bool {
        self.id_of_label(entry)
            .and_then(|id| group.labels.get(&id))
            .is_some_and(|rows| rows.get(row))
    }

    /// Whether a version a chain keeps of row `row`'s labels has the label of `entry`, among
    /// those `past` lets through by the stamp of who wrote over them.
    fn had_label(
        &self,
        group: &NodeGroup,
        row: usize,
        entry: &LabelIndex,
        past: impl Fn(u64) -> bool,
    ) -> bool {
        !group.label_updates.is_empty()
            && self.id_of_label(entry).is_some_and(|id| {
                group
                    .label_updates
                    .kept_stamped(row)
                    .any(|(stamp, kept)| past(stamp) && kept.binary_search(&id).is_ok())
            })
    }

    /// The values of key `name` row `row` holds, the newest and every one a chain keeps.
    fn every_value(group: &NodeGroup, row: usize, key: u32) -> impl Iterator<Item = Value> + '_ {
        let kept = group
            .columns
            .kept(key, row)
            .filter_map(|(_, kept)| kept.clone());
        group.columns.get(key, row).into_iter().chain(kept)
    }

    /// Brings the indexes on labels in line with node `raw`'s labels: of `label` (`None`: any).
    /// A node with the label enters with every version's value, and one no version has the
    /// label of leaves. One a chain keeps the label of stays, for the readers that see it,
    /// until `collect`.
    fn sync_label_indexes(&self, raw: u64, label: Option<&str>) {
        let Some((group, row)) = self.node(raw) else {
            return;
        };
        let id = NodeId::new(raw);
        for (name, on_label) in &self.indexes.on_label {
            let Some(key) = self.keys.get_id(name.as_str()) else {
                continue;
            };
            for entry in on_label {
                if label.is_some_and(|label| label != entry.label) {
                    continue;
                }
                if self.has_newest_label(group, row, entry) {
                    for value in Self::every_value(group, row, key) {
                        entry.index.insert(HashableValue::new(value), id);
                    }
                } else if !self.had_label(group, row, entry, |_| true) {
                    for value in Self::every_value(group, row, key) {
                        entry.index.remove(&HashableValue::new(value), id);
                    }
                }
            }
        }
    }

    /// Brings the indexes in line with a change of node `raw`'s labels, a
    /// write of label `label` (`None`: any, an undo): for the text and vector
    /// indexes an index of a label the node has now holds it, one of a label
    /// it lost does not; for the property indexes on a label, see
    /// [`sync_label_indexes`](Self::sync_label_indexes).
    pub(super) fn index_labels(&self, raw: u64, label: Option<&str>) {
        if !self.indexes.on_label.is_empty() {
            self.sync_label_indexes(raw, label);
        }
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        if self.indexes.searches() {
            self.sync_searches(raw, None, label, false);
        }
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
        for (name, on_label) in &self.indexes.on_label {
            let mut holding = on_label
                .iter()
                .filter(|entry| self.has_newest_label(group, row, entry))
                .peekable();
            if holding.peek().is_none() {
                continue;
            }
            let Some(value) = self
                .keys
                .get_id(name.as_str())
                .and_then(|key| group.columns.get(key, row))
            else {
                continue;
            };
            for entry in holding {
                entry.index.insert(HashableValue::new(value.clone()), id);
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
        for (name, on_label) in &self.indexes.on_label {
            let Some(key) = self.keys.get_id(name.as_str()) else {
                continue;
            };
            for value in Self::every_value(group, row, key) {
                let value = HashableValue::new(value);
                for entry in on_label {
                    entry.index.remove(&value, id);
                }
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
    /// entries of values only versions committed by the horizon held, and on
    /// an index on a label the entries of a node whose label only such versions
    /// held. It looks at the node row groups `groups` alone: the listed ones,
    /// which hold every stamp and chain (`collect`), or every one.
    pub(super) fn unindex_collected(&self, horizon: u64, groups: &[u64]) {
        if self.indexes.is_empty() {
            return;
        }
        let keys: Vec<(u32, &PropertyIndex)> = self
            .indexes
            .property
            .iter()
            .filter_map(|(name, index)| Some((self.keys.get_id(name.as_str())?, index)))
            .chain(self.indexes.on_label.iter().flat_map(|(name, on_label)| {
                let key = self.keys.get_id(name.as_str());
                on_label
                    .iter()
                    .filter_map(move |entry| Some((key?, &entry.index)))
            }))
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
            self.unindex_labels_collected(group, base, horizon);
        }
    }

    /// Takes out of the indexes on labels the nodes of `group`, whose rows start at `base`, whose
    /// label only versions committed by the horizon held.
    fn unindex_labels_collected(&self, group: &NodeGroup, base: u64, horizon: u64) {
        if self.indexes.on_label.is_empty() {
            return;
        }
        let Some(rows) = group.label_updates.rows() else {
            return;
        };
        let past_the_horizon = |stamp: u64| stamp & PENDING != 0 || stamp > horizon;
        for row in rows.rows() {
            let id = NodeId::new(base + row as u64);
            for (name, on_label) in &self.indexes.on_label {
                let Some(key) = self.keys.get_id(name.as_str()) else {
                    continue;
                };
                for entry in on_label {
                    if !self.has_newest_label(group, row, entry)
                        && !self.had_label(group, row, entry, past_the_horizon)
                    {
                        for value in Self::every_value(group, row, key) {
                            entry.index.remove(&HashableValue::new(value), id);
                        }
                    }
                }
            }
        }
    }

    /// Whether the index that serves `labels` on `property` holds at most `limit` candidates
    /// for a lookup of `value`; `false` without one.
    pub(super) fn property_candidates_within(
        &self,
        labels: &[&str],
        property: &str,
        value: &Value,
        limit: usize,
    ) -> bool {
        self.indexes
            .serving(labels, property)
            .is_some_and(|index| index.candidates_within(value, limit))
    }

    /// The nodes `read` sees whose `property` is `value`, through the index
    /// on the key: the candidates the index holds for the value, each read
    /// again at the view. `None` without an index on the key.
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
    /// starting from the indexed condition with the fewest candidates, by
    /// the indexes that serve `labels`. `None` when no condition's key has one.
    pub(super) fn indexed_matching_at(
        &self,
        labels: &[&str],
        conditions: &[(&str, Value)],
        read: Read,
    ) -> Option<Vec<NodeId>> {
        let candidates = conditions
            .iter()
            .filter_map(|(key, value)| {
                let index = self.indexes.serving(labels, key)?;
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

    /// The kinds the index a lookup of `property` among nodes with each of `labels` reads has
    /// held (see [`maybe_equal_at`](Self::maybe_equal_at)); `None` without one.
    pub(super) fn index_kinds(
        &self,
        labels: &[&str],
        property: &str,
    ) -> Option<crate::execution::operators::value_utils::ValueKinds> {
        self.indexes
            .serving(labels, property)
            .map(PropertyIndex::kinds)
    }

    /// [`maybe_equal`](Self::maybe_equal) at `read`: the candidates whose
    /// value as `read` sees it the index that serves `labels` finds for `key`,
    /// ascending. `None` without one.
    pub(super) fn maybe_equal_at(
        &self,
        labels: &[&str],
        property: &str,
        key: &Value,
        read: Read,
    ) -> Option<Vec<NodeId>> {
        let index = self.indexes.serving(labels, property)?;
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
        inner.counting(Touched::none(), |inner| {
            inner.build_property_index(property);
        });
        inner.settle();
    }

    /// Creates an index on node property `property` of label `label`: it holds what the
    /// module documentation says, and the writes keep it from then on. An index on them
    /// already is kept. No write of a label gives `label` an id here, so it stays out of the
    /// store's listings until a node has it.
    pub fn create_label_index(&self, label: &str, property: &str) {
        let mut inner = self.inner.write();
        inner.counting(Touched::none(), |inner| {
            inner.build_label_index(label, property);
        });
        inner.settle();
    }

    /// Drops the index on node property `property` of label `label`. Returns `true` if it
    /// existed and was removed.
    pub fn drop_label_index(&self, label: &str, property: &str) -> bool {
        let mut inner = self.inner.write();
        let dropped = inner.counting(Touched::none(), |inner| {
            inner.drop_label_index(label, property)
        });
        inner.settle();
        dropped
    }
}

impl Inner {
    /// Builds the index on `property` from every value of the key any node has in any version.
    fn build_property_index(&mut self, property: &str) {
        let inner = self;
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

    /// Builds the index of [`RowGroupStore::create_label_index`], unless one on the label and
    /// the property is there.
    fn build_label_index(&mut self, label: &str, property: &str) {
        let inner = self;
        let held = inner
            .indexes
            .on_label
            .get(property)
            .is_some_and(|on_label| on_label.iter().any(|entry| entry.label == label));
        if held {
            return;
        }
        let entry = LabelIndex::new(label);
        if let (Some(key), Some(id)) = (inner.keys.get_id(property), inner.id_of_label(&entry)) {
            for (number, group) in &inner.nodes {
                let base = number * ROWS_PER_GROUP;
                let newest = group.labels.get(&id);
                let chained = group.label_updates.rows();
                let rows = newest.iter().flat_map(|rows| rows.rows()).chain(
                    chained.iter().flat_map(|rows| rows.rows()).filter(|row| {
                        !newest.is_some_and(|rows| rows.get(*row))
                            && inner.had_label(group, *row, &entry, |_| true)
                    }),
                );
                for row in rows {
                    let node = NodeId::new(base + row as u64);
                    for value in Inner::every_value(group, row, key) {
                        entry.index.insert(HashableValue::new(value), node);
                    }
                }
            }
        }
        inner
            .indexes
            .on_label
            .entry(PropertyKey::new(property))
            .or_default()
            .push(entry);
    }

    /// Drops the index on node property `property` of label `label`; whether it existed.
    fn drop_label_index(&mut self, label: &str, property: &str) -> bool {
        let on_labels = &mut self.indexes.on_label;
        let Some(on_label) = on_labels.get_mut(property) else {
            return false;
        };
        let before = on_label.len();
        on_label.retain(|entry| entry.label != label);
        let dropped = on_label.len() != before;
        if on_label.is_empty() {
            on_labels.remove(property);
        }
        dropped
    }
}

impl RowGroupStore {
    /// The indexes on labels, as `(label, property)`, sorted.
    #[must_use]
    pub fn label_index_entries(&self) -> Vec<(String, String)> {
        let inner = self.inner.read();
        let mut entries: Vec<(String, String)> = inner
            .indexes
            .on_label
            .iter()
            .flat_map(|(name, on_label)| {
                on_label
                    .iter()
                    .map(|entry| (entry.label.clone(), name.to_string()))
            })
            .collect();
        entries.sort();
        entries
    }

    /// Drops the index on a node property that covers every label. Returns `true` if the
    /// index existed and was removed.
    pub fn drop_property_index(&self, property: &str) -> bool {
        let mut inner = self.inner.write();
        let dropped = inner.counting(Touched::none(), |inner| {
            inner.indexes.property.remove(property).is_some()
        });
        inner.settle();
        dropped
    }

    /// Returns `true` if the property has an index, on every label or on one.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        let inner = self.inner.read();
        inner.indexes.property.contains_key(property)
            || inner.indexes.on_label.contains_key(property)
    }

    /// How many distinct values the index that serves `labels` on `property` holds: old and
    /// pending values count until undo or `collect` takes them out.
    #[must_use]
    pub fn indexed_distinct_values(&self, labels: &[&str], property: &str) -> Option<u64> {
        let inner = self.inner.read();
        let index = inner.indexes.serving(labels, property)?;
        u64::try_from(index.len()).ok()
    }

    /// Whether `label` has an index of its own on `property`.
    #[must_use]
    pub fn has_label_index(&self, label: &str, property: &str) -> bool {
        self.inner
            .read()
            .indexes
            .on_label
            .get(property)
            .is_some_and(|on_label| on_label.iter().any(|entry| entry.label == label))
    }

    /// Whether a lookup of `property` whose results all have each of `labels` has an index to
    /// read (see `GraphStore::index_serves`).
    #[must_use]
    pub fn index_serves(&self, labels: &[&str], property: &str) -> bool {
        self.inner
            .read()
            .indexes
            .serving(labels, property)
            .is_some()
    }

    /// Returns the keys of the property indexes that cover every label.
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

    /// The entries of the index on `property` that may be equal to `key`
    /// under `=` (see
    /// [`GraphStore::find_nodes_maybe_equal`](crate::graph::traits::GraphStore::find_nodes_maybe_equal)),
    /// in any version: every node `=` finds equal at any view, and maybe
    /// others, for a filter to decide at its view. `None` when the property
    /// has no index. What the index holds, for its tests and diagnostics; a
    /// reader looks up at its view.
    #[must_use]
    pub fn property_index_entries(&self, property: &str, key: &Value) -> Option<Vec<NodeId>> {
        self.inner.read().maybe_equal(property, key)
    }

    /// How many distinct values the index on `property` holds, `None`
    /// without one: old and pending values count until undo or `collect`
    /// takes them out.
    #[cfg(test)]
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
