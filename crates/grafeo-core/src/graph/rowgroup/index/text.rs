//! The row-group store's text indexes: a BM25 index per label and key that
//! holds the newest text of each node with the label, and the searches at a
//! view (see the parent module).

use std::collections::HashSet;
use std::sync::Arc;

use grafeo_common::types::{ArcStr, NodeId, PropertyKey, ReadView, Value};
use parking_lot::RwLock;

use super::super::{Inner, ROWS_PER_GROUP, Read, RowGroupStore};
use super::{IndexSearch, search_key};
use crate::index::text::InvertedIndex;

/// A text index of the store, with the label and key it indexes.
pub(super) struct TextEntry {
    label: ArcStr,
    property: PropertyKey,
    index: Arc<RwLock<InvertedIndex>>,
}

/// Which documents a text search returns, best first.
#[derive(Debug, Clone, Copy)]
pub(in crate::graph::rowgroup) enum TextLimit {
    /// The best `k`.
    Top(usize),
    /// Every one scoring at least this.
    AtLeast(f64),
}

impl Inner {
    /// The text an index of `entry`'s label and key holds for node `raw`:
    /// the node's newest value of the key, when it is a string and the node
    /// has the label now.
    fn newest_text(&self, raw: u64, entry: &TextEntry) -> Option<ArcStr> {
        let (group, row) = self.node(raw)?;
        let label = self.label_id(&entry.label)?;
        if !group.labels.get(&label).is_some_and(|rows| rows.get(row)) {
            return None;
        }
        match group
            .columns
            .get(self.keys.get_id(entry.property.as_str())?, row)?
        {
            Value::String(text) => Some(text),
            _ => None,
        }
    }

    /// Brings `entry`'s index in line with node `raw`'s newest state: it
    /// holds the node's text, taken again when `value_changed` or when it
    /// lacks it, or it does not hold the node.
    fn sync_text_entry(&self, raw: u64, entry: &TextEntry, value_changed: bool) {
        let id = NodeId::new(raw);
        match self.newest_text(raw, entry) {
            Some(text) => {
                let mut index = entry.index.write();
                if value_changed || !index.contains(id) {
                    index.insert(id, &text);
                }
            }
            None => {
                // Writes of the store hold its lock: no one adds the node
                // between the test and the removal but the engine, which
                // the removal then undoes as the newest state wants.
                let held = entry.index.read().contains(id);
                if held {
                    entry.index.write().remove(id);
                }
            }
        }
    }

    /// Brings the text indexes on key `key` (`None`: any) and of label
    /// `label` (`None`: any) in line with node `raw` (see
    /// [`sync_text_entry`](Self::sync_text_entry)).
    pub(in crate::graph::rowgroup) fn sync_text(
        &self,
        raw: u64,
        key: Option<&str>,
        label: Option<&str>,
        value_changed: bool,
    ) {
        for entry in self.indexes.text.values() {
            if key.is_some_and(|key| key != entry.property.as_str())
                || label.is_some_and(|label| label != entry.label.as_str())
            {
                continue;
            }
            self.sync_text_entry(raw, entry, value_changed);
        }
    }

    /// Takes node `id` out of every text index.
    pub(in crate::graph::rowgroup) fn unindex_text(&self, id: NodeId) {
        for entry in self.indexes.text.values() {
            let held = entry.index.read().contains(id);
            if held {
                entry.index.write().remove(id);
            }
        }
    }

    /// Brings a text index the store takes in line with the newest state of
    /// the nodes whose versions readers may tell apart: an index built from
    /// what a reader at the current epoch sees lacks what open transactions
    /// wrote and the nodes deleted since an older reader's epoch.
    fn reconcile_text(&self, entry: &TextEntry) {
        let key = self.keys.get_id(entry.property.as_str());
        // Only the listed row groups hold stamps or chains (`collect`).
        for (number, group) in self.listed_node_groups() {
            let base = number * ROWS_PER_GROUP;
            for row in Self::unsettled_rows(group, key) {
                let value_changed =
                    key.is_some_and(|key| group.columns.kept(key, row).next().is_some());
                self.sync_text_entry(base + row as u64, entry, value_changed);
            }
        }
    }

    /// Node `raw`'s text of key `key` as `read` sees the node, when the
    /// node has label id `label` at the view.
    fn text_at(&self, raw: u64, label: u32, key: Option<u32>, read: Read) -> Option<ArcStr> {
        let (group, row) = self.node(raw)?;
        if !group.version(row).visible(read) || !group.has_label_at(row, label, read) {
            return None;
        }
        match group.columns.get_at(key?, row, read)? {
            Value::String(text) => Some(text),
            _ => None,
        }
    }

    /// The documents of the text index on `label` and `property` that
    /// `read` sees holding a term of `query`, best first, among the nodes of
    /// `allowed` only when given: the index's scores for the nodes every
    /// reader sees alike, the text at the view scored with the index's
    /// statistics of now for the nodes whose value or labels a chain holds.
    /// Empty without the index.
    pub(in crate::graph::rowgroup) fn text_search_at(
        &self,
        label: &str,
        property: &str,
        query: &str,
        limit: TextLimit,
        allowed: Option<&HashSet<NodeId>>,
        read: Read,
    ) -> Vec<(NodeId, f64)> {
        let Some(entry) = self.indexes.text.get(&search_key(label, property)) else {
            return Vec::new();
        };
        let key = self.keys.get_id(property);
        let changed = self.changed_rows(key);
        let index = entry.index.read();
        let mut scores = index.scores(query, |id| {
            allowed.is_none_or(|allowed| allowed.contains(&id))
                && !changed.contains(&id.as_u64())
                && self.node_visible(id.as_u64(), read)
        });
        if let Some(label) = self.label_id(label) {
            for raw in changed {
                if allowed.is_some_and(|allowed| !allowed.contains(&NodeId::new(raw))) {
                    continue;
                }
                if let Some(text) = self.text_at(raw, label, key, read)
                    && let Some(score) = index.score_text(&text, query)
                {
                    scores.insert(NodeId::new(raw), score);
                }
            }
        }
        match limit {
            TextLimit::Top(k) => InvertedIndex::top(scores, k),
            TextLimit::AtLeast(threshold) => {
                scores.retain(|_, score| *score >= threshold);
                let count = scores.len();
                InvertedIndex::top(scores, count)
            }
        }
    }

    /// The BM25 score of node `raw`'s text against `query` as `read` sees
    /// the node, through the text index on `label` and `property`: the
    /// index's score of its document where every reader sees the node
    /// alike, its text at the view scored with the statistics of now where
    /// a chain holds its value or labels, 0 for a node the view does not see
    /// with the label and a text. `None` without the index.
    pub(in crate::graph::rowgroup) fn score_text_at(
        &self,
        raw: u64,
        label: &str,
        property: &str,
        query: &str,
        read: Read,
    ) -> Option<f64> {
        let entry = self.indexes.text.get(&search_key(label, property))?;
        let index = entry.index.read();
        let key = self.keys.get_id(property);
        let Some((group, row)) = self.node(raw) else {
            return Some(0.0);
        };
        if !Self::changed(group, row, key) {
            let seen = group.version(row).visible(read);
            return Some(if seen {
                index.score_document(NodeId::new(raw), query)
            } else {
                0.0
            });
        }
        Some(
            self.label_id(label)
                .and_then(|label| self.text_at(raw, label, key, read))
                .and_then(|text| index.score_text(&text, query))
                .unwrap_or(0.0),
        )
    }

    /// Whether the store has a text index on `label` and `property`.
    pub(in crate::graph::rowgroup) fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.indexes.text.contains_key(&search_key(label, property))
    }
}

impl RowGroupStore {
    /// Stores a text index for a label and a property, in place of one on
    /// the same pair (see `LpgStore::add_text_index`). The store keeps it
    /// from then on: the writes give it each node's newest text. It first
    /// brings the index in line with the newest state of the nodes whose
    /// versions readers may tell apart (an index built from what a reader at
    /// the current epoch sees lacks open transactions' writes).
    pub fn add_text_index(&self, label: &str, property: &str, index: Arc<RwLock<InvertedIndex>>) {
        let mut inner = self.inner.write();
        let entry = TextEntry {
            label: ArcStr::from(label),
            property: PropertyKey::new(property),
            index,
        };
        inner.reconcile_text(&entry);
        inner
            .indexes
            .text
            .insert(search_key(label, property), entry);
    }

    /// The best `search.k` documents of the text index on `label` and
    /// `property` for `query` that `view` sees, best first (equal scores by
    /// ascending node id), among `search`'s allowed nodes only when given:
    /// the database's `text_search`. Scores use the index's statistics of
    /// now (decision Q5 A). `None` without the index.
    ///
    /// The store's lock is held for the search, and the index's lock only
    /// inside it, in the order the writes take them.
    #[must_use]
    pub fn text_index_search_at(
        &self,
        label: &str,
        property: &str,
        query: &str,
        search: IndexSearch<'_>,
        view: ReadView,
    ) -> Option<Vec<(NodeId, f64)>> {
        let inner = self.inner.read();
        if !inner.has_text_index(label, property) {
            return None;
        }
        Some(inner.text_search_at(
            label,
            property,
            query,
            TextLimit::Top(search.k),
            search.allowed,
            Read::view(view),
        ))
    }

    /// Retrieves the text index for a label and a property.
    #[must_use]
    pub fn get_text_index(
        &self,
        label: &str,
        property: &str,
    ) -> Option<Arc<RwLock<InvertedIndex>>> {
        self.inner
            .read()
            .indexes
            .text
            .get(&search_key(label, property))
            .map(|entry| Arc::clone(&entry.index))
    }

    /// Removes the text index for a label and a property. Returns `true` if
    /// the index existed and was removed.
    pub fn remove_text_index(&self, label: &str, property: &str) -> bool {
        self.inner
            .write()
            .indexes
            .text
            .remove(&search_key(label, property))
            .is_some()
    }

    /// Returns all text index entries as `(key, index)` pairs, keys in
    /// `"label:property"` format.
    #[must_use]
    pub fn text_index_entries(&self) -> Vec<(String, Arc<RwLock<InvertedIndex>>)> {
        self.inner
            .read()
            .indexes
            .text
            .iter()
            .map(|(key, entry)| (key.clone(), Arc::clone(&entry.index)))
            .collect()
    }
}
