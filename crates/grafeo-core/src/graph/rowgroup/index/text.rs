//! The row-group store's text indexes: a BM25 index per label and key that
//! holds the newest text of each node with the label, and the searches at a
//! view (see the parent module).

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::types::{ArcStr, NodeId, PropertyKey, ReadView, Value};
use parking_lot::RwLock;

use super::super::{Inner, ROWS_PER_GROUP, Read, RowGroupStore, Touched};
use super::IndexSearch;
use super::key::{SearchKey, search_key};
use crate::index::text::InvertedIndex;

/// A text index of the store, with the label and key it indexes.
pub(super) struct TextEntry {
    label: ArcStr,
    property: PropertyKey,
    index: Arc<RwLock<InvertedIndex>>,
    /// The index's heap bytes as the store's writes last left them (see
    /// [`InvertedIndex::counted_bytes`]), read without its lock.
    counted: AtomicUsize,
}

impl TextEntry {
    /// The index's heap bytes as the writes keep them: one load.
    pub(super) fn counted_bytes(&self) -> usize {
        self.counted.load(Ordering::Relaxed)
    }

    /// The index's heap bytes by a walk of its terms: the oracle.
    pub(super) fn walk_bytes(&self) -> usize {
        self.index.read().heap_memory_bytes()
    }

    /// Notes what `index`, this entry's under its write lock, holds now.
    fn note(&self, index: &InvertedIndex) {
        self.counted.store(index.counted_bytes(), Ordering::Relaxed);
    }
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
                    entry.note(&index);
                }
            }
            None => {
                // Writes of the store hold its lock: no one adds the node
                // between the test and the removal but the engine, which
                // the removal then undoes as the newest state wants.
                let held = entry.index.read().contains(id);
                if held {
                    let mut index = entry.index.write();
                    index.remove(id);
                    entry.note(&index);
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
                let mut index = entry.index.write();
                index.remove(id);
                entry.note(&index);
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
    ) -> grafeo_common::utils::error::Result<Vec<(NodeId, f64)>> {
        let Some(entry) = self.indexes.text.get(&search_key(label, property)) else {
            return Ok(Vec::new());
        };
        // The running query's deadline, a unit of work per posting and per changed row.
        let mut ticker = crate::execution::deadline::Ticker::new();
        let timeout = |_| {
            grafeo_common::utils::error::Error::from(
                grafeo_common::utils::error::QueryError::timeout(),
            )
        };
        let key = self.keys.get_id(property);
        let changed = self.changed_rows(key);
        let index = entry.index.read();
        let mut scores = index
            .scores_until(
                query,
                |id| {
                    allowed.is_none_or(|allowed| allowed.contains(&id))
                        && !changed.contains(&id.as_u64())
                        && self.node_visible(id.as_u64(), read)
                },
                &mut ticker,
            )
            .map_err(timeout)?;
        if let Some(label) = self.label_id(label) {
            for raw in changed {
                ticker.tick().map_err(timeout)?;
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
        Ok(match limit {
            TextLimit::Top(k) => InvertedIndex::top(scores, k),
            TextLimit::AtLeast(threshold) => {
                scores.retain(|_, score| *score >= threshold);
                let count = scores.len();
                InvertedIndex::top(scores, count)
            }
        })
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
        let counted = index.read().counted_bytes();
        let entry = TextEntry {
            label: ArcStr::from(label),
            property: PropertyKey::new(property),
            counted: AtomicUsize::new(counted),
            index,
        };
        inner.counting(Touched::none(), |inner| {
            inner.reconcile_text(&entry);
            inner
                .indexes
                .text
                .insert(SearchKey::new(label, property), entry);
        });
        inner.settle();
    }

    /// The best `search.k` documents of the text index on `label` and
    /// `property` for `query` that `view` sees, best first (equal scores by
    /// ascending node id), among `search`'s allowed nodes only when given:
    /// the database's `text_search`. Scores use the index's statistics of
    /// now (decision Q5 A). `None` without the index.
    ///
    /// The store's lock is held for the search, and the index's lock only
    /// inside it, in the order the writes take them.
    ///
    /// # Errors
    ///
    /// The timeout of a search past the running query's deadline.
    pub fn text_index_search_at(
        &self,
        label: &str,
        property: &str,
        query: &str,
        search: IndexSearch<'_>,
        view: ReadView,
    ) -> grafeo_common::utils::error::Result<Option<Vec<(NodeId, f64)>>> {
        let inner = self.inner.read();
        if !inner.has_text_index(label, property) {
            return Ok(None);
        }
        Some(inner.text_search_at(
            label,
            property,
            query,
            TextLimit::Top(search.k),
            search.allowed,
            Read::view(view),
        ))
        .transpose()
    }

    /// Retrieves the text index for a label and a property.
    ///
    /// The handle is for reads (searches, lengths, statistics). Write through the store, which
    /// counts what the index holds: an insert or a removal made through the handle leaves the
    /// store's memory account stale.
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
        let mut inner = self.inner.write();
        let removed = inner.counting(Touched::none(), |inner| {
            inner
                .indexes
                .text
                .remove(&search_key(label, property))
                .is_some()
        });
        inner.settle();
        removed
    }

    /// The text indexes of this store as the account counts them: label,
    /// property, heap bytes and documents, without a walk.
    pub(in crate::graph::rowgroup) fn text_index_figures(
        &self,
    ) -> Vec<(String, String, usize, usize)> {
        self.inner
            .read()
            .indexes
            .text
            .iter()
            .map(|(key, entry)| {
                (
                    key.label.clone(),
                    key.property.clone(),
                    entry.counted_bytes(),
                    entry.index.read().len(),
                )
            })
            .collect()
    }

    /// Returns all text indexes as `(label, property, index)`.
    #[must_use]
    pub fn text_index_entries(&self) -> Vec<(String, String, Arc<RwLock<InvertedIndex>>)> {
        self.inner
            .read()
            .indexes
            .text
            .iter()
            .map(|(key, entry)| {
                (
                    key.label.clone(),
                    key.property.clone(),
                    Arc::clone(&entry.index),
                )
            })
            .collect()
    }
}
