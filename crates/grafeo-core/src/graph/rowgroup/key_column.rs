//! Finding a key's holders by reading its column, among the rows a scope leaves.
//!
//! An index on a label holds every node with the label in its newest labels or a kept version,
//! under every value it holds in any version (see `index`). So the rows left out by the newest bits
//! of indexed labels are all in those indexes, and only the others are read here.

use grafeo_common::types::{NodeId, Value};

use super::bitset::Bitset;
use super::index::MaybeEqual;
use super::{Inner, ROWS_PER_GROUP};
use crate::graph::traits::{KeyColumnHits, KeyColumnScope};

/// The words of a row group's bits.
const GROUP_WORDS: usize = (ROWS_PER_GROUP / 64) as usize;

impl Inner {
    /// See `GraphStore::find_by_key_column`.
    pub(super) fn nodes_by_key_column(
        &self,
        scope: &KeyColumnScope<'_>,
        property: &str,
        values: &[Value],
    ) -> KeyColumnHits {
        let mut hits = KeyColumnHits::default();
        let finders: Vec<MaybeEqual> = values.iter().filter_map(MaybeEqual::new).collect();
        let Some(key) = self.keys.get_id(property) else {
            return hits;
        };
        if finders.is_empty() {
            return hits;
        }
        let ids = |labels: &[&str]| -> Vec<Option<u32>> {
            labels.iter().map(|label| self.label_id(label)).collect()
        };
        let (required, one_of, except) =
            (ids(scope.required), ids(scope.one_of), ids(scope.except));
        // A label no node has had: no node has it in any version.
        if required.iter().any(Option::is_none) {
            return hits;
        }
        for (number, group) in &self.nodes {
            if !group.columns.has(key) {
                continue;
            }
            let bits = |labels: &[Option<u32>]| -> Vec<&Bitset> {
                labels
                    .iter()
                    .flatten()
                    .filter_map(|id| group.labels.get(id))
                    .collect()
            };
            let (required_bits, one_of_bits, except_bits) =
                (bits(&required), bits(&one_of), bits(&except));
            // Every row with a label of the scope in its newest labels, less those the indexes
            // hold. A label's bits are empty in a group without it.
            let in_scope = |word: usize| -> u64 {
                let mut rows = u64::MAX;
                if required_bits.len() < required.len() {
                    rows = 0;
                }
                for label in &required_bits {
                    rows &= label.word(word);
                }
                if !scope.one_of.is_empty() {
                    rows &= one_of_bits
                        .iter()
                        .fold(0, |any, label| any | label.word(word));
                }
                for label in &except_bits {
                    rows &= !label.word(word);
                }
                rows
            };
            let present = group.columns.hot_present(key);
            let value_chains = group.columns.chained_rows(key);
            let label_chains = group.label_updates.rows();
            let word_of = |chains: &Option<Bitset>, word: usize| {
                chains.as_ref().map_or(0, |rows| rows.word(word))
            };
            let base = number * ROWS_PER_GROUP;
            for word in 0..GROUP_WORDS {
                let scoped = in_scope(word);
                // A row a chain holds may differ at a reader's view whatever it holds now, in
                // its labels or its value: a candidate, read at the view by the caller.
                let always = word_of(&label_chains, word) | (word_of(&value_chains, word) & scoped);
                let to_compare = scoped & present.map_or(u64::MAX, |rows| rows.word(word));
                let mut rows = always | to_compare;
                while rows != 0 {
                    let bit = rows.trailing_zeros() as usize;
                    rows &= rows - 1;
                    let row = word * 64 + bit;
                    let candidate = if always & (1 << bit) != 0 {
                        true
                    } else if !group.versions.exists(row) {
                        continue;
                    } else {
                        group
                            .columns
                            .get(key, row)
                            .is_some_and(|value| finders.iter().any(|finder| finder.finds(&value)))
                    };
                    hits.compared += 1;
                    if candidate {
                        hits.nodes.push(NodeId::new(base + row as u64));
                    }
                }
            }
        }
        hits
    }
}
