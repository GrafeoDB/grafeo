//! The key of a text or vector index: its label and its property, apart, so
//! that a name holding a `:` cannot give two indexes one key (the indexes on
//! (`a`, `b:emb`) and (`a:b`, `emb`) were both `a:b:emb` when the key was the
//! joined `label:property`).

/// The key of a text or vector index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct SearchKey {
    pub(super) label: String,
    pub(super) property: String,
}

impl SearchKey {
    pub(super) fn new(label: &str, property: &str) -> Self {
        Self {
            label: label.to_string(),
            property: property.to_string(),
        }
    }
}

/// A [`SearchKey`] borrowed, to look an index up without building its key.
/// Its fields hash as the key's do (a `String` hashes as its `str`).
#[derive(Hash)]
pub(super) struct SearchKeyRef<'a> {
    label: &'a str,
    property: &'a str,
}

impl hashbrown::Equivalent<SearchKey> for SearchKeyRef<'_> {
    fn equivalent(&self, key: &SearchKey) -> bool {
        self.label == key.label && self.property == key.property
    }
}

/// The key of the text or vector index on `label` and `property`, for a
/// lookup.
pub(super) fn search_key<'a>(label: &'a str, property: &'a str) -> SearchKeyRef<'a> {
    SearchKeyRef { label, property }
}

#[cfg(test)]
mod tests {
    use std::hash::BuildHasher;

    use grafeo_common::utils::hash::{FxBuildHasher, FxHashMap};

    use super::*;

    /// A borrowed key hashes as the key it stands for, so a lookup finds it,
    /// and keys whose joined names are equal stay apart.
    #[test]
    fn a_borrowed_key_finds_its_key_and_only_it() {
        let hasher = FxBuildHasher::default();
        assert_eq!(
            hasher.hash_one(SearchKey::new("a:b", "emb")),
            hasher.hash_one(search_key("a:b", "emb"))
        );

        let mut map = FxHashMap::default();
        map.insert(SearchKey::new("a", "b:emb"), "Alix");
        map.insert(SearchKey::new("a:b", "emb"), "Gus");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&search_key("a", "b:emb")), Some(&"Alix"));
        assert_eq!(map.get(&search_key("a:b", "emb")), Some(&"Gus"));
        assert_eq!(map.get(&search_key("a:b:emb", "")), None);
    }
}
