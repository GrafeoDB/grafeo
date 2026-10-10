//! Append-only name dictionaries: the ids of a graph's labels, edge types and
//! property keys.
//!
//! A [`NameDictionary`] gives a name a `u32` id the first time it is used and
//! never reassigns or reuses one: a name nothing uses any more keeps its id,
//! and an id that names nothing (a gap a file may hold) is never given out,
//! as [`next_id`](NameDictionary::next_id) only grows. The one exception is
//! a load that drops a name no node or edge has
//! ([`remove`](NameDictionary::remove)): its id becomes such a gap. It is not
//! transactional: a name first used by a transaction that rolls back keeps
//! its id. The LPG section writes a graph's dictionaries as they are and a
//! load restores them id for id, so the ids a file's chunks hold are the ids
//! the store uses, checkpoint after checkpoint.

use arcstr::ArcStr;
use grafeo_common::utils::hash::FxHashMap;

/// Names and their ids, both ways; see the module docs.
#[derive(Debug, Default, Clone)]
pub(crate) struct NameDictionary {
    /// Name to id.
    by_name: FxHashMap<ArcStr, u32>,
    /// Id to name (index = id); `None` for an id that names nothing.
    by_id: Vec<Option<ArcStr>>,
}

impl NameDictionary {
    /// An empty dictionary: the first name gets id 0.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The id of `name`, `None` when it has none.
    pub(crate) fn get_id(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    /// The name of `id`, `None` when it names nothing.
    pub(crate) fn get_name(&self, id: u32) -> Option<&ArcStr> {
        self.by_id.get(id as usize).and_then(Option::as_ref)
    }

    /// The id of `name`, given the next id when it has none.
    ///
    /// # Panics
    ///
    /// Panics when every `u32` id is taken: 4,294,967,296 names, which no
    /// store holds in memory.
    pub(crate) fn get_or_create(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let id = u32::try_from(self.by_id.len())
            .expect("a name dictionary holds at most 4,294,967,296 ids");
        let name: ArcStr = name.into();
        self.by_name.insert(name.clone(), id);
        self.by_id.push(Some(name));
        id
    }

    /// Gives `name` the id `id`, as a load restores it; the ids below `id`
    /// that name nothing become gaps.
    ///
    /// # Errors
    ///
    /// Returns what is wrong when `id` already names something or `name`
    /// already has an id (also the same pair: a load lists each once).
    pub(crate) fn insert_at(&mut self, id: u32, name: &str) -> Result<(), String> {
        if let Some(known) = self.get_name(id) {
            return Err(format!("id {id} already names {known:?}"));
        }
        if let Some(known) = self.get_id(name) {
            return Err(format!("{name:?} already has id {known}"));
        }
        let at = id as usize;
        if self.by_id.len() <= at {
            self.by_id.resize(at + 1, None);
        }
        let name: ArcStr = name.into();
        self.by_name.insert(name.clone(), id);
        self.by_id[at] = Some(name);
        Ok(())
    }

    /// Forgets `name`: its id becomes a gap, which is never given out again
    /// (a later use of the name gets a new id). Returns the id it had, `None`
    /// when it had none. Only a load uses it, for a name no node or edge has
    /// (see `LpgStore::drop_unused_label`).
    pub(crate) fn remove(&mut self, name: &str) -> Option<u32> {
        let id = self.by_name.remove(name)?;
        if let Some(slot) = self.by_id.get_mut(id as usize) {
            *slot = None;
        }
        Some(id)
    }

    /// The id the next new name gets: one past every id given out or
    /// restored, so no id is ever given twice.
    ///
    /// # Panics
    ///
    /// Panics when every `u32` id is taken (see
    /// [`get_or_create`](Self::get_or_create)).
    pub(crate) fn next_id(&self) -> u32 {
        u32::try_from(self.by_id.len()).expect("a name dictionary holds at most 4,294,967,296 ids")
    }

    /// Makes `next` the next id at least, as a load restores a dictionary
    /// whose last ids name nothing: those ids are never given out.
    pub(crate) fn reserve_below(&mut self, next: u32) {
        let next = next as usize;
        if self.by_id.len() < next {
            self.by_id.resize(next, None);
        }
    }

    /// The number of names.
    pub(crate) fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Every name with its id, in id order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, &ArcStr)> {
        self.by_id
            .iter()
            .zip(0u32..)
            .filter_map(|(name, id)| Some((id, name.as_ref()?)))
    }

    /// Estimates heap memory usage in bytes.
    pub(crate) fn heap_bytes(&self) -> usize {
        let map_bytes =
            self.by_name.capacity() * (std::mem::size_of::<ArcStr>() + std::mem::size_of::<u32>());
        let vec_bytes = self.by_id.capacity() * std::mem::size_of::<Option<ArcStr>>();
        let string_bytes: usize = self.by_name.keys().map(|name| name.len()).sum();
        map_bytes + vec_bytes + string_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::NameDictionary;

    fn listed(dictionary: &NameDictionary) -> Vec<(u32, String)> {
        dictionary
            .iter()
            .map(|(id, name)| (id, name.to_string()))
            .collect()
    }

    #[test]
    fn a_name_gets_the_next_id_once_and_keeps_it() {
        let mut dictionary = NameDictionary::new();
        assert_eq!(dictionary.get_or_create("Person"), 0);
        assert_eq!(dictionary.get_or_create("City"), 1);
        assert_eq!(
            dictionary.get_or_create("Person"),
            0,
            "the same name, the same id"
        );
        assert_eq!(dictionary.get_id("City"), Some(1));
        assert_eq!(
            dictionary.get_name(0).map(|name| name.as_str()),
            Some("Person")
        );
        assert_eq!(dictionary.get_id("Town"), None);
        assert_eq!(dictionary.get_name(2), None);
        assert_eq!(dictionary.next_id(), 2);
        assert_eq!(dictionary.len(), 2);
        assert_eq!(
            listed(&dictionary),
            [(0, "Person".to_string()), (1, "City".to_string())]
        );
    }

    /// A load restores ids with gaps; a gap is never given out, the next new
    /// name gets the id after the highest.
    #[test]
    fn restored_ids_keep_their_gaps() {
        let mut dictionary = NameDictionary::new();
        dictionary.insert_at(5, "Prague").unwrap();
        dictionary.insert_at(0, "Amsterdam").unwrap();
        dictionary.insert_at(9, "Berlin").unwrap();
        assert_eq!(dictionary.next_id(), 10);
        assert_eq!(dictionary.get_name(3), None, "a gap names nothing");
        assert_eq!(
            dictionary.get_or_create("Paris"),
            10,
            "after the highest id"
        );
        assert_eq!(dictionary.len(), 4);
        assert_eq!(
            listed(&dictionary),
            [
                (0, "Amsterdam".to_string()),
                (5, "Prague".to_string()),
                (9, "Berlin".to_string()),
                (10, "Paris".to_string()),
            ],
            "in id order, without the gaps"
        );
    }

    #[test]
    fn a_reserved_next_id_is_never_given_out_below() {
        let mut dictionary = NameDictionary::new();
        dictionary.insert_at(1, "KNOWS").unwrap();
        dictionary.reserve_below(19);
        assert_eq!(dictionary.next_id(), 19);
        assert_eq!(dictionary.get_or_create("VISITED"), 19);
        dictionary.reserve_below(3);
        assert_eq!(dictionary.next_id(), 20, "reserving never lowers it");
    }

    /// A removed name leaves a gap: its id names nothing and is never given
    /// out again, and the name, used again, gets a new id.
    #[test]
    fn a_removed_name_leaves_a_gap_that_is_never_given_out() {
        let mut dictionary = NameDictionary::new();
        dictionary.get_or_create("Graph");
        dictionary.get_or_create("Graph|Repository");
        dictionary.get_or_create("Repository");
        assert_eq!(dictionary.remove("Graph|Repository"), Some(1));
        assert_eq!(dictionary.remove("Graph|Repository"), None, "once");
        assert_eq!(dictionary.remove("Missing"), None);
        assert_eq!(dictionary.get_id("Graph|Repository"), None);
        assert_eq!(dictionary.get_name(1), None, "a gap");
        assert_eq!(dictionary.len(), 2);
        assert_eq!(dictionary.next_id(), 3, "the gap stays below the next id");
        assert_eq!(
            listed(&dictionary),
            [(0, "Graph".to_string()), (2, "Repository".to_string())]
        );
        assert_eq!(dictionary.get_or_create("Starred"), 3);
        assert_eq!(
            dictionary.get_or_create("Graph|Repository"),
            4,
            "a new id, not the gap"
        );
    }

    #[test]
    fn a_taken_id_or_name_is_refused() {
        let mut dictionary = NameDictionary::new();
        dictionary.insert_at(3, "name").unwrap();
        let error = dictionary.insert_at(3, "age").unwrap_err();
        assert!(error.contains("id 3 already names \"name\""), "{error}");
        let error = dictionary.insert_at(4, "name").unwrap_err();
        assert!(error.contains("\"name\" already has id 3"), "{error}");
        let error = dictionary.insert_at(3, "name").unwrap_err();
        assert!(error.contains("id 3"), "the same pair twice: {error}");
        assert_eq!(listed(&dictionary), [(3, "name".to_string())], "unchanged");
    }
}
