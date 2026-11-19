//! Trie index for Worst-Case Optimal Joins (WCOJ).
//!
//! When you're finding triangles, cliques, or other complex patterns, traditional
//! binary joins can be slow. This trie enables the Leapfrog Trie Join algorithm
//! which is worst-case optimal - it runs in time proportional to the output size.
//!
//! Built lazily on-demand. You won't interact with this directly unless you're
//! implementing custom join algorithms.

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::hash::FxHashMap;
use smallvec::SmallVec;

/// A trie node in the edge trie.
#[derive(Debug, Clone)]
struct TrieNode {
    /// Children indexed by node ID.
    children: FxHashMap<NodeId, TrieNode>,
    /// Edge IDs at this level (for leaf nodes or intermediate data).
    edges: SmallVec<[EdgeId; 4]>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: FxHashMap::default(),
            edges: SmallVec::new(),
        }
    }

    fn insert(&mut self, path: &[NodeId], edge_id: EdgeId) {
        if path.is_empty() {
            self.edges.push(edge_id);
            return;
        }

        self.children
            .entry(path[0])
            .or_insert_with(TrieNode::new)
            .insert(&path[1..], edge_id);
    }

    fn get_child(&self, key: NodeId) -> Option<&TrieNode> {
        self.children.get(&key)
    }

    fn children_sorted(&self) -> Vec<NodeId> {
        let mut keys: Vec<_> = self.children.keys().copied().collect();
        keys.sort();
        keys
    }
}

/// The bytes one key of a trie takes in its parent's table: the key and its
/// node, the table's own overhead aside.
pub(crate) const TRIE_ENTRY_BYTES: usize = std::mem::size_of::<(NodeId, TrieNode)>();

/// A trie index for edge patterns in multi-way joins.
///
/// Edges are indexed by path (usually [src, dst]). The trie structure
/// enables efficient intersection of multiple edge sets via leapfrogging.
pub struct TrieIndex {
    /// Root of the trie.
    root: TrieNode,
    /// Number of entries in the trie.
    size: usize,
}

impl TrieIndex {
    /// Creates a new empty trie index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            size: 0,
        }
    }

    /// Inserts an edge into the trie.
    ///
    /// The path typically represents [src, dst] for a directed edge.
    pub fn insert(&mut self, path: &[NodeId], edge_id: EdgeId) {
        self.root.insert(path, edge_id);
        self.size += 1;
    }

    /// Inserts a directed edge (src -> dst).
    pub fn insert_edge(&mut self, src: NodeId, dst: NodeId, edge_id: EdgeId) {
        self.insert(&[src, dst], edge_id);
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the trie is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Creates an iterator at the root level.
    #[allow(clippy::iter_not_returning_iterator)] // TrieIterator is a cursor, not std::iter::Iterator
    pub fn iter(&self) -> TrieIterator<'_> {
        TrieIterator::new(&self.root)
    }

    /// Creates an iterator at a specific path.
    pub fn iter_at(&self, path: &[NodeId]) -> Option<TrieIterator<'_>> {
        let mut node = &self.root;
        for &key in path {
            node = node.get_child(key)?;
        }
        Some(TrieIterator::new(node))
    }

    /// Gets all values at the end of a path.
    pub fn get(&self, path: &[NodeId]) -> Option<&[EdgeId]> {
        let mut node = &self.root;
        for &key in path {
            node = node.get_child(key)?;
        }
        if node.edges.is_empty() {
            None
        } else {
            Some(&node.edges)
        }
    }
}

impl Default for TrieIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// An iterator over trie children at a single level.
pub struct TrieIterator<'a> {
    node: &'a TrieNode,
    keys: Vec<NodeId>,
    pos: usize,
}

impl<'a> TrieIterator<'a> {
    fn new(node: &'a TrieNode) -> Self {
        let keys = node.children_sorted();
        Self { node, keys, pos: 0 }
    }

    /// Returns the current key, if any.
    pub fn key(&self) -> Option<NodeId> {
        self.keys.get(self.pos).copied()
    }

    /// Advances to the next key.
    pub fn next(&mut self) -> bool {
        if self.pos < self.keys.len() {
            self.pos += 1;
            self.pos < self.keys.len()
        } else {
            false
        }
    }

    /// Seeks to the first key >= target.
    ///
    /// Returns true if a key was found.
    pub fn seek(&mut self, target: NodeId) -> bool {
        // Binary search for the target
        match self.keys[self.pos..].binary_search(&target) {
            Ok(offset) => {
                self.pos += offset;
                true
            }
            Err(offset) => {
                self.pos += offset;
                self.pos < self.keys.len()
            }
        }
    }

    /// Opens the current key's child node.
    pub fn open(&self) -> Option<TrieIterator<'a>> {
        let key = self.key()?;
        let child = self.node.get_child(key)?;
        Some(TrieIterator::new(child))
    }

    /// Returns whether the iterator is at a valid position.
    pub fn is_valid(&self) -> bool {
        self.pos < self.keys.len()
    }
}

/// Leapfrog trie join - the workhorse of WCOJ.
///
/// Given multiple sorted iterators, finds their intersection by "leapfrogging"
/// - each iterator jumps ahead to match the maximum of the others. This avoids
///   the Cartesian product explosion of naive binary joins.
pub struct LeapfrogJoin<'a> {
    iters: Vec<TrieIterator<'a>>,
    current_key: Option<NodeId>,
}

impl<'a> LeapfrogJoin<'a> {
    /// Creates a new leapfrog join over the given iterators.
    ///
    /// All iterators must be at the same level (representing the same variable).
    pub fn new(iters: Vec<TrieIterator<'a>>) -> Self {
        let mut join = Self {
            iters,
            current_key: None,
        };
        join.init();
        join
    }

    fn init(&mut self) {
        if self.iters.is_empty() {
            return;
        }

        // Sort iterators by current key
        self.iters.sort_by_key(|it| it.key());

        // Check if all at same position (intersection found)
        self.search();
    }

    fn search(&mut self) {
        if self.iters.is_empty() || !self.iters[0].is_valid() {
            self.current_key = None;
            return;
        }

        loop {
            let max_key = self.iters.last().and_then(|it| it.key());
            let min_key = self.iters.first().and_then(|it| it.key());

            match (min_key, max_key) {
                (Some(min), Some(max)) if min == max => {
                    // All iterators at the same key - found intersection
                    self.current_key = Some(min);
                    return;
                }
                (Some(_), Some(max)) => {
                    // Seek minimum to max
                    if !self.iters[0].seek(max) {
                        self.current_key = None;
                        return;
                    }
                    // Re-sort after seek
                    self.iters.sort_by_key(|it| it.key());
                }
                _ => {
                    self.current_key = None;
                    return;
                }
            }
        }
    }

    /// Returns the current intersection key.
    pub fn key(&self) -> Option<NodeId> {
        self.current_key
    }

    /// Advances to the next intersection.
    pub fn next(&mut self) -> bool {
        if self.current_key.is_none() || self.iters.is_empty() {
            return false;
        }

        // Advance the first iterator
        self.iters[0].next();
        self.iters.sort_by_key(|it| it.key());
        self.search();

        self.current_key.is_some()
    }

    /// Opens the current level and returns iterators for the next level.
    pub fn open(&self) -> Option<Vec<TrieIterator<'a>>> {
        self.current_key?;

        self.iters.iter().map(|it| it.open()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_basic() {
        let mut trie = TrieIndex::new();

        trie.insert_edge(NodeId::new(1), NodeId::new(2), EdgeId::new(0));
        trie.insert_edge(NodeId::new(1), NodeId::new(3), EdgeId::new(1));
        trie.insert_edge(NodeId::new(2), NodeId::new(3), EdgeId::new(2));

        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn test_trie_iterator() {
        let mut trie = TrieIndex::new();

        trie.insert_edge(NodeId::new(1), NodeId::new(10), EdgeId::new(0));
        trie.insert_edge(NodeId::new(2), NodeId::new(20), EdgeId::new(1));
        trie.insert_edge(NodeId::new(3), NodeId::new(30), EdgeId::new(2));

        let mut iter = trie.iter();

        // First level keys should be 1, 2, 3
        assert_eq!(iter.key(), Some(NodeId::new(1)));
        assert!(iter.next());
        assert_eq!(iter.key(), Some(NodeId::new(2)));
        assert!(iter.next());
        assert_eq!(iter.key(), Some(NodeId::new(3)));
        assert!(!iter.next());
    }

    #[test]
    fn test_trie_seek() {
        let mut trie = TrieIndex::new();

        for i in [1, 3, 5, 7, 9] {
            trie.insert_edge(NodeId::new(i), NodeId::new(100), EdgeId::new(i));
        }

        let mut iter = trie.iter();

        // Seek to 4 should land on 5
        assert!(iter.seek(NodeId::new(4)));
        assert_eq!(iter.key(), Some(NodeId::new(5)));

        // Seek to 7 should land on 7
        assert!(iter.seek(NodeId::new(7)));
        assert_eq!(iter.key(), Some(NodeId::new(7)));

        // Seek to 10 should fail (past end)
        assert!(!iter.seek(NodeId::new(10)));
    }

    #[test]
    fn test_leapfrog_join() {
        // Create two tries representing different edge sets
        let mut trie1 = TrieIndex::new();
        let mut trie2 = TrieIndex::new();

        // Trie 1: edges from nodes 1, 2, 3, 5
        for &i in &[1, 2, 3, 5] {
            trie1.insert_edge(NodeId::new(i), NodeId::new(100), EdgeId::new(i));
        }

        // Trie 2: edges from nodes 2, 3, 4, 5
        for &i in &[2, 3, 4, 5] {
            trie2.insert_edge(NodeId::new(i), NodeId::new(100), EdgeId::new(i + 10));
        }

        // Intersection should be {2, 3, 5}
        let iters = vec![trie1.iter(), trie2.iter()];
        let mut join = LeapfrogJoin::new(iters);

        let mut results = Vec::new();
        loop {
            if let Some(key) = join.key() {
                results.push(key);
                if !join.next() {
                    break;
                }
            } else {
                break;
            }
        }

        assert_eq!(results.len(), 3);
        assert!(results.contains(&NodeId::new(2)));
        assert!(results.contains(&NodeId::new(3)));
        assert!(results.contains(&NodeId::new(5)));
    }

    #[test]
    fn test_trie_get_existing_path() {
        let mut trie = TrieIndex::new();
        trie.insert_edge(NodeId::new(1), NodeId::new(2), EdgeId::new(10));
        trie.insert_edge(NodeId::new(1), NodeId::new(3), EdgeId::new(11));

        let edges = trie.get(&[NodeId::new(1), NodeId::new(2)]);
        assert!(edges.is_some());
        assert_eq!(edges.unwrap(), &[EdgeId::new(10)]);
    }

    #[test]
    fn test_trie_get_nonexistent_path() {
        let mut trie = TrieIndex::new();
        trie.insert_edge(NodeId::new(1), NodeId::new(2), EdgeId::new(0));

        assert!(trie.get(&[NodeId::new(99)]).is_none());
        assert!(trie.get(&[NodeId::new(1), NodeId::new(99)]).is_none());
    }

    #[test]
    fn test_trie_get_empty_path() {
        let trie = TrieIndex::new();
        // Empty path returns None since root has no edges
        assert!(trie.get(&[]).is_none());
    }

    #[test]
    fn test_trie_iter_at_existing() {
        let mut trie = TrieIndex::new();
        trie.insert_edge(NodeId::new(1), NodeId::new(2), EdgeId::new(0));
        trie.insert_edge(NodeId::new(1), NodeId::new(3), EdgeId::new(1));

        let iter = trie.iter_at(&[NodeId::new(1)]);
        assert!(iter.is_some());
        let iter = iter.unwrap();
        // Should iterate over children of node 1: keys 2 and 3
        assert_eq!(iter.key(), Some(NodeId::new(2)));
    }

    #[test]
    fn test_trie_iter_at_nonexistent() {
        let trie = TrieIndex::new();
        assert!(trie.iter_at(&[NodeId::new(99)]).is_none());
    }

    #[test]
    fn test_leapfrog_join_open() {
        let mut trie1 = TrieIndex::new();
        let mut trie2 = TrieIndex::new();

        // Both have edges from node 1 to different targets
        trie1.insert_edge(NodeId::new(1), NodeId::new(10), EdgeId::new(0));
        trie1.insert_edge(NodeId::new(1), NodeId::new(20), EdgeId::new(1));
        trie2.insert_edge(NodeId::new(1), NodeId::new(15), EdgeId::new(2));
        trie2.insert_edge(NodeId::new(1), NodeId::new(20), EdgeId::new(3));

        let iters = vec![trie1.iter(), trie2.iter()];
        let join = LeapfrogJoin::new(iters);

        // Current key should be 1 (intersection of first-level keys)
        assert_eq!(join.key(), Some(NodeId::new(1)));

        // Opening should descend into node 1's children
        let child_iters = join.open();
        assert!(child_iters.is_some());
        let child_iters = child_iters.unwrap();
        assert_eq!(child_iters.len(), 2);
    }

    #[test]
    fn test_leapfrog_join_empty_intersection() {
        let mut trie1 = TrieIndex::new();
        let mut trie2 = TrieIndex::new();

        trie1.insert_edge(NodeId::new(1), NodeId::new(10), EdgeId::new(0));
        trie2.insert_edge(NodeId::new(2), NodeId::new(20), EdgeId::new(1));

        let iters = vec![trie1.iter(), trie2.iter()];
        let join = LeapfrogJoin::new(iters);

        // No intersection, key should be None
        assert!(join.key().is_none());
    }

    #[test]
    fn test_trie_seek_backward_stays_forward() {
        let mut trie = TrieIndex::new();
        for i in [1, 3, 5, 7] {
            trie.insert_edge(NodeId::new(i), NodeId::new(100), EdgeId::new(i));
        }

        let mut iter = trie.iter();
        // Advance to 5
        assert!(iter.seek(NodeId::new(5)));
        assert_eq!(iter.key(), Some(NodeId::new(5)));

        // Seeking backward to 1 should not go back (binary search in sorted keys)
        assert!(iter.seek(NodeId::new(5)));
        assert_eq!(iter.key(), Some(NodeId::new(5)));
    }
}
