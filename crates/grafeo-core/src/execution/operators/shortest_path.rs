//! Shortest path operator for finding paths between nodes.
//!
//! This operator computes shortest paths between source and target nodes
//! using BFS for unweighted graphs.

use super::expand::visible_edges_from;
use super::{Operator, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EpochId, LogicalType, NodeId, TransactionId, Value};
use grafeo_common::utils::hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;

/// Operator that finds shortest paths between source and target nodes.
///
/// For each input row containing source and target nodes, this operator
/// computes the shortest path within the hop bounds and outputs its length:
/// one row per shortest path for `allShortestPaths`, one row otherwise, and no
/// row when no path fits (`OPTIONAL MATCH` adds the nulls with a left join).
pub struct ShortestPathOperator {
    /// The graph store.
    store: Arc<dyn GraphStoreSearch>,
    /// Input operator providing source/target node pairs.
    input: Box<dyn Operator>,
    /// Column index of the source node.
    source_column: usize,
    /// Column index of the target node.
    target_column: usize,
    /// Edge type filter (empty means all types).
    edge_types: Vec<String>,
    /// Direction of edge traversal.
    direction: Direction,
    /// Whether to find all shortest paths (vs. just one).
    all_paths: bool,
    /// Minimum number of edges in a path.
    min_hops: u32,
    /// Maximum number of edges in a path (`None` = unbounded).
    max_hops: Option<u32>,
    /// Transaction ID for MVCC visibility (None = use the viewing epoch).
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility (None = every edge in the store).
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
    /// Whether the operator has been exhausted.
    exhausted: bool,
}

impl ShortestPathOperator {
    /// Creates a new shortest path operator.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        source_column: usize,
        target_column: usize,
        edge_types: Vec<String>,
        direction: Direction,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            target_column,
            edge_types,
            direction,
            all_paths: false,
            min_hops: 0,
            max_hops: None,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            exhausted: false,
        }
    }

    /// Sets the transaction context for MVCC visibility: the search then
    /// walks only the edges (and reaches only the nodes) visible at `epoch`
    /// to `transaction_id`, its own uncommitted edges included.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this search as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Sets whether to find all shortest paths.
    pub fn with_all_paths(mut self, all_paths: bool) -> Self {
        self.all_paths = all_paths;
        self
    }

    /// Limits the paths to between `min_hops` and `max_hops` edges (`None` =
    /// unbounded). The default is 0 to unbounded, so a node paired with
    /// itself has a zero-length path.
    pub fn with_hop_bounds(mut self, min_hops: u32, max_hops: Option<u32>) -> Self {
        self.min_hops = min_hops;
        self.max_hops = max_hops;
        self
    }

    /// Finds the shortest path between source and target using BFS.
    /// Returns the path length (number of edges).
    fn find_shortest_path(&self, source: NodeId, target: NodeId) -> Option<i64> {
        if source == target {
            return Some(0);
        }

        let mut visited: FxHashMap<NodeId, i64> = FxHashMap::default();
        let mut queue: VecDeque<(NodeId, i64)> = VecDeque::new();

        visited.insert(source, 0);
        queue.push_back((source, 0));

        while let Some((current, depth)) = queue.pop_front() {
            // Get neighbors based on direction
            let neighbors = self.get_neighbors(current);

            for neighbor in neighbors {
                if neighbor == target {
                    return Some(depth + 1);
                }

                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, depth + 1);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        None // No path found
    }

    /// Finds the shortest paths from `source` to `target` within the hop
    /// bounds: their length and how many there are, or `None` when there is
    /// no such path.
    ///
    /// A path of at least `min_hops` edges starts with exactly `min_hops`
    /// steps, so this first counts those walks per end node, then runs one
    /// breadth-first search from all of their end nodes at once, adding up the
    /// path counts per node. With `min_hops` 0 that is a plain BFS from
    /// `source`; with `min_hops` 1 and `source == target` it finds the
    /// shortest cycles through `source`. A shortest continuation never passes
    /// another end node of the first steps, so each path is counted once.
    fn find_shortest_walks(&self, source: NodeId, target: NodeId) -> Option<(i64, usize)> {
        let min_hops = i64::from(self.min_hops);
        let max_hops = self.max_hops.map(i64::from);
        if max_hops.is_some_and(|max| max < min_hops) {
            return None;
        }

        // Walks of exactly `min_hops` edges, counted per end node
        let mut counts: FxHashMap<NodeId, usize> = FxHashMap::default();
        counts.insert(source, 1);
        for _ in 0..self.min_hops {
            let mut next: FxHashMap<NodeId, usize> = FxHashMap::default();
            for (&node, &count) in &counts {
                for neighbor in self.get_neighbors(node) {
                    let entry = next.entry(neighbor).or_insert(0);
                    *entry = entry.saturating_add(count);
                }
            }
            if next.is_empty() {
                return None;
            }
            counts = next;
        }

        // Level-by-level BFS from all of them; `counts` holds the number of
        // shortest paths to every node reached so far.
        let mut lengths: FxHashMap<NodeId, i64> =
            counts.keys().map(|&node| (node, min_hops)).collect();
        let mut frontier: Vec<NodeId> = counts.keys().copied().collect();
        let mut length = min_hops;
        loop {
            if let Some(&count) = counts.get(&target) {
                return Some((length, count));
            }
            if frontier.is_empty() || max_hops.is_some_and(|max| length >= max) {
                return None;
            }
            length += 1;

            let mut next_frontier = Vec::new();
            for node in frontier {
                let count = counts[&node];
                for neighbor in self.get_neighbors(node) {
                    match lengths.get(&neighbor) {
                        None => {
                            lengths.insert(neighbor, length);
                            counts.insert(neighbor, count);
                            next_frontier.push(neighbor);
                        }
                        Some(&reached_at) if reached_at == length => {
                            let paths = counts
                                .get_mut(&neighbor)
                                .expect("BFS: a reached node has a path count");
                            *paths = paths.saturating_add(count);
                        }
                        // Already reached by a shorter path
                        Some(_) => {}
                    }
                }
            }
            frontier = next_frontier;
        }
    }

    /// Finds the length of one shortest path from `source` to `target` within
    /// the hop bounds.
    fn find_one_shortest_path(&self, source: NodeId, target: NodeId) -> Option<i64> {
        // Without a minimum, or with a minimum of one hop between two different
        // nodes, the plain shortest path has enough hops.
        let plain = self.min_hops == 0 || (self.min_hops == 1 && source != target);
        if !plain {
            return self
                .find_shortest_walks(source, target)
                .map(|(length, _)| length);
        }
        self.find_shortest_path_bidirectional(source, target)
            .filter(|&length| self.max_hops.is_none_or(|max| length <= i64::from(max)))
    }

    /// The lengths of the shortest paths from `source` to `target`: one per
    /// path when finding all shortest paths, otherwise at most one, and none
    /// when no path fits the hop bounds.
    fn path_lengths(&self, source: NodeId, target: NodeId) -> Vec<i64> {
        if self.all_paths {
            self.find_shortest_walks(source, target)
                .map_or_else(Vec::new, |(length, count)| vec![length; count])
        } else {
            self.find_one_shortest_path(source, target)
                .into_iter()
                .collect()
        }
    }

    /// Gets neighbors of a node in a specific direction, respecting the edge
    /// type filter and visibility (see [`visible_edges_from`]).
    ///
    /// This is the direction-parameterized variant used by bidirectional BFS
    /// to traverse the forward and backward frontiers independently.
    fn get_neighbors_directed(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        visible_edges_from(
            self.store.as_ref(),
            node,
            direction,
            &self.edge_types,
            self.viewing_epoch,
            self.transaction_id,
            self.read_only,
        )
        .into_iter()
        .map(|(target, _)| target)
        .collect()
    }

    /// Gets neighbors of a node respecting edge type filter and direction.
    fn get_neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.get_neighbors_directed(node, self.direction)
    }

    /// Finds shortest path using bidirectional BFS.
    ///
    /// Maintains forward and backward frontiers, alternating expansion of the
    /// smaller one. When a node is found in both visited sets, the shortest
    /// path is `forward_depth + backward_depth`. This reduces the search space
    /// from O(b^d) to O(b^(d/2)) where b is the branching factor and d is
    /// the path length.
    ///
    /// Falls back to unidirectional BFS if backward adjacency is unavailable.
    fn find_shortest_path_bidirectional(&self, source: NodeId, target: NodeId) -> Option<i64> {
        if source == target {
            return Some(0);
        }

        // Fall back to unidirectional if backward adjacency is unavailable
        if !self.store.has_backward_adjacency() {
            return self.find_shortest_path(source, target);
        }

        let reverse_dir = self.direction.reverse();

        // Forward BFS state
        let mut forward_visited: FxHashMap<NodeId, i64> = FxHashMap::default();
        let mut forward_queue: VecDeque<(NodeId, i64)> = VecDeque::new();
        forward_visited.insert(source, 0);
        forward_queue.push_back((source, 0));

        // Backward BFS state
        let mut backward_visited: FxHashMap<NodeId, i64> = FxHashMap::default();
        let mut backward_queue: VecDeque<(NodeId, i64)> = VecDeque::new();
        backward_visited.insert(target, 0);
        backward_queue.push_back((target, 0));

        // Best known path length (upper bound)
        let mut best: Option<i64> = None;

        loop {
            // Decide which frontier to expand, or stop
            let expand_forward = match (forward_queue.front(), backward_queue.front()) {
                (Some(_), Some(_)) => forward_queue.len() <= backward_queue.len(),
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };

            if expand_forward {
                let Some((current, depth)) = forward_queue.pop_front() else {
                    break;
                };

                // If this depth alone exceeds best, this frontier is exhausted
                if let Some(b) = best
                    && depth + 1 > b
                {
                    // Clear the queue; no further expansion can improve
                    forward_queue.clear();
                    continue;
                }

                for neighbor in self.get_neighbors_directed(current, self.direction) {
                    let new_depth = depth + 1;

                    // Check if backward frontier already visited this node
                    if let Some(&backward_depth) = backward_visited.get(&neighbor) {
                        let total = new_depth + backward_depth;
                        best = Some(best.map_or(total, |b: i64| b.min(total)));
                    }

                    if !forward_visited.contains_key(&neighbor) {
                        forward_visited.insert(neighbor, new_depth);
                        if best.is_none_or(|b| new_depth < b) {
                            forward_queue.push_back((neighbor, new_depth));
                        }
                    }
                }
            } else {
                let Some((current, depth)) = backward_queue.pop_front() else {
                    break;
                };

                if let Some(b) = best
                    && depth + 1 > b
                {
                    backward_queue.clear();
                    continue;
                }

                for neighbor in self.get_neighbors_directed(current, reverse_dir) {
                    let new_depth = depth + 1;

                    // Check if forward frontier already visited this node
                    if let Some(&forward_depth) = forward_visited.get(&neighbor) {
                        let total = forward_depth + new_depth;
                        best = Some(best.map_or(total, |b: i64| b.min(total)));
                    }

                    if !backward_visited.contains_key(&neighbor) {
                        backward_visited.insert(neighbor, new_depth);
                        if best.is_none_or(|b| new_depth < b) {
                            backward_queue.push_back((neighbor, new_depth));
                        }
                    }
                }
            }
        }

        best
    }
}

impl Operator for ShortestPathOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // A pair without a path has no row, so a whole chunk can produce
        // nothing: keep reading until one produces rows or the input ends.
        loop {
            let Some(input_chunk) = self.input.next()? else {
                self.exhausted = true;
                return Ok(None);
            };

            // Build output: input columns + path length
            let num_input_cols = input_chunk.column_count();
            let mut output_schema: Vec<LogicalType> = (0..num_input_cols)
                .map(|i| {
                    input_chunk
                        .column(i)
                        .map_or(LogicalType::Any, |c| c.data_type().clone())
                })
                .collect();
            output_schema.push(LogicalType::Any); // Path column (stores length as int)

            // For allShortestPaths, we may need more rows than input
            let initial_capacity = if self.all_paths {
                input_chunk.row_count() * 4 // Estimate 4x for multiple paths
            } else {
                input_chunk.row_count()
            };
            let mut builder = DataChunkBuilder::with_capacity(&output_schema, initial_capacity);

            for row in input_chunk.selected_indices() {
                // Get source and target nodes
                let source = input_chunk
                    .column(self.source_column)
                    .and_then(|c| c.get_node_id(row));
                let target = input_chunk
                    .column(self.target_column)
                    .and_then(|c| c.get_node_id(row));

                // A null endpoint (from an earlier OPTIONAL MATCH) has no path
                let path_lengths = match (source, target) {
                    (Some(s), Some(t)) => self.path_lengths(s, t),
                    _ => Vec::new(),
                };

                // Output one row per path
                for path_length in path_lengths {
                    // Copy input columns
                    for col_idx in 0..num_input_cols {
                        if let Some(in_col) = input_chunk.column(col_idx)
                            && let Some(out_col) = builder.column_mut(col_idx)
                        {
                            if let Some(node_id) = in_col.get_node_id(row) {
                                out_col.push_node_id(node_id);
                            } else if let Some(edge_id) = in_col.get_edge_id(row) {
                                out_col.push_edge_id(edge_id);
                            } else if let Some(value) = in_col.get_value(row) {
                                out_col.push_value(value);
                            } else {
                                out_col.push_value(Value::Null);
                            }
                        }
                    }

                    // Add path length column
                    if let Some(out_col) = builder.column_mut(num_input_cols) {
                        out_col.push_value(Value::Int64(path_length));
                    }

                    builder.advance_row();
                }
            }

            let chunk = builder.finish();
            if chunk.row_count() > 0 {
                return Ok(Some(chunk));
            }
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "ShortestPath"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

    /// A mock operator that returns source/target node pairs, one chunk per
    /// inner vector (empty ones are skipped).
    struct MockPairOperator {
        chunks: Vec<Vec<(NodeId, NodeId)>>,
        position: usize,
    }

    impl MockPairOperator {
        fn new(pairs: Vec<(NodeId, NodeId)>) -> Self {
            Self::chunks(vec![pairs])
        }

        fn chunks(chunks: Vec<Vec<(NodeId, NodeId)>>) -> Self {
            Self {
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockPairOperator {
        fn next(&mut self) -> OperatorResult {
            while let Some(pairs) = self.chunks.get(self.position) {
                self.position += 1;
                if pairs.is_empty() {
                    continue;
                }

                let schema = vec![LogicalType::Node, LogicalType::Node];
                let mut builder = DataChunkBuilder::with_capacity(&schema, pairs.len());
                for (source, target) in pairs {
                    builder.column_mut(0).unwrap().push_node_id(*source);
                    builder.column_mut(1).unwrap().push_node_id(*target);
                    builder.advance_row();
                }
                return Ok(Some(builder.finish()));
            }
            Ok(None)
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "MockPair"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// Runs the operator to the end: one `(source, target, length)` per row.
    fn rows(op: &mut ShortestPathOperator) -> Vec<(NodeId, NodeId, Value)> {
        let mut rows = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in 0..chunk.row_count() {
                rows.push((
                    chunk.column(0).unwrap().get_node_id(row).unwrap(),
                    chunk.column(1).unwrap().get_node_id(row).unwrap(),
                    chunk.column(2).unwrap().get_value(row).unwrap(),
                ));
            }
        }
        rows
    }

    /// Searches outgoing paths of any edge type for the pairs in `chunks`.
    fn search(
        store: &Arc<LpgStore>,
        chunks: Vec<Vec<(NodeId, NodeId)>>,
        all_paths: bool,
        (min_hops, max_hops): (u32, Option<u32>),
    ) -> Vec<(NodeId, NodeId, Value)> {
        let mut op = ShortestPathOperator::new(
            Arc::clone(store) as Arc<dyn GraphStoreSearch>,
            Box::new(MockPairOperator::chunks(chunks)),
            0,
            1,
            vec![],
            Direction::Outgoing,
        )
        .with_all_paths(all_paths)
        .with_hop_bounds(min_hops, max_hops);
        rows(&mut op)
    }

    #[test]
    fn test_find_shortest_path_direct() {
        let store = Arc::new(LpgStore::new().unwrap());

        // a -> b (1 hop)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0, // source column
            1, // target column
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Path length should be 1
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(1));
    }

    #[test]
    fn test_find_shortest_path_same_node() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, a)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Path length should be 0 (same node)
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(0));
    }

    #[test]
    fn test_find_shortest_path_two_hops() {
        let store = Arc::new(LpgStore::new().unwrap());

        // a -> b -> c (2 hops from a to c)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(2));
    }

    #[test]
    fn test_find_shortest_path_no_path() {
        let store = Arc::new(LpgStore::new().unwrap());

        // a and b are disconnected
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        // No path, no row: MATCH drops the pair, OPTIONAL MATCH adds the nulls
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_find_shortest_path_prefers_shorter() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create two paths: a -> d (1 hop) and a -> b -> c -> d (3 hops)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        // Long path
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        // Short path (direct)
        store.create_edge(a, d, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(1)); // Should find direct path
    }

    #[test]
    fn test_find_shortest_path_with_edge_type_filter() {
        let store = Arc::new(LpgStore::new().unwrap());

        // a -KNOWS-> b -LIKES-> c
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "LIKES");

        // Path with KNOWS filter should only reach b, not c
        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec!["KNOWS".to_string()],
            Direction::Outgoing,
        );

        assert!(op.next().unwrap().is_none()); // Can't reach c via KNOWS only
    }

    #[test]
    fn test_all_shortest_paths_single_path() {
        let store = Arc::new(LpgStore::new().unwrap());

        // a -> b (single path)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        )
        .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1); // Only one path exists
    }

    #[test]
    fn test_all_shortest_paths_multiple_paths() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create diamond: a -> b -> d and a -> c -> d (two paths of length 2)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, d, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        )
        .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        // Should return 2 rows (two paths of length 2)
        assert_eq!(chunk.row_count(), 2);

        // Both should have length 2
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
        assert_eq!(path_col.get_value(1).unwrap(), Value::Int64(2));
    }

    #[test]
    fn test_multiple_pairs_in_chunk() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create: a -> b, c -> d
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        // Test multiple pairs at once
        let input = Box::new(MockPairOperator::new(vec![(a, b), (c, d), (a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        // a->d has no path, so it has no row
        assert_eq!(
            rows(&mut op),
            vec![(a, b, Value::Int64(1)), (c, d, Value::Int64(1))]
        );
    }

    #[test]
    fn test_operator_reset() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        // First iteration
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());

        // After reset
        op.reset();
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
    }

    #[test]
    fn test_operator_name() {
        let store = Arc::new(LpgStore::new().unwrap());
        let input = Box::new(MockPairOperator::new(vec![]));
        let op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        assert_eq!(op.name(), "ShortestPath");
    }

    #[test]
    fn test_empty_input() {
        let store = Arc::new(LpgStore::new().unwrap());
        let input = Box::new(MockPairOperator::new(vec![]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        // Empty input should return None
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_all_shortest_paths_no_path() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Disconnected nodes
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        )
        .with_all_paths(true);

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_all_shortest_paths_same_node() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, a)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        )
        .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(0));
    }

    // === Bidirectional BFS Tests ===

    #[test]
    fn test_bidirectional_bfs_long_chain() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Chain: n0 -> n1 -> n2 -> ... -> n9 (9 hops)
        let nodes: Vec<NodeId> = (0..10).map(|_| store.create_node(&["Node"])).collect();
        for i in 0..9 {
            store.create_edge(nodes[i], nodes[i + 1], "NEXT");
        }

        let input = Box::new(MockPairOperator::new(vec![(nodes[0], nodes[9])]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(9));
    }

    #[test]
    fn test_bidirectional_bfs_diamond() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Diamond: a -> b -> d, a -> c -> d (two paths of length 2)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, d, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
    }

    #[test]
    fn test_bidirectional_bfs_no_path() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        // No edges between a and b

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_bidirectional_bfs_same_node() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, a)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(0));
    }

    #[test]
    fn test_bidirectional_bfs_prefers_shorter() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        // Long path: a -> b -> c -> d (3 hops)
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, d, "KNOWS");
        // Short path: a -> d (1 hop)
        store.create_edge(a, d, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(1));
    }

    #[test]
    fn test_bidirectional_bfs_with_edge_type_filter() {
        let store = Arc::new(LpgStore::new().unwrap());

        // a -KNOWS-> b -LIKES-> c
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "LIKES");

        // Only KNOWS edges: a can reach b but not c
        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec!["KNOWS".to_string()],
            Direction::Outgoing,
        );

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_bidirectional_bfs_has_backward_adjacency() {
        // Default store has backward adjacency enabled
        let store = Arc::new(LpgStore::new().unwrap());
        assert!(store.has_backward_adjacency());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        // Bidirectional BFS should work and find shortest path
        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
    }

    #[test]
    fn test_shortest_path_into_any() {
        let store = Arc::new(LpgStore::new().unwrap());
        let input = Box::new(MockPairOperator::new(vec![]));
        let op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStoreSearch>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ShortestPathOperator>().is_ok());
    }

    // === Pairs without a path and hop bounds (#514) ===

    #[test]
    fn a_chunk_without_paths_does_not_end_the_output() {
        // a -> b; c is isolated. No pair in the first input chunk has a path,
        // the pair in the second one does.
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        for all_paths in [false, true] {
            let chunks = vec![vec![(a, c), (c, b)], vec![(a, b)]];
            assert_eq!(
                search(&store, chunks, all_paths, (0, None)),
                vec![(a, b, Value::Int64(1))],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn zero_hop_minimum_keeps_the_zero_length_self_path() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);

        for all_paths in [false, true] {
            assert_eq!(
                search(&store, vec![vec![(a, a)]], all_paths, (0, None)),
                vec![(a, a, Value::Int64(0))],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn one_hop_minimum_excludes_the_zero_length_self_path() {
        // a has no edges, so no path of one or more hops leads back to it
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);

        for all_paths in [false, true] {
            assert_eq!(
                search(&store, vec![vec![(a, a)]], all_paths, (1, None)),
                vec![],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn one_hop_minimum_finds_the_shortest_cycle() {
        // a -> b -> a (2 hops) and a -> c -> d -> a (3 hops)
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, a, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(c, d, "KNOWS");
        store.create_edge(d, a, "KNOWS");

        for all_paths in [false, true] {
            assert_eq!(
                search(&store, vec![vec![(a, a)]], all_paths, (1, None)),
                vec![(a, a, Value::Int64(2))],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn a_self_loop_is_a_one_hop_cycle() {
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);
        store.create_edge(a, a, "KNOWS");

        for all_paths in [false, true] {
            assert_eq!(
                search(&store, vec![vec![(a, a)]], all_paths, (1, None)),
                vec![(a, a, Value::Int64(1))],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn all_shortest_cycles_are_returned() {
        // a -> b -> d -> a and a -> c -> d -> a: two shortest cycles of 3 hops
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, d, "KNOWS");
        store.create_edge(c, d, "KNOWS");
        store.create_edge(d, a, "KNOWS");

        assert_eq!(
            search(&store, vec![vec![(a, a)]], true, (1, None)),
            vec![(a, a, Value::Int64(3)); 2]
        );
        assert_eq!(
            search(&store, vec![vec![(a, a)]], false, (1, None)),
            vec![(a, a, Value::Int64(3))]
        );
    }

    #[test]
    fn minimum_above_the_shortest_distance() {
        // Triangle a -> b -> c -> a: b is 1 hop from a, and 4 hops when the
        // path has to go around the triangle once more.
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");

        for all_paths in [false, true] {
            let pairs = || vec![vec![(a, b)]];
            assert_eq!(
                search(&store, pairs(), all_paths, (1, None)),
                vec![(a, b, Value::Int64(1))],
                "all_paths: {all_paths}"
            );
            assert_eq!(
                search(&store, pairs(), all_paths, (2, None)),
                vec![(a, b, Value::Int64(4))],
                "all_paths: {all_paths}"
            );
            assert_eq!(
                search(&store, pairs(), all_paths, (2, Some(3))),
                vec![],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn maximum_drops_longer_paths() {
        // a -> b -> c
        let store = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        for all_paths in [false, true] {
            let pairs = || vec![vec![(a, b), (a, c)]];
            assert_eq!(
                search(&store, pairs(), all_paths, (1, Some(1))),
                vec![(a, b, Value::Int64(1))],
                "all_paths: {all_paths}"
            );
            assert_eq!(
                search(&store, pairs(), all_paths, (1, Some(2))),
                vec![(a, b, Value::Int64(1)), (a, c, Value::Int64(2))],
                "all_paths: {all_paths}"
            );
        }
    }
}
