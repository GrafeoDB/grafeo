//! Shortest path operator for finding paths between nodes.
//!
//! This operator computes shortest paths between source and target nodes
//! using BFS for unweighted graphs.

use super::{Operator, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use crate::graph::Direction;
use crate::graph::lpg::LpgStore;
use grafeo_common::types::{LogicalType, NodeId, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Operator that finds shortest paths between source and target nodes.
///
/// For each input row containing source and target nodes, this operator
/// computes the shortest path and outputs the path as a value.
pub struct ShortestPathOperator {
    /// The graph store.
    store: Arc<LpgStore>,
    /// Input operator providing source/target node pairs.
    input: Box<dyn Operator>,
    /// Column index of the source node.
    source_column: usize,
    /// Column index of the target node.
    target_column: usize,
    /// Optional edge type filter.
    edge_type: Option<String>,
    /// Direction of edge traversal.
    direction: Direction,
    /// Whether to find all shortest paths (vs. just one).
    all_paths: bool,
    /// Whether the operator has been exhausted.
    exhausted: bool,
}

impl ShortestPathOperator {
    /// Creates a new shortest path operator.
    pub fn new(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        target_column: usize,
        edge_type: Option<String>,
        direction: Direction,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            target_column,
            edge_type,
            direction,
            all_paths: false,
            exhausted: false,
        }
    }

    /// Sets whether to find all shortest paths.
    pub fn with_all_paths(mut self, all_paths: bool) -> Self {
        self.all_paths = all_paths;
        self
    }

    /// Finds the shortest path between source and target using BFS.
    /// Returns the path length (number of edges).
    fn find_shortest_path(&self, source: NodeId, target: NodeId) -> Option<i64> {
        if source == target {
            return Some(0);
        }

        let mut visited: HashMap<NodeId, i64> = HashMap::new();
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

    /// Finds all shortest paths between source and target using BFS.
    /// Returns a vector of path lengths (all will be the same minimum length).
    /// For allShortestPaths, we return the count of paths with minimum length.
    fn find_all_shortest_paths(&self, source: NodeId, target: NodeId) -> Vec<i64> {
        if source == target {
            return vec![0];
        }

        // BFS that tracks number of paths to each node at each depth
        let mut distances: HashMap<NodeId, i64> = HashMap::new();
        let mut path_counts: HashMap<NodeId, usize> = HashMap::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();

        distances.insert(source, 0);
        path_counts.insert(source, 1);
        queue.push_back(source);

        let mut target_depth: Option<i64> = None;
        let mut target_path_count = 0;

        while let Some(current) = queue.pop_front() {
            let current_depth = *distances.get(&current).unwrap();
            let current_paths = *path_counts.get(&current).unwrap();

            // If we've found target and we're past its depth, stop
            if let Some(td) = target_depth {
                if current_depth >= td {
                    continue;
                }
            }

            for neighbor in self.get_neighbors(current) {
                let new_depth = current_depth + 1;

                if neighbor == target {
                    // Found target
                    if target_depth.is_none() {
                        target_depth = Some(new_depth);
                        target_path_count = current_paths;
                    } else if Some(new_depth) == target_depth {
                        target_path_count += current_paths;
                    }
                    continue;
                }

                // If not visited or same depth (for counting all paths)
                if let Some(&existing_depth) = distances.get(&neighbor) {
                    if existing_depth == new_depth {
                        // Same depth, add to path count
                        *path_counts.get_mut(&neighbor).unwrap() += current_paths;
                    }
                    // If existing_depth < new_depth, skip (already processed at shorter distance)
                } else {
                    // New node
                    distances.insert(neighbor, new_depth);
                    path_counts.insert(neighbor, current_paths);
                    queue.push_back(neighbor);
                }
            }
        }

        // Return one entry per path
        if let Some(depth) = target_depth {
            vec![depth; target_path_count]
        } else {
            vec![]
        }
    }

    /// Gets neighbors of a node respecting edge type filter and direction.
    fn get_neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.store
            .edges_from(node, self.direction)
            .filter(|(_target, edge_id)| {
                // Filter by edge type if specified
                if let Some(ref filter_type) = self.edge_type {
                    if let Some(edge_type) = self.store.edge_type(*edge_id) {
                        edge_type
                            .as_str()
                            .eq_ignore_ascii_case(filter_type.as_str())
                    } else {
                        false
                    }
                } else {
                    true
                }
            })
            .map(|(target, _)| target)
            .collect()
    }
}

impl Operator for ShortestPathOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // Get input chunk
        let input_chunk = match self.input.next()? {
            Some(chunk) => chunk,
            None => {
                self.exhausted = true;
                return Ok(None);
            }
        };

        // Build output: input columns + path length
        let num_input_cols = input_chunk.column_count();
        let mut output_schema: Vec<LogicalType> = (0..num_input_cols)
            .map(|i| {
                input_chunk
                    .column(i)
                    .map(|c| c.data_type().clone())
                    .unwrap_or(LogicalType::Any)
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

            // Compute shortest path(s)
            let path_lengths: Vec<Option<i64>> = match (source, target) {
                (Some(s), Some(t)) => {
                    if self.all_paths {
                        let paths = self.find_all_shortest_paths(s, t);
                        if paths.is_empty() {
                            vec![None] // No path found, still output one row with null
                        } else {
                            paths.into_iter().map(Some).collect()
                        }
                    } else {
                        vec![self.find_shortest_path(s, t)]
                    }
                }
                _ => vec![None],
            };

            // Output one row per path
            for path_length in path_lengths {
                // Copy input columns
                for col_idx in 0..num_input_cols {
                    if let Some(in_col) = input_chunk.column(col_idx) {
                        if let Some(out_col) = builder.column_mut(col_idx) {
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
                }

                // Add path length column
                if let Some(out_col) = builder.column_mut(num_input_cols) {
                    match path_length {
                        Some(len) => out_col.push_value(Value::Int64(len)),
                        None => out_col.push_value(Value::Null),
                    }
                }

                builder.advance_row();
            }
        }

        let chunk = builder.finish();
        if chunk.row_count() > 0 {
            Ok(Some(chunk))
        } else {
            self.exhausted = true;
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "ShortestPath"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock operator that returns a single chunk with source/target node pairs.
    struct MockPairOperator {
        pairs: Vec<(NodeId, NodeId)>,
        exhausted: bool,
    }

    impl MockPairOperator {
        fn new(pairs: Vec<(NodeId, NodeId)>) -> Self {
            Self {
                pairs,
                exhausted: false,
            }
        }
    }

    impl Operator for MockPairOperator {
        fn next(&mut self) -> OperatorResult {
            if self.exhausted || self.pairs.is_empty() {
                return Ok(None);
            }
            self.exhausted = true;

            let schema = vec![LogicalType::Node, LogicalType::Node];
            let mut builder = DataChunkBuilder::with_capacity(&schema, self.pairs.len());

            for (source, target) in &self.pairs {
                builder.column_mut(0).unwrap().push_node_id(*source);
                builder.column_mut(1).unwrap().push_node_id(*target);
                builder.advance_row();
            }

            Ok(Some(builder.finish()))
        }

        fn reset(&mut self) {
            self.exhausted = false;
        }

        fn name(&self) -> &'static str {
            "MockPair"
        }
    }

    #[test]
    fn test_find_shortest_path_direct() {
        let store = Arc::new(LpgStore::new());

        // a -> b (1 hop)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            Arc::clone(&store),
            input,
            0, // source column
            1, // target column
            None,
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
        let store = Arc::new(LpgStore::new());
        let a = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, a)]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Path length should be 0 (same node)
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(0));
    }

    #[test]
    fn test_find_shortest_path_two_hops() {
        let store = Arc::new(LpgStore::new());

        // a -> b -> c (2 hops from a to c)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(2));
    }

    #[test]
    fn test_find_shortest_path_no_path() {
        let store = Arc::new(LpgStore::new());

        // a and b are disconnected
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Path length should be null (no path)
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Null);
    }

    #[test]
    fn test_find_shortest_path_prefers_shorter() {
        let store = Arc::new(LpgStore::new());

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
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing);

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(1)); // Should find direct path
    }

    #[test]
    fn test_find_shortest_path_with_edge_type_filter() {
        let store = Arc::new(LpgStore::new());

        // a -KNOWS-> b -LIKES-> c
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "LIKES");

        // Path with KNOWS filter should only reach b, not c
        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            Arc::clone(&store),
            input,
            0,
            1,
            Some("KNOWS".to_string()),
            Direction::Outgoing,
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Null); // Can't reach c via KNOWS only
    }

    #[test]
    fn test_all_shortest_paths_single_path() {
        let store = Arc::new(LpgStore::new());

        // a -> b (single path)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing)
                .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1); // Only one path exists
    }

    #[test]
    fn test_all_shortest_paths_multiple_paths() {
        let store = Arc::new(LpgStore::new());

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
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing)
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
        let store = Arc::new(LpgStore::new());

        // Create: a -> b, c -> d
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        // Test multiple pairs at once
        let input = Box::new(MockPairOperator::new(vec![(a, b), (c, d), (a, d)]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 3);

        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(1)); // a->b = 1
        assert_eq!(path_col.get_value(1).unwrap(), Value::Int64(1)); // c->d = 1
        assert_eq!(path_col.get_value(2).unwrap(), Value::Null); // a->d = no path
    }

    #[test]
    fn test_operator_reset() {
        let store = Arc::new(LpgStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing);

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
        let store = Arc::new(LpgStore::new());
        let input = Box::new(MockPairOperator::new(vec![]));
        let op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing);

        assert_eq!(op.name(), "ShortestPath");
    }

    #[test]
    fn test_empty_input() {
        let store = Arc::new(LpgStore::new());
        let input = Box::new(MockPairOperator::new(vec![]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing);

        // Empty input should return None
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_all_shortest_paths_no_path() {
        let store = Arc::new(LpgStore::new());

        // Disconnected nodes
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing)
                .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1); // Still returns one row with null

        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Null);
    }

    #[test]
    fn test_all_shortest_paths_same_node() {
        let store = Arc::new(LpgStore::new());
        let a = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, a)]));
        let mut op =
            ShortestPathOperator::new(Arc::clone(&store), input, 0, 1, None, Direction::Outgoing)
                .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(0));
    }
}
