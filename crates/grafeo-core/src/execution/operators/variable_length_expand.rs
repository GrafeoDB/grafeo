//! Variable-length expand operator for multi-hop path traversal.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TransactionId};
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

/// Path traversal mode controlling which paths are allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PathMode {
    /// Allows repeated nodes and edges (default).
    #[default]
    Walk,
    /// No repeated edges in a path.
    Trail,
    /// No repeated nodes except the start and end may be equal.
    Simple,
    /// No repeated nodes at all.
    Acyclic,
}

/// An expand operator that handles variable-length path patterns like `*1..3`.
///
/// For each input row containing a source node, this operator produces
/// output rows for each neighbor reachable within the hop range.
#[allow(clippy::struct_excessive_bools)]
pub struct VariableLengthExpandOperator {
    /// The store to traverse.
    store: Arc<dyn GraphStoreSearch>,
    /// Input operator providing source nodes.
    input: Box<dyn Operator>,
    /// Index of the source node column in input.
    source_column: usize,
    /// Direction of edge traversal.
    direction: Direction,
    /// Edge type filter (empty = match all types, multiple = match any).
    edge_types: Vec<String>,
    /// Minimum number of hops.
    min_hops: u32,
    /// Maximum number of hops.
    max_hops: u32,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Transaction ID for MVCC visibility.
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
    /// When true, skip versioned MVCC lookups (fast path for read-only queries).
    read_only: bool,
    /// Materialized input rows.
    input_rows: Option<Vec<InputRow>>,
    /// Current input row index.
    current_input_idx: usize,
    /// Output buffer for pending results.
    output_buffer: Vec<OutputRow>,
    /// Whether the operator is exhausted.
    exhausted: bool,
    /// Whether to output path length as an additional column.
    output_path_length: bool,
    /// Whether to output full path detail (node list and edge list).
    output_path_detail: bool,
    /// Whether the edge column holds the path's edges as a list instead of
    /// its last edge (the variable of a variable-length edge pattern).
    output_edge_list: bool,
    /// Path traversal mode (WALK, TRAIL, SIMPLE, ACYCLIC).
    path_mode: PathMode,
}

/// A materialized input row.
struct InputRow {
    /// All column values from the input.
    columns: Vec<ColumnValue>,
    /// The source node ID for expansion.
    source_node: NodeId,
}

/// A column value that can be node ID, edge ID, or generic value.
#[derive(Clone)]
enum ColumnValue {
    NodeId(NodeId),
    EdgeId(EdgeId),
    Value(grafeo_common::types::Value),
}

/// A ready output row.
struct OutputRow {
    /// Index into input_rows for the source row.
    input_idx: usize,
    /// The final edge in the path (`None` for zero-length paths).
    edge_id: Option<EdgeId>,
    /// The target node.
    target_id: NodeId,
    /// The path length (number of edges/hops).
    path_length: u32,
    /// All nodes along the path (source through target), populated when tracking.
    path_nodes: Option<Vec<NodeId>>,
    /// All edges along the path, populated when tracking.
    path_edges: Option<Vec<EdgeId>>,
}

/// A shared-prefix path segment for efficient BFS path tracking.
///
/// Instead of cloning entire `Vec<NodeId>` / `Vec<EdgeId>` at each BFS expansion
/// step (O(depth) per clone), segments form an `Rc`-linked list that shares common
/// prefixes. Expansion costs O(1) (one `Rc::clone` + one allocation). Full paths
/// are only materialized when emitting output rows.
struct PathSegment {
    /// The node at this position in the path.
    node: NodeId,
    /// The edge taken to reach this node. `None` for the source/root node.
    edge: Option<EdgeId>,
    /// Parent segment, or `None` for the root.
    parent: Option<Rc<PathSegment>>,
}

impl PathSegment {
    /// Materializes the full node path from root to this segment.
    fn collect_nodes(&self, depth: u32) -> Vec<NodeId> {
        let mut nodes = Vec::with_capacity(depth as usize + 1);
        self.collect_nodes_into(&mut nodes);
        nodes
    }

    fn collect_nodes_into(&self, nodes: &mut Vec<NodeId>) {
        if let Some(parent) = &self.parent {
            parent.collect_nodes_into(nodes);
        }
        nodes.push(self.node);
    }

    /// Materializes the full edge path from root to this segment.
    fn collect_edges(&self, depth: u32) -> Vec<EdgeId> {
        let mut edges = Vec::with_capacity(depth as usize);
        self.collect_edges_into(&mut edges);
        edges
    }

    fn collect_edges_into(&self, edges: &mut Vec<EdgeId>) {
        if let Some(parent) = &self.parent {
            parent.collect_edges_into(edges);
        }
        if let Some(edge) = self.edge {
            edges.push(edge);
        }
    }

    /// Checks whether a node already appears in this path segment chain.
    fn contains_node(&self, target: NodeId) -> bool {
        if self.node == target {
            return true;
        }
        if let Some(parent) = &self.parent {
            return parent.contains_node(target);
        }
        false
    }

    /// Checks whether an edge already appears in this path segment chain.
    fn contains_edge(&self, target: EdgeId) -> bool {
        if self.edge == Some(target) {
            return true;
        }
        if let Some(parent) = &self.parent {
            return parent.contains_edge(target);
        }
        false
    }
}

impl VariableLengthExpandOperator {
    /// Creates a new variable-length expand operator.
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
        min_hops: u32,
        max_hops: u32,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_types,
            min_hops,
            max_hops: max_hops.max(min_hops), // Ensure max >= min
            chunk_capacity: 2048,
            transaction_id: None,
            viewing_epoch: None,
            read_only: false,
            input_rows: None,
            current_input_idx: 0,
            output_buffer: Vec::new(),
            exhausted: false,
            output_path_length: false,
            output_path_detail: false,
            output_edge_list: false,
            path_mode: PathMode::Walk,
        }
    }

    /// Sets the path traversal mode.
    pub fn with_path_mode(mut self, mode: PathMode) -> Self {
        self.path_mode = mode;
        self
    }

    /// Enables path length output as an additional column.
    pub fn with_path_length_output(mut self) -> Self {
        self.output_path_length = true;
        self
    }

    /// Enables full path detail output (node list and edge list columns).
    pub fn with_path_detail_output(mut self) -> Self {
        self.output_path_detail = true;
        self
    }

    /// Makes the edge column hold every edge of the path, in order, as a
    /// list of edge ids (typed `List(Edge)`), which is what the variable of a
    /// variable-length edge pattern binds to.
    pub fn with_edge_list_output(mut self) -> Self {
        self.output_edge_list = true;
        self
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Sets the transaction context for MVCC visibility.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Marks this expand as read-only, enabling fast-path lookups.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Materializes all input rows.
    fn materialize_input(&mut self) -> Result<(), OperatorError> {
        let mut rows = Vec::new();

        /// Minimum chunk size for locality sort to be worthwhile.
        const LOCALITY_SORT_THRESHOLD: usize = 1024;

        while let Some(mut chunk) = self.input.next()? {
            // Flatten to handle selection vectors
            chunk.flatten();
            // Sort by source node ID for cache locality during adjacency lookups
            if chunk.len() > LOCALITY_SORT_THRESHOLD {
                chunk = chunk.sort_by_column(self.source_column);
            }

            for row_idx in 0..chunk.row_count() {
                // Extract the source node ID
                let col = chunk.column(self.source_column).ok_or_else(|| {
                    OperatorError::ColumnNotFound(format!(
                        "Column {} not found",
                        self.source_column
                    ))
                })?;

                let source_node = col.get_node_id(row_idx).ok_or_else(|| {
                    OperatorError::Execution("Expected node ID in source column".into())
                })?;

                // Materialize all columns
                let mut columns = Vec::with_capacity(chunk.column_count());
                for col_idx in 0..chunk.column_count() {
                    let col = chunk
                        .column(col_idx)
                        .expect("col_idx within column_count range");
                    let value = if let Some(node_id) = col.get_node_id(row_idx) {
                        ColumnValue::NodeId(node_id)
                    } else if let Some(edge_id) = col.get_edge_id(row_idx) {
                        ColumnValue::EdgeId(edge_id)
                    } else if let Some(val) = col.get_value(row_idx) {
                        ColumnValue::Value(val)
                    } else {
                        ColumnValue::Value(grafeo_common::types::Value::Null)
                    };
                    columns.push(value);
                }

                rows.push(InputRow {
                    columns,
                    source_node,
                });
            }
        }

        self.input_rows = Some(rows);
        Ok(())
    }

    /// Gets edges from a node, respecting filters and visibility.
    fn get_edges(&self, node_id: NodeId) -> Vec<(NodeId, EdgeId)> {
        let epoch = self.viewing_epoch;
        let transaction_id = self.transaction_id;
        let use_versioned = !self.read_only;

        self.store
            .edges_from(node_id, self.direction)
            .into_iter()
            .filter(|(target_id, edge_id)| {
                // Filter by edge type if specified
                let type_matches = if self.edge_types.is_empty() {
                    true
                } else if let Some(actual_type) = self.store.edge_type(*edge_id) {
                    self.edge_types
                        .iter()
                        .any(|t| actual_type.as_str().eq_ignore_ascii_case(t.as_str()))
                } else {
                    false
                };

                if !type_matches {
                    return false;
                }

                // Filter by visibility
                if let Some(epoch) = epoch {
                    if use_versioned && let Some(tx) = transaction_id {
                        self.store.is_edge_visible_versioned(*edge_id, epoch, tx)
                            && self.store.is_node_visible_versioned(*target_id, epoch, tx)
                    } else {
                        self.store.is_edge_visible_at_epoch(*edge_id, epoch)
                            && self.store.is_node_visible_at_epoch(*target_id, epoch)
                    }
                } else {
                    true
                }
            })
            .collect()
    }

    /// Checks whether a candidate expansion is allowed under the current path mode.
    fn is_expansion_allowed(
        &self,
        segment: &PathSegment,
        target: NodeId,
        edge_id: EdgeId,
        source_node: NodeId,
    ) -> bool {
        match self.path_mode {
            PathMode::Walk => true,
            PathMode::Trail => !segment.contains_edge(edge_id),
            PathMode::Simple => {
                // No repeated nodes except the start may equal the end
                target == source_node || !segment.contains_node(target)
            }
            PathMode::Acyclic => !segment.contains_node(target),
        }
    }

    /// Process one input row, generating all reachable outputs.
    fn process_input_row(&self, input_idx: usize, source_node: NodeId) -> Vec<OutputRow> {
        let mut results = Vec::new();
        let needs_edges = self.output_path_detail || self.output_edge_list;
        let needs_tracking = needs_edges || self.path_mode != PathMode::Walk;

        // Zero-length path: when min_hops is 0 the source node matches itself
        // with no edges traversed. Emit it before starting the BFS.
        if self.min_hops == 0 {
            results.push(OutputRow {
                input_idx,
                edge_id: None,
                target_id: source_node,
                path_length: 0,
                path_nodes: if self.output_path_detail {
                    Some(vec![source_node])
                } else {
                    None
                },
                path_edges: if needs_edges { Some(Vec::new()) } else { None },
            });
        }

        if needs_tracking {
            // BFS with shared-prefix path tracking via Rc<PathSegment>.
            // Required for path detail output or non-Walk path modes.
            let mut frontier: VecDeque<(NodeId, u32, EdgeId, Rc<PathSegment>)> = VecDeque::new();

            let root = Rc::new(PathSegment {
                node: source_node,
                edge: None,
                parent: None,
            });

            for (target, edge_id) in self.get_edges(source_node) {
                if !self.is_expansion_allowed(&root, target, edge_id, source_node) {
                    continue;
                }
                let segment = Rc::new(PathSegment {
                    node: target,
                    edge: Some(edge_id),
                    parent: Some(Rc::clone(&root)),
                });
                frontier.push_back((target, 1, edge_id, segment));
            }

            while let Some((current_node, depth, edge_id, segment)) = frontier.pop_front() {
                if depth >= self.min_hops && depth <= self.max_hops {
                    results.push(OutputRow {
                        input_idx,
                        edge_id: Some(edge_id),
                        target_id: current_node,
                        path_length: depth,
                        path_nodes: if self.output_path_detail {
                            Some(segment.collect_nodes(depth))
                        } else {
                            None
                        },
                        path_edges: if needs_edges {
                            Some(segment.collect_edges(depth))
                        } else {
                            None
                        },
                    });
                }

                if depth < self.max_hops {
                    for (target, next_edge_id) in self.get_edges(current_node) {
                        if !self.is_expansion_allowed(&segment, target, next_edge_id, source_node) {
                            continue;
                        }
                        let new_segment = Rc::new(PathSegment {
                            node: target,
                            edge: Some(next_edge_id),
                            parent: Some(Rc::clone(&segment)),
                        });
                        frontier.push_back((target, depth + 1, next_edge_id, new_segment));
                    }
                }
            }
        } else {
            // BFS without path tracking (lightweight, Walk mode only)
            let mut frontier: VecDeque<(NodeId, u32, EdgeId)> = VecDeque::new();

            for (target, edge_id) in self.get_edges(source_node) {
                frontier.push_back((target, 1, edge_id));
            }

            while let Some((current_node, depth, edge_id)) = frontier.pop_front() {
                if depth >= self.min_hops && depth <= self.max_hops {
                    results.push(OutputRow {
                        input_idx,
                        edge_id: Some(edge_id),
                        target_id: current_node,
                        path_length: depth,
                        path_nodes: None,
                        path_edges: None,
                    });
                }

                if depth < self.max_hops {
                    for (target, next_edge_id) in self.get_edges(current_node) {
                        frontier.push_back((target, depth + 1, next_edge_id));
                    }
                }
            }
        }

        results
    }

    /// Fill the output buffer with results from the next input row.
    fn fill_output_buffer(&mut self) {
        let Some(input_rows) = &self.input_rows else {
            return;
        };

        while self.output_buffer.is_empty() && self.current_input_idx < input_rows.len() {
            let source_node = input_rows[self.current_input_idx].source_node;
            let results = self.process_input_row(self.current_input_idx, source_node);
            self.output_buffer.extend(results);
            self.current_input_idx += 1;
        }
    }
}

impl Operator for VariableLengthExpandOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // Materialize input on first call
        if self.input_rows.is_none() {
            self.materialize_input()?;
            if self.input_rows.as_ref().map_or(true, Vec::is_empty) {
                self.exhausted = true;
                return Ok(None);
            }
        }

        // Fill output buffer if empty
        self.fill_output_buffer();

        if self.output_buffer.is_empty() {
            self.exhausted = true;
            return Ok(None);
        }

        let input_rows = self
            .input_rows
            .as_ref()
            .expect("input_rows is Some: populated during BFS");

        // Build output chunk from buffer
        let num_input_cols = input_rows.first().map_or(0, |r| r.columns.len());

        // Schema: [input_columns..., edge, target, (path_length)?, (path_nodes)?, (path_edges)?, (path)?]
        let extra_cols =
            2 + usize::from(self.output_path_length) + usize::from(self.output_path_detail) * 3;
        let mut schema: Vec<LogicalType> = Vec::with_capacity(num_input_cols + extra_cols);
        if let Some(first_row) = input_rows.first() {
            for col_val in &first_row.columns {
                let ty = match col_val {
                    ColumnValue::NodeId(_) => LogicalType::Node,
                    ColumnValue::EdgeId(_) => LogicalType::Edge,
                    ColumnValue::Value(_) => LogicalType::Any,
                };
                schema.push(ty);
            }
        }
        schema.push(if self.output_edge_list {
            LogicalType::List(Box::new(LogicalType::Edge))
        } else {
            LogicalType::Edge
        });
        schema.push(LogicalType::Node);
        if self.output_path_length {
            schema.push(LogicalType::Int64);
        }
        if self.output_path_detail {
            schema.push(LogicalType::List(Box::new(LogicalType::Node))); // path nodes (ids)
            schema.push(LogicalType::List(Box::new(LogicalType::Edge))); // path edges (ids)
            schema.push(LogicalType::Any); // Value::Path (first-class path)
        }

        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        // Take up to chunk_capacity rows from buffer
        let take_count = self.output_buffer.len().min(self.chunk_capacity);
        let to_output: Vec<_> = self.output_buffer.drain(..take_count).collect();

        for out_row in &to_output {
            let input_row = &input_rows[out_row.input_idx];

            // Copy input columns
            for (col_idx, col_val) in input_row.columns.iter().enumerate() {
                if let Some(out_col) = chunk.column_mut(col_idx) {
                    match col_val {
                        ColumnValue::NodeId(id) => out_col.push_node_id(*id),
                        ColumnValue::EdgeId(id) => out_col.push_edge_id(*id),
                        ColumnValue::Value(v) => out_col.push_value(v.clone()),
                    }
                }
            }

            // Add edge column: the path's edges (an empty list for a
            // zero-length path), or its last edge (Null for zero length)
            if let Some(col) = chunk.column_mut(num_input_cols) {
                if self.output_edge_list {
                    let edges = edge_id_list(out_row.path_edges.as_deref().unwrap_or(&[]))?;
                    col.push_value(grafeo_common::types::Value::List(edges.into()));
                } else if let Some(edge_id) = out_row.edge_id {
                    col.push_edge_id(edge_id);
                } else {
                    col.push_value(grafeo_common::types::Value::Null);
                }
            }

            // Add target node column
            if let Some(col) = chunk.column_mut(num_input_cols + 1) {
                col.push_node_id(out_row.target_id);
            }

            // Add path length column if requested
            if self.output_path_length
                && let Some(col) = chunk.column_mut(num_input_cols + 2)
            {
                col.push_value(grafeo_common::types::Value::Int64(i64::from(
                    out_row.path_length,
                )));
            }

            // Add path detail columns if requested
            if self.output_path_detail {
                let base = num_input_cols + 2 + usize::from(self.output_path_length);

                // Path nodes column
                if let Some(col) = chunk.column_mut(base) {
                    let nodes_list: Vec<grafeo_common::types::Value> = out_row
                        .path_nodes
                        .as_deref()
                        .unwrap_or(&[])
                        .iter()
                        .map(|id| {
                            let signed = i64::try_from(id.0).map_err(|_| {
                                OperatorError::Execution(format!(
                                    "NodeId {} exceeds i64 range",
                                    id.0
                                ))
                            })?;
                            Ok(grafeo_common::types::Value::Int64(signed))
                        })
                        .collect::<Result<Vec<_>, OperatorError>>()?;
                    col.push_value(grafeo_common::types::Value::List(nodes_list.into()));
                }

                // Path edges column
                if let Some(col) = chunk.column_mut(base + 1) {
                    let edges_list = edge_id_list(out_row.path_edges.as_deref().unwrap_or(&[]))?;
                    col.push_value(grafeo_common::types::Value::List(edges_list.into()));
                }

                // Value::Path column (first-class path value)
                if let Some(col) = chunk.column_mut(base + 2) {
                    let nodes: Vec<grafeo_common::types::Value> = out_row
                        .path_nodes
                        .as_deref()
                        .unwrap_or(&[])
                        .iter()
                        .map(|id| {
                            let signed = i64::try_from(id.0).map_err(|_| {
                                OperatorError::Execution(format!(
                                    "NodeId {} exceeds i64 range",
                                    id.0
                                ))
                            })?;
                            Ok(grafeo_common::types::Value::Int64(signed))
                        })
                        .collect::<Result<Vec<_>, OperatorError>>()?;
                    let edges: Vec<grafeo_common::types::Value> = out_row
                        .path_edges
                        .as_deref()
                        .unwrap_or(&[])
                        .iter()
                        .map(|id| {
                            let signed = i64::try_from(id.0).map_err(|_| {
                                OperatorError::Execution(format!(
                                    "EdgeId {} exceeds i64 range",
                                    id.0
                                ))
                            })?;
                            Ok(grafeo_common::types::Value::Int64(signed))
                        })
                        .collect::<Result<Vec<_>, OperatorError>>()?;
                    col.push_value(grafeo_common::types::Value::Path {
                        nodes: nodes.into(),
                        edges: edges.into(),
                    });
                }
            }
        }

        chunk.set_count(to_output.len());
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.input.reset();
        self.input_rows = None;
        self.current_input_idx = 0;
        self.output_buffer.clear();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "VariableLengthExpand"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Edge ids as the `Value::Int64` items of an edge list.
fn edge_id_list(edges: &[EdgeId]) -> Result<Vec<grafeo_common::types::Value>, OperatorError> {
    edges
        .iter()
        .map(|id| {
            i64::try_from(id.0)
                .map(grafeo_common::types::Value::Int64)
                .map_err(|_| OperatorError::Execution(format!("EdgeId {} exceeds i64 range", id.0)))
        })
        .collect()
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::operators::ScanOperator;
    use crate::graph::lpg::LpgStore;

    #[test]
    fn test_variable_length_expand_chain() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create chain: a -> b -> c -> d
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.set_node_property(a, "name", "a".into());
        store.set_node_property(b, "name", "b".into());
        store.set_node_property(c, "name", "c".into());
        store.set_node_property(d, "name", "d".into());

        store.create_edge(a, b, "NEXT");
        store.create_edge(b, c, "NEXT");
        store.create_edge(c, d, "NEXT");

        // Create scan for all nodes
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));

        // Expand 1-3 hops from all nodes
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["NEXT".to_string()],
            1,
            3,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a', we should reach b (1 hop), c (2 hops), d (3 hops)
        let a_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == a)
            .map(|(_, t)| *t)
            .collect();
        assert!(a_targets.contains(&b), "a should reach b");
        assert!(a_targets.contains(&c), "a should reach c");
        assert!(a_targets.contains(&d), "a should reach d");
        assert_eq!(a_targets.len(), 3, "a should reach exactly 3 nodes");
    }

    #[test]
    fn test_variable_length_expand_min_hops() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create chain: a -> b -> c
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);

        store.create_edge(a, b, "NEXT");
        store.create_edge(b, c, "NEXT");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));

        // Expand 2-3 hops only (skip 1 hop)
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["NEXT".to_string()],
            2, // min 2 hops
            3, // max 3 hops
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a', we should reach c (2 hops) but NOT b (1 hop)
        let a_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == a)
            .map(|(_, t)| *t)
            .collect();
        assert!(
            !a_targets.contains(&b),
            "a should NOT reach b with min_hops=2"
        );
        assert!(a_targets.contains(&c), "a should reach c");
    }

    #[test]
    fn test_variable_length_expand_diamond() {
        let store = Arc::new(LpgStore::new().unwrap());

        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "EDGE");
        store.create_edge(a, c, "EDGE");
        store.create_edge(b, d, "EDGE");
        store.create_edge(c, d, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            2,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a': b (1 hop), c (1 hop), d (2 hops via b), d (2 hops via c)
        let a_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == a)
            .map(|(_, t)| *t)
            .collect();
        assert!(a_targets.contains(&b));
        assert!(a_targets.contains(&c));
        assert!(a_targets.contains(&d));
        // d appears twice (two paths)
        assert_eq!(a_targets.iter().filter(|&&t| t == d).count(), 2);
    }

    #[test]
    fn test_variable_length_expand_no_matching_edges() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        // Filter for LIKES edges (which don't exist)
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["LIKES".to_string()],
            1,
            3,
        );

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_variable_length_expand_single_hop() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        // Exactly 1 hop
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            1,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Only a -> b (1 hop)
        let a_results: Vec<_> = results.iter().filter(|(s, _)| *s == a).collect();
        assert_eq!(a_results.len(), 1);
        assert_eq!(a_results[0].1, b);
    }

    #[test]
    fn test_variable_length_expand_with_path_length() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");
        store.create_edge(b, c, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            2,
        )
        .with_path_length_output();

        let mut found_path_lengths = false;
        while let Ok(Some(chunk)) = expand.next() {
            // With path_length_output, there should be an extra column
            assert!(chunk.column_count() >= 4); // source, edge, target, path_length
            found_path_lengths = true;
        }
        assert!(found_path_lengths);
    }

    #[test]
    fn test_variable_length_expand_reset() {
        let store = Arc::new(LpgStore::new().unwrap());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            1,
        );

        // First pass
        let mut count1 = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count1 += chunk.row_count();
        }

        expand.reset();

        // Second pass
        let mut count2 = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count2 += chunk.row_count();
        }

        assert_eq!(count1, count2);
    }

    #[test]
    fn test_variable_length_expand_name() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        );
        assert_eq!(expand.name(), "VariableLengthExpand");
    }

    #[test]
    fn test_variable_length_expand_empty_input() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Nonexistent",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        );

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_variable_length_expand_with_chunk_capacity() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create a star graph: center -> 5 outer nodes
        let center = store.create_node(&["Node"]);
        for _ in 0..5 {
            let outer = store.create_node(&["Node"]);
            store.create_edge(center, outer, "EDGE");
        }

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            1,
        )
        .with_chunk_capacity(2);

        let mut total = 0;
        let mut chunk_count = 0;
        while let Ok(Some(chunk)) = expand.next() {
            chunk_count += 1;
            total += chunk.row_count();
        }

        assert_eq!(total, 5);
        assert!(chunk_count >= 2);
    }

    #[test]
    fn test_trail_mode_no_repeated_edges() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create cycle: a -> b -> a (same edge types)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");
        store.create_edge(b, a, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            4,
        )
        .with_path_mode(PathMode::Trail);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a': Trail allows a->b (1 hop) and a->b->a (2 hops, different edges)
        // but NOT a->b->a->b (3 hops, would reuse the a->b edge)
        let a_results: Vec<_> = results.iter().filter(|(s, _)| *s == a).collect();
        assert_eq!(a_results.len(), 2, "Trail from a: a->b and a->b->a only");
    }

    #[test]
    fn test_acyclic_mode_no_repeated_nodes() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create cycle: a -> b -> a
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "EDGE");
        store.create_edge(b, a, "EDGE");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            4,
        )
        .with_path_mode(PathMode::Acyclic);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From 'a': Acyclic allows a->b only (cannot revisit a)
        let a_results: Vec<_> = results.iter().filter(|(s, _)| *s == a).collect();
        assert_eq!(a_results.len(), 1, "Acyclic from a: only a->b");
        assert_eq!(a_results[0].1, b);
    }

    #[test]
    fn test_variable_length_expand_into_any() {
        let store = Arc::new(LpgStore::new().unwrap());
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Node",
        ));
        let op = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<VariableLengthExpandOperator>().is_ok());
    }

    // --- PathSegment collection tests ---

    #[test]
    fn test_path_segment_collect_nodes_single_hop() {
        // Root (Alix) -> target (Gus): one hop
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });

        let nodes = hop1.collect_nodes(1);
        assert_eq!(nodes, vec![NodeId(1), NodeId(2)]);
    }

    #[test]
    fn test_path_segment_collect_nodes_multi_hop() {
        // Chain: Alix(1) -> Gus(2) -> Vincent(3) -> Jules(4)
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });
        let hop2 = Rc::new(PathSegment {
            node: NodeId(3),
            edge: Some(EdgeId(101)),
            parent: Some(Rc::clone(&hop1)),
        });
        let hop3 = Rc::new(PathSegment {
            node: NodeId(4),
            edge: Some(EdgeId(102)),
            parent: Some(Rc::clone(&hop2)),
        });

        let nodes = hop3.collect_nodes(3);
        assert_eq!(nodes, vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)]);
    }

    #[test]
    fn test_path_segment_collect_edges_single_hop() {
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });

        let edges = hop1.collect_edges(1);
        assert_eq!(edges, vec![EdgeId(100)]);
    }

    #[test]
    fn test_path_segment_collect_edges_multi_hop() {
        // Chain: 3 edges connecting 4 nodes
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(10)),
            parent: Some(Rc::clone(&root)),
        });
        let hop2 = Rc::new(PathSegment {
            node: NodeId(3),
            edge: Some(EdgeId(20)),
            parent: Some(Rc::clone(&hop1)),
        });
        let hop3 = Rc::new(PathSegment {
            node: NodeId(4),
            edge: Some(EdgeId(30)),
            parent: Some(Rc::clone(&hop2)),
        });

        let edges = hop3.collect_edges(3);
        assert_eq!(edges, vec![EdgeId(10), EdgeId(20), EdgeId(30)]);
    }

    #[test]
    fn test_path_segment_root_has_no_edges() {
        let root = PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        };

        let edges = root.collect_edges(0);
        assert!(edges.is_empty(), "Root segment should yield no edges");

        let nodes = root.collect_nodes(0);
        assert_eq!(
            nodes,
            vec![NodeId(1)],
            "Root should yield only its own node"
        );
    }

    #[test]
    fn test_path_segment_contains_node() {
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });

        assert!(hop1.contains_node(NodeId(1)), "Should find root node");
        assert!(hop1.contains_node(NodeId(2)), "Should find current node");
        assert!(
            !hop1.contains_node(NodeId(3)),
            "Should not find absent node"
        );
    }

    #[test]
    fn test_path_segment_contains_edge() {
        let root = Rc::new(PathSegment {
            node: NodeId(1),
            edge: None,
            parent: None,
        });
        let hop1 = Rc::new(PathSegment {
            node: NodeId(2),
            edge: Some(EdgeId(100)),
            parent: Some(Rc::clone(&root)),
        });

        assert!(hop1.contains_edge(EdgeId(100)), "Should find current edge");
        assert!(
            !hop1.contains_edge(EdgeId(999)),
            "Should not find absent edge"
        );
    }

    // --- Expansion validation tests (via operator integration) ---

    #[test]
    fn test_walk_mode_allows_everything() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create cycle: Alix -> Gus -> Alix
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, alix, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Person",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            3,
        )
        .with_path_mode(PathMode::Walk);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Walk mode should allow repeated nodes and edges
        // From Alix: Gus(1), Alix(2), Gus(3) = 3 results
        let alix_results: Vec<_> = results.iter().filter(|(s, _)| *s == alix).collect();
        assert_eq!(
            alix_results.len(),
            3,
            "Walk mode should allow all 3 hops from Alix in a cycle"
        );
    }

    #[test]
    fn test_simple_mode_rejects_repeated_node() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Triangle: Vincent -> Jules -> Mia -> Vincent
        let vincent = store.create_node(&["Person"]);
        let jules = store.create_node(&["Person"]);
        let mia = store.create_node(&["Person"]);
        store.create_edge(vincent, jules, "KNOWS");
        store.create_edge(jules, mia, "KNOWS");
        store.create_edge(mia, vincent, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Person",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec![],
            1,
            5,
        )
        .with_path_mode(PathMode::Simple);

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From Vincent: Jules(1), Mia(2), Vincent(3, allowed: start=end)
        // No further expansion because Vincent was already visited
        let vincent_results: Vec<_> = results.iter().filter(|(s, _)| *s == vincent).collect();
        assert_eq!(
            vincent_results.len(),
            3,
            "Simple: Vincent -> Jules, Mia, back to Vincent (start=end allowed)"
        );
    }

    // --- Path detail output tests ---

    #[test]
    fn test_path_detail_output_node_and_edge_lists() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Chain: Alix -> Gus -> Vincent
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);
        let e1 = store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Person",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
            1,
            2,
        )
        .with_path_detail_output();

        let mut found_path_nodes = false;
        let mut found_edge_with_correct_id = false;
        while let Ok(Some(chunk)) = expand.next() {
            // With path detail, extra columns: path_nodes (list), path_edges (list), path (Path)
            // Schema: [source_node, edge, target, path_nodes, path_edges, path]
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();

                // Check path nodes column (index 3) for any Alix-sourced path
                if src == alix
                    && let Some(col) = chunk.column(3)
                    && let Some(val) = col.get_value(i)
                    && let Some(list) = val.as_list()
                {
                    assert!(
                        list.len() >= 2,
                        "Path node list should have at least 2 entries"
                    );
                    found_path_nodes = true;
                }

                // Check path edges column (index 4) for the Alix->Gus single-hop path
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                if src == alix
                    && dst == gus
                    && let Some(col) = chunk.column(4)
                    && let Some(val) = col.get_value(i)
                    && let Some(list) = val.as_list()
                {
                    assert_eq!(list.len(), 1, "Single-hop path should have exactly 1 edge");
                    assert_eq!(list[0].as_int64(), Some(e1.0.cast_signed()));
                    found_edge_with_correct_id = true;
                }
            }
        }
        assert!(
            found_path_nodes,
            "Should have found path node lists in output"
        );
        assert!(
            found_edge_with_correct_id,
            "Should have found edge list with correct edge ID"
        );
    }

    // --- Edge type filtering tests ---

    #[test]
    fn test_edge_type_filter_case_insensitive() {
        let store = Arc::new(LpgStore::new().unwrap());

        // Alix -[:KNOWS]-> Gus, Alix -[:LIKES]-> Vincent
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "LIKES");

        // Filter with lowercase "knows", should still match "KNOWS"
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            "Person",
        ));
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan,
            0,
            Direction::Outgoing,
            vec!["knows".to_string()],
            1,
            1,
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // From Alix, only Gus should be reached (KNOWS matches "knows")
        let alix_targets: Vec<NodeId> = results
            .iter()
            .filter(|(s, _)| *s == alix)
            .map(|(_, t)| *t)
            .collect();
        assert!(
            alix_targets.contains(&gus),
            "Case-insensitive match should find KNOWS edge"
        );
        assert!(
            !alix_targets.contains(&vincent),
            "LIKES edge should be filtered out"
        );
    }
}
