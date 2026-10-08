//! Variable-length expand operator for multi-hop path traversal.

use super::expand::visible_edges_from;
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TransactionId};
use grafeo_common::utils::hash::FxHashSet;
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
    /// Whether to emit each reachable node once instead of one row per walk
    /// (see [`Self::with_reachability`]).
    reachability: Reachability,
    /// The nodes emitted for earlier input rows, with
    /// [`Self::with_reachability_across_rows`].
    emitted_across_rows: FxHashSet<NodeId>,
}

/// Which rows a variable-length expand emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reachability {
    /// One row per walk.
    Off,
    /// Each node an input row reaches once ([`VariableLengthExpandOperator::with_reachability`]).
    PerInputRow,
    /// Each node once over all input rows
    /// ([`VariableLengthExpandOperator::with_reachability_across_rows`]).
    AcrossInputRows,
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
            reachability: Reachability::Off,
            emitted_across_rows: FxHashSet::default(),
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

    /// Emits every node an input row reaches within the hop range once, at
    /// the position of its first walk, instead of one row per walk.
    ///
    /// For rows that only reach an operator which ignores duplicate rows
    /// (`DISTINCT`, `count(DISTINCT ...)`, `min`): the targets and their order
    /// are those of the walks, with each repeat of an (input row, target)
    /// pair left out. The edge column and the path columns hold NULL, as a
    /// target no longer stands for one walk. Only in WALK mode: the other
    /// path modes keep their search, where a node reached once can still be
    /// on another path.
    pub fn with_reachability(mut self) -> Self {
        self.reachability = Reachability::PerInputRow;
        self
    }

    /// Like [`Self::with_reachability`], and emits each node once over all
    /// input rows: for the first input row that reaches it, at the position
    /// the search per input row first emits it. For rows whose consumer reads
    /// nothing of the input row but the target (`RETURN DISTINCT m.id`): it
    /// sees the first row of each target, in the same order as before, and
    /// only loses copies of rows it has seen.
    ///
    /// Each input row still runs its own search, with its own layers and its
    /// own emitted nodes; only its output leaves out the nodes emitted for
    /// earlier rows. A node an earlier row emitted can be a stepping stone for
    /// this one: reached at another depth, or below `min_hops`, it leads on
    /// to nodes only this row reaches within `max_hops`, so the search must
    /// expand it as if it were new.
    pub fn with_reachability_across_rows(mut self) -> Self {
        self.reachability = Reachability::AcrossInputRows;
        self
    }

    /// Whether this expand runs the reachability search of
    /// [`Self::with_reachability`].
    fn searches_reachability(&self) -> bool {
        self.reachability != Reachability::Off && self.path_mode == PathMode::Walk
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

    /// Gets edges from a node, respecting filters and visibility (see
    /// [`visible_edges_from`]).
    fn get_edges(&self, node_id: NodeId) -> Vec<(NodeId, EdgeId)> {
        visible_edges_from(
            self.store.as_ref(),
            node_id,
            self.direction,
            &self.edge_types,
            self.viewing_epoch,
            self.transaction_id,
            self.read_only,
        )
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
        if self.searches_reachability() {
            return self.reachable_targets(input_idx, source_node);
        }
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

    /// The targets of one input row in reachability mode, each once.
    ///
    /// Layer `L` is the set of nodes a walk of exactly `L` edges ends at, in
    /// the order the walk BFS above first reaches them. That BFS emits every
    /// walk of `L` edges before the walks of `L + 1`, which extend the walks
    /// of `L` in order, each by the edges of `get_edges`; a repeat of a node
    /// in layer `L` only repeats the ends its first occurrence gave. So
    /// expanding each node of layer `L` once, in order, gives layer `L + 1`
    /// in first-reached order, and a node emitted at the first layer in
    /// `min_hops..=max_hops` that holds it is emitted at the position of its
    /// first walk.
    ///
    /// From `min_hops` on, a layer keeps only the nodes it emits: a node
    /// emitted at an earlier layer `j` was expanded after it, and whatever it
    /// reaches in `m` more edges is in layer `j + m`, emitted already. So
    /// expanding it again could give no new node and change no order. Each
    /// node is then expanded once from `min_hops` on (twice if it was in
    /// layer `min_hops - 1`), also for the hundred hops of an unbounded
    /// pattern.
    fn reachable_targets(&self, input_idx: usize, source_node: NodeId) -> Vec<OutputRow> {
        let reached = |target_id, path_length| OutputRow {
            input_idx,
            edge_id: None,
            target_id,
            path_length,
            path_nodes: None,
            path_edges: None,
        };
        let mut results = Vec::new();
        let mut emitted: FxHashSet<NodeId> = FxHashSet::default();
        let mut layer = vec![source_node];
        let mut next_layer = Vec::new();
        // The nodes of a layer below `min_hops`, which is kept whole
        let mut in_next_layer: FxHashSet<NodeId> = FxHashSet::default();

        if self.min_hops == 0 {
            emitted.insert(source_node);
            results.push(reached(source_node, 0));
        }
        for depth in 1..=self.max_hops {
            let emitting = depth >= self.min_hops;
            for &node in &layer {
                for (target, _) in self.get_edges(node) {
                    if emitting {
                        if emitted.insert(target) {
                            results.push(reached(target, depth));
                            next_layer.push(target);
                        }
                    } else if in_next_layer.insert(target) {
                        next_layer.push(target);
                    }
                }
            }
            if next_layer.is_empty() {
                break;
            }
            std::mem::swap(&mut layer, &mut next_layer);
            next_layer.clear();
            in_next_layer.clear();
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
            let mut results = self.process_input_row(self.current_input_idx, source_node);
            if self.searches_reachability() && self.reachability == Reachability::AcrossInputRows {
                results.retain(|row| self.emitted_across_rows.insert(row.target_id));
            }
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
            // zero-length path), or its last edge (Null for zero length and
            // for a reachability search, which follows no single path)
            if let Some(col) = chunk.column_mut(num_input_cols) {
                if self.output_edge_list
                    && let Some(path_edges) = &out_row.path_edges
                {
                    let edges = edge_id_list(path_edges)?;
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

            // A reachability search has no path to describe either
            if self.searches_reachability() {
                for col_idx in num_input_cols + 2..schema.len() {
                    if let Some(col) = chunk.column_mut(col_idx) {
                        col.push_value(grafeo_common::types::Value::Null);
                    }
                }
                continue;
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
        self.emitted_across_rows.clear();
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

    // --- Reachability mode ---

    /// How a test expand emits its targets.
    #[derive(Clone, Copy)]
    enum Search {
        /// One row per walk.
        Walks,
        /// Each node once per input row ([`VariableLengthExpandOperator::with_reachability`]).
        PerInputRow,
        /// Each node once over all input rows
        /// ([`VariableLengthExpandOperator::with_reachability_across_rows`]).
        AcrossInputRows,
    }

    /// The (source, target) pairs of a variable-length expand from the nodes
    /// `input` returns, in output order.
    fn expand_pairs(
        store: &Arc<LpgStore>,
        input: Box<dyn Operator>,
        direction: Direction,
        edge_types: &[&str],
        hops: (u32, u32),
        search: Search,
    ) -> Vec<(NodeId, NodeId)> {
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(store) as Arc<dyn GraphStoreSearch>,
            input,
            0,
            direction,
            edge_types.iter().map(ToString::to_string).collect(),
            hops.0,
            hops.1,
        );
        expand = match search {
            Search::Walks => expand,
            Search::PerInputRow => expand.with_reachability(),
            Search::AcrossInputRows => expand.with_reachability_across_rows(),
        };
        let mut pairs = Vec::new();
        while let Some(chunk) = expand.next().unwrap() {
            for i in 0..chunk.row_count() {
                pairs.push((
                    chunk.column(0).unwrap().get_node_id(i).unwrap(),
                    chunk.column(2).unwrap().get_node_id(i).unwrap(),
                ));
            }
        }
        pairs
    }

    fn scan(store: &Arc<LpgStore>, label: &str) -> Box<dyn Operator> {
        Box::new(ScanOperator::with_label(
            Arc::clone(store) as Arc<dyn GraphStoreSearch>,
            label,
        ))
    }

    /// `pairs` without the repeats of a pair, each pair at its first position.
    fn first_occurrences(pairs: &[(NodeId, NodeId)]) -> Vec<(NodeId, NodeId)> {
        let mut seen = std::collections::HashSet::new();
        pairs
            .iter()
            .copied()
            .filter(|pair| seen.insert(*pair))
            .collect()
    }

    /// `pairs` without the repeats of a target, each target with the source of
    /// its first pair.
    fn first_occurrences_of_targets(pairs: &[(NodeId, NodeId)]) -> Vec<(NodeId, NodeId)> {
        let mut seen = std::collections::HashSet::new();
        pairs
            .iter()
            .copied()
            .filter(|(_, target)| seen.insert(*target))
            .collect()
    }

    /// Runs the walk enumeration and both reachability searches from every
    /// node with `label`. Asserts that the search per input row emits exactly
    /// the first walk to each (source, target) pair, and the search across
    /// input rows the first walk to each target, in walk order. Returns the
    /// pairs of the search per input row.
    fn assert_reachability_matches_walks(
        store: &Arc<LpgStore>,
        label: &str,
        direction: Direction,
        edge_types: &[&str],
        hops: (u32, u32),
    ) -> Vec<(NodeId, NodeId)> {
        let pairs = |search| {
            expand_pairs(
                store,
                scan(store, label),
                direction,
                edge_types,
                hops,
                search,
            )
        };
        let walks = pairs(Search::Walks);
        let reached = pairs(Search::PerInputRow);
        let context = format!("{direction:?} {edge_types:?} *{}..{}", hops.0, hops.1);
        assert_eq!(
            reached,
            first_occurrences(&walks),
            "per input row: {context}"
        );
        assert_eq!(
            pairs(Search::AcrossInputRows),
            first_occurrences_of_targets(&walks),
            "across input rows: {context}"
        );
        reached
    }

    /// The targets `pairs` has for `source`, in order.
    fn targets_of(pairs: &[(NodeId, NodeId)], source: NodeId) -> Vec<NodeId> {
        pairs
            .iter()
            .filter(|(s, _)| *s == source)
            .map(|(_, t)| *t)
            .collect()
    }

    #[test]
    fn reachability_chain() {
        // Alix -> Gus -> Vincent -> Jules
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        let jules = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "NEXT");
        store.create_edge(gus, vincent, "NEXT");
        store.create_edge(vincent, jules, "NEXT");

        let reached =
            assert_reachability_matches_walks(&store, "Node", Direction::Outgoing, &[], (1, 3));
        assert_eq!(targets_of(&reached, alix), vec![gus, vincent, jules]);
        assert_eq!(targets_of(&reached, vincent), vec![jules]);
        assert_eq!(targets_of(&reached, jules), Vec::<NodeId>::new());
    }

    #[test]
    fn reachability_diamond_emits_the_meeting_node_once() {
        // Alix -> Gus -> Jules and Alix -> Vincent -> Jules: two walks to Jules
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        let jules = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");
        store.create_edge(gus, jules, "KNOWS");
        store.create_edge(vincent, jules, "KNOWS");

        let walks = expand_pairs(
            &store,
            scan(&store, "Node"),
            Direction::Outgoing,
            &[],
            (1, 2),
            Search::Walks,
        );
        assert_eq!(targets_of(&walks, alix).len(), 4);
        let reached =
            assert_reachability_matches_walks(&store, "Node", Direction::Outgoing, &[], (1, 2));
        let from_alix = targets_of(&reached, alix);
        assert_eq!(from_alix.len(), 3);
        assert_eq!(from_alix.last(), Some(&jules));
    }

    #[test]
    fn reachability_triangle_returns_to_the_source_once() {
        // Alix -> Gus -> Vincent -> Alix, walked up to five times round
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");
        store.create_edge(vincent, alix, "KNOWS");

        let reached =
            assert_reachability_matches_walks(&store, "Node", Direction::Outgoing, &[], (1, 5));
        assert_eq!(targets_of(&reached, alix), vec![gus, vincent, alix]);
        assert_eq!(targets_of(&reached, vincent), vec![alix, gus, vincent]);
        // Many times round: still one row per node
        let reached =
            assert_reachability_matches_walks(&store, "Node", Direction::Outgoing, &[], (1, 100));
        assert_eq!(reached.len(), 9);
    }

    #[test]
    fn reachability_self_loop() {
        // Alix -> Alix and Alix -> Gus
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        store.create_edge(alix, alix, "KNOWS");
        store.create_edge(alix, gus, "KNOWS");

        for direction in [Direction::Outgoing, Direction::Incoming, Direction::Both] {
            let reached = assert_reachability_matches_walks(&store, "Node", direction, &[], (1, 3));
            let from_alix = targets_of(&reached, alix);
            assert!(
                from_alix.contains(&alix),
                "{direction:?}: the loop reaches Alix"
            );
            assert_eq!(
                from_alix.len(),
                if direction == Direction::Incoming {
                    1
                } else {
                    2
                }
            );
        }
    }

    #[test]
    fn reachability_comes_back_to_the_source_as_walks_do() {
        // Alix -> Gus, walked both ways: Alix - Gus - Alix is a walk of two edges
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");

        let reached =
            assert_reachability_matches_walks(&store, "Node", Direction::Both, &[], (1, 2));
        assert_eq!(targets_of(&reached, alix), vec![gus, alix]);
        assert_eq!(targets_of(&reached, gus), vec![alix, gus]);
    }

    #[test]
    fn reachability_min_hops() {
        // Alix -> Gus -> Vincent, walked both ways
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");

        // *0..2: the source first, as a path of no edges
        let reached =
            assert_reachability_matches_walks(&store, "Node", Direction::Both, &[], (0, 2));
        assert_eq!(targets_of(&reached, alix), vec![alix, gus, vincent]);

        // *2..3: Gus, one edge from Alix, is not emitted for the walk of one
        // edge, but is for Alix - Gus - Vincent - Gus, after the nodes two
        // edges away
        let reached =
            assert_reachability_matches_walks(&store, "Node", Direction::Both, &[], (2, 3));
        let from_alix = targets_of(&reached, alix);
        assert_eq!(from_alix.len(), 3);
        assert_eq!(from_alix.last(), Some(&gus));
        assert_eq!(targets_of(&reached, gus).first(), Some(&gus));

        for hops in [(0, 0), (1, 1), (2, 2), (0, 3), (1, 3), (3, 3), (2, 5)] {
            assert_reachability_matches_walks(&store, "Node", Direction::Both, &[], hops);
            assert_reachability_matches_walks(&store, "Node", Direction::Outgoing, &[], hops);
        }
    }

    #[test]
    fn reachability_expands_a_node_from_below_min_hops_again() {
        // Alix -> Gus -> Vincent, walked both ways with *3..4. Gus is one edge
        // from Alix: expanded below min_hops, and first emitted three edges
        // away. Vincent is two edges away (below min_hops) and four, and his
        // only neighbor is Gus: only expanding Gus again, after his first
        // emission, reaches him
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");

        let reached =
            assert_reachability_matches_walks(&store, "Node", Direction::Both, &[], (3, 4));
        let from_alix = targets_of(&reached, alix);
        assert_eq!(from_alix.first(), Some(&gus));
        assert_eq!(from_alix.len(), 3);
        assert!(from_alix.contains(&vincent));
    }

    #[test]
    fn reachability_each_direction() {
        // Alix -> Gus, Vincent -> Gus, Gus -> Jules, Jules -> Alix
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        let jules = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(vincent, gus, "KNOWS");
        store.create_edge(gus, jules, "KNOWS");
        store.create_edge(jules, alix, "KNOWS");

        let outgoing =
            assert_reachability_matches_walks(&store, "Node", Direction::Outgoing, &[], (1, 3));
        assert_eq!(targets_of(&outgoing, vincent), vec![gus, jules, alix]);
        let incoming =
            assert_reachability_matches_walks(&store, "Node", Direction::Incoming, &[], (1, 3));
        assert_eq!(targets_of(&incoming, vincent), Vec::<NodeId>::new());
        let into_alix = targets_of(&incoming, alix);
        assert_eq!(into_alix[..2], [jules, gus]);
        assert_eq!(into_alix.len(), 4, "and Alix and Vincent, three edges back");
        let both = assert_reachability_matches_walks(&store, "Node", Direction::Both, &[], (1, 3));
        assert_eq!(targets_of(&both, vincent).len(), 4);
    }

    #[test]
    fn reachability_edge_type_filter() {
        // Alix -KNOWS-> Gus -KNOWS-> Vincent, Alix -LIKES-> Vincent, Gus -LIKES-> Mia
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        let mia = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");
        store.create_edge(alix, vincent, "LIKES");
        store.create_edge(gus, mia, "LIKES");

        let knows = assert_reachability_matches_walks(
            &store,
            "Node",
            Direction::Outgoing,
            &["KNOWS"],
            (1, 2),
        );
        assert_eq!(targets_of(&knows, alix), vec![gus, vincent]);
        let likes = assert_reachability_matches_walks(
            &store,
            "Node",
            Direction::Outgoing,
            &["LIKES"],
            (1, 2),
        );
        assert_eq!(targets_of(&likes, alix), vec![vincent]);
        assert_eq!(targets_of(&likes, gus), vec![mia]);
        // Vincent is one LIKES edge and two KNOWS edges away: emitted once
        let both = assert_reachability_matches_walks(
            &store,
            "Node",
            Direction::Outgoing,
            &["KNOWS", "LIKES"],
            (1, 2),
        );
        let from_alix = targets_of(&both, alix);
        assert_eq!(from_alix.len(), 3);
        assert!(from_alix.contains(&mia));
    }

    #[test]
    fn reachability_empty_input() {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");

        let pairs = expand_pairs(
            &store,
            scan(&store, "Missing"),
            Direction::Both,
            &[],
            (0, 3),
            Search::PerInputRow,
        );
        assert_eq!(pairs, Vec::<(NodeId, NodeId)>::new());
    }

    #[test]
    fn reachability_hub_shared_by_several_sources() {
        // Alix, Gus and Vincent each point at a hub with 20 more neighbors
        let store = Arc::new(LpgStore::new().unwrap());
        let sources: Vec<NodeId> = (0..3).map(|_| store.create_node(&["Source"])).collect();
        let hub = store.create_node(&["Node"]);
        for &source in &sources {
            store.create_edge(source, hub, "KNOWS");
        }
        for _ in 0..20 {
            let leaf = store.create_node(&["Node"]);
            store.create_edge(hub, leaf, "KNOWS");
        }

        // Three edges: back to the hub from each of its 23 neighbors
        let walks = expand_pairs(
            &store,
            scan(&store, "Source"),
            Direction::Both,
            &[],
            (1, 3),
            Search::Walks,
        );
        let reached =
            assert_reachability_matches_walks(&store, "Source", Direction::Both, &[], (1, 3));
        for &source in &sources {
            assert_eq!(targets_of(&walks, source).len(), 1 + 23 + 23);
            let from_source = targets_of(&reached, source);
            assert_eq!(
                from_source.len(),
                1 + 23,
                "the hub, its leaves, the sources"
            );
            assert_eq!(from_source[0], hub);
            assert!(from_source.contains(&source));
        }
    }

    #[test]
    fn reachability_searches_once_per_input_row() {
        // The same sources twice: every input row gets its own targets
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");
        store.create_edge(vincent, alix, "KNOWS");

        let once = expand_pairs(
            &store,
            scan(&store, "Node"),
            Direction::Both,
            &[],
            (1, 2),
            Search::PerInputRow,
        );
        let twice_input = Box::new(crate::execution::operators::UnionOperator::new(
            vec![scan(&store, "Node"), scan(&store, "Node")],
            vec![LogicalType::Node],
        ));
        let twice = expand_pairs(
            &store,
            twice_input,
            Direction::Both,
            &[],
            (1, 2),
            Search::PerInputRow,
        );
        assert_eq!(twice, [once.clone(), once].concat());
    }

    /// 24 nodes and 50 edges of two types from a fixed pseudo-random
    /// sequence, plus a self-loop and two parallel edges.
    fn mixed_graph() -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new().unwrap());
        let nodes: Vec<NodeId> = (0..24).map(|_| store.create_node(&["Node"])).collect();
        let mut state: u64 = 0x5eed;
        let mut pick = |bound: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            usize::try_from(state >> 33).unwrap() % bound
        };
        for _ in 0..50 {
            let (from, to) = (pick(nodes.len()), pick(nodes.len()));
            let edge_type = if pick(3) == 0 { "LIKES" } else { "KNOWS" };
            store.create_edge(nodes[from], nodes[to], edge_type);
        }
        store.create_edge(nodes[0], nodes[0], "KNOWS");
        store.create_edge(nodes[1], nodes[2], "KNOWS");
        store.create_edge(nodes[1], nodes[2], "KNOWS");
        store
    }

    #[test]
    fn reachability_matches_walks_on_a_mixed_graph() {
        let store = mixed_graph();
        for direction in [Direction::Outgoing, Direction::Incoming, Direction::Both] {
            for edge_types in [&[][..], &["KNOWS"][..]] {
                for min_hops in 0..=3 {
                    for max_hops in min_hops..=5 {
                        assert_reachability_matches_walks(
                            &store,
                            "Node",
                            direction,
                            edge_types,
                            (min_hops, max_hops),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reachability_across_rows_expands_nodes_emitted_for_earlier_rows() {
        // Alix -> Gus -> Django -> Mia, and Jules -> Vincent -> Mia -> Butch.
        // With *2..3, Alix emits Django and Mia, and Mia (three edges away)
        // is as far as Alix goes. Jules reaches Mia two edges away: emitted
        // already, so not emitted again, but Butch is three edges from Jules
        // through Mia, and from no one else
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Source"]);
        let jules = store.create_node(&["Source"]);
        let gus = store.create_node(&["Node"]);
        let django = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        let mia = store.create_node(&["Node"]);
        let butch = store.create_node(&["Node"]);
        for (from, to) in [
            (alix, gus),
            (gus, django),
            (django, mia),
            (jules, vincent),
            (vincent, mia),
            (mia, butch),
        ] {
            store.create_edge(from, to, "KNOWS");
        }

        let per_row =
            assert_reachability_matches_walks(&store, "Source", Direction::Outgoing, &[], (2, 3));
        assert_eq!(
            per_row,
            vec![(alix, django), (alix, mia), (jules, mia), (jules, butch)]
        );
        let across = expand_pairs(
            &store,
            scan(&store, "Source"),
            Direction::Outgoing,
            &[],
            (2, 3),
            Search::AcrossInputRows,
        );
        // Gus and Vincent, one edge from a source, stay out
        assert_eq!(across, vec![(alix, django), (alix, mia), (jules, butch)]);
    }

    #[test]
    fn reachability_across_rows_emits_a_source_once_too() {
        // Alix -> Gus, both sources: with *0..1 Gus is a target of Alix
        // before Gus is a source
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Source"]);
        let gus = store.create_node(&["Source"]);
        store.create_edge(alix, gus, "KNOWS");

        let across = |direction| {
            expand_pairs(
                &store,
                scan(&store, "Source"),
                direction,
                &[],
                (0, 1),
                Search::AcrossInputRows,
            )
        };
        assert_eq!(across(Direction::Outgoing), vec![(alix, alix), (alix, gus)]);
        assert_eq!(across(Direction::Incoming), vec![(alix, alix), (gus, gus)]);
        assert_reachability_matches_walks(&store, "Source", Direction::Both, &[], (0, 1));
    }

    #[test]
    fn reachability_across_rows_over_repeated_input_rows() {
        // The same sources twice: the second copy reaches nothing new
        let store = mixed_graph();
        let twice_input = Box::new(crate::execution::operators::UnionOperator::new(
            vec![scan(&store, "Node"), scan(&store, "Node")],
            vec![LogicalType::Node],
        ));
        let once = expand_pairs(
            &store,
            scan(&store, "Node"),
            Direction::Both,
            &[],
            (1, 3),
            Search::AcrossInputRows,
        );
        let twice = expand_pairs(
            &store,
            twice_input,
            Direction::Both,
            &[],
            (1, 3),
            Search::AcrossInputRows,
        );
        assert!(once.len() > 1);
        assert_eq!(twice, once);
    }

    #[test]
    fn reachability_across_rows_starts_over_after_reset() {
        let store = mixed_graph();
        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan(&store, "Node"),
            0,
            Direction::Both,
            vec![],
            1,
            2,
        )
        .with_reachability_across_rows();
        fn targets(expand: &mut VariableLengthExpandOperator) -> Vec<NodeId> {
            let mut targets = Vec::new();
            while let Some(chunk) = expand.next().unwrap() {
                for i in 0..chunk.row_count() {
                    targets.push(chunk.column(2).unwrap().get_node_id(i).unwrap());
                }
            }
            targets
        }
        let first = targets(&mut expand);
        assert!(first.len() > 1);
        expand.reset();
        assert_eq!(targets(&mut expand), first);
    }

    #[test]
    fn reachability_leaves_edge_and_path_columns_null() {
        // Alix -> Gus -> Vincent
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        let vincent = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");

        let mut expand = VariableLengthExpandOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            scan(&store, "Node"),
            0,
            Direction::Both,
            vec![],
            0,
            2,
        )
        .with_edge_list_output()
        .with_path_length_output()
        .with_reachability();
        let mut rows = 0;
        while let Some(chunk) = expand.next().unwrap() {
            assert_eq!(chunk.column_count(), 4, "source, edge, target, length");
            for i in 0..chunk.row_count() {
                assert!(chunk.column(2).unwrap().get_node_id(i).is_some());
                assert!(chunk.column(1).unwrap().is_null(i), "edge column");
                assert!(chunk.column(3).unwrap().is_null(i), "path length column");
                rows += 1;
            }
        }
        // Each of the three reaches all three
        assert_eq!(rows, 9);
    }

    #[test]
    fn reachability_applies_to_walks_only() {
        // Alix -> Gus -> Alix: a trail stops where a walk goes on
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, alix, "KNOWS");

        let pairs = |reachability: bool| {
            let mut expand = VariableLengthExpandOperator::new(
                Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
                scan(&store, "Node"),
                0,
                Direction::Outgoing,
                vec![],
                1,
                4,
            )
            .with_path_mode(PathMode::Trail);
            if reachability {
                expand = expand.with_reachability();
            }
            let mut count = 0;
            while let Some(chunk) = expand.next().unwrap() {
                count += chunk.row_count();
            }
            count
        };
        assert_eq!(pairs(true), pairs(false));
        assert_eq!(pairs(false), 4, "two trails from each node");
    }
}
