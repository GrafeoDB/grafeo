//! Expand operator for relationship traversal.

use super::node_seek::node_ids;
use super::{Operator, OperatorError, OperatorResult, ReadCounter, SeekKey};
use crate::execution::DataChunk;
use crate::graph::Direction;
use crate::graph::GraphStore;
use grafeo_common::types::{ArcStr, EdgeId, LogicalType, NodeId, ReadView, Value};
use std::sync::Arc;

/// The most nodes of a target key an expand looks up and holds: a more common
/// key costs more to find than the lists it would save.
const TARGET_KEY_CAP: usize = 1024;

/// How many entries of a list one lookup of the edges to one node is worth: past this a row
/// looks its targets up. A guess, to measure.
const ENTRIES_PER_TARGET: usize = 5;

/// What a key's labels say to the expand that finds its nodes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyLabels {
    /// The labels whose indexes together hold the nodes: none, the key's on every label.
    pub read: Vec<String>,
    /// The labels a filter over the expand requires of every target.
    pub required: Vec<String>,
}

/// What an expand knows of the node each row's edges end at, under a filter
/// that decides.
enum HopTarget {
    /// Nothing: it lists every edge.
    Unknown,
    /// The node in this input column.
    Column(usize),
    /// A node that holds this key, with these labels, not looked up yet.
    Key(SeekKey, Value, KeyLabels),
    /// No node holds the key: there is no edge to find.
    Nothing,
    /// This one node.
    Node(NodeId),
    /// One of these nodes, sorted.
    Nodes(Vec<NodeId>),
}

/// An expand operator that traverses edges from source nodes.
///
/// For each input row containing a source node, this operator produces
/// output rows for each neighbor connected via matching edges.
pub struct ExpandOperator {
    /// The store to traverse.
    store: Arc<dyn GraphStore>,
    /// Input operator providing source nodes.
    input: Box<dyn Operator>,
    /// Index of the source node column in input.
    source_column: usize,
    /// Direction of edge traversal.
    direction: Direction,
    /// The edge types to follow, as stored (every type when empty).
    edge_types: Vec<ArcStr>,
    /// What is known of the node each row's edges end at.
    target: HopTarget,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Current input chunk being processed.
    current_input: Option<DataChunk>,
    /// Current row index in the input chunk.
    current_row: usize,
    /// Current edge iterator for the current row.
    current_edges: Vec<(NodeId, EdgeId)>,
    /// Current edge index.
    current_edge_idx: usize,
    /// Whether the operator is exhausted.
    exhausted: bool,
    /// Who reads: the expand follows the edges this view sees.
    view: ReadView,
    /// Counts the edges the expansion reads.
    reads: Option<Arc<ReadCounter>>,
}

/// The nodes that may hold `value` under `property`, through the indexes on `labels` or with no
/// label the index on every label, `None` past [`TARGET_KEY_CAP`] of them or when one of the
/// indexes is not there.
fn find_few_through_label_indexes(
    store: &dyn GraphStore,
    labels: &[String],
    property: &str,
    value: &Value,
    view: ReadView,
) -> grafeo_common::utils::error::Result<Option<Vec<NodeId>>> {
    if labels.is_empty() {
        return store.find_few_nodes_maybe_equal(property, value, TARGET_KEY_CAP, view);
    }
    let mut found = Vec::new();
    for label in labels {
        let left = TARGET_KEY_CAP.saturating_sub(found.len());
        match store.find_few_indexed_maybe_equal(&[label], property, value, left, view)? {
            Some(nodes) => found.extend(nodes),
            None => return Ok(None),
        }
    }
    Ok(Some(found))
}

impl ExpandOperator {
    /// Creates an expand of the edges of `edge_types` (as stored, every type
    /// when empty) that `view` sees.
    pub fn new(
        store: Arc<dyn GraphStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        direction: Direction,
        edge_types: Vec<String>,
        view: ReadView,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            direction,
            edge_types: stored_types(edge_types),
            target: HopTarget::Unknown,
            chunk_capacity: 2048,
            current_input: None,
            current_row: 0,
            current_edges: Vec::with_capacity(16), // typical node degree
            current_edge_idx: 0,
            exhausted: false,
            view,
            reads: None,
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Counts the edges the expansion reads in `reads`.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// Looks up the edges to the node in input column `column` instead of
    /// listing every edge of the row, under a filter that the target is that
    /// node. A row whose column holds no node id lists its edges, for the
    /// filter to judge; the filter also judges an integer the column reads as
    /// an id.
    #[must_use]
    pub fn with_target_column(mut self, column: usize) -> Self {
        self.target = HopTarget::Column(column);
        self
    }

    /// Finds the nodes that may hold `value` under `key` once, before the first
    /// input row, under a filter that the target holds it: no node ends the
    /// expand without reading its input, one node is a lookup per row, and
    /// several filter each row's list. A key more than `TARGET_KEY_CAP` nodes
    /// hold, or one the store cannot look up, lists every edge as usual, so this
    /// loses at most the reads of the cap. A node found without a label `labels` requires is
    /// dropped once, as the filter would drop every edge to it.
    #[must_use]
    pub fn with_target_key(mut self, key: SeekKey, value: Value, labels: KeyLabels) -> Self {
        self.target = HopTarget::Key(key, value, labels);
        self
    }

    /// Finds the nodes of a target key, once: a re-run of the expand (a subquery
    /// per row) reads the same key at the same view.
    fn find_target_key(&mut self) -> Result<(), OperatorError> {
        let HopTarget::Key(key, value, labels) = &self.target else {
            return Ok(());
        };
        let found = match key {
            SeekKey::Id => Some(node_ids(value)),
            SeekKey::Property(property) => find_few_through_label_indexes(
                self.store.as_ref(),
                &labels.read,
                property,
                value,
                self.view,
            )?,
        };
        self.target = match found {
            None => HopTarget::Unknown,
            Some(mut nodes) => {
                super::read_counter::count(self.reads.as_deref(), nodes.len());
                nodes.sort_unstable();
                nodes.dedup();
                for label in &labels.required {
                    let mut kept = Vec::with_capacity(nodes.len());
                    for node in nodes {
                        if self.store.node_has_label(node, label, self.view)? == Some(true) {
                            kept.push(node);
                        }
                    }
                    nodes = kept;
                }
                match nodes.as_slice() {
                    [] => HopTarget::Nothing,
                    [node] => HopTarget::Node(*node),
                    _ => HopTarget::Nodes(nodes),
                }
            }
        };
        Ok(())
    }

    /// Minimum chunk size for locality sort to be worthwhile.
    /// Below this threshold, the sort cost exceeds the cache savings.
    const LOCALITY_SORT_THRESHOLD: usize = 1024;

    /// Loads the next input chunk.
    fn load_next_input(&mut self) -> Result<bool, OperatorError> {
        match self.input.next() {
            Ok(Some(mut chunk)) => {
                // Flatten the chunk if it has a selection vector so we can use direct indexing
                chunk.flatten();
                // Sort by source node ID for cache locality during adjacency lookups.
                // Only worthwhile for larger chunks where the sort cost is amortized.
                if chunk.len() > Self::LOCALITY_SORT_THRESHOLD {
                    chunk = chunk.sort_by_column(self.source_column);
                }
                self.current_input = Some(chunk);
                self.current_row = 0;
                self.current_edges.clear();
                self.current_edge_idx = 0;
                Ok(true)
            }
            Ok(None) => {
                self.exhausted = true;
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    /// The edges of `source` to each of several targets, target by target, when its list is
    /// long enough for that to beat reading it. `edges_between` reads the shorter side only for
    /// one direction and one type, so another hop reads the list.
    fn edges_to_each_target(
        &self,
        source: NodeId,
    ) -> Result<Option<Vec<(NodeId, EdgeId)>>, OperatorError> {
        let HopTarget::Nodes(nodes) = &self.target else {
            return Ok(None);
        };
        if self.direction == Direction::Both || self.edge_types.len() != 1 {
            return Ok(None);
        }
        let long = self
            .store
            .edge_entries_at_most(source, self.direction, &self.edge_types, self.view)
            .is_some_and(|entries| entries > nodes.len().saturating_mul(ENTRIES_PER_TARGET));
        if !long {
            return Ok(None);
        }
        let mut edges = Vec::new();
        for target in nodes {
            let found = self.store.edges_between(
                source,
                *target,
                self.direction,
                &self.edge_types,
                self.view,
            )?;
            super::read_counter::count(self.reads.as_deref(), found.len());
            edges.extend(found.into_iter().map(|edge| (*target, edge)));
        }
        Ok(Some(edges))
    }

    /// Loads edges for the current row.
    fn load_edges_for_current_row(&mut self) -> Result<bool, OperatorError> {
        let Some(chunk) = &self.current_input else {
            return Ok(false);
        };

        if self.current_row >= chunk.row_count() {
            return Ok(false);
        }

        let col = chunk.column(self.source_column).ok_or_else(|| {
            OperatorError::ColumnNotFound(format!("Column {} not found", self.source_column))
        })?;

        // A null source (a node an OPTIONAL MATCH did not find) has no edges:
        // a pattern from it matches nothing.
        if col.is_null(self.current_row) {
            self.current_edges = Vec::new();
            self.current_edge_idx = 0;
            return Ok(true);
        }
        let source_id = col
            .get_node_id(self.current_row)
            .ok_or_else(|| OperatorError::Internal("Expected node ID in source column".into()))?;

        let target = match &self.target {
            HopTarget::Column(column) => chunk
                .column(*column)
                .and_then(|column| column.get_node_id(self.current_row)),
            HopTarget::Node(node) => Some(*node),
            _ => None,
        };
        self.current_edges = if let Some(edges) = self.edges_to_each_target(source_id)? {
            edges
        } else if let Some(target) = target {
            let edges = self.store.edges_between(
                source_id,
                target,
                self.direction,
                &self.edge_types,
                self.view,
            )?;
            super::read_counter::count(self.reads.as_deref(), edges.len());
            edges.into_iter().map(|edge| (target, edge)).collect()
        } else {
            visible_edges_from(
                self.store.as_ref(),
                source_id,
                self.direction,
                &EdgeWalk {
                    edge_types: &self.edge_types,
                    view: self.view,
                    reads: self.reads.as_deref(),
                },
            )?
        };
        // Edges found target by target have the right ends already; this costs them a search.
        if let HopTarget::Nodes(nodes) = &self.target {
            self.current_edges
                .retain(|(end, _)| nodes.binary_search(end).is_ok());
        }
        self.current_edge_idx = 0;
        Ok(true)
    }
}

/// The edge types of a pattern as the store holds them: the planner has
/// resolved the query's names to the stored ones, which the store matches
/// exactly.
pub(crate) fn stored_types(edge_types: Vec<String>) -> Vec<ArcStr> {
    edge_types.into_iter().map(ArcStr::from).collect()
}

/// What an edge walker passes to [`visible_edges_from`] besides the node and
/// direction: the edge types it takes (any type when empty), who reads, and
/// the counter of the edges it reads.
#[derive(Clone, Copy)]
pub(super) struct EdgeWalk<'a> {
    /// The edge types to take, as stored; every type when empty.
    pub(super) edge_types: &'a [ArcStr],
    /// Who reads.
    pub(super) view: ReadView,
    /// Counts the edges read.
    pub(super) reads: Option<&'a ReadCounter>,
}

/// The edges from `node` in `direction` that a query sees, each with the
/// node at its other end: those of one of the walk's types (any type when
/// empty) that its view sees, whose other end it sees, a self-loop once
/// both ways ([`GraphStore::edges`]). The expand operators, variable-length
/// expands, factorized chains and shortest paths all walk edges with it.
///
/// # Errors
///
/// Returns the error of a read the store cannot do.
pub(super) fn visible_edges_from(
    store: &dyn GraphStore,
    node: NodeId,
    direction: Direction,
    walk: &EdgeWalk<'_>,
) -> Result<Vec<(NodeId, EdgeId)>, OperatorError> {
    let edges = store.edges(node, direction, walk.edge_types, walk.view)?;
    super::read_counter::count(walk.reads, edges.len());
    Ok(edges)
}

impl Operator for ExpandOperator {
    fn next(&mut self) -> OperatorResult {
        self.find_target_key()?;
        if self.exhausted || matches!(self.target, HopTarget::Nothing) {
            return Ok(None);
        }

        // Build output schema: preserve all input columns + edge + target
        // We need to build this dynamically based on input schema
        if self.current_input.is_none() {
            if !self.load_next_input()? {
                return Ok(None);
            }
            self.load_edges_for_current_row()?;
        }
        let input_chunk = self.current_input.as_ref().expect("input loaded above");

        // Build schema: [input_columns..., edge, target]
        let input_col_count = input_chunk.column_count();
        let mut schema: Vec<LogicalType> = (0..input_col_count)
            .map(|i| {
                input_chunk
                    .column(i)
                    .map_or(LogicalType::Any, |c| c.data_type().clone())
            })
            .collect();
        schema.push(LogicalType::Edge);
        schema.push(LogicalType::Node);

        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);
        let mut count = 0;

        while count < self.chunk_capacity {
            // If we need a new input chunk
            if self.current_input.is_none() {
                if !self.load_next_input()? {
                    break;
                }
                self.load_edges_for_current_row()?;
            }

            // If we've exhausted edges for current row, move to next row
            while self.current_edge_idx >= self.current_edges.len() {
                self.current_row += 1;

                // If we've exhausted the current input chunk, get next one
                if self.current_row >= self.current_input.as_ref().map_or(0, |c| c.row_count()) {
                    self.current_input = None;
                    if !self.load_next_input()? {
                        // No more input chunks
                        if count > 0 {
                            chunk.set_count(count);
                            return Ok(Some(chunk));
                        }
                        return Ok(None);
                    }
                }

                self.load_edges_for_current_row()?;
            }

            // Get the current edge
            let (target_id, edge_id) = self.current_edges[self.current_edge_idx];

            // Copy all input columns to output
            let input = self.current_input.as_ref().expect("input loaded above");
            for col_idx in 0..input_col_count {
                if let Some(input_col) = input.column(col_idx)
                    && let Some(output_col) = chunk.column_mut(col_idx)
                {
                    // Use copy_row_to which preserves NodeId/EdgeId types
                    input_col.copy_row_to(self.current_row, output_col);
                }
            }

            // Add edge column
            if let Some(col) = chunk.column_mut(input_col_count) {
                col.push_edge_id(edge_id);
            }

            // Add target node column
            if let Some(col) = chunk.column_mut(input_col_count + 1) {
                col.push_node_id(target_id);
            }

            count += 1;
            self.current_edge_idx += 1;
        }

        if count > 0 {
            chunk.set_count(count);
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.current_input = None;
        self.current_row = 0;
        self.current_edges.clear();
        self.current_edge_idx = 0;
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "Expand"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::ScanOperator;
    use crate::graph::rowgroup::RowGroupStore;

    /// Creates a new `RowGroupStore` wrapped in an `Arc` and returns both the
    /// concrete handle (for mutation) and a trait-object handle (for operators).
    fn test_store() -> (Arc<RowGroupStore>, Arc<dyn GraphStore>) {
        let store = Arc::new(RowGroupStore::new());
        let dyn_store: Arc<dyn GraphStore> = Arc::clone(&store) as Arc<dyn GraphStore>;
        (store, dyn_store)
    }

    #[test]
    fn test_expand_outgoing() {
        let (store, dyn_store) = test_store();

        // Create nodes
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);

        // Create edges: Alix -> Gus, Alix -> Vincent
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");

        // Scan Alix only
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));

        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0, // source column
            Direction::Outgoing,
            vec![],
            ReadView::latest(),
        );

        // Collect all results
        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let edge = chunk.column(1).unwrap().get_edge_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, edge, dst));
            }
        }

        // Alix -> Gus, Alix -> Vincent
        assert_eq!(results.len(), 2);

        // All source nodes should be Alix
        for (src, _, _) in &results {
            assert_eq!(*src, alix);
        }

        // Target nodes should be Gus and Vincent
        let targets: Vec<NodeId> = results.iter().map(|(_, _, dst)| *dst).collect();
        assert!(targets.contains(&gus));
        assert!(targets.contains(&vincent));
    }

    #[test]
    fn test_expand_with_edge_type_filter() {
        let (store, dyn_store) = test_store();

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let company = store.create_node(&["Company"]);

        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(alix, company, "WORKS_AT");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));

        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec!["KNOWS".to_string()],
            ReadView::latest(),
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push(dst);
            }
        }

        // Only KNOWS edges should be followed
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], gus);
    }

    #[test]
    fn test_expand_incoming() {
        let (store, dyn_store) = test_store();

        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);

        store.create_edge(alix, gus, "KNOWS");

        // Scan Gus
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));

        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Incoming,
            vec![],
            ReadView::latest(),
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Gus <- Alix (Gus's incoming edge from Alix)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, gus); // source in the expand is Gus
        assert_eq!(results[0].1, alix); // target is Alix (who points to Gus)
    }

    #[test]
    fn test_expand_no_edges() {
        let (store, dyn_store) = test_store();

        store.create_node(&["Person"]);

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));

        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec![],
            ReadView::latest(),
        );

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_expand_reset() {
        let (store, dyn_store) = test_store();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));
        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec![],
            ReadView::latest(),
        );

        // First pass
        let mut count1 = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count1 += chunk.row_count();
        }

        // Reset and run again
        expand.reset();
        let mut count2 = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count2 += chunk.row_count();
        }

        assert_eq!(count1, count2);
        assert_eq!(count1, 1);
    }

    #[test]
    fn test_expand_name() {
        let (_store, dyn_store) = test_store();
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));
        let expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec![],
            ReadView::latest(),
        );
        assert_eq!(expand.name(), "Expand");
    }

    #[test]
    fn test_expand_with_chunk_capacity() {
        let (store, dyn_store) = test_store();

        let a = store.create_node(&["Person"]);
        for _ in 0..5 {
            let b = store.create_node(&["Person"]);
            store.create_edge(a, b, "KNOWS");
        }

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));
        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec![],
            ReadView::latest(),
        )
        .with_chunk_capacity(2);

        // With capacity 2 and 5 edges from node a, we should get multiple chunks
        let mut total = 0;
        let mut chunk_count = 0;
        while let Ok(Some(chunk)) = expand.next() {
            chunk_count += 1;
            total += chunk.row_count();
        }

        assert_eq!(total, 5);
        assert!(
            chunk_count >= 2,
            "Expected multiple chunks with small capacity"
        );
    }

    /// The expand matches edge types as stored: the planner resolves a
    /// query's names to the stored ones (case-insensitively) before it
    /// builds the expand.
    #[test]
    fn test_expand_edge_type_matches_the_stored_name() {
        let (store, dyn_store) = test_store();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));
        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec!["knows".to_string()],
            ReadView::latest(),
        );

        let mut count = 0;
        while let Ok(Some(chunk)) = expand.next() {
            count += chunk.row_count();
        }

        assert_eq!(count, 0, "another spelling is another type to the store");
    }

    #[test]
    fn test_expand_multiple_source_nodes() {
        let (store, dyn_store) = test_store();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));
        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec![],
            ReadView::latest(),
        );

        let mut results = Vec::new();
        while let Ok(Some(chunk)) = expand.next() {
            for i in 0..chunk.row_count() {
                let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
                let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
                results.push((src, dst));
            }
        }

        // Both a->c and b->c
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_expand_empty_input() {
        let (_store, dyn_store) = test_store();

        // No nodes with this label
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Nonexistent",
            ReadView::latest(),
        ));
        let mut expand = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec![],
            ReadView::latest(),
        );

        let result = expand.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_expand_into_any() {
        let (_store, dyn_store) = test_store();
        let scan = Box::new(ScanOperator::with_label(
            Arc::clone(&dyn_store),
            "Person",
            ReadView::latest(),
        ));
        let op = ExpandOperator::new(
            Arc::clone(&dyn_store),
            scan,
            0,
            Direction::Outgoing,
            vec![],
            ReadView::latest(),
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ExpandOperator>().is_ok());
    }
}
