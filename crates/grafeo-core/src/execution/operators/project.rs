//! Project operator for selecting and transforming columns.

use super::filter::{ExpressionPredicate, FilterExpression, SessionContext};
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStoreSearch;
use crate::graph::lpg::{Edge, Node};
use grafeo_common::types::{
    EdgeId, EpochId, LogicalType, NodeId, PropertyKey, TransactionId, Value,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// A projection expression.
#[non_exhaustive]
pub enum ProjectExpr {
    /// Reference to an input column.
    Column(usize),
    /// A constant value.
    Constant(Value),
    /// Property access on a node/edge column.
    PropertyAccess {
        /// The column containing the node or edge ID.
        column: usize,
        /// The property name to access.
        property: String,
    },
    /// Edge type accessor (for type(r) function).
    EdgeType {
        /// The column containing the edge ID.
        column: usize,
    },
    /// Full expression evaluation (for CASE WHEN, etc.).
    Expression {
        /// The filter expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
    },
    /// Resolve a node ID column to a full node map with metadata and properties.
    NodeResolve {
        /// The column containing the node ID.
        column: usize,
    },
    /// Resolve an edge ID column to a full edge map with metadata and properties.
    EdgeResolve {
        /// The column containing the edge ID.
        column: usize,
    },
    /// Evaluates an expression whose value is a node, an edge or a list of
    /// them (as ids, like `relationships(p)` and the variable of a
    /// variable-length edge pattern) and returns node and edge maps, as
    /// [`NodeResolve`](Self::NodeResolve) and [`EdgeResolve`](Self::EdgeResolve)
    /// do for a column.
    Entities {
        /// The expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
        /// What the value holds.
        kind: EntityValue,
    },
    /// Returns the first non-null value from two columns (used for RIGHT/FULL join dedup).
    Coalesce {
        /// Primary column index.
        first: usize,
        /// Fallback column index.
        second: usize,
    },
}

/// What a [`ProjectExpr::Entities`] value holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntityValue {
    /// One node id.
    Node,
    /// One edge id.
    Edge,
    /// A list of node ids.
    Nodes,
    /// A list of edge ids.
    Edges,
}

/// A project operator that selects and transforms columns.
pub struct ProjectOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Projection expressions.
    projections: Vec<ProjectExpr>,
    /// Output column types.
    output_types: Vec<LogicalType>,
    /// Optional store for property access.
    store: Option<Arc<dyn GraphStoreSearch>>,
    /// Transaction ID for MVCC-aware property lookups.
    transaction_id: Option<TransactionId>,
    /// Viewing epoch for MVCC-aware property lookups.
    viewing_epoch: Option<EpochId>,
    /// Session context for introspection functions in expression evaluation.
    session_context: SessionContext,
}

impl ProjectOperator {
    /// Creates a new project operator.
    ///
    /// # Panics
    ///
    /// Panics if `projections` and `output_types` have different lengths.
    pub fn new(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: None,
            transaction_id: None,
            viewing_epoch: None,
            session_context: SessionContext::default(),
        }
    }

    /// Creates a new project operator with store access for property lookups.
    ///
    /// # Panics
    ///
    /// Panics if `projections` and `output_types` have different lengths.
    pub fn with_store(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
        store: Arc<dyn GraphStoreSearch>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: Some(store),
            transaction_id: None,
            viewing_epoch: None,
            session_context: SessionContext::default(),
        }
    }

    /// Sets the transaction context for MVCC-aware property lookups.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the session context for introspection functions.
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.session_context = context;
        self
    }

    /// Decomposes this operator into its child and projections for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<ProjectExpr>, Vec<LogicalType>) {
        (self.child, self.projections, self.output_types)
    }

    /// Creates a project operator that selects specific columns.
    pub fn select_columns(
        child: Box<dyn Operator>,
        columns: Vec<usize>,
        types: Vec<LogicalType>,
    ) -> Self {
        let projections = columns.into_iter().map(ProjectExpr::Column).collect();
        Self::new(child, projections, types)
    }
}

impl Operator for ProjectOperator {
    fn next(&mut self) -> OperatorResult {
        // Get next chunk from child
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        // Create output chunk
        let mut output = DataChunk::with_capacity(&self.output_types, input.row_count());

        // Evaluate each projection
        for (i, proj) in self.projections.iter().enumerate() {
            match proj {
                ProjectExpr::Column(col_idx) => {
                    // Copy column from input to output
                    let input_col = input.column(*col_idx).ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("Column {col_idx}"))
                    })?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    // Copy selected rows
                    for row in input.selected_indices() {
                        if let Some(value) = input_col.get_value(row) {
                            output_col.push_value(value);
                        }
                    }
                }
                ProjectExpr::Constant(value) => {
                    // Push constant for each row
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");
                    for _ in input.selected_indices() {
                        output_col.push_value(value.clone());
                    }
                }
                ProjectExpr::PropertyAccess { column, property } => {
                    // Access property from node/edge in the specified column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for property access".to_string())
                    })?;

                    // Extract property for each row.
                    // For typed columns (VectorData::NodeId / EdgeId) there is
                    // no ambiguity. For Generic/Any columns (e.g. after a hash
                    // join), both get_node_id and get_edge_id can succeed on the
                    // same Int64 value, so we verify against the store to resolve
                    // the entity type.
                    let prop_key = PropertyKey::new(property);
                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            let node = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.get_node_versioned(node_id, ep, tx)
                            } else if let Some(ep) = epoch {
                                store.get_node_at_epoch(node_id, ep)
                            } else {
                                store.get_node(node_id)
                            };
                            if let Some(prop) = node.and_then(|n| n.get_property(property).cloned())
                            {
                                prop
                            } else if let Some(edge_id) = input_col.get_edge_id(row) {
                                // Node lookup failed: the ID may belong to an
                                // edge (common with Generic columns after joins).
                                let edge = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                    store.get_edge_versioned(edge_id, ep, tx)
                                } else if let Some(ep) = epoch {
                                    store.get_edge_at_epoch(edge_id, ep)
                                } else {
                                    store.get_edge(edge_id)
                                };
                                edge.and_then(|e| e.get_property(property).cloned())
                                    .unwrap_or(Value::Null)
                            } else {
                                Value::Null
                            }
                        } else if let Some(edge_id) = input_col.get_edge_id(row) {
                            let edge = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.get_edge_versioned(edge_id, ep, tx)
                            } else if let Some(ep) = epoch {
                                store.get_edge_at_epoch(edge_id, ep)
                            } else {
                                store.get_edge(edge_id)
                            };
                            edge.and_then(|e| e.get_property(property).cloned())
                                .unwrap_or(Value::Null)
                        } else if let Some(Value::Map(map)) = input_col.get_value(row) {
                            map.get(&prop_key).cloned().unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::EdgeType { column } => {
                    // Get edge type string from an edge column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for edge type access".to_string())
                    })?;

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            let etype = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.edge_type_versioned(edge_id, ep, tx)
                            } else {
                                store.edge_type(edge_id)
                            };
                            etype.map_or(Value::Null, Value::String)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::Expression {
                    expr,
                    variable_columns,
                } => {
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution(
                            "Store required for expression evaluation".to_string(),
                        )
                    })?;

                    // Use the ExpressionPredicate for expression evaluation
                    let mut evaluator = ExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(store),
                    )
                    .with_session_context(self.session_context.clone());
                    if let (Some(ep), tx_id) = (self.viewing_epoch, self.transaction_id) {
                        evaluator = evaluator.with_transaction_context(ep, tx_id);
                    }

                    // A node or edge column holds ids: it takes a node or edge
                    // map (the items of `nodes(p)`) by its id, and anything
                    // that names no entity as null, never as entity 0.
                    let entities = matches!(
                        output_col.data_type(),
                        LogicalType::Node | LogicalType::Edge
                    );
                    for row in input.selected_indices() {
                        let value = evaluator.eval_at(&input, row).unwrap_or(Value::Null);
                        let value = if entities { entity_id(value) } else { value };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::NodeResolve { column } => {
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for node resolution".to_string())
                    })?;

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            let node = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.get_node_versioned(node_id, ep, tx)
                            } else if let Some(ep) = epoch {
                                store.get_node_at_epoch(node_id, ep)
                            } else {
                                store.get_node(node_id)
                            };
                            node.map_or(Value::Null, |n| node_to_map(&n))
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::EdgeResolve { column } => {
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution("Store required for edge resolution".to_string())
                    })?;

                    let epoch = self.viewing_epoch;
                    let tx_id = self.transaction_id;
                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            let edge = if let (Some(ep), Some(tx)) = (epoch, tx_id) {
                                store.get_edge_versioned(edge_id, ep, tx)
                            } else if let Some(ep) = epoch {
                                store.get_edge_at_epoch(edge_id, ep)
                            } else {
                                store.get_edge(edge_id)
                            };
                            edge.map_or(Value::Null, |e| edge_to_map(&e))
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::Entities {
                    expr,
                    variable_columns,
                    kind,
                } => {
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let store = self.store.as_ref().ok_or_else(|| {
                        OperatorError::Execution(
                            "Store required for expression evaluation".to_string(),
                        )
                    })?;

                    let mut evaluator = ExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(store),
                    )
                    .with_session_context(self.session_context.clone());
                    if let (Some(ep), tx_id) = (self.viewing_epoch, self.transaction_id) {
                        evaluator = evaluator.with_transaction_context(ep, tx_id);
                    }

                    for row in input.selected_indices() {
                        let value = evaluator.eval_at(&input, row).unwrap_or(Value::Null);
                        output_col.push_value(self.resolve_entities(store.as_ref(), value, *kind));
                    }
                }
                ProjectExpr::Coalesce { first, second } => {
                    let first_col = input
                        .column(*first)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {first}")))?;
                    let second_col = input
                        .column(*second)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {second}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    for row in input.selected_indices() {
                        let value = match first_col.get_value(row) {
                            Some(Value::Null) | None => {
                                second_col.get_value(row).unwrap_or(Value::Null)
                            }
                            Some(v) => v,
                        };
                        output_col.push_value(value);
                    }
                }
            }
        }

        output.set_count(input.row_count());
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "Project"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

impl ProjectOperator {
    /// Replaces the entity ids in `value` with node or edge maps. Ids of
    /// entities this read cannot see become null; values that are already
    /// maps (or null) are kept.
    fn resolve_entities(
        &self,
        store: &dyn GraphStoreSearch,
        value: Value,
        kind: EntityValue,
    ) -> Value {
        let node = |value: &Value| match value {
            Value::Int64(id) => u64::try_from(*id).ok().map_or(Value::Null, |id| {
                self.visible_node(store, NodeId::new(id))
                    .map_or(Value::Null, |n| node_to_map(&n))
            }),
            other => other.clone(),
        };
        let edge = |value: &Value| match value {
            Value::Int64(id) => u64::try_from(*id).ok().map_or(Value::Null, |id| {
                self.visible_edge(store, EdgeId::new(id))
                    .map_or(Value::Null, |e| edge_to_map(&e))
            }),
            other => other.clone(),
        };
        match (kind, &value) {
            (EntityValue::Node, _) => node(&value),
            (EntityValue::Edge, _) => edge(&value),
            (EntityValue::Nodes, Value::List(items)) => {
                Value::List(items.iter().map(node).collect::<Vec<_>>().into())
            }
            (EntityValue::Edges, Value::List(items)) => {
                Value::List(items.iter().map(edge).collect::<Vec<_>>().into())
            }
            _ => value,
        }
    }

    /// The node `id` as this operator's reads see it.
    fn visible_node(&self, store: &dyn GraphStoreSearch, id: NodeId) -> Option<Node> {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(tx)) => store.get_node_versioned(id, epoch, tx),
            (Some(epoch), None) => store.get_node_at_epoch(id, epoch),
            _ => store.get_node(id),
        }
    }

    /// The edge `id` as this operator's reads see it.
    fn visible_edge(&self, store: &dyn GraphStoreSearch, id: EdgeId) -> Option<Edge> {
        match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(tx)) => store.get_edge_versioned(id, epoch, tx),
            (Some(epoch), None) => store.get_edge_at_epoch(id, epoch),
            _ => store.get_edge(id),
        }
    }
}

/// Converts a [`Node`] to a `Value::Map` with metadata and properties.
///
/// The map contains `_id` (integer), `_labels` (list of strings), and
/// all node properties at the top level.
fn node_to_map(node: &Node) -> Value {
    let mut map = BTreeMap::new();
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let node_id_i64 = node.id.as_u64() as i64;
    map.insert(PropertyKey::new("_id"), Value::Int64(node_id_i64));
    let labels: Vec<Value> = node
        .labels
        .iter()
        .map(|l| Value::String(l.clone()))
        .collect();
    map.insert(PropertyKey::new("_labels"), Value::List(labels.into()));
    for (key, value) in &node.properties {
        map.insert(key.clone(), value.clone());
    }
    Value::Map(Arc::new(map))
}

/// Converts an [`Edge`] to a `Value::Map` with metadata and properties.
///
/// The map contains `_id`, `_type`, `_source`, `_target`, and all edge
/// properties at the top level.
fn edge_to_map(edge: &Edge) -> Value {
    let mut map = BTreeMap::new();
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let edge_id_i64 = edge.id.as_u64() as i64;
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let src_id_i64 = edge.src.as_u64() as i64;
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let dst_id_i64 = edge.dst.as_u64() as i64;
    map.insert(PropertyKey::new("_id"), Value::Int64(edge_id_i64));
    map.insert(
        PropertyKey::new("_type"),
        Value::String(edge.edge_type.clone()),
    );
    map.insert(PropertyKey::new("_source"), Value::Int64(src_id_i64));
    map.insert(PropertyKey::new("_target"), Value::Int64(dst_id_i64));
    for (key, value) in &edge.properties {
        map.insert(key.clone(), value.clone());
    }
    Value::Map(Arc::new(map))
}

/// The id a value gives a node or edge column: an id, the `_id` of an entity
/// map, or null.
fn entity_id(value: Value) -> Value {
    match value {
        Value::Int64(_) | Value::Null => value,
        Value::Map(map) => match map.get(&PropertyKey::new("_id")) {
            Some(id @ Value::Int64(_)) => id.clone(),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::graph::lpg::LpgStore;
    use grafeo_common::types::Value;

    struct MockScanOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl Operator for MockScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "MockScan"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn test_project_select_columns() {
        // Create input with 3 columns: [int, string, int]
        let mut builder =
            DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String, LogicalType::Int64]);

        builder.column_mut(0).unwrap().push_int64(1);
        builder.column_mut(1).unwrap().push_string("hello");
        builder.column_mut(2).unwrap().push_int64(100);
        builder.advance_row();

        builder.column_mut(0).unwrap().push_int64(2);
        builder.column_mut(1).unwrap().push_string("world");
        builder.column_mut(2).unwrap().push_int64(200);
        builder.advance_row();

        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Project to select columns 2 and 0 (reordering)
        let mut project = ProjectOperator::select_columns(
            Box::new(mock_scan),
            vec![2, 0],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = project.next().unwrap().unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 2);

        // Check values are reordered
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(100));
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(1));
    }

    #[test]
    fn test_project_constant() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_int64(2);
        builder.advance_row();

        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Project with a constant
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Constant(Value::String("constant".into())),
            ],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let result = project.next().unwrap().unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.column(1).unwrap().get_string(0), Some("constant"));
        assert_eq!(result.column(1).unwrap().get_string(1), Some("constant"));
    }

    #[test]
    fn test_project_empty_input() {
        let mock_scan = MockScanOperator {
            chunks: vec![],
            position: 0,
        };

        let mut project =
            ProjectOperator::select_columns(Box::new(mock_scan), vec![0], vec![LogicalType::Int64]);

        assert!(project.next().unwrap().is_none());
    }

    #[test]
    fn test_project_column_not_found() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Reference column index 5 which doesn't exist
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![ProjectExpr::Column(5)],
            vec![LogicalType::Int64],
        );

        let result = project.next();
        assert!(result.is_err(), "Should fail with ColumnNotFound");
    }

    #[test]
    fn test_project_multiple_constants() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![
                ProjectExpr::Constant(Value::Int64(42)),
                ProjectExpr::Constant(Value::String("fixed".into())),
                ProjectExpr::Constant(Value::Bool(true)),
            ],
            vec![LogicalType::Int64, LogicalType::String, LogicalType::Bool],
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column_count(), 3);
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(42));
        assert_eq!(result.column(1).unwrap().get_string(0), Some("fixed"));
        assert_eq!(
            result.column(2).unwrap().get_value(0),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn test_project_identity() {
        // Select all columns in original order
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
        builder.column_mut(0).unwrap().push_int64(10);
        builder.column_mut(1).unwrap().push_string("test");
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::select_columns(
            Box::new(mock_scan),
            vec![0, 1],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10));
        assert_eq!(result.column(1).unwrap().get_string(0), Some("test"));
    }

    #[test]
    fn test_project_name() {
        let mock_scan = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let project =
            ProjectOperator::select_columns(Box::new(mock_scan), vec![0], vec![LogicalType::Int64]);
        assert_eq!(project.name(), "Project");
    }

    /// A node or edge column takes an entity map by its `_id`; a value that
    /// names no entity is null there, never entity 0.
    #[test]
    fn test_project_entity_values_into_entity_columns() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        let mock_scan = MockScanOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let map = |entries: &[(&str, Value)]| {
            Value::Map(Arc::new(
                entries
                    .iter()
                    .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
                    .collect(),
            ))
        };
        let expression = |value: Value| ProjectExpr::Expression {
            expr: FilterExpression::Literal(value),
            variable_columns: HashMap::new(),
        };
        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![
                expression(map(&[("_id", Value::Int64(7))])),
                expression(map(&[("name", Value::from("Alix"))])),
                expression(Value::from("Alix")),
            ],
            vec![LogicalType::Node, LogicalType::Node, LogicalType::Edge],
            Arc::new(LpgStore::new().unwrap()),
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().get_node_id(0),
            Some(NodeId::new(7))
        );
        assert_eq!(result.column(1).unwrap().get_node_id(0), None);
        assert_eq!(result.column(2).unwrap().get_edge_id(0), None);
    }

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_project_node_resolve() {
        // Create a store with a test node
        let store = LpgStore::new().unwrap();
        let node_id = store.create_node(&["Person"]);
        store.set_node_property(node_id, "name", Value::String("Alix".into()));
        store.set_node_property(node_id, "age", Value::Int64(30));

        // Create input chunk with a NodeId column
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(node_id);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::NodeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column_count(), 1);

        let value = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::Map(map) = value {
            assert_eq!(
                map.get(&PropertyKey::new("_id")),
                Some(&Value::Int64(node_id.as_u64() as i64))
            );
            assert!(map.get(&PropertyKey::new("_labels")).is_some());
            assert_eq!(
                map.get(&PropertyKey::new("name")),
                Some(&Value::String("Alix".into()))
            );
            assert_eq!(map.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
        } else {
            panic!("Expected Value::Map, got {:?}", value);
        }
    }

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_project_edge_resolve() {
        let store = LpgStore::new().unwrap();
        let src = store.create_node(&["Person"]);
        let dst = store.create_node(&["Company"]);
        let edge_id = store.create_edge(src, dst, "WORKS_AT");
        store.set_edge_property(edge_id, "since", Value::Int64(2020));

        // Create input chunk with an EdgeId column
        let mut builder = DataChunkBuilder::new(&[LogicalType::Edge]);
        builder.column_mut(0).unwrap().push_edge_id(edge_id);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::EdgeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
        );

        let result = project.next().unwrap().unwrap();
        let value = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::Map(map) = value {
            assert_eq!(
                map.get(&PropertyKey::new("_id")),
                Some(&Value::Int64(edge_id.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_type")),
                Some(&Value::String("WORKS_AT".into()))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_source")),
                Some(&Value::Int64(src.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_target")),
                Some(&Value::Int64(dst.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("since")),
                Some(&Value::Int64(2020))
            );
        } else {
            panic!("Expected Value::Map, got {:?}", value);
        }
    }

    #[test]
    fn test_project_resolve_missing_entity() {
        use grafeo_common::types::NodeId;

        let store = LpgStore::new().unwrap();

        // Create input chunk with a NodeId that doesn't exist in the store
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder
            .column_mut(0)
            .unwrap()
            .push_node_id(NodeId::new(999));
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::NodeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().get_value(0), Some(Value::Null));
    }

    #[test]
    fn test_project_into_any() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let op = ProjectOperator::select_columns(Box::new(mock), vec![0], vec![LogicalType::Int64]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ProjectOperator>().is_ok());
    }

    #[test]
    fn test_project_into_parts() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let op = ProjectOperator::new(
            Box::new(mock),
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Constant(Value::Int64(1)),
            ],
            vec![LogicalType::Int64, LogicalType::Int64],
        );
        let (child, projections, output_types) = op.into_parts();
        assert_eq!(projections.len(), 2);
        assert_eq!(output_types.len(), 2);
        // Verify child is still functional
        let mut child = child;
        assert!(child.next().unwrap().is_none());
    }
}
