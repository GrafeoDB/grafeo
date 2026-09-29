//! Mutation operators for creating and deleting graph elements.
//!
//! These operators modify the graph structure:
//! - `CreateNodeOperator`: Creates new nodes
//! - `CreateEdgeOperator`: Creates new edges
//! - `DeleteNodeOperator`: Deletes nodes
//! - `DeleteEdgeOperator`: Deletes edges

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{
    EdgeId, EpochId, LogicalType, NodeId, PropertyKey, TransactionId, Value,
};

use super::filter::{ExpressionPredicate, FilterExpression};
use super::{GraphWriter, Operator, OperatorError, OperatorResult, SessionContext};
use crate::execution::chunk::{DataChunk, DataChunkBuilder};
use crate::graph::{GraphStore, GraphStoreSearch};

/// Trait for validating schema constraints during mutation operations.
///
/// Implementors check type definitions, NOT NULL, and UNIQUE constraints
/// before data is written to the store.
pub trait ConstraintValidator: Send + Sync {
    /// Validates a single property value for a node with the given labels.
    ///
    /// Checks type compatibility and NOT NULL constraints.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the value type is incompatible or a NOT NULL constraint is violated.
    fn validate_node_property(
        &self,
        labels: &[String],
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError>;

    /// Validates that all required properties are present after creating a node.
    ///
    /// Checks NOT NULL constraints for properties that were not explicitly set.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a required (NOT NULL) property is missing.
    fn validate_node_complete(
        &self,
        labels: &[String],
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError>;

    /// Checks UNIQUE constraint for a node property value.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a node with the same label already has this value.
    fn check_unique_node_property(
        &self,
        labels: &[String],
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError>;

    /// Checks the UNIQUE and NODE KEY constraints on several properties,
    /// which hold for the combination of values, against the full set of
    /// properties a node gets. `node` is the node itself, which the check
    /// leaves out. ([`check_unique_node_property`](Self::check_unique_node_property)
    /// covers constraints on one property.)
    ///
    /// # Errors
    ///
    /// Returns `Err` if another node with the constraint's label has the same
    /// values for all of its properties.
    fn check_unique_node(
        &self,
        labels: &[String],
        properties: &[(String, Value)],
        node: Option<NodeId>,
    ) -> Result<(), OperatorError> {
        let _ = (labels, properties, node);
        Ok(())
    }

    /// Validates a single property value for an edge of the given type.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the value type is incompatible with the edge schema.
    fn validate_edge_property(
        &self,
        edge_type: &str,
        key: &str,
        value: &Value,
    ) -> Result<(), OperatorError>;

    /// Validates that all required properties are present after creating an edge.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a required property is missing from the edge.
    fn validate_edge_complete(
        &self,
        edge_type: &str,
        properties: &[(String, Value)],
    ) -> Result<(), OperatorError>;

    /// Validates that the node labels are allowed by the bound graph type.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any label is not defined in the graph type.
    fn validate_node_labels_allowed(&self, labels: &[String]) -> Result<(), OperatorError> {
        let _ = labels;
        Ok(())
    }

    /// Validates that the edge type is allowed by the bound graph type.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the edge type is not defined in the graph type.
    fn validate_edge_type_allowed(&self, edge_type: &str) -> Result<(), OperatorError> {
        let _ = edge_type;
        Ok(())
    }

    /// Validates that edge endpoints have the correct node type labels.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the source or target node labels do not match the edge type definition.
    fn validate_edge_endpoints(
        &self,
        edge_type: &str,
        source_labels: &[String],
        target_labels: &[String],
    ) -> Result<(), OperatorError> {
        let _ = (edge_type, source_labels, target_labels);
        Ok(())
    }

    /// Injects default values for properties that are defined in a type but
    /// not explicitly provided.
    fn inject_defaults(&self, labels: &[String], properties: &mut Vec<(String, Value)>) {
        let _ = (labels, properties);
    }
}

/// Operator that creates new nodes.
///
/// For each input row, creates a new node with the specified labels
/// and properties, then outputs the row with the new node.
pub struct CreateNodeOperator {
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Input operator.
    input: Option<Box<dyn Operator>>,
    /// Labels for the new nodes.
    labels: Vec<String>,
    /// Properties to set (name -> column index or constant value).
    properties: Vec<(String, PropertySource)>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Column index for the created node variable.
    output_column: usize,
    /// Whether this operator has been executed (for no-input case).
    executed: bool,
    /// Evaluates computed property values (`PropertySource::Expression`).
    expressions: PropertyExpressions,
}

/// Evaluators for the computed property values (`PropertySource::Expression`)
/// of a CREATE or MERGE operator, such as `{id: toString(i)}`.
///
/// Compiled on first use: the search store and transaction context are
/// attached to the operator after construction.
#[derive(Default)]
pub(super) struct PropertyExpressions {
    search_store: Option<Arc<dyn GraphStoreSearch>>,
    session_context: SessionContext,
    compiled: Option<Vec<Option<ExpressionPredicate>>>,
}

impl PropertyExpressions {
    /// Creates evaluators that use `search_store` and `session_context`.
    pub(super) fn new(
        search_store: Option<Arc<dyn GraphStoreSearch>>,
        session_context: SessionContext,
    ) -> Self {
        Self {
            search_store,
            session_context,
            compiled: None,
        }
    }

    /// Resolves every property for one input row, evaluating computed values
    /// against that row.
    pub(super) fn resolve_row(
        &mut self,
        properties: &[(String, PropertySource)],
        chunk: &DataChunk,
        row: usize,
        store: &dyn GraphStore,
        viewing_epoch: Option<EpochId>,
        transaction_id: Option<TransactionId>,
    ) -> Result<Vec<(String, Value)>, OperatorError> {
        if self.compiled.is_none() {
            let mut compiled = Vec::with_capacity(properties.len());
            for (_, source) in properties {
                let PropertySource::Expression {
                    expr,
                    variable_columns,
                } = source
                else {
                    compiled.push(None);
                    continue;
                };
                let search_store = self.search_store.as_ref().ok_or_else(|| {
                    OperatorError::Execution(
                        "computed property value requires a search store; planner did not attach one"
                            .to_string(),
                    )
                })?;
                let mut predicate = ExpressionPredicate::new(
                    (**expr).clone(),
                    variable_columns.clone(),
                    Arc::clone(search_store),
                )
                .with_session_context(self.session_context.clone());
                if let Some(epoch) = viewing_epoch {
                    predicate = predicate.with_transaction_context(epoch, transaction_id);
                }
                compiled.push(Some(predicate));
            }
            self.compiled = Some(compiled);
        }
        let compiled = self.compiled.as_deref().unwrap_or_default();
        Ok(properties
            .iter()
            .enumerate()
            .map(|(i, (name, source))| {
                let value = match compiled.get(i).and_then(Option::as_ref) {
                    Some(predicate) => predicate.eval_at(chunk, row).unwrap_or(Value::Null),
                    None => source.resolve(chunk, row, store),
                };
                (name.clone(), value)
            })
            .collect())
    }
}

/// Source for a property value.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PropertySource {
    /// Get value from an input column.
    Column(usize),
    /// Use a constant value.
    Constant(Value),
    /// Access a named property from a map/node/edge in an input column.
    PropertyAccess {
        /// The column containing the map, node ID, or edge ID.
        column: usize,
        /// The property name to extract.
        property: String,
    },
    /// A value computed per row, such as `toString(i)` or `i * 2` in
    /// `CREATE (:X {id: toString(i)})`. Also used by `MERGE ... ON CREATE/MATCH
    /// SET`, where the expression may reference the MERGE variable, which only
    /// exists after the merge resolves.
    ///
    /// `resolve` cannot evaluate this variant and returns `Value::Null`: the
    /// CREATE and MERGE operators evaluate it (see `PropertyExpressions`).
    Expression {
        /// The expression to evaluate against an augmented row.
        expr: Box<FilterExpression>,
        /// Variable-name to column-index map for the augmented row layout.
        variable_columns: HashMap<String, usize>,
    },
}

impl PropertySource {
    /// Resolves a property value from a data chunk row.
    ///
    /// Returns `Value::Null` for [`PropertySource::Expression`]: those sources
    /// require an operator-specific augmented row and must be intercepted by
    /// the producing operator (currently the MERGE node and edge operators).
    pub fn resolve(
        &self,
        chunk: &crate::execution::chunk::DataChunk,
        row: usize,
        store: &dyn GraphStore,
    ) -> Value {
        match self {
            PropertySource::Column(col_idx) => chunk
                .column(*col_idx)
                .and_then(|c| c.get_value(row))
                .unwrap_or(Value::Null),
            PropertySource::Constant(v) => v.clone(),
            PropertySource::PropertyAccess { column, property } => {
                let Some(col) = chunk.column(*column) else {
                    return Value::Null;
                };
                // Try node ID first, then edge ID, then map value
                if let Some(node_id) = col.get_node_id(row) {
                    store
                        .get_node(node_id)
                        .and_then(|node| node.get_property(property).cloned())
                        .unwrap_or(Value::Null)
                } else if let Some(edge_id) = col.get_edge_id(row) {
                    store
                        .get_edge(edge_id)
                        .and_then(|edge| edge.get_property(property).cloned())
                        .unwrap_or(Value::Null)
                } else if let Some(Value::Map(map)) = col.get_value(row) {
                    let key = PropertyKey::new(property);
                    map.get(&key).cloned().unwrap_or(Value::Null)
                } else {
                    Value::Null
                }
            }
            // Expression sources require an augmented row built by the producer.
            // Reaching this branch means an operator forgot to intercept it.
            PropertySource::Expression { .. } => Value::Null,
        }
    }
}

impl CreateNodeOperator {
    /// Creates a new node creation operator.
    ///
    /// # Arguments
    /// * `writer` - Writes the nodes; a store alone writes without a
    ///   transaction, checks or conflict tracking.
    /// * `input` - Optional input operator (None for standalone CREATE).
    /// * `labels` - Labels to assign to created nodes.
    /// * `properties` - Properties to set on created nodes.
    /// * `output_schema` - Schema of the output.
    /// * `output_column` - Column index where the created node ID goes.
    pub fn new(
        writer: impl Into<GraphWriter>,
        input: Option<Box<dyn Operator>>,
        labels: Vec<String>,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
        output_column: usize,
    ) -> Self {
        Self {
            writer: writer.into(),
            input,
            labels,
            properties,
            output_schema,
            output_column,
            executed: false,
            expressions: PropertyExpressions::default(),
        }
    }

    /// Provides a search-store handle so computed property values
    /// (`PropertySource::Expression`) can be evaluated.
    #[must_use]
    pub fn with_search_store(mut self, search_store: Arc<dyn GraphStoreSearch>) -> Self {
        self.expressions.search_store = Some(search_store);
        self
    }

    /// Sets the session context used when evaluating computed property values.
    #[must_use]
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.expressions.session_context = context;
        self
    }
}

impl Operator for CreateNodeOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(ref mut input) = self.input {
            // For each input row, create a node
            let Some(chunk) = input.next()? else {
                return Ok(None);
            };
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let properties = self.expressions.resolve_row(
                    &self.properties,
                    &chunk,
                    row,
                    self.writer.store().as_ref() as &dyn GraphStore,
                    self.writer.viewing_epoch(),
                    self.writer.transaction_id(),
                )?;
                let node_id = self.writer.create_node(&self.labels, properties)?;

                // The input columns before the new node's column, then the node.
                copy_columns(&chunk, row, &mut builder, self.output_column);
                if let Some(dst) = builder.column_mut(self.output_column) {
                    dst.push_value(id_value(node_id.0));
                }
                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }

        // No input: create a single node
        if self.executed {
            return Ok(None);
        }
        self.executed = true;

        // Resolve constant properties. Computed values need a row: the
        // planner gives such a CREATE a single-row input instead.
        let properties: Vec<(String, Value)> = self
            .properties
            .iter()
            .filter_map(|(name, source)| {
                if let PropertySource::Constant(value) = source {
                    Some((name.clone(), value.clone()))
                } else {
                    None
                }
            })
            .collect();
        let node_id = self.writer.create_node(&self.labels, properties)?;

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);
        if let Some(dst) = builder.column_mut(self.output_column) {
            dst.push_value(id_value(node_id.0));
        }
        builder.advance_row();

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        if let Some(ref mut input) = self.input {
            input.reset();
        }
        self.executed = false;
    }

    fn name(&self) -> &'static str {
        "CreateNode"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Reads the entity id in `column` of `row`, for the error messages naming
/// the column (`from`, `node`, ...) and the id kind (`node`, `edge`, ...).
fn id_at(
    chunk: &DataChunk,
    column: usize,
    row: usize,
    column_name: &str,
    id_kind: &str,
) -> Result<u64, OperatorError> {
    let value = chunk
        .column(column)
        .and_then(|c| c.get_value(row))
        .ok_or_else(|| OperatorError::ColumnNotFound(format!("{column_name} column {column}")))?;
    match value {
        // Ids travel as the bits of an i64.
        Value::Int64(id) => Ok(id.cast_unsigned()),
        other => Err(OperatorError::TypeMismatch {
            expected: format!("Int64 ({id_kind} ID)"),
            found: format!("{other:?}"),
        }),
    }
}

/// Copies the first `columns` input columns of `row` to the output row.
fn copy_columns(chunk: &DataChunk, row: usize, builder: &mut DataChunkBuilder, columns: usize) {
    for col_idx in 0..columns.min(chunk.column_count()) {
        if let (Some(src), Some(dst)) = (chunk.column(col_idx), builder.column_mut(col_idx)) {
            dst.push_value(src.get_value(row).unwrap_or(Value::Null));
        }
    }
}

/// Encodes an entity id for an output column.
fn id_value(id: u64) -> Value {
    Value::Int64(id.cast_signed())
}

/// Operator that creates new edges.
pub struct CreateEdgeOperator {
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Input operator.
    input: Box<dyn Operator>,
    /// Column index for the source node.
    from_column: usize,
    /// Column index for the target node.
    to_column: usize,
    /// Edge type.
    edge_type: String,
    /// Properties to set.
    properties: Vec<(String, PropertySource)>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Column index for the created edge variable (if any).
    output_column: Option<usize>,
    /// Evaluates computed property values (`PropertySource::Expression`).
    expressions: PropertyExpressions,
}

impl CreateEdgeOperator {
    /// Creates a new edge creation operator.
    ///
    /// Use builder methods to set additional options:
    /// - [`with_properties`](Self::with_properties) - set edge properties
    /// - [`with_output_column`](Self::with_output_column) - output the created edge ID
    pub fn new(
        writer: impl Into<GraphWriter>,
        input: Box<dyn Operator>,
        from_column: usize,
        to_column: usize,
        edge_type: String,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            writer: writer.into(),
            input,
            from_column,
            to_column,
            edge_type,
            properties: Vec::new(),
            output_schema,
            output_column: None,
            expressions: PropertyExpressions::default(),
        }
    }

    /// Sets the properties to assign to created edges.
    pub fn with_properties(mut self, properties: Vec<(String, PropertySource)>) -> Self {
        self.properties = properties;
        self
    }

    /// Sets the output column for the created edge ID.
    pub fn with_output_column(mut self, column: usize) -> Self {
        self.output_column = Some(column);
        self
    }

    /// Provides a search-store handle so computed property values
    /// (`PropertySource::Expression`) can be evaluated.
    #[must_use]
    pub fn with_search_store(mut self, search_store: Arc<dyn GraphStoreSearch>) -> Self {
        self.expressions.search_store = Some(search_store);
        self
    }

    /// Sets the session context used when evaluating computed property values.
    #[must_use]
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.expressions.session_context = context;
        self
    }
}

impl Operator for CreateEdgeOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

        for row in chunk.selected_indices() {
            let from = NodeId(id_at(&chunk, self.from_column, row, "from", "node")?);
            let to = NodeId(id_at(&chunk, self.to_column, row, "to", "node")?);
            let properties = self.expressions.resolve_row(
                &self.properties,
                &chunk,
                row,
                self.writer.store().as_ref() as &dyn GraphStore,
                self.writer.viewing_epoch(),
                self.writer.transaction_id(),
            )?;
            let edge_id = self
                .writer
                .create_edge(from, to, &self.edge_type, properties)?;

            copy_columns(&chunk, row, &mut builder, chunk.column_count());
            if let Some(out_col) = self.output_column
                && let Some(dst) = builder.column_mut(out_col)
            {
                dst.push_value(id_value(edge_id.0));
            }
            builder.advance_row();
        }

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "CreateEdge"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that deletes nodes.
pub struct DeleteNodeOperator {
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Input operator.
    input: Box<dyn Operator>,
    /// Column index for the node to delete.
    node_column: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Whether to detach (delete connected edges) before deleting.
    detach: bool,
}

impl DeleteNodeOperator {
    /// Creates a new node deletion operator.
    pub fn new(
        writer: impl Into<GraphWriter>,
        input: Box<dyn Operator>,
        node_column: usize,
        output_schema: Vec<LogicalType>,
        detach: bool,
    ) -> Self {
        Self {
            writer: writer.into(),
            input,
            node_column,
            output_schema,
            detach,
        }
    }
}

impl Operator for DeleteNodeOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

        for row in chunk.selected_indices() {
            let node_id = NodeId(id_at(&chunk, self.node_column, row, "node", "node")?);
            self.writer.delete_node(node_id, self.detach)?;

            // Pass through all input columns so downstream RETURN can
            // reference the variable (e.g., count(n) after DELETE n).
            copy_columns(&chunk, row, &mut builder, chunk.column_count());
            builder.advance_row();
        }

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "DeleteNode"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that deletes edges.
pub struct DeleteEdgeOperator {
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Input operator.
    input: Box<dyn Operator>,
    /// Column index for the edge to delete.
    edge_column: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
}

impl DeleteEdgeOperator {
    /// Creates a new edge deletion operator.
    pub fn new(
        writer: impl Into<GraphWriter>,
        input: Box<dyn Operator>,
        edge_column: usize,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            writer: writer.into(),
            input,
            edge_column,
            output_schema,
        }
    }
}

impl Operator for DeleteEdgeOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

        for row in chunk.selected_indices() {
            let edge_id = EdgeId(id_at(&chunk, self.edge_column, row, "edge", "edge")?);
            self.writer.delete_edge(edge_id)?;
            copy_columns(&chunk, row, &mut builder, chunk.column_count());
            builder.advance_row();
        }

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "DeleteEdge"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that adds labels to nodes.
pub struct AddLabelOperator {
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Child operator providing nodes.
    input: Box<dyn Operator>,
    /// Column index containing node IDs.
    node_column: usize,
    /// Labels to add.
    labels: Vec<String>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Column index for the update count (last column).
    count_column: usize,
}

impl AddLabelOperator {
    /// Creates a new add label operator.
    pub fn new(
        writer: impl Into<GraphWriter>,
        input: Box<dyn Operator>,
        node_column: usize,
        labels: Vec<String>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        let count_column = output_schema.len() - 1;
        Self {
            writer: writer.into(),
            input,
            node_column,
            labels,
            count_column,
            output_schema,
        }
    }
}

impl Operator for AddLabelOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

        for row in chunk.selected_indices() {
            let node_id = NodeId(id_at(&chunk, self.node_column, row, "node", "node")?);
            let added = self.writer.add_labels(node_id, &self.labels)?;

            copy_columns(&chunk, row, &mut builder, chunk.column_count());
            if let Some(dst) = builder.column_mut(self.count_column) {
                dst.push_value(Value::Int64(i64::try_from(added).unwrap_or(i64::MAX)));
            }
            builder.advance_row();
        }

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "AddLabel"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that removes labels from nodes.
pub struct RemoveLabelOperator {
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Child operator providing nodes.
    input: Box<dyn Operator>,
    /// Column index containing node IDs.
    node_column: usize,
    /// Labels to remove.
    labels: Vec<String>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Column index for the update count (last column).
    count_column: usize,
}

impl RemoveLabelOperator {
    /// Creates a new remove label operator.
    pub fn new(
        writer: impl Into<GraphWriter>,
        input: Box<dyn Operator>,
        node_column: usize,
        labels: Vec<String>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        let count_column = output_schema.len() - 1;
        Self {
            writer: writer.into(),
            input,
            node_column,
            labels,
            count_column,
            output_schema,
        }
    }
}

impl Operator for RemoveLabelOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

        for row in chunk.selected_indices() {
            let node_id = NodeId(id_at(&chunk, self.node_column, row, "node", "node")?);
            let removed = self.writer.remove_labels(node_id, &self.labels)?;

            copy_columns(&chunk, row, &mut builder, chunk.column_count());
            if let Some(dst) = builder.column_mut(self.count_column) {
                dst.push_value(Value::Int64(i64::try_from(removed).unwrap_or(i64::MAX)));
            }
            builder.advance_row();
        }

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RemoveLabel"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Operator that sets properties on nodes or edges.
///
/// This operator reads node/edge IDs from a column and sets the
/// specified properties on each entity.
pub struct SetPropertyOperator {
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Child operator providing entities.
    input: Box<dyn Operator>,
    /// Column index containing entity IDs (node or edge).
    entity_column: usize,
    /// Whether the entity is an edge (false = node).
    is_edge: bool,
    /// Properties to set (name -> source).
    properties: Vec<(String, PropertySource)>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Whether to replace all properties (true) or merge (false) for map assignments.
    replace: bool,
}

impl SetPropertyOperator {
    /// Creates a new set property operator for nodes.
    pub fn new_for_node(
        writer: impl Into<GraphWriter>,
        input: Box<dyn Operator>,
        node_column: usize,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            writer: writer.into(),
            input,
            entity_column: node_column,
            is_edge: false,
            properties,
            output_schema,
            replace: false,
        }
    }

    /// Creates a new set property operator for edges.
    pub fn new_for_edge(
        writer: impl Into<GraphWriter>,
        input: Box<dyn Operator>,
        edge_column: usize,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            writer: writer.into(),
            input,
            entity_column: edge_column,
            is_edge: true,
            properties,
            output_schema,
            replace: false,
        }
    }

    /// Sets whether this operator replaces all properties (for map assignment).
    pub fn with_replace(mut self, replace: bool) -> Self {
        self.replace = replace;
        self
    }
}

impl Operator for SetPropertyOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(chunk) = self.input.next()? else {
            return Ok(None);
        };
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

        for row in chunk.selected_indices() {
            let entity_id = id_at(&chunk, self.entity_column, row, "entity", "entity")?;
            let store = self.writer.store().as_ref() as &dyn GraphStore;
            let assignments: Vec<(String, Value)> = self
                .properties
                .iter()
                .map(|(name, source)| (name.clone(), source.resolve(&chunk, row, store)))
                .collect();
            if self.is_edge {
                self.writer
                    .set_edge_properties(EdgeId(entity_id), &assignments, self.replace)?;
            } else {
                self.writer
                    .set_node_properties(NodeId(entity_id), &assignments, self.replace)?;
            }

            copy_columns(&chunk, row, &mut builder, chunk.column_count());
            builder.advance_row();
        }

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "SetProperty"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::DataChunk;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::graph::GraphStoreMut;
    use crate::graph::lpg::LpgStore;

    // ── Helpers ────────────────────────────────────────────────────

    fn create_test_store() -> Arc<dyn GraphStoreMut> {
        Arc::new(LpgStore::new().unwrap())
    }

    struct MockInput {
        chunk: Option<DataChunk>,
    }

    impl MockInput {
        fn boxed(chunk: DataChunk) -> Box<Self> {
            Box::new(Self { chunk: Some(chunk) })
        }
    }

    impl Operator for MockInput {
        fn next(&mut self) -> OperatorResult {
            Ok(self.chunk.take())
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "MockInput"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    struct EmptyInput;
    impl Operator for EmptyInput {
        fn next(&mut self) -> OperatorResult {
            Ok(None)
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "EmptyInput"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn node_id_chunk(ids: &[NodeId]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for id in ids {
            builder.column_mut(0).unwrap().push_int64(id.0 as i64);
            builder.advance_row();
        }
        builder.finish()
    }

    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn edge_id_chunk(ids: &[EdgeId]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for id in ids {
            builder.column_mut(0).unwrap().push_int64(id.0 as i64);
            builder.advance_row();
        }
        builder.finish()
    }

    // ── CreateNodeOperator ──────────────────────────────────────

    #[test]
    fn test_create_node_standalone() {
        let store = create_test_store();

        let mut op = CreateNodeOperator::new(
            Arc::clone(&store),
            None,
            vec!["Person".to_string()],
            vec![(
                "name".to_string(),
                PropertySource::Constant(Value::String("Alix".into())),
            )],
            vec![LogicalType::Int64],
            0,
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Second call should return None (standalone executes once)
        assert!(op.next().unwrap().is_none());

        assert_eq!(store.node_count(), 1);
    }

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_create_edge() {
        let store = create_test_store();

        let node1 = store.create_node(&["Person"]);
        let node2 = store.create_node(&["Person"]);

        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(node1.0 as i64);
        builder.column_mut(1).unwrap().push_int64(node2.0 as i64);
        builder.advance_row();

        let mut op = CreateEdgeOperator::new(
            Arc::clone(&store),
            MockInput::boxed(builder.finish()),
            0,
            1,
            "KNOWS".to_string(),
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let _chunk = op.next().unwrap().unwrap();
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_delete_node() {
        let store = create_test_store();

        let node_id = store.create_node(&["Person"]);
        assert_eq!(store.node_count(), 1);

        let mut op = DeleteNodeOperator::new(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node_id])),
            0,
            vec![LogicalType::Node],
            false,
        );

        let chunk = op.next().unwrap().unwrap();
        // Pass-through: output row contains the original node ID
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(store.node_count(), 0);
    }

    // ── DeleteEdgeOperator ───────────────────────────────────────

    #[test]
    fn test_delete_edge() {
        let store = create_test_store();

        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        let eid = store.create_edge(n1, n2, "KNOWS");
        assert_eq!(store.edge_count(), 1);

        let mut op = DeleteEdgeOperator::new(
            Arc::clone(&store),
            MockInput::boxed(edge_id_chunk(&[eid])),
            0,
            vec![LogicalType::Node],
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn test_delete_edge_no_input_returns_none() {
        let store = create_test_store();

        let mut op = DeleteEdgeOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec![LogicalType::Int64],
        );

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_delete_multiple_edges() {
        let store = create_test_store();

        let n1 = store.create_node(&["N"]);
        let n2 = store.create_node(&["N"]);
        let e1 = store.create_edge(n1, n2, "R");
        let e2 = store.create_edge(n2, n1, "S");
        assert_eq!(store.edge_count(), 2);

        let mut op = DeleteEdgeOperator::new(
            Arc::clone(&store),
            MockInput::boxed(edge_id_chunk(&[e1, e2])),
            0,
            vec![LogicalType::Node],
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2);
        assert_eq!(store.edge_count(), 0);
    }

    // ── DeleteNodeOperator with DETACH ───────────────────────────

    #[test]
    fn test_delete_node_detach() {
        let store = create_test_store();

        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        store.create_edge(n1, n2, "KNOWS");
        store.create_edge(n2, n1, "FOLLOWS");
        assert_eq!(store.edge_count(), 2);

        let mut op = DeleteNodeOperator::new(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[n1])),
            0,
            vec![LogicalType::Node],
            true, // detach = true
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(store.node_count(), 1);
        assert_eq!(store.edge_count(), 0); // edges detached
    }

    // ── AddLabelOperator ─────────────────────────────────────────

    #[test]
    fn test_add_label() {
        let store = create_test_store();

        let node = store.create_node(&["Person"]);

        let mut op = AddLabelOperator::new(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node])),
            0,
            vec!["Employee".to_string()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let chunk = op.next().unwrap().unwrap();
        let updated = chunk.column(1).unwrap().get_int64(0).unwrap();
        assert_eq!(updated, 1);

        // Verify label was added
        let node_data = store.get_node(node).unwrap();
        let labels: Vec<&str> = node_data.labels.iter().map(|l| l.as_ref()).collect();
        assert!(labels.contains(&"Person"));
        assert!(labels.contains(&"Employee"));
    }

    #[test]
    fn test_add_multiple_labels() {
        let store = create_test_store();

        let node = store.create_node(&["Base"]);

        let mut op = AddLabelOperator::new(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node])),
            0,
            vec!["LabelA".to_string(), "LabelB".to_string()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let chunk = op.next().unwrap().unwrap();
        let updated = chunk.column(1).unwrap().get_int64(0).unwrap();
        assert_eq!(updated, 2); // 2 labels added

        let node_data = store.get_node(node).unwrap();
        let labels: Vec<&str> = node_data.labels.iter().map(|l| l.as_ref()).collect();
        assert!(labels.contains(&"LabelA"));
        assert!(labels.contains(&"LabelB"));
    }

    #[test]
    fn test_add_label_no_input_returns_none() {
        let store = create_test_store();

        let mut op = AddLabelOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec!["Foo".to_string()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        assert!(op.next().unwrap().is_none());
    }

    // ── RemoveLabelOperator ──────────────────────────────────────

    #[test]
    fn test_remove_label() {
        let store = create_test_store();

        let node = store.create_node(&["Person", "Employee"]);

        let mut op = RemoveLabelOperator::new(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node])),
            0,
            vec!["Employee".to_string()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let chunk = op.next().unwrap().unwrap();
        let updated = chunk.column(1).unwrap().get_int64(0).unwrap();
        assert_eq!(updated, 1);

        // Verify label was removed
        let node_data = store.get_node(node).unwrap();
        let labels: Vec<&str> = node_data.labels.iter().map(|l| l.as_ref()).collect();
        assert!(labels.contains(&"Person"));
        assert!(!labels.contains(&"Employee"));
    }

    #[test]
    fn test_remove_nonexistent_label() {
        let store = create_test_store();

        let node = store.create_node(&["Person"]);

        let mut op = RemoveLabelOperator::new(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node])),
            0,
            vec!["NonExistent".to_string()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let chunk = op.next().unwrap().unwrap();
        let updated = chunk.column(1).unwrap().get_int64(0).unwrap();
        assert_eq!(updated, 0); // nothing removed
    }

    // ── SetPropertyOperator ──────────────────────────────────────

    #[test]
    fn test_set_node_property_constant() {
        let store = create_test_store();

        let node = store.create_node(&["Person"]);

        let mut op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node])),
            0,
            vec![(
                "name".to_string(),
                PropertySource::Constant(Value::String("Alix".into())),
            )],
            vec![LogicalType::Int64],
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Verify property was set
        let node_data = store.get_node(node).unwrap();
        assert_eq!(
            node_data
                .properties
                .get(&grafeo_common::types::PropertyKey::new("name")),
            Some(&Value::String("Alix".into()))
        );
    }

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_set_node_property_from_column() {
        let store = create_test_store();

        let node = store.create_node(&["Person"]);

        // Input: column 0 = node ID, column 1 = property value
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
        builder.column_mut(0).unwrap().push_int64(node.0 as i64);
        builder
            .column_mut(1)
            .unwrap()
            .push_value(Value::String("Gus".into()));
        builder.advance_row();

        let mut op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            MockInput::boxed(builder.finish()),
            0,
            vec![("name".to_string(), PropertySource::Column(1))],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        let node_data = store.get_node(node).unwrap();
        assert_eq!(
            node_data
                .properties
                .get(&grafeo_common::types::PropertyKey::new("name")),
            Some(&Value::String("Gus".into()))
        );
    }

    #[test]
    fn test_set_edge_property() {
        let store = create_test_store();

        let n1 = store.create_node(&["N"]);
        let n2 = store.create_node(&["N"]);
        let eid = store.create_edge(n1, n2, "KNOWS");

        let mut op = SetPropertyOperator::new_for_edge(
            Arc::clone(&store),
            MockInput::boxed(edge_id_chunk(&[eid])),
            0,
            vec![(
                "weight".to_string(),
                PropertySource::Constant(Value::Float64(0.75)),
            )],
            vec![LogicalType::Int64],
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        let edge_data = store.get_edge(eid).unwrap();
        assert_eq!(
            edge_data
                .properties
                .get(&grafeo_common::types::PropertyKey::new("weight")),
            Some(&Value::Float64(0.75))
        );
    }

    #[test]
    fn test_set_multiple_properties() {
        let store = create_test_store();

        let node = store.create_node(&["Person"]);

        let mut op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node])),
            0,
            vec![
                (
                    "name".to_string(),
                    PropertySource::Constant(Value::String("Alix".into())),
                ),
                (
                    "age".to_string(),
                    PropertySource::Constant(Value::Int64(30)),
                ),
            ],
            vec![LogicalType::Int64],
        );

        op.next().unwrap().unwrap();

        let node_data = store.get_node(node).unwrap();
        assert_eq!(
            node_data
                .properties
                .get(&grafeo_common::types::PropertyKey::new("name")),
            Some(&Value::String("Alix".into()))
        );
        assert_eq!(
            node_data
                .properties
                .get(&grafeo_common::types::PropertyKey::new("age")),
            Some(&Value::Int64(30))
        );
    }

    #[test]
    fn test_set_property_no_input_returns_none() {
        let store = create_test_store();

        let mut op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec![("x".to_string(), PropertySource::Constant(Value::Int64(1)))],
            vec![LogicalType::Int64],
        );

        assert!(op.next().unwrap().is_none());
    }

    // ── Error paths ──────────────────────────────────────────────

    #[test]
    fn test_delete_node_without_detach_errors_when_edges_exist() {
        let store = create_test_store();

        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        store.create_edge(n1, n2, "KNOWS");

        let mut op = DeleteNodeOperator::new(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[n1])),
            0,
            vec![LogicalType::Int64],
            false, // no detach
        );

        let err = op.next().unwrap_err();
        match err {
            OperatorError::ConstraintViolation(msg) => {
                assert!(msg.contains("connected edge"), "unexpected message: {msg}");
            }
            other => panic!("expected ConstraintViolation, got {other:?}"),
        }
        // Node should still exist
        assert_eq!(store.node_count(), 2);
    }

    // ── CreateNodeOperator with input ───────────────────────────

    #[test]
    fn test_create_node_with_input_operator() {
        let store = create_test_store();

        // Seed node to provide input rows
        let existing = store.create_node(&["Seed"]);

        let mut op = CreateNodeOperator::new(
            Arc::clone(&store),
            Some(MockInput::boxed(node_id_chunk(&[existing]))),
            vec!["Created".to_string()],
            vec![(
                "source".to_string(),
                PropertySource::Constant(Value::String("from_input".into())),
            )],
            vec![LogicalType::Int64, LogicalType::Int64], // input col + output col
            1,                                            // output column for new node ID
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Should have created one new node (2 total: Seed + Created)
        assert_eq!(store.node_count(), 2);

        // Exhausted
        assert!(op.next().unwrap().is_none());
    }

    // ── CreateEdgeOperator with properties and output column ────

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn test_create_edge_with_properties_and_output_column() {
        let store = create_test_store();

        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);

        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(n1.0 as i64);
        builder.column_mut(1).unwrap().push_int64(n2.0 as i64);
        builder.advance_row();

        let mut op = CreateEdgeOperator::new(
            Arc::clone(&store),
            MockInput::boxed(builder.finish()),
            0,
            1,
            "KNOWS".to_string(),
            vec![LogicalType::Int64, LogicalType::Int64, LogicalType::Int64],
        )
        .with_properties(vec![(
            "since".to_string(),
            PropertySource::Constant(Value::Int64(2024)),
        )])
        .with_output_column(2);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(store.edge_count(), 1);

        // Verify the output chunk contains the edge ID in column 2
        let edge_id_raw = chunk
            .column(2)
            .and_then(|c| c.get_int64(0))
            .expect("edge ID should be in output column 2");
        let edge_id = EdgeId(edge_id_raw as u64);

        // Verify the edge has the property
        let edge = store.get_edge(edge_id).expect("edge should exist");
        assert_eq!(
            edge.properties
                .get(&grafeo_common::types::PropertyKey::new("since")),
            Some(&Value::Int64(2024))
        );
    }

    // ── SetPropertyOperator with map replacement ────────────────

    #[test]
    fn test_set_property_map_replace() {
        use std::collections::BTreeMap;

        let store = create_test_store();

        let node = store.create_node(&["Person"]);
        store.set_node_property(node, "old_prop", Value::String("should_be_removed".into()));

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::new("new_key"), Value::String("new_val".into()));

        let mut op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node])),
            0,
            vec![(
                "*".to_string(),
                PropertySource::Constant(Value::Map(Arc::new(map))),
            )],
            vec![LogicalType::Int64],
        )
        .with_replace(true);

        op.next().unwrap().unwrap();

        let node_data = store.get_node(node).unwrap();
        // Old property should be gone
        assert!(
            node_data
                .properties
                .get(&PropertyKey::new("old_prop"))
                .is_none()
        );
        // New property should exist
        assert_eq!(
            node_data.properties.get(&PropertyKey::new("new_key")),
            Some(&Value::String("new_val".into()))
        );
    }

    // ── SetPropertyOperator with map merge (no replace) ─────────

    #[test]
    fn test_set_property_map_merge() {
        use std::collections::BTreeMap;

        let store = create_test_store();

        let node = store.create_node(&["Person"]);
        store.set_node_property(node, "existing", Value::Int64(42));

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::new("added"), Value::String("hello".into()));

        let mut op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            MockInput::boxed(node_id_chunk(&[node])),
            0,
            vec![(
                "*".to_string(),
                PropertySource::Constant(Value::Map(Arc::new(map))),
            )],
            vec![LogicalType::Int64],
        ); // replace defaults to false

        op.next().unwrap().unwrap();

        let node_data = store.get_node(node).unwrap();
        // Existing property should still be there
        assert_eq!(
            node_data.properties.get(&PropertyKey::new("existing")),
            Some(&Value::Int64(42))
        );
        // New property should also exist
        assert_eq!(
            node_data.properties.get(&PropertyKey::new("added")),
            Some(&Value::String("hello".into()))
        );
    }

    // ── PropertySource::PropertyAccess ──────────────────────────

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_property_source_property_access() {
        let store = create_test_store();

        let source_node = store.create_node(&["Source"]);
        store.set_node_property(source_node, "name", Value::String("Alix".into()));

        let target_node = store.create_node(&["Target"]);

        // Build chunk: col 0 = source node ID (Node type for PropertyAccess), col 1 = target node ID
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_node_id(source_node);
        builder
            .column_mut(1)
            .unwrap()
            .push_int64(target_node.0 as i64);
        builder.advance_row();

        let mut op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            MockInput::boxed(builder.finish()),
            1, // entity column = target node
            vec![(
                "copied_name".to_string(),
                PropertySource::PropertyAccess {
                    column: 0,
                    property: "name".to_string(),
                },
            )],
            vec![LogicalType::Node, LogicalType::Int64],
        );

        op.next().unwrap().unwrap();

        let target_data = store.get_node(target_node).unwrap();
        assert_eq!(
            target_data.properties.get(&PropertyKey::new("copied_name")),
            Some(&Value::String("Alix".into()))
        );
    }

    // ── ConstraintValidator integration ─────────────────────────

    #[test]
    fn test_create_node_with_constraint_validator() {
        let store = create_test_store();

        struct RejectAgeValidator;
        impl ConstraintValidator for RejectAgeValidator {
            fn validate_node_property(
                &self,
                _labels: &[String],
                key: &str,
                _value: &Value,
            ) -> Result<(), OperatorError> {
                if key == "forbidden" {
                    return Err(OperatorError::ConstraintViolation(
                        "property 'forbidden' is not allowed".to_string(),
                    ));
                }
                Ok(())
            }
            fn validate_node_complete(
                &self,
                _labels: &[String],
                _properties: &[(String, Value)],
            ) -> Result<(), OperatorError> {
                Ok(())
            }
            fn check_unique_node_property(
                &self,
                _labels: &[String],
                _key: &str,
                _value: &Value,
            ) -> Result<(), OperatorError> {
                Ok(())
            }
            fn validate_edge_property(
                &self,
                _edge_type: &str,
                _key: &str,
                _value: &Value,
            ) -> Result<(), OperatorError> {
                Ok(())
            }
            fn validate_edge_complete(
                &self,
                _edge_type: &str,
                _properties: &[(String, Value)],
            ) -> Result<(), OperatorError> {
                Ok(())
            }
        }

        // Valid property should succeed
        let mut op = CreateNodeOperator::new(
            GraphWriter::new(Arc::clone(&store)).with_validator(Arc::new(RejectAgeValidator)),
            None,
            vec!["Thing".to_string()],
            vec![(
                "name".to_string(),
                PropertySource::Constant(Value::String("ok".into())),
            )],
            vec![LogicalType::Int64],
            0,
        );

        assert!(op.next().is_ok());
        assert_eq!(store.node_count(), 1);

        // Forbidden property should fail
        let mut op = CreateNodeOperator::new(
            GraphWriter::new(Arc::clone(&store)).with_validator(Arc::new(RejectAgeValidator)),
            None,
            vec!["Thing".to_string()],
            vec![(
                "forbidden".to_string(),
                PropertySource::Constant(Value::Int64(1)),
            )],
            vec![LogicalType::Int64],
            0,
        );

        let err = op.next().unwrap_err();
        assert!(matches!(err, OperatorError::ConstraintViolation(_)));
        // Node count should still be 2 (the node is created before validation, but the error
        // propagates - this tests the validation logic fires)
    }

    // ── Reset behavior ──────────────────────────────────────────

    #[test]
    fn test_create_node_reset_allows_re_execution() {
        let store = create_test_store();

        let mut op = CreateNodeOperator::new(
            Arc::clone(&store),
            None,
            vec!["Person".to_string()],
            vec![],
            vec![LogicalType::Int64],
            0,
        );

        // First execution
        assert!(op.next().unwrap().is_some());
        assert!(op.next().unwrap().is_none());

        // Reset and re-execute
        op.reset();
        assert!(op.next().unwrap().is_some());

        assert_eq!(store.node_count(), 2);
    }

    // ── Operator name() ──────────────────────────────────────────

    #[test]
    fn test_operator_names() {
        let store = create_test_store();

        let op = CreateNodeOperator::new(
            Arc::clone(&store),
            None,
            vec![],
            vec![],
            vec![LogicalType::Int64],
            0,
        );
        assert_eq!(op.name(), "CreateNode");

        let op = CreateEdgeOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            1,
            "R".to_string(),
            vec![LogicalType::Int64],
        );
        assert_eq!(op.name(), "CreateEdge");

        let op = DeleteNodeOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec![LogicalType::Int64],
            false,
        );
        assert_eq!(op.name(), "DeleteNode");

        let op = DeleteEdgeOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec![LogicalType::Int64],
        );
        assert_eq!(op.name(), "DeleteEdge");

        let op = AddLabelOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec!["L".to_string()],
            vec![LogicalType::Int64],
        );
        assert_eq!(op.name(), "AddLabel");

        let op = RemoveLabelOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec!["L".to_string()],
            vec![LogicalType::Int64],
        );
        assert_eq!(op.name(), "RemoveLabel");

        let op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec![],
            vec![LogicalType::Int64],
        );
        assert_eq!(op.name(), "SetProperty");
    }

    // ── into_any() coverage ─────────────────────────────────────

    #[test]
    fn test_create_node_into_any() {
        let store = create_test_store();
        let op = CreateNodeOperator::new(
            Arc::clone(&store),
            None,
            vec!["Person".to_string()],
            vec![],
            vec![LogicalType::Int64],
            0,
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<CreateNodeOperator>().is_ok());
    }

    #[test]
    fn test_create_edge_into_any() {
        let store = create_test_store();
        let op = CreateEdgeOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            1,
            "KNOWS".to_string(),
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<CreateEdgeOperator>().is_ok());
    }

    #[test]
    fn test_delete_node_into_any() {
        let store = create_test_store();
        let op = DeleteNodeOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec![LogicalType::Int64],
            false,
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<DeleteNodeOperator>().is_ok());
    }

    #[test]
    fn test_delete_edge_into_any() {
        let store = create_test_store();
        let op = DeleteEdgeOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<DeleteEdgeOperator>().is_ok());
    }

    #[test]
    fn test_add_label_into_any() {
        let store = create_test_store();
        let op = AddLabelOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec!["Label".to_string()],
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<AddLabelOperator>().is_ok());
    }

    #[test]
    fn test_remove_label_into_any() {
        let store = create_test_store();
        let op = RemoveLabelOperator::new(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec!["Label".to_string()],
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<RemoveLabelOperator>().is_ok());
    }

    #[test]
    fn test_set_property_into_any() {
        let store = create_test_store();
        let op = SetPropertyOperator::new_for_node(
            Arc::clone(&store),
            Box::new(EmptyInput),
            0,
            vec![],
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<SetPropertyOperator>().is_ok());
    }

    // ── ConstraintValidator default methods ──────────────────────

    /// A minimal validator that implements only the required methods,
    /// relying on defaults for the optional ones.
    struct MinimalValidator;

    impl ConstraintValidator for MinimalValidator {
        fn validate_node_property(
            &self,
            _labels: &[String],
            _key: &str,
            _value: &Value,
        ) -> Result<(), OperatorError> {
            Ok(())
        }
        fn validate_node_complete(
            &self,
            _labels: &[String],
            _properties: &[(String, Value)],
        ) -> Result<(), OperatorError> {
            Ok(())
        }
        fn check_unique_node_property(
            &self,
            _labels: &[String],
            _key: &str,
            _value: &Value,
        ) -> Result<(), OperatorError> {
            Ok(())
        }
        fn validate_edge_property(
            &self,
            _edge_type: &str,
            _key: &str,
            _value: &Value,
        ) -> Result<(), OperatorError> {
            Ok(())
        }
        fn validate_edge_complete(
            &self,
            _edge_type: &str,
            _properties: &[(String, Value)],
        ) -> Result<(), OperatorError> {
            Ok(())
        }
    }

    #[test]
    fn test_constraint_validator_default_node_labels_allowed() {
        let v = MinimalValidator;
        assert!(
            v.validate_node_labels_allowed(&["Person".to_string(), "Actor".to_string()])
                .is_ok()
        );
    }

    #[test]
    fn test_constraint_validator_default_edge_type_allowed() {
        let v = MinimalValidator;
        assert!(v.validate_edge_type_allowed("KNOWS").is_ok());
    }

    #[test]
    fn test_constraint_validator_default_edge_endpoints() {
        let v = MinimalValidator;
        assert!(
            v.validate_edge_endpoints("KNOWS", &["Person".to_string()], &["Person".to_string()],)
                .is_ok()
        );
    }

    #[test]
    fn test_constraint_validator_default_inject_defaults() {
        let v = MinimalValidator;
        let mut props = vec![("name".to_string(), Value::String("Alix".into()))];
        v.inject_defaults(&["Person".to_string()], &mut props);
        // Default impl is a no-op
        assert_eq!(props.len(), 1);
    }

    // ── PropertySource tests ────────────────────────────────────

    #[test]
    fn test_property_source_column() {
        let store = LpgStore::new().unwrap();
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();

        let src = PropertySource::Column(0);
        assert_eq!(src.resolve(&chunk, 0, &store), Value::Int64(42));
    }

    #[test]
    fn test_property_source_constant() {
        let store = LpgStore::new().unwrap();
        let chunk = DataChunk::empty();

        let src = PropertySource::Constant(Value::String("hello".into()));
        assert_eq!(
            src.resolve(&chunk, 0, &store),
            Value::String("hello".into()),
        );
    }

    #[test]
    fn test_property_source_column_out_of_bounds() {
        let store = LpgStore::new().unwrap();
        let chunk = DataChunk::empty();

        let src = PropertySource::Column(99);
        assert_eq!(src.resolve(&chunk, 0, &store), Value::Null);
    }

    #[test]
    fn test_property_source_property_access_from_map() {
        let store = LpgStore::new().unwrap();
        let mut map = std::collections::BTreeMap::new();
        map.insert(PropertyKey::new("age"), Value::Int64(30));

        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::Map(Arc::new(map)));
        builder.advance_row();
        let chunk = builder.finish();

        let src = PropertySource::PropertyAccess {
            column: 0,
            property: "age".to_string(),
        };
        assert_eq!(src.resolve(&chunk, 0, &store), Value::Int64(30));
    }

    #[test]
    fn test_property_source_property_access_missing_column() {
        let store = LpgStore::new().unwrap();
        let chunk = DataChunk::empty();

        let src = PropertySource::PropertyAccess {
            column: 99,
            property: "name".to_string(),
        };
        assert_eq!(src.resolve(&chunk, 0, &store), Value::Null);
    }
}
