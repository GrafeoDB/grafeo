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
use super::{Operator, OperatorError, OperatorResult, SessionContext, SharedWriteTracker};
use crate::execution::chunk::{DataChunk, DataChunkBuilder};
use crate::graph::{GraphStore, GraphStoreMut, GraphStoreSearch};

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
    /// The graph store to modify.
    store: Arc<dyn GraphStoreMut>,
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
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for MVCC versioning.
    transaction_id: Option<TransactionId>,
    /// Optional constraint validator for schema enforcement.
    validator: Option<Arc<dyn ConstraintValidator>>,
    /// Optional write tracker for conflict detection.
    write_tracker: Option<SharedWriteTracker>,
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
    /// * `store` - The graph store to modify.
    /// * `input` - Optional input operator (None for standalone CREATE).
    /// * `labels` - Labels to assign to created nodes.
    /// * `properties` - Properties to set on created nodes.
    /// * `output_schema` - Schema of the output.
    /// * `output_column` - Column index where the created node ID goes.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Option<Box<dyn Operator>>,
        labels: Vec<String>,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
        output_column: usize,
    ) -> Self {
        Self {
            store,
            input,
            labels,
            properties,
            output_schema,
            output_column,
            executed: false,
            viewing_epoch: None,
            transaction_id: None,
            validator: None,
            write_tracker: None,
            expressions: PropertyExpressions::default(),
        }
    }

    /// Sets the transaction context for MVCC versioning.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the constraint validator for schema enforcement.
    pub fn with_validator(mut self, validator: Arc<dyn ConstraintValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Sets the write tracker for conflict detection.
    pub fn with_write_tracker(mut self, tracker: SharedWriteTracker) -> Self {
        self.write_tracker = Some(tracker);
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

impl CreateNodeOperator {
    /// Validates and sets properties on a newly created node.
    fn validate_and_set_properties(
        &self,
        node_id: NodeId,
        resolved_props: &mut Vec<(String, Value)>,
    ) -> Result<(), OperatorError> {
        // Phase 0: Validate that node labels are allowed by the bound graph type
        if let Some(ref validator) = self.validator {
            validator.validate_node_labels_allowed(&self.labels)?;
        }

        // Phase 0.5: Inject defaults for properties not explicitly provided
        if let Some(ref validator) = self.validator {
            validator.inject_defaults(&self.labels, resolved_props);
        }

        // Phase 1: Validate each property value
        if let Some(ref validator) = self.validator {
            for (name, value) in resolved_props.iter() {
                validator.validate_node_property(&self.labels, name, value)?;
                validator.check_unique_node_property(&self.labels, name, value)?;
            }
            // Phase 2: Validate completeness (NOT NULL checks for missing required properties)
            validator.validate_node_complete(&self.labels, resolved_props)?;
        }

        // Phase 3: Write properties to the store
        if let Some(tid) = self.transaction_id {
            for (name, value) in resolved_props.iter() {
                self.store
                    .set_node_property_versioned(node_id, name, value.clone(), tid);
            }
        } else {
            for (name, value) in resolved_props.iter() {
                self.store.set_node_property(node_id, name, value.clone());
            }
        }
        Ok(())
    }
}

impl Operator for CreateNodeOperator {
    fn next(&mut self) -> OperatorResult {
        // Get transaction context for versioned creation
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        let tx = self.transaction_id.unwrap_or(TransactionId::SYSTEM);

        if let Some(ref mut input) = self.input {
            // For each input row, create a node
            if let Some(chunk) = input.next()? {
                let mut builder =
                    DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

                for row in chunk.selected_indices() {
                    // Resolve all property values first (before creating node)
                    let mut resolved_props = self.expressions.resolve_row(
                        &self.properties,
                        &chunk,
                        row,
                        self.store.as_ref() as &dyn GraphStore,
                        self.viewing_epoch,
                        self.transaction_id,
                    )?;

                    // Create the node with MVCC versioning
                    let label_refs: Vec<&str> = self.labels.iter().map(String::as_str).collect();
                    let node_id = self.store.create_node_versioned(&label_refs, epoch, tx);

                    // Record write for conflict detection
                    if let (Some(tracker), Some(tid)) = (&self.write_tracker, self.transaction_id) {
                        tracker.record_node_write(tid, node_id)?;
                    }

                    // Validate and set properties
                    self.validate_and_set_properties(node_id, &mut resolved_props)?;

                    // Copy input columns to output
                    for col_idx in 0..chunk.column_count() {
                        if col_idx < self.output_column
                            && let (Some(src), Some(dst)) =
                                (chunk.column(col_idx), builder.column_mut(col_idx))
                        {
                            if let Some(val) = src.get_value(row) {
                                dst.push_value(val);
                            } else {
                                dst.push_value(Value::Null);
                            }
                        }
                    }

                    // Add the new node ID
                    if let Some(dst) = builder.column_mut(self.output_column) {
                        // reason: entity IDs stored as i64, standard encoding
                        #[allow(clippy::cast_possible_wrap)]
                        // reason: entity IDs stored as i64, standard encoding
                        #[allow(clippy::cast_possible_wrap)]
                        dst.push_value(Value::Int64(node_id.0 as i64));
                    }

                    builder.advance_row();
                }

                return Ok(Some(builder.finish()));
            }
            Ok(None)
        } else {
            // No input - create a single node
            if self.executed {
                return Ok(None);
            }
            self.executed = true;

            // Resolve constant properties. Computed values need a row: the
            // planner gives such a CREATE a single-row input instead.
            let mut resolved_props: Vec<(String, Value)> = self
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

            // Create the node with MVCC versioning
            let label_refs: Vec<&str> = self.labels.iter().map(String::as_str).collect();
            let node_id = self.store.create_node_versioned(&label_refs, epoch, tx);

            // Record write for conflict detection
            if let (Some(tracker), Some(tid)) = (&self.write_tracker, self.transaction_id) {
                tracker.record_node_write(tid, node_id)?;
            }

            // Validate and set properties
            self.validate_and_set_properties(node_id, &mut resolved_props)?;

            // Build output chunk with just the node ID
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);
            if let Some(dst) = builder.column_mut(self.output_column) {
                // reason: entity IDs stored as i64, standard encoding
                #[allow(clippy::cast_possible_wrap)]
                dst.push_value(Value::Int64(node_id.0 as i64));
            }
            builder.advance_row();

            Ok(Some(builder.finish()))
        }
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

/// Operator that creates new edges.
pub struct CreateEdgeOperator {
    /// The graph store to modify.
    store: Arc<dyn GraphStoreMut>,
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
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for MVCC versioning.
    transaction_id: Option<TransactionId>,
    /// Optional constraint validator for schema enforcement.
    validator: Option<Arc<dyn ConstraintValidator>>,
    /// Optional write tracker for conflict detection.
    write_tracker: Option<SharedWriteTracker>,
    /// Evaluates computed property values (`PropertySource::Expression`).
    expressions: PropertyExpressions,
}

impl CreateEdgeOperator {
    /// Creates a new edge creation operator.
    ///
    /// Use builder methods to set additional options:
    /// - [`with_properties`](Self::with_properties) - set edge properties
    /// - [`with_output_column`](Self::with_output_column) - output the created edge ID
    /// - [`with_transaction_context`](Self::with_transaction_context) - set transaction context
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        from_column: usize,
        to_column: usize,
        edge_type: String,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            from_column,
            to_column,
            edge_type,
            properties: Vec::new(),
            output_schema,
            output_column: None,
            viewing_epoch: None,
            transaction_id: None,
            validator: None,
            write_tracker: None,
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

    /// Sets the transaction context for MVCC versioning.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the constraint validator for schema enforcement.
    pub fn with_validator(mut self, validator: Arc<dyn ConstraintValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Sets the write tracker for conflict detection.
    pub fn with_write_tracker(mut self, tracker: SharedWriteTracker) -> Self {
        self.write_tracker = Some(tracker);
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
        // Get transaction context for versioned creation
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        let tx = self.transaction_id.unwrap_or(TransactionId::SYSTEM);

        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                // Get source and target node IDs
                let from_id = chunk
                    .column(self.from_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("from column {}", self.from_column))
                    })?;

                let to_id = chunk
                    .column(self.to_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("to column {}", self.to_column))
                    })?;

                // Extract node IDs
                let from_node_id = match from_id {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{from_id:?}"),
                        });
                    }
                };

                let to_node_id = match to_id {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{to_id:?}"),
                        });
                    }
                };

                // Validate graph type and edge endpoint constraints
                if let Some(ref validator) = self.validator {
                    validator.validate_edge_type_allowed(&self.edge_type)?;

                    // Look up source and target node labels for endpoint validation
                    let source_labels: Vec<String> = self
                        .store
                        .get_node(from_node_id)
                        .map(|n| n.labels.iter().map(|l| l.to_string()).collect())
                        .unwrap_or_default();
                    let target_labels: Vec<String> = self
                        .store
                        .get_node(to_node_id)
                        .map(|n| n.labels.iter().map(|l| l.to_string()).collect())
                        .unwrap_or_default();
                    validator.validate_edge_endpoints(
                        &self.edge_type,
                        &source_labels,
                        &target_labels,
                    )?;
                }

                // Resolve property values
                let resolved_props = self.expressions.resolve_row(
                    &self.properties,
                    &chunk,
                    row,
                    self.store.as_ref() as &dyn GraphStore,
                    self.viewing_epoch,
                    self.transaction_id,
                )?;

                // Validate constraints before writing
                if let Some(ref validator) = self.validator {
                    for (name, value) in &resolved_props {
                        validator.validate_edge_property(&self.edge_type, name, value)?;
                    }
                    validator.validate_edge_complete(&self.edge_type, &resolved_props)?;
                }

                // Create the edge with MVCC versioning
                let edge_id = self.store.create_edge_versioned(
                    from_node_id,
                    to_node_id,
                    &self.edge_type,
                    epoch,
                    tx,
                );

                // Record write for conflict detection
                if let (Some(tracker), Some(tid)) = (&self.write_tracker, self.transaction_id) {
                    tracker.record_edge_write(tid, edge_id)?;
                }

                // Set properties
                if let Some(tid) = self.transaction_id {
                    for (name, value) in resolved_props {
                        self.store
                            .set_edge_property_versioned(edge_id, &name, value, tid);
                    }
                } else {
                    for (name, value) in resolved_props {
                        self.store.set_edge_property(edge_id, &name, value);
                    }
                }

                // Copy input columns
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src), Some(dst)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(val) = src.get_value(row) {
                            dst.push_value(val);
                        } else {
                            dst.push_value(Value::Null);
                        }
                    }
                }

                // Add edge ID if requested
                if let Some(out_col) = self.output_column
                    && let Some(dst) = builder.column_mut(out_col)
                {
                    // reason: entity IDs stored as i64, standard encoding
                    #[allow(clippy::cast_possible_wrap)]
                    dst.push_value(Value::Int64(edge_id.0 as i64));
                }

                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }
        Ok(None)
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
    /// The graph store to modify.
    store: Arc<dyn GraphStoreMut>,
    /// Input operator.
    input: Box<dyn Operator>,
    /// Column index for the node to delete.
    node_column: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Whether to detach (delete connected edges) before deleting.
    detach: bool,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for MVCC versioning.
    transaction_id: Option<TransactionId>,
    /// Optional write tracker for conflict detection.
    write_tracker: Option<SharedWriteTracker>,
}

impl DeleteNodeOperator {
    /// Creates a new node deletion operator.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        node_column: usize,
        output_schema: Vec<LogicalType>,
        detach: bool,
    ) -> Self {
        Self {
            store,
            input,
            node_column,
            output_schema,
            detach,
            viewing_epoch: None,
            transaction_id: None,
            write_tracker: None,
        }
    }

    /// Sets the transaction context for MVCC versioning.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the write tracker for conflict detection.
    pub fn with_write_tracker(mut self, tracker: SharedWriteTracker) -> Self {
        self.write_tracker = Some(tracker);
        self
    }
}

impl Operator for DeleteNodeOperator {
    fn next(&mut self) -> OperatorResult {
        // Get transaction context for versioned deletion
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        let tx = self.transaction_id.unwrap_or(TransactionId::SYSTEM);

        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let node_val = chunk
                    .column(self.node_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("node column {}", self.node_column))
                    })?;

                let node_id = match node_val {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{node_val:?}"),
                        });
                    }
                };

                if self.detach {
                    // Delete all connected edges first, using versioned deletion
                    // so rollback can restore them
                    let outgoing = self
                        .store
                        .edges_from(node_id, crate::graph::Direction::Outgoing);
                    let incoming = self
                        .store
                        .edges_from(node_id, crate::graph::Direction::Incoming);
                    for (_, edge_id) in outgoing.into_iter().chain(incoming) {
                        self.store.delete_edge_versioned(edge_id, epoch, tx);
                        if let (Some(tracker), Some(tid)) =
                            (&self.write_tracker, self.transaction_id)
                        {
                            tracker.record_edge_write(tid, edge_id)?;
                        }
                    }
                } else {
                    // NODETACH: check that node has no connected edges
                    let degree = self.store.out_degree(node_id) + self.store.in_degree(node_id);
                    if degree > 0 {
                        return Err(OperatorError::ConstraintViolation(format!(
                            "Cannot delete node with {} connected edge(s). Use DETACH DELETE.",
                            degree
                        )));
                    }
                }

                // Delete the node with MVCC versioning
                self.store.delete_node_versioned(node_id, epoch, tx);

                // Record write for conflict detection
                if let (Some(tracker), Some(tid)) = (&self.write_tracker, self.transaction_id) {
                    tracker.record_node_write(tid, node_id)?;
                }

                // Pass through all input columns so downstream RETURN can
                // reference the variable (e.g., count(n) after DELETE n).
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src), Some(dst)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(val) = src.get_value(row) {
                            dst.push_value(val);
                        } else {
                            dst.push_value(Value::Null);
                        }
                    }
                }
                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }
        Ok(None)
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
    /// The graph store to modify.
    store: Arc<dyn GraphStoreMut>,
    /// Input operator.
    input: Box<dyn Operator>,
    /// Column index for the edge to delete.
    edge_column: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for MVCC versioning.
    transaction_id: Option<TransactionId>,
    /// Optional write tracker for conflict detection.
    write_tracker: Option<SharedWriteTracker>,
}

impl DeleteEdgeOperator {
    /// Creates a new edge deletion operator.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        edge_column: usize,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            edge_column,
            output_schema,
            viewing_epoch: None,
            transaction_id: None,
            write_tracker: None,
        }
    }

    /// Sets the transaction context for MVCC versioning.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the write tracker for conflict detection.
    pub fn with_write_tracker(mut self, tracker: SharedWriteTracker) -> Self {
        self.write_tracker = Some(tracker);
        self
    }
}

impl Operator for DeleteEdgeOperator {
    fn next(&mut self) -> OperatorResult {
        // Get transaction context for versioned deletion
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        let tx = self.transaction_id.unwrap_or(TransactionId::SYSTEM);

        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let edge_val = chunk
                    .column(self.edge_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("edge column {}", self.edge_column))
                    })?;

                let edge_id = match edge_val {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(id) => EdgeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (edge ID)".to_string(),
                            found: format!("{edge_val:?}"),
                        });
                    }
                };

                // Delete the edge with MVCC versioning
                self.store.delete_edge_versioned(edge_id, epoch, tx);

                // Record write for conflict detection
                if let (Some(tracker), Some(tid)) = (&self.write_tracker, self.transaction_id) {
                    tracker.record_edge_write(tid, edge_id)?;
                }

                // Pass through all input columns
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src), Some(dst)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(val) = src.get_value(row) {
                            dst.push_value(val);
                        } else {
                            dst.push_value(Value::Null);
                        }
                    }
                }
                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }
        Ok(None)
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
    /// The graph store.
    store: Arc<dyn GraphStoreMut>,
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
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for undo log tracking.
    transaction_id: Option<TransactionId>,
    /// Optional write tracker for conflict detection.
    write_tracker: Option<SharedWriteTracker>,
}

impl AddLabelOperator {
    /// Creates a new add label operator.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        node_column: usize,
        labels: Vec<String>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        let count_column = output_schema.len() - 1;
        Self {
            store,
            input,
            node_column,
            labels,
            count_column,
            output_schema,
            viewing_epoch: None,
            transaction_id: None,
            write_tracker: None,
        }
    }

    /// Sets the transaction context for versioned label mutations.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the write tracker for conflict detection.
    pub fn with_write_tracker(mut self, tracker: SharedWriteTracker) -> Self {
        self.write_tracker = Some(tracker);
        self
    }
}

impl Operator for AddLabelOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let node_val = chunk
                    .column(self.node_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("node column {}", self.node_column))
                    })?;

                let node_id = match node_val {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{node_val:?}"),
                        });
                    }
                };

                // Record write for conflict detection
                if let (Some(tracker), Some(tid)) = (&self.write_tracker, self.transaction_id) {
                    tracker.record_node_write(tid, node_id)?;
                }

                // Add all labels
                let mut row_count: i64 = 0;
                for label in &self.labels {
                    let added = if let Some(tid) = self.transaction_id {
                        self.store.add_label_versioned(node_id, label, tid)
                    } else {
                        self.store.add_label(node_id, label)
                    };
                    if added {
                        row_count += 1;
                    }
                }

                // Copy input columns to output (pass-through)
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src), Some(dst)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(val) = src.get_value(row) {
                            dst.push_value(val);
                        } else {
                            dst.push_value(Value::Null);
                        }
                    }
                }
                // Append the update count column
                if let Some(dst) = builder.column_mut(self.count_column) {
                    dst.push_value(Value::Int64(row_count));
                }

                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }
        Ok(None)
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
    /// The graph store.
    store: Arc<dyn GraphStoreMut>,
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
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for undo log tracking.
    transaction_id: Option<TransactionId>,
    /// Optional write tracker for conflict detection.
    write_tracker: Option<SharedWriteTracker>,
}

impl RemoveLabelOperator {
    /// Creates a new remove label operator.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        node_column: usize,
        labels: Vec<String>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        let count_column = output_schema.len() - 1;
        Self {
            store,
            input,
            node_column,
            labels,
            count_column,
            output_schema,
            viewing_epoch: None,
            transaction_id: None,
            write_tracker: None,
        }
    }

    /// Sets the transaction context for versioned label mutations.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the write tracker for conflict detection.
    pub fn with_write_tracker(mut self, tracker: SharedWriteTracker) -> Self {
        self.write_tracker = Some(tracker);
        self
    }
}

impl Operator for RemoveLabelOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let node_val = chunk
                    .column(self.node_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("node column {}", self.node_column))
                    })?;

                let node_id = match node_val {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{node_val:?}"),
                        });
                    }
                };

                // Record write for conflict detection
                if let (Some(tracker), Some(tid)) = (&self.write_tracker, self.transaction_id) {
                    tracker.record_node_write(tid, node_id)?;
                }

                // Remove all labels
                let mut row_count: i64 = 0;
                for label in &self.labels {
                    let removed = if let Some(tid) = self.transaction_id {
                        self.store.remove_label_versioned(node_id, label, tid)
                    } else {
                        self.store.remove_label(node_id, label)
                    };
                    if removed {
                        row_count += 1;
                    }
                }

                // Copy input columns to output (pass-through)
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src), Some(dst)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(val) = src.get_value(row) {
                            dst.push_value(val);
                        } else {
                            dst.push_value(Value::Null);
                        }
                    }
                }
                // Append the update count column
                if let Some(dst) = builder.column_mut(self.count_column) {
                    dst.push_value(Value::Int64(row_count));
                }

                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }
        Ok(None)
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
    /// The graph store.
    store: Arc<dyn GraphStoreMut>,
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
    /// Optional constraint validator for schema enforcement.
    validator: Option<Arc<dyn ConstraintValidator>>,
    /// Entity labels (for node constraint validation).
    labels: Vec<String>,
    /// Edge type (for edge constraint validation).
    edge_type_name: Option<String>,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for undo log tracking.
    transaction_id: Option<TransactionId>,
    /// Optional write tracker for conflict detection.
    write_tracker: Option<SharedWriteTracker>,
}

impl SetPropertyOperator {
    /// Creates a new set property operator for nodes.
    pub fn new_for_node(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        node_column: usize,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            entity_column: node_column,
            is_edge: false,
            properties,
            output_schema,
            replace: false,
            validator: None,
            labels: Vec::new(),
            edge_type_name: None,
            viewing_epoch: None,
            transaction_id: None,
            write_tracker: None,
        }
    }

    /// Creates a new set property operator for edges.
    pub fn new_for_edge(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        edge_column: usize,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            entity_column: edge_column,
            is_edge: true,
            properties,
            output_schema,
            replace: false,
            validator: None,
            labels: Vec::new(),
            edge_type_name: None,
            viewing_epoch: None,
            transaction_id: None,
            write_tracker: None,
        }
    }

    /// Sets whether this operator replaces all properties (for map assignment).
    pub fn with_replace(mut self, replace: bool) -> Self {
        self.replace = replace;
        self
    }

    /// Sets the constraint validator for schema enforcement.
    pub fn with_validator(mut self, validator: Arc<dyn ConstraintValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Sets the entity labels (for node constraint validation).
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Sets the edge type name (for edge constraint validation).
    pub fn with_edge_type(mut self, edge_type: String) -> Self {
        self.edge_type_name = Some(edge_type);
        self
    }

    /// Sets the transaction context for versioned property mutations.
    ///
    /// When a transaction ID is provided, property changes are recorded in
    /// an undo log so they can be restored on rollback.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the write tracker for conflict detection.
    pub fn with_write_tracker(mut self, tracker: SharedWriteTracker) -> Self {
        self.write_tracker = Some(tracker);
        self
    }
}

impl Operator for SetPropertyOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let entity_val = chunk
                    .column(self.entity_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!(
                            "entity column {}",
                            self.entity_column
                        ))
                    })?;

                let entity_id = match entity_val {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(id) => id as u64,
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (entity ID)".to_string(),
                            found: format!("{entity_val:?}"),
                        });
                    }
                };

                // Record write for conflict detection
                if let (Some(tracker), Some(tid)) = (&self.write_tracker, self.transaction_id) {
                    if self.is_edge {
                        tracker.record_edge_write(tid, EdgeId(entity_id))?;
                    } else {
                        tracker.record_node_write(tid, NodeId(entity_id))?;
                    }
                }

                // Resolve all property values
                let resolved_props: Vec<(String, Value)> = self
                    .properties
                    .iter()
                    .map(|(name, source)| {
                        let value =
                            source.resolve(&chunk, row, self.store.as_ref() as &dyn GraphStore);
                        (name.clone(), value)
                    })
                    .collect();

                // Validate constraints before writing
                if let Some(ref validator) = self.validator {
                    if self.is_edge {
                        if let Some(ref et) = self.edge_type_name {
                            for (name, value) in &resolved_props {
                                validator.validate_edge_property(et, name, value)?;
                            }
                        }
                    } else {
                        for (name, value) in &resolved_props {
                            validator.validate_node_property(&self.labels, name, value)?;
                            validator.check_unique_node_property(&self.labels, name, value)?;
                        }
                    }
                }

                // Write all properties (use versioned methods when inside a transaction)
                let tx_id = self.transaction_id;
                for (prop_name, value) in resolved_props {
                    if prop_name == "*" {
                        // Map assignment: value should be a Map
                        if let Value::Map(map) = value {
                            if self.replace {
                                // Replace: remove all existing properties first
                                if self.is_edge {
                                    if let Some(edge) = self.store.get_edge(EdgeId(entity_id)) {
                                        let keys: Vec<String> = edge
                                            .properties
                                            .iter()
                                            .map(|(k, _)| k.as_str().to_string())
                                            .collect();
                                        for key in keys {
                                            if let Some(tid) = tx_id {
                                                self.store.remove_edge_property_versioned(
                                                    EdgeId(entity_id),
                                                    &key,
                                                    tid,
                                                );
                                            } else {
                                                self.store
                                                    .remove_edge_property(EdgeId(entity_id), &key);
                                            }
                                        }
                                    }
                                } else if let Some(node) = self.store.get_node(NodeId(entity_id)) {
                                    let keys: Vec<String> = node
                                        .properties
                                        .iter()
                                        .map(|(k, _)| k.as_str().to_string())
                                        .collect();
                                    for key in keys {
                                        if let Some(tid) = tx_id {
                                            self.store.remove_node_property_versioned(
                                                NodeId(entity_id),
                                                &key,
                                                tid,
                                            );
                                        } else {
                                            self.store
                                                .remove_node_property(NodeId(entity_id), &key);
                                        }
                                    }
                                }
                            }
                            // Set each map entry (null values remove the property)
                            for (key, val) in map.iter() {
                                if val.is_null() {
                                    // Null in SET += removes the property (Cypher/GQL semantics)
                                    if self.is_edge {
                                        if let Some(tid) = tx_id {
                                            self.store.remove_edge_property_versioned(
                                                EdgeId(entity_id),
                                                key.as_str(),
                                                tid,
                                            );
                                        } else {
                                            self.store.remove_edge_property(
                                                EdgeId(entity_id),
                                                key.as_str(),
                                            );
                                        }
                                    } else if let Some(tid) = tx_id {
                                        self.store.remove_node_property_versioned(
                                            NodeId(entity_id),
                                            key.as_str(),
                                            tid,
                                        );
                                    } else {
                                        self.store
                                            .remove_node_property(NodeId(entity_id), key.as_str());
                                    }
                                } else if self.is_edge {
                                    if let Some(tid) = tx_id {
                                        self.store.set_edge_property_versioned(
                                            EdgeId(entity_id),
                                            key.as_str(),
                                            val.clone(),
                                            tid,
                                        );
                                    } else {
                                        self.store.set_edge_property(
                                            EdgeId(entity_id),
                                            key.as_str(),
                                            val.clone(),
                                        );
                                    }
                                } else if let Some(tid) = tx_id {
                                    self.store.set_node_property_versioned(
                                        NodeId(entity_id),
                                        key.as_str(),
                                        val.clone(),
                                        tid,
                                    );
                                } else {
                                    self.store.set_node_property(
                                        NodeId(entity_id),
                                        key.as_str(),
                                        val.clone(),
                                    );
                                }
                            }
                        }
                    } else if self.is_edge {
                        if let Some(tid) = tx_id {
                            self.store.set_edge_property_versioned(
                                EdgeId(entity_id),
                                &prop_name,
                                value,
                                tid,
                            );
                        } else {
                            self.store
                                .set_edge_property(EdgeId(entity_id), &prop_name, value);
                        }
                    } else if let Some(tid) = tx_id {
                        self.store.set_node_property_versioned(
                            NodeId(entity_id),
                            &prop_name,
                            value,
                            tid,
                        );
                    } else {
                        self.store
                            .set_node_property(NodeId(entity_id), &prop_name, value);
                    }
                }

                // Copy input columns to output
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src), Some(dst)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(val) = src.get_value(row) {
                            dst.push_value(val);
                        } else {
                            dst.push_value(Value::Null);
                        }
                    }
                }

                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }
        Ok(None)
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
            Arc::clone(&store),
            None,
            vec!["Thing".to_string()],
            vec![(
                "name".to_string(),
                PropertySource::Constant(Value::String("ok".into())),
            )],
            vec![LogicalType::Int64],
            0,
        )
        .with_validator(Arc::new(RejectAgeValidator));

        assert!(op.next().is_ok());
        assert_eq!(store.node_count(), 1);

        // Forbidden property should fail
        let mut op = CreateNodeOperator::new(
            Arc::clone(&store),
            None,
            vec!["Thing".to_string()],
            vec![(
                "forbidden".to_string(),
                PropertySource::Constant(Value::Int64(1)),
            )],
            vec![LogicalType::Int64],
            0,
        )
        .with_validator(Arc::new(RejectAgeValidator));

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
