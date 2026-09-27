//! Merge operator for MERGE clause execution.
//!
//! The MERGE operator implements the Cypher MERGE semantics:
//! 1. Try to match the pattern in the graph
//! 2. If found, return existing element (optionally apply ON MATCH SET)
//! 3. If not found, create the element (optionally apply ON CREATE SET)

use super::{
    ConstraintValidator, ExpressionPredicate, Operator, OperatorError, OperatorResult,
    PropertySource, SessionContext,
};
use crate::execution::chunk::{DataChunk, DataChunkBuilder};
use crate::graph::{GraphStore, GraphStoreMut, GraphStoreSearch};
use grafeo_common::types::{
    EdgeId, EpochId, LogicalType, NodeId, PropertyKey, TransactionId, Value,
};
use std::sync::Arc;

/// Configuration for a node merge operation.
pub struct MergeConfig {
    /// Variable name for the merged node.
    pub variable: String,
    /// Labels to match/create.
    pub labels: Vec<String>,
    /// Properties that must match (also used for creation).
    pub match_properties: Vec<(String, PropertySource)>,
    /// Properties to set on CREATE.
    pub on_create_properties: Vec<(String, PropertySource)>,
    /// Properties to set on MATCH.
    pub on_match_properties: Vec<(String, PropertySource)>,
    /// Output schema (input columns + node column).
    pub output_schema: Vec<LogicalType>,
    /// Column index where the merged node ID is placed.
    pub output_column: usize,
    /// If the merge variable was already bound in the input, this column index
    /// is used to detect NULL references (e.g., from unmatched OPTIONAL MATCH).
    /// `None` for standalone MERGE that introduces a new variable.
    pub bound_variable_column: Option<usize>,
}

/// Merge operator for MERGE clause.
///
/// Tries to match a node with the given labels and properties.
/// If found, returns the existing node. If not found, creates a new node.
///
/// When an input operator is provided (chained MERGE), input rows are
/// passed through with the merged node ID appended as an additional column.
pub struct MergeOperator {
    /// The graph store.
    store: Arc<dyn GraphStoreMut>,
    /// Optional input operator (for chained MERGE patterns).
    input: Option<Box<dyn Operator>>,
    /// Merge configuration.
    config: MergeConfig,
    /// Whether we've already executed (standalone mode only).
    executed: bool,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for undo log tracking.
    transaction_id: Option<TransactionId>,
    /// Optional constraint validator for schema enforcement.
    validator: Option<Arc<dyn ConstraintValidator>>,
    /// Search-store handle used to evaluate `PropertySource::Expression`
    /// runtime expressions in `ON CREATE` / `ON MATCH SET`. None when no
    /// expression sources are present (the planner skips threading it).
    search_store: Option<Arc<dyn GraphStoreSearch>>,
    /// Session context for expression evaluation (info, schema, etc.).
    session_context: SessionContext,
}

impl MergeOperator {
    /// Creates a new merge operator.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Option<Box<dyn Operator>>,
        config: MergeConfig,
    ) -> Self {
        Self {
            store,
            input,
            config,
            executed: false,
            viewing_epoch: None,
            transaction_id: None,
            validator: None,
            search_store: None,
            session_context: SessionContext::default(),
        }
    }

    /// Returns the variable name for the merged node.
    #[must_use]
    pub fn variable(&self) -> &str {
        &self.config.variable
    }

    /// Sets the transaction context for versioned mutations.
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

    /// Provides a search-store handle so `PropertySource::Expression`
    /// sources in `ON CREATE` / `ON MATCH SET` can be evaluated.
    #[must_use]
    pub fn with_search_store(mut self, search_store: Arc<dyn GraphStoreSearch>) -> Self {
        self.search_store = Some(search_store);
        self
    }

    /// Sets the session context used during expression evaluation.
    #[must_use]
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.session_context = context;
        self
    }

    /// Resolves property sources to concrete values for a given row.
    ///
    /// Skips [`PropertySource::Expression`] sources: those need an augmented
    /// row containing the merged node/edge and are evaluated separately by
    /// [`Self::resolve_action_properties`].
    fn resolve_properties(
        props: &[(String, PropertySource)],
        chunk: Option<&DataChunk>,
        row: usize,
        store: &dyn GraphStore,
    ) -> Vec<(String, Value)> {
        props
            .iter()
            .map(|(name, source)| {
                let value = if let Some(chunk) = chunk {
                    source.resolve(chunk, row, store)
                } else {
                    // Standalone mode: only constants are valid
                    match source {
                        PropertySource::Constant(v) => v.clone(),
                        _ => Value::Null,
                    }
                };
                (name.clone(), value)
            })
            .collect()
    }

    /// True when at least one property source in the slice requires the
    /// augmented-row evaluation path.
    fn has_expression_source(props: &[(String, PropertySource)]) -> bool {
        props
            .iter()
            .any(|(_, src)| matches!(src, PropertySource::Expression { .. }))
    }

    /// Builds a one-row chunk containing the input row plus the merged node
    /// in the column reserved for the MERGE variable.
    ///
    /// Used to evaluate `PropertySource::Expression` sources for ON CREATE /
    /// ON MATCH SET. The augmented chunk's schema matches `output_schema`.
    fn build_augmented_node_chunk(
        &self,
        chunk: Option<&DataChunk>,
        row: usize,
        merged_node: NodeId,
    ) -> DataChunk {
        let mut builder = DataChunkBuilder::with_capacity(&self.config.output_schema, 1);
        if let Some(input) = chunk {
            for col_idx in 0..input.column_count() {
                let val = input
                    .column(col_idx)
                    .and_then(|c| c.get_value(row))
                    .unwrap_or(Value::Null);
                if let Some(dst) = builder.column_mut(col_idx) {
                    dst.push_value(val);
                }
            }
        }
        if let Some(dst) = builder.column_mut(self.config.output_column) {
            dst.push_node_id(merged_node);
        }
        builder.advance_row();
        builder.finish()
    }

    /// Resolves an action-property source list (ON CREATE or ON MATCH) given
    /// the merged node id. Lazily builds the augmented chunk only if at least
    /// one source needs it.
    ///
    /// Returns an error only when an expression source is present but no
    /// search store was attached, which would be a planner/wiring bug.
    fn resolve_action_properties(
        &self,
        props: &[(String, PropertySource)],
        chunk: Option<&DataChunk>,
        row: usize,
        merged_node: NodeId,
    ) -> Result<Vec<(String, Value)>, super::OperatorError> {
        if !Self::has_expression_source(props) {
            // Fast path: no runtime expressions, fall through to the existing
            // resolver which understands Column/Constant/PropertyAccess.
            return Ok(Self::resolve_properties(
                props,
                chunk,
                row,
                self.store.as_ref(),
            ));
        }

        let augmented = self.build_augmented_node_chunk(chunk, row, merged_node);
        let mut out = Vec::with_capacity(props.len());
        for (name, source) in props {
            let value = match source {
                PropertySource::Expression {
                    expr,
                    variable_columns,
                } => {
                    let search_store = self.search_store.as_ref().ok_or_else(|| {
                        super::OperatorError::Execution(
                            "MERGE expression source requires search store; planner did not attach one"
                                .to_string(),
                        )
                    })?;
                    let mut predicate = ExpressionPredicate::new(
                        (**expr).clone(),
                        variable_columns.clone(),
                        Arc::clone(search_store),
                    )
                    .with_session_context(self.session_context.clone());
                    if let Some(epoch) = self.viewing_epoch {
                        predicate = predicate.with_transaction_context(epoch, self.transaction_id);
                    }
                    predicate.eval_at(&augmented, 0).unwrap_or(Value::Null)
                }
                _ => source.resolve(&augmented, 0, self.store.as_ref()),
            };
            out.push((name.clone(), value));
        }
        Ok(out)
    }

    /// Tries to find a matching node with the given resolved properties.
    fn find_matching_node(&self, resolved_match_props: &[(String, Value)]) -> Option<NodeId> {
        // Use a property index when available to avoid a full label scan.
        // Null conditions are excluded from the index query and verified in the loop.
        let use_index = resolved_match_props
            .iter()
            .any(|(k, v)| !v.is_null() && self.store.has_property_index(k));

        let candidates: Vec<NodeId> = if use_index {
            let conditions: Vec<(&str, Value)> = resolved_match_props
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            self.store.find_nodes_by_properties(&conditions)
        } else if let Some(first_label) = self.config.labels.first() {
            self.store.nodes_by_label(first_label)
        } else {
            self.store.node_ids()
        };

        for node_id in candidates {
            // Transactional creates write their version at `EpochId::PENDING`,
            // so the unversioned `get_node` (which checks visibility against
            // the current real epoch) hides nodes this same transaction has
            // just created. UNWIND-driven MERGE relies on seeing those rows
            // to dedupe, so route through the versioned read when we have a
            // transaction context attached.
            let node_opt = match (self.viewing_epoch, self.transaction_id) {
                (Some(epoch), Some(tid)) => self.store.get_node_versioned(node_id, epoch, tid),
                _ => self.store.get_node(node_id),
            };
            let Some(node) = node_opt else { continue };

            let has_all_labels = self.config.labels.iter().all(|label| node.has_label(label));
            if !has_all_labels {
                continue;
            }

            let has_all_props = resolved_match_props.iter().all(|(key, expected_value)| {
                let prop = node.properties.get(&PropertyKey::new(key.as_str()));
                if expected_value.is_null() {
                    // Null in a MERGE pattern matches both absent and explicitly null properties
                    prop.map_or(true, |v| v.is_null())
                } else {
                    prop.is_some_and(|v| v == expected_value)
                }
            });

            if has_all_props {
                return Some(node_id);
            }
        }

        None
    }

    /// Merges match and ON CREATE property lists, with ON CREATE values
    /// overriding match values for the same key.
    fn merge_node_props(
        resolved_match_props: &[(String, Value)],
        resolved_create_props: &[(String, Value)],
    ) -> Vec<(String, Value)> {
        let mut merged: Vec<(String, Value)> = resolved_match_props.to_vec();
        for (k, v) in resolved_create_props {
            if let Some(existing) = merged.iter_mut().find(|(key, _)| key == k) {
                existing.1 = v.clone();
            } else {
                merged.push((k.clone(), v.clone()));
            }
        }
        merged
    }

    /// Writes a freshly-created node's properties through the versioned
    /// API when the operator is participating in a transaction, so that
    /// rollback can undo them via the MVCC undo log. Falls back to the
    /// non-versioned setter only when no transaction context is attached
    /// (test paths and standalone operator construction).
    fn write_node_props(&self, id: NodeId, props: &[(PropertyKey, Value)]) {
        if let Some(tid) = self.transaction_id {
            for (key, value) in props {
                self.store
                    .set_node_property_versioned(id, key.as_str(), value.clone(), tid);
            }
        } else {
            for (key, value) in props {
                self.store
                    .set_node_property(id, key.as_str(), value.clone());
            }
        }
    }

    /// Creates a node through the versioned API so the create itself is
    /// tagged with the operator's transaction (when one is attached) and
    /// can be undone by transaction rollback. The non-versioned
    /// `create_node_with_props` would tag the create with
    /// [`TransactionId::SYSTEM`], leaving the node visible after the
    /// surrounding session transaction rolls back.
    fn store_create_node(&self, label_refs: &[&str]) -> NodeId {
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        let tx = self.transaction_id.unwrap_or(TransactionId::SYSTEM);
        self.store.create_node_versioned(label_refs, epoch, tx)
    }

    /// Creates a new node with the specified labels and resolved properties.
    fn create_node(
        &self,
        resolved_match_props: &[(String, Value)],
        resolved_create_props: &[(String, Value)],
    ) -> Result<NodeId, super::OperatorError> {
        let all_props = Self::merge_node_props(resolved_match_props, resolved_create_props);

        // Validate constraints before creating the node
        if let Some(ref validator) = self.validator {
            validator.validate_node_labels_allowed(&self.config.labels)?;
            for (name, value) in &all_props {
                validator.validate_node_property(&self.config.labels, name, value)?;
                validator.check_unique_node_property(&self.config.labels, name, value)?;
            }
            validator.validate_node_complete(&self.config.labels, &all_props)?;
        }

        let prop_pairs: Vec<(PropertyKey, Value)> = all_props
            .into_iter()
            .map(|(k, v)| (PropertyKey::new(k.as_str()), v))
            .collect();

        let labels: Vec<&str> = self.config.labels.iter().map(String::as_str).collect();
        let id = self.store_create_node(&labels);
        self.write_node_props(id, &prop_pairs);
        Ok(id)
    }

    /// Phase one of the two-phase create path: creates the node from match
    /// properties only, deferring the completeness check until ON CREATE
    /// expression properties are resolved (since those properties may
    /// satisfy NOT NULL / PRIMARY KEY requirements that match props alone
    /// would fail). Per-property type checks and uniqueness checks for the
    /// match properties still run here.
    ///
    /// Both the create and the property writes go through the versioned
    /// API, so a failure in phase two (or in `apply_on_match`) is undone
    /// when the surrounding session transaction rolls back. Without that,
    /// the node would persist as an orphan visible to later queries.
    fn create_node_phase_one(
        &self,
        resolved_match_props: &[(String, Value)],
    ) -> Result<NodeId, super::OperatorError> {
        if let Some(ref validator) = self.validator {
            validator.validate_node_labels_allowed(&self.config.labels)?;
            for (name, value) in resolved_match_props {
                validator.validate_node_property(&self.config.labels, name, value)?;
                validator.check_unique_node_property(&self.config.labels, name, value)?;
            }
        }

        let prop_pairs: Vec<(PropertyKey, Value)> = resolved_match_props
            .iter()
            .map(|(k, v)| (PropertyKey::new(k.as_str()), v.clone()))
            .collect();

        let labels: Vec<&str> = self.config.labels.iter().map(String::as_str).collect();
        let id = self.store_create_node(&labels);
        self.write_node_props(id, &prop_pairs);
        Ok(id)
    }

    /// Phase two of the two-phase create path: validates ON CREATE
    /// properties (type, uniqueness) and the full property set
    /// (completeness) after expressions have been evaluated against the
    /// freshly created node, but before the values are written. The just
    /// created node holds only match properties at this point, so a
    /// uniqueness check on an ON CREATE property cannot conflict with the
    /// node itself.
    fn validate_on_create_phase_two(
        &self,
        resolved_match_props: &[(String, Value)],
        resolved_create_props: &[(String, Value)],
    ) -> Result<(), super::OperatorError> {
        let Some(ref validator) = self.validator else {
            return Ok(());
        };
        for (name, value) in resolved_create_props {
            validator.validate_node_property(&self.config.labels, name, value)?;
            validator.check_unique_node_property(&self.config.labels, name, value)?;
        }
        let all_props = Self::merge_node_props(resolved_match_props, resolved_create_props);
        validator.validate_node_complete(&self.config.labels, &all_props)?;
        Ok(())
    }

    /// Finds or creates a matching node for a single row, applying ON MATCH/ON CREATE.
    fn merge_node_for_row(
        &self,
        chunk: Option<&DataChunk>,
        row: usize,
    ) -> Result<NodeId, super::OperatorError> {
        let store_ref: &dyn GraphStore = self.store.as_ref();
        // Match properties cannot reference the MERGE variable (ISO §15.5),
        // so they resolve against the input chunk directly.
        let resolved_match = resolve_match_properties(
            &self.config.match_properties,
            chunk,
            row,
            store_ref,
            MatchContext {
                search_store: self.search_store.as_ref(),
                session_context: &self.session_context,
                viewing_epoch: self.viewing_epoch,
                transaction_id: self.transaction_id,
            },
        )?;

        if let Some(existing_id) = self.find_matching_node(&resolved_match) {
            // Resolve ON MATCH SET against an augmented row containing the
            // matched node id, so `coalesce(n.x, 0)` can read the live value.
            let resolved_on_match = self.resolve_action_properties(
                &self.config.on_match_properties,
                chunk,
                row,
                existing_id,
            )?;
            self.apply_on_match(existing_id, &resolved_on_match)?;
            Ok(existing_id)
        } else if Self::has_expression_source(&self.config.on_create_properties) {
            // Two-phase create: build the node from match properties first so
            // the new id exists, then evaluate ON CREATE against an augmented
            // row referencing it, then write those properties via the same
            // path used for ON MATCH SET. Completeness and uniqueness on the
            // ON CREATE properties are validated between phases via
            // `validate_on_create_phase_two` so neither premature rejection
            // (when ON CREATE supplies a NOT NULL / PRIMARY KEY property) nor
            // silent constraint bypass (UNIQUE on an ON CREATE property)
            // occurs.
            let new_id = self.create_node_phase_one(&resolved_match)?;
            let resolved_on_create = self.resolve_action_properties(
                &self.config.on_create_properties,
                chunk,
                row,
                new_id,
            )?;
            self.validate_on_create_phase_two(&resolved_match, &resolved_on_create)?;
            self.apply_on_match(new_id, &resolved_on_create)?;
            Ok(new_id)
        } else {
            // Fast path: no runtime expressions; create with all properties at once.
            let resolved_on_create =
                Self::resolve_properties(&self.config.on_create_properties, chunk, row, store_ref);
            self.create_node(&resolved_match, &resolved_on_create)
        }
    }

    /// Applies ON MATCH properties to an existing node.
    fn apply_on_match(
        &self,
        node_id: NodeId,
        resolved_on_match: &[(String, Value)],
    ) -> Result<(), super::OperatorError> {
        for (key, value) in resolved_on_match {
            if let Some(ref validator) = self.validator {
                validator.validate_node_property(&self.config.labels, key, value)?;
            }
            if let Some(tid) = self.transaction_id {
                self.store
                    .set_node_property_versioned(node_id, key.as_str(), value.clone(), tid);
            } else {
                self.store
                    .set_node_property(node_id, key.as_str(), value.clone());
            }
        }
        Ok(())
    }
}

impl Operator for MergeOperator {
    fn next(&mut self) -> OperatorResult {
        // When we have an input operator, pass through input rows with the
        // merged node ID appended (used for chained inline MERGE patterns).
        if let Some(ref mut input) = self.input {
            if let Some(chunk) = input.next()? {
                let mut builder =
                    DataChunkBuilder::with_capacity(&self.config.output_schema, chunk.row_count());

                for row in chunk.selected_indices() {
                    // Reject NULL bound variables (e.g., from unmatched OPTIONAL MATCH)
                    if let Some(bound_col) = self.config.bound_variable_column {
                        let is_null = chunk.column(bound_col).map_or(true, |col| col.is_null(row));
                        if is_null {
                            return Err(super::OperatorError::TypeMismatch {
                                expected: format!(
                                    "non-null node for MERGE variable '{}'",
                                    self.config.variable
                                ),
                                found: "NULL".to_string(),
                            });
                        }
                    }

                    // Merge the node per-row: resolve properties from this row
                    let node_id = self.merge_node_for_row(Some(&chunk), row)?;

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

                    // Append the merged node ID
                    if let Some(dst) = builder.column_mut(self.config.output_column) {
                        dst.push_node_id(node_id);
                    }

                    builder.advance_row();
                }

                return Ok(Some(builder.finish()));
            }
            return Ok(None);
        }

        // Standalone mode (no input operator)
        if self.executed {
            return Ok(None);
        }
        self.executed = true;

        let node_id = self.merge_node_for_row(None, 0)?;

        let mut builder = DataChunkBuilder::new(&self.config.output_schema);
        if let Some(dst) = builder.column_mut(self.config.output_column) {
            dst.push_node_id(node_id);
        }
        builder.advance_row();

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.executed = false;
        if let Some(ref mut input) = self.input {
            input.reset();
        }
    }

    fn name(&self) -> &'static str {
        "Merge"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Configuration for a relationship merge operation.
pub struct MergeRelationshipConfig {
    /// Column index for the source node ID in the input.
    pub source_column: usize,
    /// Column index for the target node ID in the input.
    pub target_column: usize,
    /// Variable name for the source node (for error messages).
    pub source_variable: String,
    /// Variable name for the target node (for error messages).
    pub target_variable: String,
    /// Relationship type to match/create.
    pub edge_type: String,
    /// Properties that must match (also used for creation).
    pub match_properties: Vec<(String, PropertySource)>,
    /// Properties to set on CREATE.
    pub on_create_properties: Vec<(String, PropertySource)>,
    /// Properties to set on MATCH.
    pub on_match_properties: Vec<(String, PropertySource)>,
    /// Output schema (input columns + edge column).
    pub output_schema: Vec<LogicalType>,
    /// Column index for the edge variable in the output.
    pub edge_output_column: usize,
}

/// Merge operator for relationship patterns.
///
/// Takes input rows containing source and target node IDs, then for each row:
/// 1. Searches for an existing relationship matching the type and properties
/// 2. If found, applies ON MATCH properties and returns the existing edge
/// 3. If not found, creates a new relationship and applies ON CREATE properties
pub struct MergeRelationshipOperator {
    /// The graph store.
    store: Arc<dyn GraphStoreMut>,
    /// Input operator providing rows with source/target node columns.
    input: Box<dyn Operator>,
    /// Merge configuration.
    config: MergeRelationshipConfig,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for undo log tracking.
    transaction_id: Option<TransactionId>,
    /// Optional constraint validator for schema enforcement.
    validator: Option<Arc<dyn ConstraintValidator>>,
    /// Search-store handle for evaluating `PropertySource::Expression`.
    search_store: Option<Arc<dyn GraphStoreSearch>>,
    /// Session context for expression evaluation.
    session_context: SessionContext,
}

impl MergeRelationshipOperator {
    /// Creates a new merge relationship operator.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        config: MergeRelationshipConfig,
    ) -> Self {
        Self {
            store,
            input,
            config,
            viewing_epoch: None,
            transaction_id: None,
            validator: None,
            search_store: None,
            session_context: SessionContext::default(),
        }
    }

    /// Sets the transaction context for versioned mutations.
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

    /// Provides a search-store handle for runtime expression evaluation.
    #[must_use]
    pub fn with_search_store(mut self, search_store: Arc<dyn GraphStoreSearch>) -> Self {
        self.search_store = Some(search_store);
        self
    }

    /// Sets the session context used during expression evaluation.
    #[must_use]
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.session_context = context;
        self
    }

    /// Builds a one-row chunk containing the input row plus the merged edge
    /// in the column reserved for the MERGE relationship variable.
    fn build_augmented_edge_chunk(
        &self,
        chunk: &DataChunk,
        row: usize,
        merged_edge: EdgeId,
    ) -> DataChunk {
        let mut builder = DataChunkBuilder::with_capacity(&self.config.output_schema, 1);
        for col_idx in 0..chunk.column_count() {
            let val = chunk
                .column(col_idx)
                .and_then(|c| c.get_value(row))
                .unwrap_or(Value::Null);
            if let Some(dst) = builder.column_mut(col_idx) {
                dst.push_value(val);
            }
        }
        if let Some(dst) = builder.column_mut(self.config.edge_output_column) {
            dst.push_edge_id(merged_edge);
        }
        builder.advance_row();
        builder.finish()
    }

    /// Resolves an action-property list (ON CREATE / ON MATCH SET) against
    /// an augmented row that includes the merged edge id. Falls back to the
    /// fast path when no expression sources are present.
    fn resolve_action_properties(
        &self,
        props: &[(String, PropertySource)],
        chunk: &DataChunk,
        row: usize,
        merged_edge: EdgeId,
    ) -> Result<Vec<(String, Value)>, super::OperatorError> {
        if !MergeOperator::has_expression_source(props) {
            return Ok(MergeOperator::resolve_properties(
                props,
                Some(chunk),
                row,
                self.store.as_ref(),
            ));
        }

        let augmented = self.build_augmented_edge_chunk(chunk, row, merged_edge);
        let mut out = Vec::with_capacity(props.len());
        for (name, source) in props {
            let value = match source {
                PropertySource::Expression {
                    expr,
                    variable_columns,
                } => {
                    let search_store = self.search_store.as_ref().ok_or_else(|| {
                        super::OperatorError::Execution(
                            "MERGE expression source requires search store; planner did not attach one"
                                .to_string(),
                        )
                    })?;
                    let mut predicate = ExpressionPredicate::new(
                        (**expr).clone(),
                        variable_columns.clone(),
                        Arc::clone(search_store),
                    )
                    .with_session_context(self.session_context.clone());
                    if let Some(epoch) = self.viewing_epoch {
                        predicate = predicate.with_transaction_context(epoch, self.transaction_id);
                    }
                    predicate.eval_at(&augmented, 0).unwrap_or(Value::Null)
                }
                _ => source.resolve(&augmented, 0, self.store.as_ref()),
            };
            out.push((name.clone(), value));
        }
        Ok(out)
    }

    /// Tries to find a matching relationship between source and target.
    fn find_matching_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        resolved_match_props: &[(String, Value)],
    ) -> Option<EdgeId> {
        use crate::graph::Direction;

        for (target, edge_id) in self.store.edges_from(src, Direction::Outgoing) {
            if target != dst {
                continue;
            }

            // Same as `find_matching_node`: edges this transaction created
            // earlier in the statement sit at `EpochId::PENDING`, so the
            // unversioned read would hide them and every repeated row would
            // create another edge.
            let edge_opt = match (self.viewing_epoch, self.transaction_id) {
                (Some(epoch), Some(tid)) => self.store.get_edge_versioned(edge_id, epoch, tid),
                _ => self.store.get_edge(edge_id),
            };
            if let Some(edge) = edge_opt {
                if edge.edge_type.as_str() != self.config.edge_type {
                    continue;
                }

                let has_all_props = resolved_match_props.iter().all(|(key, expected)| {
                    let prop = edge.get_property(key);
                    if expected.is_null() {
                        // Null in a MERGE pattern matches both absent and explicitly null properties
                        prop.is_none_or(Value::is_null)
                    } else {
                        prop.is_some_and(|v| v == expected)
                    }
                });

                if has_all_props {
                    return Some(edge_id);
                }
            }
        }

        None
    }

    /// Versioned-API edge create. See [`MergeOperator::store_create_node`]
    /// for the rationale: the create itself must be tagged with the
    /// operator's transaction so that rollback can undo it.
    fn store_create_edge(&self, src: NodeId, dst: NodeId) -> EdgeId {
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        let tx = self.transaction_id.unwrap_or(TransactionId::SYSTEM);
        self.store
            .create_edge_versioned(src, dst, &self.config.edge_type, epoch, tx)
    }

    /// Writes a freshly-created edge's properties through the versioned
    /// setter when a transaction is attached, mirroring
    /// [`MergeOperator::write_node_props`].
    fn write_edge_props(&self, id: EdgeId, props: &[(PropertyKey, Value)]) {
        if let Some(tid) = self.transaction_id {
            for (key, value) in props {
                self.store
                    .set_edge_property_versioned(id, key.as_str(), value.clone(), tid);
            }
        } else {
            for (key, value) in props {
                self.store
                    .set_edge_property(id, key.as_str(), value.clone());
            }
        }
    }

    /// Creates a new edge with resolved match and on_create properties.
    fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        resolved_match_props: &[(String, Value)],
        resolved_create_props: &[(String, Value)],
    ) -> Result<EdgeId, super::OperatorError> {
        let all_props =
            MergeOperator::merge_node_props(resolved_match_props, resolved_create_props);

        // Validate constraints before creating the edge
        if let Some(ref validator) = self.validator {
            validator.validate_edge_type_allowed(&self.config.edge_type)?;
            for (name, value) in &all_props {
                validator.validate_edge_property(&self.config.edge_type, name, value)?;
            }
            validator.validate_edge_complete(&self.config.edge_type, &all_props)?;
        }

        let prop_pairs: Vec<(PropertyKey, Value)> = all_props
            .into_iter()
            .map(|(k, v)| (PropertyKey::new(k.as_str()), v))
            .collect();

        let id = self.store_create_edge(src, dst);
        self.write_edge_props(id, &prop_pairs);
        Ok(id)
    }

    /// Phase one of the two-phase edge create path: validates per-property
    /// types on match props and writes the edge, deferring the completeness
    /// check until ON CREATE expression properties are resolved. See
    /// [`MergeOperator::create_node_phase_one`] for the rationale.
    ///
    /// Both the create and the property writes go through the versioned
    /// API, so a failure in phase two (or in `apply_on_match_edge`) is
    /// undone when the surrounding session transaction rolls back.
    fn create_edge_phase_one(
        &self,
        src: NodeId,
        dst: NodeId,
        resolved_match_props: &[(String, Value)],
    ) -> Result<EdgeId, super::OperatorError> {
        if let Some(ref validator) = self.validator {
            validator.validate_edge_type_allowed(&self.config.edge_type)?;
            for (name, value) in resolved_match_props {
                validator.validate_edge_property(&self.config.edge_type, name, value)?;
            }
        }

        let prop_pairs: Vec<(PropertyKey, Value)> = resolved_match_props
            .iter()
            .map(|(k, v)| (PropertyKey::new(k.as_str()), v.clone()))
            .collect();

        let id = self.store_create_edge(src, dst);
        self.write_edge_props(id, &prop_pairs);
        Ok(id)
    }

    /// Phase two of the two-phase edge create path: validates ON CREATE
    /// edge properties and the full property set for completeness after
    /// expressions have been evaluated against the freshly created edge.
    fn validate_on_create_edge_phase_two(
        &self,
        resolved_match_props: &[(String, Value)],
        resolved_create_props: &[(String, Value)],
    ) -> Result<(), super::OperatorError> {
        let Some(ref validator) = self.validator else {
            return Ok(());
        };
        for (name, value) in resolved_create_props {
            validator.validate_edge_property(&self.config.edge_type, name, value)?;
        }
        let all_props =
            MergeOperator::merge_node_props(resolved_match_props, resolved_create_props);
        validator.validate_edge_complete(&self.config.edge_type, &all_props)?;
        Ok(())
    }

    /// Applies ON MATCH properties to an existing edge.
    fn apply_on_match_edge(
        &self,
        edge_id: EdgeId,
        resolved_on_match: &[(String, Value)],
    ) -> Result<(), super::OperatorError> {
        for (key, value) in resolved_on_match {
            if let Some(ref validator) = self.validator {
                validator.validate_edge_property(&self.config.edge_type, key, value)?;
            }
            if let Some(tid) = self.transaction_id {
                self.store
                    .set_edge_property_versioned(edge_id, key.as_str(), value.clone(), tid);
            } else {
                self.store
                    .set_edge_property(edge_id, key.as_str(), value.clone());
            }
        }
        Ok(())
    }
}

impl Operator for MergeRelationshipOperator {
    fn next(&mut self) -> OperatorResult {
        use super::OperatorError;

        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.config.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let src_val = chunk
                    .column(self.config.source_column)
                    .and_then(|c| c.get_node_id(row))
                    .ok_or_else(|| OperatorError::TypeMismatch {
                        expected: format!(
                            "non-null node for MERGE variable '{}'",
                            self.config.source_variable
                        ),
                        found: "NULL".to_string(),
                    })?;

                let dst_val = chunk
                    .column(self.config.target_column)
                    .and_then(|c| c.get_node_id(row))
                    .ok_or_else(|| OperatorError::TypeMismatch {
                        expected: format!(
                            "non-null node for MERGE variable '{}'",
                            self.config.target_variable
                        ),
                        found: "None".to_string(),
                    })?;

                let store_ref: &dyn GraphStore = self.store.as_ref();
                let resolved_match = resolve_match_properties(
                    &self.config.match_properties,
                    Some(&chunk),
                    row,
                    store_ref,
                    MatchContext {
                        search_store: self.search_store.as_ref(),
                        session_context: &self.session_context,
                        viewing_epoch: self.viewing_epoch,
                        transaction_id: self.transaction_id,
                    },
                )?;

                let edge_id = if let Some(existing) =
                    self.find_matching_edge(src_val, dst_val, &resolved_match)
                {
                    let resolved_on_match = self.resolve_action_properties(
                        &self.config.on_match_properties,
                        &chunk,
                        row,
                        existing,
                    )?;
                    self.apply_on_match_edge(existing, &resolved_on_match)?;
                    existing
                } else if MergeOperator::has_expression_source(&self.config.on_create_properties) {
                    // Two-phase create so ON CREATE expressions can reference
                    // the new edge. Completeness validation is deferred to
                    // `validate_on_create_edge_phase_two` so an ON CREATE
                    // property is allowed to satisfy a NOT NULL constraint
                    // that match properties alone would fail.
                    let new_id = self.create_edge_phase_one(src_val, dst_val, &resolved_match)?;
                    let resolved_on_create = self.resolve_action_properties(
                        &self.config.on_create_properties,
                        &chunk,
                        row,
                        new_id,
                    )?;
                    self.validate_on_create_edge_phase_two(&resolved_match, &resolved_on_create)?;
                    self.apply_on_match_edge(new_id, &resolved_on_create)?;
                    new_id
                } else {
                    let resolved_on_create = MergeOperator::resolve_properties(
                        &self.config.on_create_properties,
                        Some(&chunk),
                        row,
                        store_ref,
                    );
                    self.create_edge(src_val, dst_val, &resolved_match, &resolved_on_create)?
                };

                // Copy input columns to output, then add the edge column
                for col_idx in 0..self.config.output_schema.len() {
                    if col_idx == self.config.edge_output_column {
                        if let Some(dst_col) = builder.column_mut(col_idx) {
                            dst_col.push_edge_id(edge_id);
                        }
                    } else if let (Some(src_col), Some(dst_col)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                        && let Some(val) = src_col.get_value(row)
                    {
                        dst_col.push_value(val);
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
        "MergeRelationship"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// What a MERGE needs to evaluate computed match properties.
struct MatchContext<'a> {
    search_store: Option<&'a Arc<dyn GraphStoreSearch>>,
    session_context: &'a SessionContext,
    viewing_epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
}

/// Resolves MERGE match properties for one row. Computed values such as
/// `MERGE (:X {id: toString(i)})` are evaluated against the input row; they
/// used to resolve to NULL, so every row matched or created the same node.
fn resolve_match_properties(
    props: &[(String, PropertySource)],
    chunk: Option<&DataChunk>,
    row: usize,
    store: &dyn GraphStore,
    context: MatchContext<'_>,
) -> Result<Vec<(String, Value)>, OperatorError> {
    if !MergeOperator::has_expression_source(props) {
        return Ok(MergeOperator::resolve_properties(props, chunk, row, store));
    }
    let chunk = chunk.ok_or_else(|| {
        OperatorError::Execution(
            "computed MERGE property without an input row; planner did not provide one".to_string(),
        )
    })?;
    super::mutation::PropertyExpressions::new(
        context.search_store.cloned(),
        context.session_context.clone(),
    )
    .resolve_row(
        props,
        chunk,
        row,
        store,
        context.viewing_epoch,
        context.transaction_id,
    )
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

    fn const_props(props: Vec<(&str, Value)>) -> Vec<(String, PropertySource)> {
        props
            .into_iter()
            .map(|(k, v)| (k.to_string(), PropertySource::Constant(v)))
            .collect()
    }

    #[test]
    fn test_merge_creates_new_node() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // MERGE should create a new node since none exists
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Alix".into()))]),
                on_create_properties: vec![],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let result = merge.next().unwrap();
        assert!(result.is_some());

        // Verify node was created
        let nodes = store.nodes_by_label("Person");
        assert_eq!(nodes.len(), 1);

        let node = store.get_node(nodes[0]).unwrap();
        assert!(node.has_label("Person"));
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String("Alix".into()))
        );
    }

    #[test]
    fn test_merge_matches_existing_node() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // Create an existing node
        store.create_node_with_props(
            &["Person"],
            &[(PropertyKey::new("name"), Value::String("Gus".into()))],
        );

        // MERGE should find the existing node
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Gus".into()))]),
                on_create_properties: vec![],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let result = merge.next().unwrap();
        assert!(result.is_some());

        // Verify only one node exists (no new node created)
        let nodes = store.nodes_by_label("Person");
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_merge_with_on_create() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // MERGE with ON CREATE SET
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Vincent".into()))]),
                on_create_properties: const_props(vec![("created", Value::Bool(true))]),
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let _ = merge.next().unwrap();

        // Verify node has both match properties and on_create properties
        let nodes = store.nodes_by_label("Person");
        let node = store.get_node(nodes[0]).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String("Vincent".into()))
        );
        assert_eq!(
            node.properties.get(&PropertyKey::new("created")),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn test_merge_with_on_match() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // Create an existing node
        let node_id = store.create_node_with_props(
            &["Person"],
            &[(PropertyKey::new("name"), Value::String("Jules".into()))],
        );

        // MERGE with ON MATCH SET
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Jules".into()))]),
                on_create_properties: vec![],
                on_match_properties: const_props(vec![("updated", Value::Bool(true))]),
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let _ = merge.next().unwrap();

        // Verify node has the on_match property added
        let node = store.get_node(node_id).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("updated")),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn test_merge_uses_property_index() {
        let lpg_store = Arc::new(LpgStore::new().unwrap());
        lpg_store.create_property_index("name");
        assert!(lpg_store.has_property_index("name"));

        // Use the trait object for node creation so the &[(PropertyKey, Value)] signature applies.
        let store: Arc<dyn GraphStoreMut> = lpg_store;

        for i in 0..50u32 {
            store.create_node_with_props(
                &["Person"],
                &[(
                    PropertyKey::new("name"),
                    Value::String(format!("person_{i}").into()),
                )],
            );
        }

        let target_id = store.create_node_with_props(
            &["Person"],
            &[(PropertyKey::new("name"), Value::String("Beatrix".into()))],
        );

        // MERGE should find the existing node via index lookup
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Beatrix".into()))]),
                on_create_properties: vec![],
                on_match_properties: const_props(vec![("found", Value::Bool(true))]),
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let result = merge.next().unwrap();
        assert!(result.is_some());

        // ON MATCH should have fired on the correct node
        let node = store.get_node(target_id).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("found")),
            Some(&Value::Bool(true))
        );

        // No new node should have been created
        let persons = store.nodes_by_label("Person");
        assert_eq!(persons.len(), 51);
    }

    #[test]
    fn test_merge_creates_via_index_miss() {
        let lpg_store = Arc::new(LpgStore::new().unwrap());
        lpg_store.create_property_index("name");

        let store: Arc<dyn GraphStoreMut> = lpg_store;

        store.create_node_with_props(
            &["Person"],
            &[(PropertyKey::new("name"), Value::String("Django".into()))],
        );

        // MERGE for a name not in the index — should create
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Shosanna".into()))]),
                on_create_properties: const_props(vec![("created", Value::Bool(true))]),
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let result = merge.next().unwrap();
        assert!(result.is_some());

        let persons = store.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);

        let new_nodes: Vec<_> = persons
            .iter()
            .filter_map(|&id| store.get_node(id))
            .filter(|n| {
                n.properties.get(&PropertyKey::new("name"))
                    == Some(&Value::String("Shosanna".into()))
            })
            .collect();
        assert_eq!(new_nodes.len(), 1);
        assert_eq!(
            new_nodes[0].properties.get(&PropertyKey::new("created")),
            Some(&Value::Bool(true))
        );
    }

    // GrafeoDB/grafeo#317. Operator-level test: a `PropertySource::Expression`
    // for ON CREATE / ON MATCH SET must evaluate against an augmented row that
    // contains the merged node, not against the (potentially absent) input row.

    #[test]
    fn test_merge_on_match_resolves_expression_against_merged_node() {
        use super::super::filter::FilterExpression;
        use crate::graph::lpg::LpgStore;
        use std::collections::HashMap;

        let lpg = Arc::new(LpgStore::new().unwrap());
        let store: Arc<dyn GraphStoreMut> = Arc::clone(&lpg) as Arc<dyn GraphStoreMut>;
        let search: Arc<dyn GraphStoreSearch> = Arc::clone(&lpg) as Arc<dyn GraphStoreSearch>;

        // Pre-create the matching node so the MERGE goes into the ON MATCH branch.
        let id = store.create_node_with_props(
            &["Item"],
            &[
                (PropertyKey::new("val"), Value::Int64(1)),
                (PropertyKey::new("x"), Value::Int64(7)),
            ],
        );

        // ON MATCH SET n.x = n.x + 5
        let expr = FilterExpression::Binary {
            left: Box::new(FilterExpression::Property {
                variable: "n".to_string(),
                property: "x".to_string(),
            }),
            op: super::super::filter::BinaryFilterOp::Add,
            right: Box::new(FilterExpression::Literal(Value::Int64(5))),
        };
        let mut variable_columns = HashMap::new();
        // Standalone MERGE: input is None, so the augmented row only has the
        // MERGE variable column at index 0.
        variable_columns.insert("n".to_string(), 0_usize);

        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Item".to_string()],
                match_properties: const_props(vec![("val", Value::Int64(1))]),
                on_create_properties: vec![],
                on_match_properties: vec![(
                    "x".to_string(),
                    PropertySource::Expression {
                        expr: Box::new(expr),
                        variable_columns,
                    },
                )],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        )
        .with_search_store(Arc::clone(&search));

        merge.next().unwrap();

        let node = store.get_node(id).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("x")),
            Some(&Value::Int64(12)),
            "ON MATCH expression must read the merged node, not NULL"
        );
    }

    #[test]
    fn test_merge_on_create_resolves_expression_against_new_node() {
        // ON CREATE coalesce(n.x, 99) must see the freshly-created node and
        // fall back to 99 because `x` is not yet set on it.
        use super::super::filter::FilterExpression;
        use crate::graph::lpg::LpgStore;
        use std::collections::HashMap;

        let lpg = Arc::new(LpgStore::new().unwrap());
        let store: Arc<dyn GraphStoreMut> = Arc::clone(&lpg) as Arc<dyn GraphStoreMut>;
        let search: Arc<dyn GraphStoreSearch> = Arc::clone(&lpg) as Arc<dyn GraphStoreSearch>;

        let coalesce = FilterExpression::FunctionCall {
            name: "coalesce".to_string(),
            args: vec![
                FilterExpression::Property {
                    variable: "n".to_string(),
                    property: "x".to_string(),
                },
                FilterExpression::Literal(Value::Int64(99)),
            ],
        };
        let mut variable_columns = HashMap::new();
        variable_columns.insert("n".to_string(), 0_usize);

        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Item".to_string()],
                match_properties: const_props(vec![("val", Value::Int64(1))]),
                on_create_properties: vec![(
                    "x".to_string(),
                    PropertySource::Expression {
                        expr: Box::new(coalesce),
                        variable_columns,
                    },
                )],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        )
        .with_search_store(Arc::clone(&search));

        merge.next().unwrap();

        let nodes = store.nodes_by_label("Item");
        assert_eq!(nodes.len(), 1);
        let node = store.get_node(nodes[0]).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("x")),
            Some(&Value::Int64(99))
        );
    }

    // ── Two-phase constraint validation regression tests ──────────────
    //
    // The two-phase create path (ON CREATE expression sources) used to call
    // `create_node` / `create_edge` with an empty on_create list, which made
    // `validate_node_complete` and `check_unique_node_property` only see the
    // match properties. The fix routes the two phases through dedicated
    // helpers that validate the full property set at the right time.

    use super::ConstraintValidator;

    /// Minimal validator that enforces NOT NULL on a single named property.
    struct RequirePropertyValidator {
        required_property: &'static str,
    }

    impl ConstraintValidator for RequirePropertyValidator {
        fn validate_node_property(
            &self,
            _labels: &[String],
            _key: &str,
            _value: &Value,
        ) -> Result<(), super::super::OperatorError> {
            Ok(())
        }
        fn validate_node_complete(
            &self,
            _labels: &[String],
            properties: &[(String, Value)],
        ) -> Result<(), super::super::OperatorError> {
            if !properties.iter().any(|(k, _)| k == self.required_property) {
                return Err(super::super::OperatorError::ConstraintViolation(format!(
                    "missing required property '{}'",
                    self.required_property
                )));
            }
            Ok(())
        }
        fn check_unique_node_property(
            &self,
            _labels: &[String],
            _key: &str,
            _value: &Value,
        ) -> Result<(), super::super::OperatorError> {
            Ok(())
        }
        fn validate_edge_property(
            &self,
            _edge_type: &str,
            _key: &str,
            _value: &Value,
        ) -> Result<(), super::super::OperatorError> {
            Ok(())
        }
        fn validate_edge_complete(
            &self,
            _edge_type: &str,
            properties: &[(String, Value)],
        ) -> Result<(), super::super::OperatorError> {
            if !properties.iter().any(|(k, _)| k == self.required_property) {
                return Err(super::super::OperatorError::ConstraintViolation(format!(
                    "missing required edge property '{}'",
                    self.required_property
                )));
            }
            Ok(())
        }
    }

    /// Validator that records every uniqueness check it sees, so the test
    /// can assert ON CREATE properties were not silently bypassed.
    struct RecordingUniqueValidator {
        seen: std::sync::Mutex<Vec<(String, Value)>>,
    }

    impl RecordingUniqueValidator {
        fn new() -> Self {
            Self {
                seen: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl ConstraintValidator for RecordingUniqueValidator {
        fn validate_node_property(
            &self,
            _labels: &[String],
            _key: &str,
            _value: &Value,
        ) -> Result<(), super::super::OperatorError> {
            Ok(())
        }
        fn validate_node_complete(
            &self,
            _labels: &[String],
            _properties: &[(String, Value)],
        ) -> Result<(), super::super::OperatorError> {
            Ok(())
        }
        fn check_unique_node_property(
            &self,
            _labels: &[String],
            key: &str,
            value: &Value,
        ) -> Result<(), super::super::OperatorError> {
            self.seen
                .lock()
                .unwrap()
                .push((key.to_string(), value.clone()));
            Ok(())
        }
        fn validate_edge_property(
            &self,
            _edge_type: &str,
            _key: &str,
            _value: &Value,
        ) -> Result<(), super::super::OperatorError> {
            Ok(())
        }
        fn validate_edge_complete(
            &self,
            _edge_type: &str,
            _properties: &[(String, Value)],
        ) -> Result<(), super::super::OperatorError> {
            Ok(())
        }
    }

    fn coalesce_n_x_else(default: i64) -> super::super::filter::FilterExpression {
        use super::super::filter::FilterExpression;
        FilterExpression::FunctionCall {
            name: "coalesce".to_string(),
            args: vec![
                FilterExpression::Property {
                    variable: "n".to_string(),
                    property: "x".to_string(),
                },
                FilterExpression::Literal(Value::Int64(default)),
            ],
        }
    }

    #[test]
    fn test_merge_two_phase_completeness_uses_full_property_set() {
        // Regression: phase one used to run completeness against match
        // properties only, falsely rejecting an ON CREATE property that
        // satisfies a NOT NULL requirement. With the fix, completeness is
        // checked once both phases have produced their properties.
        use crate::graph::lpg::LpgStore;
        use std::collections::HashMap;

        let lpg = Arc::new(LpgStore::new().unwrap());
        let store: Arc<dyn GraphStoreMut> = Arc::clone(&lpg) as Arc<dyn GraphStoreMut>;
        let search: Arc<dyn GraphStoreSearch> = Arc::clone(&lpg) as Arc<dyn GraphStoreSearch>;

        let mut variable_columns = HashMap::new();
        variable_columns.insert("n".to_string(), 0_usize);

        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Item".to_string()],
                match_properties: const_props(vec![("val", Value::Int64(1))]),
                // ON CREATE supplies the NOT NULL property `x`.
                on_create_properties: vec![(
                    "x".to_string(),
                    PropertySource::Expression {
                        expr: Box::new(coalesce_n_x_else(99)),
                        variable_columns,
                    },
                )],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        )
        .with_search_store(Arc::clone(&search))
        .with_validator(Arc::new(RequirePropertyValidator {
            required_property: "x",
        }));

        merge
            .next()
            .expect("MERGE must succeed because ON CREATE supplies the required property");

        let nodes = store.nodes_by_label("Item");
        assert_eq!(nodes.len(), 1);
        let node = store.get_node(nodes[0]).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("x")),
            Some(&Value::Int64(99)),
            "ON CREATE expression value must be persisted"
        );
    }

    #[test]
    fn test_merge_two_phase_unique_check_runs_on_on_create_props() {
        // Regression: phase one used to skip uniqueness checks on ON CREATE
        // properties because the empty list passed to `create_node` hid
        // them. The fix runs `check_unique_node_property` for ON CREATE
        // values in phase two.
        use crate::graph::lpg::LpgStore;
        use std::collections::HashMap;

        let lpg = Arc::new(LpgStore::new().unwrap());
        let store: Arc<dyn GraphStoreMut> = Arc::clone(&lpg) as Arc<dyn GraphStoreMut>;
        let search: Arc<dyn GraphStoreSearch> = Arc::clone(&lpg) as Arc<dyn GraphStoreSearch>;

        let mut variable_columns = HashMap::new();
        variable_columns.insert("n".to_string(), 0_usize);

        let recorder = Arc::new(RecordingUniqueValidator::new());

        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Item".to_string()],
                match_properties: const_props(vec![("val", Value::Int64(1))]),
                on_create_properties: vec![(
                    "x".to_string(),
                    PropertySource::Expression {
                        expr: Box::new(coalesce_n_x_else(42)),
                        variable_columns,
                    },
                )],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        )
        .with_search_store(Arc::clone(&search))
        .with_validator(Arc::clone(&recorder) as Arc<dyn ConstraintValidator>);

        merge.next().unwrap();

        let seen = recorder.seen.lock().unwrap().clone();
        assert!(
            seen.iter().any(|(k, v)| k == "x" && *v == Value::Int64(42)),
            "uniqueness check must fire for ON CREATE expression property `x`, observed: {seen:?}"
        );
    }

    #[test]
    fn test_merge_relationship_two_phase_completeness_uses_full_property_set() {
        // Edge-equivalent of the node completeness regression.
        use super::super::filter::FilterExpression;
        use crate::execution::chunk::DataChunkBuilder;
        use crate::graph::lpg::LpgStore;
        use std::collections::HashMap;

        let lpg = Arc::new(LpgStore::new().unwrap());
        let store: Arc<dyn GraphStoreMut> = Arc::clone(&lpg) as Arc<dyn GraphStoreMut>;
        let search: Arc<dyn GraphStoreSearch> = Arc::clone(&lpg) as Arc<dyn GraphStoreSearch>;

        let src_id = store.create_node_with_props(
            &["Node"],
            &[(PropertyKey::new("name"), Value::String("Vincent".into()))],
        );
        let dst_id = store.create_node_with_props(
            &["Node"],
            &[(PropertyKey::new("name"), Value::String("Mia".into()))],
        );

        // Build an input chunk: [src_id, dst_id] with the edge column at index 2.
        let input_schema = vec![LogicalType::Node, LogicalType::Node];
        let mut builder = DataChunkBuilder::with_capacity(&input_schema, 1);
        builder.column_mut(0).unwrap().push_node_id(src_id);
        builder.column_mut(1).unwrap().push_node_id(dst_id);
        builder.advance_row();
        let chunk = builder.finish();

        struct OneShot(Option<DataChunk>);
        impl Operator for OneShot {
            fn next(&mut self) -> OperatorResult {
                Ok(self.0.take())
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "OneShot"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        // ON CREATE supplies NOT NULL property `x` via expression.
        let coalesce = FilterExpression::FunctionCall {
            name: "coalesce".to_string(),
            args: vec![
                FilterExpression::Property {
                    variable: "r".to_string(),
                    property: "x".to_string(),
                },
                FilterExpression::Literal(Value::Int64(7)),
            ],
        };
        let mut variable_columns = HashMap::new();
        // Augmented edge chunk: [src, dst, r] → r at index 2.
        variable_columns.insert("r".to_string(), 2_usize);

        let mut merge_rel = MergeRelationshipOperator::new(
            Arc::clone(&store),
            Box::new(OneShot(Some(chunk))),
            MergeRelationshipConfig {
                source_column: 0,
                target_column: 1,
                source_variable: "a".to_string(),
                target_variable: "b".to_string(),
                edge_type: "KNOWS".to_string(),
                match_properties: vec![],
                on_create_properties: vec![(
                    "x".to_string(),
                    PropertySource::Expression {
                        expr: Box::new(coalesce),
                        variable_columns,
                    },
                )],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node, LogicalType::Node, LogicalType::Edge],
                edge_output_column: 2,
            },
        )
        .with_search_store(Arc::clone(&search))
        .with_validator(Arc::new(RequirePropertyValidator {
            required_property: "x",
        }));

        merge_rel.next().expect(
            "MERGE relationship must succeed because ON CREATE supplies the required property",
        );

        // Confirm the edge was created with `x` set to the expression value.
        use crate::graph::Direction;
        let edges: Vec<EdgeId> = store
            .edges_from(src_id, Direction::Outgoing)
            .into_iter()
            .filter_map(|(target, edge_id)| (target == dst_id).then_some(edge_id))
            .collect();
        assert_eq!(edges.len(), 1, "expected exactly one outgoing edge");
        let edge = store.get_edge(edges[0]).unwrap();
        assert_eq!(edge.get_property("x"), Some(&Value::Int64(7)));
    }

    #[test]
    fn test_merge_in_transaction_dedupes_within_unwind() {
        // Regression: MERGE inside UNWIND, executed in a transaction (auto-
        // commit or otherwise), tags its creates at `EpochId::PENDING`.
        // `find_matching_node`'s read path used to call the unversioned
        // `get_node`, which rejects PENDING records, so subsequent rows of
        // the same UNWIND could not see the node the operator had just
        // created and produced a duplicate per row.
        use crate::execution::chunk::DataChunkBuilder;
        use crate::graph::lpg::LpgStore;
        use grafeo_common::types::EpochId;

        let lpg = Arc::new(LpgStore::new().unwrap());
        let store: Arc<dyn GraphStoreMut> = Arc::clone(&lpg) as Arc<dyn GraphStoreMut>;

        // Build an input chunk emulating `UNWIND [1, 1, 1] AS i`.
        let input_schema = vec![LogicalType::Int64];
        let mut builder = DataChunkBuilder::with_capacity(&input_schema, 3);
        for _ in 0..3 {
            builder.column_mut(0).unwrap().push_value(Value::Int64(1));
            builder.advance_row();
        }
        let chunk = builder.finish();

        struct OneShot(Option<DataChunk>);
        impl Operator for OneShot {
            fn next(&mut self) -> OperatorResult {
                Ok(self.0.take())
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "OneShot"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        // Use a non-SYSTEM transaction so versioned creates land at PENDING.
        let tx = TransactionId::new(1);
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            Some(Box::new(OneShot(Some(chunk)))),
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Item".to_string()],
                match_properties: vec![("val".to_string(), PropertySource::Column(0))],
                on_create_properties: vec![],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Int64, LogicalType::Node],
                output_column: 1,
                bound_variable_column: None,
            },
        )
        .with_transaction_context(EpochId::INITIAL, Some(tx));

        while merge.next().unwrap().is_some() {}

        // All three rows had val = 1, so MERGE must observe the node it
        // created on iteration 1 in iterations 2 and 3 and skip the create.
        let nodes = store.nodes_by_label("Item");
        let visible: Vec<_> = nodes
            .iter()
            .filter_map(|&id| store.get_node_versioned(id, EpochId::INITIAL, tx))
            .collect();
        assert_eq!(
            visible.len(),
            1,
            "MERGE inside UNWIND must dedupe nodes its own transaction created in earlier rows"
        );
    }

    #[test]
    fn test_merge_into_any() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let op = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: vec![],
                on_create_properties: vec![],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<MergeOperator>().is_ok());
    }
}
