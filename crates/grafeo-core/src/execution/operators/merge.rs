//! Merge operator for MERGE clause execution.
//!
//! The MERGE operator implements the Cypher MERGE semantics:
//! 1. Try to match the pattern in the graph
//! 2. If found, return existing element (optionally apply ON MATCH SET)
//! 3. If not found, create the element (optionally apply ON CREATE SET)

use super::{
    ExpressionPredicate, GraphWriter, Operator, OperatorError, OperatorResult, PropertySource,
    SessionContext,
};
use crate::execution::chunk::{DataChunk, DataChunkBuilder, copied_column_types};
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EdgeId, LogicalType, NodeId, PropertyKey, Value};
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
    /// Labels a created node gets besides `labels` (`ON CREATE SET n:Label`).
    pub on_create_labels: Vec<String>,
    /// Labels added to each matched node (`ON MATCH SET n:Label`).
    pub on_match_labels: Vec<String>,
    /// Output schema (input columns + node column).
    pub output_schema: Vec<LogicalType>,
    /// Column index where the merged node ID is placed.
    pub output_column: usize,
    /// If the merge variable was already bound in the input, this column index
    /// is used to detect NULL references (e.g., from unmatched OPTIONAL MATCH).
    /// `None` for standalone MERGE that introduces a new variable.
    pub bound_variable_column: Option<usize>,
}

/// The column types of output rows for `chunk` (none for a standalone MERGE),
/// as many as the declared ones: its own for the input columns, which are
/// copied (a node or edge stays one, see `ColumnTypes`), then the declared
/// ones, with `entity` for the column of the merged node or edge.
fn output_types(
    chunk: Option<&DataChunk>,
    declared: &[LogicalType],
    entity_column: usize,
    entity: LogicalType,
) -> Vec<LogicalType> {
    let mut input = chunk.map(DataChunk::column_types).unwrap_or_default();
    input.truncate(declared.len());
    let mut types = copied_column_types(&input, declared);
    types.extend(declared.iter().skip(input.len()).cloned());
    if let Some(column_type) = types.get_mut(entity_column) {
        *column_type = entity;
    }
    types
}

/// Merge operator for MERGE clause.
///
/// Tries to match a node with the given labels and properties.
/// If found, returns the existing node. If not found, creates a new node.
///
/// When an input operator is provided (chained MERGE), input rows are
/// passed through with the merged node ID appended as an additional column.
pub struct MergeOperator {
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Optional input operator (for chained MERGE patterns).
    input: Option<Box<dyn Operator>>,
    /// Merge configuration.
    config: MergeConfig,
    /// The labels of a created node: the pattern's, then the ON CREATE ones
    /// it does not have yet.
    created_labels: Vec<String>,
    /// Whether we've already executed (standalone mode only).
    executed: bool,
    /// Search-store handle used to evaluate `PropertySource::Expression`
    /// runtime expressions in `ON CREATE` / `ON MATCH SET`. None when no
    /// expression sources are present (the planner skips threading it).
    search_store: Option<Arc<dyn GraphStoreSearch>>,
    /// Compiled computed match properties, built on first use.
    match_expressions: Option<super::mutation::PropertyExpressions>,
    /// Session context for expression evaluation (info, schema, etc.).
    session_context: SessionContext,
}

impl MergeOperator {
    /// Creates a new merge operator.
    pub fn new(
        writer: impl Into<GraphWriter>,
        input: Option<Box<dyn Operator>>,
        config: MergeConfig,
    ) -> Self {
        let mut created_labels = config.labels.clone();
        for label in &config.on_create_labels {
            if !created_labels.contains(label) {
                created_labels.push(label.clone());
            }
        }
        Self {
            writer: writer.into(),
            input,
            config,
            created_labels,
            executed: false,
            search_store: None,
            match_expressions: None,
            session_context: SessionContext::default(),
        }
    }

    /// Returns the variable name for the merged node.
    #[must_use]
    pub fn variable(&self) -> &str {
        &self.config.variable
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
    /// [`Self::resolve_action_properties`]. A property of a node or edge is
    /// read as `writer`'s transaction sees it.
    fn resolve_properties(
        props: &[(String, PropertySource)],
        chunk: Option<&DataChunk>,
        row: usize,
        writer: &GraphWriter,
    ) -> Vec<(String, Value)> {
        props
            .iter()
            .map(|(name, source)| {
                let value = if let Some(chunk) = chunk {
                    source.resolve(chunk, row, writer)
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
        let types = output_types(
            chunk,
            &self.config.output_schema,
            self.config.output_column,
            LogicalType::Node,
        );
        let mut builder = DataChunkBuilder::with_capacity(&types, 1);
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
            return Ok(Self::resolve_properties(props, chunk, row, &self.writer));
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
                    if let Some(epoch) = self.writer.viewing_epoch() {
                        predicate =
                            predicate.with_transaction_context(epoch, self.writer.transaction_id());
                    }
                    predicate.eval_at(&augmented, 0).unwrap_or(Value::Null)
                }
                _ => source.resolve(&augmented, 0, &self.writer),
            };
            out.push((name.clone(), value));
        }
        Ok(out)
    }

    /// The label whose nodes MERGE reads when no property index applies: the
    /// pattern's label with the fewest nodes (`nodes_by_label_count`), and of
    /// labels with as many nodes the first written. Each node read is checked
    /// for all the labels, so the matches are the same whichever label is
    /// read, in the same (ID) order.
    fn scan_label(&self) -> Option<&str> {
        match self.config.labels.as_slice() {
            [] => None,
            [label] => Some(label),
            labels => {
                let store = self.writer.store();
                labels
                    .iter()
                    .map(String::as_str)
                    .min_by_key(|label| store.nodes_by_label_count(label))
            }
        }
    }

    /// The nodes that match the given resolved properties (every one, as in
    /// openCypher, where MERGE binds each match).
    fn find_matching_nodes(&self, resolved_match_props: &[(String, Value)]) -> Vec<NodeId> {
        // Use a property index when available to avoid a full label scan.
        // Null conditions are excluded from the index query and verified in the loop.
        let use_index = resolved_match_props
            .iter()
            .any(|(k, v)| !v.is_null() && self.writer.store().has_property_index(k));

        let candidates: Vec<NodeId> = if use_index {
            let conditions: Vec<(&str, Value)> = resolved_match_props
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            self.writer.store().find_nodes_by_properties(&conditions)
        } else if let Some(label) = self.scan_label() {
            self.writer.store().nodes_by_label(label)
        } else {
            self.writer.store().node_ids()
        };

        let mut matches = Vec::new();
        for node_id in candidates {
            // Transactional creates write their version at `EpochId::PENDING`,
            // so the unversioned `get_node` (which checks visibility against
            // the current real epoch) hides nodes this same transaction has
            // just created. UNWIND-driven MERGE relies on seeing those rows
            // to dedupe, so route through the versioned read when we have a
            // transaction context attached.
            let node_opt = match (self.writer.viewing_epoch(), self.writer.transaction_id()) {
                (Some(epoch), Some(tid)) => {
                    self.writer.store().get_node_versioned(node_id, epoch, tid)
                }
                _ => self.writer.store().get_node(node_id),
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
                matches.push(node_id);
            }
        }

        matches
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

    /// Finds the matching nodes for a single row and applies ON MATCH to each,
    /// or creates one and applies ON CREATE.
    fn merge_node_for_row(
        &mut self,
        chunk: Option<&DataChunk>,
        row: usize,
    ) -> Result<Vec<NodeId>, super::OperatorError> {
        // Match properties cannot reference the MERGE variable (ISO §15.5),
        // so they resolve against the input chunk directly.
        let resolved_match = resolve_match_properties(
            &self.config.match_properties,
            chunk,
            row,
            MatchContext {
                search_store: self.search_store.as_ref(),
                session_context: &self.session_context,
                writer: &self.writer,
                cache: &mut self.match_expressions,
            },
        )?;

        let matches = self.find_matching_nodes(&resolved_match);
        if !matches.is_empty() {
            for &existing_id in &matches {
                // Resolve ON MATCH SET against an augmented row containing the
                // matched node id, so `coalesce(n.x, 0)` can read the live value.
                let resolved_on_match = self.resolve_action_properties(
                    &self.config.on_match_properties,
                    chunk,
                    row,
                    existing_id,
                )?;
                self.writer
                    .set_node_properties(existing_id, &resolved_on_match, false)?;
                if !self.config.on_match_labels.is_empty() {
                    self.writer
                        .add_labels(existing_id, &self.config.on_match_labels)?;
                }
            }
            Ok(matches)
        } else if Self::has_expression_source(&self.config.on_create_properties) {
            // ON CREATE expressions read the new node, so it is created from
            // the match properties first; the whole property set is checked
            // before the ON CREATE values are written.
            self.writer
                .create_node_with(&self.created_labels, resolved_match, |new_id| {
                    self.resolve_action_properties(
                        &self.config.on_create_properties,
                        chunk,
                        row,
                        new_id,
                    )
                })
                .map(|created| vec![created])
        } else {
            // No runtime expressions: create with all properties at once.
            let resolved_on_create = Self::resolve_properties(
                &self.config.on_create_properties,
                chunk,
                row,
                &self.writer,
            );
            self.writer
                .create_node(
                    &self.created_labels,
                    Self::merge_node_props(&resolved_match, &resolved_on_create),
                )
                .map(|created| vec![created])
        }
    }
}

impl Operator for MergeOperator {
    fn next(&mut self) -> OperatorResult {
        // When we have an input operator, pass through input rows with the
        // merged node ID appended (used for chained inline MERGE patterns).
        if let Some(ref mut input) = self.input {
            if let Some(chunk) = input.next()? {
                let types = output_types(
                    Some(&chunk),
                    &self.config.output_schema,
                    self.config.output_column,
                    LogicalType::Node,
                );
                // A row comes out once per node it merges (every match).
                let mut merged = Vec::with_capacity(chunk.row_count());
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
                    for node_id in self.merge_node_for_row(Some(&chunk), row)? {
                        merged.push((row, node_id));
                    }
                }

                let mut builder = DataChunkBuilder::with_capacity(&types, merged.len().max(1));
                for (row, node_id) in merged {
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

        let node_ids = self.merge_node_for_row(None, 0)?;

        let types = output_types(
            None,
            &self.config.output_schema,
            self.config.output_column,
            LogicalType::Node,
        );
        let mut builder = DataChunkBuilder::with_capacity(&types, node_ids.len().max(1));
        for node_id in node_ids {
            if let Some(dst) = builder.column_mut(self.config.output_column) {
                dst.push_node_id(node_id);
            }
            builder.advance_row();
        }

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
    /// Whether a relationship from target to source matches too (a pattern
    /// without a direction). One is created from source to target.
    pub undirected: bool,
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
    /// Validated, versioned writes.
    writer: GraphWriter,
    /// Input operator providing rows with source/target node columns.
    input: Box<dyn Operator>,
    /// Merge configuration.
    config: MergeRelationshipConfig,
    /// Search-store handle for evaluating `PropertySource::Expression`.
    search_store: Option<Arc<dyn GraphStoreSearch>>,
    /// Compiled computed match properties, built on first use.
    match_expressions: Option<super::mutation::PropertyExpressions>,
    /// Session context for expression evaluation.
    session_context: SessionContext,
}

impl MergeRelationshipOperator {
    /// Creates a new merge relationship operator.
    pub fn new(
        writer: impl Into<GraphWriter>,
        input: Box<dyn Operator>,
        config: MergeRelationshipConfig,
    ) -> Self {
        Self {
            writer: writer.into(),
            input,
            config,
            search_store: None,
            match_expressions: None,
            session_context: SessionContext::default(),
        }
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
        let types = output_types(
            Some(chunk),
            &self.config.output_schema,
            self.config.edge_output_column,
            LogicalType::Edge,
        );
        let mut builder = DataChunkBuilder::with_capacity(&types, 1);
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
                &self.writer,
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
                    if let Some(epoch) = self.writer.viewing_epoch() {
                        predicate =
                            predicate.with_transaction_context(epoch, self.writer.transaction_id());
                    }
                    predicate.eval_at(&augmented, 0).unwrap_or(Value::Null)
                }
                _ => source.resolve(&augmented, 0, &self.writer),
            };
            out.push((name.clone(), value));
        }
        Ok(out)
    }

    /// The relationships between source and target that match (every one,
    /// as in openCypher, where MERGE binds each match): from source to
    /// target, and the other way round too for a pattern without a direction
    /// (a relationship from a node to itself counts once).
    fn find_matching_edges(
        &self,
        src: NodeId,
        dst: NodeId,
        resolved_match_props: &[(String, Value)],
    ) -> Vec<EdgeId> {
        use crate::graph::Direction;

        let store = self.writer.store();
        let mut candidates: Vec<EdgeId> = store
            .edges_from(src, Direction::Outgoing)
            .into_iter()
            .filter(|&(target, _)| target == dst)
            .map(|(_, edge_id)| edge_id)
            .collect();
        if self.config.undirected {
            for (source, edge_id) in store.edges_from(src, Direction::Incoming) {
                if source == dst && !candidates.contains(&edge_id) {
                    candidates.push(edge_id);
                }
            }
        }

        let mut matches = Vec::new();
        for edge_id in candidates {
            // Same as `find_matching_node`: edges this transaction created
            // earlier in the statement sit at `EpochId::PENDING`, so the
            // unversioned read would hide them and every repeated row would
            // create another edge.
            let edge_opt = match (self.writer.viewing_epoch(), self.writer.transaction_id()) {
                (Some(epoch), Some(tid)) => {
                    self.writer.store().get_edge_versioned(edge_id, epoch, tid)
                }
                _ => self.writer.store().get_edge(edge_id),
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
                    matches.push(edge_id);
                }
            }
        }

        matches
    }
}

impl Operator for MergeRelationshipOperator {
    fn next(&mut self) -> OperatorResult {
        use super::OperatorError;

        if let Some(chunk) = self.input.next()? {
            let types = output_types(
                Some(&chunk),
                &self.config.output_schema,
                self.config.edge_output_column,
                LogicalType::Edge,
            );
            // A row comes out once per relationship it merges (every match).
            let mut merged = Vec::with_capacity(chunk.row_count());
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

                let resolved_match = resolve_match_properties(
                    &self.config.match_properties,
                    Some(&chunk),
                    row,
                    MatchContext {
                        search_store: self.search_store.as_ref(),
                        session_context: &self.session_context,
                        writer: &self.writer,
                        cache: &mut self.match_expressions,
                    },
                )?;

                let matches = self.find_matching_edges(src_val, dst_val, &resolved_match);
                if !matches.is_empty() {
                    for &existing in &matches {
                        let resolved_on_match = self.resolve_action_properties(
                            &self.config.on_match_properties,
                            &chunk,
                            row,
                            existing,
                        )?;
                        self.writer
                            .set_edge_properties(existing, &resolved_on_match, false)?;
                        merged.push((row, existing));
                    }
                    continue;
                }
                let edge_id =
                    if MergeOperator::has_expression_source(&self.config.on_create_properties) {
                        // ON CREATE expressions read the new edge: see MergeOperator.
                        self.writer.create_edge_with(
                            src_val,
                            dst_val,
                            &self.config.edge_type,
                            resolved_match,
                            |new_id| {
                                self.resolve_action_properties(
                                    &self.config.on_create_properties,
                                    &chunk,
                                    row,
                                    new_id,
                                )
                            },
                        )?
                    } else {
                        let resolved_on_create = MergeOperator::resolve_properties(
                            &self.config.on_create_properties,
                            Some(&chunk),
                            row,
                            &self.writer,
                        );
                        self.writer.create_edge(
                            src_val,
                            dst_val,
                            &self.config.edge_type,
                            MergeOperator::merge_node_props(&resolved_match, &resolved_on_create),
                        )?
                    };
                merged.push((row, edge_id));
            }

            let mut builder = DataChunkBuilder::with_capacity(&types, merged.len().max(1));
            for (row, edge_id) in merged {
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
    /// The MERGE's writer: values are read as its transaction sees them.
    writer: &'a GraphWriter,
    /// The operator's compiled evaluators, reused across rows.
    cache: &'a mut Option<super::mutation::PropertyExpressions>,
}

/// Resolves MERGE match properties for one row. Computed values such as
/// `MERGE (:X {id: toString(i)})` are evaluated against the input row; they
/// used to resolve to NULL, so every row matched or created the same node.
fn resolve_match_properties(
    props: &[(String, PropertySource)],
    chunk: Option<&DataChunk>,
    row: usize,
    context: MatchContext<'_>,
) -> Result<Vec<(String, Value)>, OperatorError> {
    if !MergeOperator::has_expression_source(props) {
        return Ok(MergeOperator::resolve_properties(
            props,
            chunk,
            row,
            context.writer,
        ));
    }
    let chunk = chunk.ok_or_else(|| {
        OperatorError::Execution(
            "computed MERGE property without an input row; planner did not provide one".to_string(),
        )
    })?;
    context
        .cache
        .get_or_insert_with(|| {
            super::mutation::PropertyExpressions::new(
                context.search_store.cloned(),
                context.session_context.clone(),
            )
        })
        .resolve_row(props, chunk, row, context.writer)
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::operators::ConstraintValidator;
    use crate::graph::GraphStoreMut;
    use crate::graph::lpg::LpgStore;
    use grafeo_common::types::TransactionId;

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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
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
            GraphWriter::new(Arc::clone(&store)).with_validator(Arc::new(
                RequirePropertyValidator {
                    required_property: "x",
                },
            )),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        )
        .with_search_store(Arc::clone(&search));

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
            GraphWriter::new(Arc::clone(&store))
                .with_validator(Arc::clone(&recorder) as Arc<dyn ConstraintValidator>),
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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        )
        .with_search_store(Arc::clone(&search));

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
            GraphWriter::new(Arc::clone(&store)).with_validator(Arc::new(
                RequirePropertyValidator {
                    required_property: "x",
                },
            )),
            Box::new(OneShot(Some(chunk))),
            MergeRelationshipConfig {
                source_column: 0,
                target_column: 1,
                source_variable: "a".to_string(),
                target_variable: "b".to_string(),
                undirected: false,
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
        .with_search_store(Arc::clone(&search));

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
            GraphWriter::new(Arc::clone(&store))
                .with_transaction_context(EpochId::INITIAL, Some(tx)),
            Some(Box::new(OneShot(Some(chunk)))),
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Item".to_string()],
                match_properties: vec![("val".to_string(), PropertySource::Column(0))],
                on_create_properties: vec![],
                on_match_properties: vec![],
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
                output_schema: vec![LogicalType::Int64, LogicalType::Node],
                output_column: 1,
                bound_variable_column: None,
            },
        );

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
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<MergeOperator>().is_ok());
    }

    /// A pattern with several labels reads the nodes of the label with the
    /// fewest; of labels with as many nodes, the first written (#457).
    #[test]
    fn merge_reads_the_label_with_the_fewest_nodes() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        for i in 0..19 {
            if i < 3 {
                store.create_node(&["Graph", "Repository"]);
            } else {
                store.create_node(&["Graph"]);
            }
        }
        for _ in 0..3 {
            store.create_node(&["Tag"]);
        }
        let merge = |labels: &[&str]| {
            MergeOperator::new(
                Arc::clone(&store),
                None,
                MergeConfig {
                    variable: "n".to_string(),
                    labels: labels.iter().map(ToString::to_string).collect(),
                    match_properties: const_props(vec![("id", Value::from("r1"))]),
                    on_create_properties: vec![],
                    on_match_properties: vec![],
                    on_create_labels: Vec::new(),
                    on_match_labels: Vec::new(),
                    output_schema: vec![LogicalType::Node],
                    output_column: 0,
                    bound_variable_column: None,
                },
            )
        };
        for (labels, expected) in [
            (&["Graph", "Repository"][..], Some("Repository")),
            (&["Repository", "Graph"][..], Some("Repository")),
            (&["Repository", "Tag"][..], Some("Repository")),
            (&["Tag", "Repository"][..], Some("Tag")),
            (&["Graph", "Missing"][..], Some("Missing")),
            (&["Tag"][..], Some("Tag")),
            (&[][..], None),
        ] {
            assert_eq!(merge(labels).scan_label(), expected, "{labels:?}");
        }
    }
}
