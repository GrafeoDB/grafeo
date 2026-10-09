//! Mutation planning (CREATE, DELETE, SET, MERGE, CALL, labels).

use super::{
    AddLabelOp, AddLabelOperator, AntiJoinOp, Arc, CreateEdgeOp, CreateElement, CreateNodeOp,
    CreateOp, CreateOperator, CreateStep, DeleteEdgeOp, DeleteEdgeOperator, DeleteNodeOp,
    DeleteNodeOperator, Direction, EagerOperator, EntityValue, Error, ExpandDirection,
    ExpressionPredicate, HashMap, LeftJoinOp, LogicalExpression, LogicalOperator, LogicalType,
    MergeConfig, MergeOp, MergeOperator, MergeRelationshipConfig, MergeRelationshipOp,
    MergeRelationshipOperator, Operator, ProjectExpr, ProjectOperator, PropertySource,
    RemoveLabelOp, RemoveLabelOperator, Result, SetPropertyOp, SetPropertyOperator, ShortestPathOp,
    ShortestPathOperator, UnaryOp, UnwindOp, UnwindOperator, Value,
};
#[cfg(feature = "algos")]
use super::{CallProcedureOp, StaticResultOperator};
use crate::query::plan::{PathMode, PathSelection};
use grafeo_common::utils::error::{QueryError, QueryErrorKind};
use grafeo_core::execution::operators::{
    ExecutionPathMode, ExecutionPathSelection, JoinCondition, JoinedRowCondition,
};

impl super::Planner {
    /// Plans a CREATE NODE operator (Gremlin `addV`, a GraphQL mutation): a
    /// [`CreateOperator`] of one node, as for a [`CreateOp`].
    pub(super) fn plan_create_node(
        &self,
        create: &CreateNodeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let node = CreateElement::Node {
            variable: create.variable.clone(),
            labels: create.labels.clone(),
            properties: create.properties.clone(),
        };
        self.plan_creation(create.input.as_deref(), std::slice::from_ref(&node))
    }

    /// Plans a CREATE EDGE operator: a [`CreateOperator`] of one edge.
    pub(super) fn plan_create_edge(
        &self,
        create: &CreateEdgeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let edge = CreateElement::Edge {
            variable: create.variable.clone(),
            from_variable: create.from_variable.clone(),
            to_variable: create.to_variable.clone(),
            edge_type: create.edge_type.clone(),
            properties: create.properties.clone(),
        };
        self.plan_creation(Some(&create.input), std::slice::from_ref(&edge))
    }

    /// Plans the [`CreateOp`] of an INSERT or CREATE clause: one
    /// [`CreateOperator`] that creates every node and edge for each row.
    pub(super) fn plan_create(
        &self,
        create: &CreateOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        self.plan_creation(create.input.as_deref(), &create.elements)
    }

    /// Plans a [`CreateOperator`] that creates `elements` for each row of
    /// `input`. The input is planned in this small frame and the operator
    /// built in another: a chain of creations (Gremlin `addV` steps) plans
    /// each one below the next (see `crate::query::limits`).
    fn plan_creation(
        &self,
        input: Option<&LogicalOperator>,
        elements: &[CreateElement],
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let input = match input {
            Some(input) => Some(self.plan_operator(input)?),
            None => None,
        };
        self.build_creation(input, elements)
    }

    /// Builds the [`CreateOperator`] of `elements` over the planned `input`;
    /// a clause that starts the statement (no input) runs once, for a single
    /// row.
    #[inline(never)]
    fn build_creation(
        &self,
        input: Option<(Box<dyn Operator>, Vec<String>)>,
        elements: &[CreateElement],
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let has_input = input.is_some();
        let (input_op, mut columns) = input.unwrap_or_else(|| (single_row_input(), Vec::new()));
        let input_width = columns.len();
        // Where each variable is: its first column, as for a single edge.
        let mut positions: HashMap<String, usize> = HashMap::new();
        for (position, column) in columns.iter().enumerate() {
            positions.entry(column.clone()).or_insert(position);
        }
        let column_of = |positions: &HashMap<String, usize>, variable: &str, role: &str| {
            positions
                .get(variable)
                .copied()
                .ok_or_else(|| Error::Internal(format!("{role} variable '{variable}' not found")))
        };

        let mut steps = Vec::with_capacity(elements.len());
        for element in elements {
            match element {
                CreateElement::Node {
                    variable,
                    labels,
                    properties,
                } => {
                    // A bare variable of the input rows refers to their node
                    // (e.g. from MATCH): nothing to create.
                    if labels.is_empty()
                        && properties.is_empty()
                        && positions.contains_key(variable)
                    {
                        continue;
                    }
                    let properties = self.row_property_sources(properties, &columns)?;
                    positions.entry(variable.clone()).or_insert(columns.len());
                    columns.push(variable.clone());
                    steps.push(CreateStep::node(labels.clone(), properties));
                }
                CreateElement::Edge {
                    variable,
                    from_variable,
                    to_variable,
                    edge_type,
                    properties,
                } => {
                    let from_column = column_of(&positions, from_variable, "Source")?;
                    let to_column = column_of(&positions, to_variable, "Target")?;
                    let properties = self.row_property_sources(properties, &columns)?;
                    if let Some(variable) = variable {
                        positions.entry(variable.clone()).or_insert(columns.len());
                        columns.push(variable.clone());
                        self.edge_columns.borrow_mut().insert(variable.clone());
                    }
                    steps.push(CreateStep::edge(
                        from_column,
                        to_column,
                        edge_type.clone(),
                        properties,
                        variable.is_some(),
                    ));
                }
            }
        }

        // Nothing to create (only nodes of the input rows): the rows as they
        // are.
        if steps.is_empty() && has_input {
            return Ok((input_op, columns));
        }
        let operator = CreateOperator::new(self.graph_writer()?, input_op, input_width, steps)
            .with_search_store(Arc::clone(&self.store))
            .with_session_context(self.session_context.clone());
        Ok((Box::new(operator), columns))
    }

    /// Plans a DELETE NODE operator.
    ///
    /// If the variable is tracked as an edge (via `edge_columns`), this
    /// automatically delegates to [`DeleteEdgeOperator`] instead.
    pub(super) fn plan_delete_node(
        &self,
        delete: &DeleteNodeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let col_idx = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Preserve input columns so downstream RETURN/aggregate can reference
        // the deleted variable (e.g., DETACH DELETE n RETURN count(n)).
        let output_schema = self.derive_schema_from_columns(&columns);
        let output_columns = columns.clone();

        // Auto-detect edge variables and use the correct operator
        let is_edge = self.edge_columns.borrow().contains(&delete.variable);

        let writer = self.graph_writer()?;
        if is_edge {
            let op = DeleteEdgeOperator::new(writer, input_op, col_idx, output_schema);
            Ok((Box::new(op), output_columns))
        } else {
            let op =
                DeleteNodeOperator::new(writer, input_op, col_idx, output_schema, delete.detach);
            Ok((Box::new(op), output_columns))
        }
    }

    /// Plans a DELETE EDGE operator.
    pub(super) fn plan_delete_edge(
        &self,
        delete: &DeleteEdgeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let edge_column = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Preserve input columns so downstream clauses can reference the
        // deleted variable (same pass-through pattern as delete_node).
        let output_schema = self.derive_schema_from_columns(&columns);
        let output_columns = columns.clone();

        let op =
            DeleteEdgeOperator::new(self.graph_writer()?, input_op, edge_column, output_schema);

        Ok((Box::new(op), output_columns))
    }

    /// Plans a LEFT JOIN operator (for OPTIONAL MATCH).
    pub(super) fn plan_left_join(
        &self,
        left_join: &LeftJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // An OPTIONAL MATCH that comes first matches from one empty row.
        let (left_op, left_columns) = self.plan_input(&left_join.left)?;
        // After a write the right side reads the store as the write left it.
        let left_writes = super::after_write::right_side_runs_after_a_write(&left_join.left);
        let (right_op, right_columns) = self.plan_after_a_write(left_writes, &left_join.right)?;
        let left_types = self.derive_schema_from_columns(&left_columns);
        let right_types = self.derive_schema_from_columns(&right_columns);

        // A condition that reads both sides (the WHERE of an OPTIONAL MATCH on a
        // variable bound before it) decides which pairs are matches, so a left
        // row none of whose pairs pass it keeps nulls. It reads the joined row:
        // the left columns, then the right ones (a name both sides have reads
        // the left one).
        let residual = match &left_join.condition {
            Some(condition) => {
                let filter_expr = self.convert_expression(condition)?;
                let mut variable_columns: HashMap<String, usize> = HashMap::new();
                for (i, name) in left_columns.iter().chain(&right_columns).enumerate() {
                    variable_columns.entry(name.clone()).or_insert(i);
                }
                let predicate = ExpressionPredicate::new(
                    filter_expr,
                    variable_columns,
                    Arc::clone(&self.store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone());
                let condition: Box<dyn JoinCondition> =
                    Box::new(JoinedRowCondition::new(Box::new(predicate)));
                Some(condition)
            }
            None => None,
        };
        // After a write (`... CREATE ... WITH i OPTIONAL MATCH (t {k: 301 - i})`)
        // the left side is read first, so that the right side sees what every
        // left row wrote; the hash join reads its right side first otherwise.
        let (join_op, join_columns, _join_types) = super::common::build_left_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            &left_types,
            &right_types,
            |mut join| {
                if let Some(residual) = residual {
                    join = join.with_residual(residual);
                }
                if left_writes {
                    join = join.with_probe_first();
                }
                join
            },
        );

        Ok((join_op, join_columns))
    }

    /// Plans an ANTI JOIN operator (for WHERE NOT EXISTS patterns).
    pub(super) fn plan_anti_join(
        &self,
        anti_join: &AntiJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&anti_join.left)?;
        let (right_op, right_columns) = self.plan_operator(&anti_join.right)?;
        let schema = self.derive_schema_from_columns(&left_columns);
        Ok(super::common::build_anti_join(
            left_op,
            right_op,
            left_columns,
            &right_columns,
            schema,
        ))
    }

    /// Plans an unwind operator.
    pub(super) fn plan_unwind(
        &self,
        unwind: &UnwindOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // An UNWIND that comes first unwinds a list it computes once, in a
        // column of its own on the one empty row (see `plan_input`).
        let unwinds_a_constant = matches!(&*unwind.input, LogicalOperator::Empty);
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) = if unwinds_a_constant {
            let literal_list = self.convert_expression(&unwind.expression)?;
            let (single_row_op, _) = self.plan_input(&unwind.input)?;
            let project_op: Box<dyn Operator> = Box::new(
                ProjectOperator::with_store(
                    single_row_op,
                    vec![ProjectExpr::Expression {
                        expr: literal_list,
                        variable_columns: HashMap::new(),
                    }],
                    vec![LogicalType::Any],
                    Arc::clone(&self.store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
            (project_op, vec!["__list__".to_string()])
        } else {
            let (op, columns) = self.plan_operator(&unwind.input)?;
            // A list that reads the graph after a write (`SET n.list = ...
            // UNWIND n.list AS x`) reads what the whole write left.
            if super::after_write::unwind_reads(unwind) {
                (read_first_after_a_write(op, &unwind.input), columns)
            } else {
                (op, columns)
            }
        };

        // The list is a column of the input (the one row of a constant list, or
        // a variable), or an expression evaluated per row in a column of its
        // own: a literal, a property, `range(1, n.k)`, `nodes(p)`, ...
        let list_col_idx = if unwinds_a_constant {
            Some(0)
        } else {
            match &unwind.expression {
                LogicalExpression::Variable(var) => input_columns.iter().position(|c| c == var),
                _ => None,
            }
        };

        let (final_input_op, final_input_columns, col_idx) = if let Some(idx) = list_col_idx {
            (input_op, input_columns, idx)
        } else {
            // Wrap input in a ProjectOperator that adds the list as an extra column
            let literal_list = self.convert_expression(&unwind.expression)?;
            let mut proj_exprs: Vec<ProjectExpr> =
                (0..input_columns.len()).map(ProjectExpr::Column).collect();
            let var_cols: HashMap<String, usize> = input_columns
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i))
                .collect();
            proj_exprs.push(ProjectExpr::Expression {
                expr: literal_list,
                variable_columns: var_cols,
            });
            let mut proj_schema = self.derive_schema_from_columns(&input_columns);
            proj_schema.push(LogicalType::Any);
            let project_op: Box<dyn Operator> = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    proj_exprs,
                    proj_schema,
                    Arc::clone(&self.store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
            let list_col = input_columns.len();
            let mut cols = input_columns;
            cols.push("__unwind_list__".to_string());
            (project_op, cols, list_col)
        };

        // Build output columns: all input columns plus the new variable
        let mut columns = final_input_columns.clone();
        columns.push(unwind.variable.clone());

        // The items of a node or edge list (a collected list, `nodes(p)`,
        // `relationships(p)`, a list literal of nodes, ...) are nodes or
        // edges, so a property read takes the right entity, and the items of
        // a list of maps or paths with nodes in them keep them; the items of
        // any other list are values. A list literal whose items differ
        // (`[n, 1]`) gives values: one column holds one kind.
        let item = self
            .entity_value(&unwind.expression)
            .and_then(|list| list.item());

        // Build output schema
        let mut output_schema = self.derive_schema_from_columns(&final_input_columns);
        output_schema.push(
            item.as_ref()
                .map_or(LogicalType::Any, EntityValue::logical_type),
        );
        self.set_column_entity(&unwind.variable, item);

        // Add ORDINALITY column (1-based index) if requested
        let emit_ordinality = unwind.ordinality_var.is_some();
        if let Some(ref ord_var) = unwind.ordinality_var {
            columns.push(ord_var.clone());
            output_schema.push(LogicalType::Int64);
            self.scalar_columns.borrow_mut().insert(ord_var.clone());
        }

        // Add OFFSET column (0-based index) if requested
        let emit_offset = unwind.offset_var.is_some();
        if let Some(ref off_var) = unwind.offset_var {
            columns.push(off_var.clone());
            output_schema.push(LogicalType::Int64);
            self.scalar_columns.borrow_mut().insert(off_var.clone());
        }

        let unwind_op = UnwindOperator::new(
            final_input_op,
            col_idx,
            unwind.variable.clone(),
            output_schema,
            emit_ordinality,
            emit_offset,
        );
        // A variable that holds the list stays in scope, so the rows pass the
        // list on; a list computed for the UNWIND alone (a constant, or an
        // expression in a column of its own) stays out of them.
        let list_is_a_variable = !unwinds_a_constant && list_col_idx.is_some();
        let operator: Box<dyn Operator> = Box::new(if list_is_a_variable {
            unwind_op
        } else {
            unwind_op.without_the_list()
        });

        Ok((operator, columns))
    }

    /// Plans a MERGE operator.
    pub(super) fn plan_merge(&self, merge: &MergeOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // A MERGE that comes first runs once without an input, or on the one
        // empty row when it computes a property (below).
        let starts_the_query = matches!(merge.input.as_ref(), LogicalOperator::Empty);
        let (mut input_op, mut columns) = if starts_the_query {
            (None, Vec::new())
        } else {
            let (op, cols) = self.plan_operator(&merge.input)?;
            (Some(read_first_after_a_write(op, &merge.input)), cols)
        };

        // Match properties cannot reference the MERGE variable (ISO §15.5).
        // Computed values (`toString(i)`) are evaluated per input row.
        let match_properties = self.row_property_sources(&merge.match_properties, &columns)?;
        if starts_the_query {
            if has_computed_source(&match_properties) {
                input_op = Some(self.plan_input(&merge.input)?.0);
            } else {
                // PROFILE walks `Empty` too, which no operator reads here.
                self.record_absorbed_scan_entry("Empty", &merge.input);
            }
        }

        // ON CREATE / ON MATCH expressions are evaluated against an augmented row
        // that includes the merged node. Build the action-scope columns now so
        // `coalesce(n.x, 0)` and similar expressions can resolve `n`.
        let mut action_scope_columns = columns.clone();
        action_scope_columns.push(merge.variable.clone());

        let on_create_properties: Vec<(String, PropertySource)> = merge
            .on_create
            .iter()
            .map(|(name, expr)| {
                let source = self.merge_action_property_source(expr, &action_scope_columns)?;
                Ok::<_, Error>((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        let on_match_properties: Vec<(String, PropertySource)> = merge
            .on_match
            .iter()
            .map(|(name, expr)| {
                let source = self.merge_action_property_source(expr, &action_scope_columns)?;
                Ok::<_, Error>((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        // Detect if the merge variable is already bound from the input.
        // If so, record its column index for NULL-reference checking at runtime.
        let bound_variable_column = columns.iter().position(|c| c == &merge.variable);

        // Column index for the merged node ID in the output
        let output_column = columns.len();
        columns.push(merge.variable.clone());

        // Build output schema: type-aware pass-through for input columns,
        // Node for the newly-added merge variable column.
        let input_cols = &columns[..output_column];
        let mut output_schema = self.derive_schema_from_columns(input_cols);
        output_schema.push(LogicalType::Node);

        let merge_op = MergeOperator::new(
            self.graph_writer()?,
            input_op,
            MergeConfig {
                variable: merge.variable.clone(),
                labels: merge.labels.clone(),
                match_properties,
                on_create_properties,
                on_match_properties,
                on_create_labels: merge.on_create_labels.clone(),
                on_match_labels: merge.on_match_labels.clone(),
                output_schema,
                output_column,
                bound_variable_column,
            },
        )
        .with_search_store(Arc::clone(&self.store))
        .with_session_context(self.session_context.clone());

        let operator: Box<dyn Operator> = Box::new(merge_op);

        Ok((operator, columns))
    }

    /// Plans a MERGE RELATIONSHIP operator.
    pub(super) fn plan_merge_relationship(
        &self,
        merge_rel: &MergeRelationshipOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, mut columns) = self.plan_operator(&merge_rel.input)?;
        let input_op = read_first_after_a_write(input_op, &merge_rel.input);

        // Find source and target node columns
        let source_column = columns
            .iter()
            .position(|c| c == &merge_rel.source_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found for MERGE relationship",
                    merge_rel.source_variable
                ))
            })?;

        let target_column = columns
            .iter()
            .position(|c| c == &merge_rel.target_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Target variable '{}' not found for MERGE relationship",
                    merge_rel.target_variable
                ))
            })?;

        // Convert match properties to PropertySource (supports variables from
        // input); computed values are evaluated per input row.
        let match_properties = self.row_property_sources(&merge_rel.match_properties, &columns)?;

        // ON CREATE / ON MATCH SET on a MERGE relationship may reference the
        // edge variable itself: build an augmented scope that includes it.
        let mut action_scope_columns = columns.clone();
        action_scope_columns.push(merge_rel.variable.clone());

        let on_create_properties: Vec<(String, PropertySource)> = merge_rel
            .on_create
            .iter()
            .map(|(name, expr)| {
                let source = self.merge_action_property_source(expr, &action_scope_columns)?;
                Ok::<_, Error>((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        let on_match_properties: Vec<(String, PropertySource)> = merge_rel
            .on_match
            .iter()
            .map(|(name, expr)| {
                let source = self.merge_action_property_source(expr, &action_scope_columns)?;
                Ok::<_, Error>((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        // Add the edge variable to output columns and track it as an edge
        let edge_output_column = columns.len();
        columns.push(merge_rel.variable.clone());
        self.edge_columns
            .borrow_mut()
            .insert(merge_rel.variable.clone());

        // Build output schema: type-aware pass-through for input columns,
        // Edge for the newly-added merge relationship column.
        let input_cols = &columns[..edge_output_column];
        let mut output_schema = self.derive_schema_from_columns(input_cols);
        output_schema.push(LogicalType::Edge);

        let config = MergeRelationshipConfig {
            source_column,
            target_column,
            source_variable: merge_rel.source_variable.clone(),
            target_variable: merge_rel.target_variable.clone(),
            undirected: merge_rel.undirected,
            edge_type: merge_rel.edge_type.clone(),
            match_properties,
            on_create_properties,
            on_match_properties,
            output_schema,
            edge_output_column,
        };

        let merge_rel_op = MergeRelationshipOperator::new(self.graph_writer()?, input_op, config)
            .with_search_store(Arc::clone(&self.store))
            .with_session_context(self.session_context.clone());

        let operator: Box<dyn Operator> = Box::new(merge_rel_op);

        Ok((operator, columns))
    }

    /// Plans a SHORTEST PATH operator.
    pub(super) fn plan_shortest_path(
        &self,
        sp: &ShortestPathOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator
        let (input_op, mut columns) = self.plan_operator(&sp.input)?;

        // Find source and target node columns
        let source_column = columns
            .iter()
            .position(|c| c == &sp.source_var)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found for shortestPath",
                    sp.source_var
                ))
            })?;

        // A search that binds the target has none in its input
        let target_column = if sp.binds_target {
            None
        } else {
            let column = columns
                .iter()
                .position(|c| c == &sp.target_var)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "Target variable '{}' not found for shortestPath",
                        sp.target_var
                    ))
                })?;
            Some(column)
        };

        // Convert direction
        let direction = match sp.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };
        // ANY k keeps the k shortest paths, which are k paths of the pair
        let selection = match sp.selection {
            PathSelection::Any(count) | PathSelection::Shortest(count) => {
                ExecutionPathSelection::Shortest(count)
            }
            PathSelection::ShortestGroups(count) => ExecutionPathSelection::ShortestGroups(count),
        };
        let path_mode = match sp.path_mode {
            PathMode::Walk => ExecutionPathMode::Walk,
            PathMode::Trail => ExecutionPathMode::Trail,
            PathMode::Simple => ExecutionPathMode::Simple,
            PathMode::Acyclic => ExecutionPathMode::Acyclic,
        };

        // Create the shortest path operator: it walks the edges this query
        // sees, its transaction's uncommitted ones included, as an expand does.
        let operator = match target_column {
            Some(target_column) => ShortestPathOperator::new(
                Arc::clone(&self.store),
                input_op,
                source_column,
                target_column,
                sp.edge_types.clone(),
                direction,
            ),
            None => ShortestPathOperator::from_source(
                Arc::clone(&self.store),
                input_op,
                source_column,
                sp.edge_types.clone(),
                direction,
            ),
        };
        let mut operator = operator
            .with_selection(selection)
            .with_path_mode(path_mode)
            .with_hop_bounds(sp.min_hops, sp.max_hops)
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_read_only(self.read_only)
            .with_memory_budget(self.path_search_budget)
            .with_path_output();

        // The edge condition reads a row of the input columns and the
        // candidate edge after them
        if let Some(condition) = &sp.edge_condition {
            let mut variable_columns: HashMap<String, usize> = columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            variable_columns.insert(condition.variable.clone(), columns.len());
            let predicate = ExpressionPredicate::new(
                self.convert_expression(&condition.predicate)?,
                variable_columns,
                Arc::clone(&self.store),
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            operator = operator.with_edge_condition(Box::new(predicate));
        }

        // The target the search binds comes before the length
        if sp.binds_target {
            columns.push(sp.target_var.clone());
        }

        // Add path length column with the expected naming convention
        // The translator expects _path_length_{alias} format for length(p) calls
        let path_col_name = format!("_path_length_{}", sp.path_alias);
        columns.push(path_col_name.clone());

        // Mark path length as scalar so plan_return uses LogicalType::Any, not Node
        self.scalar_columns.borrow_mut().insert(path_col_name);

        // The edge variable: the list of the path's edges for a quantified
        // edge pattern (a group variable, as a variable-length expand binds
        // it), the path's one edge otherwise
        if let Some(edge_variable) = &sp.edge_variable {
            operator = operator.with_edge_output(sp.quantified);
            let edge_column = self.register_edge_column(&Some(edge_variable.clone()));
            if sp.quantified {
                self.edge_columns.borrow_mut().remove(&edge_column);
                self.entity_list_columns
                    .borrow_mut()
                    .insert(edge_column.clone(), EntityValue::Edges);
                self.group_list_variables
                    .borrow_mut()
                    .insert(edge_column.clone());
            }
            columns.push(edge_column);
        }

        // The path's nodes, edges and the path itself, as a variable-length
        // expand writes them for a named path; the path holds node and edge
        // ids, which RETURN gives as nodes and edges.
        for column in [
            format!("_path_nodes_{}", sp.path_alias),
            format!("_path_edges_{}", sp.path_alias),
        ] {
            self.scalar_columns.borrow_mut().insert(column.clone());
            columns.push(column);
        }
        self.set_column_entity(&sp.path_alias, Some(EntityValue::Nested(LogicalType::Path)));
        columns.push(sp.path_alias.clone());

        Ok((Box::new(operator), columns))
    }

    /// Plans a CALL procedure operator.
    #[cfg(feature = "algos")]
    pub(super) fn plan_call_procedure(
        &self,
        call: &CallProcedureOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::procedures::{self, BuiltinProcedures};

        static PROCEDURES: std::sync::OnceLock<BuiltinProcedures> = std::sync::OnceLock::new();
        let registry = PROCEDURES.get_or_init(BuiltinProcedures::new);

        // Special case: grafeo.procedures() lists all procedures
        let resolved_name = call.name.join(".");
        if resolved_name == "grafeo.procedures" || resolved_name == "procedures" {
            let result = procedures::procedures_result(registry);
            return self.plan_static_result(result, &call.yield_items);
        }

        // Check user-defined procedures first (requires GQL for body re-parsing)
        #[cfg(feature = "gql")]
        if let Some(catalog) = &self.catalog {
            let proc_name = if call.name.len() == 1 {
                &call.name[0]
            } else {
                // For dotted names, try the last segment as procedure name
                call.name.last().expect("name has at least one segment")
            };
            if let Some(proc_def) = catalog.get_procedure(proc_name) {
                return self.plan_user_procedure(call, &proc_def);
            }
        }

        // Look up the procedure
        let procedure = registry.get(&call.name).ok_or_else(|| {
            Error::Internal(format!(
                "Unknown procedure: '{}'. Use CALL grafeo.procedures() to list available procedures.",
                call.name.join(".")
            ))
        })?;

        // Evaluate the arguments, constants all, to the procedure's parameters
        let values = self.procedure_argument_values(
            &resolved_name,
            &call.arguments,
            procedure.parameters(),
        )?;
        let params =
            procedures::evaluate_arguments(&resolved_name, &values, procedure.parameters())?;

        // Canonical column names for this procedure (user-facing names)
        let canonical_columns = procedure.output_columns();

        // Determine output columns from YIELD or procedure defaults
        let yield_columns = call.yield_items.as_ref().map(|items| {
            items
                .iter()
                .map(|item| (item.field_name.clone(), item.alias.clone()))
                .collect::<Vec<_>>()
        });

        let output_columns = if let Some(yield_cols) = &yield_columns {
            yield_cols
                .iter()
                .map(|(name, alias)| alias.clone().unwrap_or_else(|| name.clone()))
                .collect()
        } else {
            canonical_columns.clone()
        };

        let mut op = crate::query::executor::procedure_call::ProcedureCallOperator::new(
            self.procedure_store(&params)?,
            procedure,
            params,
            yield_columns,
            canonical_columns,
        );
        #[cfg(feature = "lpg")]
        if let Some(lpg_store) = self.lpg_store.as_ref() {
            op = op.with_lpg_store(Arc::clone(lpg_store));
        }
        let operator: Box<dyn Operator> = Box::new(op);

        // Procedure outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        Ok((operator, output_columns))
    }

    /// The values of a procedure call's arguments, each a constant (see
    /// [`constant_argument`](Self::constant_argument)). One map argument
    /// keeps its keys: it names the parameters (see
    /// [`evaluate_arguments`](crate::procedures::evaluate_arguments)).
    #[cfg(feature = "algos")]
    fn procedure_argument_values(
        &self,
        procedure: &str,
        arguments: &[LogicalExpression],
        param_defs: &[grafeo_adapters::plugins::ParameterDef],
    ) -> Result<Vec<Value>> {
        if let [LogicalExpression::Map(entries)] = arguments {
            let mut named = std::collections::BTreeMap::new();
            for (key, expression) in entries {
                let value = self.constant_argument(procedure, key, expression)?;
                named.insert(grafeo_common::types::PropertyKey::new(key.as_str()), value);
            }
            return Ok(vec![Value::Map(Arc::new(named))]);
        }
        arguments
            .iter()
            .enumerate()
            .map(|(index, expression)| {
                let name = crate::procedures::argument_name(param_defs, index);
                self.constant_argument(procedure, &name, expression)
            })
            .collect()
    }

    /// The value of `expression`, argument `argument` of `procedure`: a
    /// literal, a parameter (filled in before planning) or an expression of
    /// them, such as `$d / 2` or `[1.0, 0.0 + 0.0]`.
    ///
    /// # Errors
    ///
    /// A procedure runs once, before any row, so an argument that reads a row
    /// (a variable an earlier clause binds) or the graph (a subquery) is an
    /// error, as is a parameter nobody supplied and an expression that cannot
    /// be evaluated: the procedure would otherwise run with the default.
    #[cfg(feature = "algos")]
    fn constant_argument(
        &self,
        procedure: &str,
        argument: &str,
        expression: &LogicalExpression,
    ) -> Result<Value> {
        let semantic =
            |message: String| Error::Query(QueryError::new(QueryErrorKind::Semantic, message));
        let reads = match first_non_constant(expression, &mut Vec::new()) {
            None => None,
            Some(NonConstant::Row(variable)) => Some(format!("the variable '{variable}'")),
            Some(NonConstant::Graph) => Some("the graph (a subquery or a pattern)".to_string()),
            Some(NonConstant::Parameter(name)) => {
                return Err(semantic(format!("Missing parameter: ${name}")));
            }
        };
        if let Some(reads) = reads {
            return Err(semantic(format!(
                "Argument '{argument}' of {procedure} reads {reads}: a procedure argument must \
                 be a constant (a literal, a parameter or an expression of them)"
            )));
        }
        if let LogicalExpression::Literal(value) = expression {
            return Ok(value.clone());
        }
        let evaluator = ExpressionPredicate::new(
            self.convert_expression(expression)?,
            HashMap::new(),
            Arc::clone(&self.store),
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone());
        evaluator
            .eval_at(&grafeo_core::execution::DataChunk::empty(), 0)
            .ok_or_else(|| {
                semantic(format!(
                    "Argument '{argument}' of {procedure} cannot be evaluated"
                ))
            })
    }

    /// The store a procedure reads: the projection its `projection` argument
    /// names (`CALL grafeo.pagerank({projection: 'p'})`), else the planner's
    /// store, the selected graph. Projection names are database-wide.
    #[cfg(feature = "algos")]
    fn procedure_store(
        &self,
        params: &grafeo_adapters::plugins::Parameters,
    ) -> Result<Arc<dyn grafeo_core::graph::GraphStoreSearch>> {
        let Some(name) = params.get_string("projection") else {
            return Ok(Arc::clone(&self.store));
        };
        #[cfg(feature = "lpg")]
        if let Some(projection) = self
            .projections
            .as_ref()
            .and_then(|projections| projections.read().get(name).cloned())
        {
            return Ok(projection as Arc<dyn grafeo_core::graph::GraphStoreSearch>);
        }
        Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            format!("Projection '{name}' does not exist"),
        )))
    }

    /// Plans a static result set (e.g., from `grafeo.procedures()`).
    #[cfg(feature = "algos")]
    pub(super) fn plan_static_result(
        &self,
        result: grafeo_adapters::plugins::AlgorithmResult,
        yield_items: &Option<Vec<crate::query::plan::ProcedureYield>>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Determine output columns and column indices
        let (output_columns, column_indices) = if let Some(items) = yield_items {
            let mut cols = Vec::new();
            let mut indices = Vec::new();
            for item in items {
                let idx = result
                    .columns
                    .iter()
                    .position(|c| c == &item.field_name)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "YIELD column '{}' not found (available: {})",
                            item.field_name,
                            result.columns.join(", ")
                        ))
                    })?;
                indices.push(idx);
                cols.push(
                    item.alias
                        .clone()
                        .unwrap_or_else(|| item.field_name.clone()),
                );
            }
            (cols, indices)
        } else {
            let indices: Vec<usize> = (0..result.columns.len()).collect();
            (result.columns.clone(), indices)
        };

        let operator = Box::new(StaticResultOperator {
            rows: result.rows,
            column_indices,
            row_index: 0,
        });

        // Static result outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        Ok((operator, output_columns))
    }

    /// Plans a user-defined procedure call.
    #[cfg(all(feature = "algos", feature = "gql"))]
    fn plan_user_procedure(
        &self,
        call: &CallProcedureOp,
        proc_def: &crate::catalog::ProcedureDefinition,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::executor::user_procedure::{ProcedureContext, UserProcedureOperator};

        // Validate argument count
        if call.arguments.len() != proc_def.params.len() {
            return Err(Error::Internal(format!(
                "Procedure '{}' expects {} arguments, got {}",
                proc_def.name,
                proc_def.params.len(),
                call.arguments.len()
            )));
        }

        // Build parameter map: param_name -> value, each argument a constant
        let mut param_map = std::collections::HashMap::new();
        for (param, argument) in proc_def.params.iter().zip(&call.arguments) {
            let value = self.constant_argument(&proc_def.name, &param.0, argument)?;
            param_map.insert(param.0.clone(), value);
        }

        // Determine output columns
        let return_columns: Vec<String> = proc_def.returns.iter().map(|r| r.0.clone()).collect();

        let output_columns = if let Some(yield_items) = &call.yield_items {
            yield_items
                .iter()
                .map(|item| {
                    item.alias
                        .clone()
                        .unwrap_or_else(|| item.field_name.clone())
                })
                .collect()
        } else {
            return_columns.clone()
        };

        let yield_columns = call.yield_items.as_ref().map(|items| {
            items
                .iter()
                .map(|item| item.field_name.clone())
                .collect::<Vec<_>>()
        });

        let operator = Box::new(UserProcedureOperator::new(
            proc_def.body.clone(),
            param_map,
            return_columns,
            yield_columns,
            ProcedureContext {
                store: Arc::clone(&self.store),
                store_mut: self.write_store.as_ref().map(Arc::clone),
                transaction_manager: self.transaction_manager.clone(),
                transaction_id: self.transaction_id,
                viewing_epoch: self.viewing_epoch,
                catalog: self.catalog.clone(),
                write_counter: self.write_counter(),
                #[cfg(feature = "lpg")]
                projections: self.projections.clone(),
            },
        ));

        // Procedure outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        Ok((operator, output_columns))
    }

    /// Plans an ADD LABEL operator.
    pub(super) fn plan_add_label(
        &self,
        add_label: &AddLabelOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&add_label.input)?;

        // Find the node column
        let node_column = columns
            .iter()
            .position(|c| c == &add_label.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for ADD LABEL",
                    add_label.variable
                ))
            })?;

        // Preserve input columns (like SetPropertyOperator) and append update count
        let mut output_schema = self.derive_schema_from_columns(&columns);
        output_schema.push(LogicalType::Int64);
        let mut output_columns = columns.clone();
        output_columns.push("labels_added".to_string());

        let op = AddLabelOperator::new(
            self.graph_writer()?,
            input_op,
            node_column,
            add_label.labels.clone(),
            output_schema,
        );

        Ok((Box::new(op), output_columns))
    }

    /// Plans a REMOVE LABEL operator.
    pub(super) fn plan_remove_label(
        &self,
        remove_label: &RemoveLabelOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&remove_label.input)?;

        // Find the node column
        let node_column = columns
            .iter()
            .position(|c| c == &remove_label.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for REMOVE LABEL",
                    remove_label.variable
                ))
            })?;

        // Preserve input columns (like SetPropertyOperator) and append update count
        let mut output_schema = self.derive_schema_from_columns(&columns);
        output_schema.push(LogicalType::Int64);
        let mut output_columns = columns.clone();
        output_columns.push("labels_removed".to_string());

        let op = RemoveLabelOperator::new(
            self.graph_writer()?,
            input_op,
            node_column,
            remove_label.labels.clone(),
            output_schema,
        );

        Ok((Box::new(op), output_columns))
    }

    /// Plans a SET PROPERTY operator.
    pub(super) fn plan_set_property(
        &self,
        set_prop: &SetPropertyOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&set_prop.input)?;

        // Find the entity column (node or edge variable)
        let entity_column = columns
            .iter()
            .position(|c| c == &set_prop.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for SET",
                    set_prop.variable
                ))
            })?;

        // Convert properties to PropertySource (supports constants, variables, and
        // complex expressions like `c.value + 1`). Expressions that cannot be resolved
        // to a simple PropertySource are pre-computed via a projection operator.
        let mut properties: Vec<(String, PropertySource)> = Vec::new();
        let mut projection_exprs: Vec<ProjectExpr> = Vec::new();
        let mut projection_columns: Vec<String> = columns.clone();

        // Start with pass-through for all existing columns
        for i in 0..columns.len() {
            projection_exprs.push(ProjectExpr::Column(i));
        }

        let mut needs_projection = false;

        for (name, expr) in &set_prop.properties {
            let source = match self.expression_to_property_source(expr, &columns) {
                Ok(s) => s,
                Err(_) => {
                    // Fallback: try constant folding for complex expressions
                    // (e.g., vector([1,2,3]), date('2024-01-01')).
                    if let Some(v) = Self::try_fold_expression(expr) {
                        PropertySource::Constant(v)
                    } else {
                        // Complex runtime expression (e.g., c.value + 1): add a
                        // projection column that evaluates it, then SET from that column.
                        match self.convert_expression(expr) {
                            Ok(filter_expr) => {
                                let col_idx = projection_columns.len();
                                let col_name = format!("__set_expr_{name}");
                                let variable_columns: HashMap<String, usize> = columns
                                    .iter()
                                    .enumerate()
                                    .map(|(i, c)| (c.clone(), i))
                                    .collect();
                                projection_exprs.push(ProjectExpr::Expression {
                                    expr: filter_expr,
                                    variable_columns,
                                });
                                projection_columns.push(col_name);
                                needs_projection = true;
                                PropertySource::Column(col_idx)
                            }
                            Err(_) => {
                                return Err(Error::Internal(format!(
                                    "Cannot resolve SET expression for property '{name}': \
                                     variable not in scope or unsupported expression"
                                )));
                            }
                        }
                    }
                }
            };
            properties.push((name.clone(), source));
        }

        // If any SET expression needed runtime evaluation, wrap input in a projection.
        let actual_input: Box<dyn Operator> = if needs_projection {
            let proj_schema = self.derive_schema_from_columns(&projection_columns);
            Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projection_exprs,
                    proj_schema,
                    Arc::clone(&self.store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            )
        } else {
            input_op
        };

        // Output schema: type-aware pass-through for input columns.
        let output_schema = self.derive_schema_from_columns(&columns);
        let output_columns = columns.clone();

        // Determine if this is a node or edge using tracked edge columns
        let is_edge = set_prop.is_edge || self.edge_columns.borrow().contains(&set_prop.variable);
        let operator: Box<dyn Operator> = if is_edge {
            Box::new(
                SetPropertyOperator::new_for_edge(
                    self.graph_writer()?,
                    actual_input,
                    entity_column,
                    properties,
                    output_schema,
                )
                .with_replace(set_prop.replace),
            )
        } else {
            Box::new(
                SetPropertyOperator::new_for_node(
                    self.graph_writer()?,
                    actual_input,
                    entity_column,
                    properties,
                    output_schema,
                )
                .with_replace(set_prop.replace),
            )
        };

        Ok((operator, output_columns))
    }

    /// Lowers an ON CREATE / ON MATCH SET expression for a MERGE clause.
    ///
    /// Resolution order:
    /// 1. Simple lowering against the augmented action scope (input columns +
    ///    the MERGE variable). Catches literals, plain variable refs, and
    ///    direct property access.
    /// 2. Constant folding for plan-time-evaluable expressions like
    ///    `vector([1,2,3])` or `date('2024-01-01')`.
    /// 3. Fall back to a runtime [`PropertySource::Expression`] carrying the
    ///    converted [`FilterExpression`] and the variable-column map. The
    ///    operator builds an augmented row containing the merged node/edge
    ///    and evaluates the expression via [`ExpressionPredicate`].
    pub(super) fn merge_action_property_source(
        &self,
        expr: &LogicalExpression,
        action_scope_columns: &[String],
    ) -> Result<PropertySource> {
        if let Ok(source) = self.expression_to_property_source(expr, action_scope_columns) {
            return Ok(source);
        }
        if let Some(value) = Self::try_fold_expression(expr) {
            return Ok(PropertySource::Constant(value));
        }
        let filter_expr = self.convert_expression(expr)?;
        // When the merge variable is already bound from input, it appears
        // twice in `action_scope_columns`. Collecting via `HashMap::insert`
        // keeps the LAST occurrence, which is the appended column matching
        // the operator's augmented row position. Do not switch to a
        // first-wins collector here.
        let variable_columns: HashMap<String, usize> = action_scope_columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        Ok(PropertySource::Expression {
            expr: Box::new(filter_expr),
            variable_columns,
        })
    }

    /// Lowers the property map of a CREATE, INSERT or MERGE pattern against
    /// the input `columns`.
    ///
    /// Tries a direct source (literal, variable, property access), then
    /// constant folding, then a runtime [`PropertySource::Expression`] that
    /// the operator evaluates per input row. Computed values such as
    /// `toString(i)` used to fail with an internal error in CREATE, and
    /// silently became NULL in MERGE.
    pub(super) fn row_property_sources(
        &self,
        properties: &[(String, LogicalExpression)],
        columns: &[String],
    ) -> Result<Vec<(String, PropertySource)>> {
        properties
            .iter()
            .map(|(name, expr)| {
                let source = self.row_property_source(expr, columns).map_err(|e| {
                    Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Cannot evaluate the value of property '{name}': {e}"),
                    ))
                })?;
                Ok((name.clone(), source))
            })
            .collect()
    }

    fn row_property_source(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Result<PropertySource> {
        if let Ok(source) = self.expression_to_property_source(expr, columns) {
            return Ok(source);
        }
        if let Some(value) = Self::try_fold_expression(expr) {
            return Ok(PropertySource::Constant(value));
        }
        let filter_expr = self.convert_expression(expr)?;
        // Last occurrence wins, matching `expression_to_property_source`.
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        Ok(PropertySource::Expression {
            expr: Box::new(filter_expr),
            variable_columns,
        })
    }

    /// Converts a logical expression to a PropertySource.
    ///
    /// Variable resolution uses `rposition` rather than `position` so that
    /// when `columns` legitimately contains a duplicated variable name,
    /// references resolve to the most recently added (innermost / latest)
    /// column. The MERGE planner relies on this for ON CREATE / ON MATCH
    /// expressions whose action scope appends the merge variable on top of
    /// an input that may already bind it: the appended column is the one
    /// the operator's augmented row populates with the merged node/edge id,
    /// and resolving to the earlier bound copy would read the pre-merge
    /// (potentially stale) value instead. For unique columns the two are
    /// equivalent.
    pub(super) fn expression_to_property_source(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Result<PropertySource> {
        match expr {
            LogicalExpression::Literal(value) => Ok(PropertySource::Constant(value.clone())),
            LogicalExpression::Variable(name) => {
                let col_idx = columns.iter().rposition(|c| c == name).ok_or_else(|| {
                    Error::Internal(format!("Variable '{}' not found for property source", name))
                })?;
                Ok(PropertySource::Column(col_idx))
            }
            LogicalExpression::Property { variable, property } => {
                let col_idx = columns.iter().rposition(|c| c == variable).ok_or_else(|| {
                    Error::Internal(format!(
                        "Variable '{}' not found for property access '{}.{}'",
                        variable, variable, property
                    ))
                })?;
                Ok(PropertySource::PropertyAccess {
                    column: col_idx,
                    property: property.clone(),
                })
            }
            LogicalExpression::Parameter(name) => {
                // Parameters should be resolved before planning
                // For now, treat as a placeholder
                Ok(PropertySource::Constant(
                    grafeo_common::types::Value::String(format!("${}", name).into()),
                ))
            }
            _ => {
                if let Some(value) = Self::try_fold_expression(expr) {
                    Ok(PropertySource::Constant(value))
                } else {
                    Err(Error::Internal(format!(
                        "Unsupported expression type for property source: {:?}",
                        expr
                    )))
                }
            }
        }
    }

    /// Tries to evaluate a constant expression at plan time.
    ///
    /// Recursively folds literals, unary operators, lists, and known function calls
    /// (like `vector()`) into concrete values. Returns `None` if the expression
    /// contains non-constant parts (variables, property accesses, etc.).
    pub(super) fn try_fold_expression(expr: &LogicalExpression) -> Option<Value> {
        match expr {
            LogicalExpression::Literal(v) => Some(v.clone()),
            LogicalExpression::List(items) => {
                let values: Option<Vec<Value>> =
                    items.iter().map(Self::try_fold_expression).collect();
                Some(Value::List(values?.into()))
            }
            LogicalExpression::FunctionCall { name, args, .. } => {
                match name.to_lowercase().as_str() {
                    "vector" => {
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::List(items) => {
                                // reason: intentional lossy f64/i64 to f32 for vector elements
                                #[allow(clippy::cast_possible_truncation)]
                                let floats: Vec<f32> = items
                                    .iter()
                                    .filter_map(|v| match v {
                                        Value::Float64(f) => Some(*f as f32),
                                        Value::Int64(i) => Some(*i as f32),
                                        _ => None,
                                    })
                                    .collect();
                                if floats.len() == items.len() {
                                    Some(Value::Vector(floats.into()))
                                } else {
                                    None
                                }
                            }
                            // Already a vector (from all-numeric list folding)
                            Value::Vector(v) => Some(Value::Vector(v)),
                            _ => None,
                        }
                    }
                    "timestamp" => {
                        if !args.is_empty() {
                            return None;
                        }
                        Some(Value::Int64(
                            grafeo_common::types::Timestamp::now().as_millis(),
                        ))
                    }
                    "now" | "current_timestamp" | "currenttimestamp" => {
                        if !args.is_empty() {
                            return None;
                        }
                        Some(Value::Timestamp(grafeo_common::types::Timestamp::now()))
                    }
                    "date" | "todate" | "current_date" | "currentdate" => {
                        if args.is_empty() {
                            return Some(Value::Date(grafeo_common::types::Date::today()));
                        }
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                grafeo_common::types::Date::parse(&s).map(Value::Date)
                            }
                            _ => None,
                        }
                    }
                    "time" | "totime" | "local_time" | "current_time" | "currenttime" => {
                        if args.is_empty() {
                            return Some(Value::Time(grafeo_common::types::Time::now()));
                        }
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                grafeo_common::types::Time::parse(&s).map(Value::Time)
                            }
                            _ => None,
                        }
                    }
                    "datetime" | "localdatetime" | "local_datetime" | "todatetime" => {
                        if args.is_empty() {
                            return Some(Value::Timestamp(grafeo_common::types::Timestamp::now()));
                        }
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                if let Some(d) = grafeo_common::types::Date::parse(&s) {
                                    return Some(Value::Timestamp(d.to_timestamp()));
                                }
                                if let Some(pos) = s.find('T') {
                                    let (date_part, time_part) = (&s[..pos], &s[pos + 1..]);
                                    if let (Some(d), Some(t)) = (
                                        grafeo_common::types::Date::parse(date_part),
                                        grafeo_common::types::Time::parse(time_part),
                                    ) {
                                        return Some(Value::Timestamp(
                                            grafeo_common::types::Timestamp::from_date_time(d, t),
                                        ));
                                    }
                                }
                                None
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            LogicalExpression::Map(entries) => {
                let folded: Option<Vec<(String, Value)>> = entries
                    .iter()
                    .map(|(k, v)| Self::try_fold_expression(v).map(|val| (k.clone(), val)))
                    .collect();
                let folded = folded?;
                let map: std::collections::BTreeMap<grafeo_common::types::PropertyKey, Value> =
                    folded
                        .into_iter()
                        .map(|(k, v)| (grafeo_common::types::PropertyKey::from(k), v))
                        .collect();
                Some(Value::Map(std::sync::Arc::new(map)))
            }
            LogicalExpression::Unary { op, operand } => {
                let value = Self::try_fold_expression(operand)?;
                match op {
                    UnaryOp::Neg => match value {
                        Value::Int64(n) => Some(Value::Int64(-n)),
                        Value::Float64(f) => Some(Value::Float64(-f)),
                        _ => None,
                    },
                    UnaryOp::Not => match value {
                        Value::Bool(b) => Some(Value::Bool(!b)),
                        _ => None,
                    },
                    UnaryOp::IsNull | UnaryOp::IsNotNull => None,
                }
            }
            _ => None,
        }
    }
}

/// Whether any property value must be computed per input row.
fn has_computed_source(properties: &[(String, PropertySource)]) -> bool {
    properties
        .iter()
        .any(|(_, source)| !matches!(source, PropertySource::Constant(_)))
}

/// A one-row input, so a CREATE or MERGE without a MATCH still has a row to
/// evaluate computed property values against.
fn single_row_input() -> Box<dyn Operator> {
    Box::new(grafeo_core::execution::operators::single_row::SingleRowOperator::new())
}

/// The planned input `op` of a clause that reads after a write (see
/// [`after_write`](super::after_write)), read whole before the first row
/// comes out while its plan `input` still writes as its rows come out: a
/// MERGE after a write (`UNWIND ... CREATE (:P {k: i}) WITH i MERGE (:P {k:
/// 301 - i})`) then finds what every row of the earlier clauses wrote, not
/// only what the rows before its own did, as a MATCH there does (see
/// `plan_node_scan`), and a RETURN after a SET reads what the whole SET left.
/// An input read whole already (by a MATCH after the write) is not read again.
/// The MERGE itself still runs row by row, so a row sees what the MERGE
/// created for the rows before it.
pub(super) fn read_first_after_a_write(
    op: Box<dyn Operator>,
    input: &LogicalOperator,
) -> Box<dyn Operator> {
    if super::after_write::writes_pending(input) {
        Box::new(EagerOperator::new(op))
    } else {
        op
    }
}

/// What keeps an expression from being a constant.
#[cfg(feature = "algos")]
enum NonConstant {
    /// It reads this variable of the row.
    Row(String),
    /// It reads the graph: a subquery or a pattern.
    Graph,
    /// It reads this parameter, which nobody supplied.
    Parameter(String),
}

/// The first part of `expression` that keeps it from being a constant, if
/// any. The variables a list comprehension, a list predicate or a reduce
/// binds (`local`) are not row variables: `[x IN [1, 2] | x * 2]` is a
/// constant.
#[cfg(feature = "algos")]
fn first_non_constant(
    expression: &LogicalExpression,
    local: &mut Vec<String>,
) -> Option<NonConstant> {
    let row = |name: &String, local: &[String]| {
        (!local.contains(name)).then(|| NonConstant::Row(name.clone()))
    };
    match expression {
        LogicalExpression::Literal(_) => None,
        LogicalExpression::Parameter(name) => Some(NonConstant::Parameter(name.clone())),
        LogicalExpression::Variable(name)
        | LogicalExpression::Labels(name)
        | LogicalExpression::Type(name)
        | LogicalExpression::Id(name)
        | LogicalExpression::Property { variable: name, .. } => row(name, local),
        LogicalExpression::MapProjection { base, entries } => row(base, local).or_else(|| {
            entries.iter().find_map(|entry| match entry {
                crate::query::plan::MapProjectionEntry::LiteralEntry(_, value) => {
                    first_non_constant(value, local)
                }
                _ => None,
            })
        }),
        LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_)
        | LogicalExpression::PatternComprehension { .. } => Some(NonConstant::Graph),
        LogicalExpression::Binary { left, right, .. } => {
            first_non_constant(left, local).or_else(|| first_non_constant(right, local))
        }
        LogicalExpression::Unary { operand, .. } => first_non_constant(operand, local),
        LogicalExpression::FunctionCall { args: items, .. } | LogicalExpression::List(items) => {
            items
                .iter()
                .find_map(|item| first_non_constant(item, local))
        }
        LogicalExpression::Map(entries) => entries
            .iter()
            .find_map(|(_, value)| first_non_constant(value, local)),
        LogicalExpression::IndexAccess { base, index } => {
            first_non_constant(base, local).or_else(|| first_non_constant(index, local))
        }
        LogicalExpression::MapAccess { base, .. } => first_non_constant(base, local),
        LogicalExpression::SliceAccess { base, start, end } => first_non_constant(base, local)
            .or_else(|| {
                [start, end]
                    .into_iter()
                    .flatten()
                    .find_map(|bound| first_non_constant(bound, local))
            }),
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => operand
            .iter()
            .chain(else_clause)
            .find_map(|part| first_non_constant(part, local))
            .or_else(|| {
                when_clauses.iter().find_map(|(condition, result)| {
                    first_non_constant(condition, local)
                        .or_else(|| first_non_constant(result, local))
                })
            }),
        LogicalExpression::ListComprehension {
            variable,
            list_expr,
            filter_expr,
            map_expr,
        } => first_non_constant(list_expr, local).or_else(|| {
            local.push(variable.clone());
            let found = filter_expr
                .iter()
                .chain(std::iter::once(map_expr))
                .find_map(|part| first_non_constant(part, local));
            local.pop();
            found
        }),
        LogicalExpression::ListPredicate {
            variable,
            list_expr,
            predicate,
            ..
        } => first_non_constant(list_expr, local).or_else(|| {
            local.push(variable.clone());
            let found = first_non_constant(predicate, local);
            local.pop();
            found
        }),
        LogicalExpression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => first_non_constant(initial, local)
            .or_else(|| first_non_constant(list, local))
            .or_else(|| {
                local.push(accumulator.clone());
                local.push(variable.clone());
                let found = first_non_constant(expression, local);
                local.truncate(local.len() - 2);
                found
            }),
    }
}
