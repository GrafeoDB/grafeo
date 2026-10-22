//! Semantic validation - catching errors before execution.
//!
//! The binder walks the logical plan and validates that everything makes sense:
//! - Is that variable actually defined? (You can't use `RETURN x` if `x` wasn't matched)
//! - Does that property access make sense? (Accessing `.age` on an integer fails)
//! - Are types compatible? (Can't compare a string to an integer)
//!
//! Better to catch these errors early than waste time executing a broken query.

use crate::query::plan::{
    CreateElement, ExpandOp, FilterOp, LogicalExpression, LogicalOperator, LogicalPlan, NodeScanOp,
    ReturnItem, ReturnOp, TripleScanOp,
};
use grafeo_common::types::LogicalType;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_common::utils::strings::{find_similar, format_suggestion};
use indexmap::IndexMap;
use std::collections::HashSet;

/// Creates a semantic binding error.
fn binding_error(message: impl Into<String>) -> Error {
    Error::Query(QueryError::new(QueryErrorKind::Semantic, message))
}

/// Creates a semantic binding error with a hint.
fn binding_error_with_hint(message: impl Into<String>, hint: impl Into<String>) -> Error {
    Error::Query(QueryError::new(QueryErrorKind::Semantic, message).with_hint(hint))
}

/// What a pattern binds a variable to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Element {
    Node,
    Edge,
}

/// Whether a variable holds a value of a known type that is not a node or an
/// edge (`WITH 1 AS r`). A variable of unknown type may hold either.
fn holds_a_value(info: &VariableInfo) -> bool {
    !info.is_node
        && !info.is_edge
        && !matches!(
            info.data_type,
            LogicalType::Any
                | LogicalType::Null
                | LogicalType::Node
                | LogicalType::Edge
                | LogicalType::List(_)
                | LogicalType::Path
        )
}

/// Creates an "undefined variable" error with a suggestion if a similar variable exists.
fn undefined_variable_error(variable: &str, context: &BindingContext, suffix: &str) -> Error {
    let candidates: Vec<String> = context.variable_names();
    let candidates_ref: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();

    if let Some(suggestion) = find_similar(variable, &candidates_ref) {
        binding_error_with_hint(
            format!("Undefined variable '{variable}'{suffix}"),
            format_suggestion(suggestion),
        )
    } else {
        binding_error(format!("Undefined variable '{variable}'{suffix}"))
    }
}

/// The parts of a subquery body joined by `UNION` (under the `DISTINCT` of
/// a plain `UNION`), or `None` for a body of one part.
fn union_parts(plan: &LogicalOperator) -> Option<&[LogicalOperator]> {
    match plan {
        LogicalOperator::Union(union) => Some(&union.inputs),
        LogicalOperator::Distinct(distinct) => match distinct.input.as_ref() {
            LogicalOperator::Union(union) => Some(&union.inputs),
            _ => None,
        },
        _ => None,
    }
}

/// The outer variables one part of a subquery imports: the columns of the
/// parameter scan its plan starts from, none without one.
fn leading_imports(plan: &LogicalOperator) -> Vec<String> {
    match plan {
        LogicalOperator::ParameterScan(scan) => scan.columns.clone(),
        LogicalOperator::NodeScan(scan) => {
            scan.input.as_deref().map_or_else(Vec::new, leading_imports)
        }
        LogicalOperator::EdgeScan(scan) => {
            scan.input.as_deref().map_or_else(Vec::new, leading_imports)
        }
        LogicalOperator::Expand(op) => leading_imports(&op.input),
        LogicalOperator::Filter(op) => leading_imports(&op.input),
        LogicalOperator::Project(op) => leading_imports(&op.input),
        LogicalOperator::Return(op) => leading_imports(&op.input),
        LogicalOperator::Aggregate(op) => leading_imports(&op.input),
        LogicalOperator::Limit(op) => leading_imports(&op.input),
        LogicalOperator::Skip(op) => leading_imports(&op.input),
        LogicalOperator::Sort(op) => leading_imports(&op.input),
        LogicalOperator::Distinct(op) => leading_imports(&op.input),
        LogicalOperator::Unwind(op) => leading_imports(&op.input),
        LogicalOperator::Bind(op) => leading_imports(&op.input),
        LogicalOperator::Join(join) => leading_imports(&join.left),
        LogicalOperator::LeftJoin(join) => leading_imports(&join.left),
        LogicalOperator::Apply(apply) => leading_imports(&apply.input),
        _ => Vec::new(),
    }
}

/// Information about a bound variable.
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// The name of the variable.
    pub name: String,
    /// The inferred type of the variable.
    pub data_type: LogicalType,
    /// Whether this variable is a node.
    pub is_node: bool,
    /// Whether this variable is an edge.
    pub is_edge: bool,
}

/// Context containing all bound variables and their information.
///
/// Uses `IndexMap` to maintain insertion order without a separate `Vec`,
/// removing redundant storage and making `remove_variable` O(n) instead of
/// two separate O(n) operations.
#[derive(Debug, Clone, Default)]
pub struct BindingContext {
    /// Map from variable name to its info, in definition order.
    variables: IndexMap<String, VariableInfo>,
}

impl BindingContext {
    /// Creates a new empty binding context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: IndexMap::new(),
        }
    }

    /// Adds a variable to the context.
    ///
    /// If the variable is already defined, replaces its info but preserves its
    /// position in definition order.
    pub fn add_variable(&mut self, name: String, info: VariableInfo) {
        self.variables.insert(name, info);
    }

    /// Looks up a variable by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&VariableInfo> {
        self.variables.get(name)
    }

    /// Checks if a variable is defined.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Returns all variable names in definition order.
    #[must_use]
    pub fn variable_names(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }

    /// Returns the number of bound variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Returns true if no variables are bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// Removes a variable from the context (used for temporary scoping).
    pub fn remove_variable(&mut self, name: &str) {
        self.variables.shift_remove(name);
    }
}

/// Semantic binder for query plans.
///
/// The binder walks the logical plan and:
/// 1. Collects all variable definitions
/// 2. Validates that all variable references are valid
/// 3. Infers types where possible
/// 4. Reports semantic errors
pub struct Binder {
    /// The current binding context.
    context: BindingContext,
    /// The outer variables the `CALL` subquery being bound imports, with what
    /// they are in the outer query: a parameter scan after a `WITH` in its
    /// body that leaves one out brings it back as that.
    imports: BindingContext,
}

impl Binder {
    /// Creates a new binder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            context: BindingContext::new(),
            imports: BindingContext::new(),
        }
    }

    /// Binds a logical plan, returning the binding context.
    ///
    /// # Errors
    ///
    /// Returns an error if semantic validation fails.
    pub fn bind(&mut self, plan: &LogicalPlan) -> Result<BindingContext> {
        self.bind_operator(&plan.root)?;
        Ok(self.context.clone())
    }

    /// Binds a single logical operator.
    fn bind_operator(&mut self, op: &LogicalOperator) -> Result<()> {
        match op {
            LogicalOperator::NodeScan(scan) => self.bind_node_scan(scan),
            LogicalOperator::Expand(expand) => self.bind_expand(expand),
            LogicalOperator::Filter(filter) => self.bind_filter(filter),
            LogicalOperator::Return(ret) => self.bind_return(ret),
            LogicalOperator::Project(project) => {
                self.bind_operator(&project.input)?;
                // A projection that does not pass its input through (a WITH)
                // ends the scope of what it leaves out: its rows hold only
                // the projected columns.
                let mut projected = (!project.pass_through_input).then(BindingContext::new);
                for projection in &project.projections {
                    self.validate_expression(&projection.expression)?;
                    // Add the projection alias to the context (for WITH clause support)
                    if let Some(ref alias) = projection.alias {
                        // Determine the type from the expression
                        let data_type = self.infer_expression_type(&projection.expression);
                        // Propagate node/edge status when projecting a variable
                        // or a Case that selects between node variables (used
                        // by optional() and union() translations).
                        let (is_node, is_edge) = self.infer_entity_status(&projection.expression);
                        let info = VariableInfo {
                            name: alias.clone(),
                            data_type,
                            is_node,
                            is_edge,
                        };
                        if let Some(projected) = &mut projected {
                            projected.add_variable(alias.clone(), info.clone());
                        }
                        self.context.add_variable(alias.clone(), info);
                    } else if let Some(projected) = &mut projected {
                        // An unaliased variable passes on as itself; any
                        // other unaliased item is a column named after its
                        // expression (`a.name`), which does not keep `a`.
                        let name = crate::query::planner::common::expression_to_string(
                            &projection.expression,
                        );
                        let info = self.context.get(&name).cloned().unwrap_or(VariableInfo {
                            name: name.clone(),
                            data_type: LogicalType::Any,
                            is_node: false,
                            is_edge: false,
                        });
                        projected.add_variable(name, info);
                    }
                }
                if let Some(projected) = projected {
                    self.context = projected;
                }
                Ok(())
            }
            LogicalOperator::Limit(limit) => self.bind_operator(&limit.input),
            LogicalOperator::Skip(skip) => self.bind_operator(&skip.input),
            LogicalOperator::Sort(sort) => {
                self.bind_operator(&sort.input)?;
                for key in &sort.keys {
                    self.validate_expression(&key.expression)?;
                }
                Ok(())
            }
            LogicalOperator::CreateNode(create) => {
                if let Some(ref input) = create.input {
                    self.bind_operator(input)?;
                }
                self.bind_created_node(&create.variable, &create.properties)
            }
            LogicalOperator::Create(create) => {
                if let Some(ref input) = create.input {
                    self.bind_operator(input)?;
                }
                for element in &create.elements {
                    match element {
                        CreateElement::Node {
                            variable,
                            properties,
                            ..
                        } => self.bind_created_node(variable, properties)?,
                        CreateElement::Edge {
                            variable,
                            from_variable,
                            to_variable,
                            properties,
                            ..
                        } => self.bind_created_edge(
                            variable.as_ref(),
                            from_variable,
                            to_variable,
                            properties,
                        )?,
                    }
                }
                Ok(())
            }
            LogicalOperator::EdgeScan(scan) => {
                if let Some(ref input) = scan.input {
                    self.bind_operator(input)?;
                }
                self.bind_element(&scan.variable, Element::Edge)
            }
            LogicalOperator::Distinct(distinct) => self.bind_operator(&distinct.input),
            LogicalOperator::Join(join) => self.bind_join(join),
            LogicalOperator::Aggregate(agg) => self.bind_aggregate(agg),
            LogicalOperator::CreateEdge(create) => {
                self.bind_operator(&create.input)?;
                self.bind_created_edge(
                    create.variable.as_ref(),
                    &create.from_variable,
                    &create.to_variable,
                    &create.properties,
                )
            }
            LogicalOperator::DeleteNode(delete) => {
                self.bind_operator(&delete.input)?;
                // Validate that the variable to delete is defined
                if !self.context.contains(&delete.variable) {
                    return Err(undefined_variable_error(
                        &delete.variable,
                        &self.context,
                        " in DELETE",
                    ));
                }
                Ok(())
            }
            LogicalOperator::DeleteEdge(delete) => {
                self.bind_operator(&delete.input)?;
                // Validate that the variable to delete is defined
                if !self.context.contains(&delete.variable) {
                    return Err(undefined_variable_error(
                        &delete.variable,
                        &self.context,
                        " in DELETE",
                    ));
                }
                Ok(())
            }
            LogicalOperator::SetProperty(set) => {
                self.bind_operator(&set.input)?;
                // Validate that the variable to update is defined
                if !self.context.contains(&set.variable) {
                    return Err(undefined_variable_error(
                        &set.variable,
                        &self.context,
                        " in SET",
                    ));
                }
                // Validate property value expressions
                for (_, expr) in &set.properties {
                    self.validate_expression(expr)?;
                }
                Ok(())
            }
            LogicalOperator::Empty => Ok(()),

            LogicalOperator::Unwind(unwind) => {
                // First bind the input
                self.bind_operator(&unwind.input)?;
                // Validate the expression being unwound
                self.validate_expression(&unwind.expression)?;
                // Add the new variable to the context
                self.context.add_variable(
                    unwind.variable.clone(),
                    VariableInfo {
                        name: unwind.variable.clone(),
                        data_type: LogicalType::Any, // Unwound elements can be any type
                        is_node: false,
                        is_edge: false,
                    },
                );
                // Add ORDINALITY variable if present (1-based index)
                if let Some(ref ord_var) = unwind.ordinality_var {
                    self.context.add_variable(
                        ord_var.clone(),
                        VariableInfo {
                            name: ord_var.clone(),
                            data_type: LogicalType::Int64,
                            is_node: false,
                            is_edge: false,
                        },
                    );
                }
                // Add OFFSET variable if present (0-based index)
                if let Some(ref off_var) = unwind.offset_var {
                    self.context.add_variable(
                        off_var.clone(),
                        VariableInfo {
                            name: off_var.clone(),
                            data_type: LogicalType::Int64,
                            is_node: false,
                            is_edge: false,
                        },
                    );
                }
                Ok(())
            }

            // RDF/SPARQL operators
            LogicalOperator::TripleScan(scan) => self.bind_triple_scan(scan),
            LogicalOperator::Union(union) => self.bind_branches(&union.inputs),
            LogicalOperator::LeftJoin(lj) => {
                self.bind_join_inputs(&lj.left, &lj.right)?;
                if let Some(ref cond) = lj.condition {
                    self.validate_expression(cond)?;
                }
                Ok(())
            }
            LogicalOperator::AntiJoin(aj) => self.bind_join_inputs(&aj.left, &aj.right),
            LogicalOperator::Bind(bind) => {
                self.bind_operator(&bind.input)?;
                self.validate_expression(&bind.expression)?;
                self.context.add_variable(
                    bind.variable.clone(),
                    VariableInfo {
                        name: bind.variable.clone(),
                        data_type: LogicalType::Any,
                        is_node: false,
                        is_edge: false,
                    },
                );
                Ok(())
            }
            LogicalOperator::Merge(merge) => {
                // First bind the input
                self.bind_operator(&merge.input)?;
                // Match properties are validated against the outer scope: ISO/IEC 39075:2024
                // §15.5 forbids the MERGE pattern from referencing the variable it introduces.
                for (_, expr) in &merge.match_properties {
                    self.validate_expression(expr)?;
                }
                // The MERGE variable is in scope inside ON CREATE / ON MATCH SET, so add it
                // before validating those action expressions. Without this, expressions like
                // `ON MATCH SET n.x = coalesce(n.x, 0)` failed with `Undefined variable 'n'`.
                self.context.add_variable(
                    merge.variable.clone(),
                    VariableInfo {
                        name: merge.variable.clone(),
                        data_type: LogicalType::Node,
                        is_node: true,
                        is_edge: false,
                    },
                );
                for (_, expr) in &merge.on_create {
                    self.validate_expression(expr)?;
                }
                for (_, expr) in &merge.on_match {
                    self.validate_expression(expr)?;
                }
                Ok(())
            }
            LogicalOperator::MergeRelationship(merge_rel) => {
                self.bind_operator(&merge_rel.input)?;
                // Validate source and target variables exist
                if !self.context.contains(&merge_rel.source_variable) {
                    return Err(undefined_variable_error(
                        &merge_rel.source_variable,
                        &self.context,
                        " in MERGE relationship source",
                    ));
                }
                if !self.context.contains(&merge_rel.target_variable) {
                    return Err(undefined_variable_error(
                        &merge_rel.target_variable,
                        &self.context,
                        " in MERGE relationship target",
                    ));
                }
                // Match properties cannot reference the edge variable (same rule as MERGE node).
                for (_, expr) in &merge_rel.match_properties {
                    self.validate_expression(expr)?;
                }
                // Edge variable is in scope inside ON CREATE / ON MATCH SET.
                self.context.add_variable(
                    merge_rel.variable.clone(),
                    VariableInfo {
                        name: merge_rel.variable.clone(),
                        data_type: LogicalType::Edge,
                        is_node: false,
                        is_edge: true,
                    },
                );
                for (_, expr) in &merge_rel.on_create {
                    self.validate_expression(expr)?;
                }
                for (_, expr) in &merge_rel.on_match {
                    self.validate_expression(expr)?;
                }
                Ok(())
            }
            LogicalOperator::AddLabel(add_label) => {
                self.bind_operator(&add_label.input)?;
                // Validate that the variable exists
                if !self.context.contains(&add_label.variable) {
                    return Err(undefined_variable_error(
                        &add_label.variable,
                        &self.context,
                        " in SET labels",
                    ));
                }
                Ok(())
            }
            LogicalOperator::RemoveLabel(remove_label) => {
                self.bind_operator(&remove_label.input)?;
                // Validate that the variable exists
                if !self.context.contains(&remove_label.variable) {
                    return Err(undefined_variable_error(
                        &remove_label.variable,
                        &self.context,
                        " in REMOVE labels",
                    ));
                }
                Ok(())
            }
            LogicalOperator::ShortestPath(sp) => {
                // First bind the input
                self.bind_operator(&sp.input)?;
                // Validate that source and target variables are defined
                if !self.context.contains(&sp.source_var) {
                    return Err(undefined_variable_error(
                        &sp.source_var,
                        &self.context,
                        " (source in shortestPath)",
                    ));
                }
                // A search to every node binds the target itself
                if sp.binds_target {
                    self.bind_element(&sp.target_var, Element::Node)?;
                } else if !self.context.contains(&sp.target_var) {
                    return Err(undefined_variable_error(
                        &sp.target_var,
                        &self.context,
                        " (target in shortestPath)",
                    ));
                }
                // The edge condition reads the candidate edge and the input row
                if let Some(condition) = &sp.edge_condition {
                    self.bind_element(&condition.variable, Element::Edge)?;
                    self.validate_expression(&condition.predicate)?;
                }
                if let Some(edge_variable) = &sp.edge_variable {
                    self.bind_element(edge_variable, Element::Edge)?;
                }
                // nodes(p) and edges(p) read these columns
                for column in [
                    format!("_path_nodes_{}", sp.path_alias),
                    format!("_path_edges_{}", sp.path_alias),
                ] {
                    self.context.add_variable(
                        column.clone(),
                        VariableInfo {
                            name: column,
                            data_type: LogicalType::Any,
                            is_node: false,
                            is_edge: false,
                        },
                    );
                }
                // Add the path alias variable to the context
                self.context.add_variable(
                    sp.path_alias.clone(),
                    VariableInfo {
                        name: sp.path_alias.clone(),
                        data_type: LogicalType::Any, // Path is a complex type
                        is_node: false,
                        is_edge: false,
                    },
                );
                // Also add the path length variable for length(p) calls
                let path_length_var = format!("_path_length_{}", sp.path_alias);
                self.context.add_variable(
                    path_length_var.clone(),
                    VariableInfo {
                        name: path_length_var,
                        data_type: LogicalType::Int64,
                        is_node: false,
                        is_edge: false,
                    },
                );
                Ok(())
            }
            // SPARQL Update operators - these don't require variable binding
            LogicalOperator::InsertTriple(insert) => {
                if let Some(ref input) = insert.input {
                    self.bind_operator(input)?;
                }
                Ok(())
            }
            LogicalOperator::DeleteTriple(delete) => {
                if let Some(ref input) = delete.input {
                    self.bind_operator(input)?;
                }
                Ok(())
            }
            LogicalOperator::Modify(modify) => {
                self.bind_operator(&modify.where_clause)?;
                Ok(())
            }
            LogicalOperator::ClearGraph(_)
            | LogicalOperator::CreateGraph(_)
            | LogicalOperator::DropGraph(_)
            | LogicalOperator::LoadGraph(_)
            | LogicalOperator::CopyGraph(_)
            | LogicalOperator::MoveGraph(_)
            | LogicalOperator::AddGraph(_) => Ok(()),
            // A horizontal aggregate passes its input's rows on with the
            // value it computes over the list column of each row.
            LogicalOperator::HorizontalAggregate(aggregate) => {
                self.bind_operator(&aggregate.input)?;
                if !self.context.contains(&aggregate.list_column) {
                    return Err(undefined_variable_error(
                        &aggregate.list_column,
                        &self.context,
                        " in a horizontal aggregate",
                    ));
                }
                self.context.add_variable(
                    aggregate.alias.clone(),
                    VariableInfo {
                        name: aggregate.alias.clone(),
                        data_type: LogicalType::Any,
                        is_node: false,
                        is_edge: false,
                    },
                );
                Ok(())
            }
            LogicalOperator::VectorScan(scan) => {
                // VectorScan introduces a variable for matched nodes
                if let Some(ref input) = scan.input {
                    self.bind_operator(input)?;
                }
                self.context.add_variable(
                    scan.variable.clone(),
                    VariableInfo {
                        name: scan.variable.clone(),
                        data_type: LogicalType::Node,
                        is_node: true,
                        is_edge: false,
                    },
                );
                // Validate the query vector expression
                self.validate_expression(&scan.query_vector)?;
                Ok(())
            }
            LogicalOperator::VectorJoin(join) => {
                // VectorJoin takes input from left side and produces right-side matches
                self.bind_operator(&join.input)?;
                // Add right variable for matched nodes
                self.context.add_variable(
                    join.right_variable.clone(),
                    VariableInfo {
                        name: join.right_variable.clone(),
                        data_type: LogicalType::Node,
                        is_node: true,
                        is_edge: false,
                    },
                );
                // Optionally add score variable
                if let Some(ref score_var) = join.score_variable {
                    self.context.add_variable(
                        score_var.clone(),
                        VariableInfo {
                            name: score_var.clone(),
                            data_type: LogicalType::Float64,
                            is_node: false,
                            is_edge: false,
                        },
                    );
                }
                // Validate the query vector expression
                self.validate_expression(&join.query_vector)?;
                Ok(())
            }
            LogicalOperator::MapCollect(mc) => {
                self.bind_operator(&mc.input)?;
                self.context.add_variable(
                    mc.alias.clone(),
                    VariableInfo {
                        name: mc.alias.clone(),
                        data_type: LogicalType::Any,
                        is_node: false,
                        is_edge: false,
                    },
                );
                Ok(())
            }
            LogicalOperator::Except(except) => {
                self.bind_branches([except.left.as_ref(), except.right.as_ref()])
            }
            LogicalOperator::Intersect(intersect) => {
                self.bind_branches([intersect.left.as_ref(), intersect.right.as_ref()])
            }
            LogicalOperator::Otherwise(otherwise) => {
                self.bind_branches([otherwise.left.as_ref(), otherwise.right.as_ref()])
            }
            LogicalOperator::Apply(apply) => {
                // Snapshot context BEFORE binding the input, so we can detect
                // which variables were added by the input plan.
                let pre_apply_names: HashSet<String> =
                    self.context.variable_names().iter().cloned().collect();

                self.bind_operator(&apply.input)?;

                // Scope down: when the input plan exposes a Return/Aggregate
                // projection (not a raw scan/expand), remove its internal-only
                // variables. Only the projected output columns should be visible
                // to the subplan — this prevents variables internal to a sibling
                // CALL block from leaking into the next CALL block.
                let mut input_output_ctx = BindingContext::new();
                Self::register_subplan_columns(&apply.input, &mut input_output_ctx);
                let input_output_names: HashSet<String> =
                    input_output_ctx.variable_names().iter().cloned().collect();

                if !input_output_names.is_empty() {
                    // Input has an explicit projection: remove its internals.
                    let input_internals: Vec<String> = self
                        .context
                        .variable_names()
                        .iter()
                        .filter(|n| {
                            !pre_apply_names.contains(*n) && !input_output_names.contains(*n)
                        })
                        .cloned()
                        .collect();
                    for name in input_internals {
                        self.context.remove_variable(&name);
                    }
                }

                // Snapshot the permitted outer context for the subplan.
                let outer_names: HashSet<String> =
                    self.context.variable_names().iter().cloned().collect();

                // A name the subquery imports must be a variable of the outer
                // query (`*` imports all of them).
                if let Some(missing) = apply
                    .shared_variables
                    .iter()
                    .find(|name| *name != "*" && !outer_names.contains(*name))
                {
                    return Err(undefined_variable_error(
                        missing,
                        &self.context,
                        " imported into CALL",
                    ));
                }

                // A subquery sees the outer variables it imports: the ones its
                // scope clause or importing `WITH` names (`CALL (a, b)`,
                // `WITH a`), all of them for `*`, and none when it names none
                // (`CALL () { ... }`, a Cypher `CALL { ... }` without an
                // importing `WITH`): the planner then runs it without the
                // outer row. It binds in a context of its own, so neither its
                // internal variables nor a `WITH` in it that drops outer ones
                // change the outer scope.
                let imports_all = apply.shared_variables.iter().any(|name| name == "*");
                let bound = match union_parts(&apply.subplan) {
                    // Each part of a UNION imports its own outer variables (the
                    // Apply imports all of them): the ones its scan starts from.
                    Some(parts) if !imports_all => parts.iter().try_for_each(|part| {
                        let part_context = self.imported(&leading_imports(part));
                        self.bind_subquery_part(part, part_context)
                    }),
                    _ => {
                        let subplan_context = if imports_all {
                            self.context.clone()
                        } else {
                            self.imported(&apply.shared_variables)
                        };
                        self.bind_subquery_part(&apply.subplan, subplan_context)
                    }
                };
                bound?;

                // A subquery returns new variables only, as in openCypher: an
                // outer one, imported or not, would be bound twice.
                let mut subplan_output_ctx = BindingContext::new();
                Self::register_subplan_columns(&apply.subplan, &mut subplan_output_ctx);
                if let Some(clash) = subplan_output_ctx
                    .variable_names()
                    .into_iter()
                    .find(|name| outer_names.contains(name))
                {
                    return Err(binding_error_with_hint(
                        format!("Variable '{clash}' is already declared outside the CALL subquery"),
                        "return it under a new name",
                    ));
                }

                // Register output columns so downstream operators can reference them.
                Self::register_subplan_columns(&apply.subplan, &mut self.context);
                Ok(())
            }
            LogicalOperator::MultiWayJoin(mwj) => {
                for input in &mwj.inputs {
                    let before = self.context.clone();
                    self.bind_operator(input)?;
                    self.keep_join_scope(before);
                }
                for cond in &mwj.conditions {
                    self.validate_expression(&cond.left)?;
                    self.validate_expression(&cond.right)?;
                }
                Ok(())
            }
            LogicalOperator::ParameterScan(param_scan) => {
                // Register parameter columns as variables (injected by outer
                // Apply). A variable of the outer query keeps what it is, so a
                // CALL subquery can match an imported edge as an edge, also
                // where a scan brings an import back after a `WITH` that left
                // it out.
                for col in &param_scan.columns {
                    if self.context.contains(col) {
                        continue;
                    }
                    let info = self.imports.get(col).cloned().unwrap_or(VariableInfo {
                        name: col.clone(),
                        data_type: LogicalType::Any,
                        is_node: true,
                        is_edge: false,
                    });
                    self.context.add_variable(col.clone(), info);
                }
                Ok(())
            }
            // DDL operators don't need binding: they're handled before the binder
            LogicalOperator::CreatePropertyGraph(_) => Ok(()),
            // Procedure calls: register yielded columns as variables for downstream operators
            LogicalOperator::CallProcedure(call) => {
                if let Some(yields) = &call.yield_items {
                    for item in yields {
                        let var_name = item.alias.as_deref().unwrap_or(&item.field_name);
                        self.context.add_variable(
                            var_name.to_string(),
                            VariableInfo {
                                name: var_name.to_string(),
                                data_type: LogicalType::Any,
                                is_node: false,
                                is_edge: false,
                            },
                        );
                    }
                }
                Ok(())
            }
            LogicalOperator::LoadData(load) => {
                // The row variable is bound as Any (Map or List depending on WITH HEADERS)
                self.context.add_variable(
                    load.variable.clone(),
                    VariableInfo {
                        name: load.variable.clone(),
                        data_type: LogicalType::Any,
                        is_node: false,
                        is_edge: false,
                    },
                );
                Ok(())
            }
            LogicalOperator::Construct(construct) => self.bind_operator(&construct.input),
            LogicalOperator::TextScan(scan) => {
                self.context.add_variable(
                    scan.variable.clone(),
                    VariableInfo {
                        name: scan.variable.clone(),
                        data_type: LogicalType::Node,
                        is_node: true,
                        is_edge: false,
                    },
                );
                if let Some(ref score_col) = scan.score_column {
                    self.context.add_variable(
                        score_col.clone(),
                        VariableInfo {
                            name: score_col.clone(),
                            data_type: LogicalType::Float64,
                            is_node: false,
                            is_edge: false,
                        },
                    );
                }
                self.validate_expression(&scan.query)?;
                Ok(())
            }
        }
    }

    /// Binds a triple scan operator (for RDF/SPARQL).
    fn bind_triple_scan(&mut self, scan: &TripleScanOp) -> Result<()> {
        use crate::query::plan::TripleComponent;

        // First bind the input if present
        if let Some(ref input) = scan.input {
            self.bind_operator(input)?;
        }

        // Add variables for subject, predicate, object
        if let TripleComponent::Variable(name) = &scan.subject
            && !self.context.contains(name)
        {
            self.context.add_variable(
                name.clone(),
                VariableInfo {
                    name: name.clone(),
                    data_type: LogicalType::Any, // RDF term
                    is_node: false,
                    is_edge: false,
                },
            );
        }

        if let TripleComponent::Variable(name) = &scan.predicate
            && !self.context.contains(name)
        {
            self.context.add_variable(
                name.clone(),
                VariableInfo {
                    name: name.clone(),
                    data_type: LogicalType::Any, // IRI
                    is_node: false,
                    is_edge: false,
                },
            );
        }

        if let TripleComponent::Variable(name) = &scan.object
            && !self.context.contains(name)
        {
            self.context.add_variable(
                name.clone(),
                VariableInfo {
                    name: name.clone(),
                    data_type: LogicalType::Any, // RDF term
                    is_node: false,
                    is_edge: false,
                },
            );
        }

        if let Some(TripleComponent::Variable(name)) = &scan.graph
            && !self.context.contains(name)
        {
            self.context.add_variable(
                name.clone(),
                VariableInfo {
                    name: name.clone(),
                    data_type: LogicalType::Any, // IRI
                    is_node: false,
                    is_edge: false,
                },
            );
        }

        Ok(())
    }

    /// Binds a node scan operator.
    fn bind_node_scan(&mut self, scan: &NodeScanOp) -> Result<()> {
        // First bind the input if present
        if let Some(ref input) = scan.input {
            self.bind_operator(input)?;
        }

        // Add the scanned variable to scope
        self.bind_element(&scan.variable, Element::Node)
    }

    /// Binds `name` to a node or an edge of a pattern. A name already bound
    /// to the other kind, or to a value, is an error: matching it would
    /// compare a node with an edge (or a number) and match by a coincidence
    /// of IDs. A name bound to the same kind, or to something of unknown
    /// kind (an UNWIND variable, a procedure result), is bound again.
    fn bind_element(&mut self, name: &str, element: Element) -> Result<()> {
        if let Some(info) = self.context.get(name) {
            let conflict = match element {
                Element::Node if info.is_edge => Some("is an edge, so it cannot also be a node"),
                Element::Edge if info.is_node => Some("is a node, so it cannot also be an edge"),
                Element::Node if holds_a_value(info) => {
                    Some("holds a value, so it cannot be a node")
                }
                Element::Edge if holds_a_value(info) => {
                    Some("holds a value, so it cannot be an edge")
                }
                _ => None,
            };
            if let Some(conflict) = conflict {
                return Err(binding_error(format!("Variable '{name}' {conflict}")));
            }
        }
        let is_edge = element == Element::Edge;
        self.context.add_variable(
            name.to_string(),
            VariableInfo {
                name: name.to_string(),
                data_type: if is_edge {
                    LogicalType::Edge
                } else {
                    LogicalType::Node
                },
                is_node: !is_edge,
                is_edge,
            },
        );
        Ok(())
    }

    /// Binds the branches of a set operation, each in the scope the first one
    /// started from: a branch is a query of its own, so a name may be a node
    /// in one branch and an edge in the next. After the last branch the scope
    /// has every name a branch bound; a name the branches bind differently is
    /// of unknown kind.
    fn bind_branches<'a>(
        &mut self,
        branches: impl IntoIterator<Item = &'a LogicalOperator>,
    ) -> Result<()> {
        let before = self.context.clone();
        let mut after: IndexMap<String, VariableInfo> = IndexMap::new();
        for branch in branches {
            self.context = before.clone();
            self.bind_operator(branch)?;
            for (name, info) in &self.context.variables {
                match after.get_mut(name) {
                    Some(seen) if seen.is_node != info.is_node || seen.is_edge != info.is_edge => {
                        seen.is_node = false;
                        seen.is_edge = false;
                        seen.data_type = LogicalType::Any;
                    }
                    Some(_) => {}
                    None => {
                        after.insert(name.clone(), info.clone());
                    }
                }
            }
        }
        self.context = BindingContext { variables: after };
        Ok(())
    }

    /// Binds an expand operator.
    fn bind_expand(&mut self, expand: &ExpandOp) -> Result<()> {
        // First bind the input
        self.bind_operator(&expand.input)?;

        // Validate that the source variable is defined
        if !self.context.contains(&expand.from_variable) {
            return Err(undefined_variable_error(
                &expand.from_variable,
                &self.context,
                " in EXPAND",
            ));
        }

        // Validate that the source is a node
        if let Some(info) = self.context.get(&expand.from_variable)
            && !info.is_node
        {
            return Err(binding_error(format!(
                "Variable '{}' is not a node, cannot expand from it",
                expand.from_variable
            )));
        }

        // Add edge variable if present
        if let Some(ref edge_var) = expand.edge_variable {
            self.bind_element(edge_var, Element::Edge)?;
        }

        // Add target variable
        self.bind_element(&expand.to_variable, Element::Node)?;

        // Add path variables for variable-length paths
        if let Some(ref path_alias) = expand.path_alias {
            // Register the path variable itself (e.g. p in MATCH p=...)
            self.context.add_variable(
                path_alias.clone(),
                VariableInfo {
                    name: path_alias.clone(),
                    data_type: LogicalType::Any,
                    is_node: false,
                    is_edge: false,
                },
            );
            // length(p) → _path_length_p
            let path_length_var = format!("_path_length_{}", path_alias);
            self.context.add_variable(
                path_length_var.clone(),
                VariableInfo {
                    name: path_length_var,
                    data_type: LogicalType::Int64,
                    is_node: false,
                    is_edge: false,
                },
            );
            // nodes(p) → _path_nodes_p
            let path_nodes_var = format!("_path_nodes_{}", path_alias);
            self.context.add_variable(
                path_nodes_var.clone(),
                VariableInfo {
                    name: path_nodes_var,
                    data_type: LogicalType::Any,
                    is_node: false,
                    is_edge: false,
                },
            );
            // edges(p) → _path_edges_p
            let path_edges_var = format!("_path_edges_{}", path_alias);
            self.context.add_variable(
                path_edges_var.clone(),
                VariableInfo {
                    name: path_edges_var,
                    data_type: LogicalType::Any,
                    is_node: false,
                    is_edge: false,
                },
            );
        }

        Ok(())
    }

    /// Binds a filter operator.
    fn bind_filter(&mut self, filter: &FilterOp) -> Result<()> {
        // First bind the input
        self.bind_operator(&filter.input)?;

        // Validate the predicate expression
        self.validate_expression(&filter.predicate)?;

        Ok(())
    }

    /// Registers output columns from a subplan into the binding context.
    /// Walks through wrapping operators to find a Return and extracts column names.
    fn register_subplan_columns(plan: &LogicalOperator, ctx: &mut BindingContext) {
        match plan {
            LogicalOperator::Return(ret) => {
                for item in &ret.items {
                    let col_name = if let Some(alias) = &item.alias {
                        alias.clone()
                    } else {
                        match &item.expression {
                            LogicalExpression::Variable(name) => name.clone(),
                            LogicalExpression::Property { variable, property } => {
                                format!("{variable}.{property}")
                            }
                            _ => continue,
                        }
                    };
                    ctx.add_variable(
                        col_name.clone(),
                        VariableInfo {
                            name: col_name,
                            data_type: LogicalType::Any,
                            is_node: false,
                            is_edge: false,
                        },
                    );
                }
            }
            LogicalOperator::Sort(s) => Self::register_subplan_columns(&s.input, ctx),
            LogicalOperator::Limit(l) => Self::register_subplan_columns(&l.input, ctx),
            LogicalOperator::Skip(s) => Self::register_subplan_columns(&s.input, ctx),
            // The branches of a UNION return the same columns.
            LogicalOperator::Union(u) => {
                if let Some(first) = u.inputs.first() {
                    Self::register_subplan_columns(first, ctx);
                }
            }
            LogicalOperator::Distinct(d) => Self::register_subplan_columns(&d.input, ctx),
            LogicalOperator::Aggregate(agg) => {
                // Aggregate produces named output columns
                for expr in &agg.aggregates {
                    if let Some(alias) = &expr.alias {
                        ctx.add_variable(
                            alias.clone(),
                            VariableInfo {
                                name: alias.clone(),
                                data_type: LogicalType::Any,
                                is_node: false,
                                is_edge: false,
                            },
                        );
                    }
                }
            }
            _ => {}
        }
    }

    /// Binds a return operator.
    fn bind_return(&mut self, ret: &ReturnOp) -> Result<()> {
        // First bind the input
        self.bind_operator(&ret.input)?;

        // Validate all return expressions and register aliases
        // (aliases must be visible to parent Sort for ORDER BY resolution)
        for item in &ret.items {
            self.validate_return_item(item)?;
            if let Some(ref alias) = item.alias {
                let data_type = self.infer_expression_type(&item.expression);
                self.context.add_variable(
                    alias.clone(),
                    VariableInfo {
                        name: alias.clone(),
                        data_type,
                        is_node: false,
                        is_edge: false,
                    },
                );
            }
        }

        Ok(())
    }

    /// Validates a return item.
    fn validate_return_item(&mut self, item: &ReturnItem) -> Result<()> {
        self.validate_expression(&item.expression)
    }

    /// Validates that an expression only references defined variables.
    fn validate_expression(&mut self, expr: &LogicalExpression) -> Result<()> {
        match expr {
            LogicalExpression::Variable(name) => {
                // "*" is a wildcard marker for RETURN *, expanded by the planner
                if name == "*" {
                    return Ok(());
                }
                if !self.context.contains(name) {
                    return Err(undefined_variable_error(name, &self.context, ""));
                }
                Ok(())
            }
            LogicalExpression::Property { variable, .. } => {
                if !self.context.contains(variable) {
                    return Err(undefined_variable_error(
                        variable,
                        &self.context,
                        " in property access",
                    ));
                }
                Ok(())
            }
            LogicalExpression::Literal(_) => Ok(()),
            LogicalExpression::Binary { left, right, .. } => {
                self.validate_expression(left)?;
                self.validate_expression(right)
            }
            LogicalExpression::Unary { operand, .. } => self.validate_expression(operand),
            LogicalExpression::FunctionCall { args, .. } => {
                for arg in args {
                    self.validate_expression(arg)?;
                }
                Ok(())
            }
            LogicalExpression::List(items) => {
                for item in items {
                    self.validate_expression(item)?;
                }
                Ok(())
            }
            LogicalExpression::Map(pairs) => {
                for (_, value) in pairs {
                    self.validate_expression(value)?;
                }
                Ok(())
            }
            LogicalExpression::IndexAccess { base, index } => {
                self.validate_expression(base)?;
                self.validate_expression(index)
            }
            LogicalExpression::MapAccess { base, .. } => self.validate_expression(base),
            LogicalExpression::SliceAccess { base, start, end } => {
                self.validate_expression(base)?;
                if let Some(s) = start {
                    self.validate_expression(s)?;
                }
                if let Some(e) = end {
                    self.validate_expression(e)?;
                }
                Ok(())
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    self.validate_expression(op)?;
                }
                for (cond, result) in when_clauses {
                    self.validate_expression(cond)?;
                    self.validate_expression(result)?;
                }
                if let Some(else_expr) = else_clause {
                    self.validate_expression(else_expr)?;
                }
                Ok(())
            }
            // Parameter references are validated externally
            LogicalExpression::Parameter(_) => Ok(()),
            // labels(n), type(e), id(n) need the variable to be defined
            LogicalExpression::Labels(var)
            | LogicalExpression::Type(var)
            | LogicalExpression::Id(var) => {
                if !self.context.contains(var) {
                    return Err(undefined_variable_error(var, &self.context, " in function"));
                }
                Ok(())
            }
            LogicalExpression::ListComprehension { list_expr, .. } => {
                // Validate the list expression against the outer context.
                // The filter and map expressions use the iteration variable
                // which is locally scoped, so we skip validating them here.
                self.validate_expression(list_expr)?;
                Ok(())
            }
            LogicalExpression::ListPredicate { list_expr, .. } => {
                // Validate the list expression against the outer context.
                // The predicate uses the iteration variable which is locally
                // scoped, so we skip validating it against the outer context.
                self.validate_expression(list_expr)?;
                Ok(())
            }
            LogicalExpression::ExistsSubquery(subquery)
            | LogicalExpression::CountSubquery(subquery)
            | LogicalExpression::ValueSubquery(subquery) => {
                // Subqueries have their own binding context
                // For now, just validate the structure exists
                let _ = subquery; // Would need recursive binding
                Ok(())
            }
            LogicalExpression::PatternComprehension {
                subplan,
                projection,
            } => {
                // Bind the subplan to register pattern variables (e.g., `f` in `(p)-[:KNOWS]->(f)`)
                self.bind_operator(subplan)?;
                // Now validate the projection expression (e.g., `f.name`)
                self.validate_expression(projection)
            }
            LogicalExpression::MapProjection { base, entries } => {
                if !self.context.contains(base) {
                    return Err(undefined_variable_error(
                        base,
                        &self.context,
                        " in map projection",
                    ));
                }
                for entry in entries {
                    if let crate::query::plan::MapProjectionEntry::LiteralEntry(_, expr) = entry {
                        self.validate_expression(expr)?;
                    }
                }
                Ok(())
            }
            LogicalExpression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                self.validate_expression(initial)?;
                self.validate_expression(list)?;
                // accumulator and variable are locally scoped: inject them
                // into context, validate body, then remove
                let had_acc = self.context.contains(accumulator);
                let had_var = self.context.contains(variable);
                if !had_acc {
                    self.context.add_variable(
                        accumulator.clone(),
                        VariableInfo {
                            name: accumulator.clone(),
                            data_type: LogicalType::Any,
                            is_node: false,
                            is_edge: false,
                        },
                    );
                }
                if !had_var {
                    self.context.add_variable(
                        variable.clone(),
                        VariableInfo {
                            name: variable.clone(),
                            data_type: LogicalType::Any,
                            is_node: false,
                            is_edge: false,
                        },
                    );
                }
                self.validate_expression(expression)?;
                if !had_acc {
                    self.context.remove_variable(accumulator);
                }
                if !had_var {
                    self.context.remove_variable(variable);
                }
                Ok(())
            }
        }
    }

    /// Infers the type of an expression for use in WITH clause aliasing.
    fn infer_expression_type(&self, expr: &LogicalExpression) -> LogicalType {
        match expr {
            LogicalExpression::Variable(name) => {
                // Look up the variable type from context
                self.context
                    .get(name)
                    .map_or(LogicalType::Any, |info| info.data_type.clone())
            }
            LogicalExpression::Property { .. } => LogicalType::Any, // Properties can be any type
            LogicalExpression::Literal(value) => {
                // Infer type from literal value
                use grafeo_common::types::Value;
                match value {
                    Value::Bool(_) => LogicalType::Bool,
                    Value::Int64(_) => LogicalType::Int64,
                    Value::Float64(_) => LogicalType::Float64,
                    Value::String(_) => LogicalType::String,
                    Value::List(_) => LogicalType::Any, // Complex type
                    Value::Map(_) => LogicalType::Any,  // Complex type
                    Value::Null => LogicalType::Any,
                    _ => LogicalType::Any,
                }
            }
            LogicalExpression::Binary { .. } => LogicalType::Any, // Could be bool or numeric
            LogicalExpression::Unary { .. } => LogicalType::Any,
            LogicalExpression::FunctionCall { name, .. } => {
                // Infer based on function name
                match name.to_lowercase().as_str() {
                    "count" | "sum" | "id" => LogicalType::Int64,
                    "avg" => LogicalType::Float64,
                    "type" => LogicalType::String,
                    // List-returning functions use Any since we don't track element type
                    "labels" | "collect" => LogicalType::Any,
                    _ => LogicalType::Any,
                }
            }
            LogicalExpression::List(_) => LogicalType::Any, // Complex type
            LogicalExpression::Map(_) => LogicalType::Any,  // Complex type
            _ => LogicalType::Any,
        }
    }

    /// Infers whether an expression resolves to a node or edge entity.
    ///
    /// Returns `(is_node, is_edge)`. This propagates entity status through
    /// simple Variable references and Case expressions whose branches all
    /// agree on entity kind (used by optional() translation).
    fn infer_entity_status(&self, expr: &LogicalExpression) -> (bool, bool) {
        match expr {
            LogicalExpression::Variable(src) => self
                .context
                .get(src)
                .map_or((false, false), |info| (info.is_node, info.is_edge)),
            LogicalExpression::Case {
                when_clauses,
                else_clause,
                ..
            } => {
                // Collect entity status from all THEN and ELSE branches
                let mut all_node = true;
                let mut all_edge = true;
                let mut any_branch = false;
                for (_, then_expr) in when_clauses {
                    let (n, e) = self.infer_entity_status(then_expr);
                    all_node &= n;
                    all_edge &= e;
                    any_branch = true;
                }
                if let Some(else_expr) = else_clause {
                    let (n, e) = self.infer_entity_status(else_expr);
                    all_node &= n;
                    all_edge &= e;
                    any_branch = true;
                }
                if any_branch {
                    (all_node, all_edge)
                } else {
                    (false, false)
                }
            }
            _ => (false, false),
        }
    }

    /// Binds a node a CREATE or INSERT creates: it introduces `variable`.
    fn bind_created_node(
        &mut self,
        variable: &str,
        properties: &[(String, LogicalExpression)],
    ) -> Result<()> {
        self.context.add_variable(
            variable.to_string(),
            VariableInfo {
                name: variable.to_string(),
                data_type: LogicalType::Node,
                is_node: true,
                is_edge: false,
            },
        );
        for (_, expr) in properties {
            self.validate_expression(expr)?;
        }
        Ok(())
    }

    /// Binds an edge a CREATE or INSERT creates: its endpoints must be
    /// defined, and it introduces `variable` when it has one.
    fn bind_created_edge(
        &mut self,
        variable: Option<&String>,
        from_variable: &str,
        to_variable: &str,
        properties: &[(String, LogicalExpression)],
    ) -> Result<()> {
        if !self.context.contains(from_variable) {
            return Err(undefined_variable_error(
                from_variable,
                &self.context,
                " (source in CREATE EDGE)",
            ));
        }
        if !self.context.contains(to_variable) {
            return Err(undefined_variable_error(
                to_variable,
                &self.context,
                " (target in CREATE EDGE)",
            ));
        }
        if let Some(var) = variable {
            self.context.add_variable(
                var.clone(),
                VariableInfo {
                    name: var.clone(),
                    data_type: LogicalType::Edge,
                    is_node: false,
                    is_edge: true,
                },
            );
        }
        for (_, expr) in properties {
            self.validate_expression(expr)?;
        }
        Ok(())
    }

    /// Binds a join operator.
    fn bind_join(&mut self, join: &crate::query::plan::JoinOp) -> Result<()> {
        // Bind both sides of the join
        self.bind_join_inputs(&join.left, &join.right)?;

        // Validate join conditions
        for condition in &join.conditions {
            self.validate_expression(&condition.left)?;
            self.validate_expression(&condition.right)?;
        }

        Ok(())
    }

    /// Binds `part`, a `CALL` subquery's body or one part of a `UNION` in it,
    /// in a context of its own that starts as `imported`, the outer
    /// variables it imports. Neither its own variables nor a `WITH` in it
    /// change the outer scope.
    fn bind_subquery_part(
        &mut self,
        part: &LogicalOperator,
        imported: BindingContext,
    ) -> Result<()> {
        let enclosing_imports = std::mem::replace(&mut self.imports, imported.clone());
        let outer_context = std::mem::replace(&mut self.context, imported);
        let bound = self.bind_operator(part);
        self.context = outer_context;
        self.imports = enclosing_imports;
        bound
    }

    /// The outer variables named in `names`, with what they are in the
    /// current context.
    fn imported(&self, names: &[String]) -> BindingContext {
        let mut imported = BindingContext::new();
        for name in names {
            if let Some(info) = self.context.get(name) {
                imported.add_variable(name.clone(), info.clone());
            }
        }
        imported
    }

    /// Binds the two inputs of a join. The right one sees the left one's
    /// variables, and the join's rows hold the columns of both, so a
    /// projection inside the right input does not end the scope of the left
    /// one's variables.
    fn bind_join_inputs(&mut self, left: &LogicalOperator, right: &LogicalOperator) -> Result<()> {
        self.bind_operator(left)?;
        let after_left = self.context.clone();
        self.bind_operator(right)?;
        self.keep_join_scope(after_left);
        Ok(())
    }

    /// Makes the context the variables of `before` followed by those bound
    /// since, for an input that joins its rows to the ones `before` holds.
    fn keep_join_scope(&mut self, before: BindingContext) {
        let since = std::mem::replace(&mut self.context, before);
        for (name, info) in since.variables {
            self.context.add_variable(name, info);
        }
    }

    /// Binds an aggregate operator.
    fn bind_aggregate(&mut self, agg: &crate::query::plan::AggregateOp) -> Result<()> {
        // Bind the input first
        self.bind_operator(&agg.input)?;

        // Validate group by expressions
        for expr in &agg.group_by {
            self.validate_expression(expr)?;
        }

        // Validate aggregate expressions
        for agg_expr in &agg.aggregates {
            if let Some(ref expr) = agg_expr.expression {
                self.validate_expression(expr)?;
            }
            // Add the alias as a new variable if present
            if let Some(ref alias) = agg_expr.alias {
                self.context.add_variable(
                    alias.clone(),
                    VariableInfo {
                        name: alias.clone(),
                        data_type: LogicalType::Any,
                        is_node: false,
                        is_edge: false,
                    },
                );
            }
        }

        // Register group-by output column names so ORDER BY / HAVING
        // can reference them (e.g. "n.city" from Property(n, city)).
        for expr in &agg.group_by {
            let col_name = crate::query::planner::common::expression_to_string(expr);
            if !self.context.contains(&col_name) {
                self.context.add_variable(
                    col_name.clone(),
                    VariableInfo {
                        name: col_name,
                        data_type: LogicalType::Any,
                        is_node: false,
                        is_edge: false,
                    },
                );
            }
        }

        Ok(())
    }
}

impl Default for Binder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{BinaryOp, FilterOp};

    #[test]
    fn test_bind_simple_scan() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(ctx.contains("n"));
        assert!(ctx.get("n").unwrap().is_node);
    }

    #[test]
    fn test_bind_undefined_variable() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("undefined".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Undefined variable"));
    }

    #[test]
    fn test_bind_property_access() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        assert!(result.is_ok());
    }

    #[test]
    fn test_bind_filter_with_undefined_variable() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "m".to_string(), // undefined!
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(
                        grafeo_common::types::Value::Int64(30),
                    )),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                pushdown_hint: None,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Undefined variable 'm'"));
    }

    #[test]
    fn test_bind_expand() {
        use crate::query::plan::{ExpandDirection, ExpandOp, PathMode};

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                quantified: false,
                from_variable: "a".to_string(),
                to_variable: "b".to_string(),
                edge_variable: Some("e".to_string()),
                direction: ExpandDirection::Outgoing,
                edge_types: vec!["KNOWS".to_string()],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(ctx.contains("a"));
        assert!(ctx.contains("b"));
        assert!(ctx.contains("e"));
        assert!(ctx.get("a").unwrap().is_node);
        assert!(ctx.get("b").unwrap().is_node);
        assert!(ctx.get("e").unwrap().is_edge);
    }

    #[test]
    fn test_bind_expand_from_undefined_variable() {
        // Tests that expanding from an undefined variable produces a clear error
        use crate::query::plan::{ExpandDirection, ExpandOp, PathMode};

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("b".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                quantified: false,
                from_variable: "undefined".to_string(), // not defined!
                to_variable: "b".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_types: vec![],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Undefined variable 'undefined'"),
            "Expected error about undefined variable, got: {}",
            err
        );
    }

    #[test]
    fn test_bind_return_with_aggregate_and_non_aggregate() {
        // Tests binding of aggregate functions alongside regular expressions
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::FunctionCall {
                        name: "count".to_string(),
                        args: vec![LogicalExpression::Variable("n".to_string())],
                        distinct: false,
                    },
                    alias: Some("cnt".to_string()),
                },
                ReturnItem {
                    expression: LogicalExpression::Literal(grafeo_common::types::Value::Int64(1)),
                    alias: Some("one".to_string()),
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        // This should succeed - count(n) with literal is valid
        assert!(result.is_ok());
    }

    #[test]
    fn test_bind_nested_property_access() {
        // Tests that nested property access on the same variable works
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "name".to_string(),
                    },
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    },
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        assert!(result.is_ok());
    }

    #[test]
    fn test_bind_binary_expression_with_undefined() {
        // Tests that binary expressions with undefined variables produce errors
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Add,
                    right: Box::new(LogicalExpression::Property {
                        variable: "m".to_string(), // undefined!
                        property: "age".to_string(),
                    }),
                },
                alias: Some("total".to_string()),
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Undefined variable 'm'")
        );
    }

    #[test]
    fn test_bind_duplicate_variable_definition() {
        // Tests behavior when the same variable is defined twice (via two NodeScans)
        // This is typically not allowed or the second shadows the first
        use crate::query::plan::{JoinOp, JoinType};

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: Some("A".to_string()),
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "m".to_string(), // different variable is fine
                    label: Some("B".to_string()),
                    input: None,
                })),
                join_type: JoinType::Inner,
                conditions: vec![],
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        // Join with different variables should work
        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert!(ctx.contains("n"));
        assert!(ctx.contains("m"));
    }

    #[test]
    fn test_bind_function_with_wrong_arity() {
        // Tests that functions with wrong number of arguments are handled
        // (behavior depends on whether binder validates arity)
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::FunctionCall {
                    name: "count".to_string(),
                    args: vec![], // count() needs an argument
                    distinct: false,
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);

        // The binder may or may not catch this - if it passes, execution will fail
        // This test documents current behavior
        // If binding fails, that's fine; if it passes, execution will handle it
        let _ = result; // We're just testing it doesn't panic
    }

    // --- Mutation operator validation ---

    #[test]
    fn test_create_edge_rejects_undefined_source() {
        use crate::query::plan::CreateEdgeOp;

        let plan = LogicalPlan::new(LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: Some("e".to_string()),
            from_variable: "ghost".to_string(), // not defined!
            to_variable: "b".to_string(),
            edge_type: "KNOWS".to_string(),
            properties: vec![],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "b".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(
            err.to_string().contains("Undefined variable 'ghost'"),
            "Should reject undefined source variable, got: {err}"
        );
    }

    #[test]
    fn test_create_edge_rejects_undefined_target() {
        use crate::query::plan::CreateEdgeOp;

        let plan = LogicalPlan::new(LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: None,
            from_variable: "a".to_string(),
            to_variable: "missing".to_string(), // not defined!
            edge_type: "KNOWS".to_string(),
            properties: vec![],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(
            err.to_string().contains("Undefined variable 'missing'"),
            "Should reject undefined target variable, got: {err}"
        );
    }

    #[test]
    fn test_create_edge_validates_property_expressions() {
        use crate::query::plan::CreateEdgeOp;

        // Source and target defined, but property references undefined variable
        let plan = LogicalPlan::new(LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: Some("e".to_string()),
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_type: "KNOWS".to_string(),
            properties: vec![(
                "since".to_string(),
                LogicalExpression::Property {
                    variable: "x".to_string(), // undefined!
                    property: "year".to_string(),
                },
            )],
            input: Box::new(LogicalOperator::Join(crate::query::plan::JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: None,
                    input: None,
                })),
                join_type: crate::query::plan::JoinType::Inner,
                conditions: vec![],
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(err.to_string().contains("Undefined variable 'x'"));
    }

    #[test]
    fn test_set_property_rejects_undefined_variable() {
        use crate::query::plan::SetPropertyOp;

        let plan = LogicalPlan::new(LogicalOperator::SetProperty(SetPropertyOp {
            variable: "ghost".to_string(),
            properties: vec![(
                "name".to_string(),
                LogicalExpression::Literal(grafeo_common::types::Value::String("Alix".into())),
            )],
            replace: false,
            is_edge: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(
            err.to_string().contains("in SET"),
            "Error should indicate SET context, got: {err}"
        );
    }

    #[test]
    fn test_delete_node_rejects_undefined_variable() {
        use crate::query::plan::DeleteNodeOp;

        let plan = LogicalPlan::new(LogicalOperator::DeleteNode(DeleteNodeOp {
            variable: "phantom".to_string(),
            detach: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(err.to_string().contains("Undefined variable 'phantom'"));
    }

    #[test]
    fn test_delete_edge_rejects_undefined_variable() {
        use crate::query::plan::DeleteEdgeOp;

        let plan = LogicalPlan::new(LogicalOperator::DeleteEdge(DeleteEdgeOp {
            variable: "gone".to_string(),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(err.to_string().contains("Undefined variable 'gone'"));
    }

    // --- WITH/Project clause ---

    #[test]
    fn test_project_alias_becomes_available_downstream() {
        use crate::query::plan::{ProjectOp, Projection};

        // WITH n.name AS person_name RETURN person_name
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("person_name".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Project(ProjectOp {
                projections: vec![Projection {
                    expression: LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "name".to_string(),
                    },
                    alias: Some("person_name".to_string()),
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                pass_through_input: false,
            })),
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(
            ctx.contains("person_name"),
            "WITH alias should be available to RETURN"
        );
    }

    #[test]
    fn test_project_rejects_undefined_expression() {
        use crate::query::plan::{ProjectOp, Projection};

        let plan = LogicalPlan::new(LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Variable("nope".to_string()),
                alias: Some("x".to_string()),
            }],
            input: Box::new(LogicalOperator::Empty),
            pass_through_input: false,
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);
        assert!(result.is_err(), "WITH on undefined variable should fail");
    }

    /// `MATCH (n) WITH n.name AS name`, then a projection of `n` or `name`.
    fn project_after_with(read: &str, pass_through_input: bool) -> LogicalPlan {
        use crate::query::plan::{ProjectOp, Projection};

        let with = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                },
                alias: Some("name".to_string()),
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
            pass_through_input,
        });
        LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable(read.to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(with),
        }))
    }

    #[test]
    fn test_project_ends_the_scope_of_what_it_leaves_out() {
        let error = Binder::new()
            .bind(&project_after_with("n", false))
            .unwrap_err();
        assert!(
            error.to_string().contains("Undefined variable 'n'"),
            "{error}"
        );
        // A pass-through projection (GQL LET) keeps its input's variables.
        let ctx = Binder::new().bind(&project_after_with("n", true)).unwrap();
        assert!(ctx.contains("n") && ctx.contains("name"));
    }

    #[test]
    fn test_an_unaliased_projection_passes_on_its_column_only() {
        use crate::query::plan::{ProjectOp, Projection};

        // The column of an unaliased `n.name` is named after it; `n` is gone.
        let plan = LogicalPlan::new(LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                },
                alias: None,
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
            pass_through_input: false,
        }));
        let ctx = Binder::new().bind(&plan).unwrap();
        assert_eq!(ctx.variable_names(), ["n.name"]);
    }

    #[test]
    fn test_a_projection_in_a_join_input_keeps_the_other_side() {
        use crate::query::plan::{JoinOp, JoinType, ProjectOp, Projection};

        // The right input projects `b` away; the join's rows still hold `a`.
        let scan = |variable: &str| {
            LogicalOperator::NodeScan(NodeScanOp {
                variable: variable.to_string(),
                label: None,
                input: None,
            })
        };
        let right = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Variable("b".to_string()),
                alias: Some("c".to_string()),
            }],
            input: Box::new(scan("b")),
            pass_through_input: false,
        });
        let plan = LogicalPlan::new(LogicalOperator::Join(JoinOp {
            left: Box::new(scan("a")),
            right: Box::new(right),
            join_type: JoinType::Cross,
            conditions: vec![],
        }));
        let ctx = Binder::new().bind(&plan).unwrap();
        assert_eq!(ctx.variable_names(), ["a", "c"]);
    }

    // --- UNWIND ---

    #[test]
    fn test_unwind_adds_element_variable() {
        use crate::query::plan::UnwindOp;

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("item".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Unwind(UnwindOp {
                expression: LogicalExpression::List(vec![
                    LogicalExpression::Literal(grafeo_common::types::Value::Int64(1)),
                    LogicalExpression::Literal(grafeo_common::types::Value::Int64(2)),
                ]),
                variable: "item".to_string(),
                ordinality_var: None,
                offset_var: None,
                input: Box::new(LogicalOperator::Empty),
            })),
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("item"), "UNWIND variable should be in scope");
        let info = ctx.get("item").unwrap();
        assert!(
            !info.is_node && !info.is_edge,
            "UNWIND variable is not a graph element"
        );
    }

    // --- MERGE ---

    #[test]
    fn test_merge_adds_variable_and_validates_properties() {
        use crate::query::plan::MergeOp;

        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("m".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Merge(MergeOp {
                variable: "m".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: vec![(
                    "name".to_string(),
                    LogicalExpression::Literal(grafeo_common::types::Value::String("Alix".into())),
                )],
                on_create: vec![(
                    "created".to_string(),
                    LogicalExpression::Literal(grafeo_common::types::Value::Bool(true)),
                )],
                on_match: vec![(
                    "updated".to_string(),
                    LogicalExpression::Literal(grafeo_common::types::Value::Bool(true)),
                )],
                on_create_labels: Vec::new(),
                on_match_labels: Vec::new(),
                input: Box::new(LogicalOperator::Empty),
            })),
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("m"));
        assert!(
            ctx.get("m").unwrap().is_node,
            "MERGE variable should be a node"
        );
    }

    #[test]
    fn test_merge_rejects_undefined_in_on_create() {
        use crate::query::plan::MergeOp;

        let plan = LogicalPlan::new(LogicalOperator::Merge(MergeOp {
            variable: "m".to_string(),
            labels: vec![],
            match_properties: vec![],
            on_create: vec![(
                "name".to_string(),
                LogicalExpression::Property {
                    variable: "other".to_string(), // undefined!
                    property: "name".to_string(),
                },
            )],
            on_match: vec![],
            on_create_labels: Vec::new(),
            on_match_labels: Vec::new(),
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);
        assert!(
            result.is_err(),
            "ON CREATE referencing undefined variable should fail"
        );
    }

    // GrafeoDB/grafeo#317: the MERGE variable must be in scope inside
    // ON CREATE / ON MATCH SET. Before the fix the binder added the variable
    // *after* validating these clauses, causing `n.x`-style references inside
    // them to be rejected as undefined.

    #[test]
    fn test_merge_on_create_can_reference_merge_variable() {
        use crate::query::plan::MergeOp;

        let plan = LogicalPlan::new(LogicalOperator::Merge(MergeOp {
            variable: "n".to_string(),
            labels: vec!["Item".to_string()],
            match_properties: vec![(
                "val".to_string(),
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(1)),
            )],
            on_create: vec![(
                "x".to_string(),
                LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "val".to_string(),
                },
            )],
            on_match: vec![],
            on_create_labels: Vec::new(),
            on_match_labels: Vec::new(),
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        binder
            .bind(&plan)
            .expect("MERGE variable must be in scope inside ON CREATE SET");
    }

    #[test]
    fn test_merge_on_match_can_reference_merge_variable() {
        use crate::query::plan::MergeOp;

        let plan = LogicalPlan::new(LogicalOperator::Merge(MergeOp {
            variable: "n".to_string(),
            labels: vec!["Item".to_string()],
            match_properties: vec![(
                "val".to_string(),
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(1)),
            )],
            on_create: vec![],
            on_match: vec![(
                "x".to_string(),
                LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "x".to_string(),
                },
            )],
            on_create_labels: Vec::new(),
            on_match_labels: Vec::new(),
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        binder
            .bind(&plan)
            .expect("MERGE variable must be in scope inside ON MATCH SET");
    }

    #[test]
    fn test_merge_match_properties_cannot_reference_merge_variable() {
        // ISO §15.5: the variable bound by the pattern is not in scope inside
        // the pattern's own property predicates. Match-property validation
        // must run *before* the variable is added to the binding context.
        use crate::query::plan::MergeOp;

        let plan = LogicalPlan::new(LogicalOperator::Merge(MergeOp {
            variable: "n".to_string(),
            labels: vec!["Item".to_string()],
            match_properties: vec![(
                "val".to_string(),
                LogicalExpression::Property {
                    variable: "n".to_string(), // self-reference inside the pattern: invalid
                    property: "val".to_string(),
                },
            )],
            on_create: vec![],
            on_match: vec![],
            on_create_labels: Vec::new(),
            on_match_labels: Vec::new(),
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);
        assert!(
            result.is_err(),
            "match properties must not see the MERGE variable"
        );
    }

    // --- ShortestPath ---

    #[test]
    fn test_shortest_path_rejects_undefined_source() {
        use crate::query::plan::{ExpandDirection, PathMode, PathSelection, ShortestPathOp};

        let plan = LogicalPlan::new(LogicalOperator::ShortestPath(ShortestPathOp {
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "b".to_string(),
                label: None,
                input: None,
            })),
            source_var: "missing".to_string(), // not defined
            target_var: "b".to_string(),
            edge_types: vec![],
            direction: ExpandDirection::Both,
            path_alias: "p".to_string(),
            selection: PathSelection::Shortest(1),
            path_mode: PathMode::Walk,
            binds_target: false,
            min_hops: 1,
            max_hops: None,
            edge_variable: None,
            quantified: true,
            edge_condition: None,
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(
            err.to_string().contains("source in shortestPath"),
            "Error should mention shortestPath source context, got: {err}"
        );
    }

    #[test]
    fn test_shortest_path_adds_path_and_length_variables() {
        use crate::query::plan::{
            ExpandDirection, JoinOp, JoinType, PathMode, PathSelection, ShortestPathOp,
        };

        let plan = LogicalPlan::new(LogicalOperator::ShortestPath(ShortestPathOp {
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: None,
                    input: None,
                })),
                join_type: JoinType::Cross,
                conditions: vec![],
            })),
            source_var: "a".to_string(),
            target_var: "b".to_string(),
            edge_types: vec!["ROAD".to_string()],
            direction: ExpandDirection::Outgoing,
            path_alias: "p".to_string(),
            selection: PathSelection::Shortest(1),
            path_mode: PathMode::Walk,
            binds_target: false,
            min_hops: 1,
            max_hops: None,
            edge_variable: None,
            quantified: true,
            edge_condition: None,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("p"), "Path alias should be bound");
        assert!(
            ctx.contains("_path_length_p"),
            "Path length variable should be auto-created"
        );
    }

    // --- Expression validation edge cases ---

    #[test]
    fn test_case_expression_validates_all_branches() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Case {
                    operand: None,
                    when_clauses: vec![
                        (
                            LogicalExpression::Binary {
                                left: Box::new(LogicalExpression::Property {
                                    variable: "n".to_string(),
                                    property: "age".to_string(),
                                }),
                                op: BinaryOp::Gt,
                                right: Box::new(LogicalExpression::Literal(
                                    grafeo_common::types::Value::Int64(18),
                                )),
                            },
                            LogicalExpression::Literal(grafeo_common::types::Value::String(
                                "adult".into(),
                            )),
                        ),
                        (
                            // This branch references undefined variable
                            LogicalExpression::Property {
                                variable: "ghost".to_string(),
                                property: "flag".to_string(),
                            },
                            LogicalExpression::Literal(grafeo_common::types::Value::String(
                                "flagged".into(),
                            )),
                        ),
                    ],
                    else_clause: Some(Box::new(LogicalExpression::Literal(
                        grafeo_common::types::Value::String("other".into()),
                    ))),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(
            err.to_string().contains("ghost"),
            "CASE should validate all when-clause conditions"
        );
    }

    #[test]
    fn test_case_expression_validates_else_clause() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Case {
                    operand: None,
                    when_clauses: vec![(
                        LogicalExpression::Literal(grafeo_common::types::Value::Bool(true)),
                        LogicalExpression::Literal(grafeo_common::types::Value::Int64(1)),
                    )],
                    else_clause: Some(Box::new(LogicalExpression::Property {
                        variable: "missing".to_string(),
                        property: "x".to_string(),
                    })),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(
            err.to_string().contains("missing"),
            "CASE ELSE should validate its expression too"
        );
    }

    #[test]
    fn test_slice_access_validates_expressions() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::SliceAccess {
                    base: Box::new(LogicalExpression::Variable("n".to_string())),
                    start: Some(Box::new(LogicalExpression::Variable(
                        "undefined_start".to_string(),
                    ))),
                    end: None,
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(err.to_string().contains("undefined_start"));
    }

    #[test]
    fn test_list_comprehension_validates_list_source() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::ListComprehension {
                    variable: "x".to_string(),
                    list_expr: Box::new(LogicalExpression::Variable("not_defined".to_string())),
                    filter_expr: None,
                    map_expr: Box::new(LogicalExpression::Variable("x".to_string())),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(
            err.to_string().contains("not_defined"),
            "List comprehension should validate source list expression"
        );
    }

    #[test]
    fn test_labels_type_id_reject_undefined() {
        // labels(x) where x is not defined
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Labels("x".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        assert!(
            binder.bind(&plan).is_err(),
            "labels(x) on undefined x should fail"
        );

        // type(e) where e is not defined
        let plan2 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Type("e".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder2 = Binder::new();
        assert!(
            binder2.bind(&plan2).is_err(),
            "type(e) on undefined e should fail"
        );

        // id(n) where n is not defined
        let plan3 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Id("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder3 = Binder::new();
        assert!(
            binder3.bind(&plan3).is_err(),
            "id(n) on undefined n should fail"
        );
    }

    #[test]
    fn test_expand_rejects_non_node_source() {
        use crate::query::plan::{ExpandDirection, ExpandOp, PathMode, UnwindOp};

        // UNWIND [1,2] AS x  -- x is not a node
        // MATCH (x)-[:E]->(b)  -- should fail: x isn't a node
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("b".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Expand(ExpandOp {
                quantified: false,
                from_variable: "x".to_string(),
                to_variable: "b".to_string(),
                edge_variable: None,
                direction: ExpandDirection::Outgoing,
                edge_types: vec![],
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::Unwind(UnwindOp {
                    expression: LogicalExpression::List(vec![]),
                    variable: "x".to_string(),
                    ordinality_var: None,
                    offset_var: None,
                    input: Box::new(LogicalOperator::Empty),
                })),
                path_alias: None,
                path_mode: PathMode::Walk,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(
            err.to_string().contains("not a node"),
            "Expanding from non-node should fail, got: {err}"
        );
    }

    #[test]
    fn test_add_label_rejects_undefined_variable() {
        use crate::query::plan::AddLabelOp;

        let plan = LogicalPlan::new(LogicalOperator::AddLabel(AddLabelOp {
            variable: "missing".to_string(),
            labels: vec!["Admin".to_string()],
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(err.to_string().contains("SET labels"));
    }

    #[test]
    fn test_remove_label_rejects_undefined_variable() {
        use crate::query::plan::RemoveLabelOp;

        let plan = LogicalPlan::new(LogicalOperator::RemoveLabel(RemoveLabelOp {
            variable: "missing".to_string(),
            labels: vec!["Admin".to_string()],
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(err.to_string().contains("REMOVE labels"));
    }

    #[test]
    fn test_sort_validates_key_expressions() {
        use crate::query::plan::{SortKey, SortOp, SortOrder};

        let plan = LogicalPlan::new(LogicalOperator::Sort(SortOp {
            keys: vec![SortKey {
                expression: LogicalExpression::Property {
                    variable: "missing".to_string(),
                    property: "name".to_string(),
                },
                order: SortOrder::Ascending,
                nulls: None,
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        assert!(
            binder.bind(&plan).is_err(),
            "ORDER BY on undefined variable should fail"
        );
    }

    #[test]
    fn test_create_node_adds_variable_before_property_validation() {
        use crate::query::plan::CreateNodeOp;

        // CREATE (n:Person {friend: n.name}) - referencing the node being created
        // The variable should be available for property expressions (self-reference)
        let plan = LogicalPlan::new(LogicalOperator::CreateNode(CreateNodeOp {
            variable: "n".to_string(),
            labels: vec!["Person".to_string()],
            properties: vec![(
                "self_ref".to_string(),
                LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                },
            )],
            input: None,
        }));

        let mut binder = Binder::new();
        // This should succeed because CreateNode adds the variable before validating properties
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.get("n").unwrap().is_node);
    }

    #[test]
    fn test_undefined_variable_suggests_similar() {
        // 'person' is defined, user types 'persn' - should get a suggestion
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("persn".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "person".to_string(),
                label: None,
                input: None,
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        let msg = err.to_string();
        // The error should contain the variable name at minimum
        assert!(
            msg.contains("persn"),
            "Error should mention the undefined variable"
        );
    }

    #[test]
    fn an_undefined_variable_named_like_a_generated_one_is_undefined() {
        // A generated name never collides with one the statement spells, so
        // `_anon_42` that nothing binds is the user's undefined variable
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("_anon_42".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        let message = binder.bind(&plan).unwrap_err().to_string();
        assert!(
            message.contains("Undefined variable '_anon_42'"),
            "{message}"
        );
    }

    #[test]
    fn test_map_expression_validates_values() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Map(vec![(
                    "key".to_string(),
                    LogicalExpression::Variable("undefined".to_string()),
                )]),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        assert!(
            binder.bind(&plan).is_err(),
            "Map values should be validated"
        );
    }

    #[test]
    fn test_vector_scan_validates_query_vector() {
        use crate::query::plan::VectorScanOp;

        let plan = LogicalPlan::new(LogicalOperator::VectorScan(VectorScanOp {
            variable: "result".to_string(),
            index_name: None,
            property: "embedding".to_string(),
            label: Some("Doc".to_string()),
            query_vector: LogicalExpression::Variable("undefined_vec".to_string()),
            k: Some(10),
            metric: None,
            min_similarity: None,
            max_distance: None,
            input: None,
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        assert!(err.to_string().contains("undefined_vec"));
    }

    /// UNWIND with ORDINALITY and OFFSET binds three variables: the element
    /// (Any), the 1-based ORDINALITY index (Int64), and the 0-based OFFSET
    /// index (Int64).
    #[test]
    fn test_bind_unwind_ordinality_and_offset() {
        use crate::query::plan::UnwindOp;

        let plan = LogicalPlan::new(LogicalOperator::Unwind(UnwindOp {
            expression: LogicalExpression::List(vec![
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(1)),
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(2)),
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(3)),
            ]),
            variable: "x".to_string(),
            ordinality_var: Some("i".to_string()),
            offset_var: Some("j".to_string()),
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();

        assert!(ctx.contains("x"));
        assert!(ctx.contains("i"));
        assert!(ctx.contains("j"));
        // ORDINALITY and OFFSET are both typed Int64.
        assert_eq!(ctx.get("i").unwrap().data_type, LogicalType::Int64);
        assert_eq!(ctx.get("j").unwrap().data_type, LogicalType::Int64);
        // Element variable is Any (could be any list-element type).
        assert_eq!(ctx.get("x").unwrap().data_type, LogicalType::Any);
        // None of the three is a node or edge.
        for v in ["x", "i", "j"] {
            let info = ctx.get(v).unwrap();
            assert!(!info.is_node);
            assert!(!info.is_edge);
        }
    }

    /// MERGE on a relationship pattern validates its source/target variables
    /// and registers the edge variable. Match-property expressions are validated.
    #[test]
    fn test_bind_merge_relationship_properties() {
        use crate::query::plan::{JoinOp, JoinType, MergeRelationshipOp};

        // Build a plan with two bound nodes (a, b) via a cross-join NodeScans,
        // then MERGE a WORKS relationship between them with a `start` property.
        let plan = LogicalPlan::new(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable: "r".to_string(),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            edge_type: "WORKS".to_string(),
            match_properties: vec![(
                "start".to_string(),
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(2026)),
            )],
            on_create: vec![],
            on_match: vec![],
            undirected: false,
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: Some("Company".to_string()),
                    input: None,
                })),
                join_type: JoinType::Cross,
                conditions: vec![],
            })),
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("a"));
        assert!(ctx.contains("b"));
        assert!(ctx.contains("r"));
        let rel = ctx.get("r").unwrap();
        assert!(rel.is_edge);
        assert!(!rel.is_node);
        assert_eq!(rel.data_type, LogicalType::Edge);

        // Referencing an undefined variable in a match-property should fail.
        let bad_plan = LogicalPlan::new(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable: "r".to_string(),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            edge_type: "WORKS".to_string(),
            match_properties: vec![(
                "start".to_string(),
                LogicalExpression::Property {
                    variable: "ghost".to_string(),
                    property: "year".to_string(),
                },
            )],
            on_create: vec![],
            on_match: vec![],
            undirected: false,
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: None,
                    input: None,
                })),
                join_type: JoinType::Cross,
                conditions: vec![],
            })),
        }));
        let mut binder2 = Binder::new();
        let err = binder2.bind(&bad_plan).unwrap_err();
        assert!(err.to_string().contains("Undefined variable 'ghost'"));
    }

    /// Undefined variable 'xx' when only 'x' is defined produces a
    /// "Did you mean 'x'?" suggestion in the error message.
    #[test]
    fn test_undefined_variable_suggestion() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("xx".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Unwind(crate::query::plan::UnwindOp {
                expression: LogicalExpression::List(vec![LogicalExpression::Literal(
                    grafeo_common::types::Value::Int64(1),
                )]),
                variable: "x".to_string(),
                ordinality_var: None,
                offset_var: None,
                input: Box::new(LogicalOperator::Empty),
            })),
        }));

        let mut binder = Binder::new();
        let err = binder.bind(&plan).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Undefined variable 'xx'"), "got: {msg}");
        assert!(
            msg.contains("Did you mean 'x'?"),
            "should suggest the similar variable 'x', got: {msg}"
        );
    }

    /// Aggregate registers both aggregate aliases (e.g. `cnt`) and the
    /// stringified group-by expression ("n.age") so downstream clauses
    /// (ORDER BY / HAVING / RETURN) can reference them.
    #[test]
    fn test_bind_aggregate_group_by_alias() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};

        // Aggregate groups by n.age, plus a COUNT(*) aliased as "cnt".
        // Downstream ORDER BY would reference either "n.age" (group column)
        // or "cnt" (aggregate alias), so both must end up in the binding
        // context after binding the Aggregate.
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("cnt".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "age".to_string(),
                }],
                aggregates: vec![AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: None,
                    expression2: None,
                    distinct: false,
                    alias: Some("cnt".to_string()),
                    percentile: None,
                    separator: None,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                having: None,
            })),
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        // Aggregate alias is registered as a variable.
        assert!(ctx.contains("cnt"), "aggregate alias 'cnt' should be bound");
        // The group-by output column "n.age" is registered so RETURN /
        // ORDER BY / HAVING can reference it.
        assert!(
            ctx.contains("n.age"),
            "group-by output column 'n.age' should be registered, got: {:?}",
            ctx.variable_names()
        );
    }

    // ========================================================================
    // Coverage tests: BindingContext helpers
    // ========================================================================

    #[test]
    fn test_binding_context_len_and_is_empty_and_remove() {
        let mut ctx = BindingContext::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.len(), 0);

        ctx.add_variable(
            "vincent".to_string(),
            VariableInfo {
                name: "vincent".to_string(),
                data_type: LogicalType::Node,
                is_node: true,
                is_edge: false,
            },
        );
        ctx.add_variable(
            "jules".to_string(),
            VariableInfo {
                name: "jules".to_string(),
                data_type: LogicalType::Node,
                is_node: true,
                is_edge: false,
            },
        );
        assert!(!ctx.is_empty());
        assert_eq!(ctx.len(), 2);
        assert_eq!(
            ctx.variable_names(),
            vec!["vincent".to_string(), "jules".to_string()]
        );

        ctx.remove_variable("vincent");
        assert_eq!(ctx.len(), 1);
        assert!(!ctx.contains("vincent"));
        assert!(ctx.contains("jules"));

        // Removing nonexistent is a no-op.
        ctx.remove_variable("nonexistent");
        assert_eq!(ctx.len(), 1);
    }

    #[test]
    fn test_binder_default_impl() {
        let binder = Binder::default();
        assert!(binder.context.is_empty());
    }

    // ========================================================================
    // Coverage tests: simple leaf / passthrough operators
    // ========================================================================

    #[test]
    fn test_bind_empty_operator_alone() {
        let plan = LogicalPlan::new(LogicalOperator::Empty);
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_bind_limit_and_skip_delegate_to_input() {
        use crate::query::plan::{CountExpr, LimitOp, SkipOp};

        let plan = LogicalPlan::new(LogicalOperator::Limit(LimitOp {
            count: CountExpr::Literal(5),
            input: Box::new(LogicalOperator::Skip(SkipOp {
                count: CountExpr::Literal(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "mia".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
            })),
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("mia"));
    }

    #[test]
    fn test_bind_distinct_delegates_to_input() {
        use crate::query::plan::DistinctOp;

        let plan = LogicalPlan::new(LogicalOperator::Distinct(DistinctOp {
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "butch".to_string(),
                label: None,
                input: None,
            })),
            columns: None,
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("butch"));
    }

    #[test]
    fn test_bind_edge_scan_with_input_binds_edge_variable() {
        use crate::query::plan::EdgeScanOp;

        let plan = LogicalPlan::new(LogicalOperator::EdgeScan(EdgeScanOp {
            variable: "e".to_string(),
            edge_types: vec!["KNOWS".to_string()],
            input: Some(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "django".to_string(),
                label: None,
                input: None,
            }))),
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("django"));
        let edge_info = ctx.get("e").expect("edge variable bound");
        assert!(edge_info.is_edge);
        assert!(!edge_info.is_node);
    }

    #[test]
    fn test_bind_edge_scan_without_input() {
        use crate::query::plan::EdgeScanOp;

        let plan = LogicalPlan::new(LogicalOperator::EdgeScan(EdgeScanOp {
            variable: "rel".to_string(),
            edge_types: vec![],
            input: None,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.get("rel").unwrap().is_edge);
    }

    // ========================================================================
    // Coverage tests: SPARQL / RDF operators
    // ========================================================================

    #[test]
    fn test_bind_triple_scan_registers_all_variable_components() {
        use crate::query::plan::{TripleComponent, TripleScanOp};

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: Some(TripleComponent::Variable("g".to_string())),
            input: None,
            dataset: None,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("s"));
        assert!(ctx.contains("p"));
        assert!(ctx.contains("o"));
        assert!(ctx.contains("g"));
    }

    #[test]
    fn test_bind_triple_scan_skips_constant_components() {
        use crate::query::plan::{TripleComponent, TripleScanOp};
        use grafeo_common::types::Value;

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Literal(Value::Int64(42)),
            graph: None,
            input: Some(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "existing".to_string(),
                label: None,
                input: None,
            }))),
            dataset: None,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        // Input was bound.
        assert!(ctx.contains("existing"));
        // No variable components registered.
        assert_eq!(ctx.len(), 1);
    }

    #[test]
    fn test_bind_triple_scan_does_not_rebind_existing_variable() {
        use crate::query::plan::{TripleComponent, TripleScanOp};

        // Use an input that already binds 's' as a node. The triple scan
        // must not clobber it with the RDF-term type.
        let plan = LogicalPlan::new(LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: Some(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "s".to_string(),
                label: None,
                input: None,
            }))),
            dataset: None,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        // 's' retains its node status because the triple scan's branch is
        // skipped once the variable is already present.
        assert!(ctx.get("s").unwrap().is_node);
        assert!(ctx.contains("p"));
        assert!(ctx.contains("o"));
    }

    #[test]
    fn test_bind_insert_triple_and_delete_triple_with_and_without_input() {
        use crate::query::plan::{DeleteTripleOp, InsertTripleOp, TripleComponent};

        // With input.
        let plan = LogicalPlan::new(LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: Some(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "hans".to_string(),
                label: None,
                input: None,
            }))),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("hans"));

        // Without input.
        let plan2 = LogicalPlan::new(LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Iri("http://example.org/o".to_string()),
            graph: None,
            input: None,
        }));
        let mut binder2 = Binder::new();
        assert!(binder2.bind(&plan2).is_ok());

        // Delete with input.
        let plan3 = LogicalPlan::new(LogicalOperator::DeleteTriple(DeleteTripleOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: Some(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "shosanna".to_string(),
                label: None,
                input: None,
            }))),
        }));
        let mut binder3 = Binder::new();
        let ctx3 = binder3.bind(&plan3).unwrap();
        assert!(ctx3.contains("shosanna"));

        // Delete without input.
        let plan4 = LogicalPlan::new(LogicalOperator::DeleteTriple(DeleteTripleOp {
            subject: TripleComponent::Iri("http://example.org/s".to_string()),
            predicate: TripleComponent::Iri("http://example.org/p".to_string()),
            object: TripleComponent::Iri("http://example.org/o".to_string()),
            graph: None,
            input: None,
        }));
        let mut binder4 = Binder::new();
        assert!(binder4.bind(&plan4).is_ok());
    }

    #[test]
    fn test_bind_modify_operator_walks_where_clause() {
        use crate::query::plan::ModifyOp;

        let plan = LogicalPlan::new(LogicalOperator::Modify(ModifyOp {
            delete_templates: vec![],
            insert_templates: vec![],
            where_clause: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "beatrix".to_string(),
                label: None,
                input: None,
            })),
            graph: None,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("beatrix"));
    }

    #[test]
    fn test_bind_construct_walks_input() {
        use crate::query::plan::ConstructOp;

        let plan = LogicalPlan::new(LogicalOperator::Construct(ConstructOp {
            templates: vec![],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "node".to_string(),
                label: None,
                input: None,
            })),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("node"));
    }

    #[test]
    fn test_bind_graph_ddl_operators_succeed_without_input() {
        use crate::query::plan::{
            AddGraphOp, ClearGraphOp, CopyGraphOp, CreateGraphOp, DropGraphOp, LoadGraphOp,
            MoveGraphOp,
        };

        // Each DDL operator should bind cleanly even without input. We cover
        // every arm of the consolidated DDL match block.
        let cases: Vec<LogicalOperator> = vec![
            LogicalOperator::ClearGraph(ClearGraphOp {
                target: grafeo_common::storage::log_record::RdfGraphTarget::Default,
                silent: false,
            }),
            LogicalOperator::CreateGraph(CreateGraphOp {
                graph: "http://example.org/g".to_string(),
                silent: false,
            }),
            LogicalOperator::DropGraph(DropGraphOp {
                target: grafeo_common::storage::log_record::RdfGraphTarget::Default,
                silent: true,
            }),
            LogicalOperator::LoadGraph(LoadGraphOp {
                source: "http://example.org/data".to_string(),
                destination: None,
                silent: false,
            }),
            LogicalOperator::CopyGraph(CopyGraphOp {
                source: None,
                destination: None,
                silent: false,
            }),
            LogicalOperator::MoveGraph(MoveGraphOp {
                source: None,
                destination: None,
                silent: false,
            }),
            LogicalOperator::AddGraph(AddGraphOp {
                source: None,
                destination: None,
                silent: false,
            }),
        ];

        for op in cases {
            let plan = LogicalPlan::new(op);
            let mut binder = Binder::new();
            assert!(binder.bind(&plan).is_ok());
        }
    }

    #[test]
    fn test_bind_create_property_graph_is_noop() {
        use crate::query::plan::CreatePropertyGraphOp;

        let plan = LogicalPlan::new(LogicalOperator::CreatePropertyGraph(
            CreatePropertyGraphOp {
                name: "social".to_string(),
                node_tables: vec![],
                edge_tables: vec![],
            },
        ));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.is_empty());
    }

    // ========================================================================
    // Coverage tests: joins, set ops, union, bind
    // ========================================================================

    #[test]
    fn test_bind_union_walks_all_inputs() {
        use crate::query::plan::UnionOp;

        let plan = LogicalPlan::new(LogicalOperator::Union(UnionOp {
            inputs: vec![
                LogicalOperator::NodeScan(NodeScanOp {
                    variable: "amsterdam".to_string(),
                    label: None,
                    input: None,
                }),
                LogicalOperator::NodeScan(NodeScanOp {
                    variable: "berlin".to_string(),
                    label: None,
                    input: None,
                }),
            ],
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("amsterdam"));
        assert!(ctx.contains("berlin"));
    }

    #[test]
    fn test_bind_left_join_with_condition_validates_it() {
        use crate::query::plan::LeftJoinOp;

        // Left and right both bind variables; a condition referencing a
        // defined variable validates, whereas an unknown variable fails.
        let ok_plan = LogicalPlan::new(LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "paris".to_string(),
                label: None,
                input: None,
            })),
            right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "prague".to_string(),
                label: None,
                input: None,
            })),
            condition: Some(LogicalExpression::Variable("paris".to_string())),
        }));
        let mut binder = Binder::new();
        assert!(binder.bind(&ok_plan).is_ok());

        // Missing condition is accepted too.
        let plan_no_cond = LogicalPlan::new(LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "paris".to_string(),
                label: None,
                input: None,
            })),
            right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "prague".to_string(),
                label: None,
                input: None,
            })),
            condition: None,
        }));
        assert!(Binder::new().bind(&plan_no_cond).is_ok());

        // Condition referencing undefined variable rejects.
        let bad_plan = LogicalPlan::new(LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "paris".to_string(),
                label: None,
                input: None,
            })),
            right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "prague".to_string(),
                label: None,
                input: None,
            })),
            condition: Some(LogicalExpression::Variable("missing".to_string())),
        }));
        assert!(Binder::new().bind(&bad_plan).is_err());
    }

    #[test]
    fn test_bind_anti_join_walks_both_sides() {
        use crate::query::plan::AntiJoinOp;

        let plan = LogicalPlan::new(LogicalOperator::AntiJoin(AntiJoinOp {
            left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "left_side".to_string(),
                label: None,
                input: None,
            })),
            right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "right_side".to_string(),
                label: None,
                input: None,
            })),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("left_side"));
        assert!(ctx.contains("right_side"));
    }

    #[test]
    fn test_bind_bind_operator_adds_variable_and_validates_expression() {
        use crate::query::plan::BindOp;

        // Bind with a defined expression reference.
        let plan = LogicalPlan::new(LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Variable("n".to_string()),
            variable: "x".to_string(),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("x"));

        // BIND expression referencing undefined variable must fail.
        let bad_plan = LogicalPlan::new(LogicalOperator::Bind(BindOp {
            expression: LogicalExpression::Variable("ghost".to_string()),
            variable: "x".to_string(),
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&bad_plan).is_err());
    }

    #[test]
    fn test_bind_except_intersect_otherwise_walk_both_sides() {
        use crate::query::plan::{ExceptOp, IntersectOp, OtherwiseOp};

        for op in [
            LogicalOperator::Except(ExceptOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "l".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "r".to_string(),
                    label: None,
                    input: None,
                })),
                all: false,
            }),
            LogicalOperator::Intersect(IntersectOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "l".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "r".to_string(),
                    label: None,
                    input: None,
                })),
                all: false,
            }),
            LogicalOperator::Otherwise(OtherwiseOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "l".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "r".to_string(),
                    label: None,
                    input: None,
                })),
            }),
        ] {
            let plan = LogicalPlan::new(op);
            let mut binder = Binder::new();
            let ctx = binder.bind(&plan).unwrap();
            assert!(ctx.contains("l"));
            assert!(ctx.contains("r"));
        }
    }

    #[test]
    fn test_bind_multi_way_join_validates_conditions() {
        use crate::query::plan::{JoinCondition, MultiWayJoinOp};

        let plan = LogicalPlan::new(LogicalOperator::MultiWayJoin(MultiWayJoinOp {
            inputs: vec![
                LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                }),
                LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: None,
                    input: None,
                }),
            ],
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
            }],
            shared_variables: vec![],
        }));
        let mut binder = Binder::new();
        assert!(binder.bind(&plan).is_ok());

        // Condition with undefined variable fails.
        let bad_plan = LogicalPlan::new(LogicalOperator::MultiWayJoin(MultiWayJoinOp {
            inputs: vec![LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: None,
                input: None,
            })],
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("nope".to_string()),
            }],
            shared_variables: vec![],
        }));
        assert!(Binder::new().bind(&bad_plan).is_err());
    }

    // ========================================================================
    // Coverage tests: Apply / parameter scan / subplan scoping
    // ========================================================================

    #[test]
    fn test_bind_parameter_scan_registers_columns() {
        use crate::query::plan::ParameterScanOp;

        let plan = LogicalPlan::new(LogicalOperator::ParameterScan(ParameterScanOp {
            columns: vec!["vincent".to_string(), "jules".to_string()],
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("vincent"));
        assert!(ctx.contains("jules"));
    }

    #[test]
    fn test_bind_apply_removes_internal_variables_from_input_and_subplan() {
        use crate::query::plan::ApplyOp;

        // Outer input: a Return that exposes only an aliased column. The
        // internal NodeScan variable "inner_scan" is bound by Return's input,
        // then scoped out by Apply because Return registers explicit outputs.
        let plan = LogicalPlan::new(LogicalOperator::Apply(ApplyOp {
            input: Box::new(LogicalOperator::Return(ReturnOp {
                items: vec![ReturnItem {
                    expression: LogicalExpression::Variable("inner_scan".to_string()),
                    alias: Some("out_col".to_string()),
                }],
                distinct: false,
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "inner_scan".to_string(),
                    label: None,
                    input: None,
                })),
            })),
            subplan: Box::new(LogicalOperator::Return(ReturnOp {
                items: vec![ReturnItem {
                    expression: LogicalExpression::Literal(grafeo_common::types::Value::Int64(1)),
                    alias: Some("sub_col".to_string()),
                }],
                distinct: false,
                input: Box::new(LogicalOperator::Empty),
            })),
            shared_variables: vec![],
            optional: false,
            unit: false,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();

        // Output of the Apply should include projected columns from both
        // sides but not the internal node-scan variable.
        assert!(ctx.contains("out_col"), "input projection exposed");
        assert!(ctx.contains("sub_col"), "subplan output registered");
        assert!(
            !ctx.contains("inner_scan"),
            "internal scan variable should be scoped out"
        );
    }

    #[test]
    fn test_bind_apply_without_explicit_projection_keeps_input_variables() {
        use crate::query::plan::ApplyOp;

        // When the outer input has no projection, the scoping branch is
        // skipped and input variables stay visible.
        let plan = LogicalPlan::new(LogicalOperator::Apply(ApplyOp {
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "outer".to_string(),
                label: None,
                input: None,
            })),
            subplan: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "inner".to_string(),
                label: None,
                input: None,
            })),
            shared_variables: vec![],
            optional: false,
            unit: false,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("outer"));
    }

    // ========================================================================
    // Coverage tests: MERGE relationship
    // ========================================================================

    #[test]
    fn test_merge_relationship_rejects_undefined_source_and_target() {
        use crate::query::plan::MergeRelationshipOp;

        // Undefined source.
        let plan = LogicalPlan::new(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable: "r".to_string(),
            source_variable: "phantom".to_string(),
            target_variable: "b".to_string(),
            edge_type: "KNOWS".to_string(),
            match_properties: vec![],
            on_create: vec![],
            on_match: vec![],
            undirected: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "b".to_string(),
                label: None,
                input: None,
            })),
        }));
        let err = Binder::new().bind(&plan).unwrap_err();
        assert!(err.to_string().contains("MERGE relationship source"));

        // Undefined target.
        let plan2 = LogicalPlan::new(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable: "r".to_string(),
            source_variable: "a".to_string(),
            target_variable: "phantom".to_string(),
            edge_type: "KNOWS".to_string(),
            match_properties: vec![],
            on_create: vec![],
            on_match: vec![],
            undirected: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: None,
                input: None,
            })),
        }));
        let err2 = Binder::new().bind(&plan2).unwrap_err();
        assert!(err2.to_string().contains("MERGE relationship target"));
    }

    #[test]
    fn test_merge_relationship_happy_path_binds_edge_variable() {
        use crate::query::plan::{JoinOp, JoinType, MergeRelationshipOp};

        let plan = LogicalPlan::new(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable: "r".to_string(),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            edge_type: "KNOWS".to_string(),
            match_properties: vec![(
                "since".to_string(),
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(2020)),
            )],
            on_create: vec![(
                "created_at".to_string(),
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(1)),
            )],
            on_match: vec![(
                "updated_at".to_string(),
                LogicalExpression::Literal(grafeo_common::types::Value::Int64(2)),
            )],
            undirected: false,
            input: Box::new(LogicalOperator::Join(JoinOp {
                left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: None,
                    input: None,
                })),
                right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "b".to_string(),
                    label: None,
                    input: None,
                })),
                join_type: JoinType::Cross,
                conditions: vec![],
            })),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        let edge = ctx.get("r").expect("edge variable bound");
        assert!(edge.is_edge);
        assert!(!edge.is_node);
    }

    // ========================================================================
    // Coverage tests: UNWIND, CallProcedure, LoadData, MapCollect, VectorJoin
    // ========================================================================

    #[test]
    fn test_unwind_ordinality_and_offset_variables() {
        use crate::query::plan::UnwindOp;

        let plan = LogicalPlan::new(LogicalOperator::Unwind(UnwindOp {
            expression: LogicalExpression::List(vec![LogicalExpression::Literal(
                grafeo_common::types::Value::Int64(1),
            )]),
            variable: "item".to_string(),
            ordinality_var: Some("ord".to_string()),
            offset_var: Some("off".to_string()),
            input: Box::new(LogicalOperator::Empty),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert_eq!(ctx.get("ord").unwrap().data_type, LogicalType::Int64);
        assert_eq!(ctx.get("off").unwrap().data_type, LogicalType::Int64);
    }

    #[test]
    fn test_call_procedure_with_and_without_yields() {
        use crate::query::plan::{CallProcedureOp, ProcedureYield};

        // With yield items including an alias.
        let plan = LogicalPlan::new(LogicalOperator::CallProcedure(CallProcedureOp {
            name: vec!["grafeo".to_string(), "pagerank".to_string()],
            arguments: vec![],
            yield_items: Some(vec![
                ProcedureYield {
                    field_name: "nodeId".to_string(),
                    alias: None,
                },
                ProcedureYield {
                    field_name: "score".to_string(),
                    alias: Some("rank".to_string()),
                },
            ]),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("nodeId"));
        assert!(ctx.contains("rank"));
        assert!(!ctx.contains("score"), "aliased yield hides raw name");

        // Without yields: nothing to register.
        let plan2 = LogicalPlan::new(LogicalOperator::CallProcedure(CallProcedureOp {
            name: vec!["grafeo".to_string(), "noop".to_string()],
            arguments: vec![],
            yield_items: None,
        }));
        let mut binder2 = Binder::new();
        let ctx2 = binder2.bind(&plan2).unwrap();
        assert!(ctx2.is_empty());
    }

    #[test]
    fn test_load_data_binds_row_variable() {
        use crate::query::plan::{LoadDataFormat, LoadDataOp};

        let plan = LogicalPlan::new(LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Csv,
            with_headers: true,
            path: "/tmp/data.csv".to_string(),
            variable: "row".to_string(),
            field_terminator: None,
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("row"));
        assert_eq!(ctx.get("row").unwrap().data_type, LogicalType::Any);
    }

    #[test]
    fn test_map_collect_registers_alias() {
        use crate::query::plan::MapCollectOp;

        let plan = LogicalPlan::new(LogicalOperator::MapCollect(MapCollectOp {
            key_var: "k".to_string(),
            value_var: "v".to_string(),
            alias: "grouped".to_string(),
            input: Box::new(LogicalOperator::Empty),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("grouped"));
    }

    #[test]
    fn test_vector_join_registers_right_and_score_variables() {
        use crate::query::plan::VectorJoinOp;

        let plan = LogicalPlan::new(LogicalOperator::VectorJoin(VectorJoinOp {
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: None,
                input: None,
            })),
            left_vector_variable: None,
            left_property: None,
            query_vector: LogicalExpression::Literal(grafeo_common::types::Value::Int64(0)),
            right_variable: "b".to_string(),
            right_property: "embedding".to_string(),
            right_label: None,
            index_name: None,
            k: 5,
            metric: None,
            min_similarity: None,
            max_distance: None,
            score_variable: Some("score".to_string()),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.get("b").unwrap().is_node);
        assert_eq!(ctx.get("score").unwrap().data_type, LogicalType::Float64);

        // VectorJoin with undefined query_vector reference should fail.
        let bad_plan = LogicalPlan::new(LogicalOperator::VectorJoin(VectorJoinOp {
            input: Box::new(LogicalOperator::Empty),
            left_vector_variable: None,
            left_property: None,
            query_vector: LogicalExpression::Variable("undef_vec".to_string()),
            right_variable: "b".to_string(),
            right_property: "embedding".to_string(),
            right_label: None,
            index_name: None,
            k: 5,
            metric: None,
            min_similarity: None,
            max_distance: None,
            score_variable: None,
        }));
        assert!(Binder::new().bind(&bad_plan).is_err());
    }

    // ========================================================================
    // Coverage tests: expression validation edge cases
    // ========================================================================

    #[test]
    fn test_expression_validation_unary_index_map_projection() {
        use crate::query::plan::{MapProjectionEntry, UnaryOp};

        // Unary with undefined operand fails.
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(LogicalExpression::Variable("ghost".to_string())),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan).is_err());

        // IndexAccess validates both base and index.
        let plan2 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::IndexAccess {
                    base: Box::new(LogicalExpression::Variable("xs".to_string())),
                    index: Box::new(LogicalExpression::Variable("idx_ghost".to_string())),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "xs".to_string(),
                label: None,
                input: None,
            })),
        }));
        let err = Binder::new().bind(&plan2).unwrap_err();
        assert!(err.to_string().contains("idx_ghost"));

        // MapProjection validates literal entries.
        let plan3 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::MapProjection {
                    base: "n".to_string(),
                    entries: vec![
                        MapProjectionEntry::PropertySelector("name".to_string()),
                        MapProjectionEntry::AllProperties,
                        MapProjectionEntry::LiteralEntry(
                            "extra".to_string(),
                            LogicalExpression::Variable("missing".to_string()),
                        ),
                    ],
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
        }));
        assert!(Binder::new().bind(&plan3).is_err());

        // MapProjection base undefined fails.
        let plan4 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::MapProjection {
                    base: "unknown".to_string(),
                    entries: vec![MapProjectionEntry::AllProperties],
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        let err4 = Binder::new().bind(&plan4).unwrap_err();
        assert!(err4.to_string().contains("map projection"));
    }

    #[test]
    fn test_expression_validation_list_and_parameter_and_subquery() {
        // Parameter is always valid.
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Parameter("p".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan).is_ok());

        // List with undefined element.
        let plan2 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::List(vec![LogicalExpression::Variable(
                    "no_such".to_string(),
                )]),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan2).is_err());

        // Subquery expressions are accepted without recursive binding.
        let plan3 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::ExistsSubquery(Box::new(LogicalOperator::Empty)),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan3).is_ok());

        let plan4 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::CountSubquery(Box::new(LogicalOperator::Empty)),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan4).is_ok());

        let plan5 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::ValueSubquery(Box::new(LogicalOperator::Empty)),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan5).is_ok());
    }

    #[test]
    fn test_expression_validation_list_predicate_and_pattern_comprehension() {
        use crate::query::plan::ListPredicateKind;

        // ListPredicate validates source list.
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::ListPredicate {
                    kind: ListPredicateKind::All,
                    variable: "x".to_string(),
                    list_expr: Box::new(LogicalExpression::Variable("missing_list".to_string())),
                    predicate: Box::new(LogicalExpression::Literal(
                        grafeo_common::types::Value::Bool(true),
                    )),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan).is_err());

        // PatternComprehension binds its subplan and then validates projection.
        let plan2 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::PatternComprehension {
                    subplan: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "f".to_string(),
                        label: None,
                        input: None,
                    })),
                    projection: Box::new(LogicalExpression::Property {
                        variable: "f".to_string(),
                        property: "name".to_string(),
                    }),
                },
                alias: Some("names".to_string()),
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan2).is_ok());

        // Pattern comprehension with a bad projection.
        let plan3 = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::PatternComprehension {
                    subplan: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "f".to_string(),
                        label: None,
                        input: None,
                    })),
                    projection: Box::new(LogicalExpression::Variable("ghost".to_string())),
                },
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));
        assert!(Binder::new().bind(&plan3).is_err());
    }

    #[test]
    fn test_expression_validation_reduce_adds_and_removes_locals() {
        // reduce() adds accumulator and iteration variable during validation.
        // After the call, they should not remain in outer scope.
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Reduce {
                    accumulator: "acc".to_string(),
                    initial: Box::new(LogicalExpression::Literal(
                        grafeo_common::types::Value::Int64(0),
                    )),
                    variable: "x".to_string(),
                    list: Box::new(LogicalExpression::Variable("xs".to_string())),
                    expression: Box::new(LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Variable("acc".to_string())),
                        op: crate::query::plan::BinaryOp::Add,
                        right: Box::new(LogicalExpression::Variable("x".to_string())),
                    }),
                },
                alias: Some("sum".to_string()),
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Project(crate::query::plan::ProjectOp {
                projections: vec![crate::query::plan::Projection {
                    expression: LogicalExpression::List(vec![LogicalExpression::Literal(
                        grafeo_common::types::Value::Int64(1),
                    )]),
                    alias: Some("xs".to_string()),
                }],
                input: Box::new(LogicalOperator::Empty),
                pass_through_input: false,
            })),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        // 'acc' and 'x' must not leak to outer scope.
        assert!(!ctx.contains("acc"));
        assert!(!ctx.contains("x"));
        // Alias 'sum' is registered.
        assert!(ctx.contains("sum"));
    }

    #[test]
    fn test_expression_validation_reduce_preserves_preexisting_locals() {
        // If accumulator or variable name already exists in scope, the reduce
        // branch should leave them intact when it finishes.
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Reduce {
                    accumulator: "acc".to_string(),
                    initial: Box::new(LogicalExpression::Literal(
                        grafeo_common::types::Value::Int64(0),
                    )),
                    variable: "x".to_string(),
                    list: Box::new(LogicalExpression::Variable("acc".to_string())),
                    expression: Box::new(LogicalExpression::Variable("acc".to_string())),
                },
                alias: Some("r".to_string()),
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Project(crate::query::plan::ProjectOp {
                projections: vec![
                    crate::query::plan::Projection {
                        expression: LogicalExpression::List(vec![]),
                        alias: Some("acc".to_string()),
                    },
                    crate::query::plan::Projection {
                        expression: LogicalExpression::Literal(grafeo_common::types::Value::Int64(
                            0,
                        )),
                        alias: Some("x".to_string()),
                    },
                ],
                input: Box::new(LogicalOperator::Empty),
                pass_through_input: false,
            })),
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        // Pre-existing bindings survive.
        assert!(ctx.contains("acc"));
        assert!(ctx.contains("x"));
    }

    // ========================================================================
    // Coverage tests: type inference and entity status propagation
    // ========================================================================

    #[test]
    fn test_infer_expression_type_for_literals_and_functions() {
        use crate::query::plan::{ProjectOp, Projection};
        use grafeo_common::types::Value;

        // Wrap a series of projections with typed aliases.
        let projections = vec![
            Projection {
                expression: LogicalExpression::Literal(Value::Bool(true)),
                alias: Some("b".to_string()),
            },
            Projection {
                expression: LogicalExpression::Literal(Value::Int64(1)),
                alias: Some("i".to_string()),
            },
            Projection {
                expression: LogicalExpression::Literal(Value::Float64(1.5)),
                alias: Some("f".to_string()),
            },
            Projection {
                expression: LogicalExpression::Literal(Value::String("s".into())),
                alias: Some("s".to_string()),
            },
            Projection {
                expression: LogicalExpression::Literal(Value::List(std::sync::Arc::from(Vec::<
                    Value,
                >::new(
                )))),
                alias: Some("l".to_string()),
            },
            Projection {
                expression: LogicalExpression::Literal(Value::Map(std::sync::Arc::new(
                    std::collections::BTreeMap::new(),
                ))),
                alias: Some("m".to_string()),
            },
            Projection {
                expression: LogicalExpression::Literal(Value::Null),
                alias: Some("n".to_string()),
            },
            Projection {
                expression: LogicalExpression::FunctionCall {
                    name: "count".to_string(),
                    args: vec![LogicalExpression::Literal(Value::Int64(1))],
                    distinct: false,
                },
                alias: Some("cnt".to_string()),
            },
            Projection {
                expression: LogicalExpression::FunctionCall {
                    name: "AVG".to_string(),
                    args: vec![LogicalExpression::Literal(Value::Int64(1))],
                    distinct: false,
                },
                alias: Some("avg_val".to_string()),
            },
            Projection {
                expression: LogicalExpression::FunctionCall {
                    name: "type".to_string(),
                    args: vec![],
                    distinct: false,
                },
                alias: Some("tname".to_string()),
            },
            Projection {
                expression: LogicalExpression::FunctionCall {
                    name: "labels".to_string(),
                    args: vec![],
                    distinct: false,
                },
                alias: Some("lbls".to_string()),
            },
            Projection {
                expression: LogicalExpression::FunctionCall {
                    name: "unknown_fn".to_string(),
                    args: vec![],
                    distinct: false,
                },
                alias: Some("u".to_string()),
            },
            // List and Map expressions are Any.
            Projection {
                expression: LogicalExpression::List(vec![]),
                alias: Some("lit_list".to_string()),
            },
            Projection {
                expression: LogicalExpression::Map(vec![]),
                alias: Some("lit_map".to_string()),
            },
            // Unary / Binary / Property fall through to Any.
            Projection {
                expression: LogicalExpression::Unary {
                    op: crate::query::plan::UnaryOp::Not,
                    operand: Box::new(LogicalExpression::Literal(Value::Bool(true))),
                },
                alias: Some("unary_ty".to_string()),
            },
            Projection {
                expression: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Literal(Value::Int64(1))),
                    op: crate::query::plan::BinaryOp::Add,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(2))),
                },
                alias: Some("bin_ty".to_string()),
            },
        ];

        let plan = LogicalPlan::new(LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(LogicalOperator::Empty),
            pass_through_input: false,
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();

        assert_eq!(ctx.get("b").unwrap().data_type, LogicalType::Bool);
        assert_eq!(ctx.get("i").unwrap().data_type, LogicalType::Int64);
        assert_eq!(ctx.get("f").unwrap().data_type, LogicalType::Float64);
        assert_eq!(ctx.get("s").unwrap().data_type, LogicalType::String);
        assert_eq!(ctx.get("l").unwrap().data_type, LogicalType::Any);
        assert_eq!(ctx.get("m").unwrap().data_type, LogicalType::Any);
        assert_eq!(ctx.get("n").unwrap().data_type, LogicalType::Any);
        assert_eq!(ctx.get("cnt").unwrap().data_type, LogicalType::Int64);
        assert_eq!(ctx.get("avg_val").unwrap().data_type, LogicalType::Float64);
        assert_eq!(ctx.get("tname").unwrap().data_type, LogicalType::String);
        assert_eq!(ctx.get("lbls").unwrap().data_type, LogicalType::Any);
        assert_eq!(ctx.get("u").unwrap().data_type, LogicalType::Any);
        assert_eq!(ctx.get("lit_list").unwrap().data_type, LogicalType::Any);
        assert_eq!(ctx.get("lit_map").unwrap().data_type, LogicalType::Any);
        assert_eq!(ctx.get("unary_ty").unwrap().data_type, LogicalType::Any);
        assert_eq!(ctx.get("bin_ty").unwrap().data_type, LogicalType::Any);
    }

    #[test]
    fn test_infer_entity_status_for_case_projection() {
        use crate::query::plan::{ProjectOp, Projection};

        // Case with both branches selecting node variables should propagate
        // node status. The planner uses this for optional()/union() rewrites.
        let plan = LogicalPlan::new(LogicalOperator::Project(ProjectOp {
            projections: vec![
                Projection {
                    expression: LogicalExpression::Variable("n".to_string()),
                    alias: Some("original".to_string()),
                },
                Projection {
                    expression: LogicalExpression::Case {
                        operand: None,
                        when_clauses: vec![(
                            LogicalExpression::Literal(grafeo_common::types::Value::Bool(true)),
                            LogicalExpression::Variable("n".to_string()),
                        )],
                        else_clause: Some(Box::new(LogicalExpression::Variable("n".to_string()))),
                    },
                    alias: Some("case_node".to_string()),
                },
                Projection {
                    // Empty Case (should produce neither node nor edge).
                    expression: LogicalExpression::Case {
                        operand: None,
                        when_clauses: vec![],
                        else_clause: None,
                    },
                    alias: Some("empty_case".to_string()),
                },
                Projection {
                    // Case mixing node var and literal should lose node-ness.
                    expression: LogicalExpression::Case {
                        operand: None,
                        when_clauses: vec![(
                            LogicalExpression::Literal(grafeo_common::types::Value::Bool(true)),
                            LogicalExpression::Variable("n".to_string()),
                        )],
                        else_clause: Some(Box::new(LogicalExpression::Literal(
                            grafeo_common::types::Value::Int64(0),
                        ))),
                    },
                    alias: Some("mixed_case".to_string()),
                },
            ],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
            pass_through_input: false,
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();

        assert!(ctx.get("original").unwrap().is_node);
        assert!(ctx.get("case_node").unwrap().is_node);
        let empty = ctx.get("empty_case").unwrap();
        assert!(!empty.is_node && !empty.is_edge);
        let mixed = ctx.get("mixed_case").unwrap();
        assert!(
            !mixed.is_node && !mixed.is_edge,
            "mixed Case branches lose entity status"
        );
    }

    // ========================================================================
    // Coverage tests: aggregate, register_subplan_columns
    // ========================================================================

    #[test]
    fn test_bind_aggregate_registers_group_by_column_and_alias() {
        use crate::query::plan::{AggregateExpr, AggregateFunction, AggregateOp};

        let plan = LogicalPlan::new(LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![LogicalExpression::Property {
                variable: "n".to_string(),
                property: "city".to_string(),
            }],
            aggregates: vec![AggregateExpr {
                function: AggregateFunction::Count,
                expression: Some(LogicalExpression::Variable("n".to_string())),
                expression2: None,
                distinct: false,
                alias: Some("c".to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: None,
                input: None,
            })),
            having: None,
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("c"), "aggregate alias registered");
        assert!(ctx.contains("n.city"), "group-by column name registered");
    }

    #[test]
    fn test_apply_registers_aggregate_subplan_columns() {
        use crate::query::plan::{
            AggregateExpr, AggregateFunction, AggregateOp, ApplyOp, DistinctOp,
        };

        // Apply whose subplan is Distinct(Aggregate(...)). This exercises the
        // Distinct -> Aggregate recursion inside register_subplan_columns.
        let subplan = LogicalOperator::Distinct(DistinctOp {
            input: Box::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: vec![],
                aggregates: vec![AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: None,
                    expression2: None,
                    distinct: false,
                    alias: Some("total".to_string()),
                    percentile: None,
                    separator: None,
                }],
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: None,
                    input: None,
                })),
                having: None,
            })),
            columns: None,
        });

        let plan = LogicalPlan::new(LogicalOperator::Apply(ApplyOp {
            input: Box::new(LogicalOperator::Empty),
            subplan: Box::new(subplan),
            shared_variables: vec![],
            optional: false,
            unit: false,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(
            ctx.contains("total"),
            "aggregate alias surfaces through Distinct wrapper"
        );
    }

    #[test]
    fn test_apply_registers_sort_and_limit_wrapped_return_columns() {
        use crate::query::plan::{ApplyOp, CountExpr, LimitOp, SortKey, SortOp, SortOrder};

        // Apply(subplan = Limit(Sort(Return(var, property))))
        let subplan = LogicalOperator::Limit(LimitOp {
            count: CountExpr::Literal(10),
            input: Box::new(LogicalOperator::Sort(SortOp {
                keys: vec![SortKey {
                    expression: LogicalExpression::Variable("n".to_string()),
                    order: SortOrder::Ascending,
                    nulls: None,
                }],
                input: Box::new(LogicalOperator::Return(ReturnOp {
                    items: vec![
                        ReturnItem {
                            expression: LogicalExpression::Variable("n".to_string()),
                            alias: None,
                        },
                        ReturnItem {
                            expression: LogicalExpression::Property {
                                variable: "n".to_string(),
                                property: "name".to_string(),
                            },
                            alias: None,
                        },
                        ReturnItem {
                            // Literal with no alias hits the `_ => continue` branch.
                            expression: LogicalExpression::Literal(
                                grafeo_common::types::Value::Int64(1),
                            ),
                            alias: None,
                        },
                    ],
                    distinct: false,
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "n".to_string(),
                        label: None,
                        input: None,
                    })),
                })),
            })),
        });

        let plan = LogicalPlan::new(LogicalOperator::Apply(ApplyOp {
            input: Box::new(LogicalOperator::Empty),
            subplan: Box::new(subplan),
            shared_variables: vec![],
            optional: false,
            unit: false,
        }));

        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        // Unaliased Variable registers under the variable name itself.
        assert!(ctx.contains("n"));
        // Unaliased Property registers as "variable.property".
        assert!(ctx.contains("n.name"));
    }

    /// A horizontal aggregate keeps the variables of its input, adds its
    /// result, and needs its list column to be bound.
    #[test]
    fn test_horizontal_aggregate_binds_its_input_and_result() {
        use crate::query::plan::{AggregateFunction, EntityKind, HorizontalAggregateOp};

        let aggregate = |list_column: &str| {
            LogicalPlan::new(LogicalOperator::HorizontalAggregate(
                HorizontalAggregateOp {
                    list_column: list_column.to_string(),
                    entity_kind: EntityKind::Edge,
                    function: AggregateFunction::Sum,
                    distinct: false,
                    percentile: None,
                    separator: None,
                    property: "weight".to_string(),
                    alias: "total".to_string(),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "e".to_string(),
                        label: None,
                        input: None,
                    })),
                },
            ))
        };
        let ctx = Binder::new().bind(&aggregate("e")).unwrap();
        assert!(ctx.contains("e"), "the input's variable stays bound");
        assert!(ctx.contains("total"), "the result is bound");
        let error = Binder::new().bind(&aggregate("missing")).unwrap_err();
        assert!(
            error.to_string().contains("missing"),
            "an unbound list column is an error: {error}"
        );
    }

    // ========================================================================
    // Coverage tests: expand path alias (variable-length paths)
    // ========================================================================

    #[test]
    fn test_expand_with_path_alias_registers_auxiliary_variables() {
        use crate::query::plan::{ExpandDirection, ExpandOp, PathMode};

        let plan = LogicalPlan::new(LogicalOperator::Expand(ExpandOp {
            quantified: true,
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["ROAD".to_string()],
            min_hops: 1,
            max_hops: Some(5),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: None,
                input: None,
            })),
            path_alias: Some("p".to_string()),
            path_mode: PathMode::Walk,
        }));
        let mut binder = Binder::new();
        let ctx = binder.bind(&plan).unwrap();
        assert!(ctx.contains("p"));
        assert!(ctx.contains("_path_length_p"));
        assert!(ctx.contains("_path_nodes_p"));
        assert!(ctx.contains("_path_edges_p"));
    }
}
