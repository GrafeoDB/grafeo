//! Semantic validation - catching errors before execution.
//!
//! The binder walks the logical plan and validates that everything makes sense:
//! - Is that variable actually defined? (You can't use `RETURN x` if `x` wasn't matched)
//! - Does that property access make sense? (Accessing `.age` on an integer fails)
//! - Are types compatible? (Can't compare a string to an integer)
//!
//! Better to catch these errors early than waste time executing a broken query.

use crate::query::plan::{
    ExpandOp, FilterOp, LogicalExpression, LogicalOperator, LogicalPlan, NodeScanOp, ReturnItem,
    ReturnOp, TripleScanOp,
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
}

impl Binder {
    /// Creates a new binder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            context: BindingContext::new(),
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
                        self.context.add_variable(
                            alias.clone(),
                            VariableInfo {
                                name: alias.clone(),
                                data_type,
                                is_node,
                                is_edge,
                            },
                        );
                    }
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
                // CreateNode introduces a new variable
                if let Some(ref input) = create.input {
                    self.bind_operator(input)?;
                }
                self.context.add_variable(
                    create.variable.clone(),
                    VariableInfo {
                        name: create.variable.clone(),
                        data_type: LogicalType::Node,
                        is_node: true,
                        is_edge: false,
                    },
                );
                // Validate property expressions
                for (_, expr) in &create.properties {
                    self.validate_expression(expr)?;
                }
                Ok(())
            }
            LogicalOperator::EdgeScan(scan) => {
                if let Some(ref input) = scan.input {
                    self.bind_operator(input)?;
                }
                self.context.add_variable(
                    scan.variable.clone(),
                    VariableInfo {
                        name: scan.variable.clone(),
                        data_type: LogicalType::Edge,
                        is_node: false,
                        is_edge: true,
                    },
                );
                Ok(())
            }
            LogicalOperator::Distinct(distinct) => self.bind_operator(&distinct.input),
            LogicalOperator::Join(join) => self.bind_join(join),
            LogicalOperator::Aggregate(agg) => self.bind_aggregate(agg),
            LogicalOperator::CreateEdge(create) => {
                self.bind_operator(&create.input)?;
                // Validate that source and target variables are defined
                if !self.context.contains(&create.from_variable) {
                    return Err(undefined_variable_error(
                        &create.from_variable,
                        &self.context,
                        " (source in CREATE EDGE)",
                    ));
                }
                if !self.context.contains(&create.to_variable) {
                    return Err(undefined_variable_error(
                        &create.to_variable,
                        &self.context,
                        " (target in CREATE EDGE)",
                    ));
                }
                // Add edge variable if present
                if let Some(ref var) = create.variable {
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
                // Validate property expressions
                for (_, expr) in &create.properties {
                    self.validate_expression(expr)?;
                }
                Ok(())
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
            LogicalOperator::Union(union) => {
                for input in &union.inputs {
                    self.bind_operator(input)?;
                }
                Ok(())
            }
            LogicalOperator::LeftJoin(lj) => {
                self.bind_operator(&lj.left)?;
                self.bind_operator(&lj.right)?;
                if let Some(ref cond) = lj.condition {
                    self.validate_expression(cond)?;
                }
                Ok(())
            }
            LogicalOperator::AntiJoin(aj) => {
                self.bind_operator(&aj.left)?;
                self.bind_operator(&aj.right)?;
                Ok(())
            }
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
                // Validate the match property expressions
                for (_, expr) in &merge.match_properties {
                    self.validate_expression(expr)?;
                }
                // Validate the ON CREATE property expressions
                for (_, expr) in &merge.on_create {
                    self.validate_expression(expr)?;
                }
                // Validate the ON MATCH property expressions
                for (_, expr) in &merge.on_match {
                    self.validate_expression(expr)?;
                }
                // MERGE introduces a new variable
                self.context.add_variable(
                    merge.variable.clone(),
                    VariableInfo {
                        name: merge.variable.clone(),
                        data_type: LogicalType::Node,
                        is_node: true,
                        is_edge: false,
                    },
                );
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
                for (_, expr) in &merge_rel.match_properties {
                    self.validate_expression(expr)?;
                }
                for (_, expr) in &merge_rel.on_create {
                    self.validate_expression(expr)?;
                }
                for (_, expr) in &merge_rel.on_match {
                    self.validate_expression(expr)?;
                }
                // MERGE relationship introduces the edge variable
                self.context.add_variable(
                    merge_rel.variable.clone(),
                    VariableInfo {
                        name: merge_rel.variable.clone(),
                        data_type: LogicalType::Edge,
                        is_node: false,
                        is_edge: true,
                    },
                );
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
                if !self.context.contains(&sp.target_var) {
                    return Err(undefined_variable_error(
                        &sp.target_var,
                        &self.context,
                        " (target in shortestPath)",
                    ));
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
            | LogicalOperator::AddGraph(_)
            | LogicalOperator::HorizontalAggregate(_) => Ok(()),
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
                self.bind_operator(&except.left)?;
                self.bind_operator(&except.right)?;
                Ok(())
            }
            LogicalOperator::Intersect(intersect) => {
                self.bind_operator(&intersect.left)?;
                self.bind_operator(&intersect.right)?;
                Ok(())
            }
            LogicalOperator::Otherwise(otherwise) => {
                self.bind_operator(&otherwise.left)?;
                self.bind_operator(&otherwise.right)?;
                Ok(())
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

                self.bind_operator(&apply.subplan)?;

                // Remove internal-only variables added by the subplan (those that
                // are not output columns). Prevents subplan internals from leaking
                // into the outer query or sibling CALL blocks.
                let mut subplan_output_ctx = BindingContext::new();
                Self::register_subplan_columns(&apply.subplan, &mut subplan_output_ctx);
                let subplan_output_names: HashSet<String> = subplan_output_ctx
                    .variable_names()
                    .iter()
                    .cloned()
                    .collect();

                let to_remove: Vec<String> = self
                    .context
                    .variable_names()
                    .iter()
                    .filter(|n| !outer_names.contains(*n) && !subplan_output_names.contains(*n))
                    .cloned()
                    .collect();
                for name in to_remove {
                    self.context.remove_variable(&name);
                }

                // Register output columns so downstream operators can reference them.
                Self::register_subplan_columns(&apply.subplan, &mut self.context);
                Ok(())
            }
            LogicalOperator::MultiWayJoin(mwj) => {
                for input in &mwj.inputs {
                    self.bind_operator(input)?;
                }
                for cond in &mwj.conditions {
                    self.validate_expression(&cond.left)?;
                    self.validate_expression(&cond.right)?;
                }
                Ok(())
            }
            LogicalOperator::ParameterScan(param_scan) => {
                // Register parameter columns as variables (injected by outer Apply)
                for col in &param_scan.columns {
                    self.context.add_variable(
                        col.clone(),
                        VariableInfo {
                            name: col.clone(),
                            data_type: LogicalType::Any,
                            is_node: true,
                            is_edge: false,
                        },
                    );
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
                // Register the node variable
                self.context.add_variable(
                    scan.variable.clone(),
                    VariableInfo {
                        name: scan.variable.clone(),
                        data_type: LogicalType::Node,
                        is_node: true,
                        is_edge: false,
                    },
                );
                // Optionally register the score column
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
                // Validate the query expression
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
        self.context.add_variable(
            scan.variable.clone(),
            VariableInfo {
                name: scan.variable.clone(),
                data_type: LogicalType::Node,
                is_node: true,
                is_edge: false,
            },
        );

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
            self.context.add_variable(
                edge_var.clone(),
                VariableInfo {
                    name: edge_var.clone(),
                    data_type: LogicalType::Edge,
                    is_node: false,
                    is_edge: true,
                },
            );
        }

        // Add target variable
        self.context.add_variable(
            expand.to_variable.clone(),
            VariableInfo {
                name: expand.to_variable.clone(),
                data_type: LogicalType::Node,
                is_node: true,
                is_edge: false,
            },
        );

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
                if !self.context.contains(name) && !name.starts_with("_anon_") {
                    return Err(undefined_variable_error(name, &self.context, ""));
                }
                Ok(())
            }
            LogicalExpression::Property { variable, .. } => {
                if !self.context.contains(variable) && !variable.starts_with("_anon_") {
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
                if !self.context.contains(var) && !var.starts_with("_anon_") {
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
                if !self.context.contains(base) && !base.starts_with("_anon_") {
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

    /// Binds a join operator.
    fn bind_join(&mut self, join: &crate::query::plan::JoinOp) -> Result<()> {
        // Bind both sides of the join
        self.bind_operator(&join.left)?;
        self.bind_operator(&join.right)?;

        // Validate join conditions
        for condition in &join.conditions {
            self.validate_expression(&condition.left)?;
            self.validate_expression(&condition.right)?;
        }

        Ok(())
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
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);
        assert!(
            result.is_err(),
            "ON CREATE referencing undefined variable should fail"
        );
    }

    // --- ShortestPath ---

    #[test]
    fn test_shortest_path_rejects_undefined_source() {
        use crate::query::plan::{ExpandDirection, ShortestPathOp};

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
            all_paths: false,
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
        use crate::query::plan::{ExpandDirection, JoinOp, JoinType, ShortestPathOp};

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
            all_paths: false,
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
    fn test_anon_variables_skip_validation() {
        // Variables starting with _anon_ are anonymous and should be silently accepted
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("_anon_42".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        }));

        let mut binder = Binder::new();
        let result = binder.bind(&plan);
        assert!(
            result.is_ok(),
            "Anonymous variables should bypass validation"
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
}
