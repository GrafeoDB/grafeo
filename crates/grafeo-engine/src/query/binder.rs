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
use std::collections::HashMap;

/// Creates a semantic binding error.
fn binding_error(message: impl Into<String>) -> Error {
    Error::Query(QueryError::new(QueryErrorKind::Semantic, message))
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
#[derive(Debug, Clone, Default)]
pub struct BindingContext {
    /// Map from variable name to its info.
    variables: HashMap<String, VariableInfo>,
    /// Variables in order of definition.
    order: Vec<String>,
}

impl BindingContext {
    /// Creates a new empty binding context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Adds a variable to the context.
    pub fn add_variable(&mut self, name: String, info: VariableInfo) {
        if !self.variables.contains_key(&name) {
            self.order.push(name.clone());
        }
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
    pub fn variable_names(&self) -> &[String] {
        &self.order
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
                    return Err(binding_error(format!(
                        "Undefined source variable '{}' in CREATE EDGE",
                        create.from_variable
                    )));
                }
                if !self.context.contains(&create.to_variable) {
                    return Err(binding_error(format!(
                        "Undefined target variable '{}' in CREATE EDGE",
                        create.to_variable
                    )));
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
                    return Err(binding_error(format!(
                        "Undefined variable '{}' in DELETE",
                        delete.variable
                    )));
                }
                Ok(())
            }
            LogicalOperator::DeleteEdge(delete) => {
                self.bind_operator(&delete.input)?;
                // Validate that the variable to delete is defined
                if !self.context.contains(&delete.variable) {
                    return Err(binding_error(format!(
                        "Undefined variable '{}' in DELETE",
                        delete.variable
                    )));
                }
                Ok(())
            }
            LogicalOperator::SetProperty(set) => {
                self.bind_operator(&set.input)?;
                // Validate that the variable to update is defined
                if !self.context.contains(&set.variable) {
                    return Err(binding_error(format!(
                        "Undefined variable '{}' in SET",
                        set.variable
                    )));
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
            LogicalOperator::AddLabel(add_label) => {
                self.bind_operator(&add_label.input)?;
                // Validate that the variable exists
                if !self.context.contains(&add_label.variable) {
                    return Err(binding_error(format!(
                        "Undefined variable '{}' in SET labels",
                        add_label.variable
                    )));
                }
                Ok(())
            }
            LogicalOperator::RemoveLabel(remove_label) => {
                self.bind_operator(&remove_label.input)?;
                // Validate that the variable exists
                if !self.context.contains(&remove_label.variable) {
                    return Err(binding_error(format!(
                        "Undefined variable '{}' in REMOVE labels",
                        remove_label.variable
                    )));
                }
                Ok(())
            }
            LogicalOperator::ShortestPath(sp) => {
                // First bind the input
                self.bind_operator(&sp.input)?;
                // Validate that source and target variables are defined
                if !self.context.contains(&sp.source_var) {
                    return Err(binding_error(format!(
                        "Undefined source variable '{}' in shortestPath",
                        sp.source_var
                    )));
                }
                if !self.context.contains(&sp.target_var) {
                    return Err(binding_error(format!(
                        "Undefined target variable '{}' in shortestPath",
                        sp.target_var
                    )));
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
        if let TripleComponent::Variable(name) = &scan.subject {
            if !self.context.contains(name) {
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
        }

        if let TripleComponent::Variable(name) = &scan.predicate {
            if !self.context.contains(name) {
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
        }

        if let TripleComponent::Variable(name) = &scan.object {
            if !self.context.contains(name) {
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
        }

        if let Some(TripleComponent::Variable(name)) = &scan.graph {
            if !self.context.contains(name) {
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
            return Err(binding_error(format!(
                "Undefined variable '{}' in EXPAND",
                expand.from_variable
            )));
        }

        // Validate that the source is a node
        if let Some(info) = self.context.get(&expand.from_variable) {
            if !info.is_node {
                return Err(binding_error(format!(
                    "Variable '{}' is not a node, cannot expand from it",
                    expand.from_variable
                )));
            }
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

        // Add path length variable for variable-length paths (for length(p) calls)
        if let Some(ref path_alias) = expand.path_alias {
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

    /// Binds a return operator.
    fn bind_return(&mut self, ret: &ReturnOp) -> Result<()> {
        // First bind the input
        self.bind_operator(&ret.input)?;

        // Validate all return expressions
        for item in &ret.items {
            self.validate_return_item(item)?;
        }

        Ok(())
    }

    /// Validates a return item.
    fn validate_return_item(&self, item: &ReturnItem) -> Result<()> {
        self.validate_expression(&item.expression)
    }

    /// Validates that an expression only references defined variables.
    fn validate_expression(&self, expr: &LogicalExpression) -> Result<()> {
        match expr {
            LogicalExpression::Variable(name) => {
                if !self.context.contains(name) && !name.starts_with("_anon_") {
                    return Err(binding_error(format!("Undefined variable '{name}'")));
                }
                Ok(())
            }
            LogicalExpression::Property { variable, .. } => {
                if !self.context.contains(variable) && !variable.starts_with("_anon_") {
                    return Err(binding_error(format!(
                        "Undefined variable '{variable}' in property access"
                    )));
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
                    return Err(binding_error(format!(
                        "Undefined variable '{var}' in function"
                    )));
                }
                Ok(())
            }
            LogicalExpression::ListComprehension {
                list_expr,
                filter_expr,
                map_expr,
                ..
            } => {
                // Validate the list expression
                self.validate_expression(list_expr)?;
                // Note: filter_expr and map_expr use the comprehension variable
                // which is defined within the comprehension scope, so we don't
                // need to validate it against the outer context
                if let Some(filter) = filter_expr {
                    self.validate_expression(filter)?;
                }
                self.validate_expression(map_expr)?;
                Ok(())
            }
            LogicalExpression::ExistsSubquery(subquery)
            | LogicalExpression::CountSubquery(subquery) => {
                // Subqueries have their own binding context
                // For now, just validate the structure exists
                let _ = subquery; // Would need recursive binding
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
                    .map(|info| info.data_type.clone())
                    .unwrap_or(LogicalType::Any)
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
        use crate::query::plan::{ExpandDirection, ExpandOp};

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
                edge_type: Some("KNOWS".to_string()),
                min_hops: 1,
                max_hops: Some(1),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "a".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                path_alias: None,
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
}
