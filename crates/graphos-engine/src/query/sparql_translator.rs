//! SPARQL to LogicalPlan translator.
//!
//! Translates SPARQL 1.1 AST to the common logical plan representation.

use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, BindOp, DistinctOp, FilterOp, JoinOp,
    JoinType, LimitOp, LogicalExpression, LogicalOperator, LogicalPlan, Projection, ProjectOp,
    SkipOp, SortKey, SortOp, SortOrder, TripleComponent, TripleScanOp, UnaryOp, UnionOp,
};
use graphos_adapters::query::sparql::{self, ast};
use graphos_common::types::Value;
use graphos_common::utils::error::{Error, Result};
use std::collections::HashMap;

/// Translates a SPARQL query string to a logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let sparql_query = sparql::parse(query)?;
    let mut translator = SparqlTranslator::new();
    translator.translate_query(&sparql_query)
}

/// Translator from SPARQL AST to LogicalPlan.
struct SparqlTranslator {
    /// Prefix mappings for IRI resolution.
    prefixes: HashMap<String, String>,
    /// Base IRI for relative IRI resolution.
    base: Option<String>,
    /// Counter for generating anonymous variables.
    anon_counter: u32,
}

impl SparqlTranslator {
    fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base: None,
            anon_counter: 0,
        }
    }

    fn translate_query(&mut self, query: &ast::Query) -> Result<LogicalPlan> {
        // Process prologue
        if let Some(base) = &query.base {
            self.base = Some(base.as_str().to_string());
        }

        for prefix in &query.prefixes {
            self.prefixes
                .insert(prefix.prefix.clone(), prefix.namespace.as_str().to_string());
        }

        // Translate query form
        match &query.query_form {
            ast::QueryForm::Select(select) => self.translate_select(select),
            ast::QueryForm::Ask(ask) => self.translate_ask(ask),
            ast::QueryForm::Construct(construct) => self.translate_construct(construct),
            ast::QueryForm::Describe(describe) => self.translate_describe(describe),
        }
    }

    fn translate_select(&mut self, select: &ast::SelectQuery) -> Result<LogicalPlan> {
        // Start with the WHERE clause pattern
        let mut plan = self.translate_graph_pattern(&select.where_clause)?;

        // Apply GROUP BY if present
        if let Some(group_by) = &select.solution_modifiers.group_by {
            let (aggregates, _group_exprs) = self.extract_aggregates_for_select(select)?;
            let group_by_exprs = group_by
                .iter()
                .map(|g| self.translate_group_condition(g))
                .collect::<Result<Vec<_>>>()?;

            plan = LogicalOperator::Aggregate(AggregateOp {
                group_by: group_by_exprs,
                aggregates,
                input: Box::new(plan),
            });

            // Apply HAVING if present
            if let Some(having) = &select.solution_modifiers.having {
                plan = LogicalOperator::Filter(FilterOp {
                    predicate: self.translate_expression(having)?,
                    input: Box::new(plan),
                });
            }
        }

        // Apply ORDER BY
        if let Some(order_by) = &select.solution_modifiers.order_by {
            let keys = order_by
                .iter()
                .map(|oc| {
                    Ok(SortKey {
                        expression: self.translate_expression(&oc.expression)?,
                        order: match oc.direction {
                            ast::SortDirection::Ascending => SortOrder::Ascending,
                            ast::SortDirection::Descending => SortOrder::Descending,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            plan = LogicalOperator::Sort(SortOp {
                keys,
                input: Box::new(plan),
            });
        }

        // Apply OFFSET
        if let Some(offset) = select.solution_modifiers.offset {
            plan = LogicalOperator::Skip(SkipOp {
                count: offset as usize,
                input: Box::new(plan),
            });
        }

        // Apply LIMIT
        if let Some(limit) = select.solution_modifiers.limit {
            plan = LogicalOperator::Limit(LimitOp {
                count: limit as usize,
                input: Box::new(plan),
            });
        }

        // Apply DISTINCT/REDUCED
        if select.modifier == ast::SelectModifier::Distinct {
            plan = LogicalOperator::Distinct(DistinctOp {
                input: Box::new(plan),
            });
        }

        // Apply projection
        let projections = self.translate_projection(&select.projection)?;
        if !projections.is_empty() {
            plan = LogicalOperator::Project(ProjectOp {
                projections,
                input: Box::new(plan),
            });
        }

        Ok(LogicalPlan::new(plan))
    }

    fn translate_ask(&mut self, ask: &ast::AskQuery) -> Result<LogicalPlan> {
        // ASK returns true if the pattern has any matches
        let plan = self.translate_graph_pattern(&ask.where_clause)?;

        // Limit to 1 result for efficiency
        let plan = LogicalOperator::Limit(LimitOp {
            count: 1,
            input: Box::new(plan),
        });

        Ok(LogicalPlan::new(plan))
    }

    fn translate_construct(&mut self, construct: &ast::ConstructQuery) -> Result<LogicalPlan> {
        // For CONSTRUCT, we need to evaluate the WHERE pattern and then
        // produce triples according to the template
        let plan = self.translate_graph_pattern(&construct.where_clause)?;

        // Apply solution modifiers
        let mut plan = plan;
        if let Some(limit) = construct.solution_modifiers.limit {
            plan = LogicalOperator::Limit(LimitOp {
                count: limit as usize,
                input: Box::new(plan),
            });
        }

        // The template will be processed at execution time
        Ok(LogicalPlan::new(plan))
    }

    fn translate_describe(&mut self, describe: &ast::DescribeQuery) -> Result<LogicalPlan> {
        // DESCRIBE returns information about resources
        if let Some(where_clause) = &describe.where_clause {
            let plan = self.translate_graph_pattern(where_clause)?;
            Ok(LogicalPlan::new(plan))
        } else {
            // DESCRIBE with just IRIs - create a scan for each resource
            Ok(LogicalPlan::new(LogicalOperator::Empty))
        }
    }

    fn translate_projection(&mut self, projection: &ast::Projection) -> Result<Vec<Projection>> {
        match projection {
            ast::Projection::Wildcard => Ok(Vec::new()), // Empty means select all
            ast::Projection::Variables(vars) => {
                vars.iter()
                    .map(|pv| {
                        Ok(Projection {
                            expression: self.translate_expression(&pv.expression)?,
                            alias: pv.alias.clone(),
                        })
                    })
                    .collect()
            }
        }
    }

    fn translate_graph_pattern(&mut self, pattern: &ast::GraphPattern) -> Result<LogicalOperator> {
        match pattern {
            ast::GraphPattern::Basic(triples) => self.translate_basic_pattern(triples),

            ast::GraphPattern::Group(patterns) => {
                let mut plan = LogicalOperator::Empty;
                for p in patterns {
                    let p_plan = self.translate_graph_pattern(p)?;
                    plan = self.join_patterns(plan, p_plan);
                }
                Ok(plan)
            }

            ast::GraphPattern::Optional(inner) => {
                // OPTIONAL creates a left outer join
                let inner_plan = self.translate_graph_pattern(inner)?;
                Ok(inner_plan)
            }

            ast::GraphPattern::Union(alternatives) => {
                let inputs = alternatives
                    .iter()
                    .map(|p| self.translate_graph_pattern(p))
                    .collect::<Result<Vec<_>>>()?;

                Ok(LogicalOperator::Union(UnionOp { inputs }))
            }

            ast::GraphPattern::Minus(inner) => {
                // MINUS is an anti-join
                let inner_plan = self.translate_graph_pattern(inner)?;
                Ok(inner_plan)
            }

            ast::GraphPattern::Filter(expr) => {
                // Standalone FILTER - will be combined with its context
                let predicate = self.translate_expression(expr)?;
                Ok(LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(LogicalOperator::Empty),
                }))
            }

            ast::GraphPattern::Bind { expression, variable } => {
                let expr = self.translate_expression(expression)?;
                Ok(LogicalOperator::Bind(BindOp {
                    expression: expr,
                    variable: variable.clone(),
                    input: Box::new(LogicalOperator::Empty),
                }))
            }

            ast::GraphPattern::NamedGraph { graph: _, pattern } => {
                // For named graph, we add the graph as context
                self.translate_graph_pattern(pattern)
            }

            ast::GraphPattern::SubSelect(subquery) => {
                let plan = self.translate_select(subquery)?;
                Ok(plan.root)
            }

            ast::GraphPattern::Service { endpoint: _, pattern, silent: _ } => {
                // SERVICE queries remote endpoints - for now, translate the pattern
                self.translate_graph_pattern(pattern)
            }

            ast::GraphPattern::InlineData(_data) => {
                // VALUES clause - inline data
                // For now, return empty; full implementation needs a Values operator
                Ok(LogicalOperator::Empty)
            }
        }
    }

    fn translate_basic_pattern(
        &mut self,
        triples: &[ast::TriplePattern],
    ) -> Result<LogicalOperator> {
        if triples.is_empty() {
            return Ok(LogicalOperator::Empty);
        }

        let mut plan = LogicalOperator::Empty;

        for triple in triples {
            let triple_scan = self.translate_triple_pattern(triple)?;
            plan = self.join_patterns(plan, triple_scan);
        }

        Ok(plan)
    }

    fn translate_triple_pattern(&mut self, triple: &ast::TriplePattern) -> Result<LogicalOperator> {
        let subject = self.translate_triple_term(&triple.subject)?;
        let predicate = self.translate_property_path(&triple.predicate)?;
        let object = self.translate_triple_term(&triple.object)?;

        Ok(LogicalOperator::TripleScan(TripleScanOp {
            subject,
            predicate,
            object,
            graph: None,
            input: None,
        }))
    }

    fn translate_triple_term(&mut self, term: &ast::TripleTerm) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::TripleTerm::Iri(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::TripleTerm::Literal(lit) => {
                let value = self.literal_to_value(lit);
                Ok(TripleComponent::Literal(value))
            }
            ast::TripleTerm::BlankNode(bnode) => {
                // Treat blank nodes as variables
                match bnode {
                    ast::BlankNode::Labeled(label) => {
                        Ok(TripleComponent::Variable(format!("_:{}", label)))
                    }
                    ast::BlankNode::Anonymous(_) => {
                        let var = format!("_:anon{}", self.next_anon());
                        Ok(TripleComponent::Variable(var))
                    }
                }
            }
        }
    }

    fn translate_property_path(&mut self, path: &ast::PropertyPath) -> Result<TripleComponent> {
        match path {
            ast::PropertyPath::Predicate(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::PropertyPath::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::PropertyPath::RdfType => {
                Ok(TripleComponent::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()))
            }
            // Complex property paths are not fully supported yet
            _ => Err(Error::Internal(
                "Complex property paths not yet supported".to_string(),
            )),
        }
    }

    fn translate_expression(&mut self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),

            ast::Expression::Iri(iri) => {
                Ok(LogicalExpression::Literal(Value::String(
                    self.resolve_iri(iri).into(),
                )))
            }

            ast::Expression::Literal(lit) => {
                Ok(LogicalExpression::Literal(self.literal_to_value(lit)))
            }

            ast::Expression::Binary { left, operator, right } => {
                let left = self.translate_expression(left)?;
                let right = self.translate_expression(right)?;
                let op = self.translate_binary_op(*operator);
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                })
            }

            ast::Expression::Unary { operator, operand } => {
                let operand = self.translate_expression(operand)?;
                let op = self.translate_unary_op(*operator);
                Ok(LogicalExpression::Unary {
                    op,
                    operand: Box::new(operand),
                })
            }

            ast::Expression::FunctionCall { function, arguments } => {
                let name = self.translate_function_name(function);
                let args = arguments
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::FunctionCall { name, args })
            }

            ast::Expression::Bound(var) => {
                // BOUND(?x) checks if variable is bound
                Ok(LogicalExpression::FunctionCall {
                    name: "BOUND".to_string(),
                    args: vec![LogicalExpression::Variable(var.clone())],
                })
            }

            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let cond = self.translate_expression(condition)?;
                let then_expr = self.translate_expression(then_expression)?;
                let else_expr = self.translate_expression(else_expression)?;
                Ok(LogicalExpression::Case {
                    operand: None,
                    when_clauses: vec![(cond, then_expr)],
                    else_clause: Some(Box::new(else_expr)),
                })
            }

            ast::Expression::Coalesce(exprs) => {
                let args = exprs
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::FunctionCall {
                    name: "COALESCE".to_string(),
                    args,
                })
            }

            ast::Expression::Exists(_pattern) => {
                // EXISTS { pattern } - check if pattern has any matches
                Ok(LogicalExpression::FunctionCall {
                    name: "EXISTS".to_string(),
                    args: vec![],
                })
            }

            ast::Expression::NotExists(_pattern) => {
                // NOT EXISTS { pattern }
                Ok(LogicalExpression::FunctionCall {
                    name: "NOT_EXISTS".to_string(),
                    args: vec![],
                })
            }

            ast::Expression::In { expression, list } => {
                let expr = self.translate_expression(expression)?;
                let items = list
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::In,
                    right: Box::new(LogicalExpression::List(items)),
                })
            }

            ast::Expression::NotIn { expression, list } => {
                let expr = self.translate_expression(expression)?;
                let items = list
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(LogicalExpression::Binary {
                        left: Box::new(expr),
                        op: BinaryOp::In,
                        right: Box::new(LogicalExpression::List(items)),
                    }),
                })
            }

            ast::Expression::Aggregate(agg) => {
                self.translate_aggregate_expression(agg)
            }

            ast::Expression::Bracketed(inner) => {
                self.translate_expression(inner)
            }
        }
    }

    fn translate_aggregate_expression(
        &mut self,
        agg: &ast::AggregateExpression,
    ) -> Result<LogicalExpression> {
        let func_name = match agg {
            ast::AggregateExpression::Count { .. } => "COUNT",
            ast::AggregateExpression::Sum { .. } => "SUM",
            ast::AggregateExpression::Average { .. } => "AVG",
            ast::AggregateExpression::Minimum { .. } => "MIN",
            ast::AggregateExpression::Maximum { .. } => "MAX",
            ast::AggregateExpression::Sample { .. } => "SAMPLE",
            ast::AggregateExpression::GroupConcat { .. } => "GROUP_CONCAT",
        };

        let args = match agg {
            ast::AggregateExpression::Count { expression, .. } => {
                if let Some(expr) = expression {
                    vec![self.translate_expression(expr)?]
                } else {
                    vec![]
                }
            }
            ast::AggregateExpression::Sum { expression, .. }
            | ast::AggregateExpression::Average { expression, .. }
            | ast::AggregateExpression::Minimum { expression, .. }
            | ast::AggregateExpression::Maximum { expression, .. }
            | ast::AggregateExpression::Sample { expression, .. }
            | ast::AggregateExpression::GroupConcat { expression, .. } => {
                vec![self.translate_expression(expression)?]
            }
        };

        Ok(LogicalExpression::FunctionCall {
            name: func_name.to_string(),
            args,
        })
    }

    fn translate_binary_op(&self, op: ast::BinaryOperator) -> BinaryOp {
        match op {
            ast::BinaryOperator::Or => BinaryOp::Or,
            ast::BinaryOperator::And => BinaryOp::And,
            ast::BinaryOperator::Equal => BinaryOp::Eq,
            ast::BinaryOperator::NotEqual => BinaryOp::Ne,
            ast::BinaryOperator::LessThan => BinaryOp::Lt,
            ast::BinaryOperator::LessOrEqual => BinaryOp::Le,
            ast::BinaryOperator::GreaterThan => BinaryOp::Gt,
            ast::BinaryOperator::GreaterOrEqual => BinaryOp::Ge,
            ast::BinaryOperator::Add => BinaryOp::Add,
            ast::BinaryOperator::Subtract => BinaryOp::Sub,
            ast::BinaryOperator::Multiply => BinaryOp::Mul,
            ast::BinaryOperator::Divide => BinaryOp::Div,
        }
    }

    fn translate_unary_op(&self, op: ast::UnaryOperator) -> UnaryOp {
        match op {
            ast::UnaryOperator::Not => UnaryOp::Not,
            ast::UnaryOperator::Plus => UnaryOp::Not, // No direct mapping, use identity
            ast::UnaryOperator::Minus => UnaryOp::Neg,
        }
    }

    fn translate_function_name(&self, func: &ast::FunctionName) -> String {
        match func {
            ast::FunctionName::BuiltIn(builtin) => format!("{:?}", builtin).to_uppercase(),
            ast::FunctionName::Custom(iri) => self.resolve_iri(iri),
        }
    }

    fn translate_group_condition(
        &mut self,
        cond: &ast::GroupCondition,
    ) -> Result<LogicalExpression> {
        match cond {
            ast::GroupCondition::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::GroupCondition::Expression { expression, .. } => {
                self.translate_expression(expression)
            }
            ast::GroupCondition::BuiltInCall(expr) => self.translate_expression(expr),
        }
    }

    fn extract_aggregates_for_select(
        &mut self,
        select: &ast::SelectQuery,
    ) -> Result<(Vec<AggregateExpr>, Vec<LogicalExpression>)> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();

        if let ast::Projection::Variables(vars) = &select.projection {
            for pv in vars {
                if self.is_aggregate_expression(&pv.expression) {
                    if let Some(agg) = self.extract_aggregate(&pv.expression, &pv.alias)? {
                        aggregates.push(agg);
                    }
                } else {
                    group_by.push(self.translate_expression(&pv.expression)?);
                }
            }
        }

        Ok((aggregates, group_by))
    }

    fn is_aggregate_expression(&self, expr: &ast::Expression) -> bool {
        matches!(expr, ast::Expression::Aggregate(_))
    }

    fn extract_aggregate(
        &mut self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        if let ast::Expression::Aggregate(agg) = expr {
            let (func, expr_inner, distinct) = match agg {
                ast::AggregateExpression::Count { distinct, expression } => {
                    (AggregateFunction::Count, expression.as_ref().map(|e| e.as_ref()), *distinct)
                }
                ast::AggregateExpression::Sum { distinct, expression } => {
                    (AggregateFunction::Sum, Some(expression.as_ref()), *distinct)
                }
                ast::AggregateExpression::Average { distinct, expression } => {
                    (AggregateFunction::Avg, Some(expression.as_ref()), *distinct)
                }
                ast::AggregateExpression::Minimum { expression } => {
                    (AggregateFunction::Min, Some(expression.as_ref()), false)
                }
                ast::AggregateExpression::Maximum { expression } => {
                    (AggregateFunction::Max, Some(expression.as_ref()), false)
                }
                ast::AggregateExpression::Sample { expression } => {
                    // Map SAMPLE to Collect for now
                    (AggregateFunction::Collect, Some(expression.as_ref()), false)
                }
                ast::AggregateExpression::GroupConcat { distinct, expression, .. } => {
                    (AggregateFunction::Collect, Some(expression.as_ref()), *distinct)
                }
            };

            let expression = if let Some(e) = expr_inner {
                Some(self.translate_expression(e)?)
            } else {
                None
            };

            Ok(Some(AggregateExpr {
                function: func,
                expression,
                distinct,
                alias: alias.clone(),
            }))
        } else {
            Ok(None)
        }
    }

    fn join_patterns(
        &self,
        left: LogicalOperator,
        right: LogicalOperator,
    ) -> LogicalOperator {
        if matches!(left, LogicalOperator::Empty) {
            return right;
        }
        if matches!(right, LogicalOperator::Empty) {
            return left;
        }

        // For basic patterns, use inner join on shared variables
        LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions: vec![], // Shared variables are implicit join conditions
        })
    }

    fn resolve_iri(&self, iri: &ast::Iri) -> String {
        let iri_str = iri.as_str();

        // Check if it's a prefixed name
        if let Some(colon_pos) = iri_str.find(':') {
            let prefix = &iri_str[..colon_pos];
            let local = &iri_str[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                return format!("{}{}", namespace, local);
            }
        }

        // Return as-is if no prefix match or already a full IRI
        iri_str.to_string()
    }

    fn literal_to_value(&self, lit: &ast::Literal) -> Value {
        // Check for typed literals
        if let Some(datatype) = &lit.datatype {
            let dt = self.resolve_iri(datatype);
            match dt.as_str() {
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long" => {
                    if let Ok(n) = lit.value.parse::<i64>() {
                        return Value::Int64(n);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => {
                    if let Ok(n) = lit.value.parse::<f64>() {
                        return Value::Float64(n);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#boolean" => {
                    return Value::Bool(lit.value == "true" || lit.value == "1");
                }
                _ => {}
            }
        }

        // Default to string
        Value::String(lit.value.clone().into())
    }

    fn next_anon(&mut self) -> u32 {
        let n = self.anon_counter;
        self.anon_counter += 1;
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_simple_select() {
        let query = "SELECT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_prefix() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE { ?x foaf:name ?name }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_filter() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_ask() {
        let query = "ASK { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_limit() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } LIMIT 10";
        let result = translate(query);
        assert!(result.is_ok());
    }
}
