//! Cypher AST to Logical Plan translator.
//!
//! Translates parsed Cypher queries into the common logical plan representation
//! that can be optimized and executed.

use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, CreateEdgeOp, CreateNodeOp,
    DeleteNodeOp, DistinctOp, ExpandDirection, ExpandOp, FilterOp, LeftJoinOp, LimitOp,
    LogicalExpression, LogicalOperator, LogicalPlan, MergeOp, NodeScanOp, ProjectOp, Projection,
    RemoveLabelOp, ReturnItem, ReturnOp, SetPropertyOp, SkipOp, SortKey, SortOp, SortOrder,
    UnaryOp, UnwindOp,
};
use grafeo_adapters::query::cypher::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};

/// Translates a Cypher query string to a logical plan.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let statement = cypher::parse(query)?;
    let translator = CypherTranslator::new();
    translator.translate_statement(&statement)
}

/// Cypher AST to logical plan translator.
struct CypherTranslator {
    /// Variable counter for generating unique variable names.
    #[allow(dead_code)]
    var_counter: u32,
}

impl CypherTranslator {
    fn new() -> Self {
        Self { var_counter: 0 }
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Query(query) => self.translate_query(query),
            ast::Statement::Create(create) => self.translate_create_statement(create),
            ast::Statement::Merge(merge) => self.translate_merge_statement(&merge),
            ast::Statement::Delete(_) => Err(Error::Internal("DELETE not yet supported".into())),
            ast::Statement::Set(_) => Err(Error::Internal("SET not yet supported".into())),
            ast::Statement::Remove(_) => Err(Error::Internal("REMOVE not yet supported".into())),
        }
    }

    fn translate_query(&self, query: &ast::Query) -> Result<LogicalPlan> {
        let mut plan: Option<LogicalOperator> = None;

        for clause in &query.clauses {
            plan = Some(self.translate_clause(clause, plan)?);
        }

        let root = plan.ok_or_else(|| Error::Internal("Empty query".into()))?;
        Ok(LogicalPlan::new(root))
    }

    fn translate_clause(
        &self,
        clause: &ast::Clause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match clause {
            ast::Clause::Match(match_clause) => self.translate_match(match_clause, input),
            ast::Clause::OptionalMatch(match_clause) => {
                self.translate_optional_match(match_clause, input)
            }
            ast::Clause::Where(where_clause) => self.translate_where(where_clause, input),
            ast::Clause::With(with_clause) => self.translate_with(with_clause, input),
            ast::Clause::Return(return_clause) => self.translate_return(return_clause, input),
            ast::Clause::Unwind(unwind_clause) => self.translate_unwind(unwind_clause, input),
            ast::Clause::OrderBy(order_by) => self.translate_order_by(order_by, input),
            ast::Clause::Skip(expr) => self.translate_skip(expr, input),
            ast::Clause::Limit(expr) => self.translate_limit(expr, input),
            ast::Clause::Create(create_clause) => {
                self.translate_create_clause(create_clause, input)
            }
            ast::Clause::Merge(merge_clause) => self.translate_merge(merge_clause, input),
            ast::Clause::Delete(delete_clause) => self.translate_delete(delete_clause, input),
            ast::Clause::Set(set_clause) => self.translate_set(set_clause, input),
            ast::Clause::Remove(remove_clause) => self.translate_remove(remove_clause, input),
        }
    }

    fn translate_match(
        &self,
        match_clause: &ast::MatchClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut plan = input;

        for pattern in &match_clause.patterns {
            plan = Some(self.translate_pattern(pattern, plan)?);
        }

        plan.ok_or_else(|| Error::Internal("Empty MATCH pattern".into()))
    }

    fn translate_optional_match(
        &self,
        match_clause: &ast::MatchClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // OPTIONAL MATCH uses LEFT JOIN semantics
        let input = input.ok_or_else(|| Error::Internal("OPTIONAL MATCH requires input".into()))?;

        // Build the match pattern
        let mut right: Option<LogicalOperator> = None;
        for pattern in &match_clause.patterns {
            right = Some(self.translate_pattern(pattern, right)?);
        }

        let right = right.ok_or_else(|| Error::Internal("Empty OPTIONAL MATCH pattern".into()))?;

        Ok(LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(input),
            right: Box::new(right),
            condition: None,
        }))
    }

    fn translate_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node_pattern) => self.translate_node_pattern(node_pattern, input),
            ast::Pattern::Path(path_pattern) => self.translate_path_pattern(path_pattern, input),
            ast::Pattern::NamedPath { pattern, .. } => self.translate_pattern(pattern, input),
        }
    }

    fn translate_node_pattern(
        &self,
        node: &ast::NodePattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let variable = node.variable.clone().unwrap_or_else(|| "_anon".to_string());
        let label = node.labels.first().cloned();

        Ok(LogicalOperator::NodeScan(NodeScanOp {
            variable,
            label,
            input: input.map(Box::new),
        }))
    }

    fn translate_path_pattern(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut plan = self.translate_node_pattern(&path.start, input)?;

        for rel in &path.chain {
            plan = self.translate_relationship_pattern(rel, plan)?;
        }

        Ok(plan)
    }

    fn translate_relationship_pattern(
        &self,
        rel: &ast::RelationshipPattern,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let from_variable = Self::get_last_variable(&input)?;
        let edge_variable = rel.variable.clone();
        let edge_type = rel.types.first().cloned();
        let to_variable = rel
            .target
            .variable
            .clone()
            .unwrap_or_else(|| "_anon".to_string());
        let target_label = rel.target.labels.first().cloned();

        let direction = match rel.direction {
            ast::Direction::Outgoing => ExpandDirection::Outgoing,
            ast::Direction::Incoming => ExpandDirection::Incoming,
            ast::Direction::Undirected => ExpandDirection::Both,
        };

        let (min_hops, max_hops) = if let Some(range) = &rel.length {
            (range.min.unwrap_or(1), range.max)
        } else {
            (1, Some(1))
        };

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable,
            to_variable: to_variable.clone(),
            edge_variable,
            direction,
            edge_type,
            min_hops,
            max_hops,
            input: Box::new(input),
        });

        if let Some(label) = target_label {
            Ok(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::FunctionCall {
                    name: "hasLabel".into(),
                    args: vec![
                        LogicalExpression::Variable(to_variable),
                        LogicalExpression::Literal(Value::from(label)),
                    ],
                },
                input: Box::new(expand),
            }))
        } else {
            Ok(expand)
        }
    }

    fn translate_where(
        &self,
        where_clause: &ast::WhereClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("WHERE requires input".into()))?;
        let predicate = self.translate_expression(&where_clause.predicate)?;

        Ok(LogicalOperator::Filter(FilterOp {
            predicate,
            input: Box::new(input),
        }))
    }

    fn translate_with(
        &self,
        with_clause: &ast::WithClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("WITH requires input".into()))?;

        let projections: Vec<Projection> = with_clause
            .items
            .iter()
            .map(|item| {
                Ok(Projection {
                    expression: self.translate_expression(&item.expression)?,
                    alias: item.alias.clone(),
                })
            })
            .collect::<Result<_>>()?;

        let mut plan = LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
        });

        if let Some(where_clause) = &with_clause.where_clause {
            let predicate = self.translate_expression(&where_clause.predicate)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate,
                input: Box::new(plan),
            });
        }

        if with_clause.distinct {
            plan = LogicalOperator::Distinct(DistinctOp {
                input: Box::new(plan),
            });
        }

        Ok(plan)
    }

    fn translate_unwind(
        &self,
        unwind_clause: &ast::UnwindClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // UNWIND can work with or without prior input
        // If there's no input, create an implicit single row (Empty with one result)
        let input = input.unwrap_or(LogicalOperator::Empty);

        let expression = self.translate_expression(&unwind_clause.expression)?;

        Ok(LogicalOperator::Unwind(UnwindOp {
            expression,
            variable: unwind_clause.variable.clone(),
            input: Box::new(input),
        }))
    }

    fn translate_merge_statement(&self, merge: &ast::MergeClause) -> Result<LogicalPlan> {
        let op = self.translate_merge(merge, None)?;
        Ok(LogicalPlan { root: op })
    }

    fn translate_merge(
        &self,
        merge_clause: &ast::MergeClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.unwrap_or(LogicalOperator::Empty);

        // Extract node information from the pattern
        // For now, we only support simple single-node patterns: (n:Label {props})
        let pattern = &merge_clause.pattern;

        // Extract node from the pattern
        let node = match pattern {
            ast::Pattern::Node(n) => n,
            ast::Pattern::Path(path) => &path.start,
            ast::Pattern::NamedPath { pattern: inner, .. } => match inner.as_ref() {
                ast::Pattern::Node(n) => n,
                ast::Pattern::Path(path) => &path.start,
                _ => {
                    return Err(Error::Internal(
                        "MERGE NamedPath must contain a node".into(),
                    ));
                }
            },
        };

        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_merge_{}", 0));
        let labels: Vec<String> = node.labels.clone();

        // Extract properties from the node pattern
        let match_properties: Vec<(String, LogicalExpression)> = node
            .properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect::<Result<Vec<_>>>()?;

        // Extract ON CREATE properties
        let on_create: Vec<(String, LogicalExpression)> =
            if let Some(set_clause) = &merge_clause.on_create {
                self.extract_set_properties(set_clause)?
            } else {
                Vec::new()
            };

        // Extract ON MATCH properties
        let on_match: Vec<(String, LogicalExpression)> =
            if let Some(set_clause) = &merge_clause.on_match {
                self.extract_set_properties(set_clause)?
            } else {
                Vec::new()
            };

        Ok(LogicalOperator::Merge(MergeOp {
            variable,
            labels,
            match_properties,
            on_create,
            on_match,
            input: Box::new(input),
        }))
    }

    /// Extracts properties from a map expression.
    #[allow(dead_code)]
    fn extract_map_properties(
        &self,
        expr: &ast::Expression,
    ) -> Result<Vec<(String, LogicalExpression)>> {
        match expr {
            ast::Expression::Map(pairs) => pairs
                .iter()
                .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                .collect(),
            _ => Err(Error::Internal(
                "Expected map expression for properties".into(),
            )),
        }
    }

    /// Extracts properties from a SET clause.
    fn extract_set_properties(
        &self,
        set_clause: &ast::SetClause,
    ) -> Result<Vec<(String, LogicalExpression)>> {
        let mut properties = Vec::new();
        for item in &set_clause.items {
            match item {
                ast::SetItem::Property {
                    variable: _,
                    property,
                    value,
                } => {
                    properties.push((property.clone(), self.translate_expression(value)?));
                }
                ast::SetItem::AllProperties {
                    variable: _,
                    properties: prop_expr,
                } => {
                    // n = {props} - extract all properties from the map
                    if let ast::Expression::Map(pairs) = prop_expr {
                        for (k, v) in pairs {
                            properties.push((k.clone(), self.translate_expression(v)?));
                        }
                    }
                }
                ast::SetItem::MergeProperties {
                    variable: _,
                    properties: prop_expr,
                } => {
                    // n += {props} - merge properties
                    if let ast::Expression::Map(pairs) = prop_expr {
                        for (k, v) in pairs {
                            properties.push((k.clone(), self.translate_expression(v)?));
                        }
                    }
                }
                ast::SetItem::Labels { .. } => {
                    // Labels are handled separately
                }
            }
        }
        Ok(properties)
    }

    fn translate_return(
        &self,
        return_clause: &ast::ReturnClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("RETURN requires input".into()))?;

        // Check if RETURN contains aggregate functions
        let has_aggregates = match &return_clause.items {
            ast::ReturnItems::All => false,
            ast::ReturnItems::Explicit(items) => items
                .iter()
                .any(|item| contains_aggregate(&item.expression)),
        };

        if has_aggregates {
            // Extract aggregates and group-by expressions
            let (aggregates, group_by) = self.extract_aggregates_and_groups(return_clause)?;

            Ok(LogicalOperator::Aggregate(AggregateOp {
                group_by,
                aggregates,
                input: Box::new(input),
            }))
        } else {
            // Normal return without aggregates
            let items = match &return_clause.items {
                ast::ReturnItems::All => {
                    vec![ReturnItem {
                        expression: LogicalExpression::Variable("*".into()),
                        alias: None,
                    }]
                }
                ast::ReturnItems::Explicit(items) => items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<_>>()?,
            };

            Ok(LogicalOperator::Return(ReturnOp {
                items,
                distinct: return_clause.distinct,
                input: Box::new(input),
            }))
        }
    }

    /// Extracts aggregate and group-by expressions from RETURN items.
    fn extract_aggregates_and_groups(
        &self,
        return_clause: &ast::ReturnClause,
    ) -> Result<(Vec<AggregateExpr>, Vec<LogicalExpression>)> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();

        let items = match &return_clause.items {
            ast::ReturnItems::All => {
                return Err(Error::Internal(
                    "Cannot use RETURN * with aggregates".into(),
                ));
            }
            ast::ReturnItems::Explicit(items) => items,
        };

        for item in items {
            if let Some(agg_expr) = self.try_extract_aggregate(&item.expression, &item.alias)? {
                aggregates.push(agg_expr);
            } else {
                // Non-aggregate expressions become group-by keys
                let expr = self.translate_expression(&item.expression)?;
                group_by.push(expr);
            }
        }

        Ok((aggregates, group_by))
    }

    /// Tries to extract an aggregate expression from an AST expression.
    fn try_extract_aggregate(
        &self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        match expr {
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                if let Some(function) = to_aggregate_function(name) {
                    let expression = if args.is_empty() {
                        None
                    } else {
                        Some(self.translate_expression(&args[0])?)
                    };

                    Ok(Some(AggregateExpr {
                        function,
                        expression,
                        distinct: *distinct,
                        alias: alias.clone(),
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn translate_order_by(
        &self,
        order_by: &ast::OrderByClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("ORDER BY requires input".into()))?;

        let keys: Vec<SortKey> = order_by
            .items
            .iter()
            .map(|item| {
                Ok(SortKey {
                    expression: self.translate_expression(&item.expression)?,
                    order: match item.direction {
                        ast::SortDirection::Asc => SortOrder::Ascending,
                        ast::SortDirection::Desc => SortOrder::Descending,
                    },
                })
            })
            .collect::<Result<_>>()?;

        Ok(LogicalOperator::Sort(SortOp {
            keys,
            input: Box::new(input),
        }))
    }

    fn translate_skip(
        &self,
        expr: &ast::Expression,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("SKIP requires input".into()))?;
        let count = self.eval_as_usize(expr)?;

        Ok(LogicalOperator::Skip(SkipOp {
            count,
            input: Box::new(input),
        }))
    }

    fn translate_limit(
        &self,
        expr: &ast::Expression,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("LIMIT requires input".into()))?;
        let count = self.eval_as_usize(expr)?;

        Ok(LogicalOperator::Limit(LimitOp {
            count,
            input: Box::new(input),
        }))
    }

    fn translate_create_clause(
        &self,
        create_clause: &ast::CreateClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut plan = input;

        for pattern in &create_clause.patterns {
            plan = Some(self.translate_create_pattern(pattern, plan)?);
        }

        plan.ok_or_else(|| Error::Internal("Empty CREATE pattern".into()))
    }

    fn translate_create_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node) => {
                let variable = node.variable.clone().unwrap_or_else(|| "_anon".to_string());
                let labels = node.labels.clone();
                let properties: Vec<(String, LogicalExpression)> = node
                    .properties
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;

                Ok(LogicalOperator::CreateNode(CreateNodeOp {
                    variable,
                    labels,
                    properties,
                    input: input.map(Box::new),
                }))
            }
            ast::Pattern::Path(path) => {
                let mut current =
                    self.translate_create_pattern(&ast::Pattern::Node(path.start.clone()), input)?;

                for rel in &path.chain {
                    let from_variable = self.get_last_node_variable(&Some(current.clone()))?;
                    let to_variable = rel
                        .target
                        .variable
                        .clone()
                        .unwrap_or_else(|| "_anon".to_string());
                    let edge_type = rel
                        .types
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "RELATED".to_string());

                    let target_labels = rel.target.labels.clone();
                    let target_props: Vec<(String, LogicalExpression)> = rel
                        .target
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<_>>()?;

                    current = LogicalOperator::CreateNode(CreateNodeOp {
                        variable: to_variable.clone(),
                        labels: target_labels,
                        properties: target_props,
                        input: Some(Box::new(current)),
                    });

                    let edge_props: Vec<(String, LogicalExpression)> = rel
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<_>>()?;

                    current = LogicalOperator::CreateEdge(CreateEdgeOp {
                        variable: rel.variable.clone(),
                        from_variable,
                        to_variable,
                        edge_type,
                        properties: edge_props,
                        input: Box::new(current),
                    });
                }

                Ok(current)
            }
            ast::Pattern::NamedPath { pattern, .. } => {
                self.translate_create_pattern(pattern, input)
            }
        }
    }

    fn translate_create_statement(&self, create: &ast::CreateClause) -> Result<LogicalPlan> {
        let mut plan: Option<LogicalOperator> = None;

        for pattern in &create.patterns {
            plan = Some(self.translate_create_pattern(pattern, plan)?);
        }

        let root = plan.ok_or_else(|| Error::Internal("Empty CREATE".into()))?;
        Ok(LogicalPlan::new(root))
    }

    fn translate_delete(
        &self,
        delete_clause: &ast::DeleteClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("DELETE requires input".into()))?;

        let mut plan = input;

        // Delete each expression (typically variables)
        for expr in &delete_clause.expressions {
            if let ast::Expression::Variable(var) = expr {
                // Check if it's a node or edge - for simplicity, try node first
                plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                    variable: var.clone(),
                    input: Box::new(plan),
                });
            } else {
                return Err(Error::Internal(
                    "DELETE only supports variable expressions".into(),
                ));
            }
        }

        Ok(plan)
    }

    fn translate_set(
        &self,
        set_clause: &ast::SetClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("SET requires input".into()))?;

        let mut plan = input;

        // Group items by variable
        for item in &set_clause.items {
            match item {
                ast::SetItem::Property {
                    variable,
                    property,
                    value,
                } => {
                    // SET n.prop = value
                    let value_expr = self.translate_expression(value)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![(property.clone(), value_expr)],
                        replace: false,
                        input: Box::new(plan),
                    });
                }
                ast::SetItem::AllProperties {
                    variable,
                    properties,
                } => {
                    // SET n = {...} or SET n = m
                    let value_expr = self.translate_expression(properties)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![("*".to_string(), value_expr)],
                        replace: true,
                        input: Box::new(plan),
                    });
                }
                ast::SetItem::MergeProperties {
                    variable,
                    properties,
                } => {
                    // SET n += {...}
                    let value_expr = self.translate_expression(properties)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![("*".to_string(), value_expr)],
                        replace: false,
                        input: Box::new(plan),
                    });
                }
                ast::SetItem::Labels { .. } => {
                    return Err(Error::Internal("SET labels not yet supported".into()));
                }
            }
        }

        Ok(plan)
    }

    fn translate_remove(
        &self,
        remove_clause: &ast::RemoveClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| Error::Internal("REMOVE requires input".into()))?;

        let mut plan = input;

        for item in &remove_clause.items {
            match item {
                ast::RemoveItem::Property { variable, property } => {
                    // REMOVE n.prop sets the property to null
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![(
                            property.clone(),
                            LogicalExpression::Literal(Value::Null),
                        )],
                        replace: false,
                        input: Box::new(plan),
                    });
                }
                ast::RemoveItem::Labels { variable, labels } => {
                    // REMOVE n:Label removes labels from the node
                    plan = LogicalOperator::RemoveLabel(RemoveLabelOp {
                        variable: variable.clone(),
                        labels: labels.clone(),
                        input: Box::new(plan),
                    });
                }
            }
        }

        Ok(plan)
    }

    fn translate_expression(&self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => self.translate_literal(lit),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { base, property } => {
                if let ast::Expression::Variable(var) = base.as_ref() {
                    Ok(LogicalExpression::Property {
                        variable: var.clone(),
                        property: property.clone(),
                    })
                } else {
                    Err(Error::Internal(
                        "Nested property access not supported".into(),
                    ))
                }
            }
            ast::Expression::IndexAccess { base, index } => {
                let base_expr = self.translate_expression(base)?;
                let index_expr = self.translate_expression(index)?;
                Ok(LogicalExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            ast::Expression::SliceAccess { base, start, end } => {
                let base_expr = self.translate_expression(base)?;
                let start_expr = start
                    .as_ref()
                    .map(|s| self.translate_expression(s))
                    .transpose()?
                    .map(Box::new);
                let end_expr = end
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(LogicalExpression::SliceAccess {
                    base: Box::new(base_expr),
                    start: start_expr,
                    end: end_expr,
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let left_expr = self.translate_expression(left)?;
                let right_expr = self.translate_expression(right)?;
                let binary_op = self.translate_binary_op(op)?;

                Ok(LogicalExpression::Binary {
                    left: Box::new(left_expr),
                    op: binary_op,
                    right: Box::new(right_expr),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand_expr = self.translate_expression(operand)?;
                let unary_op = self.translate_unary_op(op)?;

                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(operand_expr),
                })
            }
            ast::Expression::FunctionCall { name, args, .. } => {
                let translated_args: Vec<LogicalExpression> = args
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                })
            }
            ast::Expression::List(items) => {
                let translated: Vec<LogicalExpression> = items
                    .iter()
                    .map(|i| self.translate_expression(i))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::List(translated))
            }
            ast::Expression::Map(pairs) => {
                let translated: Vec<(String, LogicalExpression)> = pairs
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;
                Ok(LogicalExpression::Map(translated))
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let translated_operand = if let Some(op) = input {
                    Some(Box::new(self.translate_expression(op)?))
                } else {
                    None
                };

                let translated_when: Vec<(LogicalExpression, LogicalExpression)> = whens
                    .iter()
                    .map(|(when, then)| {
                        Ok((
                            self.translate_expression(when)?,
                            self.translate_expression(then)?,
                        ))
                    })
                    .collect::<Result<_>>()?;

                let translated_else = if let Some(el) = else_clause {
                    Some(Box::new(self.translate_expression(el)?))
                } else {
                    None
                };

                Ok(LogicalExpression::Case {
                    operand: translated_operand,
                    when_clauses: translated_when,
                    else_clause: translated_else,
                })
            }
            ast::Expression::ListComprehension {
                variable,
                list,
                filter,
                projection,
            } => {
                let list_expr = self.translate_expression(list)?;
                let filter_expr = filter
                    .as_ref()
                    .map(|f| self.translate_expression(f))
                    .transpose()?
                    .map(Box::new);
                // If no projection, use the variable itself as the map expression
                let map_expr = if let Some(proj) = projection {
                    self.translate_expression(proj)?
                } else {
                    LogicalExpression::Variable(variable.clone())
                };

                Ok(LogicalExpression::ListComprehension {
                    variable: variable.clone(),
                    list_expr: Box::new(list_expr),
                    filter_expr,
                    map_expr: Box::new(map_expr),
                })
            }
            ast::Expression::PatternComprehension { .. } => Err(Error::Internal(
                "Pattern comprehension not yet supported".into(),
            )),
            ast::Expression::Exists(_) => Err(Error::Internal("EXISTS not yet supported".into())),
            ast::Expression::CountSubquery(_) => {
                Err(Error::Internal("COUNT subquery not yet supported".into()))
            }
        }
    }

    fn translate_literal(&self, lit: &ast::Literal) -> Result<LogicalExpression> {
        let value = match lit {
            ast::Literal::Null => Value::Null,
            ast::Literal::Bool(b) => Value::Bool(*b),
            ast::Literal::Integer(i) => Value::Int64(*i),
            ast::Literal::Float(f) => Value::Float64(*f),
            ast::Literal::String(s) => Value::from(s.as_str()),
        };
        Ok(LogicalExpression::Literal(value))
    }

    fn translate_binary_op(&self, op: &ast::BinaryOp) -> Result<BinaryOp> {
        Ok(match op {
            ast::BinaryOp::Eq => BinaryOp::Eq,
            ast::BinaryOp::Ne => BinaryOp::Ne,
            ast::BinaryOp::Lt => BinaryOp::Lt,
            ast::BinaryOp::Le => BinaryOp::Le,
            ast::BinaryOp::Gt => BinaryOp::Gt,
            ast::BinaryOp::Ge => BinaryOp::Ge,
            ast::BinaryOp::And => BinaryOp::And,
            ast::BinaryOp::Or => BinaryOp::Or,
            ast::BinaryOp::Xor => BinaryOp::Xor,
            ast::BinaryOp::Add => BinaryOp::Add,
            ast::BinaryOp::Sub => BinaryOp::Sub,
            ast::BinaryOp::Mul => BinaryOp::Mul,
            ast::BinaryOp::Div => BinaryOp::Div,
            ast::BinaryOp::Mod => BinaryOp::Mod,
            ast::BinaryOp::Pow => {
                return Err(Error::Internal("Power operator not yet supported".into()));
            }
            ast::BinaryOp::Concat => BinaryOp::Concat,
            ast::BinaryOp::StartsWith => BinaryOp::StartsWith,
            ast::BinaryOp::EndsWith => BinaryOp::EndsWith,
            ast::BinaryOp::Contains => BinaryOp::Contains,
            ast::BinaryOp::RegexMatch => BinaryOp::Regex,
            ast::BinaryOp::In => BinaryOp::In,
        })
    }

    fn translate_unary_op(&self, op: &ast::UnaryOp) -> Result<UnaryOp> {
        Ok(match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Neg => UnaryOp::Neg,
            ast::UnaryOp::Pos => {
                return Err(Error::Internal("Unary positive not yet supported".into()));
            }
            ast::UnaryOp::IsNull => UnaryOp::IsNull,
            ast::UnaryOp::IsNotNull => UnaryOp::IsNotNull,
        })
    }

    fn eval_as_usize(&self, expr: &ast::Expression) -> Result<usize> {
        match expr {
            ast::Expression::Literal(ast::Literal::Integer(i)) => {
                if *i >= 0 {
                    Ok(*i as usize)
                } else {
                    Err(Error::Internal("Expected non-negative integer".into()))
                }
            }
            _ => Err(Error::Internal("Expected integer literal".into())),
        }
    }

    fn get_last_variable(plan: &LogicalOperator) -> Result<String> {
        match plan {
            LogicalOperator::NodeScan(scan) => Ok(scan.variable.clone()),
            LogicalOperator::Expand(expand) => Ok(expand.to_variable.clone()),
            LogicalOperator::Filter(filter) => Self::get_last_variable(&filter.input),
            LogicalOperator::Project(project) => Self::get_last_variable(&project.input),
            _ => Err(Error::Internal("Cannot get variable from operator".into())),
        }
    }

    fn get_last_node_variable(&self, plan: &Option<LogicalOperator>) -> Result<String> {
        match plan {
            Some(LogicalOperator::CreateNode(node)) => Ok(node.variable.clone()),
            Some(LogicalOperator::NodeScan(scan)) => Ok(scan.variable.clone()),
            Some(LogicalOperator::CreateEdge(edge)) => Ok(edge.to_variable.clone()),
            Some(other) => self.get_last_node_variable(&self.extract_input(other)),
            None => Err(Error::Internal("No previous node variable".into())),
        }
    }

    fn extract_input(&self, plan: &LogicalOperator) -> Option<LogicalOperator> {
        match plan {
            LogicalOperator::CreateNode(n) => n.input.as_ref().map(|b| b.as_ref().clone()),
            LogicalOperator::CreateEdge(e) => Some(e.input.as_ref().clone()),
            LogicalOperator::Filter(f) => Some(f.input.as_ref().clone()),
            _ => None,
        }
    }
}

/// Checks if an AST expression contains an aggregate function call.
fn contains_aggregate(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::FunctionCall { name, .. } => is_aggregate_function(name),
        ast::Expression::Binary { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        ast::Expression::Unary { operand, .. } => contains_aggregate(operand),
        _ => false,
    }
}

/// Returns true if the function name is an aggregate function.
fn is_aggregate_function(name: &str) -> bool {
    matches!(
        name.to_uppercase().as_str(),
        "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "COLLECT"
    )
}

/// Converts a function name to an AggregateFunction enum.
fn to_aggregate_function(name: &str) -> Option<AggregateFunction> {
    match name.to_uppercase().as_str() {
        "COUNT" => Some(AggregateFunction::Count),
        "SUM" => Some(AggregateFunction::Sum),
        "AVG" => Some(AggregateFunction::Avg),
        "MIN" => Some(AggregateFunction::Min),
        "MAX" => Some(AggregateFunction::Max),
        "COLLECT" => Some(AggregateFunction::Collect),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Basic MATCH Tests ===

    #[test]
    fn test_translate_simple_match() {
        let plan = translate("MATCH (n:Person) RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalOperator::NodeScan(scan) = ret.input.as_ref() {
                assert_eq!(scan.variable, "n");
                assert_eq!(scan.label, Some("Person".into()));
            } else {
                panic!("Expected NodeScan");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_with_where() {
        let plan = translate("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Gt);
                }
            } else {
                panic!("Expected Filter");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_return_distinct() {
        let plan = translate("MATCH (n:Person) RETURN DISTINCT n.name").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert!(ret.distinct);
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_return_all() {
        let plan = translate("MATCH (n:Person) RETURN *").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalExpression::Variable(v) = &ret.items[0].expression {
                assert_eq!(v, "*");
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === Path Pattern Tests ===

    #[test]
    fn test_translate_outgoing_relationship() {
        let plan = translate("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b").unwrap();

        // Find the Expand operator
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Outgoing);
        assert_eq!(expand.edge_type.as_deref(), Some("KNOWS"));
    }

    #[test]
    fn test_translate_incoming_relationship() {
        let plan = translate("MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a, b").unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Incoming);
    }

    #[test]
    fn test_translate_variable_length_path() {
        let plan = translate("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) RETURN a, b").unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.min_hops, 1);
        assert_eq!(expand.max_hops, Some(3));
    }

    // === Mutation Tests ===

    #[test]
    fn test_translate_create_node() {
        let plan = translate("CREATE (n:Person {name: 'Alice'})").unwrap();

        if let LogicalOperator::CreateNode(create) = &plan.root {
            assert_eq!(create.variable, "n");
            assert_eq!(create.labels, vec!["Person".to_string()]);
            assert_eq!(create.properties.len(), 1);
            assert_eq!(create.properties[0].0, "name");
        } else {
            panic!("Expected CreateNode, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_create_path() {
        let plan = translate("CREATE (a:Person)-[:KNOWS]->(b:Person)").unwrap();

        // Should have CreateEdge at root
        if let LogicalOperator::CreateEdge(edge) = &plan.root {
            assert_eq!(edge.edge_type, "KNOWS");
            // Input should be CreateNode for b
            if let LogicalOperator::CreateNode(node_b) = edge.input.as_ref() {
                assert_eq!(node_b.variable, "b");
            }
        } else {
            panic!("Expected CreateEdge, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_delete_node() {
        let plan = translate("MATCH (n:Person) DELETE n").unwrap();

        if let LogicalOperator::DeleteNode(delete) = &plan.root {
            assert_eq!(delete.variable, "n");
            if let LogicalOperator::NodeScan(scan) = delete.input.as_ref() {
                assert_eq!(scan.variable, "n");
                assert_eq!(scan.label, Some("Person".into()));
            } else {
                panic!("Expected NodeScan input");
            }
        } else {
            panic!("Expected DeleteNode, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_set_property() {
        let plan = translate("MATCH (n:Person) SET n.name = 'Bob' RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::SetProperty(set) = ret.input.as_ref() {
                assert_eq!(set.variable, "n");
                assert_eq!(set.properties.len(), 1);
                assert_eq!(set.properties[0].0, "name");
                assert!(!set.replace);
            } else {
                panic!("Expected SetProperty");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_set_multiple_properties() {
        let plan = translate("MATCH (n:Person) SET n.name = 'Alice', n.age = 30 RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            // SET creates chained SetProperty operators
            if let LogicalOperator::SetProperty(set2) = ret.input.as_ref() {
                if let LogicalOperator::SetProperty(set1) = set2.input.as_ref() {
                    // Properties are set in order
                    assert_eq!(set1.properties[0].0, "name");
                    assert_eq!(set2.properties[0].0, "age");
                } else {
                    panic!("Expected nested SetProperty");
                }
            } else {
                panic!("Expected SetProperty");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_remove_property() {
        let plan = translate("MATCH (n:Person) REMOVE n.name RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::SetProperty(set) = ret.input.as_ref() {
                // REMOVE property is translated to SET property = null
                assert_eq!(set.variable, "n");
                assert_eq!(set.properties.len(), 1);
                assert_eq!(set.properties[0].0, "name");
                // Value should be Null
                if let LogicalExpression::Literal(Value::Null) = &set.properties[0].1 {
                    // OK
                } else {
                    panic!("Expected Null value for REMOVE");
                }
            } else {
                panic!("Expected SetProperty");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_remove_label() {
        let plan = translate("MATCH (n:Person:Admin) REMOVE n:Admin RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::RemoveLabel(remove) = ret.input.as_ref() {
                assert_eq!(remove.variable, "n");
                assert_eq!(remove.labels, vec!["Admin".to_string()]);
            } else {
                panic!("Expected RemoveLabel");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    // === WITH, UNWIND, ORDER BY, SKIP, LIMIT Tests ===

    #[test]
    fn test_translate_with_clause() {
        let plan = translate("MATCH (n:Person) WITH n.name AS name RETURN name").unwrap();

        // Find Project operator
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                LogicalOperator::Return(r) => find_project(&r.input),
                LogicalOperator::Filter(f) => find_project(&f.input),
                _ => None,
            }
        }

        let project = find_project(&plan.root).expect("Expected Project");
        assert_eq!(project.projections.len(), 1);
        assert_eq!(project.projections[0].alias.as_deref(), Some("name"));
    }

    #[test]
    fn test_translate_with_distinct() {
        let plan = translate("MATCH (n:Person) WITH DISTINCT n.city AS city RETURN city").unwrap();

        // Find Distinct operator
        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Return(r) => find_distinct(&r.input),
                LogicalOperator::Project(p) => find_distinct(&p.input),
                LogicalOperator::Filter(f) => find_distinct(&f.input),
                _ => false,
            }
        }

        assert!(find_distinct(&plan.root));
    }

    #[test]
    fn test_translate_unwind() {
        let plan = translate("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        // Find Unwind operator
        fn find_unwind(op: &LogicalOperator) -> Option<&UnwindOp> {
            match op {
                LogicalOperator::Unwind(u) => Some(u),
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => None,
            }
        }

        let unwind = find_unwind(&plan.root).expect("Expected Unwind");
        assert_eq!(unwind.variable, "x");
    }

    #[test]
    fn test_translate_order_by() {
        let plan = translate("MATCH (n:Person) RETURN n ORDER BY n.name").unwrap();

        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Return(r) => find_sort(&r.input),
                _ => None,
            }
        }

        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys.len(), 1);
        assert_eq!(sort.keys[0].order, SortOrder::Ascending);
    }

    #[test]
    fn test_translate_order_by_desc() {
        let plan = translate("MATCH (n:Person) RETURN n ORDER BY n.age DESC").unwrap();

        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Return(r) => find_sort(&r.input),
                _ => None,
            }
        }

        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys[0].order, SortOrder::Descending);
    }

    #[test]
    fn test_translate_limit() {
        let plan = translate("MATCH (n:Person) RETURN n LIMIT 10").unwrap();

        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                LogicalOperator::Return(r) => find_limit(&r.input),
                _ => None,
            }
        }

        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 10);
    }

    #[test]
    fn test_translate_skip() {
        let plan = translate("MATCH (n:Person) RETURN n SKIP 5").unwrap();

        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Return(r) => find_skip(&r.input),
                LogicalOperator::Limit(l) => find_skip(&l.input),
                _ => None,
            }
        }

        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    // === MERGE Tests ===

    #[test]
    fn test_translate_merge() {
        let plan = translate("MERGE (n:Person {name: 'Alice'})").unwrap();

        if let LogicalOperator::Merge(merge) = &plan.root {
            assert_eq!(merge.variable, "n");
            assert_eq!(merge.labels, vec!["Person".to_string()]);
            assert_eq!(merge.match_properties.len(), 1);
            assert_eq!(merge.match_properties[0].0, "name");
        } else {
            panic!("Expected Merge, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_merge_on_create() {
        let plan =
            translate("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.created = true").unwrap();

        if let LogicalOperator::Merge(merge) = &plan.root {
            assert_eq!(merge.on_create.len(), 1);
            assert_eq!(merge.on_create[0].0, "created");
        } else {
            panic!("Expected Merge, got {:?}", plan.root);
        }
    }

    // === Expression Tests ===

    #[test]
    fn test_translate_list_expression() {
        // Cypher requires MATCH before RETURN, so use UNWIND to test list
        let plan = translate("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        fn find_unwind(op: &LogicalOperator) -> Option<&UnwindOp> {
            match op {
                LogicalOperator::Unwind(u) => Some(u),
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => None,
            }
        }

        let unwind = find_unwind(&plan.root).expect("Expected Unwind");
        if let LogicalExpression::List(items) = &unwind.expression {
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected List expression");
        }
    }

    #[test]
    fn test_translate_map_expression() {
        // Test map in CREATE with properties
        let plan = translate("CREATE (n:Person {name: 'Alice', age: 30})").unwrap();

        if let LogicalOperator::CreateNode(create) = &plan.root {
            assert_eq!(create.properties.len(), 2);
        } else {
            panic!("Expected CreateNode");
        }
    }

    #[test]
    fn test_translate_function_call() {
        // Use toUpper which is a simple function
        let plan = translate("MATCH (n:Person) RETURN toUpper(n.name)").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalExpression::FunctionCall { name, args } = &ret.items[0].expression {
                assert_eq!(name.to_lowercase(), "toupper");
                assert_eq!(args.len(), 1);
            } else {
                panic!("Expected FunctionCall, got {:?}", ret.items[0].expression);
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_case_expression() {
        let plan =
            translate("MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END")
                .unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalExpression::Case {
                when_clauses,
                else_clause,
                ..
            } = &ret.items[0].expression
            {
                assert_eq!(when_clauses.len(), 1);
                assert!(else_clause.is_some());
            } else {
                panic!("Expected Case expression");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_parameter() {
        let plan = translate("MATCH (n:Person) WHERE n.name = $name RETURN n").unwrap();

        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { right, .. } = &filter.predicate {
            if let LogicalExpression::Parameter(p) = right.as_ref() {
                assert_eq!(p, "name");
            } else {
                panic!("Expected Parameter");
            }
        }
    }

    // === Error Handling Tests ===

    #[test]
    fn test_translate_binary_op_all() {
        let translator = CypherTranslator::new();

        // Test all supported binary ops
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Eq).unwrap(),
            BinaryOp::Eq
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Ne).unwrap(),
            BinaryOp::Ne
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Lt).unwrap(),
            BinaryOp::Lt
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Le).unwrap(),
            BinaryOp::Le
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Gt).unwrap(),
            BinaryOp::Gt
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Ge).unwrap(),
            BinaryOp::Ge
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::And).unwrap(),
            BinaryOp::And
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Or).unwrap(),
            BinaryOp::Or
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Xor).unwrap(),
            BinaryOp::Xor
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Add).unwrap(),
            BinaryOp::Add
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Sub).unwrap(),
            BinaryOp::Sub
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Mul).unwrap(),
            BinaryOp::Mul
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Div).unwrap(),
            BinaryOp::Div
        );
        assert_eq!(
            translator.translate_binary_op(&ast::BinaryOp::Mod).unwrap(),
            BinaryOp::Mod
        );
    }

    #[test]
    fn test_translate_unary_op_all() {
        let translator = CypherTranslator::new();

        assert_eq!(
            translator.translate_unary_op(&ast::UnaryOp::Not).unwrap(),
            UnaryOp::Not
        );
        assert_eq!(
            translator.translate_unary_op(&ast::UnaryOp::Neg).unwrap(),
            UnaryOp::Neg
        );
        assert_eq!(
            translator
                .translate_unary_op(&ast::UnaryOp::IsNull)
                .unwrap(),
            UnaryOp::IsNull
        );
        assert_eq!(
            translator
                .translate_unary_op(&ast::UnaryOp::IsNotNull)
                .unwrap(),
            UnaryOp::IsNotNull
        );
    }

    #[test]
    fn test_translate_literal_types() {
        let translator = CypherTranslator::new();

        // Test all literal types
        let null_lit = translator.translate_literal(&ast::Literal::Null).unwrap();
        assert!(matches!(null_lit, LogicalExpression::Literal(Value::Null)));

        let bool_lit = translator
            .translate_literal(&ast::Literal::Bool(true))
            .unwrap();
        assert!(matches!(
            bool_lit,
            LogicalExpression::Literal(Value::Bool(true))
        ));

        let int_lit = translator
            .translate_literal(&ast::Literal::Integer(42))
            .unwrap();
        assert!(matches!(
            int_lit,
            LogicalExpression::Literal(Value::Int64(42))
        ));

        let float_lit = translator
            .translate_literal(&ast::Literal::Float(3.14))
            .unwrap();
        if let LogicalExpression::Literal(Value::Float64(f)) = float_lit {
            assert!((f - 3.14).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }
    }
}
