//! GraphQL to LogicalPlan translator.
//!
//! Translates GraphQL queries to the common logical plan representation for LPG.
//!
//! # Mapping Strategy
//!
//! GraphQL's hierarchical selection model maps to LPG traversals:
//! - Root fields → NodeScan (field name is the type/label)
//! - Field arguments → Filter predicates
//! - Nested selections → Expand (field name is relationship type)
//! - Scalar fields → Return projections

use super::common::{
    VarGen, capitalize_first, graphql_directives_allow, wrap_filter, wrap_limit, wrap_return,
    wrap_skip, wrap_sort,
};
use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, BinaryOp, CountExpr, CreateNodeOp, DeleteNodeOp,
    ExpandDirection, ExpandOp, LogicalExpression, LogicalOperator, LogicalPlan, NodeScanOp,
    PathMode, ReturnItem, SetPropertyOp, SortKey, SortOrder, UnionOp,
};
use grafeo_adapters::query::graphql::{self, ast};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use std::collections::HashMap;

/// Translates a GraphQL query string to a logical plan.
///
/// Supports an `EXPLAIN` or `EXPLAIN ANALYZE` prefix (case-insensitive,
/// non-standard extension) to request plan-only or profiled execution.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let trimmed = query.trim_start();
    let (explain, profile, actual_query) = if trimmed
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("EXPLAIN"))
        && trimmed
            .as_bytes()
            .get(7)
            .is_some_and(u8::is_ascii_whitespace)
    {
        let rest = trimmed[7..].trim_start();
        if rest
            .get(..7)
            .is_some_and(|s| s.eq_ignore_ascii_case("ANALYZE"))
            && rest.as_bytes().get(7).is_some_and(u8::is_ascii_whitespace)
        {
            (false, true, rest[7..].trim_start())
        } else {
            (true, false, rest)
        }
    } else {
        (false, false, query)
    };

    let doc = graphql::parse(actual_query)?;
    let translator = GraphQLTranslator::new();
    let mut plan = translator.translate_document(&doc)?;
    plan.explain = explain;
    plan.profile = profile;
    Ok(plan)
}

/// Mutation type for GraphQL mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MutationType {
    Create,
    Update,
    Delete,
}

/// Extracted special arguments from a field.
struct ExtractedArgs<'a> {
    /// Pagination: first (limit)
    first: Option<CountExpr>,
    /// Pagination: skip (offset)
    skip: Option<CountExpr>,
    /// Sort keys from orderBy
    order_by: Option<Vec<SortKey>>,
    /// Remaining filter arguments
    filters: Vec<&'a ast::Argument>,
}

/// Translator from GraphQL AST to LogicalPlan.
struct GraphQLTranslator {
    /// Generator for anonymous variable names.
    var_gen: VarGen,
    /// Fragment definitions for resolution.
    fragments: HashMap<String, ast::FragmentDefinition>,
    /// Default values from variable declarations (e.g., `query($limit: Int = 2)`).
    variable_defaults: HashMap<String, grafeo_common::types::Value>,
}

impl GraphQLTranslator {
    fn new() -> Self {
        Self {
            var_gen: VarGen::new(),
            fragments: HashMap::new(),
            variable_defaults: HashMap::new(),
        }
    }

    fn translate_document(&self, doc: &ast::Document) -> Result<LogicalPlan> {
        // First, collect all fragment definitions
        let mut fragments = HashMap::new();
        for def in &doc.definitions {
            if let ast::Definition::Fragment(frag) = def {
                fragments.insert(frag.name.clone(), frag.clone());
            }
        }

        // Find the first operation
        let operation = doc
            .definitions
            .iter()
            .find_map(|def| match def {
                ast::Definition::Operation(op) => Some(op),
                _ => None,
            })
            .ok_or_else(|| {
                Error::Query(QueryError::new(
                    QueryErrorKind::Syntax,
                    "No operation found in document",
                ))
            })?;

        // Collect default values from variable declarations
        let variable_defaults: HashMap<String, grafeo_common::types::Value> = operation
            .variables
            .iter()
            .filter_map(|v| {
                v.default_value
                    .as_ref()
                    .map(|dv| (v.name.clone(), dv.clone()))
            })
            .collect();

        // Create translator with fragments and variable defaults
        let translator = GraphQLTranslator {
            var_gen: VarGen::new(),
            fragments,
            variable_defaults,
        };

        translator.translate_operation(operation)
    }

    fn translate_operation(&self, op: &ast::OperationDefinition) -> Result<LogicalPlan> {
        match op.operation {
            ast::OperationType::Query => self.translate_query(op),
            ast::OperationType::Mutation => self.translate_mutation(op),
            ast::OperationType::Subscription => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Subscriptions are not supported",
            ))),
        }
    }

    fn translate_query(&self, op: &ast::OperationDefinition) -> Result<LogicalPlan> {
        let selections = &op.selection_set.selections;
        if selections.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Syntax,
                "Empty selection set",
            )));
        }

        // Collect all root fields from the selection set
        let root_fields: Vec<&ast::Field> = selections
            .iter()
            .filter_map(|sel| {
                if let ast::Selection::Field(field) = sel {
                    Some(field)
                } else {
                    None
                }
            })
            .collect();

        if root_fields.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Syntax,
                "No field found in selection set",
            )));
        }

        // Translate each root field independently
        let mut plans: Vec<LogicalOperator> = Vec::with_capacity(root_fields.len());
        for field in &root_fields {
            plans.push(self.translate_root_field(field)?);
        }

        // If there is only one root field, use it directly; otherwise union them
        let root = if plans.len() == 1 {
            plans.remove(0)
        } else {
            LogicalOperator::Union(UnionOp { inputs: plans })
        };

        let mut logical_plan = LogicalPlan::new(root);
        logical_plan
            .default_params
            .clone_from(&self.variable_defaults);
        Ok(logical_plan)
    }

    fn translate_mutation(&self, op: &ast::OperationDefinition) -> Result<LogicalPlan> {
        let field = self.get_first_field(&op.selection_set)?;

        // Parse mutation type from field name
        let (mutation_type, type_name) = self.parse_mutation_name(&field.name)?;

        match mutation_type {
            MutationType::Create => self.translate_create_mutation(field, &type_name),
            MutationType::Update => self.translate_update_mutation(field, &type_name),
            MutationType::Delete => self.translate_delete_mutation(field, &type_name),
        }
    }

    fn parse_mutation_name(&self, name: &str) -> Result<(MutationType, String)> {
        if let Some(type_name) = name.strip_prefix("create") {
            Ok((MutationType::Create, type_name.to_string()))
        } else if let Some(type_name) = name.strip_prefix("update") {
            Ok((MutationType::Update, type_name.to_string()))
        } else if let Some(type_name) = name.strip_prefix("delete") {
            Ok((MutationType::Delete, type_name.to_string()))
        } else {
            Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "Unknown mutation: {}. Expected createX, updateX, or deleteX",
                    name
                ),
            )))
        }
    }

    fn translate_create_mutation(
        &self,
        field: &ast::Field,
        type_name: &str,
    ) -> Result<LogicalPlan> {
        let var = self.var_gen.next();

        // Convert arguments to properties
        let properties: Vec<(String, LogicalExpression)> = field
            .arguments
            .iter()
            .map(|arg| (arg.name.clone(), self.input_value_to_expression(&arg.value)))
            .collect();

        let mut plan = LogicalOperator::CreateNode(CreateNodeOp {
            variable: var.clone(),
            labels: vec![capitalize_first(type_name)],
            properties,
            input: None,
        });

        // If there's a selection set, return the created node's properties
        plan = if let Some(selection_set) = &field.selection_set {
            self.translate_selection_set(selection_set, plan, &var)?
        } else {
            wrap_return(
                plan,
                vec![ReturnItem {
                    expression: LogicalExpression::Variable(var),
                    alias: None,
                }],
                false,
            )
        };

        Ok(LogicalPlan::new(plan))
    }

    fn translate_update_mutation(
        &self,
        field: &ast::Field,
        type_name: &str,
    ) -> Result<LogicalPlan> {
        let var = self.var_gen.next();

        // Need at least 2 arguments: one for filter, one for update
        if field.arguments.len() < 2 {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Update mutation requires a filter argument and at least one property to update",
            )));
        }

        // Determine filter: prefer 'id', otherwise use first argument
        let (filter_arg_name, filter_predicate) =
            if let Some(id_arg) = field.arguments.iter().find(|arg| arg.name == "id") {
                // Filter by id
                (
                    "id".to_string(),
                    LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Id(var.clone())),
                        op: BinaryOp::Eq,
                        right: Box::new(LogicalExpression::Literal(id_arg.value.to_value())),
                    },
                )
            } else {
                // Use first argument as property filter
                let first_arg = &field.arguments[0];
                (
                    first_arg.name.clone(),
                    LogicalExpression::Binary {
                        left: Box::new(LogicalExpression::Property {
                            variable: var.clone(),
                            property: first_arg.name.clone(),
                        }),
                        op: BinaryOp::Eq,
                        right: Box::new(LogicalExpression::Literal(first_arg.value.to_value())),
                    },
                )
            };

        // Collect properties to update (all arguments except the filter argument)
        let properties: Vec<(String, LogicalExpression)> = field
            .arguments
            .iter()
            .filter(|arg| arg.name != filter_arg_name)
            .map(|arg| (arg.name.clone(), self.input_value_to_expression(&arg.value)))
            .collect();

        if properties.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Update mutation requires at least one property to update",
            )));
        }

        // Start with a node scan for the type
        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: var.clone(),
            label: Some(capitalize_first(type_name)),
            input: None,
        });

        // Apply filter
        plan = wrap_filter(plan, filter_predicate);

        // Set the properties
        plan = LogicalOperator::SetProperty(SetPropertyOp {
            variable: var.clone(),
            properties,
            replace: false, // Merge properties, don't replace all
            is_edge: false,
            input: Box::new(plan),
        });

        // If there's a selection set, return the updated node's properties
        plan = if let Some(selection_set) = &field.selection_set {
            self.translate_selection_set(selection_set, plan, &var)?
        } else {
            wrap_return(
                plan,
                vec![ReturnItem {
                    expression: LogicalExpression::Variable(var),
                    alias: None,
                }],
                false,
            )
        };

        Ok(LogicalPlan::new(plan))
    }

    fn translate_delete_mutation(
        &self,
        field: &ast::Field,
        type_name: &str,
    ) -> Result<LogicalPlan> {
        let var = self.var_gen.next();

        // Need at least 1 argument for filter
        if field.arguments.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Delete mutation requires a filter argument (id or property)",
            )));
        }

        // Determine filter: prefer 'id', otherwise use first argument
        let filter_predicate =
            if let Some(id_arg) = field.arguments.iter().find(|arg| arg.name == "id") {
                // Filter by id
                LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Id(var.clone())),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(id_arg.value.to_value())),
                }
            } else {
                // Use first argument as property filter
                let first_arg = &field.arguments[0];
                LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: var.clone(),
                        property: first_arg.name.clone(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(first_arg.value.to_value())),
                }
            };

        // First scan for the node
        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: var.clone(),
            label: Some(capitalize_first(type_name)),
            input: None,
        });

        // Apply filter
        plan = wrap_filter(plan, filter_predicate);

        // Delete the node (GraphQL mutations are like DETACH DELETE)
        plan = LogicalOperator::DeleteNode(DeleteNodeOp {
            variable: var,
            detach: true,
            input: Box::new(plan),
        });

        Ok(LogicalPlan::new(plan))
    }

    fn translate_root_field(&self, field: &ast::Field) -> Result<LogicalOperator> {
        // Check for aggregate root field patterns:
        //   { personCount }               -> COUNT(*) over Person
        //   { personAggregate { sum_age } } -> SUM(age) over Person
        if let Some(plan) = self.try_translate_aggregate_root(field)? {
            return Ok(plan);
        }

        // Root field name is the type/label to scan
        let var = self.var_gen.next();

        // Start with a node scan using the field name as the label
        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: var.clone(),
            label: Some(capitalize_first(&field.name)),
            input: None,
        });

        // Extract special arguments (pagination, orderBy) from regular filters
        let extracted = self.extract_special_args(&field.arguments, &var);

        // Apply filters (excluding pagination and orderBy)
        if !extracted.filters.is_empty() {
            let filter = self.translate_filter_arguments(&extracted.filters, &var)?;
            plan = wrap_filter(plan, filter);
        }

        // Process nested selection set
        if let Some(selection_set) = &field.selection_set {
            plan = self.translate_selection_set(selection_set, plan, &var)?;
        } else {
            // No nested selection, return the whole node
            plan = wrap_return(
                plan,
                vec![ReturnItem {
                    expression: LogicalExpression::Variable(var),
                    alias: field.alias.clone(),
                }],
                false,
            );
        }

        // Apply ordering (before pagination)
        if let Some(keys) = extracted.order_by {
            plan = wrap_sort(plan, keys);
        }

        // Apply skip BEFORE limit
        if let Some(count) = extracted.skip {
            plan = wrap_skip(plan, count);
        }

        // Apply limit
        if let Some(count) = extracted.first {
            plan = wrap_limit(plan, count);
        }

        Ok(plan)
    }

    /// Extracts special arguments (first, skip, orderBy) from field arguments.
    fn try_translate_aggregate_root(&self, field: &ast::Field) -> Result<Option<LogicalOperator>> {
        let name = &field.name;

        // Pattern 1: <Type>Count (no selection set) -> COUNT(*)
        if let Some(type_name) = name.strip_suffix("Count")
            && !type_name.is_empty()
            && field.selection_set.is_none()
        {
            let var = self.var_gen.next();
            let alias = field.alias.clone().unwrap_or_else(|| name.clone());
            let scan = LogicalOperator::NodeScan(NodeScanOp {
                variable: var,
                label: Some(capitalize_first(type_name)),
                input: None,
            });
            let agg = LogicalOperator::Aggregate(AggregateOp {
                group_by: Vec::new(),
                aggregates: vec![AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: None,
                    expression2: None,
                    distinct: false,
                    alias: Some(alias.clone()),
                    percentile: None,
                    separator: None,
                }],
                input: Box::new(scan),
                having: None,
            });
            let plan = wrap_return(
                agg,
                vec![ReturnItem {
                    expression: LogicalExpression::Variable(alias.clone()),
                    alias: Some(alias),
                }],
                false,
            );
            return Ok(Some(plan));
        }

        // Pattern 2: <Type>Aggregate { sum_<prop>, avg_<prop>, ... }
        if let Some(type_name) = name.strip_suffix("Aggregate")
            && !type_name.is_empty()
            && let Some(selection_set) = &field.selection_set
        {
            let var = self.var_gen.next();
            let scan = LogicalOperator::NodeScan(NodeScanOp {
                variable: var.clone(),
                label: Some(capitalize_first(type_name)),
                input: None,
            });

            let mut agg_exprs = Vec::new();
            let mut return_items = Vec::new();

            for selection in &selection_set.selections {
                if let ast::Selection::Field(sub_field) = selection {
                    if !graphql_directives_allow(&sub_field.directives) {
                        continue;
                    }
                    let alias = sub_field
                        .alias
                        .clone()
                        .unwrap_or_else(|| sub_field.name.clone());

                    if let Some((func, prop)) =
                        parse_aggregate_field(&sub_field.name, &sub_field.arguments)
                    {
                        let expression = prop.map(|p| LogicalExpression::Property {
                            variable: var.clone(),
                            property: p,
                        });
                        agg_exprs.push(AggregateExpr {
                            function: func,
                            expression,
                            expression2: None,
                            distinct: false,
                            alias: Some(alias.clone()),
                            percentile: None,
                            separator: None,
                        });
                        return_items.push(ReturnItem {
                            expression: LogicalExpression::Variable(alias.clone()),
                            alias: Some(alias),
                        });
                    }
                }
            }

            if agg_exprs.is_empty() {
                return Ok(None);
            }

            let agg = LogicalOperator::Aggregate(AggregateOp {
                group_by: Vec::new(),
                aggregates: agg_exprs,
                input: Box::new(scan),
                having: None,
            });
            let plan = wrap_return(agg, return_items, false);
            return Ok(Some(plan));
        }

        Ok(None)
    }

    fn try_translate_aggregate_selection_set(
        &self,
        selection_set: &ast::SelectionSet,
        input: LogicalOperator,
        current_var: &str,
    ) -> Result<Option<LogicalOperator>> {
        let mut agg_exprs: Vec<AggregateExpr> = Vec::new();
        let mut return_items: Vec<ReturnItem> = Vec::new();

        for selection in &selection_set.selections {
            if let ast::Selection::Field(field) = selection {
                if !graphql_directives_allow(&field.directives) {
                    continue;
                }
                let Some((func, prop_name)) = parse_aggregate_field(&field.name, &field.arguments)
                else {
                    return Ok(None);
                };

                let alias = field.alias.clone().unwrap_or_else(|| field.name.clone());
                let expression = prop_name.map(|p| LogicalExpression::Property {
                    variable: current_var.to_string(),
                    property: p,
                });

                agg_exprs.push(AggregateExpr {
                    function: func,
                    expression,
                    expression2: None,
                    distinct: false,
                    alias: Some(alias.clone()),
                    percentile: None,
                    separator: None,
                });
                return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(alias.clone()),
                    alias: Some(alias),
                });
            } else {
                return Ok(None);
            }
        }

        if agg_exprs.is_empty() {
            return Ok(None);
        }

        let agg = LogicalOperator::Aggregate(AggregateOp {
            group_by: Vec::new(),
            aggregates: agg_exprs,
            input: Box::new(input),
            having: None,
        });
        Ok(Some(wrap_return(agg, return_items, false)))
    }

    fn extract_special_args<'a>(&self, args: &'a [ast::Argument], var: &str) -> ExtractedArgs<'a> {
        let mut first = None;
        let mut skip = None;
        let mut order_by = None;
        let mut filters = Vec::new();

        for arg in args {
            match arg.name.as_str() {
                "first" | "limit" => match &arg.value {
                    ast::InputValue::Int(n) => {
                        // reason: GraphQL first/limit values are non-negative in practice
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let count = *n as usize;
                        first = Some(CountExpr::Literal(count));
                    }
                    ast::InputValue::Variable(name) => {
                        first = Some(CountExpr::Parameter(name.clone()));
                    }
                    _ => {}
                },
                "skip" | "offset" => match &arg.value {
                    ast::InputValue::Int(n) => {
                        // reason: GraphQL skip/offset values are non-negative in practice
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let count = *n as usize;
                        skip = Some(CountExpr::Literal(count));
                    }
                    ast::InputValue::Variable(name) => {
                        skip = Some(CountExpr::Parameter(name.clone()));
                    }
                    _ => {}
                },
                "orderBy" => {
                    if let ast::InputValue::Object(fields) = &arg.value {
                        let keys: Vec<SortKey> = fields
                            .iter()
                            .map(|(field, dir)| {
                                let order = match dir {
                                    ast::InputValue::Enum(s) if s == "DESC" => {
                                        SortOrder::Descending
                                    }
                                    _ => SortOrder::Ascending,
                                };
                                SortKey {
                                    expression: LogicalExpression::Property {
                                        variable: var.to_string(),
                                        property: field.clone(),
                                    },
                                    order,
                                    nulls: None,
                                }
                            })
                            .collect();
                        order_by = Some(keys);
                    }
                }
                _ => filters.push(arg),
            }
        }

        ExtractedArgs {
            first,
            skip,
            order_by,
            filters,
        }
    }

    fn translate_selection_set(
        &self,
        selection_set: &ast::SelectionSet,
        input: LogicalOperator,
        current_var: &str,
    ) -> Result<LogicalOperator> {
        // Try aggregate selection set first (all fields are aggregate functions)
        if let Some(plan) =
            self.try_translate_aggregate_selection_set(selection_set, input.clone(), current_var)?
        {
            return Ok(plan);
        }

        // Collect all return items and build the plan
        let (plan, return_items) =
            self.collect_selection_items(selection_set, input, current_var)?;

        // Wrap in Return if we have items
        if !return_items.is_empty() {
            Ok(wrap_return(plan, return_items, false))
        } else {
            Ok(plan)
        }
    }

    /// Collects return items from a selection set without wrapping in Return.
    /// This allows nested selections to be collected and merged into a single Return.
    fn collect_selection_items(
        &self,
        selection_set: &ast::SelectionSet,
        input: LogicalOperator,
        current_var: &str,
    ) -> Result<(LogicalOperator, Vec<ReturnItem>)> {
        let mut return_items = Vec::new();
        let mut plan = input;

        for selection in &selection_set.selections {
            match selection {
                ast::Selection::Field(field) => {
                    // Evaluate @skip / @include directives
                    if !graphql_directives_allow(&field.directives) {
                        continue;
                    }

                    if field.selection_set.is_some() {
                        // This is a relationship traversal - collect nested items
                        let (new_plan, nested_items) =
                            self.translate_nested_field_items(field, plan, current_var)?;
                        plan = new_plan;
                        // Add nested items with proper aliasing
                        for item in nested_items {
                            let alias = field.alias.clone().unwrap_or(field.name.clone());
                            let new_alias = if let Some(existing) = &item.alias {
                                format!("{}_{}", alias, existing)
                            } else {
                                alias
                            };
                            return_items.push(ReturnItem {
                                expression: item.expression,
                                alias: Some(new_alias),
                            });
                        }
                    } else {
                        // Scalar field - add to return items
                        let alias = field.alias.clone().unwrap_or(field.name.clone());
                        return_items.push(ReturnItem {
                            expression: LogicalExpression::Property {
                                variable: current_var.to_string(),
                                property: field.name.clone(),
                            },
                            alias: Some(alias),
                        });
                    }
                }
                ast::Selection::FragmentSpread(spread) => {
                    // Evaluate @skip / @include directives on the spread
                    if !graphql_directives_allow(&spread.directives) {
                        continue;
                    }
                    // Resolve fragment and include its fields
                    if let Some(frag) = self.fragments.get(&spread.name) {
                        let (new_plan, items) = self.expand_fragment(frag, plan, current_var)?;
                        plan = new_plan;
                        return_items.extend(items);
                    }
                }
                ast::Selection::InlineFragment(inline) => {
                    // Evaluate @skip / @include directives on the inline fragment
                    if !graphql_directives_allow(&inline.directives) {
                        continue;
                    }
                    // Inline fragment with type condition
                    if let Some(type_cond) = &inline.type_condition {
                        // Add type check filter: type_cond IN labels(var)
                        plan = wrap_filter(
                            plan,
                            LogicalExpression::Binary {
                                left: Box::new(LogicalExpression::Literal(
                                    grafeo_common::types::Value::String(type_cond.clone().into()),
                                )),
                                op: BinaryOp::In,
                                right: Box::new(LogicalExpression::Labels(current_var.to_string())),
                            },
                        );
                    }
                    // Process inline fragment's selection set
                    let (new_plan, items) =
                        self.process_inline_selections(&inline.selection_set, plan, current_var)?;
                    plan = new_plan;
                    return_items.extend(items);
                }
            }
        }

        Ok((plan, return_items))
    }

    /// Translates a nested field and returns the plan + return items (not wrapped in Return).
    fn translate_nested_field_items(
        &self,
        field: &ast::Field,
        input: LogicalOperator,
        from_var: &str,
    ) -> Result<(LogicalOperator, Vec<ReturnItem>)> {
        let to_var = self.var_gen.next();

        // The field name is the edge type: preserve original case to match how edges are stored
        let mut plan = LogicalOperator::Expand(ExpandOp {
            quantified: false,
            from_variable: from_var.to_string(),
            to_variable: to_var.clone(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec![field.name.clone()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(input),
            path_alias: None,
            path_mode: PathMode::Walk,
        });

        // Apply argument filters
        if !field.arguments.is_empty() {
            let filter = self.translate_arguments(&field.arguments, &to_var)?;
            plan = wrap_filter(plan, filter);
        }

        // Collect nested selection items (without wrapping in Return)
        let return_items = if let Some(selection_set) = &field.selection_set {
            let (new_plan, items) = self.collect_selection_items(selection_set, plan, &to_var)?;
            plan = new_plan;
            items
        } else {
            // No nested selection - return the whole nested node
            vec![ReturnItem {
                expression: LogicalExpression::Variable(to_var),
                alias: None,
            }]
        };

        Ok((plan, return_items))
    }

    /// Translates filter arguments to a predicate expression.
    /// Supports:
    /// - Direct arguments: `name: "Alix"` → `name = "Alix"`
    /// - Where clause: `where: { age_gt: 30 }` → `age > 30`
    /// - Operator suffixes: `_gt`, `_gte`, `_lt`, `_lte`, `_ne`, `_contains`, `_starts_with`, `_ends_with`, `_in`
    fn translate_filter_arguments(
        &self,
        args: &[&ast::Argument],
        var: &str,
    ) -> Result<LogicalExpression> {
        let mut predicates = Vec::new();

        for arg in args {
            // Check for "where" argument with nested object
            if arg.name == "where" || arg.name == "filter" {
                if let ast::InputValue::Object(fields) = &arg.value {
                    for (field_name, value) in fields {
                        let (property, op) = self.parse_field_operator(field_name);
                        let prop = LogicalExpression::Property {
                            variable: var.to_string(),
                            property,
                        };
                        let val = self.input_value_to_expression(value);
                        predicates.push(LogicalExpression::Binary {
                            left: Box::new(prop),
                            op,
                            right: Box::new(val),
                        });
                    }
                }
            } else {
                // Direct argument: name: "Alix" → name = "Alix", age_gt: 30 → age > 30
                let (property, op) = self.parse_field_operator(&arg.name);
                let prop = LogicalExpression::Property {
                    variable: var.to_string(),
                    property,
                };
                let value = self.input_value_to_expression(&arg.value);
                predicates.push(LogicalExpression::Binary {
                    left: Box::new(prop),
                    op,
                    right: Box::new(value),
                });
            }
        }

        self.combine_with_and(predicates)
    }

    /// Legacy translate_arguments for nested fields (still uses simple equality).
    fn translate_arguments(&self, args: &[ast::Argument], var: &str) -> Result<LogicalExpression> {
        let refs: Vec<&ast::Argument> = args.iter().collect();
        self.translate_filter_arguments(&refs, var)
    }

    /// Parses a field name with optional operator suffix.
    /// Returns (property_name, operator).
    fn parse_field_operator(&self, field: &str) -> (String, BinaryOp) {
        // Check suffixes in order of length (longest first to avoid partial matches)
        let suffixes = [
            ("_starts_with", BinaryOp::StartsWith),
            ("_ends_with", BinaryOp::EndsWith),
            ("_contains", BinaryOp::Contains),
            ("_gte", BinaryOp::Ge),
            ("_lte", BinaryOp::Le),
            ("_gt", BinaryOp::Gt),
            ("_lt", BinaryOp::Lt),
            ("_ne", BinaryOp::Ne),
            ("_in", BinaryOp::In),
        ];

        for (suffix, op) in suffixes {
            if let Some(property) = field.strip_suffix(suffix) {
                return (property.to_string(), op);
            }
        }

        (field.to_string(), BinaryOp::Eq)
    }

    /// Converts an `InputValue` to a `LogicalExpression`.
    ///
    /// Variable references (`$name`) are translated to `LogicalExpression::Parameter`
    /// so the executor can substitute the actual value from the parameter map.
    fn input_value_to_expression(&self, input: &ast::InputValue) -> LogicalExpression {
        match input {
            ast::InputValue::Variable(name) => LogicalExpression::Parameter(name.clone()),
            _ => LogicalExpression::Literal(input.to_value()),
        }
    }

    /// Combines predicates with AND.
    fn combine_with_and(&self, predicates: Vec<LogicalExpression>) -> Result<LogicalExpression> {
        if predicates.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "No predicates",
            )));
        }

        let result = predicates
            .into_iter()
            .reduce(|acc, pred| LogicalExpression::Binary {
                left: Box::new(acc),
                op: BinaryOp::And,
                right: Box::new(pred),
            })
            .expect("predicates non-empty after is_empty check");

        Ok(result)
    }

    fn expand_fragment(
        &self,
        frag: &ast::FragmentDefinition,
        input: LogicalOperator,
        current_var: &str,
    ) -> Result<(LogicalOperator, Vec<ReturnItem>)> {
        let mut return_items = Vec::new();

        for selection in &frag.selection_set.selections {
            if let ast::Selection::Field(field) = selection
                && field.selection_set.is_none()
                && graphql_directives_allow(&field.directives)
            {
                // Scalar field
                let alias = field.alias.clone().unwrap_or(field.name.clone());
                return_items.push(ReturnItem {
                    expression: LogicalExpression::Property {
                        variable: current_var.to_string(),
                        property: field.name.clone(),
                    },
                    alias: Some(alias),
                });
            }
        }

        Ok((input, return_items))
    }

    fn process_inline_selections(
        &self,
        selection_set: &ast::SelectionSet,
        input: LogicalOperator,
        current_var: &str,
    ) -> Result<(LogicalOperator, Vec<ReturnItem>)> {
        let mut return_items = Vec::new();

        for selection in &selection_set.selections {
            if let ast::Selection::Field(field) = selection
                && field.selection_set.is_none()
                && graphql_directives_allow(&field.directives)
            {
                let alias = field.alias.clone().unwrap_or(field.name.clone());
                return_items.push(ReturnItem {
                    expression: LogicalExpression::Property {
                        variable: current_var.to_string(),
                        property: field.name.clone(),
                    },
                    alias: Some(alias),
                });
            }
        }

        Ok((input, return_items))
    }

    fn get_first_field<'a>(&self, selection_set: &'a ast::SelectionSet) -> Result<&'a ast::Field> {
        for selection in &selection_set.selections {
            if let ast::Selection::Field(field) = selection {
                return Ok(field);
            }
        }
        Err(Error::Query(QueryError::new(
            QueryErrorKind::Syntax,
            "No field found in selection set",
        )))
    }
}

fn parse_aggregate_field(
    name: &str,
    arguments: &[ast::Argument],
) -> Option<(AggregateFunction, Option<String>)> {
    // Underscore-prefixed style with argument: _sum(field: age)
    let field_arg = arguments
        .iter()
        .find(|a| a.name == "field")
        .and_then(|a| match &a.value {
            ast::InputValue::Enum(s) | ast::InputValue::String(s) => Some(s.clone()),
            _ => None,
        });

    match name {
        "_count" | "count" => Some((AggregateFunction::Count, None)),
        "_sum" => Some((AggregateFunction::Sum, field_arg)),
        "_avg" => Some((AggregateFunction::Avg, field_arg)),
        "_min" => Some((AggregateFunction::Min, field_arg)),
        "_max" => Some((AggregateFunction::Max, field_arg)),
        _ => {
            // Prefix style: sum_<prop>, avg_<prop>, min_<prop>, max_<prop>
            let prefixes: &[(&str, AggregateFunction)] = &[
                ("sum_", AggregateFunction::Sum),
                ("avg_", AggregateFunction::Avg),
                ("min_", AggregateFunction::Min),
                ("max_", AggregateFunction::Max),
                ("count_", AggregateFunction::CountNonNull),
            ];
            for (prefix, func) in prefixes {
                if let Some(prop) = name.strip_prefix(prefix)
                    && !prop.is_empty()
                {
                    return Some((*func, Some(prop.to_string())));
                }
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_simple_query() {
        let query = r#"
            query {
                user {
                    id
                    name
                }
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 2);
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_with_argument() {
        let query = r#"
            query {
                user(id: 123) {
                    name
                }
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();
        // Should have NodeScan -> Filter -> Return
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                // Filter should check id = 123
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Eq);
                }
            } else {
                panic!("Expected Filter operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_nested_fields() {
        let query = r#"
            query {
                user {
                    name
                    friends {
                        name
                    }
                }
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_with_alias() {
        let query = r#"
            query {
                user {
                    userName: name
                }
            }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items[0].alias, Some("userName".to_string()));
        }
    }

    // ==================== Pagination Tests ====================

    #[test]
    fn test_pagination_first() {
        let query = r#"{ user(first: 10) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Should contain Limit with count 10
        fn find_limit(op: &LogicalOperator) -> Option<usize> {
            match op {
                LogicalOperator::Limit(l) => Some(l.count.value()),
                LogicalOperator::Return(r) => find_limit(&r.input),
                LogicalOperator::Filter(f) => find_limit(&f.input),
                LogicalOperator::Sort(s) => find_limit(&s.input),
                LogicalOperator::Skip(s) => find_limit(&s.input),
                _ => None,
            }
        }
        assert_eq!(find_limit(&plan.root), Some(10));
    }

    #[test]
    fn test_pagination_skip() {
        let query = r#"{ user(skip: 5) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Should contain Skip with count 5
        fn find_skip(op: &LogicalOperator) -> Option<usize> {
            match op {
                LogicalOperator::Skip(s) => Some(s.count.value()),
                LogicalOperator::Return(r) => find_skip(&r.input),
                LogicalOperator::Filter(f) => find_skip(&f.input),
                LogicalOperator::Limit(l) => find_skip(&l.input),
                _ => None,
            }
        }
        assert_eq!(find_skip(&plan.root), Some(5));
    }

    #[test]
    fn test_pagination_first_and_skip() {
        let query = r#"{ user(first: 10, skip: 5) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Should have Limit(Skip(...))
        if let LogicalOperator::Limit(limit) = &plan.root {
            assert_eq!(limit.count, 10);
            if let LogicalOperator::Skip(skip) = limit.input.as_ref() {
                assert_eq!(skip.count, 5);
            } else {
                panic!("Expected Skip inside Limit");
            }
        } else {
            panic!("Expected Limit at root");
        }
    }

    // ==================== Ordering Tests ====================

    #[test]
    fn test_order_by_single() {
        let query = r#"{ user(orderBy: { name: ASC }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Should contain Sort
        fn find_sort(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Sort(_) => true,
                LogicalOperator::Return(r) => find_sort(&r.input),
                LogicalOperator::Limit(l) => find_sort(&l.input),
                LogicalOperator::Skip(s) => find_sort(&s.input),
                _ => false,
            }
        }
        assert!(find_sort(&plan.root));
    }

    #[test]
    fn test_order_by_desc() {
        let query = r#"{ user(orderBy: { age: DESC }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_sort_order(op: &LogicalOperator) -> Option<SortOrder> {
            match op {
                LogicalOperator::Sort(s) => s.keys.first().map(|k| k.order),
                LogicalOperator::Return(r) => find_sort_order(&r.input),
                LogicalOperator::Limit(l) => find_sort_order(&l.input),
                _ => None,
            }
        }
        assert_eq!(find_sort_order(&plan.root), Some(SortOrder::Descending));
    }

    // ==================== Where Operator Tests ====================

    #[test]
    fn test_where_gt() {
        let query = r#"{ user(where: { age_gt: 30 }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                LogicalOperator::Sort(s) => find_filter_op(&s.input),
                LogicalOperator::Limit(l) => find_filter_op(&l.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::Gt));
    }

    #[test]
    fn test_where_contains() {
        let query = r#"{ user(where: { name_contains: "Ali" }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::Contains));
    }

    #[test]
    fn test_where_multiple_operators() {
        let query = r#"{ user(where: { age_gte: 18, age_lte: 65 }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Should have Filter with AND predicate
        fn find_and(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        *op == BinaryOp::And
                    } else {
                        false
                    }
                }
                LogicalOperator::Return(r) => find_and(&r.input),
                _ => false,
            }
        }
        assert!(find_and(&plan.root));
    }

    // ==================== Mutation Tests ====================

    #[test]
    fn test_create_mutation() {
        let query = r#"mutation { createUser(name: "Alix", age: 30) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Should contain CreateNode
        fn find_create(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::CreateNode(_) => true,
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => false,
            }
        }
        assert!(find_create(&plan.root));
    }

    #[test]
    fn test_create_mutation_labels() {
        let query = r#"mutation { createPerson(name: "Gus") { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Label should be "Person" (capitalized)
        fn find_label(op: &LogicalOperator) -> Option<String> {
            match op {
                LogicalOperator::CreateNode(c) => c.labels.first().cloned(),
                LogicalOperator::Return(r) => find_label(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_label(&plan.root), Some("Person".to_string()));
    }

    #[test]
    fn test_delete_mutation() {
        let query = r#"mutation { deleteUser(id: 123) }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // Should contain DeleteNode
        fn find_delete(op: &LogicalOperator) -> bool {
            matches!(op, LogicalOperator::DeleteNode(_))
        }
        assert!(find_delete(&plan.root));
    }

    #[test]
    fn test_delete_mutation_by_property() {
        // Delete mutations can use any property as filter, not just id
        let query = r#"mutation { deleteUser(name: "Alix") }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_delete_mutation_requires_filter() {
        // Delete mutations require at least one filter argument
        let query = r#"mutation { deleteUser }"#;
        let result = translate(query);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_mutation() {
        let query = r#"mutation { doSomething(name: "test") { id } }"#;
        let result = translate(query);
        // Should fail because mutation name doesn't start with create/update/delete
        assert!(result.is_err());
    }

    #[test]
    fn test_subscription_not_supported() {
        let query = r#"subscription { userCreated { id } }"#;
        let result = translate(query);
        assert!(result.is_err());
    }

    // ==================== Update Mutation Tests ====================

    #[test]
    fn test_update_mutation() {
        let query = r#"mutation { updateUser(id: 123, name: "Alix") { name } }"#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Update mutation should work: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Should contain SetProperty
        fn find_set_property(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::SetProperty(_) => true,
                LogicalOperator::Return(r) => find_set_property(&r.input),
                _ => false,
            }
        }
        assert!(
            find_set_property(&plan.root),
            "Update should produce SetProperty"
        );
    }

    #[test]
    fn test_update_mutation_requires_filter_and_property() {
        // Only one argument - need at least 2 (filter + property to update)
        let query = r#"mutation { updateUser(name: "Alix") { name } }"#;
        let result = translate(query);
        assert!(result.is_err(), "Update with only 1 argument should fail");
    }

    #[test]
    fn test_update_mutation_without_selection_set() {
        // Update without specifying which fields to return
        let query = r#"mutation { updateUser(id: 1, name: "Gus") }"#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Update without selection set should work: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_update_mutation_property_filter() {
        // When no id, first argument becomes filter
        let query = r#"mutation { updateUser(email: "alix@test.com", name: "Alix") { name } }"#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Update with property filter should work: {:?}",
            result.err()
        );
    }

    // ==================== Where Operator Suffix Tests ====================

    #[test]
    fn test_where_ne() {
        let query = r#"{ user(where: { status_ne: "deleted" }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::Ne));
    }

    #[test]
    fn test_where_starts_with() {
        let query = r#"{ user(where: { email_starts_with: "admin" }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::StartsWith));
    }

    #[test]
    fn test_where_ends_with() {
        let query = r#"{ user(where: { email_ends_with: ".com" }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::EndsWith));
    }

    #[test]
    fn test_where_in() {
        let query = r#"{ user(where: { status_in: ["active", "pending"] }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::In));
    }

    #[test]
    fn test_where_lt_and_lte() {
        let query = r#"{ user(where: { age_lt: 18 }) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());

        let query2 = r#"{ user(where: { age_lte: 65 }) { name } }"#;
        let result2 = translate(query2);
        assert!(result2.is_ok());
    }

    // ==================== Pagination + Ordering Combined ====================

    #[test]
    fn test_pagination_with_order_by() {
        let query = r#"{ user(first: 10, skip: 5, orderBy: { name: ASC }) { name } }"#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Pagination with order should work: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_order_by_with_multiple_fields() {
        let query = r#"{ user(orderBy: { name: ASC, age: DESC }) { name age } }"#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Multiple orderBy fields should work: {:?}",
            result.err()
        );
    }

    // ==================== Direct Argument Range Filter Tests ====================

    #[test]
    fn test_direct_arg_range_gt() {
        // Direct argument (not wrapped in "where") should parse operator suffixes
        let query = r#"{ user(age_gt: 30) { name } }"#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Direct arg with _gt suffix should work: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<(BinaryOp, String)> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, left, .. } = &f.predicate
                        && let LogicalExpression::Property { property, .. } = left.as_ref()
                    {
                        return Some((*op, property.clone()));
                    }
                    None
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        let (op, prop) = find_filter_op(&plan.root).expect("Should have filter");
        assert_eq!(op, BinaryOp::Gt, "Should use Gt operator, not Eq");
        assert_eq!(prop, "age", "Should strip _gt suffix from property name");
    }

    #[test]
    fn test_direct_arg_range_lt() {
        let query = r#"{ user(age_lt: 18) { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::Lt));
    }

    #[test]
    fn test_direct_arg_compound_range() {
        // Two direct args with range suffixes should produce AND
        let query = r#"{ user(age_gt: 20, age_lt: 40) { name } }"#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Compound range direct args should work: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_and_with_ops(op: &LogicalOperator) -> Option<(BinaryOp, BinaryOp)> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary {
                        op: BinaryOp::And,
                        left,
                        right,
                    } = &f.predicate
                    {
                        let left_op = if let LogicalExpression::Binary { op, .. } = left.as_ref() {
                            Some(*op)
                        } else {
                            None
                        };
                        let right_op = if let LogicalExpression::Binary { op, .. } = right.as_ref()
                        {
                            Some(*op)
                        } else {
                            None
                        };
                        if let (Some(l), Some(r)) = (left_op, right_op) {
                            return Some((l, r));
                        }
                    }
                    None
                }
                LogicalOperator::Return(r) => find_and_with_ops(&r.input),
                _ => None,
            }
        }
        let (l, r) = find_and_with_ops(&plan.root).expect("Should have AND with two range ops");
        assert_eq!(l, BinaryOp::Gt);
        assert_eq!(r, BinaryOp::Lt);
    }

    #[test]
    fn test_direct_arg_equality_unchanged() {
        // Plain direct args (no suffix) should still use Eq
        let query = r#"{ user(name: "Alix") { age } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::Eq));
    }

    #[test]
    fn test_direct_arg_contains_suffix() {
        let query = r#"{ user(name_contains: "li") { name } }"#;
        let result = translate(query);
        assert!(result.is_ok());
        let plan = result.unwrap();

        fn find_filter_op(op: &LogicalOperator) -> Option<BinaryOp> {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        Some(*op)
                    } else {
                        None
                    }
                }
                LogicalOperator::Return(r) => find_filter_op(&r.input),
                _ => None,
            }
        }
        assert_eq!(find_filter_op(&plan.root), Some(BinaryOp::Contains));
    }

    #[test]
    fn test_nested_field_with_range_filter() {
        // Nested field args should also parse operator suffixes
        let query = r#"{ user { name friends(age_gt: 30) { name } } }"#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Nested field with _gt suffix should work: {:?}",
            result.err()
        );
    }

    // ==================== Inline Fragment Tests ====================

    #[test]
    fn test_inline_fragment_type_condition() {
        let query = r#"
            query {
                person {
                    ... on Person {
                        name
                    }
                }
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Inline fragment translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // The inline fragment with a type condition should produce a Filter
        // with a BinaryOp::In checking the type against labels.
        fn find_in_filter(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate
                        && *op == BinaryOp::In
                    {
                        return true;
                    }
                    find_in_filter(&f.input)
                }
                LogicalOperator::Return(r) => find_in_filter(&r.input),
                LogicalOperator::Limit(l) => find_in_filter(&l.input),
                LogicalOperator::Skip(s) => find_in_filter(&s.input),
                _ => false,
            }
        }
        assert!(
            find_in_filter(&plan.root),
            "Expected Filter with In operator for inline fragment type condition"
        );
    }

    // === EXPLAIN/PROFILE tests ===

    #[test]
    fn test_explain_prefix_sets_explain_flag() {
        let plan = super::translate("EXPLAIN { person { name } }").unwrap();
        assert!(plan.explain, "EXPLAIN prefix should set explain flag");
        assert!(!plan.profile, "EXPLAIN should not set profile flag");
    }

    #[test]
    fn test_explain_analyze_sets_profile_flag() {
        let plan = super::translate("EXPLAIN ANALYZE { person { name } }").unwrap();
        assert!(!plan.explain, "EXPLAIN ANALYZE should not set explain flag");
        assert!(plan.profile, "EXPLAIN ANALYZE should set profile flag");
    }

    #[test]
    fn test_no_explain_prefix_normal_query() {
        let plan = super::translate("{ person { name } }").unwrap();
        assert!(!plan.explain);
        assert!(!plan.profile);
    }
}
