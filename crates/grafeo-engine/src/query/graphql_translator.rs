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

use crate::query::plan::{
    BinaryOp, CreateNodeOp, DeleteNodeOp, ExpandDirection, ExpandOp, FilterOp, LimitOp,
    LogicalExpression, LogicalOperator, LogicalPlan, NodeScanOp, ReturnItem, ReturnOp,
    SetPropertyOp, SkipOp, SortKey, SortOp, SortOrder,
};
use grafeo_adapters::query::graphql::{self, ast};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Translates a GraphQL query string to a logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let doc = graphql::parse(query)?;
    let translator = GraphQLTranslator::new();
    translator.translate_document(&doc)
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
    first: Option<usize>,
    /// Pagination: skip (offset)
    skip: Option<usize>,
    /// Sort keys from orderBy
    order_by: Option<Vec<SortKey>>,
    /// Remaining filter arguments
    filters: Vec<&'a ast::Argument>,
}

/// Translator from GraphQL AST to LogicalPlan.
struct GraphQLTranslator {
    /// Counter for generating anonymous variables.
    var_counter: AtomicU32,
    /// Fragment definitions for resolution.
    fragments: HashMap<String, ast::FragmentDefinition>,
}

impl GraphQLTranslator {
    fn new() -> Self {
        Self {
            var_counter: AtomicU32::new(0),
            fragments: HashMap::new(),
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

        // Create translator with fragments
        let translator = GraphQLTranslator {
            var_counter: AtomicU32::new(0),
            fragments,
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

        // Get the first field
        let field = self.get_first_field(&op.selection_set)?;
        let plan = self.translate_root_field(field)?;

        Ok(LogicalPlan::new(plan))
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
                &format!(
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
        let var = self.next_var();

        // Convert arguments to properties
        let properties: Vec<(String, LogicalExpression)> = field
            .arguments
            .iter()
            .map(|arg| {
                (
                    arg.name.clone(),
                    LogicalExpression::Literal(arg.value.to_value()),
                )
            })
            .collect();

        let mut plan = LogicalOperator::CreateNode(CreateNodeOp {
            variable: var.clone(),
            labels: vec![self.capitalize_first(type_name)],
            properties,
            input: None,
        });

        // If there's a selection set, return the created node's properties
        plan = if let Some(selection_set) = &field.selection_set {
            self.translate_selection_set(selection_set, plan, &var)?
        } else {
            LogicalOperator::Return(ReturnOp {
                items: vec![ReturnItem {
                    expression: LogicalExpression::Variable(var),
                    alias: None,
                }],
                distinct: false,
                input: Box::new(plan),
            })
        };

        Ok(LogicalPlan::new(plan))
    }

    fn translate_update_mutation(
        &self,
        field: &ast::Field,
        type_name: &str,
    ) -> Result<LogicalPlan> {
        let var = self.next_var();

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
            .map(|arg| {
                (
                    arg.name.clone(),
                    LogicalExpression::Literal(arg.value.to_value()),
                )
            })
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
            label: Some(self.capitalize_first(type_name)),
            input: None,
        });

        // Apply filter
        plan = LogicalOperator::Filter(FilterOp {
            predicate: filter_predicate,
            input: Box::new(plan),
        });

        // Set the properties
        plan = LogicalOperator::SetProperty(SetPropertyOp {
            variable: var.clone(),
            properties,
            replace: false, // Merge properties, don't replace all
            input: Box::new(plan),
        });

        // If there's a selection set, return the updated node's properties
        plan = if let Some(selection_set) = &field.selection_set {
            self.translate_selection_set(selection_set, plan, &var)?
        } else {
            LogicalOperator::Return(ReturnOp {
                items: vec![ReturnItem {
                    expression: LogicalExpression::Variable(var),
                    alias: None,
                }],
                distinct: false,
                input: Box::new(plan),
            })
        };

        Ok(LogicalPlan::new(plan))
    }

    fn translate_delete_mutation(
        &self,
        field: &ast::Field,
        type_name: &str,
    ) -> Result<LogicalPlan> {
        let var = self.next_var();

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
            label: Some(self.capitalize_first(type_name)),
            input: None,
        });

        // Apply filter
        plan = LogicalOperator::Filter(FilterOp {
            predicate: filter_predicate,
            input: Box::new(plan),
        });

        // Delete the node (GraphQL mutations are like DETACH DELETE)
        plan = LogicalOperator::DeleteNode(DeleteNodeOp {
            variable: var,
            detach: true,
            input: Box::new(plan),
        });

        Ok(LogicalPlan::new(plan))
    }

    fn translate_root_field(&self, field: &ast::Field) -> Result<LogicalOperator> {
        // Root field name is the type/label to scan
        let var = self.next_var();

        // Start with a node scan using the field name as the label
        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: var.clone(),
            label: Some(self.capitalize_first(&field.name)),
            input: None,
        });

        // Extract special arguments (pagination, orderBy) from regular filters
        let extracted = self.extract_special_args(&field.arguments, &var);

        // Apply filters (excluding pagination and orderBy)
        if !extracted.filters.is_empty() {
            let filter = self.translate_filter_arguments(&extracted.filters, &var)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate: filter,
                input: Box::new(plan),
            });
        }

        // Process nested selection set
        if let Some(selection_set) = &field.selection_set {
            plan = self.translate_selection_set(selection_set, plan, &var)?;
        } else {
            // No nested selection, return the whole node
            plan = LogicalOperator::Return(ReturnOp {
                items: vec![ReturnItem {
                    expression: LogicalExpression::Variable(var),
                    alias: field.alias.clone(),
                }],
                distinct: false,
                input: Box::new(plan),
            });
        }

        // Apply ordering (before pagination)
        if let Some(keys) = extracted.order_by {
            plan = LogicalOperator::Sort(SortOp {
                keys,
                input: Box::new(plan),
            });
        }

        // Apply skip BEFORE limit
        if let Some(count) = extracted.skip {
            plan = LogicalOperator::Skip(SkipOp {
                count,
                input: Box::new(plan),
            });
        }

        // Apply limit
        if let Some(count) = extracted.first {
            plan = LogicalOperator::Limit(LimitOp {
                count,
                input: Box::new(plan),
            });
        }

        Ok(plan)
    }

    /// Extracts special arguments (first, skip, orderBy) from field arguments.
    fn extract_special_args<'a>(&self, args: &'a [ast::Argument], var: &str) -> ExtractedArgs<'a> {
        let mut first = None;
        let mut skip = None;
        let mut order_by = None;
        let mut filters = Vec::new();

        for arg in args {
            match arg.name.as_str() {
                "first" | "limit" => {
                    if let ast::InputValue::Int(n) = &arg.value {
                        first = Some(*n as usize);
                    }
                }
                "skip" | "offset" => {
                    if let ast::InputValue::Int(n) = &arg.value {
                        skip = Some(*n as usize);
                    }
                }
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
        // Collect all return items and build the plan
        let (plan, return_items) =
            self.collect_selection_items(selection_set, input, current_var)?;

        // Wrap in Return if we have items
        if !return_items.is_empty() {
            Ok(LogicalOperator::Return(ReturnOp {
                items: return_items,
                distinct: false,
                input: Box::new(plan),
            }))
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
                    // Resolve fragment and include its fields
                    if let Some(frag) = self.fragments.get(&spread.name) {
                        let (new_plan, items) = self.expand_fragment(frag, plan, current_var)?;
                        plan = new_plan;
                        return_items.extend(items);
                    }
                }
                ast::Selection::InlineFragment(inline) => {
                    // Inline fragment with type condition
                    if let Some(type_cond) = &inline.type_condition {
                        // Add type check filter
                        plan = LogicalOperator::Filter(FilterOp {
                            predicate: LogicalExpression::Binary {
                                left: Box::new(LogicalExpression::Labels(current_var.to_string())),
                                op: BinaryOp::Eq,
                                right: Box::new(LogicalExpression::Literal(
                                    grafeo_common::types::Value::String(type_cond.clone().into()),
                                )),
                            },
                            input: Box::new(plan),
                        });
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
        let to_var = self.next_var();

        // The field name is the edge type
        let mut plan = LogicalOperator::Expand(ExpandOp {
            from_variable: from_var.to_string(),
            to_variable: to_var.clone(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_type: Some(field.name.clone()),
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(input),
            path_alias: None,
        });

        // Apply argument filters
        if !field.arguments.is_empty() {
            let filter = self.translate_arguments(&field.arguments, &to_var)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate: filter,
                input: Box::new(plan),
            });
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

    #[allow(dead_code)]
    fn translate_nested_field(
        &self,
        field: &ast::Field,
        input: LogicalOperator,
        from_var: &str,
    ) -> Result<(LogicalOperator, String)> {
        let to_var = self.next_var();

        // The field name is the edge type
        let mut plan = LogicalOperator::Expand(ExpandOp {
            from_variable: from_var.to_string(),
            to_variable: to_var.clone(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_type: Some(field.name.clone()),
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(input),
            path_alias: None,
        });

        // Apply argument filters
        if !field.arguments.is_empty() {
            let filter = self.translate_arguments(&field.arguments, &to_var)?;
            plan = LogicalOperator::Filter(FilterOp {
                predicate: filter,
                input: Box::new(plan),
            });
        }

        // Process nested selections
        if let Some(selection_set) = &field.selection_set {
            plan = self.translate_selection_set(selection_set, plan, &to_var)?;
        }

        Ok((plan, to_var))
    }

    /// Translates filter arguments to a predicate expression.
    /// Supports:
    /// - Direct arguments: `name: "Alice"` → `name = "Alice"`
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
            if arg.name == "where" {
                if let ast::InputValue::Object(fields) = &arg.value {
                    for (field_name, value) in fields {
                        let (property, op) = self.parse_field_operator(field_name);
                        let prop = LogicalExpression::Property {
                            variable: var.to_string(),
                            property,
                        };
                        let val = LogicalExpression::Literal(self.input_value_to_value(value));
                        predicates.push(LogicalExpression::Binary {
                            left: Box::new(prop),
                            op,
                            right: Box::new(val),
                        });
                    }
                }
            } else {
                // Direct argument (legacy behavior): name: "Alice" → name = "Alice"
                let prop = LogicalExpression::Property {
                    variable: var.to_string(),
                    property: arg.name.clone(),
                };
                let value = LogicalExpression::Literal(arg.value.to_value());
                predicates.push(LogicalExpression::Binary {
                    left: Box::new(prop),
                    op: BinaryOp::Eq,
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
            if field.ends_with(suffix) {
                let property = field[..field.len() - suffix.len()].to_string();
                return (property, op);
            }
        }

        (field.to_string(), BinaryOp::Eq)
    }

    /// Converts an InputValue to a Value.
    fn input_value_to_value(&self, input: &ast::InputValue) -> grafeo_common::types::Value {
        input.to_value()
    }

    /// Combines predicates with AND.
    fn combine_with_and(&self, predicates: Vec<LogicalExpression>) -> Result<LogicalExpression> {
        if predicates.is_empty() {
            return Err(Error::Internal("No predicates".to_string()));
        }

        let result = predicates
            .into_iter()
            .reduce(|acc, pred| LogicalExpression::Binary {
                left: Box::new(acc),
                op: BinaryOp::And,
                right: Box::new(pred),
            })
            .unwrap();

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
            if let ast::Selection::Field(field) = selection {
                if field.selection_set.is_none() {
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
            if let ast::Selection::Field(field) = selection {
                if field.selection_set.is_none() {
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

    fn capitalize_first(&self, s: &str) -> String {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    }

    fn next_var(&self) -> String {
        let n = self.var_counter.fetch_add(1, Ordering::Relaxed);
        format!("_v{}", n)
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
                LogicalOperator::Limit(l) => Some(l.count),
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
                LogicalOperator::Skip(s) => Some(s.count),
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
        let query = r#"mutation { createUser(name: "Alice", age: 30) { name } }"#;
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
        let query = r#"mutation { createPerson(name: "Bob") { name } }"#;
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
            match op {
                LogicalOperator::DeleteNode(_) => true,
                _ => false,
            }
        }
        assert!(find_delete(&plan.root));
    }

    #[test]
    fn test_delete_mutation_by_property() {
        // Delete mutations can use any property as filter, not just id
        let query = r#"mutation { deleteUser(name: "Alice") }"#;
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
}
