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
    BinaryOp, ExpandDirection, ExpandOp, FilterOp, LogicalExpression, LogicalOperator,
    LogicalPlan, NodeScanOp, ReturnItem, ReturnOp,
};
use graphos_adapters::query::graphql::{self, ast};
use graphos_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
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
            .ok_or_else(|| Error::Query(QueryError::new(QueryErrorKind::Syntax, "No operation found in document")))?;

        // Only Query operations are supported for LPG
        if operation.operation != ast::OperationType::Query {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Only Query operations are supported for LPG",
            )));
        }

        // Create translator with fragments
        let translator = GraphQLTranslator {
            var_counter: AtomicU32::new(0),
            fragments,
        };

        translator.translate_operation(operation)
    }

    fn translate_operation(&self, op: &ast::OperationDefinition) -> Result<LogicalPlan> {
        // Each field in the root selection set is a separate query
        // For now, we only support a single root field
        let selections = &op.selection_set.selections;
        if selections.is_empty() {
            return Err(Error::Query(QueryError::new(QueryErrorKind::Syntax, "Empty selection set")));
        }

        // Get the first field
        let field = self.get_first_field(&op.selection_set)?;
        let plan = self.translate_root_field(field)?;

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

        // Apply argument filters
        if !field.arguments.is_empty() {
            let filter = self.translate_arguments(&field.arguments, &var)?;
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

        Ok(plan)
    }

    fn translate_selection_set(
        &self,
        selection_set: &ast::SelectionSet,
        input: LogicalOperator,
        current_var: &str,
    ) -> Result<LogicalOperator> {
        let mut return_items = Vec::new();
        let mut plan = input;
        let mut nested_vars = Vec::new();

        for selection in &selection_set.selections {
            match selection {
                ast::Selection::Field(field) => {
                    if field.selection_set.is_some() {
                        // This is a relationship traversal
                        let (new_plan, nested_var) =
                            self.translate_nested_field(field, plan, current_var)?;
                        plan = new_plan;
                        nested_vars.push((field.alias.clone().unwrap_or(field.name.clone()), nested_var));
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
                        let (new_plan, items) =
                            self.expand_fragment(frag, plan, current_var)?;
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
                                    graphos_common::types::Value::String(type_cond.clone().into()),
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

        // Add nested variable references
        for (alias, var) in nested_vars {
            return_items.push(ReturnItem {
                expression: LogicalExpression::Variable(var),
                alias: Some(alias),
            });
        }

        // If we have return items, wrap in Return
        if !return_items.is_empty() {
            plan = LogicalOperator::Return(ReturnOp {
                items: return_items,
                distinct: false,
                input: Box::new(plan),
            });
        }

        Ok(plan)
    }

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

    fn translate_arguments(
        &self,
        args: &[ast::Argument],
        var: &str,
    ) -> Result<LogicalExpression> {
        let mut predicates = Vec::new();

        for arg in args {
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

        // Combine with AND
        let mut result = predicates
            .pop()
            .ok_or_else(|| Error::Internal("No arguments".to_string()))?;

        for pred in predicates.into_iter().rev() {
            result = LogicalExpression::Binary {
                left: Box::new(pred),
                op: BinaryOp::And,
                right: Box::new(result),
            };
        }

        Ok(result)
    }

    fn expand_fragment(
        &self,
        frag: &ast::FragmentDefinition,
        input: LogicalOperator,
        current_var: &str,
    ) -> Result<(LogicalOperator, Vec<ReturnItem>)> {
        let mut return_items = Vec::new();
        let mut plan = input;

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

        Ok((plan, return_items))
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
        Err(Error::Query(QueryError::new(QueryErrorKind::Syntax, "No field found in selection set")))
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

    #[test]
    fn test_reject_mutation() {
        let query = r#"
            mutation {
                createUser(name: "Alice") {
                    id
                }
            }
        "#;
        let result = translate(query);
        assert!(result.is_err());
    }
}
