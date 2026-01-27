//! GraphQL to RDF LogicalPlan translator.
//!
//! Translates GraphQL queries to the common logical plan representation for RDF.
//!
//! # Mapping Strategy
//!
//! GraphQL's hierarchical structure maps to RDF triple patterns:
//! - Root fields → Triple patterns with `rdf:type` predicate
//! - Field arguments → Additional triple patterns for filtering
//! - Nested selections → Predicate-object traversals
//! - Scalar fields → Select variables from triple bindings

use crate::query::plan::{
    BinaryOp, FilterOp, JoinOp, JoinType, LogicalExpression, LogicalOperator, LogicalPlan,
    Projection, ProjectOp, TripleComponent, TripleScanOp,
};
use graphos_adapters::query::graphql::{self, ast};
use graphos_common::types::Value;
use graphos_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// RDF namespace constants.
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// Translates a GraphQL query string to an RDF logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str, namespace: &str) -> Result<LogicalPlan> {
    let doc = graphql::parse(query)?;
    let translator = GraphQLRdfTranslator::new(namespace);
    translator.translate_document(&doc)
}

/// Translator from GraphQL AST to RDF LogicalPlan.
struct GraphQLRdfTranslator {
    /// Counter for generating anonymous variables.
    var_counter: AtomicU32,
    /// Base namespace for type IRIs.
    namespace: String,
    /// Fragment definitions for resolution.
    fragments: HashMap<String, ast::FragmentDefinition>,
}

impl GraphQLRdfTranslator {
    fn new(namespace: &str) -> Self {
        Self {
            var_counter: AtomicU32::new(0),
            namespace: namespace.to_string(),
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

        // Only Query operations are supported
        if operation.operation != ast::OperationType::Query {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Only Query operations are supported for RDF",
            )));
        }

        // Create translator with fragments
        let translator = GraphQLRdfTranslator {
            var_counter: AtomicU32::new(0),
            namespace: self.namespace.clone(),
            fragments,
        };

        translator.translate_operation(operation)
    }

    fn translate_operation(&self, op: &ast::OperationDefinition) -> Result<LogicalPlan> {
        // Each field in the root selection set is a separate query
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
        // Root field name becomes the RDF type
        let subject_var = self.next_var();
        let type_iri = self.make_type_iri(&field.name);

        // Create triple pattern: ?subject rdf:type <Type>
        let mut plan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable(subject_var.clone()),
            predicate: TripleComponent::Iri(RDF_TYPE.to_string()),
            object: TripleComponent::Iri(type_iri),
            graph: None,
            input: None,
        });

        // Apply argument filters
        if !field.arguments.is_empty() {
            plan = self.translate_arguments(&field.arguments, &subject_var, plan)?;
        }

        // Process nested selection set
        let mut projections = Vec::new();
        if let Some(selection_set) = &field.selection_set {
            let (new_plan, new_projections) =
                self.translate_selection_set(selection_set, plan, &subject_var)?;
            plan = new_plan;
            projections = new_projections;
        }

        // Add projection if we have fields to return
        if !projections.is_empty() {
            plan = LogicalOperator::Project(ProjectOp {
                projections,
                input: Box::new(plan),
            });
        }

        Ok(plan)
    }

    fn translate_selection_set(
        &self,
        selection_set: &ast::SelectionSet,
        input: LogicalOperator,
        subject_var: &str,
    ) -> Result<(LogicalOperator, Vec<Projection>)> {
        let mut projections = Vec::new();
        let mut plan = input;

        for selection in &selection_set.selections {
            match selection {
                ast::Selection::Field(field) => {
                    if field.selection_set.is_some() {
                        // This is a nested object - requires another triple pattern
                        let (new_plan, nested_projections) =
                            self.translate_nested_field(field, plan, subject_var)?;
                        plan = new_plan;
                        projections.extend(nested_projections);
                    } else {
                        // Scalar field - create a triple pattern to fetch the property
                        let (new_plan, prop_var) =
                            self.translate_scalar_field(field, plan, subject_var)?;
                        plan = new_plan;

                        let alias = field.alias.clone().unwrap_or(field.name.clone());
                        projections.push(Projection {
                            expression: LogicalExpression::Variable(prop_var),
                            alias: Some(alias),
                        });
                    }
                }
                ast::Selection::FragmentSpread(spread) => {
                    // Resolve fragment and include its fields
                    if let Some(frag) = self.fragments.get(&spread.name) {
                        let (new_plan, frag_projections) =
                            self.expand_fragment(frag, plan, subject_var)?;
                        plan = new_plan;
                        projections.extend(frag_projections);
                    }
                }
                ast::Selection::InlineFragment(inline) => {
                    // Inline fragment with type condition
                    if let Some(type_cond) = &inline.type_condition {
                        // Add type check as a triple pattern
                        let type_iri = self.make_type_iri(type_cond);
                        let type_check = LogicalOperator::TripleScan(TripleScanOp {
                            subject: TripleComponent::Variable(subject_var.to_string()),
                            predicate: TripleComponent::Iri(RDF_TYPE.to_string()),
                            object: TripleComponent::Iri(type_iri),
                            graph: None,
                            input: None,
                        });
                        plan = self.join_patterns(plan, type_check);
                    }

                    // Process inline fragment's selection set
                    let (new_plan, inline_projections) =
                        self.translate_selection_set(&inline.selection_set, plan, subject_var)?;
                    plan = new_plan;
                    projections.extend(inline_projections);
                }
            }
        }

        Ok((plan, projections))
    }

    fn translate_scalar_field(
        &self,
        field: &ast::Field,
        input: LogicalOperator,
        subject_var: &str,
    ) -> Result<(LogicalOperator, String)> {
        let object_var = self.next_var();
        let predicate_iri = self.make_predicate_iri(&field.name);

        // Create triple pattern: ?subject <predicate> ?object
        let triple = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable(subject_var.to_string()),
            predicate: TripleComponent::Iri(predicate_iri),
            object: TripleComponent::Variable(object_var.clone()),
            graph: None,
            input: None,
        });

        let plan = self.join_patterns(input, triple);
        Ok((plan, object_var))
    }

    fn translate_nested_field(
        &self,
        field: &ast::Field,
        input: LogicalOperator,
        from_var: &str,
    ) -> Result<(LogicalOperator, Vec<Projection>)> {
        let to_var = self.next_var();
        let predicate_iri = self.make_predicate_iri(&field.name);

        // Create triple pattern: ?from <predicate> ?to
        let triple = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable(from_var.to_string()),
            predicate: TripleComponent::Iri(predicate_iri),
            object: TripleComponent::Variable(to_var.clone()),
            graph: None,
            input: None,
        });

        let mut plan = self.join_patterns(input, triple);

        // Apply argument filters to the target
        if !field.arguments.is_empty() {
            plan = self.translate_arguments(&field.arguments, &to_var, plan)?;
        }

        // Process nested selections
        let mut projections = Vec::new();
        if let Some(selection_set) = &field.selection_set {
            let (new_plan, nested_projections) =
                self.translate_selection_set(selection_set, plan, &to_var)?;
            plan = new_plan;
            projections = nested_projections;
        }

        Ok((plan, projections))
    }

    fn translate_arguments(
        &self,
        args: &[ast::Argument],
        subject_var: &str,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let mut plan = input;

        for arg in args {
            // Each argument creates a filter
            // First, create a triple pattern for the property
            let predicate_iri = self.make_predicate_iri(&arg.name);
            let object_var = self.next_var();

            let triple = LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable(subject_var.to_string()),
                predicate: TripleComponent::Iri(predicate_iri),
                object: TripleComponent::Variable(object_var.clone()),
                graph: None,
                input: None,
            });

            plan = self.join_patterns(plan, triple);

            // Add filter for the value
            let filter = LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable(object_var)),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(arg.value.to_value())),
            };

            plan = LogicalOperator::Filter(FilterOp {
                predicate: filter,
                input: Box::new(plan),
            });
        }

        Ok(plan)
    }

    fn expand_fragment(
        &self,
        frag: &ast::FragmentDefinition,
        input: LogicalOperator,
        subject_var: &str,
    ) -> Result<(LogicalOperator, Vec<Projection>)> {
        // Add type condition if present
        let type_iri = self.make_type_iri(&frag.type_condition);
        let type_check = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable(subject_var.to_string()),
            predicate: TripleComponent::Iri(RDF_TYPE.to_string()),
            object: TripleComponent::Iri(type_iri),
            graph: None,
            input: None,
        });

        let plan = self.join_patterns(input, type_check);

        // Process fragment's selection set
        self.translate_selection_set(&frag.selection_set, plan, subject_var)
    }

    fn join_patterns(&self, left: LogicalOperator, right: LogicalOperator) -> LogicalOperator {
        if matches!(left, LogicalOperator::Empty) {
            return right;
        }
        if matches!(right, LogicalOperator::Empty) {
            return left;
        }

        LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions: vec![], // Shared variables are implicit join conditions
        })
    }

    fn get_first_field<'a>(&self, selection_set: &'a ast::SelectionSet) -> Result<&'a ast::Field> {
        for selection in &selection_set.selections {
            if let ast::Selection::Field(field) = selection {
                return Ok(field);
            }
        }
        Err(Error::Query(QueryError::new(QueryErrorKind::Syntax, "No field found in selection set")))
    }

    fn make_type_iri(&self, type_name: &str) -> String {
        format!("{}{}", self.namespace, self.capitalize_first(type_name))
    }

    fn make_predicate_iri(&self, name: &str) -> String {
        format!("{}{}", self.namespace, name)
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

    const TEST_NS: &str = "http://example.org/";

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
        let result = translate(query, TEST_NS);
        assert!(result.is_ok());
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
        let result = translate(query, TEST_NS);
        assert!(result.is_ok());
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
        let result = translate(query, TEST_NS);
        assert!(result.is_ok());
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
        let result = translate(query, TEST_NS);
        assert!(result.is_err());
    }

    #[test]
    fn test_creates_rdf_type_triple() {
        let query = r#"
            query {
                person {
                    name
                }
            }
        "#;
        let result = translate(query, TEST_NS);
        assert!(result.is_ok());
        let plan = result.unwrap();

        // The root should contain a TripleScan with rdf:type
        fn find_type_scan(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::TripleScan(scan) => {
                    matches!(&scan.predicate, TripleComponent::Iri(iri) if iri == RDF_TYPE)
                }
                LogicalOperator::Join(join) => {
                    find_type_scan(&join.left) || find_type_scan(&join.right)
                }
                LogicalOperator::Filter(f) => find_type_scan(&f.input),
                LogicalOperator::Project(p) => find_type_scan(&p.input),
                _ => false,
            }
        }

        assert!(find_type_scan(&plan.root));
    }
}
