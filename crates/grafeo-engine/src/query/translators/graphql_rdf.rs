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

use super::common::{
    VarGen, capitalize_first, graphql_directives_allow, wrap_filter, wrap_limit, wrap_skip,
};
use crate::query::plan::{
    BinaryOp, CountExpr, JoinOp, JoinType, LogicalExpression, LogicalOperator, LogicalPlan,
    ProjectOp, Projection, TripleComponent, TripleScanOp, UnionOp,
};
use grafeo_adapters::query::graphql::{self, ast};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use std::collections::HashMap;

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
    let plan = translator.translate_document(&doc)?;
    crate::query::limits::check_plan_depth(plan)
}

/// Translator from GraphQL AST to RDF LogicalPlan.
struct GraphQLRdfTranslator {
    /// Generator for anonymous variable names.
    var_gen: VarGen,
    /// Base namespace for type IRIs.
    namespace: String,
    /// Fragment definitions for resolution.
    fragments: HashMap<String, ast::FragmentDefinition>,
    /// Default values from variable declarations (e.g., `query($limit: Int = 2)`).
    variable_defaults: HashMap<String, grafeo_common::types::Value>,
}

impl GraphQLRdfTranslator {
    fn new(namespace: &str) -> Self {
        Self {
            var_gen: VarGen::new(),
            namespace: namespace.to_string(),
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

        // Only Query operations are supported
        if operation.operation != ast::OperationType::Query {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Only Query operations are supported for RDF",
            )));
        }

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
        let translator = GraphQLRdfTranslator {
            var_gen: VarGen::new(),
            namespace: self.namespace.clone(),
            fragments,
            variable_defaults,
        };

        translator.translate_operation(operation)
    }

    fn translate_operation(&self, op: &ast::OperationDefinition) -> Result<LogicalPlan> {
        // Each field in the root selection set is a separate query
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

    fn translate_root_field(&self, field: &ast::Field) -> Result<LogicalOperator> {
        // Root field name becomes the RDF type
        let subject_var = self.var_gen.next();
        let type_iri = self.make_type_iri(&field.name);

        // Create triple pattern: ?subject rdf:type <Type>
        let mut plan = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable(subject_var.clone()),
            predicate: TripleComponent::Iri(RDF_TYPE.to_string()),
            object: TripleComponent::Iri(type_iri),
            graph: None,
            input: None,
            dataset: None,
        });

        // Extract pagination (first/limit, skip/offset) from regular filter arguments
        let (pagination_first, pagination_skip, filter_args) =
            Self::extract_pagination_args(&field.arguments);

        // Apply filter arguments (excluding pagination)
        if !filter_args.is_empty() {
            plan = self.translate_arguments(&filter_args, &subject_var, plan)?;
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
                pass_through_input: false,
            });
        }

        // Apply pagination
        if let Some(skip) = pagination_skip {
            plan = wrap_skip(plan, skip);
        }
        if let Some(first) = pagination_first {
            plan = wrap_limit(plan, first);
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
                    // Evaluate @skip / @include directives
                    if !graphql_directives_allow(&field.directives) {
                        continue;
                    }

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
                    // Evaluate @skip / @include directives on the spread
                    if !graphql_directives_allow(&spread.directives) {
                        continue;
                    }
                    // Resolve fragment and include its fields
                    if let Some(frag) = self.fragments.get(&spread.name) {
                        let (new_plan, frag_projections) =
                            self.expand_fragment(frag, plan, subject_var)?;
                        plan = new_plan;
                        projections.extend(frag_projections);
                    }
                }
                ast::Selection::InlineFragment(inline) => {
                    // Evaluate @skip / @include directives on the inline fragment
                    if !graphql_directives_allow(&inline.directives) {
                        continue;
                    }
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
                            dataset: None,
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
        let object_var = self.var_gen.next();
        let predicate_iri = self.make_predicate_iri(&field.name);

        // Create triple pattern: ?subject <predicate> ?object
        let triple = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable(subject_var.to_string()),
            predicate: TripleComponent::Iri(predicate_iri),
            object: TripleComponent::Variable(object_var.clone()),
            graph: None,
            input: None,
            dataset: None,
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
        let to_var = self.var_gen.next();
        let predicate_iri = self.make_predicate_iri(&field.name);

        // Create triple pattern: ?from <predicate> ?to
        let triple = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable(from_var.to_string()),
            predicate: TripleComponent::Iri(predicate_iri),
            object: TripleComponent::Variable(to_var.clone()),
            graph: None,
            input: None,
            dataset: None,
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

    /// Extracts pagination arguments (first/limit, skip/offset) from field arguments,
    /// returning the pagination values and the remaining filter arguments.
    fn extract_pagination_args(
        args: &[ast::Argument],
    ) -> (Option<CountExpr>, Option<CountExpr>, Vec<ast::Argument>) {
        let mut first = None;
        let mut skip = None;
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
                _ => filters.push(arg.clone()),
            }
        }
        (first, skip, filters)
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
            let object_var = self.var_gen.next();

            let triple = LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable(subject_var.to_string()),
                predicate: TripleComponent::Iri(predicate_iri),
                object: TripleComponent::Variable(object_var.clone()),
                graph: None,
                input: None,
                dataset: None,
            });

            plan = self.join_patterns(plan, triple);

            // Add filter for the value (variable refs become Parameter expressions)
            let right = match &arg.value {
                ast::InputValue::Variable(name) => LogicalExpression::Parameter(name.clone()),
                other => LogicalExpression::Literal(other.to_value()),
            };
            let filter = LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Variable(object_var)),
                op: BinaryOp::Eq,
                right: Box::new(right),
            };

            plan = wrap_filter(plan, filter);
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
            dataset: None,
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

    fn make_type_iri(&self, type_name: &str) -> String {
        format!("{}{}", self.namespace, capitalize_first(type_name))
    }

    fn make_predicate_iri(&self, name: &str) -> String {
        format!("{}{}", self.namespace, name)
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
                createUser(name: "Alix") {
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

    // === Projection Tests ===

    #[test]
    fn test_scalar_fields_produce_projections() {
        let query = r#"
            query {
                user {
                    id
                    name
                    email
                }
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        // 3 scalar fields should produce a Project with 3 projections
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                LogicalOperator::Filter(f) => find_project(&f.input),
                _ => None,
            }
        }
        let project = find_project(&plan.root).expect("Expected Project for scalar fields");
        assert_eq!(
            project.projections.len(),
            3,
            "3 scalar fields should produce 3 projections"
        );
    }

    #[test]
    fn test_field_alias() {
        let query = r#"
            query {
                user {
                    fullName: name
                }
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                LogicalOperator::Filter(f) => find_project(&f.input),
                _ => None,
            }
        }
        let project = find_project(&plan.root).expect("Expected Project");
        assert_eq!(project.projections.len(), 1);
        assert_eq!(
            project.projections[0].alias.as_deref(),
            Some("fullName"),
            "Alias should be 'fullName' not 'name'"
        );
    }

    // === Argument Filter Tests ===

    #[test]
    fn test_argument_creates_filter() {
        let query = r#"
            query {
                user(id: 42) {
                    name
                }
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        fn has_filter(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(_) => true,
                LogicalOperator::Join(j) => has_filter(&j.left) || has_filter(&j.right),
                LogicalOperator::Project(p) => has_filter(&p.input),
                _ => false,
            }
        }
        assert!(has_filter(&plan.root), "Argument should produce a Filter");
    }

    #[test]
    fn test_multiple_arguments() {
        let query = r#"
            query {
                user(id: 42, active: true) {
                    name
                }
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        fn count_filters(op: &LogicalOperator) -> usize {
            match op {
                LogicalOperator::Filter(f) => 1 + count_filters(&f.input),
                LogicalOperator::Join(j) => count_filters(&j.left) + count_filters(&j.right),
                LogicalOperator::Project(p) => count_filters(&p.input),
                _ => 0,
            }
        }
        assert!(
            count_filters(&plan.root) >= 2,
            "Two arguments should produce at least 2 filters"
        );
    }

    // === Nested Fields ===

    #[test]
    fn test_nested_field_creates_join() {
        let query = r#"
            query {
                user {
                    name
                    friends {
                        email
                    }
                }
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        fn count_joins(op: &LogicalOperator) -> usize {
            match op {
                LogicalOperator::Join(j) => 1 + count_joins(&j.left) + count_joins(&j.right),
                LogicalOperator::Project(p) => count_joins(&p.input),
                LogicalOperator::Filter(f) => count_joins(&f.input),
                _ => 0,
            }
        }
        // At minimum: type scan joined with name, joined with friends traversal, joined with email
        assert!(
            count_joins(&plan.root) >= 3,
            "Nested field should produce multiple joins"
        );
    }

    #[test]
    fn test_deep_nesting() {
        let query = r#"
            query {
                user {
                    friends {
                        posts {
                            title
                        }
                    }
                }
            }
        "#;
        let result = translate(query, TEST_NS);
        assert!(result.is_ok(), "3-level nesting should parse");
    }

    // === Fragment Tests ===

    #[test]
    fn test_fragment_spread() {
        let query = r#"
            query {
                user {
                    ...UserFields
                }
            }
            fragment UserFields on User {
                name
                email
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        // Fragment should expand into projections
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                LogicalOperator::Filter(f) => find_project(&f.input),
                _ => None,
            }
        }
        let project = find_project(&plan.root).expect("Expected Project from fragment");
        assert_eq!(
            project.projections.len(),
            2,
            "Fragment with 2 fields should produce 2 projections"
        );
    }

    #[test]
    fn test_inline_fragment_with_type_condition() {
        let query = r#"
            query {
                user {
                    name
                    ... on Admin {
                        role
                    }
                }
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        // Should produce type check for Admin
        fn count_type_scans(op: &LogicalOperator) -> usize {
            match op {
                LogicalOperator::TripleScan(scan) => usize::from(
                    matches!(&scan.predicate, TripleComponent::Iri(iri) if iri == RDF_TYPE),
                ),
                LogicalOperator::Join(j) => count_type_scans(&j.left) + count_type_scans(&j.right),
                LogicalOperator::Filter(f) => count_type_scans(&f.input),
                LogicalOperator::Project(p) => count_type_scans(&p.input),
                _ => 0,
            }
        }
        // Should have 2 type scans: one for User, one for Admin
        assert!(
            count_type_scans(&plan.root) >= 2,
            "Inline fragment type condition should add extra rdf:type scan"
        );
    }

    // === Error Handling ===

    #[test]
    fn test_reject_empty_selection() {
        let query = r#"
            query {
            }
        "#;
        let result = translate(query, TEST_NS);
        assert!(result.is_err(), "Empty selection should be rejected");
    }

    // === Namespace and IRI Tests ===

    #[test]
    fn test_type_iri_capitalization() {
        let query = r#"
            query {
                person {
                    name
                }
            }
        "#;
        let plan = translate(query, "http://schema.org/").unwrap();
        fn find_type_iri(op: &LogicalOperator) -> Option<String> {
            match op {
                LogicalOperator::TripleScan(scan) => {
                    if matches!(&scan.predicate, TripleComponent::Iri(iri) if iri == RDF_TYPE)
                        && let TripleComponent::Iri(iri) = &scan.object
                    {
                        return Some(iri.clone());
                    }
                    None
                }
                LogicalOperator::Join(j) => {
                    find_type_iri(&j.left).or_else(|| find_type_iri(&j.right))
                }
                LogicalOperator::Filter(f) => find_type_iri(&f.input),
                LogicalOperator::Project(p) => find_type_iri(&p.input),
                _ => None,
            }
        }
        let type_iri = find_type_iri(&plan.root).expect("Should find type IRI");
        assert_eq!(
            type_iri, "http://schema.org/Person",
            "Root field 'person' should become type IRI 'Person' (capitalized)"
        );
    }

    #[test]
    fn test_predicate_iri_from_field_name() {
        let query = r#"
            query {
                user {
                    name
                }
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        fn find_predicate_iris(op: &LogicalOperator) -> Vec<String> {
            match op {
                LogicalOperator::TripleScan(scan) => {
                    if let TripleComponent::Iri(iri) = &scan.predicate
                        && iri != RDF_TYPE
                    {
                        return vec![iri.clone()];
                    }
                    Vec::new()
                }
                LogicalOperator::Join(j) => {
                    let mut v = find_predicate_iris(&j.left);
                    v.extend(find_predicate_iris(&j.right));
                    v
                }
                LogicalOperator::Filter(f) => find_predicate_iris(&f.input),
                LogicalOperator::Project(p) => find_predicate_iris(&p.input),
                _ => Vec::new(),
            }
        }
        let iris = find_predicate_iris(&plan.root);
        assert!(
            iris.contains(&format!("{TEST_NS}name")),
            "Field 'name' should produce predicate IRI '{TEST_NS}name'"
        );
    }

    // === Nested Arguments ===

    #[test]
    fn test_nested_field_with_argument() {
        let query = r#"
            query {
                user {
                    friends(active: true) {
                        name
                    }
                }
            }
        "#;
        let plan = translate(query, TEST_NS).unwrap();
        fn has_filter(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(_) => true,
                LogicalOperator::Join(j) => has_filter(&j.left) || has_filter(&j.right),
                LogicalOperator::Project(p) => has_filter(&p.input),
                _ => false,
            }
        }
        assert!(
            has_filter(&plan.root),
            "Nested field argument should produce Filter"
        );
    }
}
