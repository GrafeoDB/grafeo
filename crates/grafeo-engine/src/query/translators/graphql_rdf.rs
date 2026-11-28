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
//!
//! # Names
//!
//! A type or field name becomes an IRI through [`RdfNames`]. Against a store
//! ([`StoreNames`]) a name matches the class or predicate of the data whose local
//! name (after its last `#` or `/`) it is, in any namespace: `Person` finds
//! `schema:Person`, `worksAt` finds `sv:worksAt`. Two such IRIs are an error naming
//! both; none falls back to [`DEFAULT_NAMESPACE`].

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

/// The namespace of a type or field name the data does not hold.
pub const DEFAULT_NAMESPACE: &str = "http://example.org/";

/// How GraphQL type and field names become RDF IRIs.
pub trait RdfNames {
    /// The class IRI of type `name` (`Person`, or `person` for a root field).
    ///
    /// # Errors
    ///
    /// When the name is ambiguous.
    fn type_iri(&self, name: &str) -> Result<String>;

    /// The predicate IRI of field `name`.
    ///
    /// # Errors
    ///
    /// When the name is ambiguous.
    fn predicate_iri(&self, name: &str) -> Result<String>;
}

/// Every name in one namespace: `<namespace>Person`, `<namespace>name`.
pub struct Namespace<'a>(pub &'a str);

impl RdfNames for Namespace<'_> {
    fn type_iri(&self, name: &str) -> Result<String> {
        Ok(format!("{}{}", self.0, capitalize_first(name)))
    }

    fn predicate_iri(&self, name: &str) -> Result<String> {
        Ok(format!("{}{name}", self.0))
    }
}

/// The namespace of `iri` (up to its last `#` or `/`) and its local name after it.
fn split_iri(iri: &str) -> (&str, &str) {
    let at = iri.rfind(['#', '/']).map_or(0, |i| i + 1);
    (&iri[..at], &iri[at..])
}

/// Names as the data in an RDF store uses them: a field matches the committed
/// predicates whose local name it is, a type the classes (objects of `rdf:type`
/// at the query's view) whose local name it is, in the namespaces those
/// predicates use. A predicate first written in the open transaction is matched
/// by name once it commits (listing the predicates at a view with changes reads
/// every triple), one deleted by a commit after the query's snapshot is no
/// longer matched, and a class whose namespace holds no predicate of the data is
/// not found: both fall back to [`DEFAULT_NAMESPACE`].
pub struct StoreNames<'a> {
    store: &'a grafeo_core::graph::rdf::RdfStore,
    view: &'a grafeo_core::graph::rdf::RdfView,
    predicates: std::cell::OnceCell<Vec<String>>,
}

impl<'a> StoreNames<'a> {
    /// Names of the data `view` sees in `store`.
    #[must_use]
    pub fn new(
        store: &'a grafeo_core::graph::rdf::RdfStore,
        view: &'a grafeo_core::graph::rdf::RdfView,
    ) -> Self {
        Self {
            store,
            view,
            predicates: std::cell::OnceCell::new(),
        }
    }

    /// The committed predicate IRIs, sorted, read once per query.
    fn predicates(&self) -> &[String] {
        self.predicates.get_or_init(|| {
            // The committed state: one entry per predicate, from its index.
            let committed = grafeo_core::graph::rdf::RdfView::latest(
                grafeo_core::graph::rdf::LatestUse::Statistics,
            );
            let mut iris: Vec<String> = self
                .store
                .predicates(&committed)
                .into_iter()
                .filter_map(|term| term.as_iri().map(|iri| iri.as_str().to_string()))
                .collect();
            iris.sort_unstable();
            iris.dedup();
            iris
        })
    }

    /// The one IRI of `found`; `default` when there is none or it is one of
    /// them (as 0.6.3 read every name there); an error naming them when there
    /// are more.
    fn pick(kind: &str, name: &str, found: Vec<String>, default: String) -> Result<String> {
        if found.contains(&default) {
            return Ok(default);
        }
        let mut found = found.into_iter();
        match (found.next(), found.next()) {
            (None, _) => Ok(default),
            (Some(one), None) => Ok(one),
            (Some(first), Some(second)) => {
                let all: Vec<String> = [first, second]
                    .into_iter()
                    .chain(found)
                    .map(|iri| format!("<{iri}>"))
                    .collect();
                Err(Error::from(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "GraphQL {kind} '{name}' matches {} IRIs in the data: {}; query it through SPARQL, which names the IRI",
                        all.len(),
                        all.join(", ")
                    ),
                )))
            }
        }
    }
}

impl RdfNames for StoreNames<'_> {
    fn type_iri(&self, name: &str) -> Result<String> {
        use grafeo_core::graph::rdf::{Term, TriplePattern};
        let local = capitalize_first(name);
        let mut namespaces: Vec<&str> = self
            .predicates()
            .iter()
            .map(|iri| split_iri(iri).0)
            .collect();
        namespaces.sort_unstable();
        namespaces.dedup();
        let found = namespaces
            .into_iter()
            .map(|namespace| format!("{namespace}{local}"))
            .filter(|class| {
                let pattern = TriplePattern {
                    subject: None,
                    predicate: Some(Term::iri(RDF_TYPE)),
                    object: Some(Term::iri(class.as_str())),
                };
                self.store.count_at(self.view, &pattern) > 0
            })
            .collect();
        Self::pick(
            "type",
            name,
            found,
            Namespace(DEFAULT_NAMESPACE).type_iri(name)?,
        )
    }

    fn predicate_iri(&self, name: &str) -> Result<String> {
        let found = self
            .predicates()
            .iter()
            .filter(|iri| split_iri(iri).1 == name)
            .cloned()
            .collect();
        Self::pick(
            "field",
            name,
            found,
            Namespace(DEFAULT_NAMESPACE).predicate_iri(name)?,
        )
    }
}

/// Translates a GraphQL query string to an RDF logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(
    query: &str,
    names: &dyn RdfNames,
    limits: &grafeo_adapters::query::limits::ParseLimits,
) -> Result<LogicalPlan> {
    // The statement's length is its text's, the `EXPLAIN` included.
    limits.check_query(query)?;
    let (explain, profile, query) = super::graphql::explain_prefix(query);
    let doc = graphql::parse_with(query, limits)?;
    let translator = GraphQLRdfTranslator::new(names);
    let plan = translator.translate_document(&doc)?;
    let mut plan = crate::query::limits::check_plan_depth(plan)?;
    plan.explain = explain;
    plan.profile = profile;
    Ok(plan)
}

/// Translator from GraphQL AST to RDF LogicalPlan.
struct GraphQLRdfTranslator<'a> {
    /// Generator for anonymous variable names.
    var_gen: VarGen,
    /// How type and field names become IRIs.
    names: &'a dyn RdfNames,
    /// Fragment definitions for resolution.
    fragments: HashMap<String, ast::FragmentDefinition>,
    /// Default values from variable declarations (e.g., `query($limit: Int = 2)`).
    variable_defaults: HashMap<String, grafeo_common::types::Value>,
}

impl<'a> GraphQLRdfTranslator<'a> {
    fn new(names: &'a dyn RdfNames) -> Self {
        Self {
            var_gen: VarGen::new(),
            names,
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
                Error::from(QueryError::new(
                    QueryErrorKind::Syntax,
                    "No operation found in document",
                ))
            })?;

        // Only Query operations are supported
        if operation.operation != ast::OperationType::Query {
            return Err(Error::from(QueryError::new(
                QueryErrorKind::Unsupported,
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
            names: self.names,
            fragments,
            variable_defaults,
        };

        translator.translate_operation(operation)
    }

    fn translate_operation(&self, op: &ast::OperationDefinition) -> Result<LogicalPlan> {
        // Each field in the root selection set is a separate query
        let selections = &op.selection_set.selections;
        if selections.is_empty() {
            return Err(Error::from(QueryError::new(
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
            return Err(Error::from(QueryError::new(
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
        let type_iri = self.make_type_iri(&field.name)?;

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
                        // `<field>_<column>`, as GraphQL over the property graph names them.
                        let prefix = field.alias.as_ref().unwrap_or(&field.name);
                        projections.extend(nested_projections.into_iter().map(|p| Projection {
                            alias: p.alias.map(|column| format!("{prefix}_{column}")),
                            ..p
                        }));
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
                        let type_iri = self.make_type_iri(type_cond)?;
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
        let predicate_iri = self.make_predicate_iri(&field.name)?;

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
        let predicate_iri = self.make_predicate_iri(&field.name)?;

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
            let predicate_iri = self.make_predicate_iri(&arg.name)?;
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
        let type_iri = self.make_type_iri(&frag.type_condition)?;
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

    fn make_type_iri(&self, type_name: &str) -> Result<String> {
        self.names.type_iri(type_name)
    }

    fn make_predicate_iri(&self, name: &str) -> Result<String> {
        self.names.predicate_iri(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`super::translate`] under the default parse limits.
    fn translate(query: &str, namespace: &str) -> Result<LogicalPlan> {
        super::translate(
            query,
            &Namespace(namespace),
            &grafeo_adapters::query::limits::ParseLimits::default(),
        )
    }

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
