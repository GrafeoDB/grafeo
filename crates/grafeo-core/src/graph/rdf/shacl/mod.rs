//! SHACL (Shapes Constraint Language) validation for RDF graphs.
//!
//! This module implements the W3C SHACL specification for validating RDF data
//! against shape definitions. It supports both SHACL Core constraints (evaluated
//! purely against the RDF store) and SHACL-SPARQL constraints (evaluated via
//! an optional SPARQL executor callback).
//!
//! # Architecture
//!
//! - **Shape model** (`shape.rs`): data types for shapes, targets, paths, constraints
//! - **Parser** (`parser.rs`): reads shape definitions from an RDF store
//! - **Target resolution** (`target.rs`): finds focus nodes for each shape
//! - **Path evaluation** (`path.rs`): evaluates SHACL property paths
//! - **Constraint evaluation** (`constraint.rs`): checks constraints against value nodes
//! - **Report** (`report.rs`): validation results in W3C format

pub mod constraint;
mod parser;
pub mod path;
pub mod report;
pub mod shape;
mod target;

use std::collections::{HashMap, HashSet};

use crate::graph::rdf::{GraphAt, Term};

pub use constraint::evaluate_constraint;
pub use parser::parse_shapes;
pub use path::evaluate_path;
pub use report::{ValidationReport, ValidationResult};
pub use shape::{
    Constraint, NodeKindValue, NodeShape, PropertyPath, PropertyShape, SH, Severity, ShaclError,
    Shape, Target,
};
pub use target::resolve_targets;

// =========================================================================
// SPARQL executor trait (for core/engine decoupling)
// =========================================================================

/// Trait for executing SPARQL queries from SHACL-SPARQL constraints.
///
/// Implemented by the engine layer (`SessionSparqlExecutor`) to provide
/// query execution without making grafeo-core depend on grafeo-engine.
pub trait SparqlExecutor {
    /// Executes a SPARQL SELECT query with `$this` bound to the focus node.
    ///
    /// Returns result rows as maps of variable name to term value.
    ///
    /// # Errors
    ///
    /// Returns `ShaclError::SparqlError` if execution fails.
    fn execute(
        &self,
        query: &str,
        this_binding: &Term,
    ) -> Result<Vec<HashMap<String, Term>>, ShaclError>;
}

// =========================================================================
// Top-level validation orchestration
// =========================================================================

/// Validates RDF data against SHACL shapes.
///
/// Parses shapes from the shapes graph, resolves targets in the data graph,
/// evaluates all constraints, and returns a validation report.
///
/// # Arguments
///
/// * `data_graph` - The RDF store containing data to validate
/// * `shapes_graph` - The RDF store containing SHACL shape definitions
/// * `sparql_executor` - Optional executor for SHACL-SPARQL constraints
///
/// # Errors
///
/// Returns `ShaclError` if shape parsing fails or a constraint cannot be evaluated.
pub fn validate(
    data_graph: GraphAt<'_>,
    shapes_graph: GraphAt<'_>,
    sparql_executor: Option<&dyn SparqlExecutor>,
) -> Result<ValidationReport, ShaclError> {
    let shapes = parse_shapes(shapes_graph)?;
    let mut all_results = Vec::new();

    for shape in &shapes {
        if shape.is_deactivated() {
            continue;
        }

        let focus_nodes = resolve_targets(shape, data_graph);
        for focus_node in &focus_nodes {
            let results = validate_shape(shape, focus_node, data_graph, &shapes, sparql_executor)?;
            all_results.extend(results);
        }
    }

    Ok(ValidationReport::from_results(all_results))
}

/// Validates a single shape against a focus node.
fn validate_shape(
    shape: &Shape,
    focus_node: &Term,
    data_graph: GraphAt<'_>,
    all_shapes: &[Shape],
    sparql_executor: Option<&dyn SparqlExecutor>,
) -> Result<Vec<ValidationResult>, ShaclError> {
    let mut visited = HashSet::new();
    let mut results = Vec::new();

    match shape {
        Shape::Node(ns) => {
            // Evaluate node-level constraints with focus node as the single value node
            let mut ctx = constraint::EvalContext {
                focus_node,
                shape,
                path: None,
                data_graph,
                all_shapes,
                visited: &mut visited,
            };
            for c in &ns.constraints {
                let value_nodes = vec![focus_node.clone()];
                results.extend(evaluate_constraint(c, &value_nodes, &mut ctx));
            }

            // Evaluate SPARQL constraints on the node shape
            for c in &ns.constraints {
                if let Constraint::Sparql(sc) = c {
                    results.extend(evaluate_sparql_constraint(
                        sc,
                        focus_node,
                        shape,
                        None,
                        sparql_executor,
                    )?);
                }
            }

            // Evaluate nested property shapes
            for ps in &ns.property_shapes {
                if ps.deactivated {
                    continue;
                }
                let path_values = evaluate_path(&ps.path, focus_node, data_graph);
                let ps_shape = Shape::Property(ps.clone());
                let mut ps_ctx = constraint::EvalContext {
                    focus_node,
                    shape: &ps_shape,
                    path: Some(&ps.path),
                    data_graph,
                    all_shapes,
                    visited: &mut visited,
                };
                for c in &ps.constraints {
                    if let Constraint::Sparql(sc) = c {
                        results.extend(evaluate_sparql_constraint(
                            sc,
                            focus_node,
                            &ps_shape,
                            Some(&ps.path),
                            sparql_executor,
                        )?);
                    } else {
                        results.extend(evaluate_constraint(c, &path_values, &mut ps_ctx));
                    }
                }
            }
        }
        Shape::Property(ps) => {
            let path_values = evaluate_path(&ps.path, focus_node, data_graph);
            let mut ctx = constraint::EvalContext {
                focus_node,
                shape,
                path: Some(&ps.path),
                data_graph,
                all_shapes,
                visited: &mut visited,
            };
            for c in &ps.constraints {
                if let Constraint::Sparql(sc) = c {
                    results.extend(evaluate_sparql_constraint(
                        sc,
                        focus_node,
                        shape,
                        Some(&ps.path),
                        sparql_executor,
                    )?);
                } else {
                    results.extend(evaluate_constraint(c, &path_values, &mut ctx));
                }
            }
        }
    }

    Ok(results)
}

/// Evaluates a SHACL-SPARQL constraint using the optional executor.
fn evaluate_sparql_constraint(
    sc: &shape::SparqlConstraint,
    focus_node: &Term,
    shape: &Shape,
    result_path: Option<&PropertyPath>,
    sparql_executor: Option<&dyn SparqlExecutor>,
) -> Result<Vec<ValidationResult>, ShaclError> {
    if sc.deactivated {
        return Ok(Vec::new());
    }

    let Some(executor) = sparql_executor else {
        // No executor provided: skip SPARQL constraints silently
        return Ok(Vec::new());
    };

    // Build the full query with prefix declarations
    let mut query = String::new();
    for decl in &sc.prefixes {
        use std::fmt::Write;
        let _ = writeln!(query, "PREFIX {}: <{}>", decl.prefix, decl.namespace);
    }
    query.push_str(&sc.select);

    // Execute the query, propagating errors instead of swallowing them
    let rows = executor.execute(&query, focus_node)?;

    // Each result row is a violation
    let mut results = Vec::new();
    for row in &rows {
        let value = row.get("value").cloned();
        let message = row
            .get("message")
            .and_then(|t| match t {
                Term::Literal(lit) => Some(lit.value().to_string()),
                _ => None,
            })
            .or_else(|| sc.message.clone());

        results.push(ValidationResult {
            focus_node: focus_node.clone(),
            source_constraint_component: format!("{}SPARQLConstraintComponent", SH::NS),
            source_shape: shape.id().clone(),
            value,
            result_path: result_path.cloned(),
            severity: shape.severity(),
            message,
        });
    }

    Ok(results)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{RdfStore, Triple};

    /// The view the tests read through.
    static LATEST: crate::graph::rdf::RdfView =
        crate::graph::rdf::RdfView::latest(crate::graph::rdf::LatestUse::Test);

    #[test]
    fn validate_conforming_data() {
        let data = RdfStore::new();
        let rdf_type = Term::iri(shape::RDF::TYPE);
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            rdf_type,
            Term::iri("http://ex.org/Person"),
        ));
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://ex.org/name"),
            Term::literal("Alix"),
        ));

        let shapes = RdfStore::new();
        let shape_id = Term::iri("http://ex.org/PersonShape");
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        let prop = Term::blank("p");
        shapes.insert(Triple::new(shape_id, Term::iri(SH::PROPERTY), prop.clone()));
        shapes.insert(Triple::new(
            prop.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/name"),
        ));
        shapes.insert(Triple::new(
            prop,
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        assert!(report.conforms, "Data should conform: {report}");
    }

    #[test]
    fn validate_with_violations() {
        let data = RdfStore::new();
        let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            rdf_type,
            Term::iri("http://ex.org/Person"),
        ));
        // No name property: violates minCount 1

        let shapes = RdfStore::new();
        let shape_id = Term::iri("http://ex.org/PersonShape");
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        let prop = Term::blank("p");
        shapes.insert(Triple::new(shape_id, Term::iri(SH::PROPERTY), prop.clone()));
        shapes.insert(Triple::new(
            prop.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/name"),
        ));
        shapes.insert(Triple::new(
            prop,
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        assert!(!report.conforms);
        assert_eq!(report.results.len(), 1);
    }

    #[test]
    fn empty_shapes_conforms() {
        let data = RdfStore::new();
        let shapes = RdfStore::new();
        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        assert!(report.conforms);
        assert!(report.results.is_empty());
    }

    #[test]
    fn deactivated_shape_skipped() {
        let data = RdfStore::new();
        let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            rdf_type,
            Term::iri("http://ex.org/Person"),
        ));

        let shapes = RdfStore::new();
        let shape_id = Term::iri("http://ex.org/PersonShape");
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri(SH::DEACTIVATED),
            Term::typed_literal("true", "http://www.w3.org/2001/XMLSchema#boolean"),
        ));
        let prop = Term::blank("p");
        shapes.insert(Triple::new(shape_id, Term::iri(SH::PROPERTY), prop.clone()));
        shapes.insert(Triple::new(
            prop.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/name"),
        ));
        shapes.insert(Triple::new(
            prop,
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        assert!(report.conforms, "Deactivated shape should be skipped");
    }

    #[test]
    fn report_to_triples() {
        let data = RdfStore::new();
        let rdf_type = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            rdf_type,
            Term::iri("http://ex.org/Person"),
        ));

        let shapes = RdfStore::new();
        let shape_id = Term::iri("http://ex.org/S");
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            shape_id.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        let prop = Term::blank("p");
        shapes.insert(Triple::new(shape_id, Term::iri(SH::PROPERTY), prop.clone()));
        shapes.insert(Triple::new(
            prop.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/name"),
        ));
        shapes.insert(Triple::new(
            prop,
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        let triples = report.to_triples();
        // Should have: report type, conforms, result link, result type, focus, component, shape, severity, message
        assert!(
            triples.len() >= 5,
            "Expected at least 5 triples, got {}",
            triples.len()
        );
    }

    #[test]
    fn display_format() {
        let report = ValidationReport::from_results(vec![ValidationResult {
            focus_node: Term::iri("http://ex.org/alix"),
            source_constraint_component: format!("{}MinCountConstraintComponent", SH::NS),
            source_shape: Term::iri("http://ex.org/S"),
            value: None,
            result_path: Some(PropertyPath::Predicate(Term::iri("http://ex.org/name"))),
            severity: Severity::Violation,
            message: Some("Expected at least 1 value(s), got 0".to_string()),
        }]);
        let text = format!("{report}");
        assert!(text.contains("FAILED"));
        assert!(text.contains("Violation"));
    }

    #[test]
    fn validate_property_shape_with_datatype() {
        let data = RdfStore::new();
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri("http://ex.org/Person"),
        ));
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://ex.org/age"),
            Term::literal("not-a-number"), // wrong datatype
        ));

        let shapes = RdfStore::new();
        let sid = Term::iri("http://ex.org/S");
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        let prop = Term::blank("p");
        shapes.insert(Triple::new(sid, Term::iri(SH::PROPERTY), prop.clone()));
        shapes.insert(Triple::new(
            prop.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/age"),
        ));
        shapes.insert(Triple::new(
            prop,
            Term::iri(SH::DATATYPE),
            Term::iri("http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        assert!(!report.conforms);
        assert_eq!(report.results.len(), 1);
        assert!(
            report.results[0]
                .source_constraint_component
                .contains("Datatype")
        );
        assert!(
            report.results[0].result_path.is_some(),
            "Property shape results should include path"
        );
    }

    #[test]
    fn validate_multiple_property_shapes() {
        let data = RdfStore::new();
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri("http://ex.org/Person"),
        ));
        // Missing both name and email

        let shapes = RdfStore::new();
        let sid = Term::iri("http://ex.org/S");
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        let p1 = Term::blank("p1");
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri(SH::PROPERTY),
            p1.clone(),
        ));
        shapes.insert(Triple::new(
            p1.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/name"),
        ));
        shapes.insert(Triple::new(
            p1,
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        let p2 = Term::blank("p2");
        shapes.insert(Triple::new(sid, Term::iri(SH::PROPERTY), p2.clone()));
        shapes.insert(Triple::new(
            p2.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/email"),
        ));
        shapes.insert(Triple::new(
            p2,
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        assert!(!report.conforms);
        assert_eq!(
            report.results.len(),
            2,
            "Should have violations for both name and email"
        );
    }

    #[test]
    fn validate_standalone_property_shape() {
        let data = RdfStore::new();
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri("http://ex.org/Person"),
        ));

        let shapes = RdfStore::new();
        // A property shape with its own target (not nested under a node shape)
        let sid = Term::iri("http://ex.org/PS");
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::PROPERTY_SHAPE),
        ));
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri(SH::PATH),
            Term::iri("http://ex.org/name"),
        ));
        shapes.insert(Triple::new(
            sid,
            Term::iri(SH::MIN_COUNT),
            Term::typed_literal("1", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        assert!(
            !report.conforms,
            "Missing name should violate standalone property shape"
        );
        assert_eq!(report.results.len(), 1);
    }

    #[test]
    fn validate_node_level_constraint() {
        let data = RdfStore::new();
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri("http://ex.org/Person"),
        ));

        let shapes = RdfStore::new();
        let sid = Term::iri("http://ex.org/S");
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        // Node-level constraint: focus node must be IRI
        shapes.insert(Triple::new(
            sid,
            Term::iri(SH::NODE_KIND),
            Term::iri(SH::IRI),
        ));

        let report = validate(data.at(&LATEST), shapes.at(&LATEST), None).unwrap();
        assert!(report.conforms, "alix is an IRI: {report}");
    }

    #[test]
    fn validate_with_mock_sparql_executor() {
        use std::collections::HashMap;

        struct MockExecutor;
        impl SparqlExecutor for MockExecutor {
            fn execute(
                &self,
                _query: &str,
                _this_binding: &Term,
            ) -> Result<Vec<HashMap<String, Term>>, ShaclError> {
                // Return one violation row
                let mut row = HashMap::new();
                row.insert("value".to_string(), Term::literal("bad"));
                row.insert("message".to_string(), Term::literal("Mock violation"));
                Ok(vec![row])
            }
        }

        let data = RdfStore::new();
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri("http://ex.org/Person"),
        ));

        let shapes = RdfStore::new();
        let sid = Term::iri("http://ex.org/S");
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        let sq = Term::blank("sq");
        shapes.insert(Triple::new(sid, Term::iri(SH::SPARQL), sq.clone()));
        shapes.insert(Triple::new(
            sq,
            Term::iri(SH::SELECT),
            Term::literal("SELECT $this ?value WHERE { $this ?p ?value }"),
        ));

        let executor = MockExecutor;
        let report = validate(data.at(&LATEST), shapes.at(&LATEST), Some(&executor)).unwrap();
        assert!(!report.conforms);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].message.as_deref(), Some("Mock violation"));
        assert_eq!(report.results[0].value, Some(Term::literal("bad")));
    }

    #[test]
    fn validate_sparql_error_propagates() {
        use std::collections::HashMap;

        struct FailingExecutor;
        impl SparqlExecutor for FailingExecutor {
            fn execute(
                &self,
                _query: &str,
                _this_binding: &Term,
            ) -> Result<Vec<HashMap<String, Term>>, ShaclError> {
                Err(ShaclError::SparqlError("query failed".to_string()))
            }
        }

        let data = RdfStore::new();
        data.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri("http://ex.org/Person"),
        ));

        let shapes = RdfStore::new();
        let sid = Term::iri("http://ex.org/S");
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Term::iri(SH::NODE_SHAPE),
        ));
        shapes.insert(Triple::new(
            sid.clone(),
            Term::iri(SH::TARGET_CLASS),
            Term::iri("http://ex.org/Person"),
        ));
        let sq = Term::blank("sq");
        shapes.insert(Triple::new(sid, Term::iri(SH::SPARQL), sq.clone()));
        shapes.insert(Triple::new(
            sq,
            Term::iri(SH::SELECT),
            Term::literal("SELECT $this WHERE { $this ?p ?o }"),
        ));

        let executor = FailingExecutor;
        let result = validate(data.at(&LATEST), shapes.at(&LATEST), Some(&executor));
        assert!(
            result.is_err(),
            "SPARQL errors should propagate as ShaclError"
        );
    }
}
