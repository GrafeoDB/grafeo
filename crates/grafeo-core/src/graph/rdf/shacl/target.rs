//! SHACL target resolution.
//!
//! Resolves the set of focus nodes for a shape by evaluating its target
//! declarations against the data graph.

use std::collections::HashSet;

use crate::graph::rdf::{RdfStore, Term, TriplePattern};

use super::shape::{RDF, Shape, Target};

/// Resolves all focus nodes for a shape by evaluating its target declarations.
///
/// Multiple targets on a single shape produce the deduplicated union of all
/// matching nodes.
pub fn resolve_targets(shape: &Shape, data_graph: &RdfStore) -> Vec<Term> {
    let mut focus_nodes: HashSet<Term> = HashSet::new();

    for target in shape.targets() {
        match target {
            Target::Class(class) => {
                resolve_class_target(class, data_graph, &mut focus_nodes);
            }
            Target::Node(node) => {
                focus_nodes.insert(node.clone());
            }
            Target::SubjectsOf(predicate) => {
                for triple in data_graph.triples_with_predicate(predicate) {
                    focus_nodes.insert(triple.subject().clone());
                }
            }
            Target::ObjectsOf(predicate) => {
                for triple in data_graph.triples_with_predicate(predicate) {
                    focus_nodes.insert(triple.object().clone());
                }
            }
        }
    }

    // Implicit class target: if the shape IRI itself is used as the object
    // of an rdf:type triple in the data graph, those subjects are targets.
    if shape.targets().is_empty() && shape.id().is_iri() {
        resolve_class_target(shape.id(), data_graph, &mut focus_nodes);
    }

    focus_nodes.into_iter().collect()
}

/// Finds all subjects that are `rdf:type` instances of the given class.
fn resolve_class_target(class: &Term, data_graph: &RdfStore, out: &mut HashSet<Term>) {
    let rdf_type = Term::iri(RDF::TYPE);
    let pattern = TriplePattern {
        subject: None,
        predicate: Some(rdf_type),
        object: Some(class.clone()),
    };
    for triple in data_graph.find(&pattern) {
        out.insert(triple.subject().clone());
    }
}

#[cfg(test)]
mod tests {
    use super::super::shape::{NodeShape, Severity};
    use super::*;
    use crate::graph::rdf::{RdfStore, Triple};

    fn make_node_shape(id: &str, targets: Vec<Target>) -> Shape {
        Shape::Node(NodeShape {
            id: Term::iri(id),
            targets,
            property_shapes: Vec::new(),
            constraints: Vec::new(),
            deactivated: false,
            severity: Severity::Violation,
            messages: Vec::new(),
        })
    }

    fn sample_data() -> RdfStore {
        let store = RdfStore::new();
        let rdf_type = Term::iri(RDF::TYPE);
        let person = Term::iri("http://ex.org/Person");
        let city = Term::iri("http://ex.org/City");
        let likes = Term::iri("http://ex.org/likes");

        store.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            rdf_type.clone(),
            person.clone(),
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/gus"),
            rdf_type.clone(),
            person.clone(),
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/amsterdam"),
            rdf_type,
            city,
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            likes.clone(),
            Term::iri("http://ex.org/amsterdam"),
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/gus"),
            likes,
            Term::iri("http://ex.org/berlin"),
        ));

        store
    }

    #[test]
    fn target_class_resolves_instances() {
        let data = sample_data();
        let shape = make_node_shape(
            "http://ex.org/S",
            vec![Target::Class(Term::iri("http://ex.org/Person"))],
        );
        let nodes = resolve_targets(&shape, &data);
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn target_node_returns_specified_node() {
        let data = sample_data();
        let shape = make_node_shape(
            "http://ex.org/S",
            vec![Target::Node(Term::iri("http://ex.org/alix"))],
        );
        let nodes = resolve_targets(&shape, &data);
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&Term::iri("http://ex.org/alix")));
    }

    #[test]
    fn target_subjects_of() {
        let data = sample_data();
        let shape = make_node_shape(
            "http://ex.org/S",
            vec![Target::SubjectsOf(Term::iri("http://ex.org/likes"))],
        );
        let nodes = resolve_targets(&shape, &data);
        assert_eq!(nodes.len(), 2); // alix and gus
    }

    #[test]
    fn target_objects_of() {
        let data = sample_data();
        let shape = make_node_shape(
            "http://ex.org/S",
            vec![Target::ObjectsOf(Term::iri("http://ex.org/likes"))],
        );
        let nodes = resolve_targets(&shape, &data);
        assert_eq!(nodes.len(), 2); // amsterdam and berlin
    }

    #[test]
    fn implicit_class_target() {
        let data = sample_data();
        // Shape IRI matches a class used in rdf:type, no explicit targets
        let shape = make_node_shape("http://ex.org/Person", vec![]);
        let nodes = resolve_targets(&shape, &data);
        assert_eq!(nodes.len(), 2); // alix and gus
    }

    #[test]
    fn multiple_targets_produce_union() {
        let data = sample_data();
        let shape = make_node_shape(
            "http://ex.org/S",
            vec![
                Target::Class(Term::iri("http://ex.org/Person")),
                Target::Node(Term::iri("http://ex.org/amsterdam")),
            ],
        );
        let nodes = resolve_targets(&shape, &data);
        assert_eq!(nodes.len(), 3); // alix, gus, amsterdam
    }

    #[test]
    fn empty_target_set() {
        let data = sample_data();
        let shape = make_node_shape(
            "http://ex.org/S",
            vec![Target::Class(Term::iri("http://ex.org/NonExistent"))],
        );
        let nodes = resolve_targets(&shape, &data);
        assert!(nodes.is_empty(), "{nodes:?}");
    }

    #[test]
    fn target_with_named_graph() {
        let store = RdfStore::new();
        let graph = store.graph_or_create("http://ex.org/g1");
        let rdf_type = Term::iri(RDF::TYPE);
        let person = Term::iri("http://ex.org/Person");
        graph.insert(Triple::new(
            Term::iri("http://ex.org/alix"),
            rdf_type,
            person,
        ));

        let shape = make_node_shape(
            "http://ex.org/S",
            vec![Target::Class(Term::iri("http://ex.org/Person"))],
        );
        let nodes = resolve_targets(&shape, &graph);
        assert_eq!(nodes.len(), 1);
    }
}
