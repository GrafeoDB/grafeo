//! SHACL property path evaluation.
//!
//! Evaluates SHACL property paths starting from a focus node to produce
//! the set of value nodes reachable via the path.

use std::collections::HashSet;

use crate::graph::rdf::{GraphAt, Term, TriplePattern};

use super::shape::PropertyPath;

/// Maximum depth for transitive path expansion (cycle safety bound).
const MAX_TRANSITIVE_DEPTH: usize = 1000;

/// Evaluates a property path starting from a focus node, returning all
/// reachable value nodes.
pub fn evaluate_path(path: &PropertyPath, focus: &Term, data_graph: GraphAt<'_>) -> Vec<Term> {
    match path {
        PropertyPath::Predicate(pred) => {
            let pattern = TriplePattern {
                subject: Some(focus.clone()),
                predicate: Some(pred.clone()),
                object: None,
            };
            data_graph
                .find(&pattern)
                .iter()
                .map(|t| t.object().clone())
                .collect()
        }

        PropertyPath::Inverse(inner) => evaluate_inverse(inner, focus, data_graph),

        PropertyPath::Sequence(paths) => evaluate_sequence(paths, focus, data_graph),

        PropertyPath::Alternative(paths) => {
            let mut results: HashSet<Term> = HashSet::new();
            for p in paths {
                for term in evaluate_path(p, focus, data_graph) {
                    results.insert(term);
                }
            }
            results.into_iter().collect()
        }

        PropertyPath::ZeroOrMore(inner) => {
            let mut visited = HashSet::new();
            visited.insert(focus.clone());
            expand_transitive(inner, data_graph, &mut visited, 0);
            visited.into_iter().collect()
        }

        PropertyPath::OneOrMore(inner) => {
            // First step: evaluate inner from focus
            let first_step: Vec<Term> = evaluate_path(inner, focus, data_graph);
            let mut visited: HashSet<Term> = first_step.into_iter().collect();
            // Continue expanding transitively from all reachable nodes
            expand_transitive(inner, data_graph, &mut visited, 0);
            visited.into_iter().collect()
        }

        PropertyPath::ZeroOrOne(inner) => {
            let mut results: HashSet<Term> = HashSet::new();
            results.insert(focus.clone()); // Zero steps
            for term in evaluate_path(inner, focus, data_graph) {
                results.insert(term); // One step
            }
            results.into_iter().collect()
        }
    }
}

/// Evaluates an inverse path: finds subjects where `focus` is the object.
fn evaluate_inverse(inner: &PropertyPath, focus: &Term, data_graph: GraphAt<'_>) -> Vec<Term> {
    match inner {
        PropertyPath::Predicate(pred) => {
            let pattern = TriplePattern {
                subject: None,
                predicate: Some(pred.clone()),
                object: Some(focus.clone()),
            };
            data_graph
                .find(&pattern)
                .iter()
                .map(|t| t.subject().clone())
                .collect()
        }
        // For complex inverse paths, evaluate the inner path with swapped direction
        other => {
            // General case: find all nodes N such that evaluate_path(other, N, graph) contains focus
            // This is expensive but correct for arbitrary nested paths
            let mut results = Vec::new();
            let mut all_nodes: HashSet<Term> = data_graph.subjects().into_iter().collect();
            all_nodes.extend(data_graph.objects());
            for candidate in &all_nodes {
                let reachable = evaluate_path(other, candidate, data_graph);
                if reachable.contains(focus) {
                    results.push(candidate.clone());
                }
            }
            results
        }
    }
}

/// Evaluates a sequence path: chains evaluation through each step.
fn evaluate_sequence(paths: &[PropertyPath], focus: &Term, data_graph: GraphAt<'_>) -> Vec<Term> {
    if paths.is_empty() {
        return vec![focus.clone()];
    }

    let mut current_nodes = vec![focus.clone()];
    for path in paths {
        let mut next_nodes = Vec::new();
        for node in &current_nodes {
            next_nodes.extend(evaluate_path(path, node, data_graph));
        }
        current_nodes = next_nodes;
        if current_nodes.is_empty() {
            break;
        }
    }
    current_nodes
}

/// Expands transitive closure: adds all nodes reachable via repeated
/// application of `path` to the `visited` set.
fn expand_transitive(
    path: &PropertyPath,
    data_graph: GraphAt<'_>,
    visited: &mut HashSet<Term>,
    depth: usize,
) {
    if depth >= MAX_TRANSITIVE_DEPTH {
        return;
    }

    let frontier: Vec<Term> = visited.iter().cloned().collect();
    let mut new_nodes = Vec::new();

    for node in &frontier {
        for term in evaluate_path(path, node, data_graph) {
            if !visited.contains(&term) {
                new_nodes.push(term);
            }
        }
    }

    if new_nodes.is_empty() {
        return;
    }

    for node in &new_nodes {
        visited.insert(node.clone());
    }

    expand_transitive(path, data_graph, visited, depth + 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{RdfStore, Triple};

    /// The view the tests read through.
    static LATEST: crate::graph::rdf::RdfView =
        crate::graph::rdf::RdfView::latest(crate::graph::rdf::LatestUse::Test);

    fn chain_store() -> RdfStore {
        // a -knows-> b -knows-> c -knows-> d
        let store = RdfStore::new();
        let knows = Term::iri("http://ex.org/knows");
        let name = Term::iri("http://ex.org/name");
        store.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            knows.clone(),
            Term::iri("http://ex.org/b"),
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/b"),
            knows.clone(),
            Term::iri("http://ex.org/c"),
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/c"),
            knows,
            Term::iri("http://ex.org/d"),
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            name,
            Term::literal("Alix"),
        ));
        store
    }

    #[test]
    fn simple_predicate_path() {
        let store = chain_store();
        let path = PropertyPath::Predicate(Term::iri("http://ex.org/knows"));
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        assert_eq!(result.len(), 1);
        assert!(result.contains(&Term::iri("http://ex.org/b")));
    }

    #[test]
    fn inverse_path() {
        let store = chain_store();
        let path = PropertyPath::Inverse(Box::new(PropertyPath::Predicate(Term::iri(
            "http://ex.org/knows",
        ))));
        let result = evaluate_path(&path, &Term::iri("http://ex.org/b"), store.at(&LATEST));
        assert_eq!(result.len(), 1);
        assert!(result.contains(&Term::iri("http://ex.org/a")));
    }

    #[test]
    fn sequence_of_two() {
        let store = chain_store();
        let path = PropertyPath::Sequence(vec![
            PropertyPath::Predicate(Term::iri("http://ex.org/knows")),
            PropertyPath::Predicate(Term::iri("http://ex.org/knows")),
        ]);
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        assert_eq!(result.len(), 1);
        assert!(result.contains(&Term::iri("http://ex.org/c")));
    }

    #[test]
    fn sequence_of_three() {
        let store = chain_store();
        let path = PropertyPath::Sequence(vec![
            PropertyPath::Predicate(Term::iri("http://ex.org/knows")),
            PropertyPath::Predicate(Term::iri("http://ex.org/knows")),
            PropertyPath::Predicate(Term::iri("http://ex.org/knows")),
        ]);
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        assert_eq!(result.len(), 1);
        assert!(result.contains(&Term::iri("http://ex.org/d")));
    }

    #[test]
    fn alternative_path() {
        let store = chain_store();
        let path = PropertyPath::Alternative(vec![
            PropertyPath::Predicate(Term::iri("http://ex.org/knows")),
            PropertyPath::Predicate(Term::iri("http://ex.org/name")),
        ]);
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        assert_eq!(result.len(), 2); // b and "Alix"
    }

    #[test]
    fn zero_or_more_linear() {
        let store = chain_store();
        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Predicate(Term::iri(
            "http://ex.org/knows",
        ))));
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        // a (zero steps), b (one), c (two), d (three)
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn zero_or_more_with_cycle() {
        let store = RdfStore::new();
        let knows = Term::iri("http://ex.org/knows");
        // a -> b -> c -> a (cycle)
        store.insert(Triple::new(
            Term::iri("http://ex.org/a"),
            knows.clone(),
            Term::iri("http://ex.org/b"),
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/b"),
            knows.clone(),
            Term::iri("http://ex.org/c"),
        ));
        store.insert(Triple::new(
            Term::iri("http://ex.org/c"),
            knows,
            Term::iri("http://ex.org/a"),
        ));

        let path = PropertyPath::ZeroOrMore(Box::new(PropertyPath::Predicate(Term::iri(
            "http://ex.org/knows",
        ))));
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        assert_eq!(result.len(), 3); // a, b, c (cycle detected, no infinite loop)
    }

    #[test]
    fn one_or_more() {
        let store = chain_store();
        let path = PropertyPath::OneOrMore(Box::new(PropertyPath::Predicate(Term::iri(
            "http://ex.org/knows",
        ))));
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        // b (one), c (two), d (three) — NOT a (zero steps excluded)
        assert_eq!(result.len(), 3);
        assert!(!result.contains(&Term::iri("http://ex.org/a")));
    }

    #[test]
    fn zero_or_one() {
        let store = chain_store();
        let path = PropertyPath::ZeroOrOne(Box::new(PropertyPath::Predicate(Term::iri(
            "http://ex.org/knows",
        ))));
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        assert_eq!(result.len(), 2); // a (zero) and b (one)
    }

    #[test]
    fn nested_inverse_in_sequence() {
        let store = chain_store();
        // Forward then inverse: a -knows-> b, then inverse of knows from b = a
        let path = PropertyPath::Sequence(vec![
            PropertyPath::Predicate(Term::iri("http://ex.org/knows")),
            PropertyPath::Inverse(Box::new(PropertyPath::Predicate(Term::iri(
                "http://ex.org/knows",
            )))),
        ]);
        let result = evaluate_path(&path, &Term::iri("http://ex.org/a"), store.at(&LATEST));
        // a -knows-> b, then inverse(knows) from b = a
        assert!(result.contains(&Term::iri("http://ex.org/a")));
    }
}
