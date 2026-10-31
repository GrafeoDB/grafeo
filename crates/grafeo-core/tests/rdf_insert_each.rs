//! `RdfStore::insert_each` (0.6.1): one batch, and for each triple, in order,
//! whether it was new, as `insert` one by one says: `false` for a triple the
//! store holds, for a repeat after its first, and for one that is no RDF
//! triple (a literal or blank-node predicate, a literal subject). The oracle is
//! a second store that gets the same triples through `insert`; the flags, the
//! triples held and the lookups must agree, below and above the parallel
//! index threshold (4,096 new triples), and in a named graph through
//! `insert_each_into`.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test rdf_insert_each
//! ```

#![cfg(feature = "triple-store")]

use std::collections::BTreeSet;
use std::sync::Arc;

use grafeo_core::graph::rdf::{RdfStore, Term, Triple, TriplePattern};

fn iri(name: &str) -> Term {
    Term::iri(format!("http://ex.org/{name}"))
}

fn triple(subject: &str, object: &str) -> Triple {
    Triple::new(iri(subject), iri("p"), Term::literal(object))
}

/// A literal subject: no RDF triple.
fn literal_subject() -> Triple {
    Triple::new_unchecked(Term::literal("Alix"), iri("p"), Term::literal("x"))
}

/// A blank-node predicate: no RDF triple.
fn blank_predicate() -> Triple {
    Triple::new_unchecked(iri("gus"), Term::blank("b"), Term::literal("x"))
}

fn names(triples: Vec<Arc<Triple>>) -> BTreeSet<String> {
    triples.iter().map(ToString::to_string).collect()
}

/// The flags `insert` gives for `batch`, one by one, on a store that holds
/// `held`, and that store.
fn one_by_one(held: &[Triple], batch: &[Triple]) -> (Vec<bool>, RdfStore) {
    let store = RdfStore::new();
    for triple in held {
        store.insert(triple.clone());
    }
    let flags = batch.iter().map(|t| store.insert(t.clone())).collect();
    (flags, store)
}

fn same_store(batched: &RdfStore, single: &RdfStore, batch: &[Triple]) {
    assert_eq!(batched.len(), single.len());
    assert_eq!(names(batched.triples()), names(single.triples()));
    for triple in batch.iter().filter(|t| t.is_valid()) {
        for pattern in [
            TriplePattern {
                subject: Some(triple.subject().clone()),
                predicate: None,
                object: None,
            },
            TriplePattern {
                subject: None,
                predicate: Some(triple.predicate().clone()),
                object: Some(triple.object().clone()),
            },
        ] {
            assert_eq!(names(batched.find(&pattern)), names(single.find(&pattern)));
        }
    }
}

#[test]
fn flags_of_a_mixed_batch_match_single_inserts() {
    let held = [triple("held", "H")];
    let batch = vec![
        triple("alix", "A"),
        triple("gus", "B"),
        literal_subject(),
        triple("alix", "A"),
        triple("held", "H"),
        triple("vincent", "C"),
        blank_predicate(),
        triple("gus", "B"),
        literal_subject(),
        triple("jules", "D"),
    ];
    let (want, single) = one_by_one(&held, &batch);
    assert_eq!(
        want,
        [
            true, true, false, false, false, true, false, false, false, true
        ]
    );
    let store = RdfStore::new();
    store.insert(held[0].clone());
    assert_eq!(store.insert_each(batch.clone()), want);
    same_store(&store, &single, &batch);
    assert_eq!(store.len(), 5);
}

#[test]
fn an_empty_batch_and_a_batch_of_invalid_triples_insert_nothing() {
    let store = RdfStore::new();
    store.insert(triple("held", "H"));
    assert_eq!(store.insert_each(Vec::new()), Vec::<bool>::new());
    assert_eq!(
        store.insert_each(vec![
            literal_subject(),
            blank_predicate(),
            literal_subject()
        ]),
        [false, false, false]
    );
    assert_eq!(store.len(), 1);
}

#[test]
fn a_batch_past_the_parallel_threshold_flags_as_single_inserts_do() {
    for new in [4_095, 4_096, 6_000] {
        let held: Vec<Triple> = (0..300).map(|i| triple(&format!("h{i}"), "held")).collect();
        let fresh: Vec<Triple> = (0..new)
            .map(|i| triple(&format!("s{}", i % 997), &format!("o{i}")))
            .collect();
        // Each new triple, then the held ones, a few invalid ones, then the
        // new ones again in reverse.
        let mut batch = fresh.clone();
        batch.extend(held.iter().cloned());
        batch.extend([literal_subject(), blank_predicate()]);
        batch.extend(fresh.iter().rev().cloned());

        let (want, single) = one_by_one(&held, &batch);
        let store = RdfStore::new();
        for triple in &held {
            store.insert(triple.clone());
        }
        let got = store.insert_each(batch.clone());
        assert_eq!(got.len(), batch.len());
        assert_eq!(got.iter().filter(|new| **new).count(), new, "{new}");
        assert_eq!(got, want, "{new}");
        assert!(
            got[..new].iter().all(|new| *new),
            "the first of each is new"
        );
        same_store(&store, &single, &batch);
    }
}

#[test]
fn insert_each_into_a_named_graph_creates_it_and_leaves_the_default_alone() {
    let store = RdfStore::new();
    store.insert(triple("held", "H"));
    let graph = "http://ex.org/g1";
    let batch = vec![
        triple("held", "H"),
        triple("alix", "A"),
        triple("alix", "A"),
        literal_subject(),
    ];
    assert_eq!(
        store.insert_each_into(Some(graph), batch.clone()),
        [true, true, false, false],
        "the default graph's triple is new in g1"
    );
    let named = store.graph(graph).expect("the insert created g1");
    assert_eq!(named.len(), 2);
    assert_eq!(store.len(), 1, "the default graph is unchanged");
    assert_eq!(
        store.insert_each_into(None, batch),
        [false, true, false, false]
    );
    assert_eq!(store.len(), 2);
}
