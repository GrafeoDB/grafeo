//! The RDF statistics count what the old string collector counted: the
//! totals (triples, distinct subjects, predicates and objects) and per
//! predicate its triples and distinct subjects and objects. The oracle names
//! each term as `to_string` writes it and counts the names, as the collector
//! before 0.6.1 did; the store counts the terms themselves, so the two agree
//! only when distinct terms have distinct names. The stores mix IRIs, blank
//! nodes, plain, typed and language-tagged literals with the same lexical
//! form, escapes, named graphs and a triple repeated across graphs.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test rdf_statistics_counts
//! ```

#![cfg(feature = "triple-store")]

use std::collections::{BTreeMap, BTreeSet};

use grafeo_core::graph::rdf::{RdfStore, Term, Triple};
use grafeo_core::statistics::RdfStatistics;

const EX: &str = "http://ex.org/";
const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";

fn iri(name: &str) -> Term {
    Term::iri(format!("{EX}{name}"))
}

fn triple(subject: Term, predicate: &str, object: Term) -> Triple {
    Triple::new(subject, iri(predicate), object)
}

/// Every kind of term, and literals that share a lexical form.
fn mixed() -> Vec<Triple> {
    let mut triples = Vec::new();
    for person in ["alix", "gus", "vincent", "jules"] {
        triples.push(triple(iri(person), "type", iri("Person")));
        triples.push(triple(iri(person), "name", Term::literal(person)));
    }
    triples.extend([
        // "42" as a string, an integer and a language-tagged string: three objects.
        triple(iri("alix"), "label", Term::literal("42")),
        triple(iri("gus"), "label", Term::typed_literal("42", XSD_INTEGER)),
        triple(iri("vincent"), "label", Term::lang_literal("42", "en")),
        triple(iri("jules"), "label", Term::lang_literal("42", "nl")),
        // An IRI and a literal with the same text.
        triple(iri("mia"), "seeAlso", Term::iri(format!("{EX}alix"))),
        triple(iri("mia"), "seeAlso", Term::literal(format!("{EX}alix"))),
        // Blank nodes as subjects and objects.
        triple(Term::blank("b1"), "knows", iri("alix")),
        triple(Term::blank("b2"), "knows", iri("alix")),
        triple(iri("gus"), "knows", Term::blank("b1")),
        // Escapes in literals.
        triple(iri("butch"), "quote", Term::literal("say \"hi\"\nand\tgo")),
        triple(iri("butch"), "quote", Term::literal("back\\slash")),
        // One subject, one predicate, many objects; many subjects, one object.
        triple(iri("django"), "visited", iri("Amsterdam")),
        triple(iri("django"), "visited", iri("Berlin")),
        triple(iri("django"), "visited", iri("Paris")),
        triple(iri("shosanna"), "visited", iri("Paris")),
        triple(iri("mia"), "visited", iri("Paris")),
    ]);
    triples
}

/// The statistics as the string collector counted them.
#[derive(Debug, PartialEq, Eq)]
struct Counts {
    total: u64,
    subjects: u64,
    predicates: u64,
    objects: u64,
    /// Per predicate name: its triples, distinct subjects, distinct objects.
    per_predicate: BTreeMap<String, (u64, u64, u64)>,
}

fn oracle(triples: &[Triple]) -> Counts {
    let mut subjects = BTreeSet::new();
    let mut objects = BTreeSet::new();
    let mut per_predicate: BTreeMap<String, (u64, BTreeSet<String>, BTreeSet<String>)> =
        BTreeMap::new();
    for triple in triples {
        let subject = triple.subject().to_string();
        let object = triple.object().to_string();
        subjects.insert(subject.clone());
        objects.insert(object.clone());
        let entry = per_predicate
            .entry(triple.predicate().to_string())
            .or_default();
        entry.0 += 1;
        entry.1.insert(subject);
        entry.2.insert(object);
    }
    Counts {
        total: triples.len() as u64,
        subjects: subjects.len() as u64,
        predicates: per_predicate.len() as u64,
        objects: objects.len() as u64,
        per_predicate: per_predicate
            .into_iter()
            .map(|(name, (count, subjects, objects))| {
                (name, (count, subjects.len() as u64, objects.len() as u64))
            })
            .collect(),
    }
}

fn counts(statistics: &RdfStatistics) -> Counts {
    Counts {
        total: statistics.total_triples,
        subjects: statistics.subject_count,
        predicates: statistics.predicate_count,
        objects: statistics.object_count,
        per_predicate: statistics
            .predicates
            .iter()
            .map(|(name, counted)| {
                (
                    name.clone(),
                    (
                        counted.triple_count,
                        counted.distinct_subjects,
                        counted.distinct_objects,
                    ),
                )
            })
            .collect(),
    }
}

/// The store's triples, as it holds them.
fn held(store: &RdfStore) -> Vec<Triple> {
    store.triples().iter().map(|t| t.as_ref().clone()).collect()
}

#[test]
fn collected_statistics_count_as_the_string_collector() {
    let store = RdfStore::new();
    let triples = mixed();
    store.batch_insert(triples.clone());
    let got = counts(&store.collect_statistics());
    assert_eq!(got, oracle(&triples));
    // The cases above that the totals must tell apart.
    assert_eq!(got.per_predicate[&format!("<{EX}label>")], (4, 4, 4));
    assert_eq!(got.per_predicate[&format!("<{EX}seeAlso>")], (2, 1, 2));
    assert_eq!(got.per_predicate[&format!("<{EX}visited>")], (5, 3, 3));
    assert_eq!(got.per_predicate[&format!("<{EX}knows>")], (3, 3, 2));
}

#[test]
fn predicate_flags_follow_the_counts() {
    let store = RdfStore::new();
    store.batch_insert(mixed());
    let statistics = store.collect_statistics();
    // name: one per person, each name its own: functional both ways.
    let name = statistics.get_predicate(&format!("<{EX}name>")).unwrap();
    assert!(name.is_functional && name.is_inverse_functional, "{name:?}");
    // visited: django visits three cities, Paris has three visitors.
    let visited = statistics.get_predicate(&format!("<{EX}visited>")).unwrap();
    assert!(
        !visited.is_functional && !visited.is_inverse_functional,
        "{visited:?}"
    );
}

#[test]
fn bulk_loaded_statistics_match_a_collection_and_are_cached() {
    let store = RdfStore::new();
    let triples = mixed();
    let loaded = store.bulk_load(triples.clone());
    assert_eq!(loaded.triple_count, triples.len());
    assert_eq!(counts(&loaded.statistics), oracle(&triples));
    let cached = store
        .cached_statistics()
        .expect("a bulk load caches the statistics it counted");
    assert_eq!(counts(&cached), oracle(&triples));
    assert_eq!(counts(&store.collect_statistics()), oracle(&triples));
}

#[test]
fn named_graphs_count_on_their_own() {
    let store = RdfStore::new();
    let triples = mixed();
    store.batch_insert(triples.clone());
    let shared = triple(iri("alix"), "visited", iri("Prague"));
    store.insert(shared.clone());
    let first = store.graph_or_create(&format!("{EX}g1"));
    let second = store.graph_or_create(&format!("{EX}g2"));
    first.batch_insert(triples[..6].to_vec());
    first.insert(shared.clone());
    second.insert(shared.clone());
    second.insert(triple(Term::blank("b9"), "visited", iri("Barcelona")));

    for (graph, store) in [("default", &store), ("g1", &first), ("g2", &second)] {
        let held = held(store);
        let got = counts(&store.collect_statistics());
        assert_eq!(got, oracle(&held), "{graph}");
        assert_eq!(got.total, store.len() as u64, "{graph}");
    }
    // The default graph's statistics hold none of the named graphs' triples.
    assert_eq!(
        store.collect_statistics().total_triples,
        triples.len() as u64 + 1
    );
    assert_eq!(second.collect_statistics().total_triples, 2);
}

#[test]
fn statistics_after_removals_count_what_is_left() {
    let store = RdfStore::new();
    let triples = mixed();
    store.batch_insert(triples.clone());
    assert_eq!(
        store.get_or_collect_statistics().total_triples,
        triples.len() as u64
    );
    // Remove every label and one of django's visits.
    let removed: Vec<Triple> = triples
        .iter()
        .filter(|t| {
            t.predicate() == &iri("label")
                || (t.subject() == &iri("django") && t.object() == &iri("Berlin"))
        })
        .cloned()
        .collect();
    assert_eq!(removed.len(), 5);
    for triple in &removed {
        assert!(store.remove(triple), "{triple}");
    }
    let left: Vec<Triple> = triples
        .into_iter()
        .filter(|t| !removed.contains(t))
        .collect();
    let got = counts(&store.collect_statistics());
    assert_eq!(got, oracle(&left));
    assert!(!got.per_predicate.contains_key(&format!("<{EX}label>")));
    assert_eq!(got.per_predicate[&format!("<{EX}visited>")], (4, 3, 2));
}

#[test]
fn an_empty_store_counts_nothing() {
    let store = RdfStore::new();
    assert_eq!(counts(&store.collect_statistics()), oracle(&[]));
    assert!(store.collect_statistics().predicates.is_empty());
}
