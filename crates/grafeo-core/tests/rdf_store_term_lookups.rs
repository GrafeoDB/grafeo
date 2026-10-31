//! Every lookup of the RDF store, by term, against a brute-force oracle over
//! the triples it holds (#659: the triples and the six indexes keyed by term
//! ids, lookups by term through the dictionary). The terms mix IRIs, blank
//! nodes, short and long literals with the same text in different kinds
//! (inline and dictionary terms), and literals that are equal as values but
//! not as terms ("01" and "1" as integers, "1.0" and "1" as decimals, "1" as a
//! string and as a language string). Then terms the store never had, removes,
//! named graphs, triples from another store, and a section save and reload.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test rdf_store_term_lookups
//! ```

#![cfg(feature = "triple-store")]

use std::collections::BTreeSet;
use std::sync::Arc;

use grafeo_common::storage::{ImageSource, MemoryImage, Section, SectionType};
use grafeo_core::graph::rdf::{RdfStore, RdfStoreSection, Term, Triple, TriplePattern};

const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

fn iri(name: &str) -> Term {
    Term::iri(format!("http://ex.org/{name}"))
}

fn typed(text: &str, kind: &str) -> Term {
    Term::typed_literal(text, format!("{XSD}{kind}"))
}

/// A long text, past any inline limit.
fn long(text: &str) -> String {
    format!("{text} {}", "and so on ".repeat(8))
}

/// Subjects, predicates and objects of every kind, and terms that look alike.
fn mixed() -> Vec<Triple> {
    let subjects = [
        iri("alix"),
        iri("gus"),
        iri("1"),
        Term::blank("b1"),
        Term::blank("b2"),
        Term::iri(format!("http://ex.org/{}", long("vincent"))),
    ];
    let predicates = [iri("name"), iri("value"), iri("knows")];
    let objects = [
        Term::literal("1"),
        Term::lang_literal("1", "en"),
        Term::lang_literal("1", "nl"),
        typed("1", "integer"),
        typed("01", "integer"),
        typed("1", "decimal"),
        typed("1.0", "decimal"),
        typed("1", "double"),
        typed("true", "boolean"),
        Term::literal("http://ex.org/alix"),
        Term::literal(long("Alix")),
        Term::literal(long("Alix") + "!"),
        iri("alix"),
        iri("1"),
        Term::blank("b1"),
    ];
    let mut triples = Vec::new();
    for (i, subject) in subjects.iter().enumerate() {
        for (j, predicate) in predicates.iter().enumerate() {
            for (k, object) in objects.iter().enumerate() {
                // About half of the combinations, so lookups miss too.
                if (i + 2 * j + 3 * k) % 2 == 0 {
                    triples.push(Triple::new(
                        subject.clone(),
                        predicate.clone(),
                        object.clone(),
                    ));
                }
            }
        }
    }
    triples
}

/// Terms the store never had, of each kind.
fn absent() -> Vec<Term> {
    vec![
        iri("nobody"),
        Term::blank("b9"),
        Term::literal("absent"),
        typed("2", "integer"),
        typed("001", "integer"),
        Term::lang_literal("1", "fr"),
        Term::literal(long("Nobody")),
    ]
}

fn names(triples: &[Arc<Triple>]) -> BTreeSet<String> {
    triples.iter().map(ToString::to_string).collect()
}

/// The triples of `held` that `pattern` matches.
fn brute(held: &[Triple], pattern: &TriplePattern) -> BTreeSet<String> {
    held.iter()
        .filter(|t| pattern.subject.as_ref().is_none_or(|s| t.subject() == s))
        .filter(|t| {
            pattern
                .predicate
                .as_ref()
                .is_none_or(|p| t.predicate() == p)
        })
        .filter(|t| pattern.object.as_ref().is_none_or(|o| t.object() == o))
        .map(ToString::to_string)
        .collect()
}

fn pattern(
    subject: Option<&Term>,
    predicate: Option<&Term>,
    object: Option<&Term>,
) -> TriplePattern {
    TriplePattern {
        subject: subject.cloned(),
        predicate: predicate.cloned(),
        object: object.cloned(),
    }
}

/// Every lookup of `store` agrees with the brute force over `held`: the
/// eight pattern shapes of each held triple and of absent terms, the lookups
/// by one term, the predicate counts, `contains`, and the distinct terms.
fn agrees(store: &RdfStore, held: &[Triple], context: &str) {
    assert_eq!(store.len(), held.len(), "{context}: len");
    assert_eq!(
        names(&store.triples()),
        held.iter().map(ToString::to_string).collect(),
        "{context}: triples"
    );
    let missing = absent();
    let mut terms: Vec<Term> = held
        .iter()
        .flat_map(|t| {
            [
                t.subject().clone(),
                t.predicate().clone(),
                t.object().clone(),
            ]
        })
        .collect();
    terms.extend(missing.iter().cloned());
    terms.sort_by_key(ToString::to_string);
    terms.dedup();

    for triple in held {
        let (s, p, o) = (triple.subject(), triple.predicate(), triple.object());
        for (bs, bp, bo) in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
            (true, true, false),
            (true, false, true),
            (false, true, true),
            (true, true, true),
        ] {
            let query = pattern(bs.then_some(s), bp.then_some(p), bo.then_some(o));
            assert_eq!(
                names(&store.find(&query)),
                brute(held, &query),
                "{context}: {query:?}"
            );
        }
        assert!(store.contains(triple), "{context}: contains {triple}");
    }
    assert_eq!(
        names(&store.find(&pattern(None, None, None))),
        brute(held, &pattern(None, None, None)),
        "{context}: all"
    );
    for term in &terms {
        let by = |s, p, o| brute(held, &pattern(s, p, o));
        assert_eq!(
            names(&store.triples_with_subject(term)),
            by(Some(term), None, None),
            "{context}: s {term}"
        );
        assert_eq!(
            names(&store.triples_with_predicate(term)),
            by(None, Some(term), None),
            "{context}: p {term}"
        );
        assert_eq!(
            names(&store.triples_with_object(term)),
            by(None, None, Some(term)),
            "{context}: o {term}"
        );
        assert_eq!(
            store.count_with_predicate(term),
            by(None, Some(term), None).len(),
            "{context}: count {term}"
        );
        // Absent terms in each position, alone and with held ones.
        for other in &missing {
            for query in [
                pattern(Some(term), None, Some(other)),
                pattern(Some(other), Some(term), None),
                pattern(None, Some(other), Some(term)),
            ] {
                assert_eq!(store.find(&query).len(), 0, "{context}: {query:?}");
            }
        }
    }
    for other in &missing {
        let triple = Triple::new(iri("alix"), iri("name"), other.clone());
        assert!(!store.contains(&triple), "{context}: contains {triple}");
    }
    let distinct = |pick: fn(&Triple) -> &Term| -> BTreeSet<String> {
        held.iter().map(|t| pick(t).to_string()).collect()
    };
    let listed =
        |terms: Vec<Term>| -> BTreeSet<String> { terms.iter().map(ToString::to_string).collect() };
    assert_eq!(
        listed(store.subjects()),
        distinct(Triple::subject),
        "{context}: subjects"
    );
    assert_eq!(
        listed(store.predicates()),
        distinct(Triple::predicate),
        "{context}: predicates"
    );
    assert_eq!(
        listed(store.objects()),
        distinct(Triple::object),
        "{context}: objects"
    );
}

fn filled(triples: &[Triple]) -> RdfStore {
    let store = RdfStore::new();
    store.batch_insert(triples.to_vec());
    store
}

#[test]
fn every_lookup_agrees_with_brute_force() {
    let held = mixed();
    assert!(held.len() > 100, "{}", held.len());
    agrees(&filled(&held), &held, "batch");
    let one_by_one = RdfStore::new();
    for triple in &held {
        assert!(one_by_one.insert(triple.clone()));
    }
    agrees(&one_by_one, &held, "one by one");
    let bulk = RdfStore::new();
    bulk.bulk_load(held.clone());
    agrees(&bulk, &held, "bulk load");
}

#[test]
fn literals_equal_as_values_are_different_terms() {
    // RDF term equality is by lexical form and datatype: each pair is two
    // terms, two triples, and a lookup by one finds only its own.
    let store = RdfStore::new();
    let pairs = [
        (typed("1", "integer"), typed("01", "integer")),
        (typed("1", "decimal"), typed("1.0", "decimal")),
        (typed("1", "integer"), typed("1", "decimal")),
        (Term::literal("1"), Term::lang_literal("1", "en")),
        (Term::literal("1"), typed("1", "integer")),
        (iri("1"), Term::literal("http://ex.org/1")),
    ];
    for (i, (a, b)) in pairs.iter().enumerate() {
        let subject = iri(&format!("pair{i}"));
        let first = Triple::new(subject.clone(), iri("value"), a.clone());
        let second = Triple::new(subject.clone(), iri("value"), b.clone());
        assert!(store.insert(first.clone()), "{first}");
        assert!(store.insert(second.clone()), "{second}");
        for (term, own) in [(a, &first), (b, &second)] {
            let found = store.find(&pattern(Some(&subject), None, Some(term)));
            assert_eq!(names(&found), BTreeSet::from([own.to_string()]), "{term}");
        }
    }
    assert_eq!(store.len(), 2 * pairs.len());
    // A plain literal is the xsd:string literal of the same text.
    let plain = Triple::new(iri("s"), iri("value"), Term::literal("x"));
    let string = Triple::new(iri("s"), iri("value"), typed("x", "string"));
    assert!(store.insert(plain));
    assert!(!store.insert(string.clone()), "the same term");
    assert!(store.contains(&string));
}

#[test]
fn removing_and_looking_up_terms_the_store_never_had() {
    let held = mixed();
    let store = filled(&held);
    for term in absent() {
        let triple = Triple::new(iri("alix"), iri("name"), term.clone());
        assert!(!store.remove(&triple), "{triple}");
        assert_eq!(store.triples_with_subject(&term).len(), 0, "{term}");
        assert_eq!(store.triples_with_object(&term).len(), 0, "{term}");
        assert_eq!(store.count_with_predicate(&term), 0);
    }
    agrees(&store, &held, "after absent removes");
}

#[test]
fn removes_keep_every_lookup_right() {
    let mut held = mixed();
    let store = filled(&held);
    // Remove every third triple, in an order unlike the insert order.
    let removed: Vec<Triple> = held.iter().rev().step_by(3).cloned().collect();
    for triple in &removed {
        assert!(store.remove(triple), "{triple}");
        assert!(!store.remove(triple), "twice {triple}");
    }
    held.retain(|t| !removed.contains(t));
    agrees(&store, &held, "after removes");
    // Put them back: the terms come back as they were.
    store.batch_insert(removed.clone());
    held.extend(removed);
    agrees(&store, &held, "after putting back");
}

#[test]
fn named_graphs_share_terms_and_keep_their_triples_apart() {
    let held = mixed();
    let store = filled(&held);
    let half: Vec<Triple> = held.iter().step_by(2).cloned().collect();
    let other: Vec<Triple> = held.iter().skip(1).step_by(3).cloned().collect();
    let first = store.graph_or_create("http://ex.org/g1");
    let second = store.graph_or_create("http://ex.org/g2");
    first.batch_insert(half.clone());
    second.batch_insert(other.clone());
    agrees(&store, &held, "default");
    agrees(&first, &half, "g1");
    agrees(&second, &other, "g2");

    // A remove in one graph leaves the others alone.
    let gone = half[0].clone();
    assert!(store.remove_from(Some("http://ex.org/g1"), &gone));
    let half_left: Vec<Triple> = half[1..].to_vec();
    agrees(&first, &half_left, "g1 after remove");
    agrees(&store, &held, "default after a g1 remove");
    assert_eq!(second.contains(&gone), other.contains(&gone));
}

#[test]
fn a_triple_from_another_store_is_found_by_its_terms() {
    // The second store sees other terms first, so its ids for the shared
    // terms differ from the first store's.
    let held = mixed();
    let first = filled(&held);
    let second = RdfStore::new();
    let extra: Vec<Triple> = (0..50)
        .map(|i| {
            Triple::new(
                iri(&format!("x{i}")),
                iri("value"),
                typed(&(100 + i).to_string(), "integer"),
            )
        })
        .collect();
    second.batch_insert(extra.clone());
    let reversed: Vec<Triple> = held.iter().rev().cloned().collect();
    second.batch_insert(reversed);

    let from_first = first.triples();
    for triple in &from_first {
        assert!(second.contains(triple), "{triple}");
        let found = second.find(&pattern(
            Some(triple.subject()),
            Some(triple.predicate()),
            Some(triple.object()),
        ));
        assert_eq!(names(&found), BTreeSet::from([triple.to_string()]));
    }
    for triple in second.triples() {
        assert_eq!(
            first.contains(&triple),
            !extra.contains(&triple),
            "{triple}"
        );
    }
    // Removing the first store's triples from the second leaves its own.
    for triple in &from_first {
        assert!(second.remove(triple), "{triple}");
    }
    agrees(&second, &extra, "after removing the other store's triples");
    agrees(&first, &held, "the first store, untouched");
}

#[test]
fn a_saved_and_reloaded_store_answers_the_same() {
    let held = mixed();
    let store = Arc::new(filled(&held));
    let named: Vec<Triple> = held.iter().step_by(4).cloned().collect();
    store
        .graph_or_create("http://ex.org/g1")
        .batch_insert(named.clone());

    // The whole-section bytes, and the chunked form the file writes.
    let bytes = RdfStoreSection::new(Arc::clone(&store))
        .serialize()
        .unwrap();
    let from_bytes = Arc::new(RdfStore::new());
    RdfStoreSection::new(Arc::clone(&from_bytes))
        .deserialize(&bytes)
        .unwrap();

    let section = RdfStoreSection::new(Arc::clone(&store));
    let image = MemoryImage::from_sections(&[&section as &dyn Section]).unwrap();
    let source = image.section_source(SectionType::RdfStore).unwrap();
    let from_chunks = Arc::new(RdfStore::new());
    RdfStoreSection::new(Arc::clone(&from_chunks))
        .read_from(&*source)
        .unwrap();

    for (back, how) in [(&from_bytes, "bytes"), (&from_chunks, "chunks")] {
        agrees(back, &held, how);
        let graph = back
            .graph("http://ex.org/g1")
            .expect("the named graph is back");
        agrees(&graph, &named, how);
        // The reloaded store takes new triples over its known terms.
        let more = Triple::new(iri("alix"), iri("knows"), typed("01", "integer"));
        let mut grown = held.clone();
        if !grown.contains(&more) {
            assert!(back.insert(more.clone()));
            grown.push(more);
        }
        agrees(back, &grown, &format!("{how}, then an insert"));
    }
}
