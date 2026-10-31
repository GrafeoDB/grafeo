//! When SPARQL collects the RDF statistics (0.6.1): only for a plan that joins
//! inputs, whose order the statistics choose. A load through `INSERT DATA`
//! statements, `DELETE DATA` and a query without a join read no statistics,
//! so a load through statements does not read the whole store again every
//! tenth of its growth. Collection is seen through
//! `RdfStore::cached_statistics`, which never collects: `None` until the first
//! collection, and the same `Arc` until the next one.
//!
//! - a statement load collects nothing; the first join collects once, the
//!   next joins reuse it;
//! - each kind of join collects (two patterns, OPTIONAL, MINUS, VALUES with a
//!   pattern, a sequence path); a single pattern, a filter, UNION, an
//!   alternative path, COUNT, ASK, DELETE DATA and DELETE WHERE of one
//!   pattern do not;
//! - after a collection, a join collects again only past the drift;
//! - a join is planned the same after a statement load as after a bulk load.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_statistics_collection
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::fmt::Write;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_core::graph::rdf::{Term, Triple};
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::{QueryLanguage, QueryProcessor};

const PREFIX: &str = "PREFIX e: <http://ex.org/>";
const PEOPLE: usize = 1_000;

/// The triples of person `i`: a name, an age and whom they know.
fn person(i: usize) -> [String; 3] {
    [
        format!("<http://ex.org/p{i}> <http://ex.org/name> \"Person {i}\""),
        format!("<http://ex.org/p{i}> <http://ex.org/age> {}", 20 + i % 50),
        format!(
            "<http://ex.org/p{i}> <http://ex.org/knows> <http://ex.org/p{}>",
            (i + 1) % PEOPLE
        ),
    ]
}

/// The people loaded through `INSERT DATA` statements of 100 people each.
fn loaded_by_statements() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for start in (0..PEOPLE).step_by(100) {
        let mut data = String::from("INSERT DATA {\n");
        for i in start..start + 100 {
            for triple in person(i) {
                writeln!(data, "{triple} .").unwrap();
            }
        }
        data.push('}');
        db.execute_sparql(&data).unwrap();
    }
    db
}

/// The same people, loaded in one bulk write.
fn loaded_in_bulk() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let iri = |text: String| Term::iri(text);
    let triples = (0..PEOPLE).flat_map(|i| {
        let subject = iri(format!("http://ex.org/p{i}"));
        [
            Triple::new(
                subject.clone(),
                iri("http://ex.org/name".into()),
                Term::literal(format!("Person {i}")),
            ),
            Triple::new(
                subject.clone(),
                iri("http://ex.org/age".into()),
                Term::typed_literal(
                    (20 + i % 50).to_string(),
                    "http://www.w3.org/2001/XMLSchema#integer",
                ),
            ),
            Triple::new(
                subject,
                iri("http://ex.org/knows".into()),
                iri(format!("http://ex.org/p{}", (i + 1) % PEOPLE)),
            ),
        ]
    });
    db.batch_insert_rdf(triples).unwrap();
    db
}

/// A term's text: an IRI's local name, a literal's lexical form, `-` unbound.
fn text(value: &Value) -> String {
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            Term::Iri(iri) => iri
                .as_str()
                .strip_prefix("http://ex.org/")
                .unwrap_or(iri.as_str())
                .to_string(),
            Term::Literal(literal) => literal.value().to_string(),
            other => format!("{other:?}"),
        },
        Value::String(text) => text.to_string(),
        Value::Null => "-".to_string(),
        other => format!("{other:?}"),
    }
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .execute_sparql(&format!("{PREFIX} {query}"))
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect::<Vec<_>>().join(" "))
        .collect();
    rows.sort();
    rows
}

fn cached(db: &GrafeoDB) -> Option<Arc<grafeo_core::statistics::RdfStatistics>> {
    db.rdf_store().cached_statistics()
}

/// The plan of `query`, without running it.
fn explain(db: &GrafeoDB, query: &str) -> String {
    let processor = QueryProcessor::with_rdf(db.store().unwrap(), Arc::clone(db.rdf_store()));
    let result = processor
        .process(
            &format!("EXPLAIN {PREFIX} {query}"),
            QueryLanguage::Sparql,
            None,
        )
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{other:?}"),
    }
}

const JOIN: &str = "SELECT ?n WHERE { e:p1 e:knows ?f . ?f e:name ?n }";

#[test]
fn a_statement_load_collects_nothing_until_the_first_join() {
    let db = loaded_by_statements();
    assert!(
        cached(&db).is_none(),
        "ten INSERT DATA statements collected the statistics"
    );
    assert_eq!(rows(&db, JOIN), ["Person 2"]);
    let first = cached(&db).expect("a join collects the statistics");
    assert_eq!(first.total_triples, 3 * PEOPLE as u64);
    assert_eq!(rows(&db, JOIN), ["Person 2"]);
    assert_eq!(
        rows(&db, "SELECT ?a WHERE { e:p2 e:knows ?f . ?f e:age ?a }"),
        ["23"]
    );
    assert!(
        Arc::ptr_eq(&first, &cached(&db).unwrap()),
        "the next joins collected again"
    );
}

#[test]
fn each_kind_of_join_collects() {
    for (kind, query, want) in [
        ("two patterns", JOIN, vec!["Person 2"]),
        (
            "OPTIONAL",
            "SELECT ?n ?x WHERE { e:p1 e:name ?n OPTIONAL { e:p1 e:nick ?x } }",
            vec!["Person 1 -"],
        ),
        (
            "MINUS",
            "SELECT ?f WHERE { e:p1 e:knows ?f MINUS { ?f e:age 22 } }",
            vec![],
        ),
        (
            "VALUES with a pattern",
            "SELECT ?n WHERE { VALUES ?p { e:p3 e:p4 } ?p e:name ?n }",
            vec!["Person 3", "Person 4"],
        ),
        (
            "a sequence path, two patterns joined",
            "SELECT ?f WHERE { e:p1 e:knows/e:knows ?f }",
            vec!["p3"],
        ),
    ] {
        let db = loaded_by_statements();
        assert_eq!(rows(&db, query), want, "{kind}");
        assert!(cached(&db).is_some(), "{kind} did not collect");
    }
}

#[test]
fn a_plan_without_a_join_collects_nothing() {
    for (kind, query, want) in [
        (
            "one pattern",
            "SELECT ?n WHERE { e:p7 e:name ?n }",
            vec!["Person 7"],
        ),
        (
            "a filter on one pattern",
            "SELECT ?p WHERE { ?p e:age ?a FILTER(?a = 20 && ?p = e:p50) }",
            vec!["p50"],
        ),
        (
            "ORDER BY and LIMIT",
            "SELECT ?a WHERE { ?p e:age ?a } ORDER BY DESC(?a) LIMIT 1",
            vec!["69"],
        ),
        (
            "UNION",
            "SELECT ?x WHERE { { e:p1 e:name ?x } UNION { e:p2 e:name ?x } }",
            vec!["Person 1", "Person 2"],
        ),
        (
            "an alternative path",
            "SELECT ?x WHERE { e:p1 (e:knows|e:name) ?x }",
            vec!["Person 1", "p2"],
        ),
        (
            "COUNT of one pattern",
            "SELECT (COUNT(*) AS ?c) WHERE { ?p e:knows ?f }",
            vec!["1000"],
        ),
        ("ASK", "ASK { e:p1 e:knows e:p2 }", vec!["Bool(true)"]),
    ] {
        let db = loaded_by_statements();
        let got = rows(&db, query);
        let got: Vec<&str> = got.iter().map(String::as_str).collect();
        assert_eq!(got, want, "{kind}");
        assert!(cached(&db).is_none(), "{kind} collected the statistics");
    }
}

#[test]
fn deletes_without_a_join_collect_nothing() {
    let db = loaded_by_statements();
    db.execute_sparql(&format!(
        "{PREFIX} DELETE DATA {{ e:p1 e:name \"Person 1\" }}"
    ))
    .unwrap();
    db.execute_sparql(&format!("{PREFIX} DELETE WHERE {{ e:p2 ?p ?o }}"))
        .unwrap();
    assert!(cached(&db).is_none(), "a delete collected the statistics");
    assert_eq!(
        rows(&db, "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }"),
        ["2996"]
    );
}

#[test]
fn a_join_collects_again_only_past_the_drift() {
    let db = loaded_by_statements();
    rows(&db, JOIN);
    let first = cached(&db).unwrap();
    // 200 changes: below a tenth of 3,000.
    let mut data = String::from("INSERT DATA {\n");
    for i in 0..200 {
        writeln!(
            data,
            "<http://ex.org/x{i}> <http://ex.org/name> \"X {i}\" ."
        )
        .unwrap();
    }
    data.push('}');
    db.execute_sparql(&data).unwrap();
    rows(&db, JOIN);
    assert!(
        Arc::ptr_eq(&first, &cached(&db).unwrap()),
        "200 changes collected again"
    );
    // 200 more: 400 in all, past a tenth of 3,000.
    db.execute_sparql(&data.replace("/x", "/y")).unwrap();
    assert!(
        Arc::ptr_eq(&first, &cached(&db).unwrap()),
        "an INSERT DATA collected again"
    );
    assert_eq!(rows(&db, JOIN), ["Person 2"]);
    let second = cached(&db).unwrap();
    assert!(!Arc::ptr_eq(&first, &second), "400 changes did not collect");
    assert_eq!(second.total_triples, 3 * PEOPLE as u64 + 400);
}

#[test]
fn a_join_is_planned_the_same_after_a_statement_load_as_after_a_bulk_load() {
    let by_statements = loaded_by_statements();
    let in_bulk = loaded_in_bulk();
    for query in [
        JOIN,
        "SELECT ?n WHERE { ?p e:age 21 . ?p e:knows ?f . ?f e:name ?n }",
        "SELECT ?n ?a WHERE { ?f e:name ?n . e:p9 e:knows ?f . ?f e:age ?a }",
        "SELECT ?p WHERE { ?p e:age 30 OPTIONAL { ?p e:knows ?f . ?f e:age 31 } }",
    ] {
        assert_eq!(
            rows(&by_statements, query),
            rows(&in_bulk, query),
            "{query}"
        );
        assert_eq!(
            explain(&by_statements, query),
            explain(&in_bulk, query),
            "{query}"
        );
    }
}
