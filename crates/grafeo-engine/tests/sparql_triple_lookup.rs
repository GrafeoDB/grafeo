//! A SPARQL triple pattern whose subject, predicate or object the rows before
//! it always bind looks its triples up per row through the store's indexes
//! (0.6.1, LDBC IS1 on RDF); past a limit that grows with the pattern's
//! estimate (at most 1,024 rows) it joins as a hash join does. Its rows must be
//! those of the plain join, which the same query
//! with the pattern in a sub-select gives (a sub-select is not looked up);
//! each case compares the two, and the main ones check their rows:
//!
//! - the bound subject, object or predicate, and two bound positions;
//! - VALUES of 1 row (looked up) and of 1,024, 1,025 and 5,000 rows (past
//!   the limit of a pattern of 2,000 triples), with repeated rows;
//! - a term a BIND computes (an IRI and a literal), a term no triple has,
//!   a term of another datatype or language;
//! - two triple patterns (the lookup side chosen by the estimates);
//! - an OPTIONAL left side that holds the free variable (the plain join);
//! - a constant GRAPH, FROM over two graphs that share a triple (its
//!   duplicates as before), GRAPH ?g (the plain join);
//! - a transaction's pending inserts and deletes, in both modes;
//! - PROFILE: the lookup operator and its rows.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_triple_lookup
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::fmt::Write;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::{QueryLanguage, QueryProcessor};
use grafeo_engine::{Config, GrafeoDB, GraphModel};

const PREFIX: &str = "PREFIX e: <http://example.org/>";

/// People p0 to p1999: `e:name "n{i}"`, `e:age {i % 50}`, `e:knows p{i+1}`
/// (p1999 none); p1 also has `e:name "Alix"@nl`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut data = String::new();
    for i in 0..2_000 {
        write!(data, "e:p{i} e:name \"n{i}\" ; e:age {} ", i % 50).unwrap();
        if i < 1_999 {
            write!(data, "; e:knows e:p{} ", i + 1).unwrap();
        }
        data.push_str(". ");
    }
    data.push_str("e:p1 e:name \"Alix\"@nl . ");
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ {data} }}"))
        .unwrap();
    db
}

/// A term's text: an IRI's local name, a literal's lexical form (with its
/// language), `-` unbound.
fn text(value: &Value) -> String {
    use grafeo_common::types::rdf::Term;
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            Term::Iri(iri) => iri
                .as_str()
                .strip_prefix("http://example.org/")
                .unwrap_or(iri.as_str())
                .to_string(),
            Term::Literal(literal) => match literal.language() {
                Some(language) => format!("{}@{language}", literal.value()),
                None => literal.value().to_string(),
            },
            other => format!("{other:?}"),
        },
        Value::Null => "-".to_string(),
        other => format!("{other:?}"),
    }
}

fn sorted(result: &QueryResult) -> Vec<String> {
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect::<Vec<_>>().join(" "))
        .collect();
    rows.sort();
    rows
}

type Execute<'a> = &'a dyn Fn(&str) -> Result<QueryResult>;

fn run(execute: Execute<'_>, query: &str) -> std::result::Result<Vec<String>, String> {
    execute(&format!("{PREFIX} {query}"))
        .map(|result| sorted(&result))
        .map_err(|error| error.to_string())
}

/// `SELECT {projection} {dataset} WHERE { {left} {pattern} }` against the
/// same with the pattern in a sub-select over its own variables; they must
/// agree. Returns the rows.
fn agree_with(
    execute: Execute<'_>,
    projection: &str,
    dataset: &str,
    left: &str,
    pattern: &str,
    pattern_variables: &str,
) -> Vec<String> {
    let looked_up = format!("SELECT {projection} {dataset} WHERE {{ {left} {pattern} }}");
    let joined = format!(
        "SELECT {projection} {dataset} WHERE {{ {left} {{ SELECT {pattern_variables} WHERE {{ {pattern} }} }} }}"
    );
    let got = run(execute, &looked_up);
    assert_eq!(got, run(execute, &joined), "`{looked_up}` against the join");
    got.unwrap_or_else(|error| panic!("`{looked_up}`: {error}"))
}

fn agree(
    db: &GrafeoDB,
    projection: &str,
    left: &str,
    pattern: &str,
    variables: &str,
) -> Vec<String> {
    agree_with(
        &|q| db.execute_sparql(q),
        projection,
        "",
        left,
        pattern,
        variables,
    )
}

/// The profile text of `query` on the committed store.
fn profile(db: &GrafeoDB, query: &str) -> String {
    let processor = QueryProcessor::with_rdf(db.store().unwrap(), Arc::clone(db.rdf_store()));
    let result = processor
        .process(
            &format!("EXPLAIN ANALYZE {PREFIX} {query}"),
            QueryLanguage::Sparql,
            None,
        )
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{other:?}"),
    }
}

/// `VALUES ?p { ... }` of `count` rows: p0, p1, ... wrapping at 2,000, so
/// past 2,000 rows repeat.
fn values_of_people(count: usize) -> String {
    let mut values = String::from("VALUES ?p { ");
    for i in 0..count {
        write!(values, "e:p{} ", i % 2_000).unwrap();
    }
    values.push('}');
    values
}

#[test]
fn each_bound_position() {
    let db = database();
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "VALUES ?p { e:p1 e:p7 }",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p1 Alix@nl", "p1 n1", "p7 n7"],
        "the subject"
    );
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "VALUES ?n { \"n5\" \"n9\" }",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p5 n5", "p9 n9"],
        "the object"
    );
    assert_eq!(
        agree(
            &db,
            "?r ?o",
            "VALUES ?r { e:name e:age }",
            "e:p3 ?r ?o .",
            "?r ?o"
        ),
        ["age 3", "name n3"],
        "the predicate"
    );
    let all_ages = agree(
        &db,
        "?s ?o",
        "VALUES ?r { e:age }",
        "?s ?r ?o .",
        "?s ?r ?o",
    );
    assert_eq!(all_ages.len(), 2_000, "the predicate alone");
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "VALUES (?p ?n) { (e:p1 \"n1\") (e:p1 \"n2\") (e:p2 \"n2\") }",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p1 n1", "p2 n2"],
        "subject and object"
    );
    assert_eq!(
        agree(
            &db,
            "?p ?q",
            "VALUES (?p ?r) { (e:p4 e:knows) (e:p1999 e:knows) }",
            "?p ?r ?q .",
            "?p ?r ?q"
        ),
        ["p4 p5"],
        "subject and predicate"
    );
}

/// 1 row looks up; 1,024, 1,025 and 5,000 rows are past the limit of a
/// pattern of 2,000 triples (and 1,025 past the most any pattern looks up).
/// Each row joins: repeated rows give their rows again.
#[test]
fn values_on_both_sides_of_the_switch() {
    let db = database();
    for count in [1, 1_024, 1_025, 5_000] {
        let rows = agree(
            &db,
            "?p ?a",
            &values_of_people(count),
            "?p e:age ?a .",
            "?p ?a",
        );
        assert_eq!(rows.len(), count, "{count} rows, one age each");
        let names = agree(
            &db,
            "?p ?n",
            &values_of_people(count),
            "?p e:name ?n .",
            "?p ?n",
        );
        let with_alix = (0..count).filter(|i| i % 2_000 == 1).count();
        assert_eq!(names.len(), count + with_alix, "{count}: p1 has two names");
    }
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "VALUES ?p { e:p7 e:p7 e:p7 }",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p7 n7", "p7 n7", "p7 n7"]
    );
}

/// A term a BIND computes, a term no triple has, and literals that differ
/// from a stored one only by language or datatype.
#[test]
fn computed_unknown_and_near_terms() {
    let db = database();
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "BIND(IRI(CONCAT(STR(e:), \"p\", \"7\")) AS ?p)",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p7 n7"],
        "a computed IRI"
    );
    assert_eq!(
        agree(
            &db,
            "?p ?n",
            "BIND(CONCAT(\"n\", \"9\") AS ?n)",
            "?p e:name ?n .",
            "?p ?n"
        ),
        ["p9 n9"],
        "a computed literal"
    );
    for left in [
        "VALUES ?p { e:nobody }",
        "VALUES ?n { \"zzz\" }",
        "VALUES ?n { \"n5\"@en }",
        "VALUES ?n { \"Alix\" }",
    ] {
        assert_eq!(
            agree(&db, "?p ?n", left, "?p e:name ?n .", "?p ?n"),
            Vec::<String>::new(),
            "{left}"
        );
    }
    for left in [
        "VALUES ?a { \"7\" }",
        "VALUES ?a { 7.0 }",
        "VALUES ?a { \"07\"^^<http://www.w3.org/2001/XMLSchema#integer> }",
    ] {
        agree(&db, "?p ?a", left, "?p e:age ?a .", "?p ?a");
    }
}

/// Two triple patterns: one is looked up with the rows of the other, both
/// orders written.
#[test]
fn two_triple_patterns() {
    let db = database();
    let execute = |q: &str| db.execute_sparql(q);
    for (query, every_pair) in [
        (
            "SELECT ?p ?n WHERE { ?p e:age 7 . ?p e:name ?n . }",
            "SELECT ?p ?n WHERE { ?p e:age 7 . { SELECT ?p ?n WHERE { ?p e:name ?n } } }",
        ),
        (
            "SELECT ?p ?n WHERE { ?p e:name ?n . ?p e:age 7 . }",
            "SELECT ?p ?n WHERE { { SELECT ?p ?n WHERE { ?p e:name ?n } } ?p e:age 7 . }",
        ),
        (
            "SELECT ?p ?q WHERE { ?p e:name \"n5\" . ?p e:knows ?q . ?q e:name ?m . }",
            "SELECT ?p ?q WHERE { ?p e:name \"n5\" . \
             { SELECT ?p ?q ?m WHERE { ?p e:knows ?q . ?q e:name ?m } } }",
        ),
    ] {
        assert_eq!(run(&execute, query), run(&execute, every_pair), "{query}");
    }
    assert_eq!(
        run(
            &execute,
            "SELECT ?p ?n WHERE { ?p e:age 7 . ?p e:name ?n . }"
        )
        .unwrap()
        .len(),
        40,
        "ages 7 of 2,000 people"
    );
}

/// An OPTIONAL left side holds ?n without always binding it: the plain join.
#[test]
fn an_optional_left_side_holding_the_free_variable() {
    let db = database();
    db.execute_sparql(&format!(
        "{PREFIX} INSERT DATA {{ e:p7 e:nick \"n7\" . e:p57 e:nick \"x\" . }}"
    ))
    .unwrap();
    let rows = agree(
        &db,
        "?p ?n",
        "?p e:age 7 . OPTIONAL { ?p e:nick ?n }",
        "?p e:name ?n .",
        "?p ?n",
    );
    assert_eq!(rows.len(), 39, "p57's nick is no name of it, p7's is");
    assert!(rows.contains(&"p7 n7".to_string()));
    let text = profile(
        &db,
        "SELECT ?p ?n WHERE { ?p e:age 7 . OPTIONAL { ?p e:nick ?n } ?p e:name ?n . }",
    );
    assert!(!text.contains("RdfTripleLookup"), "{text}");
}

/// A constant GRAPH; FROM over g1 and g2, which both hold `e:q1 e:name "q1"`
/// (FROM returns that triple twice today: the lookup must not change it);
/// GRAPH ?g (the plain join).
#[test]
fn graphs() {
    let db = database();
    db.execute_sparql(&format!(
        "{PREFIX} INSERT DATA {{ \
         GRAPH e:g1 {{ e:q1 e:name \"q1\" . e:q2 e:name \"q2\" . }} \
         GRAPH e:g2 {{ e:q1 e:name \"q1\" . e:q3 e:name \"q3\" . }} }}"
    ))
    .unwrap();
    let execute = |q: &str| db.execute_sparql(q);
    let left = "VALUES ?p { e:q1 e:q2 e:q3 e:p1 }";
    let in_g1 = format!("SELECT ?p ?n WHERE {{ {left} GRAPH e:g1 {{ ?p e:name ?n }} }}");
    let in_g1_joined = format!(
        "SELECT ?p ?n WHERE {{ {left} GRAPH e:g1 {{ {{ SELECT ?p ?n WHERE {{ ?p e:name ?n }} }} }} }}"
    );
    assert_eq!(run(&execute, &in_g1), run(&execute, &in_g1_joined));
    assert_eq!(
        run(&execute, &in_g1).unwrap(),
        ["q1 q1", "q2 q2"],
        "a constant GRAPH"
    );
    let from = agree_with(
        &execute,
        "?p ?n",
        "FROM e:g1 FROM e:g2",
        left,
        "?p e:name ?n .",
        "?p ?n",
    );
    let from_scan = run(
        &execute,
        "SELECT ?p ?n FROM e:g1 FROM e:g2 WHERE { ?p e:name ?n . FILTER(?p IN (e:q1, e:q2, e:q3, e:p1)) }",
    )
    .unwrap();
    assert_eq!(from, from_scan, "FROM as a scan gives it");
    let graph_free = "SELECT ?g ?n WHERE { VALUES ?p { e:q1 } GRAPH ?g { ?p e:name ?n } }";
    let graph_free_pairs = "SELECT ?g ?n WHERE { VALUES ?p { e:q1 } \
                            { SELECT ?g ?p ?n WHERE { GRAPH ?g { ?p e:name ?n } } } }";
    assert_eq!(run(&execute, graph_free), run(&execute, graph_free_pairs));
    assert_eq!(run(&execute, graph_free).unwrap(), ["g1 q1", "g2 q1"]);
}

/// A transaction's pending insert and delete, with 10 rows (looked up) and
/// 1,025 (joined as a hash join does): the writer sees them, a reader does
/// not, the rollback drops them.
#[test]
fn a_transactions_pending_changes() {
    let db = database();
    let mut writer = db.session();
    let reader = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(&format!("{PREFIX} INSERT DATA {{ e:p2 e:name \"Gus\" . }}"))
        .unwrap();
    writer
        .execute_sparql(&format!("{PREFIX} DELETE DATA {{ e:p3 e:name \"n3\" . }}"))
        .unwrap();
    let names = |session: &grafeo_engine::session::Session, count: usize| {
        let execute = |q: &str| session.execute_sparql(q);
        let rows = agree_with(
            &execute,
            "?p ?n",
            "",
            &values_of_people(count),
            "?p e:name ?n .",
            "?p ?n",
        );
        (
            rows.contains(&"p2 Gus".to_string()),
            rows.contains(&"p3 n3".to_string()),
        )
    };
    for count in [10, 1_025] {
        assert_eq!(names(&writer, count), (true, false), "writer, {count} rows");
        assert_eq!(names(&reader, count), (false, true), "reader, {count} rows");
    }
    writer.rollback().unwrap();
    for count in [10, 1_025] {
        assert_eq!(
            names(&writer, count),
            (false, true),
            "after the rollback, {count} rows"
        );
    }
}

/// PROFILE shows the lookup with its rows; the sub-select form does not.
#[test]
fn profile_shows_the_lookup() {
    let db = database();
    let text = profile(
        &db,
        "SELECT ?p ?n WHERE { VALUES ?p { e:p1 e:p7 e:p8 } ?p e:name ?n . }",
    );
    let lookups: Vec<&str> = text
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("RdfTripleLookup"))
        .collect();
    assert_eq!(lookups.len(), 1, "{text}");
    assert!(lookups[0].contains("rows=4"), "p1 has two names: {text}");
    let joined = profile(
        &db,
        "SELECT ?p ?n WHERE { VALUES ?p { e:p1 e:p7 e:p8 } { SELECT ?p ?n WHERE { ?p e:name ?n } } }",
    );
    assert!(!joined.contains("RdfTripleLookup"), "{joined}");
}
