//! An OPTIONAL of one triple pattern whose variables the rows before it
//! always bind looks its triples up per row (0.6.1, LDBC IS4 on RDF); a row
//! with no triple keeps its values with the pattern's variables unbound. Past
//! the lookup limit it joins as a left hash join does. Its rows must be those
//! of the plain left join, which the same query with the pattern in a
//! sub-select gives (a sub-select is not looked up); each case compares the
//! two, and the main ones check their rows:
//!
//! - a left row with no triple, repeated left rows, a term no triple has, a
//!   term a BIND computes;
//! - VALUES of 10 rows (looked up) and of 1,000 and 5,000 (past the limit);
//! - two bound positions;
//! - an OPTIONAL with a FILTER (a condition: the plain left join), nested
//!   OPTIONAL patterns, and an OPTIONAL sharing a variable the left side may leave
//!   unbound (the plain left join);
//! - a transaction's pending insert and delete;
//! - PROFILE: the lookup on the left join, with its rows.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_optional_lookup
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

/// People p0 to p1999 with `e:name "n{i}"` and `e:age {i % 50}`; the even
/// ones also `e:extra "x{i}"`, and p2 a second, "y2"; p{i} `e:knows` p{i+1}
/// for i below 10. p1 has `e:nick "Gus"` and p3 two nicks, "Mia" and "Jules"
/// (`e:nick` is too small a pattern to look up from more than one row).
fn database() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut data = String::new();
    for i in 0..2_000 {
        write!(data, "e:p{i} e:name \"n{i}\" ; e:age {} . ", i % 50).unwrap();
        if i % 2 == 0 {
            write!(data, "e:p{i} e:extra \"x{i}\" . ").unwrap();
        }
        if i < 10 {
            write!(data, "e:p{i} e:knows e:p{} . ", i + 1).unwrap();
        }
    }
    data.push_str("e:p1 e:nick \"Gus\" . e:p3 e:nick \"Mia\" , \"Jules\" . e:p2 e:extra \"y2\" . ");
    db.execute_sparql(&format!("{PREFIX} INSERT DATA {{ {data} }}"))
        .unwrap();
    db
}

fn text(value: &Value) -> String {
    use grafeo_common::types::rdf::Term;
    match value {
        Value::RdfTerm(term) => match term.as_ref() {
            Term::Iri(iri) => iri
                .as_str()
                .strip_prefix("http://example.org/")
                .unwrap_or(iri.as_str())
                .to_string(),
            Term::Literal(literal) => literal.value().to_string(),
            other => format!("{other:?}"),
        },
        Value::Null => "-".to_string(),
        other => format!("{other:?}"),
    }
}

type Execute<'a> = &'a dyn Fn(&str) -> Result<QueryResult>;

fn run(execute: Execute<'_>, query: &str) -> std::result::Result<Vec<String>, String> {
    execute(&format!("{PREFIX} {query}"))
        .map(|result| {
            let mut rows: Vec<String> = result
                .rows()
                .iter()
                .map(|row| row.iter().map(text).collect::<Vec<_>>().join(" "))
                .collect();
            rows.sort();
            rows
        })
        .map_err(|error| error.to_string())
}

/// `SELECT {projection} WHERE { {left} OPTIONAL { {pattern} } }` against the
/// same with the pattern in a sub-select over `variables`; they must agree.
fn agree_with(
    execute: Execute<'_>,
    projection: &str,
    left: &str,
    pattern: &str,
    variables: &str,
) -> Vec<String> {
    let looked_up = format!("SELECT {projection} WHERE {{ {left} OPTIONAL {{ {pattern} }} }}");
    let joined = format!(
        "SELECT {projection} WHERE {{ {left} OPTIONAL {{ {{ SELECT {variables} WHERE {{ {pattern} }} }} }} }}"
    );
    let got = run(execute, &looked_up);
    assert_eq!(
        got,
        run(execute, &joined),
        "`{looked_up}` against the left join"
    );
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
        left,
        pattern,
        variables,
    )
}

fn values_of_people(count: usize) -> String {
    let mut values = String::from("VALUES ?p { ");
    for i in 0..count {
        write!(values, "e:p{} ", i % 2_000).unwrap();
    }
    values.push('}');
    values
}

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

fn lookups(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| line.starts_with("RdfTripleLookup"))
        .map(str::to_string)
        .collect()
}

/// Looked up (five rows, a pattern of 1,001 triples): a left row with no
/// triple keeps its values; a row with two triples gives two rows; repeated
/// left rows give theirs again; an unknown term keeps its row unbound. The
/// same over the small `e:nick` pattern (the plain left join).
#[test]
fn rows_without_a_triple_repeated_rows_and_unknown_terms() {
    let db = database();
    assert_eq!(
        agree(
            &db,
            "?p ?x",
            "VALUES ?p { e:p1 e:p2 e:p3 e:p1 e:nobody }",
            "?p e:extra ?x",
            "?p ?x"
        ),
        ["nobody -", "p1 -", "p1 -", "p2 x2", "p2 y2", "p3 -"]
    );
    assert_eq!(
        agree(
            &db,
            "?p ?k",
            "VALUES ?p { e:p1 e:p2 e:p3 e:p1 e:nobody }",
            "?p e:nick ?k",
            "?p ?k"
        ),
        ["nobody -", "p1 Gus", "p1 Gus", "p2 -", "p3 Jules", "p3 Mia"]
    );
    assert_eq!(
        agree(
            &db,
            "?p ?x",
            "BIND(IRI(CONCAT(STR(e:), \"p\", \"8\")) AS ?p)",
            "?p e:extra ?x",
            "?p ?x"
        ),
        ["p8 x8"],
        "a computed start"
    );
}

/// 10 rows look up; 1,000 and 5,000 are past the limit of a pattern of
/// 1,001 triples. Every second person has `e:extra`, p2 two.
#[test]
fn values_on_both_sides_of_the_limit() {
    let db = database();
    for count in [10, 1_000, 5_000] {
        let rows = agree(
            &db,
            "?p ?x",
            &values_of_people(count),
            "?p e:extra ?x",
            "?p ?x",
        );
        let p2 = (0..count).filter(|i| i % 2_000 == 2).count();
        assert_eq!(rows.len(), count + p2, "{count}: one row each, p2 two");
        let unbound = rows.iter().filter(|row| row.ends_with(" -")).count();
        assert_eq!(unbound, count / 2, "{count}: the odd ones unbound");
    }
}

/// Both positions bound from the left: a pair the store holds and one it
/// does not.
#[test]
fn two_bound_positions() {
    let db = database();
    assert_eq!(
        agree(
            &db,
            "?p ?q ?r",
            "VALUES (?p ?q) { (e:p1 e:p2) (e:p1 e:p3) }",
            "?p ?r ?q",
            "?p ?r ?q"
        ),
        ["p1 p2 knows", "p1 p3 -"]
    );
}

/// An OPTIONAL with a FILTER has a condition: the plain left join. Its rows
/// are those the filter keeps.
#[test]
fn an_optional_with_a_filter() {
    let db = database();
    let query = "SELECT ?p ?a WHERE { VALUES ?p { e:p10 e:p45 } \
                 OPTIONAL { ?p e:age ?a FILTER(?a > 20) } }";
    let execute = |q: &str| db.execute_sparql(q);
    assert_eq!(run(&execute, query).unwrap(), ["p10 -", "p45 45"]);
}

/// Nested OPTIONAL patterns: p0 to p9 know the next one; the nick of whom they know,
/// when there is one.
#[test]
fn nested_optionals() {
    let db = database();
    let query = "SELECT ?p ?q ?k WHERE { VALUES ?p { e:p0 e:p2 e:p9 e:p11 } \
                 OPTIONAL { ?p e:knows ?q OPTIONAL { ?q e:nick ?k } } }";
    let joined = "SELECT ?p ?q ?k WHERE { VALUES ?p { e:p0 e:p2 e:p9 e:p11 } \
                  OPTIONAL { { SELECT ?p ?q WHERE { ?p e:knows ?q } } \
                  OPTIONAL { { SELECT ?q ?k WHERE { ?q e:nick ?k } } } } }";
    let execute = |q: &str| db.execute_sparql(q);
    assert_eq!(run(&execute, query), run(&execute, joined));
    assert_eq!(
        run(&execute, query).unwrap(),
        [
            "p0 p1 Gus",
            "p11 - -",
            "p2 p3 Jules",
            "p2 p3 Mia",
            "p9 p10 -"
        ]
    );
}

/// The left side may leave ?k unbound (an earlier OPTIONAL), and the
/// OPTIONAL's pattern holds ?k: the plain left join, with SPARQL's
/// compatible rows.
#[test]
fn a_variable_the_left_side_may_leave_unbound() {
    let db = database();
    let rows = agree(
        &db,
        "?p ?k ?q",
        "VALUES ?p { e:p1 e:p2 } OPTIONAL { ?p e:nick ?k }",
        "?q e:nick ?k",
        "?q ?k",
    );
    assert_eq!(
        rows,
        ["p1 Gus p1", "p2 Gus p1", "p2 Jules p3", "p2 Mia p3"],
        "p2's unbound ?k is compatible with every nick, and takes it"
    );
}

/// A transaction's pending insert and delete, looked up and past the limit.
#[test]
fn a_transactions_pending_changes() {
    let db = database();
    let mut writer = db.session();
    let reader = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute_sparql(&format!(
            "{PREFIX} INSERT DATA {{ e:p5 e:extra \"new\" . }}"
        ))
        .unwrap();
    writer
        .execute_sparql(&format!("{PREFIX} DELETE DATA {{ e:p4 e:extra \"x4\" . }}"))
        .unwrap();
    let extras = |session: &grafeo_engine::session::Session, count: usize| {
        let execute = |q: &str| session.execute_sparql(q);
        let rows = agree_with(
            &execute,
            "?p ?x",
            &values_of_people(count),
            "?p e:extra ?x",
            "?p ?x",
        );
        (
            rows.contains(&"p5 new".to_string()),
            rows.contains(&"p4 x4".to_string()),
        )
    };
    for count in [10, 1_000] {
        assert_eq!(extras(&writer, count), (true, false), "writer, {count}");
        assert_eq!(extras(&reader, count), (false, true), "reader, {count}");
    }
    writer.rollback().unwrap();
    assert_eq!(extras(&writer, 10), (false, true), "rolled back");
}

/// PROFILE: one lookup on the left join, with its rows; none in the
/// sub-select form or with a FILTER in the OPTIONAL.
#[test]
fn profile_shows_the_left_lookup() {
    let db = database();
    let text = profile(
        &db,
        "SELECT ?p ?x WHERE { VALUES ?p { e:p1 e:p2 e:p3 } OPTIONAL { ?p e:extra ?x } }",
    );
    let found = lookups(&text);
    assert_eq!(found.len(), 1, "{text}");
    assert!(found[0].contains("rows=4"), "none, x2, y2, none: {text}");
    let joined = profile(
        &db,
        "SELECT ?p ?x WHERE { VALUES ?p { e:p1 e:p2 e:p3 } \
         OPTIONAL { { SELECT ?p ?x WHERE { ?p e:extra ?x } } } }",
    );
    assert!(lookups(&joined).is_empty(), "{joined}");
    let filtered = profile(
        &db,
        "SELECT ?p ?a WHERE { VALUES ?p { e:p10 e:p45 } \
         OPTIONAL { ?p e:age ?a FILTER(?a > 20) } }",
    );
    assert!(lookups(&filtered).is_empty(), "{filtered}");
}
