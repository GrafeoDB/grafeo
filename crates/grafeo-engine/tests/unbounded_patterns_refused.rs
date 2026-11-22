//! A path pattern that can match without end is refused before it runs (ISO GQL and SQL/PGQ;
//! user decision of 2026-11-18): every quantifier without an upper bound (`*`, `+`, `{m,}`,
//! `-[*]->`, `-[*2..]->`, on an edge or a parenthesized pattern) must be in the scope of a
//! selector (ANY, ANY k, ANY SHORTEST, ALL SHORTEST, SHORTEST k, SHORTEST k GROUPS,
//! `shortestPath`, `allShortestPaths`), of a restrictor (TRAIL, ACYCLIC, SIMPLE) on the MATCH, the
//! pattern or a parenthesized pattern around it, or of DIFFERENT EDGES. Otherwise it fails with
//! `42001`, at once, naming the fixes: `ALL` alone, WALK, and a restrictor only inside the
//! repeated part bound nothing. Bounded quantifiers need nothing, and Cypher, whose
//! variable-length relationships are trails, is unchanged.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main unbounded_patterns_refused::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::utils::error::Error;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;

/// Three cities in a cycle of `:ROAD` edges, where a walk never ends.
fn cycle() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (v:City {name: 'Vienna'})-[:ROAD]->(a:City {name: 'Amsterdam'})\
         -[:ROAD]->(u:City {name: 'Utrecht'})-[:ROAD]->(v)",
    )
    .unwrap();
    db
}

/// The unbounded quantifiers, as path patterns from `a` to `b`: on an edge (inside the
/// brackets and after the arrow) and on a parenthesized pattern.
const UNBOUNDED: &[&str] = &[
    "(a)-[:ROAD*]->(b)",
    "(a)-[*]->(b)",
    "(a)-[:ROAD*2..]->(b)",
    "(a)-[:ROAD]->+(b)",
    "(a)-[:ROAD]->*(b)",
    "(a)-[:ROAD]->{2,}(b)",
    "((a)-[:ROAD]->(b))+",
    "((a)-[:ROAD]->(b)){1,}",
];

/// `result` is the refusal of an unbounded pattern: `42001`, naming both fixes.
fn assert_unbounded<T: std::fmt::Debug>(query: &str, result: Result<T, Error>) {
    let error = result.expect_err(query);
    let message = error.to_string();
    assert_eq!(
        GqlStatus::from(&error).as_str(),
        "42001",
        "{query}: {message}"
    );
    assert!(
        message.contains("no upper bound") && message.contains("TRAIL"),
        "{query}: names the quantifier and the fixes: {message}"
    );
    assert!(message.contains("ANY SHORTEST"), "{query}: {message}");
}

/// `query` answers.
fn assert_answers(db: &GrafeoDB, language: &str, query: &str) {
    let result = db
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
    assert_eq!(result.rows().len(), 1, "{language} `{query}`");
}

#[test]
fn an_unbounded_pattern_without_a_bound_is_refused() {
    let db = cycle();
    for pattern in UNBOUNDED {
        for query in [
            format!("MATCH {pattern} RETURN count(*)"),
            format!("MATCH p = {pattern} RETURN count(p)"),
            // `ALL` alone keeps every path, and WALK is no restrictor.
            format!("MATCH ALL {pattern} RETURN count(*)"),
            format!("MATCH p = ALL {pattern} RETURN count(p)"),
            format!("MATCH WALK {pattern} RETURN count(*)"),
            format!("MATCH p = WALK {pattern} RETURN count(p)"),
            format!("MATCH {pattern} KEEP REPEATABLE ELEMENTS RETURN count(*)"),
            // Every place a pattern is matched.
            format!("OPTIONAL MATCH {pattern} RETURN count(*)"),
            format!("MATCH (c:City) WHERE EXISTS {{ MATCH {pattern} }} RETURN count(c)"),
            format!("MATCH (c:City) CALL {{ MATCH {pattern} RETURN count(*) AS n }} RETURN sum(n)"),
        ] {
            assert_unbounded(&query, db.execute(&query));
        }
    }
}

/// A restrictor inside the repeated part restricts each repetition, not how many there are.
#[test]
fn a_restrictor_inside_the_repetition_does_not_bound_it() {
    let db = cycle();
    for query in [
        "MATCH (TRAIL (a)-[:ROAD]->(b))+ RETURN count(*)",
        "MATCH (ACYCLIC (a)-[:ROAD]->(b)){1,} RETURN count(*)",
        "MATCH (SIMPLE (a)-[:ROAD]->(b))* RETURN count(*)",
    ] {
        assert_unbounded(query, db.execute(query));
    }
}

/// The unbounded quantifiers on an edge: a path search (a selector) runs over one edge pattern
/// between two nodes, and over a parenthesized pattern it is not supported (`Q004`, as before).
const UNBOUNDED_EDGES: &[&str] = &[
    "(a)-[:ROAD*]->(b)",
    "(a)-[*]->(b)",
    "(a)-[:ROAD*2..]->(b)",
    "(a)-[:ROAD]->+(b)",
    "(a)-[:ROAD]->*(b)",
    "(a)-[:ROAD]->{2,}(b)",
];

#[test]
fn a_selector_bounds_an_unbounded_pattern() {
    let db = cycle();
    for pattern in UNBOUNDED_EDGES {
        for selector in [
            "ANY",
            "ANY 2",
            "ANY SHORTEST",
            "ALL SHORTEST",
            "SHORTEST 2",
            "SHORTEST 2 GROUPS",
        ] {
            assert_answers(
                &db,
                "gql",
                &format!("MATCH p = {selector} {pattern} RETURN count(p)"),
            );
            assert_answers(
                &db,
                "gql",
                &format!("MATCH {selector} {pattern} RETURN count(*)"),
            );
        }
    }
    for function in ["shortestPath", "allShortestPaths"] {
        assert_answers(
            &db,
            "gql",
            &format!("MATCH p = {function}((a)-[:ROAD*]->(b)) RETURN count(p)"),
        );
    }
}

#[test]
fn a_restrictor_or_different_edges_bounds_an_unbounded_pattern() {
    let db = cycle();
    for pattern in UNBOUNDED {
        for mode in ["TRAIL", "ACYCLIC", "SIMPLE"] {
            for query in [
                format!("MATCH {mode} {pattern} RETURN count(*)"),
                format!("MATCH p = {mode} {pattern} RETURN count(p)"),
                format!("MATCH p = ALL {mode} {pattern} RETURN count(p)"),
            ] {
                assert_answers(&db, "gql", &query);
            }
        }
        for query in [
            format!("MATCH DIFFERENT EDGES {pattern} RETURN count(*)"),
            format!("MATCH {pattern} KEEP DIFFERENT EDGES RETURN count(*)"),
        ] {
            assert_answers(&db, "gql", &query);
        }
    }
    // A restrictor on a parenthesized pattern bounds the quantifiers inside it.
    assert_answers(
        &db,
        "gql",
        "MATCH (TRAIL (a)-[:ROAD]->{1,}(b)){1} RETURN count(*)",
    );
}

/// A quantifier with an upper bound needs no mode or selector.
#[test]
fn a_bounded_quantifier_needs_nothing() {
    let db = cycle();
    for pattern in [
        "(a)-[:ROAD]->(b)",
        "(a)-[:ROAD*1..5]->(b)",
        "(a)-[:ROAD*3]->(b)",
        "(a)-[:ROAD]->{1,5}(b)",
        "(a)-[:ROAD]->{2}(b)",
        "(a)-[:ROAD]->?(b)",
        "((a)-[:ROAD]->(b)){1,4}",
    ] {
        assert_answers(&db, "gql", &format!("MATCH {pattern} RETURN count(*)"));
    }
}

/// SQL/PGQ's `GRAPH_TABLE` takes GQL's patterns, and the same rule.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_refuses_an_unbounded_pattern_too() {
    let db = cycle();
    let table = |r#match: &str| {
        format!("SELECT * FROM GRAPH_TABLE ({match} COLUMNS (a.name AS a, b.name AS b))")
    };
    for pattern in UNBOUNDED {
        for refused in [
            format!("MATCH {pattern}"),
            format!("MATCH ALL {pattern}"),
            format!("MATCH WALK {pattern}"),
        ] {
            let query = table(&refused);
            assert_unbounded(&query, db.execute_language(&query, "sql", None));
        }
        let selected = UNBOUNDED_EDGES
            .contains(pattern)
            .then(|| format!("MATCH p = ANY SHORTEST {pattern}"));
        for accepted in [
            Some(format!("MATCH TRAIL {pattern}")),
            selected,
            Some(format!("MATCH DIFFERENT EDGES {pattern}")),
        ]
        .into_iter()
        .flatten()
        {
            let query = table(&accepted);
            let result = db.execute_language(&query, "sql", None);
            assert!(result.is_ok(), "{query}: {result:?}");
        }
    }
}

/// Cypher's variable-length relationships are trails, which end: unchanged.
#[cfg(feature = "cypher")]
#[test]
fn cypher_is_unchanged() {
    let db = cycle();
    for query in [
        "MATCH (a)-[:ROAD*]->(b) RETURN count(*)",
        "MATCH (a)-[*]->(b) RETURN count(*)",
        "MATCH (a)-[:ROAD*2..]->(b) RETURN count(*)",
        "MATCH p = (a)-[:ROAD*]->(b) RETURN count(p)",
    ] {
        assert_answers(&db, "cypher", query);
    }
}
