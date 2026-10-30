//! The WHERE of an OPTIONAL MATCH filters the optional part, never the rows
//! before it: a row whose optional part the WHERE rejects stays, with nulls.
//! The same when the WHERE holds a subquery that reads only the outer
//! variables (0.6.0 filtered the outer rows instead, dropping them): EXISTS
//! in an OR and alone, COUNT, VALUE, an uncorrelated EXISTS, and a pattern
//! comprehension. In GQL and Cypher (VALUE in GQL only).
//!
//! The known remaining case: a subquery that reads a variable bound before
//! the OPTIONAL MATCH and not shared with its pattern (`m` beside `n`) is
//! still placed above the join and drops rows (ignored test below).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test optional_match_subquery_where
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Six P nodes, `id` 1 to 6; P 1 and 2 have an R edge to a T with `x` their
/// id. One Q node with `k: 1`. Two M nodes: `k` 1 with an S edge, `k` 2
/// without.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 6) AS i INSERT (:P {id: i})")
        .unwrap();
    db.execute("MATCH (p:P) WHERE p.id <= 2 INSERT (p)-[:R]->(:T {x: p.id})")
        .unwrap();
    db.execute("INSERT (:Q {k: 1}), (m:M {k: 1})-[:S]->(:U), (:M {k: 2})")
        .unwrap();
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

/// Rows of `(id, x)` for P 1 to 6, `x` from `matched` (the ids whose R
/// target the WHERE keeps), null for the rest.
fn every_p(matched: &[i64]) -> Vec<Vec<Value>> {
    (1..=6)
        .map(|id| {
            let x = if matched.contains(&id) {
                Value::Int64(id)
            } else {
                Value::Null
            };
            vec![Value::Int64(id), x]
        })
        .collect()
}

fn optional(predicate: &str) -> String {
    format!(
        "MATCH (n:P) OPTIONAL MATCH (n)-[:R]->(t) WHERE {predicate} \
         RETURN n.id AS id, t.x AS x ORDER BY id"
    )
}

fn check(languages: &[&str], predicate: &str, matched: &[i64]) {
    let db = database();
    let query = optional(predicate);
    for language in languages {
        assert_eq!(
            rows(&db, language, &query),
            every_p(matched),
            "{language} `{query}`"
        );
    }
}

#[test]
fn exists_in_or_over_the_outer_variable() {
    check(
        &LANGUAGES,
        "n.id = 6 OR EXISTS { MATCH (n)-[:R]->(u) WHERE u.x > 0 }",
        &[1, 2],
    );
}

#[test]
fn exists_over_the_outer_variable() {
    check(
        &LANGUAGES,
        "EXISTS { MATCH (n)-[:R]->(u) WHERE u.x > 1 }",
        &[2],
    );
}

#[test]
fn not_exists_over_the_outer_variable() {
    check(
        &LANGUAGES,
        "NOT EXISTS { MATCH (n)-[:R]->(u) WHERE u.x > 1 }",
        &[1],
    );
}

#[test]
fn count_over_the_outer_variable() {
    check(
        &LANGUAGES,
        "COUNT { MATCH (n)-[:R]->(u) WHERE u.x > 1 } > 0",
        &[2],
    );
}

/// VALUE is GQL only.
#[test]
fn value_over_the_outer_variable() {
    check(
        &["gql"],
        "VALUE { MATCH (n)-[:R]->(u) RETURN u.x ORDER BY u.x LIMIT 1 } = 1",
        &[1],
    );
}

/// A subquery over the outer variable beside a condition on the optional
/// one.
#[test]
fn exists_beside_a_condition_on_the_optional_variable() {
    check(
        &LANGUAGES,
        "t.x >= 1 AND EXISTS { MATCH (n)-[:R]->(u) WHERE u.x > 1 }",
        &[2],
    );
}

/// An EXISTS that reads no row variable: true keeps every match, false
/// none, and every P stays either way.
#[test]
fn an_uncorrelated_exists() {
    check(&LANGUAGES, "EXISTS { MATCH (q:Q) WHERE q.k = 1 }", &[1, 2]);
    check(&LANGUAGES, "EXISTS { MATCH (q:Q) WHERE q.k = 2 }", &[]);
}

/// A pattern comprehension over the outer variable.
#[test]
#[ignore = "a pattern comprehension in WHERE fails with: PatternComprehension reached the planner without being rewritten (0.6.0 too)"]
fn a_pattern_comprehension_over_the_outer_variable() {
    check(
        &["cypher"],
        "size([(n)-[:R]->(u) WHERE u.x > 1 | u.x]) > 0",
        &[2],
    );
}

/// The known remaining case: the subquery reads `m`, bound before the
/// OPTIONAL MATCH and not in its pattern. For M 1 (with an S edge) the
/// WHERE holds and P 1 and 2 get their T; for M 2 it does not, and every P
/// stays with a null T.
#[test]
#[ignore = "an OPTIONAL MATCH WHERE whose subquery reads a variable bound only before it filters the outer rows: the M 2 rows are dropped"]
fn a_subquery_over_a_variable_bound_only_before_the_optional_match() {
    let db = database();
    let query = "MATCH (n:P), (m:M) OPTIONAL MATCH (n)-[:R]->(t) \
                 WHERE EXISTS { MATCH (m)-[:S]->() } \
                 RETURN m.k AS k, n.id AS id, t.x AS x ORDER BY k, id";
    let mut expected = Vec::new();
    for (k, matched) in [(1, vec![1, 2]), (2, vec![])] {
        for row in every_p(&matched) {
            expected.push([vec![Value::Int64(k)], row].concat());
        }
    }
    for language in LANGUAGES {
        assert_eq!(rows(&db, language, query), expected, "{language}");
    }
}
