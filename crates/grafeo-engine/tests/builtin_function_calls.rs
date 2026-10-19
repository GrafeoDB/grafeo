//! Calls resolve in the translators (#540): the same errors in every
//! language, aggregates inside expressions still group, EXPLAIN prints the
//! call as written, and a volatile call is never folded.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test builtin_function_calls
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_engine::GrafeoDB;

fn error(db: &GrafeoDB, language: &str, query: &str) -> String {
    db.execute_language(query, language, None)
        .expect_err(query)
        .to_string()
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> String {
    format!(
        "{:?}",
        db.execute_language(query, language, None)
            .unwrap_or_else(|e| panic!("{query}: {e}"))
            .rows()
    )
}

#[test]
fn call_errors_keep_their_text_in_every_language() {
    let db = GrafeoDB::new_in_memory();
    for language in ["gql", "cypher"] {
        let unknown = error(&db, language, "RETURN upperr('a') AS x");
        assert!(
            unknown.contains("Unknown function 'upperr'"),
            "{language}: {unknown}"
        );
        assert!(
            unknown.contains("upper"),
            "{language}: the hint names upper: {unknown}"
        );
        let arity = error(&db, language, "RETURN toUpper('a', 'b') AS x");
        assert!(
            arity.contains("Function 'toUpper' takes 1 argument, got 2"),
            "{language}: {arity}"
        );
    }
}

#[test]
fn aggregates_inside_expressions_still_group() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix', city: 'Paris'}), (:Person {name: 'Gus', city: 'Paris'}), \
         (:Person {name: 'Vincent', city: 'Prague'})",
    )
    .unwrap();
    for language in ["gql", "cypher"] {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (p:Person) RETURN p.city AS c, size(collect(p.name)) AS n, \
                 count(DISTINCT p) AS d ORDER BY c"
            ),
            "[[String(\"Paris\"), Int64(2), Int64(2)], [String(\"Prague\"), Int64(1), Int64(1)]]",
            "{language}"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (p:Person) WITH p ORDER BY p.name RETURN head(collect(p.name)) AS h"
            ),
            "[[String(\"Alix\")]]",
            "{language}"
        );
    }
}

#[test]
fn explain_prints_the_call_as_written() {
    let db = GrafeoDB::new_in_memory();
    let plan = rows(&db, "gql", "EXPLAIN MATCH (p) RETURN toUpper(p.name) AS u");
    assert!(plan.contains("toUpper(p.name)"), "{plan}");
}

#[test]
fn folding_skips_volatile_calls() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 50) AS i INSERT (:R {v: rand(), d: date('2026-10-17')})")
        .unwrap();
    assert_eq!(
        rows(&db, "gql", "MATCH (r:R) RETURN count(DISTINCT r.v) AS n"),
        "[[Int64(50)]]",
        "rand() runs per row"
    );
    assert_eq!(
        rows(
            &db,
            "gql",
            "MATCH (r:R) RETURN DISTINCT r.d = date('2026-10-17') AS same"
        ),
        "[[Bool(true)]]"
    );
}

/// An aggregate where no rows are grouped (a `WHERE` on the match) is an
/// error: the evaluator has no value for it, so the filter kept no rows.
#[test]
fn an_aggregate_outside_a_grouping_is_an_error() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
        .unwrap();
    for language in ["gql", "cypher"] {
        let message = error(
            &db,
            language,
            "MATCH (p:Person) WHERE count(p) > 0 RETURN p.name",
        );
        assert!(
            message.contains("Aggregate function 'count'"),
            "{language}: {message}"
        );
    }
}
