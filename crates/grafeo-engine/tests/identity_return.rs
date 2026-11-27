//! A `RETURN` of nothing but its aggregates, as they are, plans no projection: `MATCH (n:Person)
//! RETURN count(n)` is the count alone, as in 0.6.3, where the projection over it named the same
//! column again and cost every statement an operator. Every other `RETURN` keeps its projection:
//! one that is `DISTINCT`, changes the columns or their order, reads an id back as its node, or
//! has an `ORDER BY`, `SKIP` or `LIMIT` over it. An aggregate's alias names its own column
//! (`count(n) AS people`), so its `RETURN` renames nothing; one that renames a column keeps its
//! projection (pinned on a plan in the optimizer's tests, as no statement writes one).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main identity_return::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// Alix (30, Amsterdam), Gus (40, Berlin) and Vincent (50, Amsterdam).
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix', age: 30, city: 'Amsterdam'}), \
         (:Person {name: 'Gus', age: 40, city: 'Berlin'}), \
         (:Person {name: 'Vincent', age: 50, city: 'Amsterdam'})",
    )
    .unwrap();
    db
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> QueryResult {
    match language {
        "gql" => db.execute(query),
        "cypher" => db.execute_cypher(query),
        other => panic!("unknown language {other}"),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
}

/// The first line of `query`'s plan: its root operator.
fn root(db: &GrafeoDB, language: &str, query: &str) -> String {
    let result = run(db, language, &format!("EXPLAIN {query}"));
    let text = match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("EXPLAIN gives text, not {other:?}"),
    };
    text.lines().next().unwrap_or_default().trim().to_string()
}

/// A `RETURN` of its aggregates as they are is the aggregate itself, with the column named as
/// written and the same value.
#[test]
fn a_return_of_its_aggregates_as_they_are_plans_no_projection() {
    let db = database();
    for language in ["gql", "cypher"] {
        for (query, column, value) in [
            (
                "MATCH (n:Person) RETURN count(n)",
                "count(n)",
                Value::Int64(3),
            ),
            (
                "MATCH (n:Person) RETURN sum(n.age)",
                "sum(n.age)",
                Value::Int64(120),
            ),
            (
                "MATCH (n:Person) RETURN count(n) AS people",
                "people",
                Value::Int64(3),
            ),
        ] {
            // Cypher's sum of no rows is 0: its RETURN computes `coalesce(sum(..), 0)`, which
            // `every_other_return_keeps_its_projection` pins.
            if language == "cypher" && query.contains("sum(") {
                continue;
            }
            let first = root(&db, language, query);
            assert!(
                !first.starts_with("Return"),
                "{language}: {query}: the root is {first}"
            );
            let result = run(&db, language, query);
            assert_eq!(
                result.columns,
                vec![column.to_string()],
                "{language}: {query}"
            );
            assert_eq!(result.rows(), &[vec![value.clone()]], "{language}: {query}");
        }
    }
}

/// Every other `RETURN` over an aggregate keeps its projection, or the operator its clause adds
/// stays over it.
#[test]
fn every_other_return_keeps_its_projection() {
    let db = database();
    for language in ["gql", "cypher"] {
        for (query, case) in [
            ("MATCH (n:Person) RETURN DISTINCT count(n)", "DISTINCT"),
            (
                "MATCH (n:Person) RETURN count(n) > 0 AS any",
                "a wrapped aggregate (other columns)",
            ),
            (
                "MATCH (n:Person) RETURN count(n), n.city",
                "a key after the aggregate (another order)",
            ),
            (
                "MATCH (n:Person) RETURN n, count(n)",
                "an id read back as its node",
            ),
            (
                "MATCH (n:Person) RETURN collect(n)",
                "ids read back as nodes",
            ),
            (
                "MATCH (n:Person) RETURN count(n) ORDER BY count(n)",
                "ORDER BY",
            ),
            ("MATCH (n:Person) RETURN count(n) SKIP 0", "SKIP"),
            ("MATCH (n:Person) RETURN count(n) LIMIT 1", "LIMIT"),
        ] {
            let first = root(&db, language, query);
            assert!(
                !first.starts_with("LabelCount") && !first.starts_with("Aggregate"),
                "{language}: {query} ({case}): the root is {first}"
            );
        }
    }
    let first = root(&db, "cypher", "MATCH (n:Person) RETURN sum(n.age)");
    assert!(
        first.starts_with("Return") && first.contains("coalesce"),
        "Cypher's sum computes its value for no rows: the root is {first}"
    );
}

/// The rows of the kept cases are as before.
#[test]
fn the_kept_cases_give_their_rows() {
    let db = database();
    for language in ["gql", "cypher"] {
        let people = run(&db, language, "MATCH (n:Person) RETURN count(n) AS people");
        assert_eq!(people.columns, vec!["people".to_string()], "{language}");
        assert_eq!(people.rows(), &[vec![Value::Int64(3)]], "{language}");
        let nodes = run(&db, language, "MATCH (n:Person) RETURN n, count(n)");
        assert_eq!(nodes.rows().len(), 3, "{language}");
        assert!(
            nodes
                .rows()
                .iter()
                .all(|row| matches!(row[0], Value::Map(_)) && row[1] == Value::Int64(1)),
            "{language}: each node read back with its count: {:?}",
            nodes.rows()
        );
        let distinct = run(&db, language, "MATCH (n:Person) RETURN DISTINCT count(n)");
        assert_eq!(distinct.rows(), &[vec![Value::Int64(3)]], "{language}");
    }
}
