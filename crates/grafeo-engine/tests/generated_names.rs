//! The names a translator makes up for anonymous nodes and edges (`_anon_0`)
//! are counted per statement and skip every name the statement spells: a
//! user variable named like one keeps its own value, and the same statement
//! gets the same plan each time. An undefined variable named like one is an
//! error like any other.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test generated_names
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix and Gus, and Alix KNOWS Gus twice. Every node and edge is named, so
/// the setup makes up no name.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (alix)-[k1:KNOWS]->(gus), (alix)-[k2:KNOWS]->(gus)",
    )
    .unwrap();
    db
}

/// The rows of `result` as text, sorted.
fn rows(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = result
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    Value::Bool(flag) => flag.to_string(),
                    Value::Int64(number) => number.to_string(),
                    other => panic!("unexpected value {other:?}"),
                })
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

/// The anonymous target of the first statement translated used to be named
/// `_anon_0` too (a counter shared by every statement of the process), which
/// made the pattern a loop on the user's `_anon_0`.
#[test]
fn a_user_variable_named_like_a_generated_one_keeps_its_node() {
    let db = graph();
    let query = "MATCH (_anon_0:Person {name: 'Alix'})-[:KNOWS]->() RETURN _anon_0.name";
    let want = vec![vec!["Alix".to_string()], vec!["Alix".to_string()]];
    assert_eq!(rows(db.execute(query)), want, "GQL");
    #[cfg(feature = "cypher")]
    assert_eq!(rows(db.execute_cypher(query)), want, "Cypher");
}

/// The names of a statement do not depend on the statements translated
/// before it: two statements of the same shape name their anonymous nodes
/// alike. (The same text twice would reuse the cached plan.)
#[test]
fn the_generated_names_of_a_statement_are_its_own() {
    let db = graph();
    let explain = |variable: &str| {
        rows(db.execute(&format!(
            "EXPLAIN MATCH ({variable}:Person)-[:KNOWS]->()-[:KNOWS]->() RETURN {variable}.name"
        )))
    };
    let first = explain("a");
    assert!(
        first.iter().flatten().any(|line| line.contains("_anon_0")),
        "the plan shows the first generated name: {first:?}"
    );
    db.execute("MATCH (n)-->() RETURN count(*)").unwrap();
    let second: Vec<Vec<String>> = explain("b")
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|line| line.replace("(b", "(a").replace("b.", "a."))
                .collect()
        })
        .collect();
    assert_eq!(second, first);
}

/// A variable no pattern binds is undefined, whatever it is named.
#[test]
fn an_undefined_variable_named_like_a_generated_one_is_an_error() {
    let db = graph();
    for query in [
        "MATCH (n:Person) RETURN _anon_5",
        "MATCH (n:Person) RETURN _anon_5.name",
    ] {
        let message = db.execute(query).unwrap_err().to_string();
        assert!(
            message.contains("Undefined variable '_anon_5'"),
            "GQL {query}: {message}"
        );
        #[cfg(feature = "cypher")]
        {
            let message = db.execute_cypher(query).unwrap_err().to_string();
            assert!(
                message.contains("Undefined variable '_anon_5'"),
                "Cypher {query}: {message}"
            );
        }
    }
}

/// An EXISTS subquery from a user variable named like a generated one checks
/// that node's edges.
#[test]
fn exists_from_a_user_variable_named_like_a_generated_one() {
    let db = graph();
    let query = "MATCH (_anon_x:Person) \
                 RETURN _anon_x.name, EXISTS { MATCH (_anon_x)-[:KNOWS]->(m) }";
    let want = vec![
        vec!["Alix".to_string(), "true".to_string()],
        vec!["Gus".to_string(), "false".to_string()],
    ];
    assert_eq!(rows(db.execute(query)), want, "GQL");
    #[cfg(feature = "cypher")]
    assert_eq!(rows(db.execute_cypher(query)), want, "Cypher");
}
