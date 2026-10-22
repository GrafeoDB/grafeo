//! SPARQL orders solutions before it projects them, and evaluates every
//! aggregate of a HAVING, wherever it stands (SPARQL 1.1, 18.2.4 and 18.2.5).
//! `ORDER BY ?s` of a variable the SELECT leaves out failed with an internal
//! error ("Variable 's' not found"), and an aggregate inside a function call
//! of a HAVING (`CONTAINS(GROUP_CONCAT(...), ...)`, `ABS(SUM(?x))`) evaluated
//! to nothing, so every group was dropped; one the SELECT did not compute was
//! not computed, and one of another argument read the projected one.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_order_by_and_having
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_engine::GrafeoDB;

/// Alix (30, Person), Gus (25, Person) and Prague (City, 1300).
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        r#"INSERT DATA {
            <http://ex.org/a> a <http://ex.org/Person> ; <http://ex.org/name> "Alix" ; <http://ex.org/v> 30 .
            <http://ex.org/b> a <http://ex.org/Person> ; <http://ex.org/name> "Gus" ; <http://ex.org/v> 25 .
            <http://ex.org/c> a <http://ex.org/City> ; <http://ex.org/name> "Prague" ; <http://ex.org/v> 1300 .
        }"#,
    )
    .unwrap();
    db
}

/// The rows of `query`, each value as text (a literal without its quotes),
/// and its columns.
fn run(db: &GrafeoDB, query: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let result = db
        .execute_sparql(query)
        .unwrap_or_else(|e| panic!("`{query}`: {e}"));
    let rows = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value.to_string().trim_matches('"').to_string())
                .collect()
        })
        .collect();
    (result.columns.clone(), rows)
}

fn texts(values: &[&str]) -> Vec<Vec<String>> {
    values
        .iter()
        .map(|value| vec![(*value).to_string()])
        .collect()
}

/// `ORDER BY ?s` orders by the subject, which the SELECT leaves out: the
/// names come in subject order, and only `name` is a column.
#[test]
fn order_by_a_variable_the_select_leaves_out() {
    let db = people();
    let (columns, rows) = run(
        &db,
        "SELECT ?name WHERE { ?s <http://ex.org/name> ?name } ORDER BY DESC(?s)",
    );
    assert_eq!(columns, ["name"]);
    assert_eq!(rows, texts(&["Prague", "Gus", "Alix"]));
    let (_, ascending) = run(
        &db,
        "SELECT ?name WHERE { ?s <http://ex.org/name> ?name } ORDER BY ?s",
    );
    assert_eq!(ascending, texts(&["Alix", "Gus", "Prague"]));
}

/// With DISTINCT, an alias and a LIMIT after it.
#[test]
fn order_by_an_unselected_variable_with_distinct_alias_and_limit() {
    let db = people();
    let (columns, rows) = run(
        &db,
        "SELECT DISTINCT (STRLEN(?name) AS ?len) WHERE { ?s <http://ex.org/name> ?name } \
         ORDER BY ?s LIMIT 2",
    );
    assert_eq!(columns, ["len"]);
    assert_eq!(rows, texts(&["4", "3"]));
}

/// An aggregate inside a function call of a HAVING is the group's aggregate.
#[test]
fn an_aggregate_inside_a_function_of_having() {
    let db = people();
    let (_, rows) = run(
        &db,
        r#"SELECT ?type (GROUP_CONCAT(?name; SEPARATOR=",") AS ?names)
           WHERE { ?s a ?type . ?s <http://ex.org/name> ?name }
           GROUP BY ?type
           HAVING (CONTAINS(GROUP_CONCAT(?name; SEPARATOR=","), "Gus"))"#,
    );
    assert_eq!(rows.len(), 1, "the Person group: {rows:?}");
    assert!(rows[0][0].contains("Person"), "{rows:?}");
    let (_, rows) = run(
        &db,
        "SELECT ?type (SUM(?v) AS ?total) WHERE { ?s a ?type . ?s <http://ex.org/v> ?v } \
         GROUP BY ?type HAVING (ABS(SUM(?v)) > 100)",
    );
    assert_eq!(rows.len(), 1, "the City group: {rows:?}");
    assert!(rows[0][0].contains("City"), "{rows:?}");
}

/// An aggregate the SELECT does not compute is computed for the HAVING, and
/// is not a column.
#[test]
fn an_aggregate_only_the_having_reads() {
    let db = people();
    let (columns, rows) = run(
        &db,
        "SELECT ?type WHERE { ?s a ?type } GROUP BY ?type HAVING (COUNT(?s) > 1)",
    );
    assert_eq!(columns, ["type"]);
    assert_eq!(rows.len(), 1, "the Person group: {rows:?}");
    assert!(rows[0][0].contains("Person"), "{rows:?}");
}

/// The HAVING's `SUM(?n)` is its own, not the projected `SUM(?v)`: names
/// have no number to sum, so no group passes.
#[test]
fn an_aggregate_of_another_argument_is_its_own() {
    let db = people();
    let (_, rows) = run(
        &db,
        "SELECT ?type (SUM(?v) AS ?total) \
         WHERE { ?s a ?type . ?s <http://ex.org/v> ?v . ?s <http://ex.org/name> ?n } \
         GROUP BY ?type HAVING (SUM(?v) > 0 && COUNT(DISTINCT ?n) > 1)",
    );
    assert_eq!(rows.len(), 1, "the Person group has two names: {rows:?}");
    assert!(rows[0][0].contains("Person"), "{rows:?}");
}
