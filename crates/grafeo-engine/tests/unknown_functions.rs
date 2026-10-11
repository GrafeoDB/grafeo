//! A call to a function that does not exist is an error before any row is
//! read, in every query language that calls functions (GQL, Cypher,
//! SQL/PGQ), with the closest function name as a hint. Such a call used to
//! be null for every row, so a misspelled function in a WHERE clause quietly
//! dropped every row.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test unknown_functions
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// Two directories, the second with a file in it.
fn directories() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Directory {path: 'src/tests'}), \
         (:Directory {path: 'src/app'})-[:CONTAINS]->(:File {path: 'src/app/main.py'})",
    )
    .unwrap();
    db
}

/// The message of `result`'s error.
fn error_of(result: Result<QueryResult>) -> String {
    match result {
        Ok(rows) => panic!("expected an error, got {:?}", rows.rows()),
        Err(error) => error.to_string(),
    }
}

/// The single value of `query`'s single row.
fn scalar(db: &GrafeoDB, query: &str) -> Value {
    let result = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    result.rows()[0][0].clone()
}

#[test]
fn an_unknown_function_is_an_error_not_a_null() {
    let db = directories();
    let error = error_of(db.execute("MATCH (d:Directory) RETURN no_such_function(d.path) AS x"));
    assert!(
        error.contains("Unknown function 'no_such_function'"),
        "{error}"
    );
    assert!(
        !error.contains("Did you mean"),
        "no function is close to no_such_function: {error}"
    );
}

#[test]
fn a_misspelled_function_names_the_function_it_resembles() {
    let db = directories();
    let error = error_of(
        db.execute("MATCH (d:Directory) WHERE upperr(d.path) = 'SRC/APP' RETURN d.path AS path"),
    );
    assert!(error.contains("Unknown function 'upperr'"), "{error}");
    assert!(error.contains("Did you mean 'upper'?"), "{error}");

    let error = error_of(db.execute("MATCH (d:Directory) RETURN toUpperr(d.path) AS p"));
    assert!(error.contains("Did you mean 'toUpper'?"), "{error}");

    // An aggregate is a function too.
    let error = error_of(db.execute("MATCH (d:Directory) RETURN coutn(d) AS c"));
    assert!(error.contains("Did you mean 'count'?"), "{error}");
}

/// Queries that call `upperr` in each place a GQL query calls a function.
const GQL_CALLS: &[(&str, &str)] = &[
    (
        "WHERE",
        "MATCH (d:Directory) WHERE upperr(d.path) = 'SRC/APP' RETURN d.path",
    ),
    ("RETURN", "MATCH (d:Directory) RETURN upperr(d.path) AS p"),
    ("RETURN without MATCH", "RETURN upperr('src') AS p"),
    (
        "ORDER BY",
        "MATCH (d:Directory) RETURN d.path ORDER BY upperr(d.path)",
    ),
    (
        "element WHERE",
        "MATCH (d:Directory WHERE upperr(d.path) = 'SRC/APP') RETURN d.path",
    ),
    (
        "FILTER",
        "MATCH (d:Directory) FILTER upperr(d.path) = 'SRC/APP' RETURN d.path",
    ),
    ("LET", "MATCH (d:Directory) LET p = upperr(d.path) RETURN p"),
    (
        "WITH",
        "MATCH (d:Directory) WITH upperr(d.path) AS p RETURN p",
    ),
    (
        "argument of a known function",
        "MATCH (d:Directory) RETURN size(upperr(d.path)) AS n",
    ),
    (
        "list comprehension",
        "MATCH (d:Directory) RETURN [x IN [d.path] | upperr(x)] AS l",
    ),
    (
        "list predicate",
        "MATCH (d:Directory) WHERE any(x IN [d.path] WHERE upperr(x) = 'A') RETURN d.path",
    ),
    (
        "CASE",
        "MATCH (d:Directory) RETURN CASE WHEN d.path = 'src/app' THEN upperr(d.path) END AS c",
    ),
    (
        "aggregate argument",
        "MATCH (d:Directory) RETURN count(upperr(d.path)) AS c",
    ),
    (
        "grouping key",
        "MATCH (d:Directory) RETURN upperr(d.path) AS k, count(*) AS c",
    ),
    (
        "EXISTS subquery",
        "MATCH (d:Directory) WHERE EXISTS { MATCH (d)-[:CONTAINS]->(f) WHERE upperr(f.path) = 'A' } RETURN d.path",
    ),
    ("UNWIND", "UNWIND upperr(['a']) AS x RETURN x"),
    ("SET", "MATCH (d:Directory) SET d.shout = upperr(d.path)"),
    ("INSERT", "INSERT (:Directory {path: upperr('lib')})"),
];

#[test]
fn an_unknown_gql_function_is_an_error_wherever_it_is_called() {
    let db = directories();
    for (place, query) in GQL_CALLS {
        let error = error_of(db.execute(query));
        assert!(
            error.contains("Unknown function 'upperr'"),
            "{place}: {query}: {error}"
        );
    }
    // The failed writes changed nothing: the error comes before any row.
    assert_eq!(
        scalar(&db, "MATCH (d:Directory) RETURN count(d) AS n"),
        Value::Int64(2)
    );
    assert_eq!(
        scalar(
            &db,
            "MATCH (d:Directory) WHERE d.shout IS NOT NULL RETURN count(d) AS n"
        ),
        Value::Int64(0)
    );
}

#[test]
fn known_functions_are_found_in_any_case_and_aggregates_stay_known() {
    let db = directories();
    assert_eq!(
        scalar(
            &db,
            "MATCH (d:Directory) WHERE UPPER(d.path) = 'SRC/APP' RETURN Upper(d.path) AS p"
        ),
        Value::from("SRC/APP")
    );
    assert_eq!(
        scalar(&db, "MATCH (d:Directory) RETURN TOUPPER(min(d.path)) AS p"),
        Value::from("SRC/APP")
    );
    assert_eq!(
        scalar(
            &db,
            "MATCH (d:Directory) RETURN COUNT(DISTINCT toLower(d.path)) AS n"
        ),
        Value::Int64(2)
    );
}

#[cfg(feature = "cypher")]
#[test]
fn an_unknown_cypher_function_is_an_error_wherever_it_is_called() {
    let db = directories();
    let calls = [
        "MATCH (d:Directory) RETURN no_such_function(d.path) AS x",
        "MATCH (d:Directory) WHERE upperr(d.path) = 'SRC/APP' RETURN d.path",
        "MATCH (d:Directory) WITH upperr(d.path) AS p RETURN p",
        "MATCH (d:Directory) RETURN d.path ORDER BY upperr(d.path)",
        "UNWIND upperr(['a']) AS x RETURN x",
        "MATCH (d:Directory) RETURN [(d)-[:CONTAINS]->(f) | upperr(f.path)] AS l",
        "MATCH (d:Directory) RETURN collect(upperr(d.path)) AS l",
        "MATCH (d:Directory) SET d.shout = upperr(d.path)",
        "CREATE (:Directory {path: upperr('lib')})",
        "MERGE (d:Directory {path: 'lib'}) ON CREATE SET d.shout = upperr(d.path)",
    ];
    for query in calls {
        let error = error_of(db.execute_cypher(query));
        assert!(error.contains("Unknown function '"), "{query}: {error}");
    }
    let error = error_of(db.execute_cypher("MATCH (d:Directory) RETURN upperr(d.path) AS p"));
    assert!(
        error.contains("Unknown function 'upperr'") && error.contains("Did you mean 'upper'?"),
        "{error}"
    );
    assert_eq!(
        scalar(&db, "MATCH (d:Directory) RETURN count(d) AS n"),
        Value::Int64(2),
        "CREATE and MERGE wrote nothing"
    );
}

#[cfg(feature = "sql-pgq")]
#[test]
fn an_unknown_sql_pgq_function_is_an_error() {
    let db = directories();
    let error = error_of(db.execute_sql(
        "SELECT * FROM GRAPH_TABLE (MATCH (d:Directory) COLUMNS (upperr(d.path) AS p))",
    ));
    assert!(
        error.contains("Unknown function 'upperr'") && error.contains("Did you mean 'upper'?"),
        "{error}"
    );
    let error = error_of(db.execute_sql(
        "SELECT upperr(g.p) AS shout FROM GRAPH_TABLE (MATCH (d:Directory) COLUMNS (d.path AS p)) AS g",
    ));
    assert!(error.contains("Unknown function 'upperr'"), "{error}");
}

#[cfg(feature = "gremlin")]
#[test]
fn gremlin_steps_that_read_properties_still_run() {
    // Gremlin has no function calls of its own; its steps plan to the
    // functions `properties` and `property_values`, which stay known.
    let db = directories();
    let result = db
        .execute_gremlin("g.V().hasLabel('Directory').valueMap('path')")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    let result = db
        .execute_gremlin("g.V().hasLabel('Directory').values('path')")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
}
