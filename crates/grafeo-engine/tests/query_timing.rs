//! Every statement reports `execution_time_ms`, the time of the whole call
//! (#461): in every query language, with and without parameters, for
//! `EXPLAIN`, `PROFILE`, `SHOW`, schema and session commands too.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test query_timing
//! ```

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

fn timed(what: &str, result: Result<QueryResult>) {
    let result = result.unwrap_or_else(|error| panic!("{what}: {error}"));
    let ms = result
        .execution_time_ms
        .unwrap_or_else(|| panic!("{what} reports no time"));
    assert!(ms >= 0.0, "{what}: {ms}");
}

fn params() -> HashMap<String, Value> {
    HashMap::from([("x".to_string(), Value::Int64(1))])
}

#[test]
fn gql_statements_of_every_kind_report_their_time() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    timed("INSERT", session.execute("INSERT (:Person {name: 'Alix'})"));
    timed("MATCH", session.execute("MATCH (p:Person) RETURN p.name"));
    timed(
        "parameters",
        session.execute_with_params("RETURN $x AS x", params()),
    );
    timed(
        "EXPLAIN",
        session.execute("EXPLAIN MATCH (p:Person) RETURN p.name"),
    );
    timed(
        "PROFILE",
        session.execute("PROFILE MATCH (p:Person) RETURN p.name"),
    );
    timed("SHOW", session.execute("SHOW GRAPHS"));
    timed("DDL", session.execute("CREATE GRAPH berlin"));
    timed(
        "session command",
        session.execute("SESSION SET GRAPH berlin"),
    );
    timed("database facade", db.execute("RETURN 1 AS x"));
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_reports_its_time() {
    let session = GrafeoDB::new_in_memory().session();
    timed("Cypher", session.execute_cypher("RETURN 1 AS x"));
    timed(
        "Cypher with parameters",
        session.execute_cypher_with_params("RETURN $x AS x", params()),
    );
    timed(
        "execute_language",
        session.execute_language("RETURN 1 AS x", "cypher", None),
    );
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_reports_its_time() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:City {name: 'Prague'})").unwrap();
    let session = db.session();
    let query = "SELECT * FROM GRAPH_TABLE (MATCH (c:City) COLUMNS (c.name AS name))";
    timed("SQL/PGQ", session.execute_sql(query));
    timed(
        "SQL/PGQ with parameters",
        session.execute_sql_with_params(query, HashMap::new()),
    );
}

#[cfg(feature = "gremlin")]
#[test]
fn gremlin_reports_its_time() {
    let session = GrafeoDB::new_in_memory().session();
    timed("Gremlin", session.execute_gremlin("g.V().count()"));
    timed(
        "Gremlin with parameters",
        session.execute_gremlin_with_params("g.V().count()", HashMap::new()),
    );
}

#[cfg(feature = "graphql")]
#[test]
fn graphql_reports_its_time() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let session = db.session();
    timed("GraphQL", session.execute_graphql("{ Person { name } }"));
    timed(
        "GraphQL with parameters",
        session.execute_graphql_with_params("{ Person { name } }", HashMap::new()),
    );
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_reports_its_time() {
    let session = GrafeoDB::new_in_memory().session();
    timed(
        "SPARQL update",
        session.execute_sparql(r#"INSERT DATA { <http://ex.org/a> <http://ex.org/p> "1" . }"#),
    );
    timed(
        "SPARQL",
        session.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }"),
    );
    timed(
        "SPARQL with parameters",
        session.execute_sparql_with_params("SELECT ?s WHERE { ?s ?p ?o }", HashMap::new()),
    );
}
