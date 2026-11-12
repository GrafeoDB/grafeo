//! Every path that runs a statement plans inside its memory scope (#674): a
//! session's planner checks it in a debug build, so a path that enters no
//! scope panics here instead of leaving its operators' memory uncounted. The
//! database's own calls and each language through a session: a run, a
//! plan-cache hit, `EXPLAIN` and `PROFILE`, with parameters (prepared),
//! streamed, at an epoch, a stored procedure's body planned when the call
//! runs, correlated subqueries, and a SPARQL `EXISTS` planned per row while
//! the rows are pulled.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main statement_memory_scope::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// A database with a few people and their cities.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix', city: 'Amsterdam'}), \
         (:Person {name: 'Gus', city: 'Berlin'}), \
         (:Person {name: 'Vincent', city: 'Amsterdam'})",
    )
    .unwrap();
    db
}

/// A sort, an aggregate, a distinct and a set operation, each as a run, a
/// plan-cache hit, an `EXPLAIN` and a `PROFILE`; with parameters; streamed;
/// and a stored procedure's body, planned when the call runs.
#[cfg(feature = "gql")]
#[test]
fn gql_plans_inside_the_statement_scope() {
    let db = people();
    let session = db.session();
    for query in [
        "MATCH (p:Person) RETURN p.name ORDER BY p.name",
        "MATCH (p:Person) RETURN p.city, count(p) AS people",
        "MATCH (p:Person) RETURN DISTINCT p.city",
        "MATCH (p:Person) RETURN p.name EXCEPT MATCH (p:Person {city: 'Berlin'}) RETURN p.name",
    ] {
        session.execute(query).unwrap();
        session.execute(query).unwrap();
        session.execute(&format!("EXPLAIN {query}")).unwrap();
        session.execute(&format!("PROFILE {query}")).unwrap();
    }
    let params = std::collections::HashMap::from([("city".to_string(), Value::from("Amsterdam"))]);
    // Twice: the second call fills the prepared plan.
    for _ in 0..2 {
        let found = session
            .execute_with_params(
                "MATCH (p:Person) WHERE p.city = $city RETURN p.name ORDER BY p.name",
                params.clone(),
            )
            .unwrap();
        assert_eq!(found.rows().len(), 2);
    }
    let streamed = session
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(streamed.rows().len(), 3);
    {
        session
            .execute(
                "CREATE PROCEDURE cities() RETURNS (city STRING) AS \
                 { MATCH (p:Person) RETURN DISTINCT p.city AS city ORDER BY city }",
            )
            .unwrap();
        let called = session
            .execute("CALL cities() YIELD city RETURN city")
            .unwrap();
        assert_eq!(called.rows().len(), 2);
    }
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_plans_inside_the_statement_scope() {
    let db = people();
    let session = db.session();
    for query in [
        "MATCH (p:Person) RETURN p.name ORDER BY p.name",
        "MATCH (p:Person) RETURN p.city, count(p) AS people",
        "MATCH (p:Person) RETURN DISTINCT p.city",
    ] {
        session.execute_cypher(query).unwrap();
        session.execute_cypher(&format!("EXPLAIN {query}")).unwrap();
        session.execute_cypher(&format!("PROFILE {query}")).unwrap();
    }
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_plans_inside_the_statement_scope() {
    let db = people();
    let rows = db
        .session()
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name)) ORDER BY name",
        )
        .unwrap();
    assert_eq!(rows.rows().len(), 3);
}

#[cfg(feature = "gremlin")]
#[test]
fn gremlin_plans_inside_the_statement_scope() {
    let db = people();
    let session = db.session();
    session
        .execute_gremlin("g.V().hasLabel('Person').values('name').order()")
        .unwrap();
    session
        .execute_gremlin("g.V().hasLabel('Person').values('city').dedup()")
        .unwrap();
}

#[cfg(feature = "graphql")]
#[test]
fn graphql_plans_inside_the_statement_scope() {
    let db = people();
    db.session()
        .execute_graphql("{ Person { name city } }")
        .unwrap();
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_plans_inside_the_statement_scope() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        "INSERT DATA { <http://ex.org/alix> <http://ex.org/lives> \"Amsterdam\" . \
         <http://ex.org/gus> <http://ex.org/lives> \"Berlin\" }",
    )
    .unwrap();
    for query in [
        "SELECT ?s ?city WHERE { ?s <http://ex.org/lives> ?city } ORDER BY ?city",
        "SELECT ?city (COUNT(?s) AS ?n) WHERE { ?s <http://ex.org/lives> ?city } GROUP BY ?city",
        "SELECT DISTINCT ?city WHERE { ?s <http://ex.org/lives> ?city }",
    ] {
        db.execute_sparql(query).unwrap();
        db.execute_sparql(&format!("EXPLAIN {query}")).unwrap();
        db.session().execute_sparql(query).unwrap();
    }
}

/// The database's own calls run through a session of their own: a run, a
/// plan-cache hit, with parameters, at an epoch, streamed, and correlated
/// subqueries (`EXISTS`, `CALL`, `OPTIONAL MATCH`).
#[cfg(feature = "gql")]
#[test]
fn database_calls_plan_inside_the_statement_scope() {
    let db = people();
    for query in [
        "MATCH (p:Person) RETURN p.name ORDER BY p.name",
        "MATCH (p:Person) WHERE EXISTS { MATCH (q:Person) WHERE q.city = p.city AND q.name <> p.name } \
         RETURN p.name ORDER BY p.name",
        "MATCH (p:Person) CALL { WITH p MATCH (q:Person) WHERE q.city = p.city RETURN count(q) AS n } \
         RETURN p.name, n ORDER BY p.name",
        "MATCH (p:Person) OPTIONAL MATCH (q:Person) WHERE q.city = p.city AND q.name <> p.name \
         RETURN p.name, q.name ORDER BY p.name",
    ] {
        db.execute(query).unwrap();
        db.execute(query).unwrap();
    }
    let params = std::collections::HashMap::from([("city".to_string(), Value::from("Berlin"))]);
    let found = db
        .execute_with_params(
            "MATCH (p:Person) WHERE p.city = $city RETURN DISTINCT p.name",
            params,
        )
        .unwrap();
    assert_eq!(found.rows().len(), 1);
    let epoch = db.current_epoch();
    db.execute_at_epoch("MATCH (p:Person) RETURN p.city, count(p) AS n", epoch)
        .unwrap();
    let streamed = db
        .execute_streaming("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(streamed.rows().len(), 3);
}

/// A SPARQL `EXISTS` that reads the row inside its pattern is planned per
/// row while the rows are pulled: in the statement's memory.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn a_sparql_exists_planned_per_row_plans_inside_the_statement_scope() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        "PREFIX : <urn:> INSERT DATA { :alix :t :k . :gus :t :k . :alix :p :amsterdam . \
         :vincent :q :berlin }",
    )
    .unwrap();
    // An OPTIONAL that reads the row inside the pattern: no run without the
    // row answers it.
    let query = "PREFIX : <urn:> SELECT ?s WHERE { ?s :t :k \
                 FILTER EXISTS { :vincent :q ?o OPTIONAL { ?s :p ?y } } } ORDER BY ?s";
    let rows = db.execute_sparql(query).unwrap();
    assert_eq!(rows.rows().len(), 2);
    let profile = db.execute_sparql(&format!("PROFILE {query}")).unwrap();
    assert!(
        format!("{:?}", profile.rows()).contains("ExistsPerRow"),
        "the test runs per row: {:?}",
        profile.rows()
    );
    db.session().execute_sparql(query).unwrap();
}
