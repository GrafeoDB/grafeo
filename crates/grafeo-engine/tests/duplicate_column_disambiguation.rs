//! Duplicate result column names (#371).
//!
//! Bindings read rows by column name, so two columns with the same name lose
//! data silently. A `QueryResult` with a repeated column name is rejected with
//! an error (eager and streaming paths), and unaliased expressions are named
//! after their source text so distinct expressions (`id(s)`, `id(t)`,
//! `count(a)`, `count(b)`) get distinct names without aliases.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Two `Person` nodes (ids 0 and 1) joined by a single `KNOWS` edge (id 0):
/// `(s {name:"s", age:30}) -[r:KNOWS]-> (t {name:"t", age:25})`.
fn one_edge() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let s = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("s".into())),
                ("age", Value::Int64(30)),
            ],
        )
        .unwrap();
    let t = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("t".into())),
                ("age", Value::Int64(25)),
            ],
        )
        .unwrap();
    session.create_edge(s, t, "KNOWS");
    db
}

// Unaliased id() calls get distinct, correctly valued columns.
#[test]
fn unaliased_id_calls_get_distinct_names() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (s)-[r]->(t) RETURN id(s), id(t), id(r)")
        .expect("R2a query should succeed with distinct names");
    assert_eq!(
        r.columns,
        vec!["id(s)", "id(t)", "id(r)"],
        "M2 must render function arguments so distinct bare expressions get distinct names"
    );
    assert_eq!(r.row_count(), 1);
    let row = &r.rows()[0];
    assert_eq!(row[0], Value::Int64(0), "id(s)");
    assert_eq!(row[1], Value::Int64(1), "id(t)");
    assert_eq!(row[2], Value::Int64(0), "id(r)");
}

// A repeated property column (a.name, a.name) is rejected.
#[test]
fn duplicate_property_column_is_rejected() {
    let db = one_edge();
    match db
        .session()
        .execute("MATCH (a:Person) RETURN a.name, a.name")
    {
        Ok(r) => panic!(
            "R2b expected a fail-closed error, got columns {:?} (silent collapse at FFI)",
            r.columns
        ),
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("duplicate column"),
            "R2b error should name the duplicate column, got: {e}"
        ),
    }
}

// A repeated alias (AS x, AS x) is rejected.
#[test]
fn duplicate_alias_is_rejected() {
    let db = one_edge();
    match db
        .session()
        .execute("MATCH (s)-[r]->(t) RETURN id(s) AS x, id(t) AS x")
    {
        Ok(r) => panic!(
            "R2c expected a fail-closed error for the duplicate alias, got columns {:?}",
            r.columns
        ),
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("duplicate column"),
            "R2c error should name the duplicate column, got: {e}"
        ),
    }
}

// Control: distinct aliases are accepted.
#[test]
fn distinct_aliases_are_accepted() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (s)-[r]->(t) RETURN id(s) AS sid, id(t) AS tid")
        .expect("R2d distinct-alias query must succeed unchanged");
    assert_eq!(r.columns, vec!["sid", "tid"]);
    assert_eq!(r.row_count(), 1);
    let row = &r.rows()[0];
    assert_eq!(row[0], Value::Int64(0), "sid");
    assert_eq!(row[1], Value::Int64(1), "tid");
}

// Different expressions get different names, or the query fails.
#[test]
fn distinct_expressions_do_not_collide() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (a:Person) RETURN a.age + 1, -a.age")
        .expect("distinct fallthrough expressions should not collide after M2");
    assert_eq!(r.columns.len(), 2);
    assert_ne!(
        r.columns[0], r.columns[1],
        "two distinct expressions must not share a column name"
    );
    assert_ne!(r.columns[0], "expr");
    assert_ne!(r.columns[1], "expr");
    // The unary negation renders faithfully (no embedded literal).
    assert_eq!(r.columns[1], "-a.age");
}

#[test]
fn unaliased_aggregates_get_distinct_names() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN count(a), count(b)")
        .expect("two unaliased aggregates must not collide");
    assert_eq!(r.columns, vec!["count(a)", "count(b)"]);
    assert_eq!(r.rows()[0], vec![Value::Int64(1), Value::Int64(1)]);

    let r = db
        .session()
        .execute("MATCH (a:Person) RETURN count(*), count(DISTINCT a)")
        .expect("count(*) and count(DISTINCT a) must not collide");
    assert_eq!(r.columns, vec!["count(*)", "count(DISTINCT a)"]);
}

// SPARQL: `SELECT ?s ?s` is rejected; a repeated `AS ?x` alias fails at parse time.
#[cfg(feature = "sparql")]
fn rdf_db_one_triple() -> GrafeoDB {
    use grafeo_engine::config::{Config, GraphModel};
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    db.session()
        .execute_sparql("INSERT DATA { <urn:s> <urn:p> <urn:o> }")
        .unwrap();
    db
}

#[cfg(feature = "sparql")]
#[test]
fn sparql_duplicate_variable_is_rejected() {
    let db = rdf_db_one_triple();
    match db
        .session()
        .execute_sparql("SELECT ?s ?s WHERE { ?s ?p ?o }")
    {
        Ok(r) => panic!(
            "R2e expected fail-closed error for `SELECT ?s ?s`, got columns {:?}",
            r.columns
        ),
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("duplicate column"),
            "R2e (M1) error should name the duplicate column, got: {e}"
        ),
    }
}

#[cfg(feature = "sparql")]
#[test]
fn sparql_duplicate_alias_is_rejected_at_parse() {
    let db = rdf_db_one_triple();
    match db
        .session()
        .execute_sparql("SELECT (?s AS ?x) (?o AS ?x) WHERE { ?s ?p ?o }")
    {
        Ok(r) => panic!(
            "R2e expected parse rejection for duplicate alias, got columns {:?}",
            r.columns
        ),
        Err(e) => {
            let msg = e.to_string().to_lowercase();
            assert!(
                msg.contains("duplicate") && msg.contains("alias") || msg.contains("fresh"),
                "R2e (D2) parse error should flag the duplicate alias, got: {e}"
            );
        }
    }
}

#[cfg(feature = "sparql")]
#[test]
fn sparql_distinct_variables_are_accepted() {
    let db = rdf_db_one_triple();
    let r = db
        .session()
        .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
        .expect("conformant SPARQL SELECT must succeed");
    assert_eq!(
        r.columns,
        vec!["s"],
        "conformant variable headers are untouched"
    );
    assert_eq!(r.row_count(), 1);
}

// Streaming: duplicate column names are rejected when the stream opens.
#[test]
fn streaming_rejects_duplicate_columns_at_open() {
    let db = one_edge();
    match db.execute_streaming("MATCH (a:Person) RETURN a.name, a.name") {
        Ok(_) => {
            panic!("R2f expected the streaming guard to reject duplicate columns at stream-open")
        }
        Err(e) => assert!(
            e.to_string().to_lowercase().contains("duplicate column"),
            "R2f streaming error should name the duplicate column, got: {e}"
        ),
    }
    // A distinct-name projection must still open on the streaming path.
    assert!(
        db.execute_streaming("MATCH (s)-[r]->(t) RETURN id(s), id(t), id(r)")
            .is_ok(),
        "distinct streaming projection must open"
    );
}
