//! Oracle suite: duplicate output-column disambiguation.
//!
//! A `QueryResult` stores its rows positionally and losslessly, but it is
//! consumed *by column name* across the FFI bindings (the PyO3 dict, the Node/C
//! JSON object, or any binding that keys rows into a structure forbidding
//! duplicate keys). When two output columns share a name those bindings lose
//! data silently — last-write-wins in some, a dropped whole row in others.
//!
//! Two engine moves close this:
//!   * a fail-closed result-column uniqueness invariant that is leg-agnostic
//!     and holds on BOTH the eager and the streaming paths: a result with a
//!     repeated column name is never built; a structured error is raised
//!     instead.
//!   * openCypher-faithful argument rendering in `expression_to_string`, so
//!     distinct bare expressions (`id(s)`, `id(t)`, `id(r)`) receive distinct
//!     names and the common case needs no alias.
//!
//! Within-projection duplicate-alias rejection in the SPARQL parser is
//! exercised here end-to-end and unit-tested in `grafeo-adapters`.
//!
//! Run: `cargo test -p grafeo-engine --features full --test duplicate_column_disambiguation`

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

// ===========================================================================
// Duplicate bare id() calls get three DISTINCT, correctly-valued columns.
//   PRE-FIX: all three columns named "id(...)" (arguments discarded).
// ===========================================================================
#[test]
fn r2a_duplicate_bare_id_gets_distinct_faithful_names() {
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

// ===========================================================================
// Duplicate property accessor (a.name, a.name) fails closed.
//   PRE-FIX: accepted, columns ["a.name", "a.name"], one key survives at FFI.
// ===========================================================================
#[test]
fn r2b_duplicate_property_fails_closed() {
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

// ===========================================================================
// Explicit duplicate alias (AS x, AS x) fails closed.
//   PRE-FIX: WRONGLY ACCEPTED, columns ["x", "x"]; surviving value = id(t)=1.
// ===========================================================================
#[test]
fn r2c_explicit_duplicate_alias_fails_closed() {
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

// ===========================================================================
// Distinct-alias control: distinct aliases still succeed (must not regress).
// ===========================================================================
#[test]
fn r2d_distinct_alias_control_unchanged() {
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

// ===========================================================================
// Distinct fallthrough expressions get distinct names, or else fail closed.
//   PRE-FIX: both render to the constant "expr" and collide.
// ===========================================================================
#[test]
fn rb5_distinct_fallthrough_expressions_do_not_collide() {
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

// ===========================================================================
// Aggregate-exclusion control — argument rendering must NOT touch aggregate
// headers (they use a separate naming path). An un-aliased count stays unchanged.
// ===========================================================================
#[test]
fn aggregate_header_unchanged_by_m2() {
    let db = one_edge();
    let r = db
        .session()
        .execute("MATCH (a:Person) RETURN count(a)")
        .expect("aggregate query should succeed");
    assert_eq!(r.columns.len(), 1);
    // If argument rendering had leaked into aggregates the header would render
    // the argument as "count(a)"; the separate aggregate path keeps the
    // collapsed form.
    assert_ne!(r.columns[0], "count(a)", "aggregates are out of M2 scope");
    assert!(r.columns[0].contains("count"));
}

// ===========================================================================
// SPARQL: `SELECT ?s ?s` fails closed via the leg-agnostic uniqueness invariant;
//   the non-conformant `(?s AS ?x)(?o AS ?x)` is rejected at parse time; the
//   conformant `SELECT ?s` variable header is unchanged.
//   PRE-FIX: both degenerate forms collapse to ["s","s"] / ["x","x"].
// ===========================================================================
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
fn r2e_sparql_duplicate_variable_fails_closed() {
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
fn r2e_sparql_duplicate_alias_rejected_at_parse() {
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
fn r2e_sparql_conformant_variable_header_unchanged() {
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

// ===========================================================================
// Streaming-path guard: the same duplicate-name query through the lazy/streaming
//   path fails closed at stream-open, before any row; a distinct-name projection
//   still opens. PRE-FIX: the stream opens and the collapse happens per-row in
//   the binding.
// ===========================================================================
#[test]
fn r2f_streaming_guard_rejects_duplicate_columns_at_open() {
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
