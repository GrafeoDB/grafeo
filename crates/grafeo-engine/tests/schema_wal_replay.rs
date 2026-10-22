//! WAL replay / reopen coverage for CREATE SCHEMA, DROP SCHEMA, and
//! schema-scoped CREATE GRAPH (ISO/IEC 39075 catalog hierarchy).
//!
//! Complements `schema_hierarchy.rs` (in-memory behavior) and `snapshot.rs`
//! (snapshot round-trip) by exercising the on-disk path: WAL append + reopen.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test schema_wal_replay
//! ```

#![cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]

use grafeo_engine::GrafeoDB;

#[test]
fn wal_replay_preserves_schemas_after_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("schemas.grafeo");

    {
        let db = GrafeoDB::open(&path).expect("open");
        let s = db.session();
        s.execute("CREATE SCHEMA reporting").unwrap();
        s.execute("CREATE SCHEMA social").unwrap();
        db.close().expect("close");
    }

    let db = GrafeoDB::open(&path).expect("reopen");
    let s = db.session();
    let schemas = s.execute("SHOW SCHEMAS").unwrap();
    assert_eq!(
        schemas.row_count(),
        2,
        "both schemas should be recovered from WAL after reopen"
    );
}

#[test]
fn wal_replay_preserves_cross_schema_isolation_after_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("iso.grafeo");

    {
        let db = GrafeoDB::open(&path).expect("open");
        let s = db.session();
        s.execute("CREATE SCHEMA a").unwrap();
        s.execute("CREATE SCHEMA b").unwrap();

        s.execute("SESSION SET SCHEMA a").unwrap();
        s.execute("INSERT (:Item {owner: 'a'})").unwrap();

        s.execute("SESSION SET SCHEMA b").unwrap();
        s.execute("INSERT (:Item {owner: 'b'})").unwrap();
        db.close().expect("close");
    }

    let db = GrafeoDB::open(&path).expect("reopen");
    let s = db.session();

    s.execute("SESSION SET SCHEMA a").unwrap();
    let result = s.execute("MATCH (n:Item) RETURN n.owner").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "schema a should have exactly 1 node after reopen"
    );

    s.execute("SESSION SET SCHEMA b").unwrap();
    let result = s.execute("MATCH (n:Item) RETURN n.owner").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "schema b should have exactly 1 node after reopen (no cross-schema bleed)"
    );
}

#[test]
fn wal_replay_preserves_drop_schema_after_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("drop.grafeo");

    {
        let db = GrafeoDB::open(&path).expect("open");
        let s = db.session();
        s.execute("CREATE SCHEMA keep").unwrap();
        s.execute("CREATE SCHEMA discard").unwrap();
        s.execute("DROP SCHEMA discard").unwrap();
        db.close().expect("close");
    }

    let db = GrafeoDB::open(&path).expect("reopen");
    let s = db.session();
    let schemas = s.execute("SHOW SCHEMAS").unwrap();
    assert_eq!(
        schemas.row_count(),
        1,
        "dropped schema should not reappear after reopen"
    );

    // The surviving schema must be usable
    s.execute("SESSION SET SCHEMA keep").unwrap();
    assert_eq!(s.current_schema(), Some("keep".to_string()));

    // The dropped one must not be settable
    let result = s.execute("SESSION SET SCHEMA discard");
    assert!(
        result.is_err(),
        "dropped schema should not be reachable after reopen"
    );
}

#[test]
fn wal_replay_preserves_named_graph_within_schema() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ng.grafeo");

    {
        let db = GrafeoDB::open(&path).expect("open");
        let s = db.session();
        s.execute("CREATE SCHEMA reports").unwrap();
        s.execute("SESSION SET SCHEMA reports").unwrap();
        s.execute("CREATE GRAPH quarterly").unwrap();
        s.execute("SESSION SET GRAPH quarterly").unwrap();
        s.execute("INSERT (:Row {q: 1})").unwrap();
        db.close().expect("close");
    }

    let db = GrafeoDB::open(&path).expect("reopen");
    let s = db.session();
    s.execute("SESSION SET SCHEMA reports").unwrap();

    let graphs = s.execute("SHOW GRAPHS").unwrap();
    assert_eq!(
        graphs.row_count(),
        1,
        "named graph within schema should survive reopen"
    );

    s.execute("SESSION SET GRAPH quarterly").unwrap();
    let rows = s.execute("MATCH (n:Row) RETURN n.q").unwrap();
    assert_eq!(
        rows.row_count(),
        1,
        "data inside the schema-scoped named graph should survive reopen"
    );
}
