//! Regression tests for [issue #327](https://github.com/GrafeoDB/grafeo/issues/327):
//! Session direct LPG writes are not durable through a WAL replay.
//!
//! `Session::create_node_with_props`, `Session::create_edge_with_props`,
//! `Session::set_node_property`, and `Session::set_edge_property` call
//! `active_lpg_store()` directly without routing through `WalGraphStore`,
//! so writes via these APIs never reach the WAL, and a database that
//! recovers from its WAL (a single-file database that crashed before its
//! next checkpoint) loses them.
//!
//! The equivalent `GrafeoDB::create_node_with_props`,
//! `GrafeoDB::create_edge_with_props`, etc. log
//! `WalRecord::CreateNode` / `SetNodeProperty` / `CreateEdge` /
//! `SetEdgeProperty` correctly. See `crates/grafeo-engine/src/database/crud.rs`
//! for the WAL-correct path and `crates/grafeo-engine/src/session/mod.rs`
//! around `create_node_with_props` for the bug.
//!
//! Each test:
//! 1. opens a single-file database in a child process,
//! 2. performs the session-direct write,
//! 3. exits without `close()`, so nothing is checkpointed,
//! 4. reopens, which replays the sidecar WAL, and
//! 5. asserts the data survived.
//!
//! Tests verify the fix for #327 and run by default:
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test session_wal_durability
//! ```

#![allow(missing_docs)]

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod common;

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod session_wal_durability {
    use super::common::replay::reopened_after_crash;
    use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use std::path::Path;

    fn open(path: &Path) -> GrafeoDB {
        let config = Config::persistent(path).with_storage_format(StorageFormat::SingleFile);
        GrafeoDB::with_config(config).expect("open")
    }

    /// The id returned by `query`, which returns one row with one id.
    fn only_id(db: &GrafeoDB, query: &str) -> u64 {
        let result = db.execute(query).unwrap();
        assert_eq!(result.row_count(), 1, "{query}");
        match &result.rows()[0][0] {
            Value::Int64(id) => u64::try_from(*id).unwrap(),
            other => panic!("{query}: expected an id, got {other:?}"),
        }
    }

    #[test]
    fn session_create_node_with_props_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_create_node_with_props_survives_reopen",
            open,
            |db| {
                let mut session = db.session();
                session.begin_transaction().expect("begin");
                session
                    .create_node_with_props(&["Probe"], [("name", Value::String("Alix".into()))])
                    .expect("create node");
                session.commit().expect("commit");
            },
        );
        assert_eq!(
            db.node_count(),
            1,
            "node created via Session::create_node_with_props was lost on reopen"
        );
    }

    #[test]
    fn session_create_edge_with_props_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_create_edge_with_props_survives_reopen",
            open,
            |db| {
                // Anchor nodes via the WAL-correct DB-direct path so we know the
                // nodes themselves survive; the test isolates the edge bug.
                let alix = db.create_node(&["Person"]).unwrap();
                let gus = db.create_node(&["Person"]).unwrap();

                let mut session = db.session();
                session.begin_transaction().expect("begin");
                session
                    .create_edge_with_props(alix, gus, "KNOWS", [("since", Value::Int64(2026))])
                    .expect("create edge");
                session.commit().expect("commit");
            },
        );
        assert_eq!(
            db.node_count(),
            2,
            "anchor nodes lost (DB-direct path should always survive)"
        );
        assert_eq!(
            db.edge_count(),
            1,
            "edge created via Session::create_edge_with_props was lost on reopen"
        );
    }

    #[test]
    fn session_set_node_property_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_set_node_property_survives_reopen",
            open,
            |db| {
                // Create node via DB-direct path so the node itself is durable;
                // the test isolates the property bug.
                let alix = db.create_node(&["Person"]).unwrap();
                assert_eq!(alix, NodeId::new(0), "the first node of a new database");

                let mut session = db.session();
                session.begin_transaction().expect("begin");
                session
                    .set_node_property(alix, "name", Value::String("Alix".into()))
                    .expect("set property");
                session.commit().expect("commit");
            },
        );
        // The writer got id 0 (checked in the child); replay keeps it.
        let alix = NodeId::new(0);
        assert_eq!(
            NodeId::new(only_id(&db, "MATCH (p:Person) RETURN id(p)")),
            alix,
            "replay changed the id of the node"
        );
        let node = db.get_node(alix).expect("anchor node lost on reopen");
        let name_key = PropertyKey::from("name");
        assert_eq!(
            node.properties.get(&name_key).cloned(),
            Some(Value::String("Alix".into())),
            "property set via Session::set_node_property was lost on reopen"
        );
    }

    #[test]
    fn session_set_edge_property_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_set_edge_property_survives_reopen",
            open,
            |db| {
                // Anchor edge via DB-direct path; isolate the property bug.
                let alix = db.create_node(&["Person"]).unwrap();
                let gus = db.create_node(&["Person"]).unwrap();
                let edge_id = db.create_edge(alix, gus, "KNOWS").unwrap();
                assert_eq!(edge_id, EdgeId::new(0), "the first edge of a new database");

                let mut session = db.session();
                session.begin_transaction().expect("begin");
                session
                    .set_edge_property(edge_id, "since", Value::Int64(2026))
                    .expect("set edge property");
                session.commit().expect("commit");
            },
        );
        // The writer got id 0 (checked in the child); replay keeps it.
        let edge_id = EdgeId::new(0);
        assert_eq!(
            EdgeId::new(only_id(&db, "MATCH ()-[r:KNOWS]->() RETURN id(r)")),
            edge_id,
            "replay changed the id of the edge"
        );
        let edge = db.get_edge(edge_id).expect("anchor edge lost on reopen");
        let since_key = PropertyKey::from("since");
        assert_eq!(
            edge.properties.get(&since_key).cloned(),
            Some(Value::Int64(2026)),
            "property set via Session::set_edge_property was lost on reopen"
        );
    }

    /// Cypher path is the durability oracle: the same shape of write through
    /// `db.execute_language(..., "cypher", ...)` MUST survive reopen.
    /// If this test ever fails, the bug is broader than #327.
    #[test]
    fn cypher_create_node_with_props_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::cypher_create_node_with_props_survives_reopen",
            open,
            |db| {
                db.execute_language("CREATE (:Probe {name: 'cypher'});", "cypher", None)
                    .expect("cypher create");
            },
        );
        assert_eq!(
            db.node_count(),
            1,
            "Cypher-created node was lost on reopen (oracle test failed; bug is broader than #327)"
        );
    }

    #[test]
    fn session_create_node_no_props_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_create_node_no_props_survives_reopen",
            open,
            |db| {
                let mut session = db.session();
                session.begin_transaction().expect("begin");
                session.create_node(&["Probe"]).unwrap();
                session.commit().expect("commit");
            },
        );
        assert_eq!(
            db.node_count(),
            1,
            "node from Session::create_node was lost on reopen"
        );
    }

    #[test]
    fn session_create_edge_no_props_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_create_edge_no_props_survives_reopen",
            open,
            |db| {
                let alix = db.create_node(&["Person"]).unwrap();
                let gus = db.create_node(&["Person"]).unwrap();

                let mut session = db.session();
                session.begin_transaction().expect("begin");
                session.create_edge(alix, gus, "KNOWS").unwrap();
                session.commit().expect("commit");
            },
        );
        assert_eq!(
            db.edge_count(),
            1,
            "edge from Session::create_edge was lost on reopen"
        );
    }

    #[test]
    fn session_delete_node_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_delete_node_survives_reopen",
            open,
            |db| {
                let alix = db.create_node(&["Person"]).unwrap();
                let _gus = db.create_node(&["Person"]).unwrap();

                let mut session = db.session();
                session.begin_transaction().expect("begin");
                let removed = session.delete_node(alix).unwrap();
                session.commit().expect("commit");
                assert!(removed, "delete_node should report it removed alix");
            },
        );
        assert_eq!(
            db.node_count(),
            1,
            "delete via Session::delete_node was not durable across reopen"
        );
    }

    #[test]
    fn session_delete_edge_survives_reopen() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_delete_edge_survives_reopen",
            open,
            |db| {
                let alix = db.create_node(&["Person"]).unwrap();
                let gus = db.create_node(&["Person"]).unwrap();
                let _kept = db.create_edge(alix, gus, "KNOWS").unwrap();
                let to_delete = db.create_edge(alix, gus, "ALSO_KNOWS").unwrap();

                let mut session = db.session();
                session.begin_transaction().expect("begin");
                let removed = session.delete_edge(to_delete).unwrap();
                session.commit().expect("commit");
                assert!(removed, "delete_edge should report it removed the edge");
            },
        );
        assert_eq!(
            db.edge_count(),
            1,
            "delete via Session::delete_edge was not durable across reopen"
        );
    }

    /// Rollback must not leak rolled-back records into a subsequent committed
    /// transaction's WAL window.
    ///
    /// WAL recovery buffers records until it sees `TransactionCommit`, at which
    /// point the entire buffer is flushed. If `Session::rollback` does not emit
    /// `TransactionAbort` to clear the buffer, rolled-back mutations are
    /// resurrected when the next transaction commits.
    #[test]
    fn session_rollback_does_not_resurrect_on_next_commit() {
        let (_dir, db) = reopened_after_crash(
            "session_wal_durability::session_rollback_does_not_resurrect_on_next_commit",
            open,
            |db| {
                // tx1: create a node, then rollback. The node must NOT survive reopen.
                let mut session = db.session();
                session.begin_transaction().expect("begin tx1");
                session
                    .create_node_with_props(
                        &["RolledBack"],
                        [("name", Value::String("ghost".into()))],
                    )
                    .expect("create rolled-back node");
                session.rollback().expect("rollback tx1");

                // tx2: create a different node and commit. This commit's WAL marker
                // must not flush tx1's records.
                session.begin_transaction().expect("begin tx2");
                session
                    .create_node_with_props(
                        &["Committed"],
                        [("name", Value::String("kept".into()))],
                    )
                    .expect("create committed node");
                session.commit().expect("commit tx2");
            },
        );
        assert_eq!(
            db.node_count(),
            1,
            "rolled-back node was resurrected by a later commit (TransactionAbort missing from rollback path)"
        );
    }
}
