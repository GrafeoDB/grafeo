//! Regression tests for [issue #327](https://github.com/GrafeoDB/grafeo/issues/327):
//! Session direct LPG writes are not durable in `wal-directory` storage.
//!
//! `Session::create_node_with_props`, `Session::create_edge_with_props`,
//! `Session::set_node_property`, and `Session::set_edge_property` call
//! `active_lpg_store()` directly without routing through `WalGraphStore`,
//! so writes via these APIs are not durable under
//! `StorageFormat::WalDirectory` (which relies on WAL replay rather than a
//! single-file checkpoint).
//!
//! The equivalent `GrafeoDB::create_node_with_props`,
//! `GrafeoDB::create_edge_with_props`, etc. log
//! `WalRecord::CreateNode` / `SetNodeProperty` / `CreateEdge` /
//! `SetEdgeProperty` correctly. See `crates/grafeo-engine/src/database/crud.rs`
//! for the WAL-correct path and `crates/grafeo-engine/src/session/mod.rs`
//! around `create_node_with_props` for the bug.
//!
//! Each test:
//! 1. opens a DB with `StorageFormat::WalDirectory`,
//! 2. performs the session-direct write,
//! 3. closes,
//! 4. reopens, and
//! 5. asserts the data survived.
//!
//! Tests verify the fix for #327 and run by default:
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test session_wal_durability
//! ```

#![allow(missing_docs)]

#[cfg(feature = "wal")]
mod session_wal_durability {
    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};

    #[test]
    fn session_create_node_with_props_survives_reopen() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");
            let mut session = db.session();
            session.begin_transaction().expect("begin");
            session
                .create_node_with_props(&["Probe"], [("name", Value::String("Alix".into()))])
                .expect("create node");
            session.commit().expect("commit");
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
        assert_eq!(
            db.node_count(),
            1,
            "node created via Session::create_node_with_props was lost on reopen"
        );
    }

    #[test]
    fn session_create_edge_with_props_survives_reopen() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");

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
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
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
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");
        let alix;

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");

            // Create node via DB-direct path so the node itself is durable;
            // the test isolates the property bug.
            alix = db.create_node(&["Person"]).expect("create anchor node");

            let mut session = db.session();
            session.begin_transaction().expect("begin");
            session
                .set_node_property(alix, "name", Value::String("Alix".into()))
                .expect("set property");
            session.commit().expect("commit");
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
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
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");
        let edge_id;

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");

            // Anchor edge via DB-direct path; isolate the property bug.
            let alix = db.create_node(&["Person"]).unwrap();
            let gus = db.create_node(&["Person"]).unwrap();
            edge_id = db.create_edge(alix, gus, "KNOWS");

            let mut session = db.session();
            session.begin_transaction().expect("begin");
            session
                .set_edge_property(edge_id, "since", Value::Int64(2026))
                .expect("set edge property");
            session.commit().expect("commit");
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
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
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");
            db.execute_language("CREATE (:Probe {name: 'cypher'});", "cypher", None)
                .expect("cypher create");
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
        assert_eq!(
            db.node_count(),
            1,
            "Cypher-created node was lost on reopen (oracle test failed; bug is broader than #327)"
        );
    }

    #[test]
    fn session_create_node_no_props_survives_reopen() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");
            let mut session = db.session();
            session.begin_transaction().expect("begin");
            session.create_node(&["Probe"]);
            session.commit().expect("commit");
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
        assert_eq!(
            db.node_count(),
            1,
            "node from Session::create_node was lost on reopen"
        );
    }

    #[test]
    fn session_create_edge_no_props_survives_reopen() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");
            let alix = db.create_node(&["Person"]).unwrap();
            let gus = db.create_node(&["Person"]).unwrap();

            let mut session = db.session();
            session.begin_transaction().expect("begin");
            session.create_edge(alix, gus, "KNOWS");
            session.commit().expect("commit");
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
        assert_eq!(
            db.edge_count(),
            1,
            "edge from Session::create_edge was lost on reopen"
        );
    }

    #[test]
    fn session_delete_node_survives_reopen() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");
            let alix = db.create_node(&["Person"]).unwrap();
            let _gus = db.create_node(&["Person"]).unwrap();

            let mut session = db.session();
            session.begin_transaction().expect("begin");
            let removed = session.delete_node(alix);
            session.commit().expect("commit");
            assert!(removed, "delete_node should report it removed alix");
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
        assert_eq!(
            db.node_count(),
            1,
            "delete via Session::delete_node was not durable across reopen"
        );
    }

    #[test]
    fn session_delete_edge_survives_reopen() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");
            let alix = db.create_node(&["Person"]).unwrap();
            let gus = db.create_node(&["Person"]).unwrap();
            let _kept = db.create_edge(alix, gus, "KNOWS");
            let to_delete = db.create_edge(alix, gus, "ALSO_KNOWS");

            let mut session = db.session();
            session.begin_transaction().expect("begin");
            let removed = session.delete_edge(to_delete);
            session.commit().expect("commit");
            assert!(removed, "delete_edge should report it removed the edge");
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
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
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("testdb");

        {
            let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
            let db = GrafeoDB::with_config(config).expect("open for write");

            // tx1: create a node, then rollback. The node must NOT survive reopen.
            let mut session = db.session();
            session.begin_transaction().expect("begin tx1");
            session
                .create_node_with_props(&["RolledBack"], [("name", Value::String("ghost".into()))])
                .expect("create rolled-back node");
            session.rollback().expect("rollback tx1");

            // tx2: create a different node and commit. This commit's WAL marker
            // must not flush tx1's records.
            session.begin_transaction().expect("begin tx2");
            session
                .create_node_with_props(&["Committed"], [("name", Value::String("kept".into()))])
                .expect("create committed node");
            session.commit().expect("commit tx2");
            drop(session);
            db.close().expect("close");
        }

        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("reopen");
        assert_eq!(
            db.node_count(),
            1,
            "rolled-back node was resurrected by a later commit (TransactionAbort missing from rollback path)"
        );
    }
}
