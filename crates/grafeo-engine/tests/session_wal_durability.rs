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
        let config = Config::persistent(path).with_storage_format(StorageFormat::Auto);
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

    // TODO(wal): the `wal_group_failures` crash-injection tests from #614 (issue
    // #498) were removed for the 0.6.0 release: this branch's engine does not
    // implement the asserted WAL acknowledgement semantics yet (there is no
    // `wal_before_group_write` failure seam), so every test failed, and policy
    // D4 forbids ignoring crash-injection tests. Restore the module from PR #614
    // or commit f165b194 once the WAL work lands.

    /// A failed WAL write reaches the real caller without publishing the
    /// failed commit's epoch (#498). Ported from PR #614 (temporaryfix) to
    /// the WAL of 0.6.0: its failure points are the write of the commit's
    /// group (`wal:write`) and the sync after it (`wal:sync`), and its
    /// errors `GRAFEO-T009` (not written: rolled back, retryable) and
    /// `GRAFEO-T010` (written, not durable: the outcome is known after a
    /// reopen); writes after a T010 are fenced with `GRAFEO-T008`. A direct
    /// call while no transaction is open cannot be rolled back, so a group it
    /// could not write fails it with T010 too (where #614 expected T009).
    #[cfg(all(
        feature = "wal",
        feature = "grafeo-file",
        feature = "lpg",
        feature = "gql",
        feature = "testing-crash-injection"
    ))]
    mod wal_group_failures {
        use super::{only_id, reopened_after_crash};
        use grafeo_common::testing::crash::{maybe_fail, with_failure_at};
        use grafeo_common::types::EpochId;
        use grafeo_common::utils::error::{Error, Result, TransactionError};
        use grafeo_engine::config::{DurabilityMode, StorageFormat};
        use grafeo_engine::{Config, GrafeoDB};
        use grafeo_storage::file::detect::sidecar_wal_path;
        use std::ffi::OsString;
        use std::path::Path;

        type Write = fn(&GrafeoDB, &str) -> Result<()>;

        /// The failure of the group's write, and of the sync after it.
        const GROUP_WRITE: u64 = 1;
        const GROUP_SYNC: u64 = 2;

        fn open_sync(path: &Path) -> GrafeoDB {
            let db = GrafeoDB::with_config(
                Config::persistent(path)
                    .with_storage_format(StorageFormat::Auto)
                    .with_wal_durability(DurabilityMode::Sync),
            )
            .expect("open with synchronous WAL durability");
            #[cfg(feature = "cdc")]
            db.set_cdc_enabled(true);
            db
        }

        fn explicit_query_commit(db: &GrafeoDB, label: &str) -> Result<()> {
            let mut session = db.session();
            session.begin_transaction()?;
            session.execute(&format!("INSERT (:{label})"))?;
            session.commit()
        }

        fn default_query_autocommit(db: &GrafeoDB, label: &str) -> Result<()> {
            db.execute(&format!("INSERT (:{label})")).map(|_| ())
        }

        fn database_create_node(db: &GrafeoDB, label: &str) -> Result<()> {
            db.create_node(&[label]).map(|_| ())
        }

        fn assert_live_empty(db: &GrafeoDB, epoch: EpochId) {
            #[cfg(feature = "cdc")]
            assert_eq!(
                db.changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                    .unwrap()
                    .len(),
                0,
                "a failed acknowledgement must not publish CDC events",
            );
            assert_eq!(
                db.node_count(),
                0,
                "direct reads must not publish the failed write"
            );
            assert_eq!(
                only_id(db, "MATCH (n) RETURN count(n)"),
                0,
                "queries must not publish the failed write"
            );
            assert_eq!(
                db.current_epoch(),
                epoch,
                "the published epoch must not advance"
            );
        }

        fn assert_code(error: &Error, code: &str, operation: &str) {
            assert_eq!(error.error_code().as_str(), code, "{operation}: {error}");
        }

        fn assert_incomplete(error: &Error, operation: &str) {
            assert!(
                matches!(
                    error,
                    Error::Transaction(TransactionError::IncompleteCommit)
                ),
                "{operation}: expected IncompleteCommit, got {error:?}"
            );
            assert_code(error, "GRAFEO-T008", operation);
        }

        fn assert_outcome_unknown(error: &Error, operation: &str) {
            assert_code(error, "GRAFEO-T010", operation);
            let message = error.to_string().to_ascii_lowercase();
            assert!(
                message.contains("reopen"),
                "{operation}: the caller must learn that reopening decides: {message}"
            );
        }

        /// Closing an incomplete commit must preserve the WAL, including its
        /// bytes, rather than checkpointing or removing it.
        fn wal_log_bytes(db: &GrafeoDB) -> Vec<(OsString, Vec<u8>)> {
            let sidecar = sidecar_wal_path(db.path().expect("a persistent database path"));
            let mut files: Vec<_> = std::fs::read_dir(&sidecar)
                .expect("the sidecar WAL must exist")
                .map(|entry| entry.expect("read WAL directory entry").path())
                .filter(|path| path.extension().is_some_and(|extension| extension == "log"))
                .map(|path| {
                    (
                        path.file_name().expect("a WAL file name").to_os_string(),
                        std::fs::read(&path).expect("read WAL bytes"),
                    )
                })
                .collect();
            files.sort_by(|left, right| left.0.cmp(&right.0));
            assert!(!files.is_empty(), "the WAL must remain on disk");
            files
        }

        /// The caller's result is meaningful only if the armed I/O failure
        /// fired inside it, rather than remaining armed after it returned.
        fn with_wal_failure(count: u64, write: impl FnOnce() -> Result<()>) -> Result<()> {
            with_failure_at(count, || {
                let result = write();
                for _ in 0..count {
                    assert!(
                        maybe_fail("unreached_wal_failure_witness").is_ok(),
                        "the write must consume its armed WAL failure before returning"
                    );
                }
                result
            })
        }

        fn rejection_before_group_write(test: &str, write: Write) {
            let (_dir, db) = reopened_after_crash(test, open_sync, |db| {
                let epoch = db.current_epoch();
                let error = with_wal_failure(GROUP_WRITE, || write(db, "Rejected"))
                    .expect_err("a failure of the WAL group write must reach the caller");
                assert!(
                    error.to_string().contains("wal:write"),
                    "the write must fail at the group's write: {error}"
                );
                assert_code(&error, "GRAFEO-T009", "the rejected write");
                assert_live_empty(db, epoch);

                write(db, "Durable").expect("a later write through the same caller must succeed");
                assert_eq!(
                    db.node_count(),
                    1,
                    "only the later write is visible directly"
                );
                assert_eq!(only_id(db, "MATCH (n) RETURN count(n)"), 1);
                assert_eq!(only_id(db, "MATCH (n:Rejected) RETURN count(n)"), 0);
            });
            assert_eq!(
                db.node_count(),
                1,
                "only the later write survives WAL replay"
            );
            assert_eq!(only_id(&db, "MATCH (n) RETURN count(n)"), 1);
            assert_eq!(only_id(&db, "MATCH (n:Durable) RETURN count(n)"), 1);
            assert_eq!(
                only_id(&db, "MATCH (n:Rejected) RETURN count(n)"),
                0,
                "the rejected write must not be resurrected by a later commit"
            );
        }

        /// A direct call while no transaction is open, whose group was not
        /// written: it cannot be rolled back, so it fails with T010, the
        /// database is fenced until a reopen, and the reopen, whose log
        /// holds nothing of the call, does not have it.
        fn direct_call_not_written_is_fenced(test: &str) {
            let (_dir, db) = reopened_after_crash(test, open_sync, |db| {
                let epoch = db.current_epoch();
                let error = with_wal_failure(GROUP_WRITE, || database_create_node(db, "Rejected"))
                    .expect_err("a failure of the WAL group write must reach the caller");
                assert!(error.to_string().contains("wal:write"), "{error}");
                assert_outcome_unknown(&error, "the direct call");
                assert_live_empty(db, epoch);
                assert_incomplete(
                    &db.execute("INSERT (:FencedQuery)")
                        .expect_err("later query writes must be fenced"),
                    "a later query write",
                );
                let wal_before_close = wal_log_bytes(db);
                assert_incomplete(
                    &db.close()
                        .expect_err("close must report the incomplete commit"),
                    "close",
                );
                assert_eq!(wal_log_bytes(db), wal_before_close);
            });
            assert_eq!(db.node_count(), 0, "the log holds nothing of the call");
        }

        fn unknown_after_group_sync(test: &str, write: Write) {
            let (_dir, db) = reopened_after_crash(test, open_sync, |db| {
                let epoch = db.current_epoch();
                let error = with_wal_failure(GROUP_SYNC, || write(db, "Unknown"))
                    .expect_err("a lost acknowledgement after the WAL write must reach the caller");
                assert!(error.to_string().contains("wal:sync"), "{error}");
                assert_outcome_unknown(&error, "the initial write");
                assert_live_empty(db, epoch);

                assert_incomplete(
                    &db.create_node(&["FencedDirect"])
                        .expect_err("later direct writes must be fenced"),
                    "a later direct write",
                );
                assert_incomplete(
                    &db.execute("INSERT (:FencedQuery)")
                        .expect_err("later query writes must be fenced"),
                    "a later query write",
                );
                assert_live_empty(db, epoch);

                let wal_before_close = wal_log_bytes(db);
                assert_incomplete(
                    &db.close()
                        .expect_err("close must report the incomplete commit"),
                    "close",
                );
                assert_eq!(
                    wal_log_bytes(db),
                    wal_before_close,
                    "close must preserve the WAL group for replay"
                );
            });
            assert_eq!(
                db.node_count(),
                1,
                "the written group survives WAL replay exactly once"
            );
            assert_eq!(only_id(&db, "MATCH (n) RETURN count(n)"), 1);
            assert_eq!(only_id(&db, "MATCH (n:Unknown) RETURN count(n)"), 1);
            assert_eq!(only_id(&db, "MATCH (n:FencedDirect) RETURN count(n)"), 0);
            assert_eq!(only_id(&db, "MATCH (n:FencedQuery) RETURN count(n)"), 0);
        }

        #[derive(Clone, Copy)]
        enum PropertyCaller {
            Explicit,
            Autocommit,
            Direct,
        }

        fn update_property(db: &GrafeoDB, caller: PropertyCaller, value: &str) -> Result<()> {
            use grafeo_common::types::Value;
            let node = db.iter_nodes().next().expect("the seed node").id;
            match caller {
                PropertyCaller::Explicit => {
                    let mut session = db.session();
                    session.begin_transaction()?;
                    session.set_node_property(node, "revision", Value::from(value))?;
                    session.commit()
                }
                PropertyCaller::Autocommit => db
                    .execute(&format!("MATCH (n:Probe) SET n.revision = '{value}'"))
                    .map(drop),
                PropertyCaller::Direct => {
                    db.set_node_property(node, "revision", Value::from(value))
                }
            }
        }

        fn assert_revision(db: &GrafeoDB, expected: &str) {
            use grafeo_common::types::Value;
            assert_eq!(db.node_count(), 1, "updates must not add nodes");
            let node = db.iter_nodes().next().expect("the seed node");
            assert_eq!(node.get_property("revision"), Some(&Value::from(expected)));
            let result = db.execute("MATCH (n:Probe) RETURN n.revision").unwrap();
            assert_eq!(result.rows(), &[vec![Value::from(expected)]]);
        }

        /// Undo must restore an existing property, not just remove new nodes.
        fn property_failure(test: &str, caller: PropertyCaller, failure: u64) {
            use grafeo_common::types::Value;
            // A direct call outside a transaction is fenced, not rolled back,
            // when its group is not written (see the module docs).
            let rolled_back = failure == GROUP_WRITE && !matches!(caller, PropertyCaller::Direct);
            let (_dir, db) = reopened_after_crash(test, open_sync, |db| {
                db.create_node_with_props(&["Probe"], [("revision", Value::from("before"))])
                    .unwrap();
                let epoch = db.current_epoch();
                let error = with_wal_failure(failure, || update_property(db, caller, "attempted"))
                    .expect_err("the property update must report its WAL failure");
                // A commit whose outcome is unknown keeps its pending
                // property version until the reopen, which no read sees
                // (#412): every caller reads the value before it.
                assert_revision(db, "before");
                #[cfg(feature = "cdc")]
                assert_eq!(
                    db.changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                        .unwrap()
                        .len(),
                    1,
                    "only the seed creation is published to CDC",
                );
                assert_eq!(db.current_epoch(), epoch);
                if rolled_back {
                    assert!(error.to_string().contains("wal:write"), "{error}");
                    assert_code(&error, "GRAFEO-T009", "the rolled-back update");
                    update_property(db, caller, "retry").unwrap();
                    assert_revision(db, "retry");
                } else {
                    assert_outcome_unknown(&error, "the lost property-update acknowledgement");
                    assert_incomplete(
                        &update_property(db, caller, "fenced")
                            .expect_err("later updates are fenced"),
                        "the later update",
                    );
                    let bytes = wal_log_bytes(db);
                    assert_incomplete(&db.close().expect_err("close refuses"), "close");
                    assert_eq!(wal_log_bytes(db), bytes);
                }
            });
            let expected = match (rolled_back, failure) {
                (true, _) => "retry",
                // Not written: the log holds nothing of the update.
                (false, GROUP_WRITE) => "before",
                (false, _) => "attempted",
            };
            assert_revision(&db, expected);
        }

        #[test]
        fn explicit_property_before_write_is_undone_and_retryable() {
            property_failure(
                "session_wal_durability::wal_group_failures::explicit_property_before_write_is_undone_and_retryable",
                PropertyCaller::Explicit,
                GROUP_WRITE,
            );
        }

        #[test]
        fn explicit_property_lost_ack_is_unknown_and_recovers() {
            property_failure(
                "session_wal_durability::wal_group_failures::explicit_property_lost_ack_is_unknown_and_recovers",
                PropertyCaller::Explicit,
                GROUP_SYNC,
            );
        }

        #[test]
        fn autocommit_property_before_write_is_undone_and_retryable() {
            property_failure(
                "session_wal_durability::wal_group_failures::autocommit_property_before_write_is_undone_and_retryable",
                PropertyCaller::Autocommit,
                GROUP_WRITE,
            );
        }

        #[test]
        fn autocommit_property_lost_ack_is_unknown_and_recovers() {
            property_failure(
                "session_wal_durability::wal_group_failures::autocommit_property_lost_ack_is_unknown_and_recovers",
                PropertyCaller::Autocommit,
                GROUP_SYNC,
            );
        }

        #[test]
        fn direct_property_not_written_is_fenced_until_reopen() {
            property_failure(
                "session_wal_durability::wal_group_failures::direct_property_not_written_is_fenced_until_reopen",
                PropertyCaller::Direct,
                GROUP_WRITE,
            );
        }

        #[test]
        fn direct_property_lost_ack_is_unknown_and_recovers() {
            property_failure(
                "session_wal_durability::wal_group_failures::direct_property_lost_ack_is_unknown_and_recovers",
                PropertyCaller::Direct,
                GROUP_SYNC,
            );
        }

        #[test]
        fn explicit_query_commit_failure_before_group_write_is_rejected_and_retryable() {
            rejection_before_group_write(
                "session_wal_durability::wal_group_failures::explicit_query_commit_failure_before_group_write_is_rejected_and_retryable",
                explicit_query_commit,
            );
        }

        #[test]
        fn default_query_autocommit_failure_before_group_write_is_rejected_and_retryable() {
            rejection_before_group_write(
                "session_wal_durability::wal_group_failures::default_query_autocommit_failure_before_group_write_is_rejected_and_retryable",
                default_query_autocommit,
            );
        }

        #[test]
        fn database_create_node_not_written_is_fenced_until_reopen() {
            direct_call_not_written_is_fenced(
                "session_wal_durability::wal_group_failures::database_create_node_not_written_is_fenced_until_reopen",
            );
        }

        #[test]
        fn explicit_query_commit_failure_after_group_sync_is_unknown_and_recovers() {
            unknown_after_group_sync(
                "session_wal_durability::wal_group_failures::explicit_query_commit_failure_after_group_sync_is_unknown_and_recovers",
                explicit_query_commit,
            );
        }

        #[test]
        fn default_query_autocommit_failure_after_group_sync_is_unknown_and_recovers() {
            unknown_after_group_sync(
                "session_wal_durability::wal_group_failures::default_query_autocommit_failure_after_group_sync_is_unknown_and_recovers",
                default_query_autocommit,
            );
        }

        #[test]
        fn database_create_node_failure_after_group_sync_is_unknown_and_recovers() {
            unknown_after_group_sync(
                "session_wal_durability::wal_group_failures::database_create_node_failure_after_group_sync_is_unknown_and_recovers",
                database_create_node,
            );
        }
    }
}
