//! Each transaction's WAL records form one group, written at commit (#411).
//!
//! Records of concurrent transactions used to interleave in the shared log,
//! so on replay one session's abort cleared another's pending records and one
//! session's commit committed another's uncommitted ones. These tests crash a
//! child process (it exits without `close()`) after interleaving sessions,
//! then reopen and check exactly which writes survived.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test wal_transaction_groups
//! ```

#![allow(missing_docs)]

#[cfg(feature = "wal")]
mod tests {
    use grafeo_common::types::{NodeId, TransactionId, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::session::Session;
    use grafeo_engine::{Config, GrafeoDB};
    use grafeo_storage::wal::{WalManager, WalRecord};
    use std::path::{Path, PathBuf};

    const SCENARIO_VAR: &str = "GRAFEO_WAL_GROUP_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_WAL_GROUP_PATH";
    const FORMAT_VAR: &str = "GRAFEO_WAL_GROUP_FORMAT";

    fn formats(dir: &Path) -> Vec<(&'static str, PathBuf)> {
        let mut formats = vec![("wal-directory", dir.join("dir-db"))];
        #[cfg(feature = "grafeo-file")]
        formats.push(("single-file", dir.join("single.grafeo")));
        formats
    }

    fn storage_format(name: &str) -> StorageFormat {
        match name {
            "wal-directory" => StorageFormat::WalDirectory,
            "single-file" => StorageFormat::SingleFile,
            other => panic!("unknown format {other}"),
        }
    }

    fn config(path: &Path, format: &str) -> Config {
        Config::persistent(path).with_storage_format(storage_format(format))
    }

    fn open(path: &Path, format: &str) -> GrafeoDB {
        GrafeoDB::with_config(config(path, format)).unwrap()
    }

    /// Directory holding the WAL files for a database.
    fn wal_dir(path: &Path, format: &str) -> PathBuf {
        match format {
            "wal-directory" => path.join("wal"),
            _ => {
                let mut sidecar = path.as_os_str().to_owned();
                sidecar.push(".wal");
                PathBuf::from(sidecar)
            }
        }
    }

    /// Runs `scenario` in a child process that exits without closing the
    /// database, like a crash.
    fn crash_after(scenario: &str, path: &Path, format: &str) {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::crash_child", "--nocapture"])
            .env(SCENARIO_VAR, scenario)
            .env(PATH_VAR, path)
            .env(FORMAT_VAR, format)
            .status()
            .unwrap();
        assert!(status.success(), "{format}: scenario {scenario} failed");
    }

    /// Sorted `n.name` of every `:Person` in the session's current graph.
    fn names_in(session: &Session) -> Vec<String> {
        let result = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            .unwrap();
        result
            .rows()
            .iter()
            .map(|row| match &row[0] {
                Value::String(s) => s.to_string(),
                other => panic!("unexpected name {other:?}"),
            })
            .collect()
    }

    fn names(db: &GrafeoDB) -> Vec<String> {
        names_in(&db.session())
    }

    fn names_in_graph(db: &GrafeoDB, graph: &str) -> Vec<String> {
        let session = db.session();
        session.use_graph(graph);
        names_in(&session)
    }

    fn insert(session: &Session, name: &str) {
        session
            .execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
    }

    fn strings(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    // ------------------------------------------------------------------
    // Scenarios, run in the child process
    // ------------------------------------------------------------------

    fn run_scenario(scenario: &str, db: &GrafeoDB) {
        match scenario {
            // The issue: B's rollback must not clear A's pending records.
            "abort_between" => {
                let mut a = db.session();
                let mut b = db.session();
                a.begin_transaction().unwrap();
                insert(&a, "Alix");
                b.begin_transaction().unwrap();
                insert(&b, "Gus");
                b.rollback().unwrap();
                a.commit().unwrap();
                // B's rolled-back records must not come back with its next write.
                insert(&b, "Jules");
            }
            // The mirror: B's commit must not commit A's uncommitted records.
            "commit_between" => {
                let mut a = db.session();
                let mut b = db.session();
                a.begin_transaction().unwrap();
                insert(&a, "Alix");
                b.begin_transaction().unwrap();
                insert(&b, "Gus");
                b.commit().unwrap();
                a.rollback().unwrap();
            }
            // Writes after a savepoint that was rolled back to are gone.
            "savepoint" => {
                let mut a = db.session();
                a.begin_transaction().unwrap();
                insert(&a, "Alix");
                a.savepoint("sp").unwrap();
                insert(&a, "Gus");
                a.rollback_to_savepoint("sp").unwrap();
                insert(&a, "Vincent");
                a.commit().unwrap();
            }
            // One transaction writes a named graph and the default graph,
            // while another session commits to the default graph in between.
            "named_graphs" => {
                db.session().execute("CREATE GRAPH g").unwrap();
                let mut a = db.session();
                a.begin_transaction().unwrap();
                a.use_graph("g");
                insert(&a, "Mia");
                a.use_graph("default");
                insert(&a, "Alix");
                // Commits while A is still open. (Inserting before A's default
                // graph write would reuse node id 0, which conflict detection
                // confuses with Mia's id in graph g.)
                insert(&db.session(), "Gus");
                a.commit().unwrap();
                insert(&db.session(), "Jules");
            }
            // Direct writes outside a transaction, including a named graph.
            "direct_writes" => {
                db.session().execute("CREATE GRAPH g").unwrap();
                let session = db.session();
                session
                    .create_node_with_props(&["Person"], [("name", Value::from("Django"))])
                    .unwrap();
                session.use_graph("g");
                session
                    .create_node_with_props(&["Person"], [("name", Value::from("Hans"))])
                    .unwrap();
                // The group above ended in graph g; this one is in the
                // default graph and must replay there.
                insert(&db.session(), "Jules");
            }
            // Same interleaving as "abort_between", through SPARQL updates.
            #[cfg(all(feature = "sparql", feature = "triple-store"))]
            "rdf_abort_between" => {
                let mut a = db.session();
                let mut b = db.session();
                a.begin_transaction().unwrap();
                a.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/p> "1" . }"#)
                    .unwrap();
                b.begin_transaction().unwrap();
                b.execute_sparql(r#"INSERT DATA { <http://ex.org/gus> <http://ex.org/p> "2" . }"#)
                    .unwrap();
                b.rollback().unwrap();
                a.commit().unwrap();
            }
            // Schema changes outside a transaction, with nothing committed after.
            "ddl_only" => {
                db.session().execute("CREATE GRAPH g").unwrap();
            }
            // Schema changes inside a transaction that is rolled back.
            "ddl_in_rolled_back_tx" => {
                let mut a = db.session();
                a.begin_transaction().unwrap();
                a.execute("CREATE GRAPH h").unwrap();
                a.execute("CREATE NODE TYPE Robot (name STRING)").unwrap();
                a.rollback().unwrap();
            }
            // A database-level SPARQL update, with nothing committed after.
            #[cfg(all(feature = "sparql", feature = "triple-store"))]
            "rdf_db_level" => {
                db.execute_sparql(r#"INSERT DATA { <http://ex.org/mia> <http://ex.org/p> "3" . }"#)
                    .unwrap();
            }
            "seed_alix" => insert(&db.session(), "Alix"),
            "insert_gus" => insert(&db.session(), "Gus"),
            other => panic!("unknown scenario {other}"),
        }
    }

    /// Child-process entry for [`crash_after`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
            return;
        };
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        let format = std::env::var(FORMAT_VAR).unwrap();
        let mut config = config(&path, &format);
        #[cfg(feature = "triple-store")]
        if scenario.starts_with("rdf_") {
            config = config.with_graph_model(grafeo_engine::GraphModel::Rdf);
        }
        let db = GrafeoDB::with_config(config).unwrap();
        run_scenario(&scenario, &db);
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------

    #[test]
    fn rollback_does_not_discard_another_transaction() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            crash_after("abort_between", &path, format);
            assert_eq!(
                names(&open(&path, format)),
                strings(&["Alix", "Jules"]),
                "{format}"
            );
        }
    }

    #[test]
    fn commit_does_not_commit_another_transaction() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            crash_after("commit_between", &path, format);
            assert_eq!(names(&open(&path, format)), strings(&["Gus"]), "{format}");
        }
    }

    #[test]
    fn rollback_to_savepoint_is_not_replayed() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            crash_after("savepoint", &path, format);
            assert_eq!(
                names(&open(&path, format)),
                strings(&["Alix", "Vincent"]),
                "{format}"
            );
        }
    }

    #[test]
    fn named_graph_writes_replay_into_their_graph() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            crash_after("named_graphs", &path, format);
            let db = open(&path, format);
            assert_eq!(
                names(&db),
                strings(&["Alix", "Gus", "Jules"]),
                "{format}: default graph"
            );
            assert_eq!(
                names_in_graph(&db, "g"),
                strings(&["Mia"]),
                "{format}: graph g"
            );
        }
    }

    #[test]
    fn direct_writes_outside_a_transaction_are_durable() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            crash_after("direct_writes", &path, format);
            let db = open(&path, format);
            assert_eq!(
                names(&db),
                strings(&["Django", "Jules"]),
                "{format}: default graph"
            );
            assert_eq!(
                names_in_graph(&db, "g"),
                strings(&["Hans"]),
                "{format}: graph g"
            );
        }
    }

    #[test]
    fn schema_change_outside_a_transaction_is_durable() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            crash_after("ddl_only", &path, format);
            assert_eq!(
                open(&path, format).list_graphs(),
                vec!["g".to_string()],
                "{format}"
            );
        }
    }

    /// Schema changes take effect immediately and a rollback does not undo
    /// them in memory, so the log records them as they are applied and a
    /// reopen matches the state before the crash. (If schema changes become
    /// transactional, this test changes with them.)
    #[test]
    fn schema_changes_in_a_rolled_back_transaction_match_memory() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            crash_after("ddl_in_rolled_back_tx", &path, format);
            let db = open(&path, format);
            assert_eq!(db.list_graphs(), vec!["h".to_string()], "{format}");
            let types = db.session().execute("SHOW NODE TYPES").unwrap();
            let names: Vec<Value> = types.rows().iter().map(|row| row[0].clone()).collect();
            assert_eq!(names, vec![Value::from("Robot")], "{format}");
        }
    }

    /// A transaction still open at `close()` is not committed on replay.
    #[test]
    fn open_transaction_at_close_is_not_recovered() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            {
                let db = open(&path, format);
                let mut open_tx = db.session();
                open_tx.begin_transaction().unwrap();
                insert(&open_tx, "Vincent");
                insert(&db.session(), "Gus");
                db.close().unwrap();
            }
            assert_eq!(names(&open(&path, format)), strings(&["Gus"]), "{format}");
        }
    }

    /// Records left without a commit marker by a crash are sealed at open,
    /// so the next commit cannot pick them up.
    #[test]
    fn torn_tail_is_not_committed_by_a_later_transaction() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            crash_after("seed_alix", &path, format);
            {
                // A group cut off by a crash before its commit marker.
                let wal = WalManager::open(wal_dir(&path, format)).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(9_000),
                    labels: vec!["Person".to_string()],
                })
                .unwrap();
                wal.log(&WalRecord::SetNodeProperty {
                    id: NodeId::new(9_000),
                    key: "name".to_string(),
                    value: Value::from("Torn"),
                })
                .unwrap();
            }
            crash_after("insert_gus", &path, format);
            assert_eq!(
                names(&open(&path, format)),
                strings(&["Alix", "Gus"]),
                "{format}"
            );
        }
    }

    /// A log written by an older version can end inside a named graph. New
    /// default-graph writes must not replay into that graph.
    #[test]
    fn default_graph_writes_after_a_log_ending_in_a_named_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dir-db");
        {
            let wal = WalManager::open(path.join("wal")).unwrap();
            let records = [
                WalRecord::CreateNamedGraph {
                    name: "g".to_string(),
                },
                WalRecord::SwitchGraph {
                    name: Some("g".to_string()),
                },
                WalRecord::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Person".to_string()],
                },
                WalRecord::SetNodeProperty {
                    id: NodeId::new(1),
                    key: "name".to_string(),
                    value: Value::from("Mia"),
                },
                WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(2),
                },
            ];
            for record in &records {
                wal.log(record).unwrap();
            }
        }
        crash_after("insert_gus", &path, "wal-directory");

        let db = open(&path, "wal-directory");
        assert_eq!(names(&db), strings(&["Gus"]));
        assert_eq!(names_in_graph(&db, "g"), strings(&["Mia"]));
    }

    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    mod rdf {
        use super::*;
        use grafeo_engine::GraphModel;

        fn open_rdf(path: &Path, format: &str) -> GrafeoDB {
            GrafeoDB::with_config(config(path, format).with_graph_model(GraphModel::Rdf)).unwrap()
        }

        fn subjects(db: &GrafeoDB) -> Vec<String> {
            let result = db
                .session()
                .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/p> ?o } ORDER BY ?s")
                .unwrap();
            result
                .rows()
                .iter()
                .map(|row| match &row[0] {
                    Value::String(s) => s.to_string(),
                    other => format!("{other:?}"),
                })
                .collect()
        }

        #[test]
        fn sparql_rollback_does_not_discard_another_transaction() {
            let dir = tempfile::tempdir().unwrap();
            for (format, path) in formats(dir.path()) {
                crash_after("rdf_abort_between", &path, format);
                let db = open_rdf(&path, format);
                assert_eq!(
                    subjects(&db),
                    vec!["http://ex.org/alix".to_string()],
                    "{format}"
                );
            }
        }

        #[test]
        fn database_level_sparql_update_is_durable() {
            let dir = tempfile::tempdir().unwrap();
            for (format, path) in formats(dir.path()) {
                crash_after("rdf_db_level", &path, format);
                assert_eq!(
                    subjects(&open_rdf(&path, format)),
                    vec!["http://ex.org/mia".to_string()],
                    "{format}"
                );
            }
        }
    }
}
