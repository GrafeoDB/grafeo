//! WAL records already contained in a `.grafeo` checkpoint must not be applied
//! a second time on reopen (#417).
//!
//! Replaying a `CreateEdge` for an edge the container already holds used to
//! add it to the adjacency lists again, so `MATCH` returned the edge twice.
//! Crashes run in a child process that exits without `close()`.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test checkpoint_replay
//! ```

#![allow(missing_docs)]

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod tests {
    use grafeo_common::types::{TransactionId, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use grafeo_storage::wal::{WalManager, WalRecord};
    use std::path::{Path, PathBuf};

    const SCENARIO_VAR: &str = "GRAFEO_CHECKPOINT_REPLAY_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_CHECKPOINT_REPLAY_PATH";

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(
            Config::persistent(path).with_storage_format(StorageFormat::SingleFile),
        )
        .unwrap()
    }

    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    fn backup_dir(path: &Path) -> PathBuf {
        path.with_extension("backups")
    }

    /// Runs `scenario` in a child process that exits without closing the
    /// database, like a crash.
    fn crash_after(scenario: &str, path: &Path) {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::crash_child", "--nocapture"])
            .env(SCENARIO_VAR, scenario)
            .env(PATH_VAR, path)
            .status()
            .unwrap();
        assert!(status.success(), "scenario {scenario} failed");
    }

    /// Child-process entry for [`crash_after`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
            return;
        };
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        let db = open(&path);
        let session = db.session();
        match scenario.as_str() {
            // The reproduction from the issue.
            "checkpoint_then_write" => {
                session
                    .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
                    .unwrap();
                db.wal_checkpoint().unwrap();
                session
                    .execute("INSERT (:Person {name: 'Vincent'})")
                    .unwrap();
            }
            // A full backup checkpoints the container without marking the WAL.
            "backup_then_write" => {
                session
                    .execute("INSERT (:N {i: 1})-[:E]->(:N {i: 2}), (:N {i: 3})-[:E]->(:N {i: 4})")
                    .unwrap();
                db.backup_full(&backup_dir(&path)).unwrap();
                session
                    .execute("INSERT (:N {i: 5})-[:E]->(:N {i: 6})")
                    .unwrap();
            }
            // Reopen (replaying the WAL), check the edges and crash again.
            "reopen" => {
                assert_eq!(edge_rows(&db), (Value::Int64(3), Value::Int64(3)));
                assert_eq!(db.edge_count(), 3);
            }
            // Crash at injection point N inside the first checkpoint
            // (`first:N`) or inside a second one after more writes (`second:N`).
            #[cfg(feature = "testing-crash-injection")]
            other if other.starts_with("first:") || other.starts_with("second:") => {
                let (phase, point) = other.split_once(':').unwrap();
                let point: u64 = point.parse().unwrap();
                session
                    .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
                    .unwrap();
                if phase == "second" {
                    db.wal_checkpoint().unwrap();
                    session
                        .execute(
                            "INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Jules'})",
                        )
                        .unwrap();
                }
                let db_ref = std::panic::AssertUnwindSafe(&db);
                let _ = grafeo_common::testing::crash::with_crash_at(point, move || {
                    let _ = db_ref.wal_checkpoint();
                });
            }
            other => panic!("unknown scenario {other}"),
        }
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    fn single_value(db: &GrafeoDB, query: &str) -> Value {
        let result = db.session().execute(query).unwrap();
        assert_eq!(result.row_count(), 1, "{query}");
        result.rows()[0][0].clone()
    }

    /// `(count(r), count(DISTINCT id(r)))` for all edges.
    fn edge_rows(db: &GrafeoDB) -> (Value, Value) {
        (
            single_value(db, "MATCH ()-[r]->() RETURN count(r)"),
            single_value(db, "MATCH ()-[r]->() RETURN count(DISTINCT id(r))"),
        )
    }

    #[test]
    fn crash_after_wal_checkpoint_does_not_duplicate_edges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        crash_after("checkpoint_then_write", &path);

        let db = open(&path);
        let knows = db
            .session()
            .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
            .unwrap();
        assert_eq!(
            knows.rows().to_vec(),
            vec![vec![Value::from("Alix"), Value::from("Gus")]]
        );
        assert_eq!(db.edge_count(), 1);
        assert_eq!(db.node_count(), 3);
    }

    /// The server scenario from the issue: a full backup, more writes, then
    /// repeated crash and reopen cycles. Each reopen replays the same WAL
    /// over the backup's checkpoint and crashes before a clean close could
    /// checkpoint again and drop the WAL.
    #[test]
    fn crash_after_full_backup_does_not_duplicate_edges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        crash_after("backup_then_write", &path);
        for _ in 0..2 {
            crash_after("reopen", &path);
        }
        assert!(sidecar_wal(&path).exists(), "no WAL left to replay");

        let db = open(&path);
        assert_eq!(edge_rows(&db), (Value::Int64(3), Value::Int64(3)));
        assert_eq!(db.edge_count(), 3);
        assert_eq!(db.node_count(), 6);
        drop(db);

        // The full backup restores to its two edges.
        let manifest = GrafeoDB::read_backup_manifest(&backup_dir(&path))
            .unwrap()
            .unwrap();
        let epoch = manifest.latest_full().unwrap().end_epoch;
        let restored_path = dir.path().join("restored.grafeo");
        GrafeoDB::restore_to_epoch(&backup_dir(&path), epoch, &restored_path).unwrap();
        let restored = open(&restored_path);
        assert_eq!(edge_rows(&restored), (Value::Int64(2), Value::Int64(2)));
    }

    /// `(name, name)` pairs of all KNOWS edges, sorted.
    fn knows(db: &GrafeoDB) -> Vec<Vec<Value>> {
        db.session()
            .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name ORDER BY a.name")
            .unwrap()
            .rows()
            .to_vec()
    }

    /// Crashes at every injection point of a checkpoint (`phase` is `first`,
    /// or `second` for a checkpoint after an earlier one) and checks that the
    /// reopened database holds exactly the committed KNOWS edges.
    #[cfg(feature = "testing-crash-injection")]
    fn crash_sweep(phase: &str, expected: &[Vec<Value>]) {
        // More points than the checkpoint has, so the last runs complete.
        for point in 1..=18 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db.grafeo");
            crash_after(&format!("{phase}:{point}"), &path);

            let db = open(&path);
            assert_eq!(
                knows(&db),
                expected,
                "{phase} checkpoint, crash point {point}"
            );
            assert_eq!(
                db.edge_count(),
                expected.len(),
                "{phase} checkpoint, crash point {point}"
            );
        }
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn crash_at_every_step_of_the_first_checkpoint() {
        crash_sweep("first", &[vec![Value::from("Alix"), Value::from("Gus")]]);
    }

    /// A later checkpoint used to overwrite the previous image in place, so a
    /// crash while writing it left a file that no longer opened (#418).
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn crash_at_every_step_of_a_later_checkpoint() {
        crash_sweep(
            "second",
            &[
                vec![Value::from("Alix"), Value::from("Gus")],
                vec![Value::from("Vincent"), Value::from("Jules")],
            ],
        );
    }

    /// A checkpoint that fails partway (here its new image cannot be created,
    /// as on a full disk) leaves the database usable, and the data readable
    /// after a reopen (#418).
    #[test]
    fn failed_checkpoint_keeps_the_database_readable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let blocker = {
            let mut name = path.as_os_str().to_owned();
            name.push(".checkpoint.tmp");
            PathBuf::from(name)
        };
        {
            let db = open(&path);
            let session = db.session();
            session
                .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
                .unwrap();
            db.wal_checkpoint().unwrap();
            session
                .execute("INSERT (:Person {name: 'Vincent'})")
                .unwrap();

            std::fs::create_dir(&blocker).unwrap();
            assert!(db.wal_checkpoint().is_err(), "the checkpoint must fail");
            session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
            assert!(db.close().is_err(), "close() checkpoints too");
        }
        std::fs::remove_dir(&blocker).unwrap();

        let db = open(&path);
        let names = db
            .session()
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(
            names,
            vec![
                vec![Value::from("Alix")],
                vec![Value::from("Gus")],
                vec![Value::from("Jules")],
                vec![Value::from("Vincent")],
            ]
        );
        assert_eq!(
            knows(&db),
            vec![vec![Value::from("Alix"), Value::from("Gus")]]
        );
    }

    fn log_files(dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "log"))
            .collect();
        files.sort();
        files
    }

    /// After a checkpoint the sidecar keeps only what the file does not hold.
    #[test]
    fn checkpoint_truncates_the_wal_it_covers() {
        use grafeo_storage::wal::WalRecovery;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = open(&path);
        for name in ["Alix", "Gus", "Vincent"] {
            db.session()
                .execute(&format!("INSERT (:Person {{name: '{name}'}})"))
                .unwrap();
            db.wal_checkpoint().unwrap();
        }

        let sidecar = sidecar_wal(&path);
        let files = log_files(&sidecar);
        assert_eq!(files.len(), 1, "only the active file is left: {files:?}");
        let recovery = WalRecovery::new(&sidecar);
        assert!(
            recovery
                .recover()
                .unwrap()
                .iter()
                .all(|record| !matches!(record, WalRecord::CreateNode { .. })),
            "records the file holds are not replayed again"
        );
        let checkpoint = recovery.checkpoint().expect("checkpoint.meta");
        assert_eq!(
            files[0].file_name().unwrap().to_str().unwrap(),
            format!("wal_{:08}.log", checkpoint.log_sequence),
            "recovery starts at the remaining file"
        );

        drop(db);
        let names = open(&path)
            .session()
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(
            names,
            vec![
                vec![Value::from("Alix")],
                vec![Value::from("Gus")],
                vec![Value::from("Vincent")],
            ]
        );
    }

    /// A checkpoint between a full and an incremental backup keeps the WAL
    /// files the incremental backup still has to copy.
    #[test]
    fn checkpoint_keeps_wal_needed_by_incremental_backup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let backups = backup_dir(&path);
        let db = open(&path);
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.backup_full(&backups).unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        db.wal_checkpoint().unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        let incremental = db.backup_incremental(&backups).unwrap();
        drop(session);
        drop(db);

        let restored_path = dir.path().join("restored.grafeo");
        GrafeoDB::restore_to_epoch(&backups, incremental.end_epoch, &restored_path).unwrap();
        let names = open(&restored_path)
            .session()
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(
            names,
            vec![
                vec![Value::from("Alix")],
                vec![Value::from("Gus")],
                vec![Value::from("Vincent")],
            ]
        );
    }

    /// Replaying records that the container already contains changes nothing,
    /// for every record type the sidecar can hold.
    #[test]
    fn replaying_records_already_in_the_container_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");

        let (alix, gus, knows, mia) = {
            let db = open(&path);
            let alix = db.create_node_with_props(&["Person"], [("name", Value::from("Alix"))]);
            db.add_node_label(alix, "Employee");
            let gus = db.create_node_with_props(&["Person"], [("name", Value::from("Gus"))]);
            let knows =
                db.create_edge_with_props(alix, gus, "KNOWS", [("since", Value::from(2020_i64))]);
            db.session().execute("CREATE GRAPH g").unwrap();
            let session = db.session();
            session.use_graph("g");
            let mia = session
                .create_node_with_props(&["Person"], [("name", Value::from("Mia"))])
                .unwrap();
            db.close().unwrap();
            (alix, gus, knows, mia)
        };
        assert!(!sidecar_wal(&path).exists(), "close() removes the sidecar");

        // A sidecar holding the same changes again, as after a checkpoint
        // whose WAL was not truncated.
        {
            let wal = WalManager::open(sidecar_wal(&path)).unwrap();
            let name = |id, name: &str| WalRecord::SetNodeProperty {
                id,
                key: "name".to_string(),
                value: Value::from(name),
            };
            let person = |id| WalRecord::CreateNode {
                id,
                labels: vec!["Person".to_string()],
            };
            let records = [
                person(alix),
                name(alix, "Alix"),
                WalRecord::AddNodeLabel {
                    id: alix,
                    label: "Employee".to_string(),
                },
                person(gus),
                name(gus, "Gus"),
                WalRecord::CreateEdge {
                    id: knows,
                    src: alix,
                    dst: gus,
                    edge_type: "KNOWS".to_string(),
                },
                WalRecord::SetEdgeProperty {
                    id: knows,
                    key: "since".to_string(),
                    value: Value::from(2020_i64),
                },
                WalRecord::CreateNamedGraph {
                    name: "g".to_string(),
                },
                WalRecord::SwitchGraph {
                    name: Some("g".to_string()),
                },
                person(mia),
                name(mia, "Mia"),
                WalRecord::SwitchGraph { name: None },
                WalRecord::TransactionCommit {
                    transaction_id: TransactionId::SYSTEM,
                },
            ];
            wal.log_batch(&records).unwrap();
        }

        let db = open(&path);
        assert_eq!(db.node_count(), 2);
        assert_eq!(db.edge_count(), 1);
        assert_eq!(edge_rows(&db), (Value::Int64(1), Value::Int64(1)));
        let knows_rows = db
            .session()
            .execute("MATCH (a)-[r:KNOWS]->(b) RETURN a.name, r.since, b.name")
            .unwrap();
        assert_eq!(
            knows_rows.rows().to_vec(),
            vec![vec![
                Value::from("Alix"),
                Value::from(2020_i64),
                Value::from("Gus")
            ]]
        );
        let labels = single_value(&db, "MATCH (n {name: 'Alix'}) RETURN size(labels(n))");
        assert_eq!(labels, Value::Int64(2), "Person and Employee, once each");

        let session = db.session();
        session.use_graph("g");
        let in_g = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(in_g.rows().to_vec(), vec![vec![Value::from("Mia")]]);
    }
}
