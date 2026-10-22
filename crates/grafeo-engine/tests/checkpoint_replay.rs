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
    use grafeo_common::testing::child_process;
    use grafeo_common::types::{TransactionId, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use grafeo_storage::wal::{
        ScanOptions, WalManager, WalRecord, WalScan, parse_segment_file_name, segment_file_name,
    };
    use std::path::{Path, PathBuf};

    const SCENARIO_VAR: &str = "GRAFEO_CHECKPOINT_REPLAY_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_CHECKPOINT_REPLAY_PATH";

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path).with_storage_format(StorageFormat::Auto))
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
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "tests::crash_child", "--nocapture"])
                .env(SCENARIO_VAR, scenario)
                .env(PATH_VAR, path),
        )
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
            // The issue's sequence in a named graph.
            "named_graph_checkpoint_then_write" => {
                session.execute("CREATE GRAPH g").unwrap();
                session.use_graph("g");
                session
                    .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
                    .unwrap();
                db.wal_checkpoint().unwrap();
                session
                    .execute(
                        "INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Jules'})",
                    )
                    .unwrap();
            }
            // An edge the checkpoint holds, deleted after it.
            "checkpoint_then_delete_edge" => {
                session
                    .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
                    .unwrap();
                db.wal_checkpoint().unwrap();
                session
                    .execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() DELETE r")
                    .unwrap();
            }
            // Reopen (replaying the WAL) and crash again, checking nothing.
            "reopen_only" => {}
            // Reopen (replaying the WAL), check the edges and crash again.
            "reopen" => {
                assert_eq!(edge_rows(&db), (Value::Int64(3), Value::Int64(3)));
                assert_eq!(db.edge_count().unwrap(), 3);
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
        assert_eq!(db.edge_count().unwrap(), 1);
        assert_eq!(db.node_count().unwrap(), 3);
    }

    /// Every file under `dir` with its bytes.
    fn files(dir: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
        let mut found = std::collections::BTreeMap::new();
        let mut pending = vec![dir.to_path_buf()];
        while let Some(next) = pending.pop() {
            for entry in std::fs::read_dir(&next).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    found.insert(path.clone(), std::fs::read(&path).unwrap());
                }
            }
        }
        found
    }

    /// The issue's crash, reopened read-only: the replay into memory adds
    /// no edge twice, and the file and its WAL keep every byte; a read-write
    /// open afterwards sees the same.
    #[test]
    fn crash_after_wal_checkpoint_read_only_reopen_does_not_duplicate_edges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        crash_after("checkpoint_then_write", &path);
        assert!(sidecar_wal(&path).exists(), "the crash left the WAL");
        let before = files(dir.path());

        let db = GrafeoDB::open_read_only(&path).unwrap();
        assert_eq!(knows(&db), [[Value::from("Alix"), Value::from("Gus")]]);
        assert_eq!(edge_rows(&db), (Value::Int64(1), Value::Int64(1)));
        assert_eq!((db.node_count().unwrap(), db.edge_count().unwrap()), (3, 1));
        db.close().unwrap();
        drop(db);
        assert!(
            files(dir.path()) == before,
            "a read-only open changes no file"
        );

        let db = open(&path);
        assert_eq!(knows(&db), [[Value::from("Alix"), Value::from("Gus")]]);
        assert_eq!((db.node_count().unwrap(), db.edge_count().unwrap()), (3, 1));
    }

    /// The issue's sequence in a named graph, then two more crashes after a
    /// reopen each: both edges of the graph are there once, and the default
    /// graph holds none.
    #[test]
    fn crash_after_wal_checkpoint_does_not_duplicate_named_graph_edges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        crash_after("named_graph_checkpoint_then_write", &path);
        for _ in 0..2 {
            crash_after("reopen_only", &path);
        }

        let db = open(&path);
        let session = db.session();
        session.use_graph("g");
        assert_eq!(
            session
                .execute("MATCH (a)-[r:KNOWS]->(b) RETURN a.name, b.name, count(DISTINCT id(r)) ORDER BY a.name")
                .unwrap()
                .rows()
                .to_vec(),
            [
                [Value::from("Alix"), Value::from("Gus"), Value::Int64(1)],
                [Value::from("Vincent"), Value::from("Jules"), Value::Int64(1)],
            ]
        );
        assert_eq!(
            session
                .execute("MATCH ()-[r]->() RETURN count(r)")
                .unwrap()
                .rows()[0][0],
            Value::Int64(2),
            "every edge of g once"
        );
        assert_eq!(
            edge_rows(&db),
            (Value::Int64(0), Value::Int64(0)),
            "the default graph"
        );
    }

    /// An edge the checkpoint holds and the WAL deletes stays deleted, over
    /// two more crashes after a reopen each: replaying the delete again finds
    /// no edge to delete and adds none back.
    #[test]
    fn crash_after_deleting_a_checkpointed_edge_keeps_it_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        crash_after("checkpoint_then_delete_edge", &path);
        for _ in 0..2 {
            crash_after("reopen_only", &path);
        }

        let db = open(&path);
        assert_eq!(edge_rows(&db), (Value::Int64(0), Value::Int64(0)));
        assert_eq!((db.node_count().unwrap(), db.edge_count().unwrap()), (2, 0));
        assert_eq!(
            db.session()
                .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
                .unwrap()
                .rows()
                .to_vec(),
            [[Value::from("Alix")], [Value::from("Gus")]]
        );
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
        assert_eq!(db.edge_count().unwrap(), 3);
        assert_eq!(db.node_count().unwrap(), 6);
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
                db.edge_count().unwrap(),
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

    /// Writes Alix and Gus and checkpoints, then writes Vincent with a query
    /// and Mia, Butch and their edge with direct calls: the writes a failed
    /// checkpoint must not lose.
    fn write_and_checkpoint_then_write_more(db: &GrafeoDB) {
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();
        db.wal_checkpoint().unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        let mia = db
            .create_node_with_props(&["Person"], [("name", Value::from("Mia"))])
            .unwrap();
        let butch = db
            .create_node_with_props(&["Person"], [("name", Value::from("Butch"))])
            .unwrap();
        db.create_edge(mia, butch, "LIKES").unwrap();
    }

    /// Every write of [`write_and_checkpoint_then_write_more`] and Jules,
    /// written after the failed checkpoint, is there once.
    fn assert_every_write_once(db: &GrafeoDB, what: &str) {
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
                vec![Value::from("Butch")],
                vec![Value::from("Gus")],
                vec![Value::from("Jules")],
                vec![Value::from("Mia")],
                vec![Value::from("Vincent")],
            ],
            "{what}"
        );
        assert_eq!(
            knows(db),
            vec![vec![Value::from("Alix"), Value::from("Gus")]],
            "{what}"
        );
        assert_eq!(
            db.edge_count().unwrap(),
            2,
            "{what}: KNOWS and LIKES, once each"
        );
    }

    /// Copies the files of the directory `from` into a new directory `to`,
    /// reading each through a handle that shares it with the open database.
    #[cfg(feature = "testing-crash-injection")]
    fn copy_files(from: &Path, to: &Path) {
        std::fs::create_dir(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let bytes = std::fs::read(entry.path()).unwrap();
            std::fs::write(to.join(entry.file_name()), bytes).unwrap();
        }
    }

    /// A checkpoint that fails inside its image write, at `point` (before or
    /// after the new header is written), returns the error and keeps the WAL
    /// it would have truncated. The database stays usable for query writes
    /// and direct writes alike, and every write reads back once after a
    /// reopen: from the file and WAL the failure left (as after a crash at
    /// that moment), and after a clean close (#418, #424).
    #[cfg(feature = "testing-crash-injection")]
    fn a_failed_image_write_keeps_the_database_readable(point: &str) {
        use grafeo_common::testing::crash::with_failure_at;

        // The failure points before `point` (the WAL's rotation among them)
        // are tried on a new database each, until the failure hits `point`.
        for count in 1..=16 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db.grafeo");
            let db = open(&path);
            write_and_checkpoint_then_write_more(&db);
            let files_before = log_files(&sidecar_wal(&path));
            let error = with_failure_at(count, || db.wal_checkpoint())
                .unwrap_err()
                .to_string();
            if error.contains(point) {
                after_a_failed_image_write(dir, db, &files_before, point);
                return;
            }
        }
        panic!("no checkpoint failed at {point}");
    }

    /// The checks of [`a_failed_image_write_keeps_the_database_readable`] on
    /// `db`, whose checkpoint failed at `point`.
    #[cfg(feature = "testing-crash-injection")]
    fn after_a_failed_image_write(
        dir: tempfile::TempDir,
        db: GrafeoDB,
        files_before: &[PathBuf],
        point: &str,
    ) {
        let path = dir.path().join("db.grafeo");
        let crashed = dir.path().join("crashed.grafeo");
        {
            let sidecar = sidecar_wal(&path);
            db.session()
                .execute("INSERT (:Person {name: 'Jules'})")
                .unwrap();

            // The failed checkpoint started a new WAL segment, and deleted
            // none.
            let files_after = log_files(&sidecar);
            assert!(
                files_before.iter().all(|file| files_after.contains(file)),
                "{point}: the WAL lost segments: {files_before:?}, then {files_after:?}"
            );
            assert_eq!(
                files_after.len(),
                files_before.len() + 1,
                "{point}: the checkpoint rotated the WAL"
            );

            // What a crash at this moment leaves: the file and its WAL.
            db.wal().unwrap().sync().unwrap();
            db.file_manager().unwrap().copy_to(&crashed).unwrap();
            copy_files(&sidecar, &sidecar_wal(&crashed));
            db.close().unwrap();
        }
        assert_every_write_once(&open(&crashed), &format!("{point}, as after a crash"));
        assert_every_write_once(&open(&path), &format!("{point}, after close"));
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn failed_checkpoint_before_its_header_keeps_the_database_readable() {
        a_failed_image_write_keeps_the_database_readable("checkpoint:after_chunks");
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn failed_checkpoint_after_its_header_keeps_the_database_readable() {
        a_failed_image_write_keeps_the_database_readable("checkpoint:after_header");
    }

    /// A `close()` whose checkpoint fails inside its image write, at `point`
    /// (before or after the new header), returns the error and leaves a
    /// database that reads back every committed write once: from the file
    /// and WAL the failure left (as after a crash right then), and twice in
    /// a row, so the first reopen's own checkpoint writes nothing twice
    /// (#418, whose `close()` hit a full disk). With `open_transaction`, a
    /// session has Hans inserted and not committed when `close()` runs: he
    /// is in neither.
    #[cfg(feature = "testing-crash-injection")]
    fn a_failed_close_keeps_the_database_readable(point: &str, open_transaction: bool) {
        use grafeo_common::testing::crash::with_failure_at;

        for count in 1..=16 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db.grafeo");
            let crashed = dir.path().join("crashed.grafeo");
            let db = open(&path);
            write_and_checkpoint_then_write_more(&db);
            db.session()
                .execute("INSERT (:Person {name: 'Jules'})")
                .unwrap();
            let mut session = db.session();
            if open_transaction {
                session.begin_transaction().unwrap();
                session.execute("INSERT (:Person {name: 'Hans'})").unwrap();
            }
            let files_before = log_files(&sidecar_wal(&path));
            let Err(error) = with_failure_at(count, || db.close()) else {
                panic!("no close() failed at {point}");
            };
            if !error.to_string().contains(point) {
                continue;
            }
            assert!(
                files_before
                    .iter()
                    .all(|file| log_files(&sidecar_wal(&path)).contains(file)),
                "{point}: the failed close() deleted WAL segments"
            );
            // What a crash right after the failure leaves: the file and WAL.
            db.file_manager().unwrap().copy_to(&crashed).unwrap();
            copy_files(&sidecar_wal(&path), &sidecar_wal(&crashed));
            drop(session);
            drop(db);

            for reopen in ["first reopen", "second reopen"] {
                let what = format!("{point}, open transaction {open_transaction}, {reopen}");
                let reopened = open(&crashed);
                assert_every_write_once(&reopened, &what);
                reopened.close().unwrap();
            }
            return;
        }
        panic!("no close() failed at {point}");
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_failed_close_inside_its_image_write_keeps_the_database_readable() {
        for point in ["checkpoint:after_chunks", "checkpoint:after_header"] {
            a_failed_close_keeps_the_database_readable(point, false);
        }
    }

    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_failed_close_with_an_open_transaction_keeps_only_committed_writes() {
        for point in ["checkpoint:after_chunks", "checkpoint:after_header"] {
            a_failed_close_keeps_the_database_readable(point, true);
        }
    }

    /// Checkpoints that fail before anything is written (the new WAL file
    /// each one starts cannot be created, as on a full disk), including those
    /// of `close()` and `Drop`, leave every write in the WAL, which a reopen
    /// replays once.
    #[test]
    fn checkpoints_whose_wal_cannot_rotate_keep_the_database_readable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let mut blockers = Vec::new();
        {
            // No background flusher, whose syncs would write markers.
            let db = GrafeoDB::with_config(
                Config::persistent(&path)
                    .with_storage_format(StorageFormat::Auto)
                    .with_wal_durability(grafeo_engine::config::DurabilityMode::NoSync),
            )
            .unwrap();
            write_and_checkpoint_then_write_more(&db);

            // Every checkpoint from here on fails: the WAL segment it starts,
            // at the end of the log, is a directory. Three are tried:
            // `wal_checkpoint()`, then after Jules `close()` and the `close()`
            // of `Drop`, both at the same end.
            // A sync writes its marker first, so the end of the log stays
            // where the checkpoint starts its segment.
            let block = |blockers: &mut Vec<PathBuf>| {
                db.wal().unwrap().sync().unwrap();
                let end = db.wal().unwrap().end_lsn();
                let blocker = sidecar_wal(&path).join(segment_file_name(end));
                std::fs::create_dir(&blocker).unwrap();
                blockers.push(blocker);
            };
            block(&mut blockers);
            assert!(db.wal_checkpoint().is_err(), "the checkpoint must fail");
            db.session()
                .execute("INSERT (:Person {name: 'Jules'})")
                .unwrap();
            block(&mut blockers);
            assert!(db.close().is_err(), "close() checkpoints too");
        }
        for blocker in &blockers {
            std::fs::remove_dir(blocker).unwrap();
        }
        assert_every_write_once(&open(&path), "reopened from the WAL");
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
        assert_eq!(files.len(), 1, "only the active segment is left: {files:?}");
        let fm = db.file_manager().unwrap();
        let checkpoint_lsn = fm.active_header().checkpoint_lsn;
        assert!(checkpoint_lsn > 0, "the header records the log position");
        assert_eq!(
            parse_segment_file_name(files[0].file_name().unwrap().to_str().unwrap()),
            Some(checkpoint_lsn),
            "replay starts at the remaining segment"
        );
        let mut scan =
            WalScan::open(&sidecar, ScanOptions::new(fm.database_id(), checkpoint_lsn)).unwrap();
        assert!(
            scan.next_group().unwrap().is_none(),
            "no group the file holds is replayed again"
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
    /// segments the incremental backup still has to copy.
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
            let alix = db
                .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
                .unwrap();
            db.add_node_label(alix, "Employee").unwrap();
            let gus = db
                .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
                .unwrap();
            let knows = db
                .create_edge_with_props(alix, gus, "KNOWS", [("since", Value::from(2020_i64))])
                .unwrap();
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

        // A sidecar of the WAL format of 0.5.x and 0.6.0-beta.1 holding the
        // same changes again, as after a checkpoint whose WAL was not
        // truncated: its records are replayed once, and it is retired.
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
        assert_eq!(db.node_count().unwrap(), 2);
        assert_eq!(db.edge_count().unwrap(), 1);
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
