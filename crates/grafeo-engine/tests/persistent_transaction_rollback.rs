//! Rolling back a transaction on a persistent database restores property and
//! label changes, the same as in memory.
//!
//! Persistent sessions write through a WAL-logging store wrapper. It must keep
//! the transactional (undo-recording) path for property and label changes,
//! otherwise a rolled-back `SET` / `REMOVE` stays applied.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test persistent_transaction_rollback
//! ```

#![allow(missing_docs)]

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod tests {
    use grafeo_common::testing::child_process;
    use grafeo_common::types::Value;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use std::path::{Path, PathBuf};

    /// The persistent databases the tests run on: a name, and whether the
    /// database records changes (CDC).
    const VARIANTS: [(&str, bool); 2] = [("single file", false), ("single file + cdc", true)];

    fn config(path: &Path, cdc: bool) -> Config {
        let config = Config::persistent(path).with_storage_format(StorageFormat::Auto);
        if cdc { config.with_cdc() } else { config }
    }

    /// Node and edge state as `(labels, n.v, n.w, n.x, r.weight, r.note)`.
    fn state(db: &GrafeoDB) -> Vec<Value> {
        let result = db
            .session()
            .execute(
                "MATCH (n:Person {name: 'Alix'})-[r:KNOWS]->() \
                 RETURN labels(n), n.v, n.w, n.x, r.weight, r.note",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        let mut row = result.rows()[0].clone();
        if let Value::List(labels) = &row[0] {
            let mut sorted: Vec<Value> = labels.to_vec();
            sorted.sort_by_key(|v| v.to_string());
            row[0] = Value::List(sorted.into());
        }
        row
    }

    fn seed(db: &GrafeoDB) {
        db.session()
            .execute(
                "INSERT (:Person:Base {name: 'Alix', v: 1, x: 'keep'})\
                 -[:KNOWS {weight: 5, note: 'n'}]->(:Person {name: 'Gus'})",
            )
            .unwrap();
    }

    fn change_everything(session: &grafeo_engine::session::Session) {
        for query in [
            "MATCH (n:Person {name: 'Alix'}) SET n.v = 2",
            "MATCH (n:Person {name: 'Alix'}) SET n.w = 3",
            "MATCH (n:Person {name: 'Alix'}) REMOVE n.x",
            "MATCH (n:Person {name: 'Alix'}) SET n:Extra",
            "MATCH (n:Person {name: 'Alix'}) REMOVE n:Base",
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() SET r.weight = 9",
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() REMOVE r.note",
        ] {
            session
                .execute(query)
                .unwrap_or_else(|e| panic!("{query}: {e}"));
        }
    }

    /// Seeds, changes everything in a transaction, and commits it or rolls
    /// it back.
    fn write(db: &GrafeoDB, commit: bool) {
        seed(db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        change_everything(&session);
        if commit {
            session.commit().unwrap();
        } else {
            session.rollback().unwrap();
        }
    }

    /// The state [`write`] leaves, from an in-memory database: the seed alone
    /// after a rollback (the in-memory control checks that rollback), the
    /// changed state after a commit.
    fn expected(commit: bool) -> Vec<Value> {
        let db = GrafeoDB::new_in_memory();
        if commit {
            write(&db, true);
        } else {
            seed(&db);
        }
        state(&db)
    }

    /// [`write`], then a check that the writer sees the state a reopen must
    /// rebuild.
    fn write_and_check(db: &GrafeoDB, commit: bool) {
        write(db, commit);
        assert_eq!(state(db), expected(commit), "live state");
    }

    /// How the process that wrote a database ends before the reopen.
    #[derive(Clone, Copy, Debug)]
    enum End {
        /// `close()` checkpoints: the reopen reads the file.
        Close,
        /// The process exits without `close()` (a child process), so nothing
        /// is checkpointed: the reopen replays the sidecar WAL.
        Crash,
    }

    const PATH_VAR: &str = "GRAFEO_PERSISTENT_ROLLBACK_PATH";
    const CDC_VAR: &str = "GRAFEO_PERSISTENT_ROLLBACK_CDC";
    const COMMIT_VAR: &str = "GRAFEO_PERSISTENT_ROLLBACK_COMMIT";

    /// Runs [`write`] on a new database at `path` and ends the process that
    /// wrote it as `end` says.
    fn write_then(end: End, path: &Path, cdc: bool, commit: bool) {
        match end {
            End::Close => {
                let db = GrafeoDB::with_config(config(path, cdc)).unwrap();
                write_and_check(&db, commit);
                db.close().unwrap();
            }
            End::Crash => {
                let status = child_process::run(
                    std::process::Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", "tests::crash_child", "--nocapture"])
                        .env(PATH_VAR, path)
                        .env(CDC_VAR, cdc.to_string())
                        .env(COMMIT_VAR, commit.to_string()),
                )
                .unwrap();
                assert!(status.success(), "the child process failed");
            }
        }
    }

    /// Child-process entry for [`write_then`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let flag = |var: &str| std::env::var(var).unwrap() == "true";
        let db = GrafeoDB::with_config(config(&PathBuf::from(path), flag(CDC_VAR))).unwrap();
        write_and_check(&db, flag(COMMIT_VAR));
        // Crash: no close(), no checkpoint, no destructors.
        std::process::exit(0);
    }

    #[test]
    fn rollback_restores_properties_and_labels() {
        let dir = tempfile::tempdir().unwrap();
        for (name, cdc) in VARIANTS {
            let path = dir.path().join(format!("cdc-{cdc}.grafeo"));
            let db = GrafeoDB::with_config(config(&path, cdc)).unwrap();
            seed(&db);
            let before = state(&db);

            let mut session = db.session();
            session.begin_transaction().unwrap();
            change_everything(&session);
            assert_ne!(
                state(&db),
                before,
                "{name}: changes are applied in the transaction"
            );
            session.rollback().unwrap();

            assert_eq!(state(&db), before, "{name}: rollback restores everything");
        }
    }

    /// Control: the same rollback on an in-memory database.
    #[test]
    fn rollback_restores_properties_and_labels_in_memory() {
        let db = GrafeoDB::new_in_memory();
        seed(&db);
        let before = state(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        change_everything(&session);
        session.rollback().unwrap();
        assert_eq!(state(&db), before);
    }

    #[test]
    fn rollback_is_not_replayed_after_reopen() {
        let before = expected(false);
        let dir = tempfile::tempdir().unwrap();
        for (name, cdc) in VARIANTS {
            for end in [End::Close, End::Crash] {
                let path = dir.path().join(format!("{end:?}-cdc-{cdc}.grafeo"));
                write_then(end, &path, cdc, false);

                let db = GrafeoDB::with_config(config(&path, cdc)).unwrap();
                assert_eq!(state(&db), before, "{name}, {end:?}: reopened state");
            }
        }
    }

    #[test]
    fn commit_persists_changes() {
        let after = expected(true);
        let dir = tempfile::tempdir().unwrap();
        for (name, cdc) in VARIANTS {
            for end in [End::Close, End::Crash] {
                let path = dir.path().join(format!("{end:?}-cdc-{cdc}.grafeo"));
                write_then(end, &path, cdc, true);

                let db = GrafeoDB::with_config(config(&path, cdc)).unwrap();
                assert_eq!(
                    state(&db),
                    after,
                    "{name}, {end:?}: committed changes survive reopen"
                );
            }
        }
    }
}
