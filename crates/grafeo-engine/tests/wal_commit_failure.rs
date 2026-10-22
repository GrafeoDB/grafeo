//! A commit whose WAL write fails is not reported as committed (#498).
//!
//! The commit writes its group to the WAL, and syncs it in `Sync` mode,
//! before anything of it becomes visible. A failure has two outcomes:
//!
//! - the group was not written (its LAST frame is not in the log): the
//!   commit is rolled back and fails with `GRAFEO-T009`, retryable; nothing of
//!   it is visible, and a reopen finds nothing;
//! - the group was written but the sync failed: the commit may or may not be
//!   durable, so it fails with `GRAFEO-T010`, stays invisible, and nothing
//!   commits until the database is reopened; `close()` keeps the WAL, and the
//!   reopen replays the group the disk holds.
//!
//! A direct call that commits at once (no transaction open) cannot undo its
//! writes, so a failed WAL write poisons the database in both cases and fails
//! with `GRAFEO-T010`; a reopen finds what the log holds.
//!
//! A crash at any point of a commit leaves it whole or absent after the
//! reopen, and whole once the call returned (crashes run in a child process
//! that exits from its panic hook, so nothing unwinds into a cleanup).
//!
//! ```bash
//! cargo test -p grafeo-engine --features full,testing-crash-injection --test wal_commit_failure
//! ```

#![allow(
    missing_docs,
    reason = "a test crate has no public API: each test documents itself"
)]

#[cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "gql",
    feature = "testing-crash-injection"
))]
mod tests {
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at, with_failure_at};
    use grafeo_common::types::Value;
    use grafeo_common::utils::error::{Error, TransactionError};
    use grafeo_engine::config::DurabilityMode;
    use grafeo_engine::{Config, GrafeoDB};

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
            .unwrap()
    }

    /// The names of the people the database holds, sorted.
    fn people(db: &GrafeoDB) -> Vec<Value> {
        db.session()
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[0].clone())
            .collect()
    }

    /// A way to write Mia.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Write {
        /// An explicit transaction.
        Transaction,
        /// An auto-commit statement.
        Statement,
        /// A direct call, which commits at once.
        Direct,
        /// A direct call while a transaction is open: a private transaction.
        PrivateDirect,
    }

    fn write_mia(db: &GrafeoDB, write: Write) -> grafeo_common::utils::error::Result<()> {
        match write {
            Write::Transaction => {
                let mut session = db.session();
                session.begin_transaction()?;
                session.execute("INSERT (:Person {name: 'Mia'})")?;
                session.commit()
            }
            Write::Statement => db
                .session()
                .execute("INSERT (:Person {name: 'Mia'})")
                .map(|_| ()),
            Write::Direct => db
                .create_node_with_props(&["Person"], [("name", Value::from("Mia"))])
                .map(|_| ()),
            Write::PrivateDirect => {
                let mut open = db.session();
                open.begin_transaction()?;
                let created = db
                    .create_node_with_props(&["Person"], [("name", Value::from("Mia"))])
                    .map(|_| ());
                open.rollback()?;
                created
            }
        }
    }

    /// Opens a new database in `dir` with Alix, then writes Mia with the
    /// `count`-th injected failure armed; returns the database and the error
    /// when it names `point`, `None` when the failure fired elsewhere first.
    fn fail_once(dir: &Path, write: Write, count: u64, point: &str) -> Option<(GrafeoDB, Error)> {
        let path = dir.join(format!("{count}.grafeo"));
        let db = open(&path);
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        match with_failure_at(count, || write_mia(&db, write)) {
            Err(error) if error.to_string().contains(point) => Some((db, error)),
            Err(_) => None,
            Ok(()) => panic!("{write:?}: the write never reached {point} ({count} tries)"),
        }
    }

    /// Writes Mia with the first injected failure that hits `point`, on a new
    /// database each try, and returns the database and the error.
    fn fail_at(dir: &Path, write: Write, point: &str) -> (GrafeoDB, Error) {
        (1..=40)
            .find_map(|count| fail_once(dir, write, count, point))
            .unwrap_or_else(|| panic!("{write:?}: no failure hit {point}"))
    }

    fn the_path_of(db: &GrafeoDB) -> std::path::PathBuf {
        db.path().expect("a persistent database").to_path_buf()
    }

    /// The group was not written: the commit is rolled back with T009, the
    /// database takes the same write again, and a reopen holds what
    /// committed.
    fn a_write_that_fails_before_its_group_is_logged_rolls_back(write: Write) {
        let dir = tempfile::tempdir().unwrap();
        let (db, error) = fail_at(dir.path(), write, "wal:write");
        assert!(
            matches!(
                error,
                Error::Transaction(TransactionError::LogWriteFailed(_))
            ),
            "{write:?}: {error}"
        );
        assert!(
            error.to_string().contains("GRAFEO-T009"),
            "{write:?}: {error}"
        );
        assert_eq!(people(&db), vec![Value::from("Alix")], "{write:?}");

        // Nothing is left over: the same write commits now.
        write_mia(&db, write).unwrap();
        assert_eq!(
            people(&db),
            vec![Value::from("Alix"), Value::from("Mia")],
            "{write:?}"
        );
        let path = the_path_of(&db);
        db.close().unwrap();
        assert_eq!(
            people(&open(&path)),
            vec![Value::from("Alix"), Value::from("Mia")],
            "{write:?}: Mia once, after the reopen"
        );
    }

    /// The group was written but not synced: the commit fails with T010 and
    /// stays invisible, nothing commits until the reopen, `close()` keeps the
    /// WAL, and the reopen replays the group the disk holds.
    fn a_write_whose_sync_fails_has_an_unknown_outcome(write: Write, point: &str) {
        let dir = tempfile::tempdir().unwrap();
        let (db, error) = fail_at(dir.path(), write, point);
        assert!(
            matches!(
                error,
                Error::Transaction(TransactionError::CommitOutcomeUnknown(_))
            ),
            "{write:?}: {error}"
        );
        assert!(
            error.to_string().contains("GRAFEO-T010"),
            "{write:?}: {error}"
        );
        assert_eq!(
            people(&db),
            vec![Value::from("Alix")],
            "{write:?}: the commit is not visible"
        );
        let later = db.session().execute("INSERT (:Person {name: 'Butch'})");
        assert!(
            later.is_err(),
            "{write:?}: nothing commits until the reopen"
        );

        let path = the_path_of(&db);
        assert!(db.close().is_err(), "{write:?}: close() reports the commit");
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        assert!(
            Path::new(&sidecar).exists(),
            "{write:?}: close() keeps the WAL"
        );
        drop(db);
        assert_eq!(
            people(&open(&path)),
            vec![Value::from("Alix"), Value::from("Mia")],
            "{write:?}: the reopen replays the written group"
        );
    }

    #[test]
    fn a_transaction_whose_group_is_not_written_rolls_back() {
        a_write_that_fails_before_its_group_is_logged_rolls_back(Write::Transaction);
    }

    #[test]
    fn a_statement_whose_group_is_not_written_rolls_back() {
        a_write_that_fails_before_its_group_is_logged_rolls_back(Write::Statement);
    }

    #[test]
    fn a_private_direct_call_whose_group_is_not_written_rolls_back() {
        a_write_that_fails_before_its_group_is_logged_rolls_back(Write::PrivateDirect);
    }

    #[test]
    fn a_transaction_whose_sync_fails_has_an_unknown_outcome() {
        a_write_whose_sync_fails_has_an_unknown_outcome(Write::Transaction, "wal:sync");
    }

    #[test]
    fn a_statement_whose_sync_fails_has_an_unknown_outcome() {
        a_write_whose_sync_fails_has_an_unknown_outcome(Write::Statement, "wal:sync");
    }

    #[test]
    fn a_private_direct_call_whose_sync_fails_has_an_unknown_outcome() {
        a_write_whose_sync_fails_has_an_unknown_outcome(Write::PrivateDirect, "wal:sync");
    }

    #[test]
    fn a_direct_call_whose_sync_fails_has_an_unknown_outcome() {
        a_write_whose_sync_fails_has_an_unknown_outcome(Write::Direct, "wal:sync");
    }

    /// A direct call that commits at once cannot undo its writes: a group
    /// that was not written poisons the database (T010), the call is not
    /// visible, and the reopen, whose log holds nothing of it, does not have
    /// it either.
    #[test]
    fn a_direct_call_whose_group_is_not_written_poisons_the_database() {
        let dir = tempfile::tempdir().unwrap();
        let (db, error) = fail_at(dir.path(), Write::Direct, "wal:write");
        assert!(
            matches!(
                error,
                Error::Transaction(TransactionError::CommitOutcomeUnknown(_))
            ),
            "{error}"
        );
        assert_eq!(people(&db), vec![Value::from("Alix")]);
        assert!(
            db.session()
                .execute("INSERT (:Person {name: 'Butch'})")
                .is_err(),
            "nothing commits until the reopen"
        );
        let path = the_path_of(&db);
        assert!(db.close().is_err());
        drop(db);
        assert_eq!(people(&open(&path)), vec![Value::from("Alix")]);
    }

    const WRITE_VAR: &str = "GRAFEO_WAL_COMMIT_CRASH_WRITE";
    const POINT_VAR: &str = "GRAFEO_WAL_COMMIT_CRASH_POINT";
    const PATH_VAR: &str = "GRAFEO_WAL_COMMIT_CRASH_PATH";

    const WRITES: [Write; 4] = [
        Write::Transaction,
        Write::Statement,
        Write::Direct,
        Write::PrivateDirect,
    ];

    /// The file the child writes once the write returned, holding whether it
    /// succeeded.
    fn returned_marker(path: &Path) -> PathBuf {
        path.with_extension("returned")
    }

    /// The file the child writes when it crashes, holding the crash point.
    fn crashed_marker(path: &Path) -> PathBuf {
        path.with_extension("crashed")
    }

    /// Child-process entry for [`a_crash_in_a_commit_leaves_it_whole_or_absent`]:
    /// writes Alix, then Mia with a crash at the given point; a no-op when run
    /// directly.
    #[test]
    fn crash_child() {
        let (Ok(write), Ok(point), Some(path)) = (
            std::env::var(WRITE_VAR),
            std::env::var(POINT_VAR),
            std::env::var_os(PATH_VAR),
        ) else {
            return;
        };
        let write = WRITES[write.parse::<usize>().unwrap()];
        let path = PathBuf::from(path);
        // The crash exits here, before anything unwinds: no cut back of a
        // group, no close(). It records where it crashed.
        let crashed_at = crashed_marker(&path);
        std::panic::set_hook(Box::new(move |info| {
            eprintln!("{info}");
            let _ = std::fs::write(&crashed_at, info.to_string());
            std::process::exit(0);
        }));
        let db = open(&path);
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        let result = {
            let db = std::panic::AssertUnwindSafe(&db);
            with_crash_at(point.parse().unwrap(), move || write_mia(&db, write))
        };
        // Reached only when no crash fired: the write returned.
        let returned = matches!(result, CrashResult::Completed(Ok(())));
        std::fs::write(returned_marker(&path), returned.to_string()).unwrap();
        std::process::exit(0);
    }

    /// A crash at every point of each way to commit Mia, in `Sync` mode: the
    /// reopen holds Alix and Mia, or Alix alone when the call had not
    /// returned success, and both once the crash came after the commit's
    /// fsync (`wal:after_sync`, before it is published). The sweep ends at
    /// the first point the call no longer reaches.
    #[test]
    fn a_crash_in_a_commit_leaves_it_whole_or_absent() {
        for (index, write) in WRITES.iter().enumerate() {
            let (mut crashed, mut synced_crashes) = (0, 0);
            for point in 1..=64 {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("db.grafeo");
                let status = child_process::run(
                    std::process::Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", "tests::crash_child", "--nocapture"])
                        .env(WRITE_VAR, index.to_string())
                        .env(POINT_VAR, point.to_string())
                        .env(PATH_VAR, &path),
                )
                .unwrap();
                assert!(
                    status.success(),
                    "{write:?}, point {point}: the child failed"
                );

                let returned = std::fs::read_to_string(returned_marker(&path)).ok();
                let found = people(&open(&path));
                let alix = vec![Value::from("Alix")];
                let both = vec![Value::from("Alix"), Value::from("Mia")];
                match returned.as_deref() {
                    Some("true") => {
                        assert_eq!(found, both, "{write:?}, point {point}: a returned commit");
                        break;
                    }
                    Some(other) => panic!("{write:?}, point {point}: the write failed ({other})"),
                    None => {
                        crashed += 1;
                        let at = std::fs::read_to_string(crashed_marker(&path)).unwrap_or_default();
                        if at.contains("wal:after_sync") {
                            synced_crashes += 1;
                            assert_eq!(found, both, "{write:?}, a crash after the fsync: {at}");
                        }
                        assert!(
                            found == alix || found == both,
                            "{write:?}, crash point {point}: {found:?}"
                        );
                    }
                }
            }
            assert!(
                crashed >= 2,
                "{write:?}: only {crashed} crash points in its commit"
            );
            assert!(
                synced_crashes >= 1,
                "{write:?}: no crash after the commit's fsync"
            );
        }
    }
}
