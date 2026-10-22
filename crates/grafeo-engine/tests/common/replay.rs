//! WAL replay in tests: the writing part of a test runs in a child process
//! that exits without `close()`, and the test reopens the database.
//!
//! A database file checkpoints on `close()` and on `Drop`, so a reopen in the
//! same process reads the checkpoint, not the WAL. Replay happens only after a
//! process exits without closing, as after a crash.

use std::path::Path;

use grafeo_common::testing::child_process;
use grafeo_engine::GrafeoDB;

/// Tells a test that [`reopened_after_crash`] runs in a child process where
/// its database is.
const CHILD_PATH_VAR: &str = "GRAFEO_REPLAY_TEST_PATH";

/// Runs `write` on a new database, opened with `open`, in a child process
/// that exits without `close()`, so nothing is checkpointed; then reopens the
/// database here with `open`, which replays its sidecar WAL.
///
/// `test` is the path of the calling test (`module::name`): the child runs it
/// with `--exact`, and with `--include-ignored` so an ignored test runs its
/// child too. In the child this function does not return, so call it once per
/// test, before anything else with effects. Bind the result as `(_dir, db)`:
/// the database must close before its directory goes.
///
/// # Panics
///
/// If the child process fails, or creates no database (`test` does not name
/// the calling test).
pub fn reopened_after_crash(
    test: &str,
    open: impl Fn(&Path) -> GrafeoDB,
    write: impl FnOnce(&GrafeoDB),
) -> (tempfile::TempDir, GrafeoDB) {
    if let Some(path) = std::env::var_os(CHILD_PATH_VAR) {
        let db = open(Path::new(&path));
        write(&db);
        // Crash: no close(), no checkpoint, no destructors.
        std::process::exit(0);
    }
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("db.grafeo");
    let status = child_process::run(
        std::process::Command::new(std::env::current_exe().expect("the test executable"))
            .args(["--exact", test, "--include-ignored", "--nocapture"])
            .env(CHILD_PATH_VAR, &path),
    )
    .expect("run the child process");
    assert!(status.success(), "{test}: the child process failed");
    assert!(
        path.exists(),
        "{test}: the child process created no database; does `{test}` name the calling test?"
    );
    let db = open(&path);
    (dir, db)
}
