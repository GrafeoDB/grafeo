//! The history retention limit: `Config::with_history_retention_limit(bytes)`, per open. Once
//! the commits since the window's start replaced more of the labeled property graph's data than
//! the limit, the start moves forward, whichever of time or bytes comes first, and a read before
//! it is refused as one before the window. Without the setting the window keeps what its
//! retention says.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main history_retention_limit::
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::config::ConfigError;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

const KIB: usize = 1024;

/// A database that keeps history forever, with the history retention limit `limit`, holding
/// Alix (`bio` "Amsterdam"); and the epoch of that commit.
fn open(limit: Option<u64>) -> (GrafeoDB, EpochId) {
    let mut config = Config::in_memory().with_history_retention(HistoryRetention::Forever);
    if let Some(bytes) = limit {
        config = config.with_history_retention_limit(bytes);
    }
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})")
        .unwrap();
    let epoch = db.current_epoch();
    (db, epoch)
}

/// Sets Alix's `bio` to round `round`'s string of `bytes` bytes, in a commit of its own, and
/// returns that commit's epoch.
fn set_bio(db: &GrafeoDB, round: usize, bytes: usize) -> EpochId {
    db.execute_with_params(
        "MATCH (p:Person {name: 'Alix'}) SET p.bio = $bio",
        HashMap::from([("bio".to_string(), bio(round, bytes))]),
    )
    .unwrap();
    db.current_epoch()
}

fn bio(round: usize, bytes: usize) -> Value {
    Value::from(format!("{round:04}{}", "V".repeat(bytes - 4)).as_str())
}

/// Alix's `bio` at `epoch`, or the refusal's text.
fn bio_at(db: &GrafeoDB, epoch: EpochId) -> Result<Value, String> {
    db.execute_at_epoch("MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio", epoch)
        .map(|result| result.rows()[0][0].clone())
        .map_err(|error| error.to_string())
}

fn assert_not_retained(read: Result<Value, String>, what: &str) {
    match read {
        Err(text) => assert!(text.contains("is not retained"), "{what}: {text}"),
        Ok(value) => panic!("{what}: answered {value:?}"),
    }
}

#[test]
fn a_zero_limit_is_refused() {
    let config = Config::in_memory().with_history_retention_limit(0);
    assert_eq!(
        config.validate(),
        Err(ConfigError::ZeroHistoryRetentionLimit)
    );
    assert!(GrafeoDB::with_config(config).is_err());
}

/// Without the setting the window keeps what its retention says: forever.
#[test]
fn without_the_limit_the_window_keeps_its_length() {
    let (db, start) = open(None);
    for round in 0..40 {
        set_bio(&db, round, 64 * KIB);
    }
    db.gc();
    assert_eq!(bio_at(&db, start), Ok(Value::from("Amsterdam")));
}

/// With 1 MiB, forty commits replacing 64 KiB each move the start forward: the first epoch is
/// refused, and one after which ten values were replaced still reads its own.
#[test]
fn the_limit_moves_the_start_of_the_window() {
    let (db, start) = open(Some(1 << 20));
    let epochs: Vec<EpochId> = (0..40).map(|round| set_bio(&db, round, 64 * KIB)).collect();
    db.gc();
    assert_not_retained(bio_at(&db, start), "the first epoch");
    assert_not_retained(bio_at(&db, epochs[20]), "twenty values since");
    assert_eq!(
        bio_at(&db, epochs[29]),
        Ok(bio(29, 64 * KIB)),
        "ten values since"
    );
    assert_eq!(bio_at(&db, epochs[39]), Ok(bio(39, 64 * KIB)), "the latest");
}

/// One commit that replaces more than the limit moves the start to its own epoch, and no
/// further: the epoch before it is refused, its own reads.
#[test]
fn one_commit_larger_than_the_limit_moves_the_start_to_itself() {
    let (db, _) = open(Some(100 * KIB as u64));
    let before = set_bio(&db, 0, 1 << 20);
    let large = set_bio(&db, 1, 1 << 20);
    db.gc();
    assert_not_retained(bio_at(&db, before), "the epoch before it");
    assert_eq!(bio_at(&db, large), Ok(bio(1, 1 << 20)));
}

/// An open transaction older than the start keeps its snapshot (the snapshot retention limit
/// bounds it, not the history's), and the floor stays at it while it is open; once it ends the
/// start applies.
#[test]
fn a_reader_older_than_the_start_keeps_its_snapshot() {
    let (db, start) = open(Some(1 << 20));
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let read = "MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio";
    assert_eq!(
        reader.execute(read).unwrap().rows()[0][0],
        Value::from("Amsterdam")
    );
    for round in 0..40 {
        set_bio(&db, round, 64 * KIB);
    }
    db.gc();
    assert_eq!(
        reader.execute(read).unwrap().rows()[0][0],
        Value::from("Amsterdam"),
        "its snapshot"
    );
    assert_eq!(
        bio_at(&db, start),
        Ok(Value::from("Amsterdam")),
        "the floor stays at the open reader"
    );
    reader.rollback().unwrap();
    db.gc();
    assert_not_retained(bio_at(&db, start), "once it ended");
}

/// A reopen counts the history the file holds and the log replays as the commits that made it
/// did: the same writes give the same answers at every epoch with and without one.
#[cfg(feature = "grafeo-file")]
mod reopen {
    use std::path::Path;

    use super::{KIB, assert_not_retained, bio, bio_at, set_bio};
    use grafeo_common::types::{EpochId, Value};
    use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

    /// The file database at `path`, keeping history forever under `limit`.
    pub(super) fn persistent(path: &Path, limit: Option<u64>) -> GrafeoDB {
        let mut config = Config::persistent(path).with_history_retention(HistoryRetention::Forever);
        if let Some(bytes) = limit {
            config = config.with_history_retention_limit(bytes);
        }
        GrafeoDB::with_config(config).unwrap()
    }

    /// Alix with `bio` "Amsterdam" in a new file database at `path`, and that commit's epoch.
    pub(super) fn created(path: &Path, limit: Option<u64>) -> (GrafeoDB, EpochId) {
        let db = persistent(path, limit);
        db.execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})")
            .unwrap();
        let epoch = db.current_epoch();
        (db, epoch)
    }

    /// What a read at each of `epochs` answers: the value's first characters, or "refused".
    pub(super) fn answers(db: &GrafeoDB, epochs: &[EpochId]) -> Vec<String> {
        epochs
            .iter()
            .map(|epoch| match bio_at(db, *epoch) {
                Ok(Value::String(text)) => text.chars().take(9).collect(),
                Ok(other) => format!("{other:?}"),
                Err(text) if text.contains("is not retained") => "refused".to_string(),
                Err(text) => panic!("{text}"),
            })
            .collect()
    }

    /// Forty rewrites of 64 KiB under 1 MiB in one run of a database, and the answers at the
    /// first epoch and every rewrite's.
    pub(super) fn reference() -> (Vec<EpochId>, Vec<String>) {
        let dir = tempfile::tempdir().unwrap();
        let (db, start) = created(&dir.path().join("one.grafeo"), Some(1 << 20));
        let mut epochs = vec![start];
        epochs.extend((0..40).map(|round| set_bio(&db, round, 64 * KIB)));
        db.gc();
        let answers = answers(&db, &epochs);
        db.close().unwrap();
        (epochs, answers)
    }

    /// The reopen comes after 35 of the 40 rewrites, so the 5 after it alone would leave the
    /// loaded epochs inside the window: the answers match only when the loaded history counts.
    #[test]
    fn a_reopen_reads_as_the_same_writes_without_one() {
        let (epochs, expected) = reference();
        assert!(expected.contains(&"refused".to_string()), "{expected:?}");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reopened.grafeo");
        let (db, start) = created(&path, Some(1 << 20));
        let mut written = vec![start];
        written.extend((0..35).map(|round| set_bio(&db, round, 64 * KIB)));
        db.close().unwrap();
        let db = persistent(&path, Some(1 << 20));
        written.extend((35..40).map(|round| set_bio(&db, round, 64 * KIB)));
        db.gc();
        assert_eq!(written, epochs, "the same epochs");
        assert_eq!(answers(&db, &epochs), expected);
        db.close().unwrap();
    }

    /// A lower limit at a reopen drops the oldest of the loaded history at the open.
    #[test]
    fn a_lower_limit_at_a_reopen_drops_the_oldest_history() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lower.grafeo");
        let (db, start) = created(&path, None);
        let epochs: Vec<EpochId> = (0..40).map(|round| set_bio(&db, round, 64 * KIB)).collect();
        db.close().unwrap();
        let db = persistent(&path, Some(1 << 20));
        assert_not_retained(bio_at(&db, start), "the first epoch");
        assert_not_retained(bio_at(&db, epochs[20]), "twenty values since");
        assert_eq!(bio_at(&db, epochs[29]), Ok(bio(29, 64 * KIB)));
        db.close().unwrap();
    }

    /// A higher limit at a reopen brings nothing back.
    #[test]
    fn a_higher_limit_at_a_reopen_brings_nothing_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("higher.grafeo");
        let (db, _) = created(&path, Some(1 << 20));
        let epochs: Vec<EpochId> = (0..40).map(|round| set_bio(&db, round, 64 * KIB)).collect();
        db.gc();
        db.close().unwrap();
        let db = persistent(&path, Some(64 << 20));
        assert_not_retained(bio_at(&db, epochs[20]), "dropped before the reopen");
        assert_eq!(bio_at(&db, epochs[29]), Ok(bio(29, 64 * KIB)));
        db.close().unwrap();
    }
}

/// A crash after commits the log holds: the reopen replays them, and counts what replay kept as
/// the commits did.
#[cfg(all(
    feature = "grafeo-file",
    feature = "wal",
    feature = "testing-crash-injection"
))]
mod crash {
    use std::path::PathBuf;

    use grafeo_common::testing::child_process;
    use grafeo_common::types::EpochId;

    use super::reopen::{answers, created, persistent, reference};
    use super::{KIB, assert_not_retained, bio, bio_at, set_bio};

    const PATH_VAR: &str = "GRAFEO_HISTORY_LIMIT_CRASH_PATH";
    const EPOCH: &str = "epoch ";

    /// Child-process entry: twenty rewrites, a checkpoint, twenty more the log alone holds, then
    /// an exit without `close()`. A no-op when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let (db, start) = created(&PathBuf::from(path), Some(1 << 20));
        let mut epochs = vec![start];
        epochs.extend((0..20).map(|round| set_bio(&db, round, 64 * KIB)));
        db.wal_checkpoint().unwrap();
        epochs.extend((20..40).map(|round| set_bio(&db, round, 64 * KIB)));
        for epoch in &epochs {
            println!("{EPOCH}{}", epoch.as_u64());
        }
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    const WINDOW_PATH_VAR: &str = "GRAFEO_HISTORY_WINDOW_CHECKPOINT_PATH";

    /// Child-process entry: a window of 200 ms, three rewrites 2 ms apart (inside it, so the
    /// collects after them keep every epoch), a checkpoint 400 ms later (the window has moved past
    /// all but the last), then an exit without `close()`. A no-op when run directly.
    #[test]
    fn window_checkpoint_child() {
        let Some(path) = std::env::var_os(WINDOW_PATH_VAR) else {
            return;
        };
        let db = grafeo_engine::GrafeoDB::with_config(
            grafeo_engine::Config::persistent(PathBuf::from(path)).with_history_retention(
                grafeo_engine::HistoryRetention::Duration(std::time::Duration::from_millis(200)),
            ),
        )
        .unwrap();
        db.execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})")
            .unwrap();
        let mut epochs = vec![db.current_epoch()];
        for round in 0..3 {
            std::thread::sleep(std::time::Duration::from_millis(2));
            epochs.push(set_bio(&db, round, 64));
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
        db.wal_checkpoint().unwrap();
        for epoch in &epochs {
            println!("{EPOCH}{}", epoch.as_u64());
        }
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    /// A checkpoint writes history from the window's start now, not from the last collect's
    /// floor: a reopen that keeps history forever reads nothing before the start the checkpoint
    /// saw, though no collect moved the floor there (L1).
    #[test]
    fn a_checkpoint_writes_history_from_the_window_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("window.grafeo");
        let output = child_process::output(
            grafeo_common::child_test!("window_checkpoint_child").env(WINDOW_PATH_VAR, &path),
        )
        .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{stdout}");
        let epochs: Vec<EpochId> = stdout
            .lines()
            .filter_map(|line| line.strip_prefix(EPOCH))
            .map(|epoch| EpochId::new(epoch.trim().parse().unwrap()))
            .collect();
        assert_eq!(epochs.len(), 4, "{stdout}");
        let db = persistent(&path, None);
        // The collects after the commits left the floor at the first; the checkpoint, 400 ms
        // after the last commit, starts at the last one.
        for epoch in &epochs[..3] {
            assert_not_retained(bio_at(&db, *epoch), "before the checkpoint's start");
        }
        assert_eq!(bio_at(&db, epochs[3]), Ok(bio(2, 64)));
        db.close().unwrap();
    }

    #[test]
    fn a_crash_and_its_replay_read_as_the_same_writes_without_one() {
        let (epochs, expected) = reference();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crashed.grafeo");
        let output =
            child_process::output(grafeo_common::child_test!("crash_child").env(PATH_VAR, &path))
                .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{stdout}");
        let written: Vec<EpochId> = stdout
            .lines()
            .filter_map(|line| line.strip_prefix(EPOCH))
            .map(|epoch| EpochId::new(epoch.trim().parse().unwrap()))
            .collect();
        assert_eq!(written, epochs, "the same epochs");
        let db = persistent(&path, Some(1 << 20));
        db.gc();
        assert_eq!(answers(&db, &epochs), expected);
        db.close().unwrap();
    }
}
