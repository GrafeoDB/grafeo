//! Automatic checkpoints are on by default (0.6.2): a file database checkpoints by itself once its
//! WAL holds `DEFAULT_CHECKPOINT_WAL_SIZE` (256 MiB) past the last checkpoint, so a database
//! opened as 0.6.1 opened it, with no setting, now checkpoints on its own.
//! `Config::without_checkpoint_wal_size()` turns it off: the WAL then grows until close or a
//! manual checkpoint. `wal_segment_size` sets where the WAL rotates (64 MiB by default).
//!
//! Both sizes have bounds: `wal_segment_size` 1 MiB to 1 GiB, `checkpoint_wal_size` at least
//! 1 MiB (`Some(0)` would checkpoint after every commit). `validate`, and so every open, refuses
//! a size out of bounds with an error naming the setting and its bounds; the setter on an open
//! database refuses the same `checkpoint_wal_size`, with the same error.
//!
//! The checkpoint thread is event-driven: it parks until a commit takes the WAL past the limit
//! (the WAL's append path signals it), or until the interval passes when one is set. An idle
//! database's thread never wakes, the checkpoint runs on the thread and never on the committing
//! one, and close returns at once. After a failed automatic checkpoint the size trigger tries
//! again only once the WAL has grown one more segment past the failure; the interval trigger
//! keeps its time backoff. After a commit that did not complete, the thread reports the stop and
//! the database takes no more writes, so the WAL stops growing.
//!
//! The thread's counts (`checkpoint_thread_counts`) and a free-space answer that holds on every
//! thread (`with_free_space_under`) are test hooks behind `testing-crash-injection`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main checkpoint_default::
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file", not(miri)))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use grafeo_common::types::Value;
use grafeo_engine::admin::CheckpointTriggers;
use grafeo_engine::config::DEFAULT_CHECKPOINT_WAL_SIZE;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::wal::{DEFAULT_SEGMENT_BYTES, list_wal_directory};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// The bytes of WAL past the last checkpoint: what the size trigger counts.
fn wal_bytes(db: &GrafeoDB) -> u64 {
    u64::try_from(db.wal_status().size_bytes).unwrap()
}

/// Waits up to ten seconds (a loaded machine runs the checkpointer late).
fn wait_until(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "never: {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn open(path: &Path, config: impl FnOnce(Config) -> Config) -> GrafeoDB {
    GrafeoDB::with_config(config(Config::persistent(path))).unwrap()
}

/// Commits one node holding `bytes` bytes of text, so the WAL grows by a little more than that.
fn write(db: &GrafeoDB, bytes: u64) {
    let text = "x".repeat(usize::try_from(bytes).unwrap());
    db.execute_with_params(
        "INSERT (:Blob {b: $b})",
        HashMap::from([("b".to_string(), Value::from(text.as_str()))]),
    )
    .unwrap();
}

/// Commits 64 KiB nodes while the WAL stays at least `margin` below `limit`.
fn write_up_to(db: &GrafeoDB, limit: u64, margin: u64) {
    while wal_bytes(db) + margin < limit {
        write(db, 64 * 1024);
    }
}

fn blobs(db: &GrafeoDB) -> i64 {
    match db
        .execute("MATCH (b:Blob) RETURN count(b) AS c")
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(count) => count,
        ref other => panic!("{other:?}"),
    }
}

fn sidecar(path: &Path) -> PathBuf {
    let mut wal = path.as_os_str().to_owned();
    wal.push(".wal");
    PathBuf::from(wal)
}

/// The WAL's segments, oldest first: their first log position and their length, from the file.
fn segments(path: &Path) -> Vec<(u64, u64)> {
    list_wal_directory(&sidecar(path))
        .unwrap()
        .segments
        .into_iter()
        .map(|(lsn, segment)| (lsn, std::fs::metadata(segment).unwrap().len()))
        .collect()
}

// ---------------------------------------------------------------------------
// The default, the off, the bounds
// ---------------------------------------------------------------------------

/// With no setting, a file database checkpoints by WAL size, 256 MiB, and by no interval; the WAL
/// rotates at 64 MiB.
#[test]
fn the_default_is_256_mib_of_wal_and_no_interval() {
    assert_eq!(DEFAULT_CHECKPOINT_WAL_SIZE, 256 * MIB);
    assert_eq!(DEFAULT_SEGMENT_BYTES, 64 * MIB);
    let dir = tempfile::tempdir().unwrap();
    let config = Config::persistent(dir.path().join("amsterdam.grafeo"));
    assert_eq!(
        config.checkpoint_wal_size(),
        Some(DEFAULT_CHECKPOINT_WAL_SIZE)
    );
    assert_eq!(config.checkpoint_interval(), None);
    assert_eq!(config.wal_segment_size(), DEFAULT_SEGMENT_BYTES);
    let db = GrafeoDB::with_config(config).unwrap();
    assert_eq!(
        db.checkpoint_status().triggers,
        CheckpointTriggers::none().with_wal_bytes(DEFAULT_CHECKPOINT_WAL_SIZE)
    );
    db.close().unwrap();
}

/// A database opened with no setting, as 0.6.1 opened it, checkpoints by itself once its WAL holds
/// 256 MiB, and not before: the WAL grows to near the default, then falls. One node's value is
/// overwritten, so the store stays small and only the WAL grows. CI covers the default's value
/// (`the_default_is_256_mib_of_wal_and_no_interval`) and the trigger firing on the thread at
/// 1 MiB; this run at the real size goes on the release checklist.
#[test]
#[ignore = "writes about 256 MiB of WAL; run before a release with --ignored"]
fn a_database_opened_without_a_setting_checkpoints_at_256_mib() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("berlin.grafeo"), |config| config);
    db.execute("INSERT (:Blob {k: 1})").unwrap();
    let step = 4 * MIB;
    let value = Value::from("x".repeat(usize::try_from(step).unwrap()).as_str());
    let mut highest = wal_bytes(&db);
    loop {
        db.execute_with_params(
            "MATCH (b:Blob) SET b.v = $v",
            HashMap::from([("v".to_string(), value.clone())]),
        )
        .unwrap();
        let now = wal_bytes(&db);
        if now < highest {
            break;
        }
        highest = now;
        if highest >= DEFAULT_CHECKPOINT_WAL_SIZE + 2 * step {
            wait_until("the automatic checkpoint", || {
                wal_bytes(&db) < DEFAULT_CHECKPOINT_WAL_SIZE
            });
            break;
        }
    }
    assert!(
        highest >= DEFAULT_CHECKPOINT_WAL_SIZE - 2 * step,
        "nothing checkpointed before the WAL neared 256 MiB: the highest was {highest}"
    );
    let status = db.checkpoint_status();
    assert_eq!((status.failures, status.stopped), (0, false));
    db.close().unwrap();
}

/// `without_checkpoint_wal_size()` turns automatic checkpoints off: the WAL grows until a manual
/// checkpoint (or close).
#[test]
fn without_a_checkpoint_size_the_wal_grows_until_a_manual_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("lisbon.grafeo"), |config| {
        config.without_checkpoint_wal_size()
    });
    assert_eq!(db.checkpoint_status().triggers, CheckpointTriggers::none());
    write_up_to(&db, 2 * MIB, 0);
    let before = wal_bytes(&db);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(wal_bytes(&db), before, "nothing checkpoints by itself");
    db.wal_checkpoint().unwrap();
    assert_eq!(wal_bytes(&db), 0);
    db.close().unwrap();
}

/// Out-of-bounds sizes are refused by `validate` and by the open, each naming the setting and its
/// bounds; the bounds themselves are accepted.
#[test]
fn sizes_out_of_bounds_are_refused_naming_their_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("porto.grafeo");
    let refused: [(&str, Config, &[&str]); 4] = [
        (
            "a segment below 1 MiB",
            Config::persistent(&path).with_wal_segment_size(MIB - 1),
            &["wal_segment_size", "1 MiB", "1 GiB"],
        ),
        (
            "a segment above 1 GiB",
            Config::persistent(&path).with_wal_segment_size(GIB + 1),
            &["wal_segment_size", "1 MiB", "1 GiB"],
        ),
        (
            "a checkpoint size below 1 MiB",
            Config::persistent(&path).with_checkpoint_wal_size(MIB - 1),
            &["checkpoint_wal_size", "1 MiB"],
        ),
        (
            "a checkpoint size of zero",
            Config::persistent(&path).with_checkpoint_wal_size(0),
            &["checkpoint_wal_size", "1 MiB"],
        ),
    ];
    for (what, config, names) in refused {
        let message = config.validate().expect_err(what).to_string();
        let opened = match GrafeoDB::with_config(config) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("{what}: opened"),
        };
        for name in names {
            assert!(
                message.contains(name),
                "{what}: validate names {name}: {message}"
            );
            assert!(
                opened.contains(name),
                "{what}: the open names {name}: {opened}"
            );
        }
        assert!(!path.exists(), "{what}: the refused open made no file");
    }
    for (what, config) in [
        (
            "a 1 MiB segment",
            Config::persistent(&path).with_wal_segment_size(MIB),
        ),
        (
            "a 1 GiB segment",
            Config::persistent(&path).with_wal_segment_size(GIB),
        ),
        (
            "a 1 MiB checkpoint size",
            Config::persistent(&path).with_checkpoint_wal_size(MIB),
        ),
    ] {
        assert_eq!(config.validate(), Ok(()), "{what}");
    }
}

/// The setter on an open database applies the same floor with the same check: 1 MiB less one byte
/// is refused with the very error the open gives for it (variant and message), exactly 1 MiB is
/// accepted. The interval is refused only at zero.
#[test]
fn the_setter_refuses_the_same_checkpoint_size_with_the_same_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("antwerp.grafeo"), |config| config);
    let set = db
        .set_checkpoint_triggers(CheckpointTriggers::none().with_wal_bytes(MIB - 1))
        .expect_err("1 MiB less one byte");
    let Err(opened) = GrafeoDB::with_config(
        Config::persistent(dir.path().join("ghent.grafeo")).with_checkpoint_wal_size(MIB - 1),
    ) else {
        panic!("opened with 1 MiB less one byte")
    };
    assert_eq!(
        std::mem::discriminant(&set),
        std::mem::discriminant(&opened),
        "the same variant: {set:?} and {opened:?}"
    );
    assert_eq!(set.to_string(), opened.to_string(), "the same message");
    db.set_checkpoint_triggers(CheckpointTriggers::none().with_wal_bytes(MIB))
        .unwrap();
    assert_eq!(
        db.checkpoint_status().triggers,
        CheckpointTriggers::none().with_wal_bytes(MIB)
    );
    for interval in [Duration::from_millis(1), Duration::from_secs(1)] {
        db.set_checkpoint_triggers(CheckpointTriggers::none().with_interval(interval))
            .unwrap_or_else(|error| panic!("{interval:?}: {error}"));
    }
    assert!(
        db.set_checkpoint_triggers(CheckpointTriggers::none().with_interval(Duration::ZERO))
            .is_err()
    );
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// The segment size
// ---------------------------------------------------------------------------

const SIZE_PATH_VAR: &str = "GRAFEO_SEGMENT_SIZE_PATH";

/// The child: 48 commits of 64 KiB at a 1 MiB segment size, no automatic checkpoint, then it ends
/// as a crash does (no close, the database alive), so the WAL keeps every commit.
#[test]
fn segment_size_child() {
    let Some(path) = std::env::var_os(SIZE_PATH_VAR) else {
        return;
    };
    let path = PathBuf::from(path);
    let db = open(&path, |config| {
        config
            .without_checkpoint_wal_size()
            .with_wal_segment_size(MIB)
    });
    for _ in 0..48 {
        write(&db, 64 * 1024);
    }
    assert!(db.wal().is_some());
    std::process::exit(0);
}

/// The reader does not depend on the setting: a WAL written at 1 MiB segments reopens at 4 MiB
/// with every commit, and the next rotation happens at 4 MiB.
#[test]
fn a_wal_written_at_one_segment_size_reopens_at_another() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rotterdam.grafeo");
    let mut child = grafeo_common::child_test!("segment_size_child");
    child.env(SIZE_PATH_VAR, &path);
    let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
    assert!(status.success(), "the child failed");

    // Written at 1 MiB: about three sealed segments of at most 1 MiB and a group.
    let written = segments(&path);
    let sealed = &written[..written.len() - 1];
    assert!(sealed.len() >= 2, "{written:?}");
    for (lsn, length) in sealed {
        assert!(
            *length > MIB / 2 && *length <= MIB + 128 * 1024,
            "the segment at {lsn} holds {length} bytes: {written:?}"
        );
    }

    let db = open(&path, |config| {
        config
            .without_checkpoint_wal_size()
            .with_wal_segment_size(4 * MIB)
    });
    assert_eq!(blobs(&db), 48, "every commit of the 1 MiB segments");
    let current = segments(&path).last().unwrap().0;
    while segments(&path).last().unwrap().0 == current {
        write(&db, 64 * 1024);
    }
    let after = segments(&path);
    let (_, rotated) = after.iter().find(|(lsn, _)| *lsn == current).unwrap();
    assert!(
        *rotated > 2 * MIB && *rotated <= 4 * MIB + 128 * 1024,
        "the first rotation after the reopen at 4 MiB came at {rotated} bytes: {after:?}"
    );
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert!(blobs(&db) > 48);
    db.close().unwrap();
}

#[cfg(feature = "testing-crash-injection")]
mod hooks {
    //! The tests that read the checkpoint thread's counts or make its checkpoint fail.

    use super::*;

    use grafeo_common::utils::error::Error;
    use grafeo_engine::admin::CheckpointThreadCounts;
    use grafeo_storage::file::InsufficientSpace;
    use grafeo_storage::file::space::{with_free_space, with_free_space_under};

    /// Free bytes no checkpoint of a test database fits in.
    const TOO_LITTLE: u64 = 4_321;

    fn counts(db: &GrafeoDB) -> CheckpointThreadCounts {
        db.checkpoint_thread_counts()
    }

    /// An idle database starts no checkpoint thread: the thread starts at its first work, so a
    /// process that only opens and reads keeps a single thread (and glibc's single-thread
    /// allocator path).
    #[test]
    fn an_idle_database_starts_no_checkpoint_thread() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("delft.grafeo"), |config| {
            config.with_checkpoint_wal_size(MIB)
        });
        std::thread::sleep(Duration::from_millis(600));
        let idle = counts(&db);
        assert_eq!(
            (idle.starts, idle.wakes, idle.attempts),
            (0, 0, 0),
            "idle: {idle:?}"
        );
        db.close().unwrap();
    }

    /// Commits that leave the WAL below the mark start no thread either: only a commit that takes
    /// the WAL past the mark has work for one.
    #[test]
    fn writes_under_the_mark_start_no_thread() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("schiedam.grafeo"), |config| {
            config.with_checkpoint_wal_size(MIB)
        });
        for _ in 0..10 {
            write(&db, 1024);
        }
        std::thread::sleep(Duration::from_millis(300));
        let below = counts(&db);
        assert_eq!(
            (below.starts, below.wakes, below.attempts),
            (0, 0, 0),
            "commits below the limit: {below:?}"
        );
        db.close().unwrap();
    }

    /// A read-only open starts no thread, however large the WAL it reads: it never checkpoints.
    #[test]
    fn a_read_only_open_starts_no_thread() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vlaardingen.grafeo");
        {
            let db = open(&path, Config::without_checkpoint_wal_size);
            for _ in 0..4 {
                write(&db, 64 * 1024);
            }
            db.close().unwrap();
        }
        let db = GrafeoDB::open_read_only(&path).unwrap();
        assert_eq!(blobs(&db), 4);
        std::thread::sleep(Duration::from_millis(300));
        let read_only = counts(&db);
        assert_eq!(
            (read_only.starts, read_only.attempts),
            (0, 0),
            "{read_only:?}"
        );
        db.close().unwrap();
    }

    /// The commit that takes the WAL past the mark starts the thread, once, and it checkpoints.
    #[test]
    fn the_crossing_starts_the_thread() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("maassluis.grafeo"), |config| {
            config.with_checkpoint_wal_size(MIB)
        });
        write_up_to(&db, MIB + 64 * 1024, 0);
        wait_until("the thread's checkpoint", || {
            counts(&db).attempts == 1 && wal_bytes(&db) < MIB
        });
        assert_eq!(counts(&db).starts, 1, "{:?}", counts(&db));
        // The next crossing reuses the running thread.
        write_up_to(&db, MIB + 64 * 1024, 0);
        wait_until("the second checkpoint", || {
            counts(&db).attempts == 2 && wal_bytes(&db) < MIB
        });
        assert_eq!(counts(&db).starts, 1, "one thread for the database's life");
        db.close().unwrap();
    }

    /// The interval trigger starts the thread at the first committed change, not at open: no
    /// change, no thread and no checkpoint however many intervals pass.
    #[test]
    fn the_interval_trigger_starts_at_the_first_change() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("rijswijk.grafeo"), |config| {
            config
                .without_checkpoint_wal_size()
                .with_checkpoint_interval(Duration::from_millis(100))
        });
        std::thread::sleep(Duration::from_millis(600));
        let unchanged = counts(&db);
        assert_eq!(
            (unchanged.starts, unchanged.attempts),
            (0, 0),
            "{unchanged:?}"
        );
        write(&db, 1024);
        wait_until("the interval checkpoint", || counts(&db).attempts >= 1);
        assert_eq!(counts(&db).starts, 1, "{:?}", counts(&db));
        db.close().unwrap();
    }

    /// `set_checkpoint_triggers` on an idle database starts nothing while the WAL is below the new
    /// mark; a mark the WAL is already past starts the thread, which checkpoints at once. Both
    /// marks keep to the setter's 1 MiB floor.
    #[test]
    fn new_triggers_start_the_thread_only_past_their_mark() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(
            &dir.path().join("zoetermeer.grafeo"),
            Config::without_checkpoint_wal_size,
        );
        for _ in 0..24 {
            write(&db, 64 * 1024);
        }
        let logged = wal_bytes(&db);
        assert!(logged >= MIB + MIB / 4, "{logged}");
        db.set_checkpoint_triggers(CheckpointTriggers::none().with_wal_bytes(4 * MIB))
            .unwrap();
        std::thread::sleep(Duration::from_millis(300));
        let below = counts(&db);
        assert_eq!((below.starts, below.attempts), (0, 0), "{below:?}");
        db.set_checkpoint_triggers(CheckpointTriggers::none().with_wal_bytes(MIB))
            .unwrap();
        wait_until("the checkpoint past the new mark", || {
            counts(&db).attempts == 1 && wal_bytes(&db) < MIB
        });
        assert_eq!(counts(&db).starts, 1, "{:?}", counts(&db));
        db.close().unwrap();
    }

    /// A commit that takes the WAL past the limit checkpoints, on the thread: the thread counts the
    /// attempt and the WAL falls; nothing checkpoints before.
    #[test]
    fn a_commit_over_the_limit_checkpoints_on_the_thread() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("leiden.grafeo"), |config| {
            config.with_checkpoint_wal_size(MIB)
        });
        write_up_to(&db, MIB, 128 * 1024);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(counts(&db).attempts, 0, "below the limit");
        let below = wal_bytes(&db);
        assert!(below < MIB, "{below}");
        write(&db, 256 * 1024);
        wait_until("the checkpoint on the thread", || {
            counts(&db).attempts == 1 && wal_bytes(&db) < MIB
        });
        let after = counts(&db);
        assert!(after.wakes >= 1, "{after:?}");
        let status = db.checkpoint_status();
        assert_eq!((status.failures, status.stopped), (0, false));
        assert!(status.last_checkpoint.is_some());
        db.close().unwrap();
    }

    /// Commits that cross the limit while a checkpoint runs give exactly one follow-up checkpoint,
    /// not one per commit: the thread is held after its checkpoint, before it sets its next mark,
    /// so no commit can wake it; once it sets the mark it finds the WAL already past it and runs
    /// the follow-up without waiting for another commit.
    #[test]
    fn crossings_during_a_checkpoint_give_exactly_one_follow_up() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("gouda.grafeo"), |config| {
            config.with_checkpoint_wal_size(MIB)
        });
        let hold = db.hold_checkpoint_rearm();
        write_up_to(&db, MIB + 64 * 1024, 0);
        wait_until("the first checkpoint", || {
            counts(&db).attempts == 1 && wal_bytes(&db) < MIB
        });
        let held = counts(&db);
        for _ in 0..40 {
            write(&db, 64 * 1024);
        }
        assert!(wal_bytes(&db) > 2 * MIB, "{} bytes", wal_bytes(&db));
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            counts(&db),
            held,
            "no commit wakes the thread while it is held"
        );
        drop(hold);
        wait_until("the one follow-up", || {
            counts(&db).attempts == 2 && wal_bytes(&db) < MIB
        });
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(counts(&db).attempts, 2, "one follow-up, not one per commit");
        assert_eq!(db.checkpoint_status().failures, 0);
        db.close().unwrap();
    }

    const PAST_PATH_VAR: &str = "GRAFEO_CHECKPOINT_PAST_PATH";

    /// The child: 2 MiB of commits with automatic checkpoints off, then it ends as a crash does
    /// (no close, the database alive), so the WAL keeps them all.
    #[test]
    fn wal_past_the_limit_child() {
        let Some(path) = std::env::var_os(PAST_PATH_VAR) else {
            return;
        };
        let db = open(&PathBuf::from(path), Config::without_checkpoint_wal_size);
        for _ in 0..32 {
            write(&db, 64 * 1024);
        }
        assert!(wal_bytes(&db) >= 2 * MIB);
        std::process::exit(0);
    }

    /// A WAL already past the limit when the database opens is checkpointed by the thread without
    /// a commit (an open replays the WAL and keeps it).
    #[test]
    fn a_wal_past_the_limit_at_open_checkpoints_without_a_commit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("alkmaar.grafeo");
        let mut child = grafeo_common::child_test!("wal_past_the_limit_child");
        child.env(PAST_PATH_VAR, &path);
        let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
        assert!(status.success(), "the child failed");

        let db = open(&path, |config| config.with_checkpoint_wal_size(MIB));
        wait_until("the checkpoint at open", || {
            counts(&db).attempts == 1 && wal_bytes(&db) < MIB
        });
        assert_eq!(
            counts(&db).starts,
            1,
            "the open started the thread: {:?}",
            counts(&db)
        );
        assert_eq!(blobs(&db), 32, "every commit of the WAL");
        db.close().unwrap();
    }

    /// Two callers changing the triggers at once leave one working thread: a thread replaced while
    /// another starts never takes the WAL's waker with it, so the size trigger still checkpoints
    /// after hundreds of concurrent restarts.
    #[test]
    fn concurrent_trigger_changes_keep_the_size_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("enschede.grafeo"), |config| {
            config.with_checkpoint_wal_size(MIB)
        });
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    for _ in 0..300 {
                        db.set_checkpoint_triggers(CheckpointTriggers::none().with_wal_bytes(MIB))
                            .unwrap();
                    }
                });
            }
        });
        assert_eq!(
            db.checkpoint_status().triggers,
            CheckpointTriggers::none().with_wal_bytes(MIB)
        );
        // The counts run for the database's life, across restarts: compare against now.
        let before = counts(&db).attempts;
        write_up_to(&db, MIB + 64 * 1024, 0);
        wait_until("a checkpoint after the restarts", || {
            counts(&db).attempts > before && wal_bytes(&db) < MIB
        });
        db.close().unwrap();
    }

    /// Close returns at once with the thread parked, also when the park's timeout (the interval)
    /// is an hour.
    #[test]
    fn close_returns_at_once_with_the_thread_parked() {
        for interval in [None, Some(Duration::from_secs(3600))] {
            let dir = tempfile::tempdir().unwrap();
            let db = open(
                &dir.path().join("utrecht.grafeo"),
                |config| match interval {
                    Some(interval) => config.with_checkpoint_interval(interval),
                    None => config,
                },
            );
            write(&db, 1024);
            std::thread::sleep(Duration::from_millis(200));
            let started = Instant::now();
            db.close().unwrap();
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "{interval:?}: close took {:?}",
                started.elapsed()
            );
        }
    }

    /// A failing automatic checkpoint (no room on the disk) is tried again once the WAL has grown
    /// one more segment past the failure: not per commit and not per time step. With room again,
    /// the next try succeeds and the failures reset.
    #[test]
    fn a_failing_checkpoint_is_retried_once_per_segment_of_growth() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("haarlem.grafeo"), |config| {
            config
                .with_checkpoint_wal_size(MIB)
                .with_wal_segment_size(MIB)
        });
        with_free_space_under(dir.path(), Some(TOO_LITTLE), || {
            write_up_to(&db, MIB + 64 * 1024, 0);
            wait_until("the first failure", || db.checkpoint_status().failures == 1);
            let failed = counts(&db);
            assert_eq!(failed.attempts, 1, "{failed:?}");
            let status = db.checkpoint_status();
            assert!(!status.stopped, "a full disk stops nothing");
            assert!(status.last_error.is_some());

            // No writes for longer than the old first backoff (1 s): no try.
            std::thread::sleep(Duration::from_millis(1500));
            assert_eq!(counts(&db), failed, "no try while nothing is written");

            // Half a segment of commits: no wake, no try.
            for _ in 0..8 {
                write(&db, 64 * 1024);
            }
            std::thread::sleep(Duration::from_millis(300));
            assert_eq!(counts(&db), failed, "half a segment past the failure");
            assert_eq!(db.checkpoint_status().failures, 1);

            // One more segment past the failure: one more try, which fails again.
            for _ in 0..10 {
                write(&db, 64 * 1024);
            }
            wait_until("the second try", || db.checkpoint_status().failures == 2);
            std::thread::sleep(Duration::from_millis(300));
            assert_eq!(counts(&db).attempts, 2, "one try for that segment");
        });

        // Room again: the next segment's try succeeds and the failures reset. Writes stop once
        // that try has started (an attempt counts before its checkpoint runs): writing on until
        // it ends could pass the next mark while it runs, and the follow-up checkpoint that
        // grants would count as a fourth try.
        let deadline = Instant::now() + Duration::from_secs(10);
        while counts(&db).attempts < 3 {
            assert!(Instant::now() < deadline, "no third try with room");
            write(&db, 64 * 1024);
        }
        wait_until("the third try succeeded", || {
            db.checkpoint_status().failures == 0
        });
        wait_until("the WAL checkpointed", || wal_bytes(&db) < MIB);
        assert_eq!(counts(&db).attempts, 3, "one try for the segment of growth");
        db.close().unwrap();
    }

    /// The interval trigger keeps its time backoff after a failure (1 s, doubling): far fewer
    /// tries than intervals, and with room again it succeeds without a write.
    #[test]
    fn a_failing_interval_checkpoint_backs_off_in_time() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("zwolle.grafeo"), |config| {
            config
                .without_checkpoint_wal_size()
                .with_checkpoint_interval(Duration::from_millis(200))
        });
        write(&db, 1024);
        with_free_space_under(dir.path(), Some(TOO_LITTLE), || {
            wait_until("the first failure", || db.checkpoint_status().failures >= 1);
            wait_until("a retry after the backoff", || counts(&db).attempts >= 2);
            std::thread::sleep(Duration::from_millis(1500));
            // Tries at about 0.2 s, 1.2 s and 3.2 s: a try per interval would be over a dozen.
            let tries = counts(&db).attempts;
            assert!(
                tries <= 4,
                "{tries} tries in about 3 s at a 200 ms interval"
            );
        });
        wait_until("a success with room", || {
            db.checkpoint_status().failures == 0
        });
        assert_eq!(wal_bytes(&db), 0);
        db.close().unwrap();
    }

    /// With both triggers set, either wake tries, and a success resets the failures of both: a
    /// later failure counts from one again.
    #[test]
    fn with_both_triggers_a_success_resets_the_failures() {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("breda.grafeo"), |config| {
            config
                .with_checkpoint_wal_size(MIB)
                .with_wal_segment_size(MIB)
                .with_checkpoint_interval(Duration::from_millis(300))
        });
        with_free_space_under(dir.path(), Some(TOO_LITTLE), || {
            write_up_to(&db, MIB + 64 * 1024, 0);
            wait_until("a failure", || db.checkpoint_status().failures >= 1);
        });
        wait_until("a success with room", || {
            db.checkpoint_status().failures == 0
        });
        with_free_space_under(dir.path(), Some(TOO_LITTLE), || {
            write(&db, 1024);
            wait_until("the next failure", || db.checkpoint_status().failures >= 1);
            assert_eq!(db.checkpoint_status().failures, 1, "counted from one again");
        });
        db.close().unwrap();
    }

    /// The headroom a checkpoint needs follows the segment size: with 4 MiB segments, the file's
    /// length plus 4 MiB.
    #[test]
    fn the_checkpoint_headroom_is_the_configured_segment_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("arnhem.grafeo");
        let db = open(&path, |config| {
            config
                .without_checkpoint_wal_size()
                .with_wal_segment_size(4 * MIB)
        });
        for _ in 0..4 {
            write(&db, 16 * 1024);
        }
        let length = std::fs::metadata(&path).unwrap().len();
        let error = with_free_space(Some(TOO_LITTLE), || db.wal_checkpoint()).unwrap_err();
        let Error::Io(io) = &error else {
            panic!("an I/O error: {error:?}")
        };
        let space = io
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<InsufficientSpace>())
            .unwrap_or_else(|| panic!("InsufficientSpace inside: {error:?}"));
        assert_eq!(space.needed, length + 4 * MIB);
        db.close().unwrap();
    }

    /// After a commit that does not complete, the automatic checkpoint its WAL growth signals
    /// reports the stop, the next write is refused with `IncompleteCommit`, and the WAL does not
    /// grow: the database cannot outgrow its checkpoints.
    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn after_an_incomplete_commit_the_thread_reports_the_stop_and_the_wal_stops_growing() {
        use grafeo_common::testing::commit_hook::after_next_commit_logged;
        use grafeo_common::utils::error::TransactionError;

        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir.path().join("maastricht.grafeo"), |config| {
            config.with_checkpoint_wal_size(MIB)
        });
        write_up_to(&db, MIB, 384 * 1024);
        assert_eq!(counts(&db).attempts, 0);

        // The commit that takes the WAL past the limit panics after its WAL records.
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_with_params(
                "INSERT (:Blob {b: $b})",
                HashMap::from([(
                    "b".to_string(),
                    Value::from("x".repeat(512 * 1024).as_str()),
                )]),
            )
            .unwrap();
        after_next_commit_logged(|| panic!("injected: the commit stops after its WAL records"));
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
        assert!(unwound.is_err(), "the commit panicked");

        wait_until("the stop reported", || db.checkpoint_status().stopped);
        let status = db.checkpoint_status();
        assert!(status.failures >= 1, "{status:?}");
        assert!(status.last_error.is_some(), "{status:?}");
        assert_eq!(counts(&db).attempts, 1);

        let end = db.wal().unwrap().end_lsn();
        let error = db
            .execute("INSERT (:Person {name: 'Mia'})")
            .expect_err("a write after the stop");
        assert!(
            matches!(
                error,
                Error::Transaction(TransactionError::IncompleteCommit)
            ),
            "{error:?}"
        );
        assert_eq!(db.wal().unwrap().end_lsn(), end, "the WAL did not grow");
    }
}
