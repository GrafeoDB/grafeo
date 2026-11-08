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

/// Commits 64 KiB nodes until the WAL is at least `margin` below `limit` no more: as many as the
/// WAL lacks, counted before the first, since each grows it by more than 64 KiB. The count is
/// fixed up front, so an automatic checkpoint between two of them cannot make the writes cross
/// the mark a second time.
fn write_up_to(db: &GrafeoDB, limit: u64, margin: u64) {
    let short = limit.saturating_sub(margin).saturating_sub(wal_bytes(db));
    for _ in 0..short.div_ceil(64 * 1024) {
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

// ---------------------------------------------------------------------------
// An Instant plus a duration a user sets (the class)
// ---------------------------------------------------------------------------

/// The crates whose source the scan below reads: every crate that can hold a deadline.
const SCANNED: [&str; 6] = [
    "grafeo-common/src",
    "grafeo-core/src",
    "grafeo-storage/src",
    "grafeo-adapters/src",
    "grafeo-engine/src",
    "grafeo/src",
];

/// Whether the right side of an `Instant` addition is a constant: `Duration::from_*` of a literal,
/// or an upper-case constant.
fn is_constant(right: &str) -> bool {
    let right = right.trim_start();
    if right.starts_with(|c: char| c.is_ascii_uppercase())
        && right
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    {
        return true;
    }
    let right = right.strip_prefix("std::time::").unwrap_or(right);
    [
        "Duration::from_secs(",
        "Duration::from_millis(",
        "Duration::from_micros(",
        "Duration::from_nanos(",
    ]
    .iter()
    .any(|prefix| {
        right.strip_prefix(prefix).is_some_and(|rest| {
            let digits = rest.find(')').map_or("", |end| &rest[..end]);
            !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit() || c == '_')
        })
    })
}

/// The `Instant` additions of one source text that add something other than a constant, as
/// `line: text` of the line the addition starts on: `Instant::now() + x`, or `now + x` for a
/// variable named `now`. The code lines are read as one, each trimmed and joined by a space, so
/// an addition rustfmt breaks before its `+` is found too.
fn unchecked_additions(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    // The code lines joined, with where each starts in the joined text.
    let mut code = String::new();
    let mut starts: Vec<(usize, usize)> = Vec::new();
    for (number, line) in lines.iter().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if !code.is_empty() {
            code.push(' ');
        }
        starts.push((code.len(), number));
        code.push_str(line);
    }
    let line_at = |at: usize| {
        let index = starts.partition_point(|&(start, _)| start <= at) - 1;
        starts[index].1
    };
    let mut found = Vec::new();
    for needle in ["Instant::now() +", "now +"] {
        let mut from = 0;
        while let Some(at) = code[from..].find(needle).map(|at| at + from) {
            from = at + needle.len();
            let before = code[..at].chars().next_back();
            let word = needle == "now +"
                && before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == ':');
            if word || is_constant(code[from..].trim_start()) {
                continue;
            }
            let number = line_at(at);
            found.push((number, format!("{}: {}", number + 1, lines[number].trim())));
        }
    }
    found.sort();
    found.into_iter().map(|(_, finding)| finding).collect()
}

/// The scan itself finds what it must: an interval, a timeout or a backoff added to an `Instant`
/// is found, a constant is not, and neither is a checked addition.
#[test]
fn the_instant_scan_tells_a_user_duration_from_a_constant() {
    let text = "let a = Instant::now() + interval;
let b = now + interval.max(pause);
let c = Instant::now() + Duration::from_secs(10);
let d = Instant::now() + LOCK_WAIT;
let e = now.checked_add(interval);
let f = std::time::Instant::now() + std::time::Duration::from_millis(1_500);
let g = snow + 1;
// let h = now + interval;
let i = state.now + 1;
let j = Instant::now()
    + timeout;
let k = now
    + Duration::from_secs(10);";
    let found = unchecked_additions(text);
    assert_eq!(
        found,
        [
            "1: let a = Instant::now() + interval;",
            "2: let b = now + interval.max(pause);",
            "10: let j = Instant::now()",
        ],
        "an addition broken before its `+` is found on the line it starts"
    );
}

/// Every `Instant` plus a duration in the source adds a constant. A duration a user sets (a
/// checkpoint interval, a query timeout, a backoff from either) goes through `checked_add`, where a
/// deadline past what an `Instant` holds means it never comes: an unchecked `+` panics for
/// `Duration::MAX`. The behavior rows are `an_interval_too_long_to_reckon_never_fires_and_the_thread_lives`
/// and `query_timeout_holds::a_timeout_too_long_to_reckon_runs_as_no_limit`. grafeo-common's
/// `testing` module is left out: test helpers whose callers pass constants.
#[test]
fn every_instant_addition_in_the_source_adds_a_constant() {
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                if !path.ends_with("testing") {
                    walk(&path, files);
                }
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    for dir in SCANNED {
        walk(&crates.join(dir), &mut files);
    }
    assert!(files.len() > 200, "{} files", files.len());
    let mut unchecked = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        for finding in unchecked_additions(&text) {
            unchecked.push(format!("{}:{finding}", file.display()));
        }
    }
    assert!(
        unchecked.is_empty(),
        "unchecked Instant additions: {unchecked:#?}"
    );
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

    /// Arming an interval on a database that already holds a committed change starts the thread
    /// at once, with no further commit, and it checkpoints. Without a WAL too: there the change
    /// lives in memory until a checkpoint writes the file, and no WAL size shows it.
    #[test]
    fn arming_an_interval_after_a_change_starts_the_thread_at_once() {
        for (name, wal) in [("leiden.grafeo", true), ("gouda.grafeo", false)] {
            let dir = tempfile::tempdir().unwrap();
            let db = open(&dir.path().join(name), |config| {
                let config = config.without_checkpoint_wal_size();
                if wal { config } else { config.without_wal() }
            });
            write(&db, 1024);
            std::thread::sleep(Duration::from_millis(300));
            let before = counts(&db);
            assert_eq!(
                (before.starts, before.attempts),
                (0, 0),
                "wal {wal}, no trigger yet: {before:?}"
            );
            db.set_checkpoint_triggers(
                CheckpointTriggers::none().with_interval(Duration::from_millis(100)),
            )
            .unwrap();
            wait_until("the checkpoint of the armed interval", || {
                counts(&db).attempts >= 1
            });
            assert_eq!(counts(&db).starts, 1, "wal {wal}: {:?}", counts(&db));
            db.close().unwrap();
        }
    }

    const REPLAY_PATH_VAR: &str = "GRAFEO_CHECKPOINT_REPLAY_PATH";

    /// The child: four small commits with automatic checkpoints off, then it ends as a crash does
    /// (no close, the database alive), so the WAL keeps them, far below any size mark.
    #[test]
    fn a_small_wal_child() {
        let Some(path) = std::env::var_os(REPLAY_PATH_VAR) else {
            return;
        };
        let db = open(&PathBuf::from(path), Config::without_checkpoint_wal_size);
        for _ in 0..4 {
            write(&db, 1024);
        }
        assert!(wal_bytes(&db) >= 4 * 1024, "{}", wal_bytes(&db));
        std::process::exit(0);
    }

    /// Changes an open replays from the WAL are changes the last checkpoint does not hold: an
    /// interval, set at open or armed afterwards, starts the thread without a new commit and
    /// checkpoints them.
    #[test]
    fn a_replayed_wal_gives_an_interval_work() {
        let interval = Duration::from_millis(100);
        for armed_later in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("haarlem.grafeo");
            let mut child = grafeo_common::child_test!("a_small_wal_child");
            child.env(REPLAY_PATH_VAR, &path);
            let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
            assert!(status.success(), "the child failed");

            let db = open(&path, |config| {
                let config = config.without_checkpoint_wal_size();
                if armed_later {
                    config
                } else {
                    config.with_checkpoint_interval(interval)
                }
            });
            assert_eq!(blobs(&db), 4, "every commit of the WAL");
            if armed_later {
                std::thread::sleep(Duration::from_millis(300));
                let before = counts(&db);
                assert_eq!((before.starts, before.attempts), (0, 0), "{before:?}");
                assert!(wal_bytes(&db) >= 4 * 1024, "{}", wal_bytes(&db));
                db.set_checkpoint_triggers(CheckpointTriggers::none().with_interval(interval))
                    .unwrap();
            }
            wait_until("the interval checkpoint of the replayed changes", || {
                counts(&db).attempts >= 1 && wal_bytes(&db) < 1024
            });
            assert_eq!(
                counts(&db).starts,
                1,
                "armed later {armed_later}: {:?}",
                counts(&db)
            );
            db.close().unwrap();
        }
    }

    /// An interval armed on a database with no change since its last checkpoint has no work: no
    /// thread starts and nothing checkpoints however many intervals pass, with or without a WAL,
    /// for a reopened database and for a new one that was never checkpointed. The first commit
    /// then starts it.
    #[test]
    fn an_unchanged_database_arming_an_interval_starts_nothing() {
        let rows = [
            ("utrecht.grafeo", true, false),
            ("breda.grafeo", false, false),
            ("tilburg.grafeo", true, true),
            ("venlo.grafeo", false, true),
        ];
        for (name, wal, new) in rows {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(name);
            let config = move |config: Config| {
                let config = config.without_checkpoint_wal_size();
                if wal { config } else { config.without_wal() }
            };
            if !new {
                let db = open(&path, config);
                write(&db, 1024);
                db.close().unwrap();
            }
            let db = open(&path, config);
            assert_eq!(blobs(&db), i64::from(!new), "wal {wal}, new {new}");
            db.set_checkpoint_triggers(
                CheckpointTriggers::none().with_interval(Duration::from_millis(100)),
            )
            .unwrap();
            std::thread::sleep(Duration::from_millis(600));
            let idle = counts(&db);
            assert_eq!(
                (idle.starts, idle.attempts),
                (0, 0),
                "wal {wal}, new {new}, unchanged: {idle:?}"
            );
            write(&db, 1024);
            wait_until("the interval checkpoint of the first change", || {
                counts(&db).attempts >= 1
            });
            assert_eq!(counts(&db).starts, 1, "wal {wal}: {:?}", counts(&db));
            db.close().unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // v4: the checked interval, the arm race, the restart window, changes
    // -----------------------------------------------------------------------

    /// A checkpoint interval past what an `Instant` holds never fires, and the thread that waits
    /// on it lives on: it still checkpoints by size, and the database closes. Three rows, one per
    /// place the interval's next deadline is reckoned: set at open, armed by the setter, and the
    /// backoff after a failed checkpoint (the larger of the interval and the backoff).
    #[test]
    fn an_interval_too_long_to_reckon_never_fires_and_the_thread_lives() {
        let both = CheckpointTriggers::none()
            .with_wal_bytes(MIB)
            .with_interval(Duration::MAX);
        for row in ["at open", "by the setter", "after a failure"] {
            let dir = tempfile::tempdir().unwrap();
            let db = open(&dir.path().join("hoorn.grafeo"), |config| {
                let config = config.with_wal_segment_size(MIB);
                if row == "at open" {
                    config
                        .with_checkpoint_wal_size(MIB)
                        .with_checkpoint_interval(Duration::MAX)
                } else {
                    config.without_checkpoint_wal_size()
                }
            });
            if row != "at open" {
                db.set_checkpoint_triggers(both).unwrap();
            }
            if row == "after a failure" {
                with_free_space_under(dir.path(), Some(TOO_LITTLE), || {
                    write_up_to(&db, MIB + 64 * 1024, 0);
                    wait_until("the failed size checkpoint", || {
                        db.checkpoint_status().failures == 1
                    });
                });
                // One more segment past the failure: the size trigger tries again, with room.
                let failed = wal_bytes(&db);
                write_up_to(&db, failed + MIB + 64 * 1024, 0);
                wait_until("the size checkpoint after the backoff", || {
                    db.checkpoint_status().failures == 0 && wal_bytes(&db) < MIB
                });
            } else {
                write_up_to(&db, MIB + 64 * 1024, 0);
                wait_until("the size checkpoint", || {
                    counts(&db).attempts >= 1 && wal_bytes(&db) < MIB
                });
            }
            assert_eq!(counts(&db).starts, 1, "{row}: {:?}", counts(&db));
            db.close().unwrap_or_else(|error| panic!("{row}: {error}"));
        }
    }

    /// A commit that crosses the WAL mark while the size trigger is being armed starts the
    /// thread: the crossing is the mark's only signal (the WAL clears the mark as it calls the
    /// waker), so the timer must already wait for work when the mark is armed. The hook runs on
    /// the arming thread after the mark is armed and before the arming checks whether the log is
    /// past it; another thread commits past the mark meanwhile. Its waker may wait for the arming
    /// to finish, so the hook waits only until the WAL has passed the mark, not for the commit.
    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn a_crossing_while_the_mark_is_armed_starts_the_thread() {
        use std::sync::{Arc, Mutex};

        use grafeo_common::testing::commit_hook;

        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(open(
            &dir.path().join("purmerend.grafeo"),
            Config::without_checkpoint_wal_size,
        ));
        for _ in 0..12 {
            write(&db, 64 * 1024);
        }
        let logged = wal_bytes(&db);
        assert!(logged < MIB, "{logged}");
        let blobs_to_cross = (MIB - logged) / (64 * 1024) + 2;
        let writer: Arc<Mutex<Option<std::thread::JoinHandle<()>>>> = Arc::default();
        let (writer_slot, writer_db, watched) =
            (Arc::clone(&writer), Arc::clone(&db), Arc::clone(&db));
        commit_hook::during_next_checkpoint_arm(move || {
            *writer_slot.lock().unwrap() = Some(std::thread::spawn(move || {
                for _ in 0..blobs_to_cross {
                    write(&writer_db, 64 * 1024);
                }
            }));
            wait_until("the WAL past the mark", || wal_bytes(&watched) >= MIB);
        });
        db.set_checkpoint_triggers(CheckpointTriggers::none().with_wal_bytes(MIB))
            .unwrap();
        writer
            .lock()
            .unwrap()
            .take()
            .expect("the arming ran the hook")
            .join()
            .unwrap();
        wait_until("the checkpoint of the crossing", || {
            counts(&db).attempts >= 1 && wal_bytes(&db) < MIB
        });
        assert_eq!(counts(&db).starts, 1, "{:?}", counts(&db));
        db.close().unwrap();
    }

    /// Without a WAL, a first commit with changes while the interval is armed again starts it,
    /// wherever it lands around the arming of the first-commit start: just before it, where the
    /// arming's check for a change since the last checkpoint sees the commit, or just after it,
    /// where the commit runs the hook. The hooks run on the restarting thread with no lock held
    /// that a commit takes; the commit runs on another thread and ends before the restart goes on.
    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn a_first_commit_around_a_restart_starts_the_interval() {
        use std::sync::Arc;

        use grafeo_common::testing::commit_hook;

        for row in ["just before the arming", "just after the arming"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("medemblik.grafeo");
            let config = |config: Config| config.without_checkpoint_wal_size().without_wal();
            {
                let db = open(&path, config);
                write(&db, 1024);
                db.close().unwrap();
            }
            let db = Arc::new(open(&path, config));
            let writer_db = Arc::clone(&db);
            let commit = move || {
                std::thread::spawn(move || write(&writer_db, 1024))
                    .join()
                    .unwrap();
            };
            if row == "just before the arming" {
                commit_hook::before_next_first_commit_arming(commit);
            } else {
                commit_hook::after_next_first_commit_arming(commit);
            }
            db.set_checkpoint_triggers(
                CheckpointTriggers::none().with_interval(Duration::from_millis(100)),
            )
            .unwrap();
            assert_eq!(blobs(&db), 2, "{row}: the hook's commit ran");
            wait_until("the interval checkpoint of the commit", || {
                counts(&db).attempts >= 1
            });
            assert_eq!(counts(&db).starts, 1, "{row}: {:?}", counts(&db));
            db.close().unwrap();
        }
    }

    /// Without a WAL, only a commit with changes is the interval's work: one the WAL would record,
    /// a change set or a schema change. A read in a transaction, a query, a DELETE that matched
    /// nothing and a direct removal of a property the node lacks commit without changes: no thread
    /// starts and nothing checkpoints, at the interval and after the triggers are armed again (the
    /// arming compares the last commit with changes, not the epoch, which such commits take). A
    /// write then starts it. A DDL statement alone, an index or a node type, is a change: it
    /// starts the thread, and the next tick checkpoints.
    #[test]
    fn only_commits_with_changes_give_an_interval_work() {
        let interval = CheckpointTriggers::none().with_interval(Duration::from_millis(100));
        let rows = [
            ("a read in a transaction", false),
            ("a query", false),
            ("a DELETE that matched nothing", false),
            ("a removal of a missing property", false),
            ("an index alone", true),
            ("a node type alone", true),
        ];
        for (row, is_change) in rows {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("edam.grafeo");
            let config = |config: Config| config.without_checkpoint_wal_size().without_wal();
            let city = {
                let db = open(&path, config);
                let city = db.create_node(&["City"]).unwrap();
                db.close().unwrap();
                city
            };
            let db = open(&path, |config_at_open| {
                config(config_at_open).with_checkpoint_interval(Duration::from_millis(100))
            });
            let rounds = if is_change { 1 } else { 3 };
            for _ in 0..rounds {
                match row {
                    "a read in a transaction" => {
                        let mut session = db.session();
                        session.begin_transaction().unwrap();
                        session.execute("MATCH (c:City) RETURN c").unwrap();
                        session.commit().unwrap();
                    }
                    "a query" => {
                        db.execute("MATCH (c:City) RETURN count(c) AS n").unwrap();
                    }
                    "a DELETE that matched nothing" => {
                        db.execute("MATCH (n:Nobody) DELETE n").unwrap();
                    }
                    "a removal of a missing property" => {
                        let removed = db.remove_node_property(city, "name").unwrap();
                        assert!(!removed, "the node has no name");
                    }
                    "an index alone" => {
                        db.execute("CREATE INDEX on_name FOR (c:City) ON (c.name)")
                            .unwrap();
                    }
                    _ => {
                        db.execute("CREATE NODE TYPE Harbor (name STRING)").unwrap();
                    }
                }
            }
            if is_change {
                wait_until("the checkpoint of the schema change", || {
                    counts(&db).attempts >= 1
                });
                assert_eq!(counts(&db).starts, 1, "{row}: {:?}", counts(&db));
                db.close().unwrap();
                continue;
            }
            std::thread::sleep(Duration::from_millis(600));
            let idle = counts(&db);
            assert_eq!((idle.starts, idle.attempts), (0, 0), "{row}: {idle:?}");
            db.set_checkpoint_triggers(interval).unwrap();
            std::thread::sleep(Duration::from_millis(600));
            let rearmed = counts(&db);
            assert_eq!(
                (rearmed.starts, rearmed.attempts),
                (0, 0),
                "{row}, armed again: {rearmed:?}"
            );
            write(&db, 1024);
            wait_until("the checkpoint of the first change", || {
                counts(&db).attempts >= 1
            });
            assert_eq!(counts(&db).starts, 1, "{row}: {:?}", counts(&db));
            db.close().unwrap();
        }
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

/// An open and close with no writes leaves the file's bytes as they were: the close
/// checkpoints only past the change count, and the open writes nothing. Five reads between them
/// change nothing either. Run with the defaults, and with `Sync` durability and history kept.
#[test]
fn an_open_and_close_with_no_writes_leaves_the_file_unchanged() {
    use grafeo_engine::config::{DurabilityMode, HistoryRetention};
    let configs: [(&str, fn(&std::path::Path) -> grafeo_engine::Config); 2] = [
        ("defaults", |path| grafeo_engine::Config::persistent(path)),
        ("sync, history kept", |path| {
            grafeo_engine::Config::persistent(path)
                .with_wal_durability(DurabilityMode::Sync)
                .with_history_retention(HistoryRetention::Forever)
        }),
    ];
    for (name, config) in configs {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("untouched.grafeo");
        {
            let db = grafeo_engine::GrafeoDB::with_config(config(&path)).unwrap();
            db.execute("INSERT (:City {name: 'Edam'})").unwrap();
            db.close().unwrap();
        }
        let bytes = std::fs::read(&path).unwrap();
        let db = grafeo_engine::GrafeoDB::with_config(config(&path)).unwrap();
        db.close().unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            bytes,
            "{name}: an open and close"
        );
        let db = grafeo_engine::GrafeoDB::with_config(config(&path)).unwrap();
        for _ in 0..5 {
            db.execute("MATCH (c:City) RETURN c").unwrap();
        }
        db.close().unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            bytes,
            "{name}: five reads between"
        );
    }
}
