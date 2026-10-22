//! An open refuses a WAL with damage before its end (#425): replaying the
//! commits after the damage on top of the missing ones builds a state that
//! never existed. The open fails naming the file and offset, keeps the WAL,
//! and `Config::with_wal_salvage()` replays the commits before the damage,
//! keeping the rest in `<file>.wal.damaged`. A torn tail is not damage.
//!
//! Both formats: the WAL of 0.5.x and 0.6.0-beta.1, which an open replays
//! once, and the segments of 0.6.0.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test wal_damage
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file", not(miri)))]

use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_common::types::{NodeId, TransactionId, Value};
use grafeo_common::utils::error::Error;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::wal::{WalManager, WalRecord};

fn sidecar(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal", path.display()))
}

fn set_aside(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal.damaged", path.display()))
}

/// The sorted names of the `:Person` nodes.
fn names(db: &GrafeoDB) -> Vec<String> {
    let mut names: Vec<String> = db
        .execute("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("a name: {other:?}"),
        })
        .collect();
    names.sort();
    names
}

/// The file and offset of the corruption `result` fails with.
fn corruption_of<T>(result: grafeo_common::utils::error::Result<T>) -> (PathBuf, u64) {
    match result {
        Err(Error::Corruption(corruption)) => {
            (corruption.file.unwrap(), corruption.offset.unwrap())
        }
        Err(other) => panic!("expected a corruption, got {other}"),
        Ok(_) => panic!("expected a corruption, the open succeeded"),
    }
}

/// The files of `dir`, sorted, with their bytes.
fn files_in(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_file())
        .map(|path| {
            let bytes = std::fs::read(&path).unwrap();
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                bytes,
            )
        })
        .collect();
    files.sort();
    files
}

// ── The WAL of 0.5.x and 0.6.0-beta.1 ──────────────────────────────

/// The people the first-format WAL of [`with_first_format_wal`] creates,
/// one per transaction.
const LOGGED: [(u64, &str); 3] = [(19, "Gus"), (88, "Vincent"), (103, "Mia")];

/// A database at `path` holding Alix, with a sidecar WAL of the format of
/// 0.5.x and 0.6.0-beta.1 that a writer left without `close()`: one
/// transaction per [`LOGGED`] person (create, name, commit). Returns the
/// log file and the offset of each transaction's first frame.
fn with_first_format_wal(path: &Path) -> (PathBuf, Vec<u64>) {
    let db = GrafeoDB::open(path).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.close().unwrap();
    drop(db);
    assert!(!sidecar(path).exists(), "close() removes the sidecar");

    let wal = WalManager::open(sidecar(path)).unwrap();
    let mut starts = Vec::new();
    let mut at = 0;
    for (transaction, (id, name)) in LOGGED.into_iter().enumerate() {
        starts.push(at);
        let records = [
            WalRecord::CreateNode {
                id: NodeId::new(id),
                labels: vec!["Person".to_string()],
            },
            WalRecord::SetNodeProperty {
                id: NodeId::new(id),
                key: "name".to_string(),
                value: Value::from(name),
            },
            WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(transaction as u64 + 1),
            },
        ];
        for record in &records {
            wal.log(record).unwrap();
        }
        wal.sync().unwrap();
        at = std::fs::metadata(only_log_file(&sidecar(path)))
            .unwrap()
            .len();
    }
    drop(wal);
    (only_log_file(&sidecar(path)), starts)
}

fn only_log_file(dir: &Path) -> PathBuf {
    let files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "log"))
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    files.into_iter().next().unwrap()
}

/// Flips a bit at `offset` of `file`.
fn damage(file: &Path, offset: u64) {
    let mut bytes = std::fs::read(file).unwrap();
    bytes[usize::try_from(offset).unwrap()] ^= 0x01;
    std::fs::write(file, bytes).unwrap();
}

/// The bug of #425: the open replayed Gus, skipped the damaged
/// transaction of Vincent and the rest of the file with a warning, and
/// removed the WAL, so Vincent and Mia were lost for good.
#[test]
fn a_damaged_first_format_wal_fails_the_open_and_stays() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let (log, starts) = with_first_format_wal(&path);
    damage(&log, starts[1] + 6);
    let before = files_in(&sidecar(&path));

    let (file, offset) = corruption_of(GrafeoDB::open(&path));
    assert_eq!((file, offset), (log.clone(), starts[1]));
    assert_eq!(files_in(&sidecar(&path)), before, "the WAL stays as it was");

    // A read-only open refuses it as well, and changes nothing.
    let (file, _) = corruption_of(GrafeoDB::with_config(Config::read_only(&path)));
    assert_eq!(file, log);
    assert_eq!(files_in(&sidecar(&path)), before);
}

/// A salvaging open replays the commits before the damage and keeps the
/// WAL it could not replay to the end; the next open is a plain one.
#[test]
fn a_salvaging_open_replays_up_to_the_damage_and_keeps_the_first_format_wal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let (log, starts) = with_first_format_wal(&path);
    damage(&log, starts[1] + 6);
    let damaged = files_in(&sidecar(&path));

    let db = GrafeoDB::with_config(Config::persistent(&path).with_wal_salvage()).unwrap();
    assert_eq!(names(&db), ["Alix", "Gus"]);
    assert_eq!(
        files_in(&set_aside(&path)),
        damaged,
        "the WAL is kept, unchanged"
    );
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(names(&db), ["Alix", "Gus", "Jules"]);
    db.close().unwrap();
    assert_eq!(files_in(&set_aside(&path)), damaged, "a close keeps it too");
}

/// A salvaging read-only open replays up to the damage and changes nothing.
#[test]
fn a_salvaging_read_only_open_replays_up_to_the_damage_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    let (log, starts) = with_first_format_wal(&path);
    damage(&log, starts[2] + 6);
    let before = files_in(&sidecar(&path));

    let db = GrafeoDB::with_config(Config::read_only(&path).with_wal_salvage()).unwrap();
    assert_eq!(names(&db), ["Alix", "Gus", "Vincent"]);
    drop(db);
    assert_eq!(files_in(&sidecar(&path)), before);
    assert!(!set_aside(&path).exists());
}

/// The end of a write a crash cut off is a torn tail, not damage: every
/// open replays the commits before it.
#[test]
fn a_torn_first_format_tail_opens_with_every_commit_before_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    let (log, starts) = with_first_format_wal(&path);
    let bytes = std::fs::read(&log).unwrap();
    // Mia's transaction cut in the middle of its first frame.
    let cut = usize::try_from(starts[2]).unwrap() + 7;
    std::fs::write(&log, &bytes[..cut]).unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(names(&db), ["Alix", "Gus", "Vincent"]);
    db.close().unwrap();
    assert!(!set_aside(&path).exists(), "nothing to keep");
}

// ── The WAL segments of 0.6.0 ──────────────────────────────────────

const PATH_VAR: &str = "GRAFEO_WAL_DAMAGE_PATH";

/// Child-process entry for [`segments_after_a_crash`]; a no-op when run
/// directly.
#[test]
fn crash_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(Path::new(&path)).unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    db.wal().unwrap().rotate().unwrap();
    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    // A crash: no close, no checkpoint, no destructors.
    std::process::exit(0);
}

/// A database at `path` holding Alix, whose writer crashed after Gus and
/// Vincent (in the first segment) and Mia (in the second). Returns the
/// segment files in log order.
fn segments_after_a_crash(path: &Path) -> Vec<PathBuf> {
    let db = GrafeoDB::open(path).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.close().unwrap();
    drop(db);
    let status = child_process::run(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env(PATH_VAR, path),
    )
    .unwrap();
    assert!(status.success(), "the child failed");
    let mut segments: Vec<PathBuf> = std::fs::read_dir(sidecar(path))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .and_then(grafeo_storage::wal::parse_segment_file_name)
                .is_some()
        })
        .collect();
    segments.sort();
    assert_eq!(segments.len(), 2, "{segments:?}");
    segments
}

/// Damage in a sealed segment, Vincent's group at its end, fails the open
/// with the segment and offset, and keeps the WAL.
#[test]
fn damage_in_a_sealed_segment_fails_the_open_and_stays() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let segments = segments_after_a_crash(&path);
    let end = std::fs::metadata(&segments[0]).unwrap().len();
    damage(&segments[0], end - 2);
    let before = files_in(&sidecar(&path));

    let (file, offset) = corruption_of(GrafeoDB::open(&path));
    assert_eq!(file, segments[0]);
    assert!(offset < end, "the offset {offset} lies in the segment");
    assert_eq!(files_in(&sidecar(&path)), before, "the WAL stays as it was");
}

/// A salvaging open replays the groups before the damage, and what it did
/// not replay outlives the close that removes the WAL.
#[test]
fn a_salvaging_open_replays_up_to_the_damage_and_keeps_the_rest_of_the_segments() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let segments = segments_after_a_crash(&path);
    let end = std::fs::metadata(&segments[0]).unwrap().len();
    damage(&segments[0], end - 2);
    let mia = std::fs::read(&segments[1]).unwrap();

    let db = GrafeoDB::with_config(Config::persistent(&path).with_wal_salvage()).unwrap();
    assert_eq!(names(&db), ["Alix", "Gus"]);
    db.execute("INSERT (:Person {name: 'Butch'})").unwrap();
    db.close().unwrap();
    drop(db);

    // The cut moved Vincent's damaged group and Mia's segment into
    // `damaged-<lsn>/`, which the close kept.
    let kept: Vec<PathBuf> = std::fs::read_dir(set_aside(&path))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(kept.len(), 1, "{kept:?}");
    assert!(
        files_in(&kept[0]).iter().any(|(_, bytes)| *bytes == mia),
        "Mia's segment is kept"
    );

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(names(&db), ["Alix", "Butch", "Gus"]);
}
