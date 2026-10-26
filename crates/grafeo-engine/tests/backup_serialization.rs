//! Backup calls serialize database state and exclude another destination writer.
//!
//! Callback events force the stale-state interleaving without race sleeps.
//! Process tests observe a named pause and check OS lock release, not power loss.

#![cfg(all(
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-statement-injection",
    feature = "testing-crash-injection"
))]
#![allow(
    missing_docs,
    reason = "integration tests exercise public backup callers"
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use grafeo_common::testing::{child_process, commit_hook, pause};
use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::backup::{BackupKind, BackupSegment};

const WAIT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug)]
enum Backup {
    Full,
    Incremental,
}

impl Backup {
    fn run(self, db: &GrafeoDB, directory: &Path) -> Result<BackupSegment> {
        match self {
            Self::Full => db.backup_full(directory),
            Self::Incremental => db.backup_incremental(directory),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Captured,
    Waiting,
}

struct Release(Option<mpsc::Sender<()>>);

impl Release {
    fn now(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        self.now();
    }
}

struct Finished(mpsc::Sender<()>);

impl Drop for Finished {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

struct BackupWork {
    handle: JoinHandle<Result<BackupSegment>>,
    finished: mpsc::Receiver<()>,
}

impl BackupWork {
    fn spawn(work: impl FnOnce() -> Result<BackupSegment> + Send + 'static) -> Self {
        let (sender, finished) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _finished = Finished(sender);
            work()
        });
        Self { handle, finished }
    }

    fn join(self) -> Result<BackupSegment> {
        self.finished
            .recv_timeout(WAIT)
            .expect("backup did not finish after release");
        self.handle.join().expect("backup worker panicked")
    }
}

/// Two people and one property-bearing edge, in one committed transaction.
fn seed(db: &GrafeoDB) -> NodeId {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let a = session
        .create_node_with_props(&["Person"], [("name", Value::from("A"))])
        .unwrap();
    let b = session
        .create_node_with_props(&["Person"], [("name", Value::from("B"))])
        .unwrap();
    session
        .create_edge_with_props(a, b, "LINK", [("rank", Value::Int64(1))])
        .unwrap();
    session.commit().unwrap();
    b
}

fn add_step(db: &GrafeoDB, from: NodeId, name: &str, rank: i64) -> NodeId {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let next = session
        .create_node_with_props(&["Person"], [("name", Value::from(name))])
        .unwrap();
    session
        .create_edge_with_props(from, next, "LINK", [("rank", Value::Int64(rank))])
        .unwrap();
    session.commit().unwrap();
    next
}

fn expected(last: &str) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let mut people = vec![vec![Value::from("A")], vec![Value::from("B")]];
    let mut edges = vec![vec![Value::from("A"), Value::from("B"), Value::Int64(1)]];
    if last == "C" || last == "D" {
        people.push(vec![Value::from("C")]);
        edges.push(vec![Value::from("B"), Value::from("C"), Value::Int64(2)]);
    }
    if last == "D" {
        people.push(vec![Value::from("D")]);
        edges.push(vec![Value::from("C"), Value::from("D"), Value::Int64(3)]);
    }
    (people, edges)
}

fn assert_restores(directory: &Path, at: EpochId, last: &str) {
    let restored = tempfile::tempdir().unwrap();
    let path = restored.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(directory, at, &path).expect("restore acknowledged backup");
    let (people, edges) = expected(last);
    for _ in 0..2 {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
                .unwrap()
                .rows(),
            &people,
        );
        assert_eq!(
            db.execute("MATCH (a:Person)-[e:LINK]->(b:Person) RETURN a.name, b.name, e.rank ORDER BY e.rank")
                .unwrap().rows(),
            &edges,
        );
        db.close().unwrap();
    }
}

fn checked_bytes(directory: &Path, segment: &BackupSegment) -> Vec<u8> {
    let bytes = std::fs::read(directory.join(&segment.filename)).unwrap();
    assert_eq!(u64::try_from(bytes.len()).unwrap(), segment.size_bytes);
    assert_eq!(
        crc32fast::hash(&bytes),
        segment.checksum,
        "{}",
        segment.filename
    );
    bytes
}

fn segment_lsn(segment: &BackupSegment) -> u64 {
    segment
        .end_wal_lsn
        .expect("new backup segments carry their WAL chain boundary")
}

fn assert_segment(actual: &BackupSegment, expected: &BackupSegment) {
    assert_eq!(actual.kind, expected.kind);
    assert_eq!(actual.filename, expected.filename);
    assert_eq!(actual.start_epoch, expected.start_epoch);
    assert_eq!(actual.end_epoch, expected.end_epoch);
    assert_eq!(actual.checksum, expected.checksum);
    assert_eq!(actual.size_bytes, expected.size_bytes);
    assert_eq!(actual.created_at_ms, expected.created_at_ms);
    assert_eq!(actual.end_wal_lsn, expected.end_wal_lsn);
}

fn two_calls(first_kind: Backup, second_kind: Backup, different_directories: bool) {
    let root = tempfile::tempdir().unwrap();
    let db = Arc::new(GrafeoDB::open(root.path().join("source.grafeo")).unwrap());
    let first_dir = root.path().join("first");
    let second_dir = if different_directories {
        root.path().join("second")
    } else {
        first_dir.clone()
    };
    let b = seed(&db);
    let baseline = db.backup_full(&first_dir).unwrap();
    let baseline_bytes = checked_bytes(&first_dir, &baseline);
    let baseline_lsn = segment_lsn(&baseline);
    let other_baseline = different_directories.then(|| db.backup_full(&second_dir).unwrap());
    let other_baseline_bytes = other_baseline
        .as_ref()
        .map(|segment| checked_bytes(&second_dir, segment));
    if let Some(other) = &other_baseline {
        assert_eq!(segment_lsn(other), baseline_lsn);
    }
    let c = add_step(&db, b, "C", 2);
    let first_epoch = db.current_epoch();
    let (paused_sender, paused) = mpsc::channel();
    let (first_sender, first_resume) = mpsc::channel();
    let mut release_first = Release(Some(first_sender));
    let first_db = Arc::clone(&db);
    let first_directory = first_dir.clone();
    let first = BackupWork::spawn(move || {
        let hook = move || {
            paused_sender.send(()).unwrap();
            first_resume
                .recv_timeout(WAIT)
                .expect("first backup was not released");
        };
        match first_kind {
            Backup::Full => commit_hook::during_next_checkpoint_write(hook),
            Backup::Incremental => commit_hook::after_next_backup_state_read(hook),
        }
        first_kind.run(&first_db, &first_directory)
    });
    paused
        .recv_timeout(WAIT)
        .expect("first backup did not consume its pause hook");
    add_step(&db, c, "D", 3);
    let last_epoch = db.current_epoch();
    assert!(last_epoch > first_epoch);

    let (event_sender, events) = mpsc::channel();
    let (second_sender, second_resume) = mpsc::channel();
    let mut release_second = Release(Some(second_sender));
    let second_db = Arc::clone(&db);
    let second_directory = second_dir.clone();
    let second = BackupWork::spawn(move || {
        let captured = event_sender.clone();
        commit_hook::after_next_backup_state_read(move || {
            captured.send(Event::Captured).unwrap();
            second_resume
                .recv_timeout(WAIT)
                .expect("second backup was not released");
        });
        commit_hook::before_next_backup_wait(move || {
            event_sender.send(Event::Waiting).unwrap();
        });
        second_kind.run(&second_db, &second_directory)
    });
    let event = events
        .recv_timeout(WAIT)
        .expect("second backup reported neither capture nor contention");
    release_first.now();
    let first = first.join();
    let first_bytes = first
        .as_ref()
        .ok()
        .map(|segment| checked_bytes(&first_dir, segment));
    // This send may precede the second state hook: it must be buffered so a
    // queued backup cannot hang after it acquires the gate.
    release_second.now();
    let second = second.join();
    drop(events); // Keep the receiver alive through the queued state hook.
    assert_eq!(
        event,
        Event::Waiting,
        "second backup captured state before the first finished"
    );
    let first = first.expect("first backup succeeds");
    let second = second.expect("queued backup succeeds");
    assert_eq!(first.end_epoch, first_epoch);
    assert_eq!(second.end_epoch, last_epoch);
    let first_lsn = segment_lsn(&first);
    let second_lsn = segment_lsn(&second);
    assert!(
        first_lsn > baseline_lsn,
        "the first backup must include C's committed WAL group"
    );
    assert!(
        second_lsn > first_lsn,
        "the queued backup must include D's later committed WAL group"
    );
    if matches!(first_kind, Backup::Incremental) {
        assert_eq!(first.start_epoch.as_u64(), baseline.end_epoch.as_u64() + 1);
    }
    if matches!(second_kind, Backup::Incremental) {
        assert_eq!(second.start_epoch.as_u64(), first.end_epoch.as_u64() + 1);
    }
    let first_path = first_dir.join(&first.filename);
    let second_path = second_dir.join(&second.filename);
    assert_ne!(
        first_path, second_path,
        "every acknowledgement owns its own file"
    );
    assert_eq!(
        checked_bytes(&first_dir, &first),
        first_bytes.unwrap(),
        "second backup overwrote first file"
    );
    checked_bytes(&second_dir, &second);
    assert_eq!(checked_bytes(&first_dir, &baseline), baseline_bytes);
    let manifest = GrafeoDB::read_backup_manifest(&first_dir).unwrap().unwrap();
    if different_directories {
        assert_eq!(manifest.segments.len(), 2);
        assert_segment(&manifest.segments[0], &baseline);
        assert_segment(&manifest.segments[1], &first);
        let other = GrafeoDB::read_backup_manifest(&second_dir)
            .unwrap()
            .unwrap();
        assert_eq!(other.segments.len(), 2);
        assert_segment(&other.segments[0], other_baseline.as_ref().unwrap());
        assert_segment(&other.segments[1], &second);
        assert_eq!(
            checked_bytes(&second_dir, other_baseline.as_ref().unwrap()),
            other_baseline_bytes.unwrap()
        );
    } else {
        assert_eq!(manifest.segments.len(), 3);
        assert_segment(&manifest.segments[0], &baseline);
        assert_segment(&manifest.segments[1], &first);
        assert_segment(&manifest.segments[2], &second);
    }
    let cursor = db.backup_cursor().unwrap();
    assert_eq!(cursor.backed_up_epoch, last_epoch);
    assert_eq!(cursor.wal_lsn, Some(second_lsn));
    db.close().unwrap();
    assert_restores(&first_dir, baseline.end_epoch, "B");
    assert_restores(&first_dir, first_epoch, "C");
    assert_restores(&second_dir, last_epoch, "D");
}

#[test]
fn two_full_backups_wait_for_the_same_database() {
    two_calls(Backup::Full, Backup::Full, false);
}
#[test]
fn two_incremental_backups_wait_for_the_same_database() {
    two_calls(Backup::Incremental, Backup::Incremental, false);
}
#[test]
fn an_incremental_waits_for_a_full_backup_on_the_same_database() {
    two_calls(Backup::Full, Backup::Incremental, false);
}
#[test]
fn a_full_backup_waits_for_an_incremental_on_the_same_database() {
    two_calls(Backup::Incremental, Backup::Full, false);
}
#[test]
fn backups_to_different_directories_still_wait_for_the_same_database() {
    two_calls(Backup::Full, Backup::Full, true);
}

const CHILD_SOURCE: &str = "GRAFEO_BACKUP_TEST_SOURCE";
const CHILD_DIRECTORY: &str = "GRAFEO_BACKUP_TEST_DIRECTORY";
const CHILD_MODE: &str = "GRAFEO_BACKUP_TEST_MODE";
const EXIT_WITH_LEASE: i32 = 73;

type Files = BTreeMap<PathBuf, Vec<u8>>;

fn files_under(directory: &Path) -> Files {
    let mut files = BTreeMap::new();
    if !directory.exists() {
        return files;
    }
    let mut pending = vec![directory.to_path_buf()];
    while let Some(path) = pending.pop() {
        if path.is_dir() {
            for entry in std::fs::read_dir(path).unwrap() {
                pending.push(entry.unwrap().path());
            }
        } else {
            files.insert(
                path.strip_prefix(directory).unwrap().to_path_buf(),
                std::fs::read(&path).unwrap(),
            );
        }
    }
    files
}

fn child_command(source: &Path, directory: &Path, mode: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--ignored",
            "--exact",
            "backup_serialization_child",
            "--nocapture",
        ])
        .env(CHILD_SOURCE, source)
        .env(CHILD_DIRECTORY, directory)
        .env(CHILD_MODE, mode)
        .env_remove(pause::POINT_VAR)
        .env_remove(pause::FLAG_VAR);
    command
}

fn child_output(source: &Path, directory: &Path, mode: &str) -> Output {
    child_process::output_with_deadline(&mut child_command(source, directory, mode), WAIT)
        .expect("backup child did not complete within its deadline")
}

struct PausedChild {
    output: Option<JoinHandle<Output>>,
    flag: PathBuf,
}

impl PausedChild {
    fn start(source: &Path, directory: &Path, mode: &str, flag: &Path) -> Self {
        let (source, directory, mode, child_flag) = (
            source.to_path_buf(),
            directory.to_path_buf(),
            mode.to_string(),
            flag.to_path_buf(),
        );
        let output = std::thread::spawn(move || {
            let mut command = child_command(&source, &directory, &mode);
            command
                .env(pause::POINT_VAR, "backup:before_manifest")
                .env(pause::FLAG_VAR, child_flag);
            child_process::output_with_deadline(&mut command, WAIT)
                .expect("paused backup child exceeded deadline")
        });
        Self {
            output: Some(output),
            flag: flag.to_path_buf(),
        }
    }

    fn wait_for_pause(&self) {
        let paused = with_suffix(&self.flag, ".paused");
        let deadline = Instant::now() + WAIT;
        // Poll a positive handshake, not a delay intended to win a race.
        while !paused.exists() {
            assert!(
                !self.output.as_ref().unwrap().is_finished(),
                "holder exited before its pause"
            );
            assert!(Instant::now() < deadline, "holder did not reach its pause");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            std::fs::read_to_string(paused).unwrap(),
            "backup:before_manifest"
        );
    }

    fn finish(mut self) -> Output {
        std::fs::write(with_suffix(&self.flag, ".resume"), b"resume").unwrap();
        self.output
            .take()
            .unwrap()
            .join()
            .expect("holder collector panicked")
    }
}

impl Drop for PausedChild {
    fn drop(&mut self) {
        if let Some(output) = self.output.take() {
            let _ = std::fs::write(with_suffix(&self.flag, ".resume"), b"resume");
            let _ = output.join();
        }
    }
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn directory_lease_survives_the_holder(exits: bool) {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source.grafeo");
    let directory = root.path().join("backups");
    let db = GrafeoDB::open(&source).unwrap();
    seed(&db);
    let epoch = db.current_epoch();
    db.close().unwrap();
    drop(db);
    let source_bytes = std::fs::read(&source).unwrap();
    let source_wal = files_under(&with_suffix(&source, ".wal"));
    let holder = PausedChild::start(
        &source,
        &directory,
        if exits { "exit" } else { "complete" },
        &root.path().join("holder"),
    );
    holder.wait_for_pause();
    let before = files_under(&directory);
    let refused = child_output(&source, &directory, "busy");
    assert!(
        refused.status.success(),
        "busy caller failed its contract: {}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert_eq!(
        files_under(&directory),
        before,
        "busy caller changed backup bytes"
    );
    assert_eq!(std::fs::read(&source).unwrap(), source_bytes);
    assert_eq!(files_under(&with_suffix(&source, ".wal")), source_wal);
    let output = holder.finish();
    if exits {
        assert_eq!(
            output.status.code(),
            Some(EXIT_WITH_LEASE),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            GrafeoDB::read_backup_manifest(&directory)
                .unwrap()
                .is_none()
        );
    } else {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let lock = directory.join("backup.lock");
    assert!(
        lock.is_file(),
        "the persistent lock name must not be unlinked"
    );
    assert_eq!(std::fs::read(&lock).unwrap(), [] as [u8; 0]);
    let completed = child_output(&source, &directory, "complete");
    assert!(
        completed.status.success(),
        "stale lock name prevented reuse: {}",
        String::from_utf8_lossy(&completed.stderr)
    );
    assert!(lock.is_file());
    assert_eq!(std::fs::read(&lock).unwrap(), [] as [u8; 0]);
    let manifest = GrafeoDB::read_backup_manifest(&directory).unwrap().unwrap();
    assert_eq!(manifest.segments.len(), if exits { 1 } else { 2 });
    let mut names = std::collections::BTreeSet::new();
    let source_lsn = segment_lsn(&manifest.segments[0]);
    for segment in &manifest.segments {
        assert_eq!(segment.kind, BackupKind::Full);
        assert_eq!(segment.end_epoch, epoch);
        assert_eq!(segment_lsn(segment), source_lsn);
        assert!(names.insert(segment.filename.clone()));
        checked_bytes(&directory, segment);
    }
    assert_restores(&directory, epoch, "B");
    assert_eq!(std::fs::read(&source).unwrap(), source_bytes);
    assert_eq!(files_under(&with_suffix(&source, ".wal")), source_wal);
}

#[test]
fn another_process_is_busy_until_the_directory_holder_finishes() {
    directory_lease_survives_the_holder(false);
}
#[test]
fn a_process_exit_releases_the_directory_lease_without_removing_its_file() {
    directory_lease_survives_the_holder(true);
}

#[test]
#[ignore = "subprocess entry point; invoked explicitly by the directory-lease tests"]
fn backup_serialization_child() {
    let source = PathBuf::from(std::env::var_os(CHILD_SOURCE).expect("source path for child"));
    let directory =
        PathBuf::from(std::env::var_os(CHILD_DIRECTORY).expect("backup directory for child"));
    let mode = std::env::var(CHILD_MODE).expect("mode for child");
    let db = GrafeoDB::open_read_only(source).unwrap();
    if mode == "exit" {
        commit_hook::after_next_backup_state_read(|| std::process::exit(EXIT_WITH_LEASE));
    }
    match (mode.as_str(), db.backup_full(&directory)) {
        ("busy", Err(error)) => {
            let message = error.to_string();
            assert!(
                message.contains(directory.to_string_lossy().as_ref()),
                "{message}"
            );
            assert!(
                message.contains("busy") || message.contains("lock") || message.contains("in use"),
                "{message}"
            );
        }
        ("busy", Ok(_)) => panic!("another process wrote the leased backup directory"),
        ("complete", Ok(_)) => {}
        (mode, result) => panic!("unexpected {mode} backup result: {result:?}"),
    }
    db.close().unwrap();
}
