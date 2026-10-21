//! History retention and time travel through the database API (H2c): the
//! stored setting, reads at an epoch and at a time, the floor before which
//! a read is refused, a reopen, a backup and its restore, and a crash in a
//! checkpoint.
//!
//! The row-group store keeps the values and labels of every retained epoch,
//! across a reopen too; these tests read which nodes exist at an epoch and a
//! time, and the epochs and times the database records. Crashes run in a
//! child process that exits without `close()`:
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test history_retention
//! ```

#![cfg(all(feature = "gql", feature = "lpg"))]

use grafeo_common::types::{EpochId, Timestamp, Value, ZonedDatetime};
use grafeo_engine::{Config, GrafeoDB, HistoryRetention, TimePoint};

/// Inserts a person named `name` in a statement of its own (one commit),
/// then waits a few milliseconds so the next commit takes a later time.
fn insert(db: &GrafeoDB, name: &str) -> (EpochId, Timestamp) {
    db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
        .unwrap_or_else(|e| panic!("insert {name}: {e}"));
    let epoch = db.current_epoch();
    std::thread::sleep(std::time::Duration::from_millis(3));
    let after = Timestamp::now();
    std::thread::sleep(std::time::Duration::from_millis(3));
    (epoch, after)
}

/// The names of the people a result lists.
fn names(result: &grafeo_engine::database::QueryResult) -> Vec<String> {
    result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("a name is a string, got {other:?}"),
        })
        .collect()
}

const PEOPLE: &str = "MATCH (p:Person) RETURN p.name ORDER BY p.name";

/// How many people a reader sees.
const COUNT: &str = "MATCH (p:Person) RETURN count(p)";

/// Checks that a reopened `db` reads `people` people at `at`, an earlier
/// commit's time: the rows of an epoch before the reopen as they were.
fn reads_after_reopen(db: &GrafeoDB, at: Timestamp, people: i64, what: &str) {
    let seen = count(&db.execute_as_of(COUNT, at).unwrap());
    assert_eq!(seen, people, "{what}");
}

fn count(result: &grafeo_engine::database::QueryResult) -> i64 {
    match result.rows()[0][0] {
        Value::Int64(count) => count,
        ref other => panic!("a count, got {other:?}"),
    }
}

fn keeping(retention: HistoryRetention) -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_history_retention(retention)).unwrap()
}

fn not_retained(error: grafeo_common::utils::error::Error) -> String {
    let text = error.to_string();
    assert!(text.contains("is not retained"), "{text}");
    text
}

/// A read at a time reads at the last epoch committed at or before it; a
/// view at an epoch or a time reads what was committed then and refuses
/// writes; a time before the history starts is refused.
#[test]
fn a_read_at_a_time_reads_the_last_commit_at_or_before_it() {
    let started = Timestamp::now();
    std::thread::sleep(std::time::Duration::from_millis(3));
    let db = keeping(HistoryRetention::Forever);
    let (alix, after_alix) = insert(&db, "Alix");
    let (gus, after_gus) = insert(&db, "Gus");
    let (_, after_vincent) = insert(&db, "Vincent");

    assert_eq!(
        names(&db.execute_as_of(PEOPLE, after_alix).unwrap()),
        ["Alix"]
    );
    assert_eq!(
        names(&db.execute_as_of(PEOPLE, after_gus).unwrap()),
        ["Alix", "Gus"]
    );
    assert_eq!(
        names(&db.execute_as_of(PEOPLE, after_vincent).unwrap()),
        ["Alix", "Gus", "Vincent"]
    );
    assert_eq!(db.epoch_as_of(after_gus).unwrap(), gus);
    assert_eq!(
        db.epoch_as_of(alix).unwrap(),
        alix,
        "an epoch reads at itself"
    );

    let view = db.as_of(after_alix).unwrap();
    assert_eq!(view.epoch(), alix);
    assert_eq!(names(&view.execute(PEOPLE).unwrap()), ["Alix"]);
    assert!(
        view.execute("INSERT (:Person {name: 'Mia'})").is_err(),
        "a view in the past is read-only"
    );
    assert_eq!(
        names(&db.as_of(gus).unwrap().execute(PEOPLE).unwrap()),
        ["Alix", "Gus"]
    );
    not_retained(db.execute_as_of(PEOPLE, started).unwrap_err());
}

/// A point in time is also a GQL `ZONED DATETIME` (its instant, whatever
/// its offset) or a duration ago (Snowflake's `OFFSET`, CockroachDB's
/// `'-10s'`): each reads at the last epoch committed at or before it.
#[test]
fn a_zoned_datetime_and_a_duration_ago_name_points_in_time() {
    let db = keeping(HistoryRetention::Forever);
    let (alix, after_alix) = insert(&db, "Alix");
    std::thread::sleep(std::time::Duration::from_millis(400));
    let (gus, _) = insert(&db, "Gus");

    // After Alix's commit, written in Amsterdam summer time (+02:00).
    let amsterdam = ZonedDatetime::from_timestamp_offset(after_alix, 7_200);
    assert_eq!(db.epoch_as_of(amsterdam).unwrap(), alix);
    assert_eq!(
        names(&db.execute_as_of(PEOPLE, amsterdam).unwrap()),
        ["Alix"]
    );
    assert_eq!(
        names(&db.as_of(amsterdam).unwrap().execute(PEOPLE).unwrap()),
        ["Alix"]
    );

    // 200 ms ago falls between the two commits; no time ago is now.
    let before_gus = TimePoint::ago(std::time::Duration::from_millis(200));
    assert_eq!(db.epoch_as_of(before_gus).unwrap(), alix);
    assert_eq!(
        names(&db.execute_as_of(PEOPLE, before_gus).unwrap()),
        ["Alix"]
    );
    assert_eq!(
        db.epoch_as_of(TimePoint::ago(std::time::Duration::ZERO))
            .unwrap(),
        gus
    );
    not_retained(
        db.as_of(TimePoint::ago(std::time::Duration::from_secs(3_600)))
            .unwrap_err(),
    );
}

/// Without history a read at a time is refused, and a read at an epoch
/// before the floor (the horizon garbage collection last ran at) is refused
/// with the floor named; with history kept forever the same read is
/// answered.
#[test]
fn a_read_before_the_floor_is_refused_never_answered() {
    let db = keeping(HistoryRetention::Off);
    let (alix, after_alix) = insert(&db, "Alix");
    insert(&db, "Gus");
    db.gc();
    let floor = db.current_epoch();
    let refused = not_retained(db.execute_at_epoch(PEOPLE, alix).unwrap_err());
    assert!(
        refused.contains(&format!(
            "history before epoch {} is not retained",
            floor.as_u64()
        )),
        "{refused}"
    );
    not_retained(db.as_of(alix).unwrap_err());
    assert!(
        db.execute_as_of(PEOPLE, after_alix).is_err(),
        "no time index"
    );
    assert_eq!(
        names(&db.execute_at_epoch(PEOPLE, floor).unwrap()),
        ["Alix", "Gus"],
        "a read at the floor"
    );
    // The direct reads are refused the same way, never answered with nothing.
    let id = match db
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN id(p)")
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(id) => grafeo_common::types::NodeId::new(u64::try_from(id).unwrap()),
        ref other => panic!("an id, got {other:?}"),
    };
    not_retained(db.get_node_at_epoch(id, alix).unwrap_err());
    not_retained(db.get_node_property_at_epoch(id, "name", alix).unwrap_err());
    not_retained(
        db.get_edge_at_epoch(grafeo_common::types::EdgeId::new(0), alix)
            .unwrap_err(),
    );
    assert!(
        db.get_node_at_epoch(id, floor).unwrap().is_some(),
        "a direct read at the floor"
    );
    assert_eq!(
        db.get_node_property_at_epoch(id, "name", floor).unwrap(),
        Some(Value::String("Alix".into()))
    );

    let db = keeping(HistoryRetention::Forever);
    let (alix, _) = insert(&db, "Alix");
    insert(&db, "Gus");
    db.gc();
    assert_eq!(names(&db.execute_at_epoch(PEOPLE, alix).unwrap()), ["Alix"]);
}

/// A view pins its epoch: garbage collection keeps it while it lives, so a
/// view outlives a window that moved past it, and a new view at that epoch
/// is refused once it is gone.
#[test]
fn a_view_pins_its_epoch_while_it_lives() {
    let db = keeping(HistoryRetention::Off);
    let (alix, _) = insert(&db, "Alix");
    let view = db.as_of(alix).unwrap();
    insert(&db, "Gus");
    db.gc();
    assert_eq!(names(&view.execute(PEOPLE).unwrap()), ["Alix"]);
    assert!(
        db.as_of(alix).is_ok(),
        "the view holds the floor at its epoch"
    );
    drop(view);
    insert(&db, "Vincent");
    db.gc();
    not_retained(db.as_of(alix).unwrap_err());
}

/// The retention is a stored setting: a database file keeps it, a reopen
/// applies it, and `set_stored_config` changes it from then on.
#[cfg(feature = "grafeo-file")]
#[test]
fn the_retention_is_stored_and_changes_from_then_on() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path).with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    let (_, after_alix) = insert(&db, "Alix");
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.config().history_retention(), HistoryRetention::Forever);
    assert_eq!(
        names(&db.execute_as_of(PEOPLE, after_alix).unwrap()),
        ["Alix"],
        "the time index came back"
    );
    assert!(db.store().keeps_history(), "the store keeps history too");
    db.set_stored_config(&Config::in_memory().with_history_retention(HistoryRetention::Off))
        .unwrap();
    assert!(
        db.execute_as_of(PEOPLE, after_alix).is_err(),
        "off from now on"
    );
    assert!(
        !db.store().keeps_history(),
        "and the store keeps none from now on (a replay keeps no history)"
    );
    db.set_stored_config(&Config::in_memory().with_history_retention(HistoryRetention::Forever))
        .unwrap();
    assert!(db.store().keeps_history(), "on again");
    db.set_stored_config(&Config::in_memory().with_history_retention(HistoryRetention::Off))
        .unwrap();
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.config().history_retention(), HistoryRetention::Off);
    assert!(db.execute_as_of(PEOPLE, after_alix).is_err());
    db.close().unwrap();
}

/// A reopen keeps the history: the time index of the file and of the WAL
/// groups after its checkpoint maps every time to the epoch it mapped to
/// before, and the reads at those epochs read what they read.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn a_reopen_keeps_the_history() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path).with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    let mut times = vec![insert(&db, "Alix"), insert(&db, "Gus")];
    db.wal_checkpoint().unwrap();
    times.push(insert(&db, "Vincent"));
    let before: Vec<i64> = times
        .iter()
        .map(|(_, at)| count(&db.execute_as_of(COUNT, *at).unwrap()))
        .collect();
    assert_eq!(before, [1, 2, 3]);
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    for ((epoch, at), people) in times.iter().zip(&before) {
        assert_eq!(
            db.epoch_as_of(*at).unwrap(),
            *epoch,
            "the time index came back"
        );
        reads_after_reopen(&db, *at, *people, "the reopen");
    }
    let (next, _) = insert(&db, "Mia");
    assert!(next > times[2].0, "the epochs continue");
    db.close().unwrap();
}

/// A backup and its restore keep the history: the restored database maps
/// the times of the backed up commits to their epochs.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn a_restore_keeps_the_history_of_its_backup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.grafeo");
    let backups = dir.path().join("backups");
    let restored_path = dir.path().join("restored.grafeo");
    let db = GrafeoDB::with_config(
        Config::persistent(&path).with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    let times = [insert(&db, "Alix"), insert(&db, "Gus")];
    db.backup_full(&backups).unwrap();
    let last = db.current_epoch();
    db.close().unwrap();

    GrafeoDB::restore_to_epoch(&backups, last, &restored_path).unwrap();
    let restored = GrafeoDB::open(&restored_path).unwrap();
    assert_eq!(
        restored.config().history_retention(),
        HistoryRetention::Forever
    );
    for ((epoch, at), people) in times.iter().zip([1, 2]) {
        assert_eq!(
            restored.epoch_as_of(*at).unwrap(),
            *epoch,
            "the time index came back"
        );
        reads_after_reopen(&restored, *at, people, "the restore");
    }
    restored.close().unwrap();
}

#[cfg(all(
    feature = "grafeo-file",
    feature = "wal",
    feature = "testing-crash-injection"
))]
mod crash {
    use std::path::PathBuf;

    use grafeo_common::testing::child_process;
    use grafeo_common::types::Timestamp;
    use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

    use super::{insert, reads_after_reopen};

    const POINT_VAR: &str = "GRAFEO_HISTORY_RETENTION_CRASH_POINT";
    const PATH_VAR: &str = "GRAFEO_HISTORY_RETENTION_CRASH_PATH";
    /// What a child prints per commit: its epoch and a time after it.
    const COMMIT: &str = "commit ";
    const COMPLETED: &str = "the checkpoint completed";

    /// Child-process entry: commits, a checkpoint, more commits, then a
    /// crash at injection point N of the next checkpoint; exits without
    /// `close()`. A no-op when run directly.
    #[test]
    fn crash_child() {
        let Ok(point) = std::env::var(POINT_VAR) else {
            return;
        };
        let point: u64 = point.parse().unwrap();
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        let db = GrafeoDB::with_config(
            Config::persistent(&path).with_history_retention(HistoryRetention::Forever),
        )
        .unwrap();
        let mut commits = vec![insert(&db, "Alix"), insert(&db, "Gus")];
        db.wal_checkpoint().unwrap();
        commits.push(insert(&db, "Vincent"));
        for (epoch, at) in &commits {
            println!("{COMMIT}{} {}", epoch.as_u64(), at.as_millis());
        }
        let target = std::panic::AssertUnwindSafe(&db);
        let result =
            grafeo_common::testing::crash::with_crash_at(point, move || target.wal_checkpoint());
        if let grafeo_common::testing::crash::CrashResult::Completed(checkpoint) = result {
            checkpoint.unwrap();
            println!("{COMPLETED}");
        }
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    /// A crash at every point of a checkpoint after commits the WAL logs:
    /// the reopen maps every commit's time to its epoch and reads there
    /// what was committed, whichever image the crash left.
    #[test]
    fn a_crash_in_a_checkpoint_keeps_the_history() {
        let mut completed = false;
        for point in 1..=16_u64 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("history.grafeo");
            let output = child_process::output(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "crash::crash_child", "--nocapture"])
                    .env(POINT_VAR, point.to_string())
                    .env(PATH_VAR, &path),
            )
            .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(output.status.success(), "point {point}: {stdout}");
            let commits: Vec<(u64, i64)> = stdout
                .lines()
                .filter_map(|line| line.strip_prefix(COMMIT))
                .map(|line| {
                    let (epoch, at) = line.split_once(' ').unwrap();
                    (epoch.parse().unwrap(), at.trim().parse().unwrap())
                })
                .collect();
            assert_eq!(commits.len(), 3, "point {point}: {stdout}");

            let db = GrafeoDB::open(&path).unwrap();
            for (people, (epoch, at)) in (1..).zip(&commits) {
                let at = Timestamp::from_millis(*at);
                assert_eq!(
                    db.epoch_as_of(at).unwrap().as_u64(),
                    *epoch,
                    "point {point}"
                );
                reads_after_reopen(&db, at, people, &format!("point {point}"));
            }
            db.close().unwrap();
            if stdout.contains(COMPLETED) {
                completed = true;
                break;
            }
        }
        assert!(completed, "a checkpoint has fewer than 16 crash points");
    }
}
