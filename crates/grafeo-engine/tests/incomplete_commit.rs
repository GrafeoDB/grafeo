//! A commit that is not complete yet, or never completes.
//!
//! A commit stamps its versions with its epoch and publishes the epoch only
//! once its events and WAL records are written. Until then the database's
//! direct reads (`get_node`, `get_edge`, `node_count`, `edge_count`,
//! `iter_nodes`, `iter_edges`, `validate`, `current_epoch`, the history reads
//! and the change history) read at the published epoch, as queries do, and do
//! not see the commit.
//!
//! When the commit code panics in between, the commit never completes: its
//! stamped versions stay in the store at an epoch that is never published. No
//! transaction commits afterwards (a later epoch would publish them), and
//! nothing checkpoints, saves, restores, merges, makes a full backup of or
//! copies the store (an incremental backup copies only the WAL). `close()`
//! keeps the WAL and leaves the file at its last checkpoint. A reopen then
//! shows the database without the failed commit when the panic came before
//! its WAL records were written; after them, the WAL holds the whole commit,
//! and a reopen replays it whole.
//!
//! What these tests do not cover: without the `temporal` feature, property
//! values have no versions, so a query or a direct read can see a property
//! value written by a commit that is not complete, or never completes (#412);
//! the property assertions below need `temporal`.
//!
//! These tests panic inside a commit, or run reads from another thread inside
//! one, with the `testing-statement-injection` commit hook:
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test incomplete_commit
//! ```

#![cfg(all(
    feature = "testing-statement-injection",
    feature = "lpg",
    feature = "gql"
))]

use std::sync::{Arc, mpsc};

use grafeo_common::testing::commit_hook::{after_next_commit_logged, after_next_commit_stamped};
use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

/// The names of the people in `db`, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// Alix's city, as a query sees it.
fn city_of_alix(db: &GrafeoDB) -> Value {
    db.execute("MATCH (p:Person {name: 'Alix'}) RETURN p.city")
        .unwrap()
        .rows()[0][0]
        .clone()
}

/// In one transaction: moves Alix to Paris, creates Gus and an edge from
/// Alix to Gus; the commit panics once its versions are stamped. Returns Gus
/// and the edge.
fn fail_a_commit(db: &GrafeoDB, alix: NodeId) -> (NodeId, EdgeId) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    let knows = session.create_edge(alix, gus, "KNOWS").unwrap();
    after_next_commit_stamped(|| panic!("injected: the commit stops after stamping"));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
    assert!(unwound.is_err(), "the commit panicked");
    (gus, knows)
}

/// What the direct reads of `db` see of Alix, Gus and their edge.
#[derive(Debug, PartialEq)]
struct Seen {
    nodes: usize,
    edges: usize,
    epoch: EpochId,
    gus: bool,
    knows: bool,
    /// Alix's city, with `temporal` (property values have versions).
    city: Option<Value>,
    /// The nodes and edges `iter_nodes` and `iter_edges` return.
    iterated: (usize, usize),
    /// Whether `validate` warns that there are nodes but no edges.
    no_edges_warning: bool,
}

fn seen(db: &GrafeoDB, alix: NodeId, gus: NodeId, knows: EdgeId) -> Seen {
    let city = if cfg!(feature = "temporal") {
        db.get_node(alix)
            .and_then(|node| node.get_property("city").cloned())
    } else {
        None
    };
    Seen {
        nodes: db.node_count(),
        edges: db.edge_count(),
        epoch: db.current_epoch(),
        gus: db.get_node(gus).is_some(),
        knows: db.get_edge(knows).is_some(),
        city,
        iterated: (db.iter_nodes().count(), db.iter_edges().count()),
        no_edges_warning: db
            .validate()
            .warnings
            .iter()
            .any(|warning| warning.code == "NO_EDGES"),
    }
}

#[test]
fn after_a_commit_that_does_not_complete_no_commit_publishes_part_of_it() {
    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let before = db.current_epoch();

    let (gus, knows) = fail_a_commit(&db, alix);
    assert_eq!(
        people(&db),
        [Value::from("Alix")],
        "the failed commit's write is not visible"
    );
    if cfg!(feature = "temporal") {
        assert_eq!(city_of_alix(&db), Value::Null, "nor its property value");
    }
    assert_eq!(
        seen(&db, alix, gus, knows),
        Seen {
            nodes: 1,
            edges: 0,
            epoch: before,
            gus: false,
            knows: false,
            city: None,
            iterated: (1, 0),
            no_edges_warning: true,
        },
        "the direct reads see the database as it was before the failed commit"
    );

    let error = db
        .execute("INSERT (:Person {name: 'Vincent'})")
        .expect_err("no commit succeeds after one that did not complete");
    assert!(
        error.to_string().contains("did not complete") && error.to_string().contains("reopen"),
        "the error says a commit did not complete and the database must be reopened: {error}"
    );
    assert!(
        db.create_node(&["Person"]).is_err(),
        "a direct write commits too, and fails the same way"
    );
    let mut explicit = db.session();
    explicit.begin_transaction().unwrap();
    explicit.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    assert!(
        explicit.commit().is_err(),
        "an explicit transaction cannot commit either"
    );

    let snapshot = {
        let other = GrafeoDB::new_in_memory();
        other.create_node(&["Person"]).unwrap();
        other.export_snapshot().unwrap()
    };
    let error = db
        .restore_snapshot(&snapshot)
        .expect_err("a restore after a failed commit could never be checkpointed");
    assert!(error.to_string().contains("did not complete"), "{error}");

    assert_eq!(
        people(&db),
        [Value::from("Alix")],
        "reads see what was published before the failed commit, never part of it"
    );
    assert_eq!(db.current_epoch(), before);
}

/// The change events of a commit are recorded before it is complete; the
/// change history returns them only once it is, and never those of a commit
/// that does not complete.
#[cfg(feature = "cdc")]
#[test]
fn change_events_of_a_commit_are_seen_only_once_it_is_complete() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    db.set_cdc_enabled(true);
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    // Events of Alix, of Gus, and of every entity.
    let events = move |db: &GrafeoDB, gus: NodeId| {
        (
            db.history(alix).unwrap().len(),
            db.history(gus).unwrap().len(),
            db.changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                .unwrap()
                .len(),
        )
    };

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    let (sender, during) = mpsc::channel();
    let reader = Arc::clone(&db);
    after_next_commit_logged(move || {
        let seen = std::thread::spawn(move || events(&reader, gus))
            .join()
            .expect("the reads during the commit panicked");
        sender.send(seen).unwrap();
    });
    session.commit().unwrap();
    assert_eq!(
        during.recv().expect("the commit ran the hook"),
        (1, 0, 1),
        "during the commit only Alix's creation is in the change history"
    );
    assert_eq!(
        events(&db, gus),
        (2, 1, 3),
        "once the commit is complete its events are"
    );

    // A commit that fails once its events and WAL records are written.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let vincent = session
        .create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
        .unwrap();
    after_next_commit_logged(|| panic!("injected: the commit stops after its WAL records"));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
    assert!(unwound.is_err(), "the commit panicked");
    assert_eq!(
        db.history(vincent).unwrap().len(),
        0,
        "the events of a commit that does not complete are never returned"
    );
}

/// A commit that panics once its WAL records are written leaves a complete
/// group in the WAL: it is not visible in the open database, and a reopen
/// replays all of it.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_commit_failing_after_its_wal_records_is_replayed_whole_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    after_next_commit_logged(|| panic!("injected: the commit stops after its WAL records"));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
    assert!(unwound.is_err(), "the commit panicked");
    assert_eq!(people(&db), [Value::from("Alix")], "not published");
    drop(session);
    assert!(db.close().is_err(), "close reports the failed commit");
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        people(&db),
        [Value::from("Alix"), Value::from("Gus")],
        "the reopen replays the commit's complete WAL group"
    );
    assert_eq!(city_of_alix(&db), Value::from("Paris"));
    db.close().unwrap();
}

/// Under memory pressure a compacted database merges its overlay into the
/// base, which has no versions: after a failed commit the merge would make
/// the commit's stamped part visible, so it does not run.
#[cfg(feature = "compact-store")]
#[test]
fn a_memory_pressure_merge_never_folds_a_failed_commit_into_the_base() {
    let mut db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    db.compact().unwrap();
    fail_a_commit(&db, alix);
    db.buffer_manager().spill_all();
    assert_eq!(
        people(&db),
        [Value::from("Alix")],
        "the merge did not fold the failed commit into the base"
    );
    assert_eq!(db.node_count(), 1);
}

/// Direct reads from another thread inside a commit, once its versions are
/// stamped, see the database without the commit; once the commit is
/// complete, they see it.
#[test]
fn direct_reads_see_a_commit_only_once_it_is_complete() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let before = db.current_epoch();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    let knows = session.create_edge(alix, gus, "KNOWS").unwrap();

    let (sender, during) = mpsc::channel();
    let reader = Arc::clone(&db);
    after_next_commit_stamped(move || {
        // Another thread reads while the commit is stamped and not complete;
        // it never waits for the commit, so it is joined here.
        let seen = std::thread::spawn(move || seen(&reader, alix, gus, knows))
            .join()
            .expect("the reads during the commit panicked");
        sender.send(seen).unwrap();
    });
    session.commit().unwrap();

    assert_eq!(
        during.recv().expect("the commit ran the hook"),
        Seen {
            nodes: 1,
            edges: 0,
            epoch: before,
            gus: false,
            knows: false,
            city: None,
            iterated: (1, 0),
            no_edges_warning: true,
        },
        "during the commit the direct reads do not see it"
    );
    let after = seen(&db, alix, gus, knows);
    assert!(after.epoch > before, "the commit published its epoch");
    assert_eq!(
        after,
        Seen {
            nodes: 2,
            edges: 1,
            epoch: after.epoch,
            gus: true,
            knows: true,
            city: cfg!(feature = "temporal").then(|| Value::from("Paris")),
            iterated: (2, 1),
            no_edges_warning: false,
        },
        "once the commit is complete the direct reads see it"
    );
}

/// After a commit that did not complete, nothing persists or copies the
/// store, which holds the commit's stamped part: every checkpoint, save,
/// backup and copy fails with the error of the failed commit and leaves the
/// file as it was. `close()` fails the same way, keeps the WAL and releases
/// the file; a reopen shows the database without the failed commit, with
/// what the WAL holds, and commits again.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_failed_commit_is_never_checkpointed_saved_or_copied() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let alix = {
        let db = GrafeoDB::open(&path).unwrap();
        let alix = db
            .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
            .unwrap();
        db.close().unwrap();
        alix
    };
    let checkpointed = std::fs::read(&path).unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    // Only in the WAL: a reopen must replay it.
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    fail_a_commit(&db, alix);
    let header = db.file_manager().unwrap().active_header();
    let assert_refused =
        |db: &GrafeoDB, operation: &str, outcome: grafeo_common::utils::error::Result<()>| {
            let error = outcome
                .err()
                .unwrap_or_else(|| panic!("{operation} succeeded after a failed commit"));
            assert!(
                error.to_string().contains("did not complete"),
                "{operation}: the error is the failed commit's: {error}"
            );
            assert_eq!(
                db.file_manager().unwrap().active_header(),
                header,
                "{operation}: the file has no new checkpoint"
            );
        };

    let copy = dir.path().join("copy.grafeo");
    let copy_directory = dir.path().join("copy");
    let backups = dir.path().join("backups");
    assert_refused(&db, "wal_checkpoint", db.wal_checkpoint());
    assert_refused(&db, "save to a .grafeo file", db.save(&copy));
    assert_refused(&db, "save to a WAL directory", db.save(&copy_directory));
    assert_refused(&db, "to_memory", db.to_memory().map(drop));
    assert_refused(&db, "export_snapshot", db.export_snapshot().map(drop));
    assert_refused(&db, "backup_full", db.backup_full(&backups).map(drop));
    #[cfg(feature = "compact-store")]
    let db = {
        let mut db = db;
        let outcome = db.compact();
        assert_refused(&db, "compact", outcome);
        db
    };
    for target in [&copy, &copy_directory] {
        assert!(
            !target.exists(),
            "nothing is written to {}",
            target.display()
        );
    }
    assert!(
        !backups.exists() || std::fs::read_dir(&backups).unwrap().next().is_none(),
        "no backup is written"
    );

    let error = db.close().expect_err("close reports the failed commit");
    assert!(error.to_string().contains("did not complete"), "{error}");
    drop(db);
    let wal = dir.path().join("amsterdam.grafeo.wal");
    assert!(
        wal.is_dir() && std::fs::read_dir(&wal).unwrap().next().is_some(),
        "close keeps the WAL, which holds Vincent"
    );
    assert!(
        std::fs::read(&path).unwrap() == checkpointed,
        "the file is byte for byte the last checkpoint"
    );

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        people(&db),
        [Value::from("Alix"), Value::from("Vincent")],
        "the reopened database has the WAL's commits and nothing of the failed one"
    );
    assert_eq!(city_of_alix(&db), Value::Null, "Alix never moved to Paris");
    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    assert_eq!(
        people(&db),
        [
            Value::from("Alix"),
            Value::from("Mia"),
            Value::from("Vincent")
        ],
        "the reopened database commits again"
    );
    db.close().unwrap();
}

/// The periodic checkpoint timer writes the store every interval; after a
/// commit that did not complete, it writes nothing more.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn the_checkpoint_timer_stops_after_a_failed_commit() {
    use std::time::{Duration, Instant};

    use grafeo_common::testing::commit_hook::checkpoints_started;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::persistent(&path)
            .with_checkpoint_interval(Duration::from_millis(200)),
    )
    .unwrap();
    let alix = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    let header = || db.file_manager().unwrap().active_header();

    // The timer is running: a checkpoint replaces the active header.
    let first = header();
    let deadline = Instant::now() + Duration::from_secs(10);
    while header() == first {
        assert!(Instant::now() < deadline, "the timer never checkpointed");
        std::thread::sleep(Duration::from_millis(50));
    }

    fail_a_commit(&db, alix);
    // A checkpoint holds commits off, so none is running now: the header
    // stays as it is from the failure on.
    let after_failure = header();
    let attempts = checkpoints_started(&path);
    std::thread::sleep(Duration::from_millis(1000));
    assert_eq!(
        header(),
        after_failure,
        "no checkpoint in five intervals after the failed commit"
    );
    let since = checkpoints_started(&path) - attempts;
    assert!(
        since <= 1,
        "the timer stops after its first failed attempt, it made {since}"
    );
    assert!(db.close().is_err(), "close reports the failed commit");
}
