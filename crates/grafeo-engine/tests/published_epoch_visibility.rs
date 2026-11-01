//! Explicit epoch queries must not read versions above the published watermark.
//!
//! The existing stamped-commit hook holds publication while another thread
//! reads, without sleeps. A panic at the same hook leaves an incomplete commit;
//! its synced WAL group is replayed whole on a normal reopen.

#![cfg(all(
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-statement-injection"
))]
#![allow(
    missing_docs,
    reason = "integration tests exercise public epoch-query callers"
)]

use std::path::Path;
use std::sync::{Arc, mpsc};

use grafeo_common::testing::commit_hook::after_next_commit_stamped;
use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_engine::config::DurabilityMode;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention, Session};

#[derive(Clone, Copy, Debug)]
enum Caller {
    ViewingSession,
    SessionAtEpoch,
    SessionAtEpochWithParams,
    DatabaseAtEpoch,
}

const CALLERS: [Caller; 4] = [
    Caller::ViewingSession,
    Caller::SessionAtEpoch,
    Caller::SessionAtEpochWithParams,
    Caller::DatabaseAtEpoch,
];

#[derive(Clone, Copy, Debug)]
enum Requested {
    FiniteFuture,
    Maximum,
}

impl Requested {
    fn epoch(self, published: EpochId) -> EpochId {
        match self {
            Self::FiniteFuture => EpochId::new(published.as_u64() + 10),
            Self::Maximum => EpochId::new(u64::MAX),
        }
    }
}

struct Reader {
    caller: Caller,
    session: Session,
    requested: EpochId,
}

impl Reader {
    fn new(db: &GrafeoDB, caller: Caller, requested: EpochId) -> Self {
        let session = db.session();
        match caller {
            Caller::ViewingSession => session.set_viewing_epoch(requested),
            // A nonempty prior override must survive the temporary parameterized
            // view, both when that query succeeds and when it returns an error.
            Caller::SessionAtEpochWithParams => session.set_viewing_epoch(db.current_epoch()),
            Caller::SessionAtEpoch | Caller::DatabaseAtEpoch => {}
        }
        Self {
            caller,
            session,
            requested,
        }
    }

    fn execute(&self, db: &GrafeoDB, query: &str, parameterized: &str) -> Result<QueryResult> {
        match self.caller {
            Caller::ViewingSession => self.session.execute(query),
            Caller::SessionAtEpoch => self.session.execute_at_epoch(query, self.requested),
            Caller::DatabaseAtEpoch => db.execute_at_epoch(query, self.requested),
            Caller::SessionAtEpochWithParams => {
                let previous = self.session.viewing_epoch();
                let result = self.session.execute_at_epoch_with_params(
                    parameterized,
                    self.requested,
                    Some(std::collections::HashMap::from([(
                        "excluded".to_string(),
                        Value::from("Excluded"),
                    )])),
                );
                assert_eq!(
                    self.session.viewing_epoch(),
                    previous,
                    "parameterized epoch query must restore the prior override"
                );
                result
            }
        }
    }
}

const PEOPLE: &str = "MATCH (n:Person) RETURN n.name, n.city ORDER BY n.name";
const PEOPLE_WITH_PARAMS: &str =
    "MATCH (n:Person) WHERE n.name <> $excluded RETURN n.name, n.city ORDER BY n.name";
const CHANGED: &str = "MATCH (n:Changed) RETURN n.name ORDER BY n.name";
const CHANGED_WITH_PARAMS: &str =
    "MATCH (n:Changed) WHERE n.name <> $excluded RETURN n.name ORDER BY n.name";
const ORIGINAL: &str = "MATCH (n:Original) RETURN n.name ORDER BY n.name";
const ORIGINAL_WITH_PARAMS: &str =
    "MATCH (n:Original) WHERE n.name <> $excluded RETURN n.name ORDER BY n.name";

#[derive(Debug)]
struct Seen {
    published: EpochId,
    people: Vec<Vec<Value>>,
    changed: Vec<Vec<Value>>,
    original: Vec<Vec<Value>>,
}

fn observe(reader: &Reader, db: &GrafeoDB) -> Seen {
    Seen {
        published: db.current_epoch(),
        people: reader
            .execute(db, PEOPLE, PEOPLE_WITH_PARAMS)
            .expect("query person values")
            .rows()
            .to_vec(),
        changed: reader
            .execute(db, CHANGED, CHANGED_WITH_PARAMS)
            .expect("query added labels")
            .rows()
            .to_vec(),
        original: reader
            .execute(db, ORIGINAL, ORIGINAL_WITH_PARAMS)
            .expect("query removed labels")
            .rows()
            .to_vec(),
    }
}

fn assert_baseline(seen: &Seen) {
    assert_eq!(
        seen.people,
        vec![vec![Value::from("Base"), Value::from("Before")]],
        "an unpublished node or property value became visible: {seen:?}"
    );
    assert!(
        seen.changed.is_empty(),
        "an unpublished label became visible: {seen:?}"
    );
    assert_eq!(
        seen.original,
        vec![vec![Value::from("Base")]],
        "an unpublished label removal became visible: {seen:?}"
    );
}

fn assert_completed(seen: &Seen) {
    assert_eq!(
        seen.people,
        vec![
            vec![Value::from("Base"), Value::from("After")],
            vec![Value::from("New"), Value::from("After")],
        ]
    );
    assert_eq!(
        seen.changed,
        vec![vec![Value::from("Base")], vec![Value::from("New")]]
    );
    assert!(seen.original.is_empty(), "{seen:?}");
}

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_wal_durability(DurabilityMode::Sync)
            .with_history_retention(HistoryRetention::Forever),
    )
    .expect("open persistent database")
}

fn baseline(db: &GrafeoDB) -> NodeId {
    db.create_node_with_props(
        &["Person", "Original"],
        [
            ("name", Value::from("Base")),
            ("city", Value::from("Before")),
        ],
    )
    .expect("commit the baseline person")
}

fn stage_change(db: &GrafeoDB, base: NodeId) -> Session {
    let mut writer = db.session();
    writer
        .begin_transaction()
        .expect("begin explicit transaction");
    writer
        .set_node_property(base, "city", Value::from("After"))
        .expect("stage property update");
    assert!(writer.add_node_label(base, "Changed").unwrap());
    assert!(writer.remove_node_label(base, "Original").unwrap());
    writer
        .create_node_with_props(
            &["Person", "Changed"],
            [("name", Value::from("New")), ("city", Value::from("After"))],
        )
        .expect("stage another person");
    writer
}

fn paused_commit_keeps_future_reads_at_the_published_epoch(caller: Caller, request: Requested) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.grafeo");
    let db = Arc::new(open(&path));
    let base = baseline(&db);
    let before = db.current_epoch();
    let requested = request.epoch(before);
    let reader = Reader::new(&db, caller, requested);
    let mut writer = stage_change(&db, base);
    let during_db = Arc::clone(&db);
    let (sender, during) = mpsc::channel();
    after_next_commit_stamped(move || {
        // Join before returning from the hook, so the observation cannot slip
        // past publication. Return the same session for the later read.
        let observed = std::thread::spawn(move || {
            let seen = observe(&reader, &during_db);
            (reader, seen)
        })
        .join()
        .expect("reader during stamped commit panicked");
        sender.send(observed).expect("send the hook's observation");
    });
    writer.commit().expect("complete the paused commit");
    let (reader, seen) = during
        .try_recv()
        .expect("the commit must consume its stamped hook");
    assert_eq!(seen.published, before, "the hook ran before publication");
    assert_baseline(&seen);
    assert!(db.current_epoch() > before);
    // In particular, a setter-time clamp must not permanently pin a requested
    // future view: this exact session sees the newly published value now.
    assert_completed(&observe(&reader, &db));
    let past = Reader::new(&db, caller, before);
    assert_baseline(&observe(&past, &db));
    drop(writer);
    drop(reader);
    drop(past);
    db.close().expect("close completed source");
    drop(db);
    let reopened = open(&path);
    assert_completed(&observe(
        &Reader::new(&reopened, caller, requested),
        &reopened,
    ));
    reopened.close().expect("close reopened source");
}

/// Stages a change on `db` and commits it with a panic injected after the
/// commit is stamped and its WAL group synced, before it is published.
/// Returns the writer's session.
fn interrupt_commit(db: &GrafeoDB, base: NodeId) -> Session {
    let before = db.current_epoch();
    let mut writer = stage_change(db, base);
    let (sender, hook_ran) = mpsc::channel();
    after_next_commit_stamped(move || {
        sender.send(()).expect("record the stamped hook");
        panic!("injected: stop the commit after stamping");
    });
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| writer.commit()));
    assert!(
        panicked.is_err(),
        "the stamped hook must interrupt the commit"
    );
    hook_ran
        .try_recv()
        .expect("the incomplete commit must consume its stamped hook");
    assert_eq!(
        db.current_epoch(),
        before,
        "the failed commit was not published"
    );
    writer
}

fn incomplete_commit_keeps_future_reads_at_the_published_epoch(request: Requested) {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("source.grafeo"));
    let base = baseline(&db);
    let requested = request.epoch(db.current_epoch());
    let writer = interrupt_commit(&db, base);
    for caller in CALLERS {
        let reader = Reader::new(&db, caller, requested);
        assert_baseline(&observe(&reader, &db));
    }
    drop(writer);
}

/// A commit whose caller saw a panic after its WAL group was synced comes
/// back whole on reopen: the group is on disk, so recovery replays it, as it
/// replays any synced group (this is recovery, not a rollback of the group).
/// The open database refuses to close cleanly over it.
#[test]
fn a_synced_commit_whose_caller_panicked_is_replayed_whole_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.grafeo");
    let db = open(&path);
    let base = baseline(&db);
    let writer = interrupt_commit(&db, base);
    let error = db.close().expect_err("close reports the incomplete commit");
    assert!(
        matches!(
            error,
            Error::Transaction(TransactionError::IncompleteCommit)
        ),
        "{error}"
    );
    drop(writer);
    drop(db);
    for _ in 0..2 {
        let reopened = open(&path);
        for caller in CALLERS {
            assert_completed(&observe(
                &Reader::new(&reopened, caller, EpochId::new(u64::MAX)),
                &reopened,
            ));
        }
        reopened.close().expect("close recovered database");
    }
}

#[test]
fn viewing_session_finite_future_does_not_see_a_stamped_commit() {
    paused_commit_keeps_future_reads_at_the_published_epoch(
        Caller::ViewingSession,
        Requested::FiniteFuture,
    );
}

#[test]
fn viewing_session_maximum_does_not_see_a_stamped_commit() {
    paused_commit_keeps_future_reads_at_the_published_epoch(
        Caller::ViewingSession,
        Requested::Maximum,
    );
}

#[test]
fn session_at_finite_future_does_not_see_a_stamped_commit() {
    paused_commit_keeps_future_reads_at_the_published_epoch(
        Caller::SessionAtEpoch,
        Requested::FiniteFuture,
    );
}

#[test]
fn session_at_maximum_does_not_see_a_stamped_commit() {
    paused_commit_keeps_future_reads_at_the_published_epoch(
        Caller::SessionAtEpoch,
        Requested::Maximum,
    );
}

#[test]
fn parameterized_session_at_finite_future_restores_its_view_and_hides_a_stamped_commit() {
    paused_commit_keeps_future_reads_at_the_published_epoch(
        Caller::SessionAtEpochWithParams,
        Requested::FiniteFuture,
    );
}

#[test]
fn parameterized_session_at_maximum_restores_its_view_and_hides_a_stamped_commit() {
    paused_commit_keeps_future_reads_at_the_published_epoch(
        Caller::SessionAtEpochWithParams,
        Requested::Maximum,
    );
}

#[test]
fn database_at_finite_future_does_not_see_a_stamped_commit() {
    paused_commit_keeps_future_reads_at_the_published_epoch(
        Caller::DatabaseAtEpoch,
        Requested::FiniteFuture,
    );
}

#[test]
fn database_at_maximum_does_not_see_a_stamped_commit() {
    paused_commit_keeps_future_reads_at_the_published_epoch(
        Caller::DatabaseAtEpoch,
        Requested::Maximum,
    );
}

#[test]
fn finite_future_does_not_see_an_incomplete_commit() {
    incomplete_commit_keeps_future_reads_at_the_published_epoch(Requested::FiniteFuture);
}

#[test]
fn maximum_does_not_see_an_incomplete_commit() {
    incomplete_commit_keeps_future_reads_at_the_published_epoch(Requested::Maximum);
}

#[test]
fn explicit_epoch_callers_keep_normal_current_and_historical_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.grafeo");
    let db = open(&path);
    let base = baseline(&db);
    let before = db.current_epoch();
    stage_change(&db, base)
        .commit()
        .expect("commit the normal change");
    let current = db.current_epoch();
    for caller in CALLERS {
        assert_baseline(&observe(&Reader::new(&db, caller, before), &db));
        assert_completed(&observe(&Reader::new(&db, caller, current), &db));
    }
    db.close().expect("close history source");
}

#[test]
fn explicit_epoch_callers_still_refuse_history_below_the_retained_floor() {
    let db =
        GrafeoDB::with_config(Config::in_memory().with_history_retention(HistoryRetention::Off))
            .unwrap();
    let base = baseline(&db);
    let before = db.current_epoch();
    stage_change(&db, base)
        .commit()
        .expect("advance the retention floor");
    db.gc();
    let floor = db.current_epoch();
    assert!(floor > before);
    for caller in CALLERS {
        let reader = Reader::new(&db, caller, before);
        let error = reader
            .execute(&db, PEOPLE, PEOPLE_WITH_PARAMS)
            .expect_err("history before the retained floor must be refused");
        assert!(
            error.to_string().contains(&format!(
                "history before epoch {} is not retained",
                floor.as_u64()
            )),
            "{caller:?}: {error}"
        );
        assert_completed(&observe(&Reader::new(&db, caller, floor), &db));
    }
}
