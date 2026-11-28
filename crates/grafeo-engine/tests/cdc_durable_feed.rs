//! The durable change feed (#679, patch 7): with CDC stored on (`set_cdc(true)`), the events of
//! every commit are read back after a close and after a crash from the WAL, through
//! `read_cdc_feed` and `acknowledge_cdc_feed`, and equal the live events of the same run field for
//! field (ids, graphs, kinds, epochs, timestamps, before and after).
//!
//! The oracle: the live log of the same run (`changes_between` over every epoch), serialized, so
//! the comparison covers every field a consumer reads.
//!
//! - every LPG write form, in the default graph and a named graph, in GQL and Cypher, with the
//!   value kinds of the codec, in auto-commit statements and transactions (folded creates,
//!   rollbacks, a savepoint rollback, a failed statement);
//! - after a clean close, and after a crash (a child exits without closing);
//! - the feed's rules: a read does not move the cursor, an acknowledgement does and a retried one
//!   is a no-op, a forged or future position is refused (22023) with the cursor unchanged, `limit`
//!   never splits a group, the position's token round-trips;
//! - off and in memory: both calls refuse; off then on starts a fresh feed;
//! - a crash during an acknowledgement leaves the old position or the new one;
//! - past the retention limit the read fails with `GRAFEO-S006`;
//! - a delete too large for its whole image reads back with its identity and without its
//!   properties, as live;
//! - the live events carry the timestamps each commit reserved with its group, which the feed
//!   reads back;
//! - an entity that lives through a transaction reads back with one update per write, as live.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_durable_feed::
//! ```

#![cfg(all(
    feature = "cdc",
    feature = "gql",
    feature = "cypher",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_common::types::EpochId;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::cdc::{CdcBatch, CdcPosition, ChangeEvent};
use grafeo_engine::{Config, GrafeoDB};

const ALL: NonZeroUsize = NonZeroUsize::new(1_000_000).unwrap();

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::open(path).unwrap()
}

fn sidecar_wal(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal", path.display()))
}

/// Each event as JSON: every field, in a form that two processes print alike.
fn as_json(events: &[ChangeEvent]) -> Vec<serde_json::Value> {
    events
        .iter()
        .map(|event| serde_json::to_value(event).unwrap())
        .collect()
}

fn live(db: &GrafeoDB) -> Vec<serde_json::Value> {
    as_json(
        &db.changes_between(EpochId::new(0), db.current_epoch())
            .unwrap(),
    )
}

fn durable(batches: &[CdcBatch]) -> Vec<serde_json::Value> {
    let events: Vec<ChangeEvent> = batches
        .iter()
        .flat_map(|batch| batch.events.iter().cloned())
        .collect();
    as_json(&events)
}

/// The workload: every write form, the value kinds, the transaction outcomes and a named graph.
fn workload(db: &GrafeoDB) {
    let run = |statement: &str| {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    };
    // Creates, with and without labels and properties.
    run("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})");
    run("INSERT ()");
    run("INSERT (:Person {name: 'Gus'})");
    run("MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) \
         INSERT (a)-[:KNOWS {since: 2020}]->(g)");
    // Properties: a new one, an overwrite, the same value, a removal, a set to null.
    run("MATCH (a:Person {name: 'Alix'}) SET a.age = 30");
    run("MATCH (a:Person {name: 'Alix'}) SET a.age = 31");
    run("MATCH (a:Person {name: 'Alix'}) SET a.age = 31");
    run("MATCH (a:Person {name: 'Alix'}) REMOVE a.city");
    run("MATCH (a:Person {name: 'Alix'}) SET a.age = NULL");
    // Labels: a new one, one already there, a removal, an absent one.
    run("MATCH (a:Person {name: 'Alix'}) SET a:Admin");
    run("MATCH (a:Person {name: 'Alix'}) SET a:Person");
    run("MATCH (a:Person {name: 'Alix'}) REMOVE a:Admin");
    run("MATCH (a:Person {name: 'Alix'}) REMOVE a:Ghost");
    // The value kinds, as the old values of a later overwrite.
    run(
        "MATCH (g:Person {name: 'Gus'}) SET g.max = 9223372036854775807, \
         g.min = -9223372036854775808, g.zero = -0.0, g.text = 'Zoë 東京', \
         g.list = [1, [2, 'x'], null], g.map = {k: 1, nested: {v: true}}, \
         g.day = date('2026-11-12'), g.at = zoned_datetime('2026-11-12T10:00:00.123456+01:00'), \
         g.local = local_datetime('2026-11-12T10:00:00.5'), g.span = duration('P1DT2H3.25S')",
    );
    run(
        "MATCH (g:Person {name: 'Gus'}) SET g.max = 0, g.min = 0, g.zero = 0, g.text = '', \
         g.list = [], g.map = {}, g.day = NULL, g.at = NULL, g.local = NULL, g.span = NULL",
    );
    // Edges: a property set, a delete.
    run("MATCH ()-[e:KNOWS]->() SET e.since = 2021");
    run("MATCH ()-[e:KNOWS]->() DELETE e");
    // DETACH DELETE of a node with a self-loop and an edge to another node.
    run("INSERT (:Person {name: 'Mia'})");
    run("MATCH (m:Person {name: 'Mia'}) INSERT (m)-[:LIKES]->(m)");
    run("MATCH (m:Person {name: 'Mia'}), (g:Person {name: 'Gus'}) \
         INSERT (m)-[:KNOWS]->(g), (g)-[:KNOWS]->(m)");
    run("MATCH (m:Person {name: 'Mia'}) DETACH DELETE m");
    // A plain node delete.
    run("MATCH (n) WHERE n.name IS NULL DELETE n");
    // Cypher writes, a MERGE among them.
    db.execute_cypher("CREATE (:Person {name: 'Jules'})")
        .unwrap();
    db.execute_cypher("MERGE (p:Person {name: 'Jules'}) SET p.city = 'Utrecht'")
        .unwrap();
    db.execute_cypher("MERGE (p:Person {name: 'Vincent'})")
        .unwrap();

    // A transaction: a create folded with its later updates, and an update and a delete of
    // nodes from before.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Butch'})").unwrap();
    session
        .execute("MATCH (b:Person {name: 'Butch'}) SET b.age = 40, b:Boxer")
        .unwrap();
    session
        .execute("MATCH (j:Person {name: 'Jules'}) SET j.age = 33")
        .unwrap();
    session
        .execute("MATCH (v:Person {name: 'Vincent'}) DELETE v")
        .unwrap();
    // A failed statement leaves nothing of itself.
    assert!(
        session
            .execute("MATCH (j:Person {name: 'Jules'}) SET j.ratio = 1 / 0")
            .is_err()
    );
    // A savepoint rollback takes back what followed it.
    session.savepoint("before_django").unwrap();
    session
        .execute("INSERT (:Person {name: 'Django'})")
        .unwrap();
    session.rollback_to_savepoint("before_django").unwrap();
    session.commit().unwrap();

    // A delete after updates in one transaction: the delete with the state before it (the net
    // rule), durable as live.
    run("INSERT (:Person {name: 'Marcus', city: 'Lyon', age: 50})");
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (m:Person {name: 'Marcus'}) SET m.city = 'Lille', m.age = 51, m:Pilot")
        .unwrap();
    session
        .execute("MATCH (m:Person {name: 'Marcus'}) REMOVE m.age")
        .unwrap();
    session
        .execute("MATCH (m:Person {name: 'Marcus'}) DELETE m")
        .unwrap();
    session.commit().unwrap();

    // An entity that lives through a transaction keeps one update per write.
    run("INSERT (:Person {name: 'Harm', age: 28})");
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (h:Person {name: 'Harm'}) SET h.age = 29")
        .unwrap();
    session
        .execute("MATCH (h:Person {name: 'Harm'}) SET h.age = 28")
        .unwrap();
    session.commit().unwrap();

    // A rolled-back transaction leaves no events.
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Hans'})").unwrap();
    session.rollback().unwrap();

    // A named graph.
    db.create_graph("cities").unwrap();
    let in_cities = db.session();
    in_cities.use_graph("cities");
    in_cities
        .execute("INSERT (:City {name: 'Amsterdam'})-[:ROAD {km: 57}]->(:City {name: 'Utrecht'})")
        .unwrap();
    in_cities
        .execute("MATCH (c:City {name: 'Utrecht'}) SET c.population = 361924")
        .unwrap();
    in_cities
        .execute("MATCH (c:City {name: 'Amsterdam'}) DETACH DELETE c")
        .unwrap();
}

/// A file database with CDC stored on and the workload committed.
fn with_workload(path: &Path) -> GrafeoDB {
    let db = open(path);
    db.set_cdc(true).unwrap();
    workload(&db);
    db
}

#[test]
fn the_durable_events_equal_the_live_ones_after_a_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let db = with_workload(&path);
    let live = live(&db);
    assert!(
        live.len() > 30,
        "the workload reports its events: {}",
        live.len()
    );
    // Before the close the feed already holds them.
    assert_eq!(
        durable(&db.read_cdc_feed(ALL).unwrap()),
        live,
        "before the close"
    );
    db.close().unwrap();
    drop(db);
    let reopened = open(&path);
    assert!(reopened.is_cdc_enabled());
    assert_eq!(
        durable(&reopened.read_cdc_feed(ALL).unwrap()),
        live,
        "after the close"
    );
}

mod crash {
    use super::*;

    const PATH_VAR: &str = "GRAFEO_CDC_FEED_CRASH_PATH";
    const LIVE_VAR: &str = "GRAFEO_CDC_FEED_CRASH_LIVE";

    /// A child commits the workload, writes its live events to a file and exits without closing;
    /// the reopened database's feed equals them.
    #[test]
    fn the_durable_events_equal_the_live_ones_after_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let live_file = dir.path().join("live.json");
        let status = child_process::run(
            grafeo_common::child_test!("workload_child")
                .env(PATH_VAR, &path)
                .env(LIVE_VAR, &live_file),
        )
        .unwrap();
        assert!(status.success(), "the child failed");
        assert!(sidecar_wal(&path).exists(), "the crash left the WAL");
        let live: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(&live_file).unwrap()).unwrap();
        let reopened = open(&path);
        assert!(reopened.is_cdc_enabled());
        assert_eq!(durable(&reopened.read_cdc_feed(ALL).unwrap()), live);
    }

    /// Child-process entry for the test above; a no-op when run directly.
    #[test]
    fn workload_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = with_workload(Path::new(&path));
        let live_file = std::env::var_os(LIVE_VAR).unwrap();
        std::fs::write(live_file, serde_json::to_vec(&live(&db)).unwrap()).unwrap();
        // Exit without closing: no checkpoint, the WAL stays.
        std::process::exit(0);
    }
}

/// Three auto-commit statements: three groups.
fn three_groups(db: &GrafeoDB) {
    for name in ["Alix", "Gus", "Mia"] {
        db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
    }
}

fn file_with_cdc(dir: &tempfile::TempDir) -> (PathBuf, GrafeoDB) {
    let path = dir.path().join("people.grafeo");
    let db = open(&path);
    db.set_cdc(true).unwrap();
    (path, db)
}

fn positions(batches: &[CdcBatch]) -> Vec<CdcPosition> {
    batches.iter().map(|batch| batch.position).collect()
}

#[test]
fn a_read_does_not_move_the_cursor() {
    let dir = tempfile::tempdir().unwrap();
    let (_, db) = file_with_cdc(&dir);
    three_groups(&db);
    let first = db.read_cdc_feed(ALL).unwrap();
    assert_eq!(first.len(), 3, "one batch per group");
    assert_eq!(durable(&db.read_cdc_feed(ALL).unwrap()), durable(&first));
}

#[test]
fn an_acknowledgement_moves_past_its_position_and_a_retry_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let (path, db) = file_with_cdc(&dir);
    three_groups(&db);
    let batches = db.read_cdc_feed(ALL).unwrap();
    db.acknowledge_cdc_feed(batches[0].position).unwrap();
    assert_eq!(
        positions(&db.read_cdc_feed(ALL).unwrap()),
        positions(&batches[1..])
    );
    db.acknowledge_cdc_feed(batches[0].position)
        .expect("an acknowledgement behind the cursor is a no-op");
    assert_eq!(
        positions(&db.read_cdc_feed(ALL).unwrap()),
        positions(&batches[1..])
    );
    db.acknowledge_cdc_feed(batches[2].position).unwrap();
    assert!(db.read_cdc_feed(ALL).unwrap().is_empty(), "nothing new");
    // The cursor is stored: a reopen reads on from it.
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    db.close().unwrap();
    drop(db);
    let reopened = open(&path);
    let after = reopened.read_cdc_feed(ALL).unwrap();
    assert_eq!(after.len(), 1, "only the group after the acknowledged one");
}

#[test]
fn a_position_the_feed_did_not_produce_is_refused_and_the_cursor_stays() {
    let dir = tempfile::tempdir().unwrap();
    let (_, db) = file_with_cdc(&dir);
    three_groups(&db);
    let batches = db.read_cdc_feed(ALL).unwrap();
    let last = batches[2].position;
    assert_eq!(CdcPosition::from_token(&last.to_token()).unwrap(), last);
    let (lsn, epoch) = last
        .to_token()
        .split_once(':')
        .map(|(lsn, epoch)| (lsn.parse::<u64>().unwrap(), epoch.parse::<u64>().unwrap()))
        .expect("the token is <lsn>:<epoch>");
    for forged in [
        format!("{}:{epoch}", lsn + 1),
        format!("{lsn}:{}", epoch + 1),
        format!("{}:{}", lsn + 1_000_000, epoch + 10),
    ] {
        let position = CdcPosition::from_token(&forged).unwrap();
        let error = db.acknowledge_cdc_feed(position).unwrap_err();
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            "22023",
            "{forged}: {error}"
        );
        assert!(error.to_string().contains(&forged), "{error}");
        assert_eq!(
            positions(&db.read_cdc_feed(ALL).unwrap()),
            positions(&batches)
        );
    }
    for malformed in ["", "12", "a:b", "1:2:3", "-1:2"] {
        assert!(CdcPosition::from_token(malformed).is_err(), "{malformed:?}");
    }
}

#[test]
fn a_limit_never_splits_a_group() {
    let dir = tempfile::tempdir().unwrap();
    let (_, db) = file_with_cdc(&dir);
    // A group of ten events, then three groups of one.
    db.execute("UNWIND range(1, 10) AS i INSERT (:Person {i: i})")
        .unwrap();
    three_groups(&db);
    let one = NonZeroUsize::new(1).unwrap();
    let first = db.read_cdc_feed(one).unwrap();
    assert_eq!(first.len(), 1, "the first group whole, past the limit");
    assert_eq!(first[0].events.len(), 10);
    let two = db.read_cdc_feed(NonZeroUsize::new(11).unwrap()).unwrap();
    assert_eq!(
        two.iter()
            .map(|batch| batch.events.len())
            .collect::<Vec<_>>(),
        [10, 1],
        "whole groups up to the limit"
    );
}

#[test]
fn the_feed_needs_the_stored_setting_and_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("people.grafeo"));
    let error = db.read_cdc_feed(ALL).unwrap_err();
    assert!(error.to_string().contains("set_cdc"), "{error}");
    let position = CdcPosition::from_token("0:0").unwrap();
    let error = db.acknowledge_cdc_feed(position).unwrap_err();
    assert!(error.to_string().contains("set_cdc"), "{error}");
    // `with_cdc()` turns on live capture for this open only: no durable feed.
    drop(db);
    let live_only =
        GrafeoDB::with_config(Config::persistent(dir.path().join("live.grafeo")).with_cdc())
            .unwrap();
    assert!(live_only.read_cdc_feed(ALL).is_err());

    let in_memory = GrafeoDB::new_in_memory();
    in_memory.set_cdc(true).unwrap();
    assert!(
        in_memory.read_cdc_feed(ALL).is_err(),
        "no durable feed in memory"
    );
    assert!(in_memory.acknowledge_cdc_feed(position).is_err());
}

#[test]
fn off_then_on_starts_a_fresh_feed() {
    let dir = tempfile::tempdir().unwrap();
    let (_, db) = file_with_cdc(&dir);
    three_groups(&db);
    db.set_cdc(false).unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    db.set_cdc(true).unwrap();
    assert!(
        db.read_cdc_feed(ALL).unwrap().is_empty(),
        "nothing from before"
    );
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    let batches = db.read_cdc_feed(ALL).unwrap();
    assert_eq!(batches.len(), 1);
    let after = serde_json::to_string(&batches[0].events).unwrap();
    assert!(
        after.contains("Jules") && !after.contains("Vincent"),
        "{after}"
    );
}

mod acknowledgement_crash {
    use super::*;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};

    const SOURCE_VAR: &str = "GRAFEO_CDC_ACK_CRASH_PATH";
    const COUNT_VAR: &str = "GRAFEO_CDC_ACK_CRASH_COUNT";

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    /// A crash at each point of an acknowledgement of the first of three groups: the reopened
    /// feed starts at the first group (the old cursor) or the second (the new one), never
    /// elsewhere.
    #[test]
    fn a_crash_during_an_acknowledgement_keeps_the_old_or_the_new_position() {
        let base = tempfile::tempdir().unwrap();
        let (path, db) = file_with_cdc(&base);
        three_groups(&db);
        let batches = positions(&db.read_cdc_feed(ALL).unwrap());
        db.close().unwrap();
        drop(db);
        let mut completed = false;
        for count in 1..=64_u64 {
            let dir = tempfile::tempdir().unwrap();
            copy_dir(base.path(), dir.path());
            let copy = dir.path().join(path.file_name().unwrap());
            let output = child_process::output_with_deadline(
                grafeo_common::child_test!("acknowledge_child")
                    .env(SOURCE_VAR, &copy)
                    .env(COUNT_VAR, count.to_string()),
                std::time::Duration::from_secs(60),
            )
            .unwrap();
            assert!(output.status.success(), "count {count}: the child failed");
            let reopened = open(&copy);
            let first = positions(&reopened.read_cdc_feed(ALL).unwrap());
            assert!(
                first == batches || first == batches[1..],
                "count {count}: the feed starts at neither the old nor the new cursor: {first:?}"
            );
            if String::from_utf8_lossy(&output.stdout).contains("COMPLETED") {
                completed = true;
                break;
            }
        }
        assert!(
            completed,
            "the acknowledgement completes within 64 crash points"
        );
    }

    /// Child-process entry for the test above; a no-op when run directly.
    #[test]
    fn acknowledge_child() {
        let Some(path) = std::env::var_os(SOURCE_VAR) else {
            return;
        };
        let count: u64 = std::env::var(COUNT_VAR).unwrap().parse().unwrap();
        let db = open(Path::new(&path));
        let first = db.read_cdc_feed(ALL).unwrap()[0].position;
        // The database lives outside the closure: an injected crash does not close it.
        match with_crash_at(
            count,
            std::panic::AssertUnwindSafe(|| db.acknowledge_cdc_feed(first)),
        ) {
            CrashResult::Completed(result) => {
                result.unwrap();
                println!("COMPLETED");
            }
            _ => println!("CRASHED"),
        }
        // Exit without closing, as a crash does.
        std::process::exit(0);
    }
}

#[test]
fn past_the_retention_limit_the_read_fails_with_s006() {
    const LIMIT: u64 = 64 * 1024;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let db =
        GrafeoDB::with_config(Config::persistent(&path).with_wal_retention_limit(LIMIT)).unwrap();
    db.set_cdc(true).unwrap();
    for round in 0..30_i64 {
        db.execute(&format!(
            "UNWIND range({}, {}) AS i INSERT (:Person {{i: i, note: '{}'}})",
            round * 100,
            round * 100 + 99,
            "x".repeat(300)
        ))
        .unwrap();
        db.wal_checkpoint().unwrap();
    }
    let error = db.read_cdc_feed(ALL).unwrap_err();
    assert_eq!(error.error_code().as_str(), "GRAFEO-S006", "{error}");
}

/// The durable delete after updates holds the state before the transaction, as live does: the
/// net rule reads the same writes from the log.
#[test]
fn a_delete_after_updates_reads_back_with_the_state_before_the_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let (_, db) = file_with_cdc(&dir);
    workload(&db);
    let events: Vec<ChangeEvent> = db
        .read_cdc_feed(ALL)
        .unwrap()
        .into_iter()
        .flat_map(|batch| batch.events)
        .collect();
    let marcus = events
        .iter()
        .rev()
        .find(|event| {
            event.kind == grafeo_engine::cdc::ChangeKind::Delete
                && format!("{:?}", event.before).contains("Marcus")
        })
        .expect("Marcus's delete");
    let before = marcus.before.as_ref().unwrap();
    assert_eq!(
        before.get("city"),
        Some(&grafeo_common::types::Value::from("Lyon"))
    );
    assert_eq!(
        before.get("age"),
        Some(&grafeo_common::types::Value::Int64(50))
    );
    assert_eq!(marcus.labels.as_deref(), Some(&["Person".to_string()][..]));
}

/// A commit found by its position at the start of a WAL segment: a checkpoint between groups
/// starts a new segment, and acknowledging the first commit after it reads on from the second.
#[test]
fn a_position_at_a_segment_start_is_found() {
    let dir = tempfile::tempdir().unwrap();
    let (_, db) = file_with_cdc(&dir);
    three_groups(&db);
    db.wal_checkpoint().unwrap();
    three_groups(&db);
    let batches = db.read_cdc_feed(ALL).unwrap();
    assert_eq!(batches.len(), 6);
    db.acknowledge_cdc_feed(batches[3].position).unwrap();
    assert_eq!(
        positions(&db.read_cdc_feed(ALL).unwrap()),
        positions(&batches[4..])
    );
}

/// Reads while commits go on: every read succeeds with whole commits only, a group still being
/// written is not read, and each read is a prefix of the final one.
#[test]
fn a_read_beside_commits_reads_whole_commits_only() {
    let dir = tempfile::tempdir().unwrap();
    let (_, db) = file_with_cdc(&dir);
    let db = std::sync::Arc::new(db);
    let writer = {
        let db = std::sync::Arc::clone(&db);
        std::thread::spawn(move || {
            for i in 0..200 {
                db.execute(&format!(
                    "INSERT (:Person {{i: {i}}}), (:Person {{i: {i}}})"
                ))
                .unwrap();
            }
        })
    };
    let mut reads = Vec::new();
    while !writer.is_finished() {
        reads.push(db.read_cdc_feed(ALL).unwrap());
    }
    writer.join().unwrap();
    let last = db.read_cdc_feed(ALL).unwrap();
    assert_eq!(last.len(), 200);
    for read in &reads {
        assert!(
            read.iter().all(|batch| batch.events.len() == 2),
            "whole commits"
        );
        assert_eq!(positions(read), positions(&last[..read.len()]));
    }
}

/// A read-only open reads the feed and refuses an acknowledgement.
#[test]
fn a_read_only_open_reads_but_does_not_acknowledge() {
    let dir = tempfile::tempdir().unwrap();
    let (path, db) = file_with_cdc(&dir);
    three_groups(&db);
    db.close().unwrap();
    drop(db);
    let reader = GrafeoDB::open_read_only(&path).unwrap();
    let batches = reader.read_cdc_feed(ALL).unwrap();
    assert_eq!(batches.len(), 3);
    let error = reader
        .acknowledge_cdc_feed(batches[0].position)
        .unwrap_err();
    assert!(error.to_string().contains("read-only"), "{error}");
}

/// Past the retention limit, the position `GRAFEO-S006` names is accepted, and the feed goes on
/// from the first commit still in the log.
#[test]
fn the_position_s006_names_goes_on_from_the_first_commit_left() {
    use grafeo_common::utils::error::{Error, StorageError};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path).with_wal_retention_limit(64 * 1024))
        .unwrap();
    db.set_cdc(true).unwrap();
    for round in 0..30_i64 {
        db.execute(&format!(
            "UNWIND range({}, {}) AS i INSERT (:Person {{i: i, note: '{}'}})",
            round * 100,
            round * 100 + 99,
            "x".repeat(300)
        ))
        .unwrap();
        db.wal_checkpoint().unwrap();
    }
    let (first_available, epoch) = match db.read_cdc_feed(ALL) {
        Err(Error::Storage(storage)) => match *storage {
            StorageError::CdcFeedTruncated {
                first_available,
                epoch,
            } => (first_available, epoch),
            other => panic!("the read is not S006: {other}"),
        },
        Err(other) => panic!("the read is not S006: {other}"),
        Ok(batches) => panic!("the read gave {} batches, not S006", batches.len()),
    };
    let restart = CdcPosition::from_token(&format!("{first_available}:{epoch}")).unwrap();
    db.acknowledge_cdc_feed(restart).unwrap();
    let batches = db.read_cdc_feed(ALL).unwrap();
    assert!(!batches.is_empty(), "the feed goes on");
    assert_eq!(
        batches[0].position, restart,
        "from the first commit still there"
    );
}

/// A delete too large for its whole before-image (140,000 properties, past the list rule) reads
/// back from the log with its identity, its labels, and without its properties (`before` is
/// `None`), as the live log reports it; a small node deleted in the same commit keeps its whole
/// image. The same after a close.
#[test]
fn an_oversized_delete_reads_back_with_its_identity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let db = open(&path);
    db.set_cdc(true).unwrap();
    for name in ["Big", "Small"] {
        db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
    }
    for chunk in 0..140 {
        let sets: Vec<String> = (chunk * 1000..(chunk + 1) * 1000)
            .map(|i| format!("`{}`: true", crate::cdc_wal_records::tiny_key(i)))
            .collect();
        db.execute(&format!(
            "MATCH (n:Person {{name: 'Big'}}) SET n += {{{}}}",
            sets.join(", ")
        ))
        .unwrap();
    }
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("MATCH (n:Person) DELETE n").unwrap();
    session.commit().unwrap();
    let epoch = db.current_epoch();
    let live = db.changes_between(epoch, epoch).unwrap();
    assert_eq!(live.len(), 2, "{live:?}");
    let identity = live
        .iter()
        .find(|event| event.before.is_none())
        .expect("Big's delete leaves its properties out");
    assert_eq!(identity.labels, Some(vec!["Person".to_string()]));
    assert!(
        live.iter().any(|event| event
            .before
            .as_ref()
            .is_some_and(|before| before.len() == 1)),
        "Small's delete keeps its name"
    );
    let delete_batch = |db: &GrafeoDB| {
        db.read_cdc_feed(ALL)
            .unwrap()
            .into_iter()
            .find(|batch| batch.position.epoch() == epoch)
            .expect("the delete's commit is in the feed")
    };
    assert_eq!(
        as_json(&delete_batch(&db).events),
        as_json(&live),
        "before the close"
    );
    db.close().unwrap();
    drop(db);
    assert_eq!(
        as_json(&delete_batch(&open(&path)).events),
        as_json(&live),
        "after the close"
    );
}

/// The live events carry the timestamps each commit reserved with its group, the run the feed
/// reads back from the group's first timestamp: per commit the two lists are equal, and they
/// rise. A publish that took fresh readings of the clock would hand out later ones.
#[test]
fn the_live_timestamps_are_the_ones_each_commit_reserved() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let db = with_workload(&path);
    let batches = db.read_cdc_feed(ALL).unwrap();
    assert!(batches.len() > 20, "{} commits", batches.len());
    let stamps = |events: &[ChangeEvent]| {
        events
            .iter()
            .map(|event| event.timestamp)
            .collect::<Vec<_>>()
    };
    for batch in &batches {
        let epoch = batch.position.epoch();
        let live = db.changes_between(epoch, epoch).unwrap();
        assert_eq!(stamps(&live), stamps(&batch.events), "epoch {epoch:?}");
        assert!(
            stamps(&batch.events)
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "epoch {epoch:?}: the timestamps rise"
        );
    }
}

/// An entity that exists before and after a transaction keeps one update per write: two sets of
/// one property read back from the log as two updates, in order, each with its own before and
/// after, the same as the live log reports them.
#[test]
fn each_update_of_an_entity_that_lives_through_a_transaction_reads_back() {
    use grafeo_common::types::Value;
    use grafeo_engine::cdc::ChangeKind;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let db = open(&path);
    db.set_cdc(true).unwrap();
    db.execute("INSERT (:Person {name: 'Gus', age: 28})")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 29")
        .unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 28")
        .unwrap();
    session.commit().unwrap();
    let epoch = db.current_epoch();
    let live = db.changes_between(epoch, epoch).unwrap();
    let age = |age: i64| Some([("age".to_string(), Value::from(age))].into());
    let updates: Vec<_> = live
        .iter()
        .map(|event| {
            (
                event.kind.clone(),
                event.before.clone(),
                event.after.clone(),
            )
        })
        .collect();
    assert_eq!(
        updates,
        [
            (ChangeKind::Update, age(28), age(29)),
            (ChangeKind::Update, age(29), age(28)),
        ]
    );
    let batch = db
        .read_cdc_feed(ALL)
        .unwrap()
        .into_iter()
        .find(|batch| batch.position.epoch() == epoch)
        .expect("the transaction's commit is in the feed");
    assert_eq!(as_json(&batch.events), as_json(&live));
}

/// Every bulk write form logs one record per entity, so its events and timestamps read back as
/// live reported them: batch creates of nodes and edges, a statement over many rows, and an
/// import, which reports none either way.
#[test]
fn bulk_writes_read_back_as_live_reported_them() {
    use std::collections::HashMap;

    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_engine::database::BatchEdge;

    let dir = tempfile::tempdir().unwrap();
    let (path, db) = file_with_cdc(&dir);
    let person = |name: &str| HashMap::from([(PropertyKey::new("name"), Value::from(name))]);
    let people = db
        .batch_create_nodes_with_props(
            "Person",
            vec![person("Vincent"), person("Jules"), person("Mia")],
        )
        .unwrap();
    db.batch_create_nodes_with_labels(&["Person", "Boxer"], vec![person("Butch")])
        .unwrap();
    db.batch_create_edges(vec![
        BatchEdge {
            src: people[0],
            dst: people[1],
            edge_type: "KNOWS".to_string(),
            properties: HashMap::from([(PropertyKey::new("since"), Value::from(1994_i64))]),
        },
        BatchEdge {
            src: people[1],
            dst: people[2],
            edge_type: "KNOWS".to_string(),
            properties: HashMap::new(),
        },
    ])
    .unwrap();
    db.execute("UNWIND ['Amsterdam', 'Utrecht', 'Leiden'] AS city INSERT (:City {name: city})")
        .unwrap();
    db.import_tsv_str("1\t2\n2\t3\n", "ROAD", true).unwrap();

    let live = live(&db);
    assert_eq!(
        live.len(),
        3 + 1 + 2 + 3,
        "the batches and the rows report one event each, the import none: {live:?}"
    );
    assert_eq!(
        durable(&db.read_cdc_feed(ALL).unwrap()),
        live,
        "before the close"
    );
    db.close().unwrap();
    drop(db);
    let reopened = open(&path);
    assert_eq!(
        durable(&reopened.read_cdc_feed(ALL).unwrap()),
        live,
        "after the close"
    );
}

/// The S006 a read fails with: the position it names.
fn truncated_at(result: grafeo_common::utils::error::Result<Vec<CdcBatch>>) -> CdcPosition {
    use grafeo_common::utils::error::{Error, StorageError};
    match result {
        Err(Error::Storage(storage)) => match *storage {
            StorageError::CdcFeedTruncated {
                first_available,
                epoch,
            } => CdcPosition::from_token(&format!("{first_available}:{epoch}")).unwrap(),
            other => panic!("the read is not S006: {other}"),
        },
        Err(other) => panic!("the read is not S006: {other}"),
        Ok(batches) => panic!("the read gave {} batches, not S006", batches.len()),
    }
}

/// A WAL directory removed while the database was closed (as a 0.6.3 close removes it, cursor
/// included) leaves the stored setting on: the next open starts the feed again at the file's
/// checkpoint and marks it restarted, so the first read fails with S006 naming where it goes on
/// from, as for a log cut past the retention limit. Acknowledging that position goes on; the
/// commits that were not acknowledged before went with the log.
#[test]
fn a_removed_wal_directory_restarts_the_feed_with_s006() {
    let dir = tempfile::tempdir().unwrap();
    let (path, db) = file_with_cdc(&dir);
    three_groups(&db);
    let first = db.read_cdc_feed(ALL).unwrap();
    db.acknowledge_cdc_feed(first[0].position).unwrap();
    db.close().unwrap();
    drop(db);
    assert!(sidecar_wal(&path).exists(), "the feed kept its log");
    std::fs::remove_dir_all(sidecar_wal(&path)).unwrap();

    let reopened = open(&path);
    assert!(
        reopened.is_cdc_enabled(),
        "the setting is stored in the file"
    );
    reopened
        .execute("INSERT (:Person {name: 'Shosanna'})")
        .unwrap();
    let restart = truncated_at(reopened.read_cdc_feed(ALL));
    assert_eq!(
        truncated_at(reopened.read_cdc_feed(ALL)),
        restart,
        "a read does not clear it"
    );
    reopened.acknowledge_cdc_feed(restart).unwrap();
    let batches = reopened.read_cdc_feed(ALL).unwrap();
    assert_eq!(
        positions(&batches),
        [restart],
        "the feed goes on from the first commit after the restart"
    );
    reopened.acknowledge_cdc_feed(batches[0].position).unwrap();
    assert!(reopened.read_cdc_feed(ALL).unwrap().is_empty());
    reopened.close().unwrap();
    drop(reopened);
    assert!(
        open(&path).read_cdc_feed(ALL).unwrap().is_empty(),
        "the acknowledgement cleared the restart for good"
    );
}

/// A feed that `set_cdc(true)` starts, and one a reopen finds with its cursor, is not restarted:
/// its reads give the commits.
#[test]
fn an_ordinary_start_is_not_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let (path, db) = file_with_cdc(&dir);
    three_groups(&db);
    assert_eq!(
        db.read_cdc_feed(ALL).unwrap().len(),
        3,
        "after set_cdc(true)"
    );
    db.close().unwrap();
    drop(db);
    assert_eq!(
        open(&path).read_cdc_feed(ALL).unwrap().len(),
        3,
        "after a reopen with the cursor there"
    );
}
