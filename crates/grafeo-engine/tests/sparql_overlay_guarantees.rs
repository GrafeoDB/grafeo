//! What a SPARQL read at an older snapshot costs and waits for, and what a commit waits for
//! (#718).
//!
//! - A transaction whose snapshot predates many deletes reads and writes in time that does not
//!   grow with those deletes: its triple lookups (a write's check of what its snapshot holds, a
//!   property path's hop) find the deleted triples by key, not by a walk of every one.
//! - A commit finishes while readers keep reading: steady SPARQL readers, whose property paths
//!   take nested reads, never keep a commit (and the LPG commits after it) waiting.
//! - A commit's conflict check never waits for another commit's RDF apply while it holds the
//!   transactions lock, so begins and writes elsewhere go on.
//! - A load's publish builds the write set of the transactions open across it before it takes
//!   the transactions lock, so a begin never waits for that build.
//!
//! The first two oracles are ratios between 1,000 and 16,000 deleted links. A property path's
//! work is counted by the deadline probe (`testing-deadline-probe`), the same on every machine,
//! and grows with its hops (sixteen times), where a walk of the deletes per hop would make it
//! about 256 times. The writes are the same at both sizes and timed: less than four times as long,
//! where a walk per write would take about sixteen. The others are deadlines far above what the
//! work takes.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_overlay_guarantees::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store", feature = "gql"))]

use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use grafeo_engine::GrafeoDB;

const PREFIX: &str = "PREFIX ex: <http://ex.org/> ";

/// The links a transaction writes at its old snapshot.
const WRITES: usize = 8_000;

/// How many rows the query returns.
fn count(session: &grafeo_engine::session::Session, query: &str) -> usize {
    session
        .execute_sparql(&format!("{PREFIX}{query}"))
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .row_count()
}

/// An `INSERT DATA` of the chain `ex:d<i> ex:v ex:d<i + step>` for `i` in `0..count`.
fn chain(count: usize, step: usize) -> String {
    let mut text = format!("{PREFIX}INSERT DATA {{ ");
    for i in 0..count {
        write!(text, "ex:d{i} ex:v ex:d{} . ", i + step).unwrap();
    }
    text.push('}');
    text
}

/// A database whose chain of `deleted` links a commit deleted after `reader` began: the
/// reader's snapshot still holds every link, the store none.
fn snapshot_of_a_deleted_chain(deleted: usize) -> (GrafeoDB, grafeo_engine::session::Session) {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&chain(deleted, 1)).unwrap();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(count(&reader, LINKS), deleted, "the snapshot");
    db.execute_sparql(&format!("{PREFIX}DELETE WHERE {{ ?s ex:v ?o }}"))
        .unwrap();
    (db, reader)
}

/// Every link of the chain, as a pattern.
const LINKS: &str = "SELECT ?s WHERE { ?s ex:v ?o }";

/// Whether the property path `ex:d0 ex:v+ ?o` reads `reader`'s snapshot of a deleted chain of
/// `deleted` links to its end within `polls` clock reads under the deadline probe. One path
/// search counts its work on one ticker, which reads the clock on its first unit and every
/// 1,024th, so the reads grow with the work. A timeout is the only other outcome.
#[cfg(feature = "testing-deadline-probe")]
fn path_finishes(reader: &grafeo_engine::session::Session, deleted: usize, polls: u32) -> bool {
    use grafeo_common::utils::error::{Error, QueryErrorKind};
    use grafeo_core::execution::deadline;

    let query = format!("{PREFIX}SELECT ?o WHERE {{ ex:d0 ex:v+ ?o }}");
    let _probe = deadline::expire_after_polls(polls);
    match reader.execute_sparql(&query) {
        Ok(result) => {
            assert_eq!(
                result.row_count(),
                deleted,
                "the path reaches the chain's end"
            );
            true
        }
        Err(Error::Query(error)) if error.kind == QueryErrorKind::Timeout => false,
        Err(error) => panic!("{query} under {polls} clock reads: {error}"),
    }
}

/// A property path at a snapshot older than sixteen times the deleted links finishes within
/// twice sixteen times the clock reads: its hops find the snapshot's deleted triples by key. The
/// work is counted by the deadline probe, not timed, so the bound is the same on every machine:
/// a path of n hops is n lookups, sixteen times the work for sixteen times the links, where a
/// walk of the deletes at each hop would be about 256 times.
#[cfg(feature = "testing-deadline-probe")]
#[test]
fn a_path_at_an_old_snapshot_does_not_walk_its_deletes() {
    let (_small_db, small) = snapshot_of_a_deleted_chain(1_000);
    assert!(
        !path_finishes(&small, 1_000, 0),
        "the path counts its work on the probe"
    );
    // The fewest clock reads the small path needs: more than `too_few`, at most `enough`.
    let mut enough = 1;
    while !path_finishes(&small, 1_000, enough) {
        enough = enough.checked_mul(2).expect("the path finishes");
    }
    let mut too_few = enough / 2;
    while enough - too_few > 1 {
        let middle = too_few + (enough - too_few) / 2;
        if path_finishes(&small, 1_000, middle) {
            enough = middle;
        } else {
            too_few = middle;
        }
    }
    let (_large_db, large) = snapshot_of_a_deleted_chain(16_000);
    assert!(
        path_finishes(&large, 16_000, enough * 32),
        "the path needs {enough} clock reads at 1,000 deleted links, more than {} at 16,000",
        enough * 32
    );
}

/// How long a transaction at a snapshot of a deleted chain of `deleted` links takes to write
/// [`WRITES`] new links between the chain's nodes, each checked against what the snapshot holds
/// (every term of them is known, so each check reaches the deleted links). The same writes at
/// every size. Checks the answers on the way: the snapshot's, and the state after its commit.
fn writes_at_an_old_snapshot(deleted: usize) -> Duration {
    let (db, mut reader) = snapshot_of_a_deleted_chain(deleted);
    // Links two steps on: new triples between known nodes (the ones past the chain's end name
    // new nodes, which cost nothing to check).
    let started = Instant::now();
    reader.execute_sparql(&chain(WRITES, 2)).unwrap();
    let writes = started.elapsed();
    assert_eq!(
        count(&reader, LINKS),
        deleted + WRITES,
        "the snapshot with the transaction's writes"
    );
    reader.commit().unwrap();
    assert_eq!(
        count(&db.session(), LINKS),
        WRITES,
        "the delete and the transaction's writes stand"
    );
    writes
}

/// The same writes at a snapshot older than sixteen times the deleted links take less than four
/// times as long: each write's check finds the deleted triples by key. The work does not change
/// with the deletes, so the expected ratio is about one, and a walk of the deletes per write
/// about sixteen. The fastest of three runs per size keeps a slow moment of the machine out.
#[test]
fn writes_at_an_old_snapshot_do_not_walk_its_deletes() {
    let fastest = |deleted| {
        (0..3)
            .map(|_| writes_at_an_old_snapshot(deleted))
            .min()
            .expect("three runs")
    };
    let small = fastest(1_000);
    let large = fastest(16_000);
    assert!(
        large < small * 4,
        "the writes: {small:?} at 1,000 deleted links, {large:?} at 16,000"
    );
}

/// A commit, and an LPG commit after it, finish while four sessions keep running property-path
/// reads, each of which takes nested reads of the store: readers never hold a commit off for
/// long. The deadline is far above what the commits take.
#[test]
fn a_commit_finishes_while_readers_keep_reading() {
    const READERS: usize = 4;
    const CHAIN: usize = 400;
    let db = Arc::new(GrafeoDB::new_in_memory());
    let mut chain = format!("{PREFIX}INSERT DATA {{ ");
    for i in 0..CHAIN {
        write!(chain, "ex:p{i} ex:knows ex:p{} . ", i + 1).unwrap();
    }
    chain.push('}');
    db.execute_sparql(&chain).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..READERS)
        .map(|_| {
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let session = db.session();
                let path = format!("{PREFIX}SELECT ?o WHERE {{ ex:p0 ex:knows+ ?o }}");
                let mut reads = 0_u64;
                while !stop.load(Ordering::Relaxed) {
                    session.execute_sparql(&path).unwrap();
                    reads += 1;
                }
                reads
            })
        })
        .collect();
    // The readers are reading before the commit starts.
    std::thread::sleep(Duration::from_millis(200));

    let (done, finished) = mpsc::channel();
    let writer = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute_sparql(&format!(
                    "{PREFIX}INSERT DATA {{ ex:alix ex:knows ex:gus . }}"
                ))
                .unwrap();
            session.commit().unwrap();
            done.send("the SPARQL commit").unwrap();
            db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
            done.send("the LPG commit after it").unwrap();
        })
    };
    let deadline = Duration::from_secs(30);
    let first = finished.recv_timeout(deadline);
    let second = finished.recv_timeout(deadline);
    stop.store(true, Ordering::Relaxed);
    let reads: u64 = readers
        .into_iter()
        .map(|reader| reader.join().unwrap())
        .sum();
    assert_eq!(
        first,
        Ok("the SPARQL commit"),
        "the SPARQL commit waited past {deadline:?} under {READERS} readers ({reads} reads)"
    );
    assert_eq!(
        second,
        Ok("the LPG commit after it"),
        "the LPG commit after it waited past {deadline:?}"
    );
    writer.join().unwrap();
    assert!(reads > 0, "the readers read");
    let session = db.session();
    assert_eq!(
        count(&session, "SELECT ?o WHERE { ex:alix ex:knows ?o }"),
        1
    );
}

/// A commit's conflict check never waits for another commit's RDF apply while it holds the
/// transactions lock (#718, C1): T1 commits a triple and is held in its apply, its change set
/// locked. T2, which began before T1 committed and wrote another triple, commits meanwhile, and
/// its check runs against T1's writes. While both are in flight, another session begins a
/// transaction and writes, which needs the transactions lock; it finishes within the deadline.
/// T2's own commit waits for T1's publish (commits publish in order), so it is not what is timed.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_conflict_check_does_not_wait_for_another_commits_apply() {
    use grafeo_common::testing::commit_hook::{RdfApplyStep, during_next_rdf_apply};

    let db = Arc::new(GrafeoDB::new_in_memory());
    db.execute_sparql(&format!("{PREFIX}INSERT DATA {{ ex:alix ex:p \"1\" . }}"))
        .unwrap();
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first
        .execute_sparql(&format!("{PREFIX}INSERT DATA {{ ex:gus ex:p \"2\" . }}"))
        .unwrap();
    second
        .execute_sparql(&format!("{PREFIX}INSERT DATA {{ ex:mia ex:p \"3\" . }}"))
        .unwrap();

    let (answer, answered) = mpsc::channel();
    let in_hook = Arc::clone(&db);
    let second_commit = Arc::new(std::sync::Mutex::new(None));
    let second_slot = Arc::clone(&second_commit);
    during_next_rdf_apply(RdfApplyStep::GuardTaken, move || {
        // T2 commits on a thread of its own and reaches its conflict check.
        let committing = std::thread::spawn(move || second.commit());
        std::thread::sleep(Duration::from_millis(300));
        // Another session begins and writes: it needs the transactions lock.
        let other = Arc::clone(&in_hook);
        let (done, finished) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            let mut session = other.session();
            session.begin_transaction().unwrap();
            session
                .execute("INSERT (:Person {name: 'Vincent'})")
                .unwrap();
            session.rollback().unwrap();
            // The hook stops listening at its deadline: a late send is not an error here.
            let _ = done.send(());
        });
        answer
            .send(finished.recv_timeout(Duration::from_secs(10)))
            .unwrap();
        // The hook returns: T1's apply goes on, so both threads end.
        *second_slot.lock().unwrap() = Some((committing, writer));
    });
    first.commit().unwrap();
    let waited = answered.recv().unwrap();
    let (committing, writer) = second_commit.lock().unwrap().take().unwrap();
    writer.join().unwrap();
    let second_result = committing.join().unwrap();
    assert!(
        waited.is_ok(),
        "a begin and a write waited for T1's apply while T2's commit checked against it"
    );
    second_result.expect("T2 wrote another triple: it commits");
    let session = db.session();
    assert_eq!(count(&session, "SELECT ?s WHERE { ?s ex:p ?o }"), 3);
}

/// A begin never waits for a load's write set (#718, E2): with a transaction open, a load into a
/// named graph builds the writes that transaction's commit checks against, held here by the
/// `during_next_load_writes` hook. Meanwhile another session begins a transaction and writes,
/// which needs the transactions lock; it finishes within the deadline, and the load lands whole.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_begin_does_not_wait_for_a_loads_write_set() {
    use grafeo_common::testing::commit_hook::during_next_load_writes;
    use grafeo_engine::RdfFormat;

    const LOADED: usize = 1_000;
    let db = Arc::new(GrafeoDB::new_in_memory());
    let mut open = db.session();
    open.begin_transaction().unwrap();
    assert_eq!(count(&open, "SELECT ?s WHERE { ?s ?p ?o }"), 0);

    let (answer, answered) = mpsc::channel();
    let in_hook = Arc::clone(&db);
    during_next_load_writes(move || {
        let other = Arc::clone(&in_hook);
        let (done, finished) = mpsc::channel();
        let writer = std::thread::spawn(move || {
            let mut session = other.session();
            session.begin_transaction().unwrap();
            session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
            session.rollback().unwrap();
            // The hook stops listening at its deadline: a late send is not an error here.
            let _ = done.send(());
        });
        answer
            .send((finished.recv_timeout(Duration::from_secs(10)), writer))
            .unwrap();
    });
    let mut text = String::new();
    for i in 0..LOADED {
        writeln!(text, "<http://ex.org/s{i}> <http://ex.org/v> \"{i}\" .").unwrap();
    }
    db.load_rdf_text(&text, RdfFormat::NTriples, Some("http://ex.org/g"))
        .unwrap();
    let (waited, writer) = answered
        .recv()
        .expect("the load built its write set for the open transaction");
    writer.join().unwrap();
    assert!(
        waited.is_ok(),
        "a begin and a write waited for the load's write set"
    );
    open.rollback().unwrap();
    assert_eq!(
        count(
            &db.session(),
            "SELECT ?s WHERE { GRAPH <http://ex.org/g> { ?s ?p ?o } }"
        ),
        LOADED
    );
}
