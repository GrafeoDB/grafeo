//! What a SPARQL read at an older snapshot waits for, and what a commit waits for (#718).
//!
//! - A commit finishes while readers keep reading: steady SPARQL readers, whose property paths
//!   take nested reads, never keep a commit (and the LPG commits after it) waiting.
//! - A commit's conflict check never waits for another commit's RDF apply while it holds the
//!   transactions lock, so begins and writes elsewhere go on.
//!
//! The oracle is a deadline far above what the work takes.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_overlay_guarantees::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store", feature = "gql"))]

use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use grafeo_engine::GrafeoDB;

const PREFIX: &str = "PREFIX ex: <http://ex.org/> ";

/// How many rows the query returns.
fn count(session: &grafeo_engine::session::Session, query: &str) -> usize {
    session
        .execute_sparql(&format!("{PREFIX}{query}"))
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .row_count()
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
