//! What the RDF store keeps for readers at an older snapshot is in the memory account, and goes
//! when no reader needs it (#718).
//!
//! A commit that deletes or clears what an open transaction still reads keeps the old triples for
//! it, and a commit of new triples keeps them for the conflict checks of the transactions that
//! began before it. The memory limit grants writes by `memory_usage()`'s account, so what is kept
//! must be in it (`rdf.snapshot_bytes`), and it must go once no reader needs it, without waiting
//! for another commit.
//!
//! - **The account against its walk:** at every step `snapshot_account_mismatch()` is `None` (the
//!   overlay's counted bytes equal a walk of what it holds), and `snapshot_bytes` is of the order
//!   the kept triples hold and grows with them.
//! - **Freed when the reader ends:** the oracle is the allocator, which this binary counts: once
//!   the reader ends, with no commit after, the process frees at least four fifths of what the
//!   same statement frees with no reader.
//! - **The allocator against the account:** over a delete and a clear a reader keeps, the commit,
//!   the reader's read and its end move the allocator and the report alike (with the RDF term
//!   text and each triple's allocation header counted).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_overlay_in_the_account -- --test-threads 1
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this one forwards every call to System"
)]
#![allow(
    deprecated,
    reason = "the overlay's walk is read from the RDF store, which no session shows"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static FREED: AtomicUsize = AtomicUsize::new(0);

/// Forwards to [`System`] and counts the bytes of each call.
struct Counting;

// SAFETY: every method forwards to `System` with the caller's arguments, so `System`'s
// guarantees are this allocator's; the counters are only read by the tests.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc`, passed on.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc_zeroed`, passed on.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc`, passed on.
        unsafe { System.dealloc(pointer, layout) };
        FREED.fetch_add(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract for `realloc`, passed on.
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
            FREED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        moved
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// One measurement at a time: another test's allocations would land in its difference.
static SERIAL: Mutex<()> = Mutex::new(());

/// The bytes the process holds now.
fn held() -> isize {
    let allocated = ALLOCATED.load(Ordering::Relaxed);
    let freed = FREED.load(Ordering::Relaxed);
    isize::try_from(allocated).unwrap() - isize::try_from(freed).unwrap()
}

/// The triples each test writes, far above any fixed cost.
const TRIPLES: usize = 20_000;

/// What a kept triple may count in the overlay, at least and at most: its slot and its `Arc`
/// at the least, its own partition's indexes at the most. A count outside these is not the kept
/// triples' (nothing counted, or counted many times over).
const PER_TRIPLE: std::ops::RangeInclusive<usize> = 32..=4_096;

const PREFIX: &str = "PREFIX ex: <http://ex.org/> ";

/// An `INSERT DATA` of `count` triples `ex:<subject><i> ex:v "<i>"`, in the default graph.
fn insert_data(subject: &str, count: usize) -> String {
    let mut text = format!("{PREFIX}INSERT DATA {{ ");
    for i in 0..count {
        write!(text, "ex:{subject}{i} ex:v \"{i}\" . ").unwrap();
    }
    text.push('}');
    text
}

const COUNT: &str = "SELECT ?s WHERE { ?s ex:v ?o }";

/// How many triples of `ex:v` the session sees.
fn count(session: &Session) -> usize {
    session
        .execute_sparql(&format!("{PREFIX}{COUNT}"))
        .unwrap()
        .row_count()
}

/// A database holding `triples` triples, and a transaction on it that has read them.
fn with_reader(db: &GrafeoDB, triples: usize) -> Session {
    db.execute_sparql(&insert_data("d", triples)).unwrap();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    assert_eq!(count(&reader), triples);
    reader
}

/// What the overlay reports holding for readers at older snapshots.
fn snapshot_bytes(db: &GrafeoDB) -> usize {
    db.memory_usage().rdf.snapshot_bytes
}

/// The overlay's account equals its walk.
fn assert_walked(db: &GrafeoDB, what: &str) {
    assert_eq!(
        db.rdf_store().snapshot_account_mismatch(),
        None,
        "{what}: the overlay's account equals its walk"
    );
}

/// `bytes` is what `triples` kept triples may count (see [`PER_TRIPLE`]).
fn assert_of_the_order_of(bytes: usize, triples: usize, what: &str) {
    assert!(
        (PER_TRIPLE.start() * triples..=PER_TRIPLE.end() * triples).contains(&bytes),
        "{what}: {bytes} B kept for {triples} triples"
    );
}

/// What a write that removes every triple frees with no reader open: the oracle of what it frees
/// once its reader ends.
fn freed_without_a_reader(statement: &str) -> isize {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&insert_data("d", TRIPLES)).unwrap();
    let before = held();
    db.execute_sparql(&format!("{PREFIX}{statement}")).unwrap();
    before - held()
}

/// What the overlay keeps for a reader after `statement` (which removes every triple), with
/// `triples` triples, checked against its walk on the way.
fn kept_for_a_reader(statement: &str, triples: usize) -> usize {
    let db = GrafeoDB::new_in_memory();
    let mut reader = with_reader(&db, triples);
    db.execute_sparql(&format!("{PREFIX}{statement}")).unwrap();
    assert_walked(&db, "after the commit");
    let kept = snapshot_bytes(&db);
    reader.rollback().unwrap();
    kept
}

/// A delete or a clear of what an older transaction still reads:
/// - the triples kept for it are in the account and its walk, of the order they hold, through
///   the commit and the reader's read after it, and twice the triples keep about twice as much;
/// - when it ends, with no commit after, the account falls to nothing and the process frees what
///   the statement frees with no reader.
fn kept_for_a_reader_and_freed_when_it_ends(statement: &str) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let freed_alone = freed_without_a_reader(statement);
    let db = GrafeoDB::new_in_memory();
    let mut reader = with_reader(&db, TRIPLES);
    assert_eq!(snapshot_bytes(&db), 0, "nothing kept before the commit");
    let before = held();

    db.execute_sparql(&format!("{PREFIX}{statement}")).unwrap();
    assert_walked(
        &db,
        &format!("`{statement}` while a transaction reads the triples"),
    );
    let kept = snapshot_bytes(&db);
    assert_of_the_order_of(kept, TRIPLES, &format!("after `{statement}`"));

    assert_eq!(count(&reader), TRIPLES, "the reader reads its snapshot");
    assert_walked(&db, "the reader's read after the commit");
    assert!(
        snapshot_bytes(&db) >= kept,
        "the reader's view is counted beside what the overlay keeps"
    );

    reader.rollback().unwrap();
    assert_walked(&db, "the reader's end");
    assert_eq!(
        snapshot_bytes(&db),
        0,
        "`{statement}`: nothing is kept once the reader ends, with no commit after"
    );
    let freed = before - held();
    assert!(
        freed * 5 >= freed_alone * 4,
        "`{statement}`: {freed} B freed once the reader ended, {freed_alone} B with no reader"
    );
    assert_eq!(count(&db.session()), 0);

    let twice = kept_for_a_reader(statement, 2 * TRIPLES);
    assert!(
        twice * 2 >= kept * 3 && twice * 2 <= kept * 5,
        "`{statement}`: {kept} B kept for {TRIPLES} triples, {twice} B for twice as many"
    );
}

#[test]
fn deleted_triples_kept_for_a_reader_are_counted_and_freed_when_it_ends() {
    kept_for_a_reader_and_freed_when_it_ends("DELETE WHERE { ?s ex:v ?o }");
}

#[test]
fn a_cleared_graph_kept_for_a_reader_is_counted_and_freed_when_it_ends() {
    kept_for_a_reader_and_freed_when_it_ends("CLEAR DEFAULT");
}

/// A commit of new triples while an older transaction is open keeps its writes for that
/// transaction's conflict check: they are in the account and its walk, of the order they hold,
/// and they go when the older transaction ends.
#[test]
fn writes_kept_for_an_older_transaction_are_counted_and_released() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let db = GrafeoDB::new_in_memory();
    let mut older = with_reader(&db, TRIPLES);
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute_sparql(&insert_data("n", TRIPLES)).unwrap();
    writer.commit().unwrap();
    assert_walked(
        &db,
        "a commit of new triples while an older transaction is open",
    );
    assert_of_the_order_of(snapshot_bytes(&db), TRIPLES, "the writes kept");

    older.rollback().unwrap();
    assert_walked(&db, "the older transaction's end");
    assert_eq!(
        snapshot_bytes(&db),
        0,
        "nothing is kept once the older transaction ends"
    );
    assert_eq!(count(&db.session()), 2 * TRIPLES);
}

/// What the allocator and the report may differ by over a step: the few kilobytes a statement
/// keeps (its cached plan, a dictionary's next room).
const MARGIN: isize = 64 * 1024;

/// The allocator against the account while a reader keeps what `statement` removed: over the
/// commit, the reader's read and its end, the allocator and the report move by the same bytes
/// (the kept triples with their headers and text, the swapped-out partition, the reader's view).
fn kept_for_a_reader_is_counted_as_the_allocator_holds_it(statement: &str) {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let db = GrafeoDB::new_in_memory();
    let mut reader = with_reader(&db, TRIPLES);
    let mut wrong = Vec::new();
    let mut step = |what: &str, run: &mut dyn FnMut()| {
        let report_before = isize::try_from(db.memory_usage().total_bytes).unwrap();
        let before = held();
        run();
        let after = held();
        let report_after = isize::try_from(db.memory_usage().total_bytes).unwrap();
        let (allocated, reported) = (after - before, report_after - report_before);
        if (allocated - reported).abs() > MARGIN {
            wrong.push(format!(
                "`{statement}`, {what}: {allocated} B by the allocator, {reported} B in the report"
            ));
        }
    };
    step("the commit", &mut || {
        db.execute_sparql(&format!("{PREFIX}{statement}")).unwrap();
    });
    step("the reader's read", &mut || {
        assert_eq!(count(&reader), TRIPLES);
    });
    step("the reader's end", &mut || reader.rollback().unwrap());
    assert!(
        wrong.is_empty(),
        "{}",
        wrong.join(
            "
"
        )
    );
}

#[test]
fn deleted_triples_kept_for_a_reader_are_counted_as_the_allocator_holds_them() {
    kept_for_a_reader_is_counted_as_the_allocator_holds_it("DELETE WHERE { ?s ex:v ?o }");
}

#[test]
fn a_cleared_graph_kept_for_a_reader_is_counted_as_the_allocator_holds_it() {
    kept_for_a_reader_is_counted_as_the_allocator_holds_it("CLEAR DEFAULT");
}
