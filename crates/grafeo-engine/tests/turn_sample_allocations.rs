//! The turn's sample reads a few lists, not the label: the first plan of a candidate form allocates
//! about the same over a label of 20,000 starts as over one of 200,000, hot and reopened. Counted in
//! bytes, so it holds in any build and on any machine.
//!
//! A test binary of its own: the counting allocator is global to the binary.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test turn_sample_allocations
//! ```

#![allow(unsafe_code, reason = "a global allocator that counts")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

struct Counting;

static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to the system allocator and only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Held by each test for its whole run: the tests of a binary run in parallel, and another test's
/// allocations would count in this one's bytes.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

const CANDIDATE: &str = "EXPLAIN MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.s, m.k";

fn fill(db: &GrafeoDB, starts: i64) {
    for s in 0..starts {
        let a = db
            .create_node_with_props(&["Start"], [("s", Value::Int64(s))])
            .unwrap();
        let far = db
            .create_node_with_props(&["Far"], [("k", Value::Int64(s % 5_000))])
            .unwrap();
        db.create_edge(a, far, "R").unwrap();
    }
    db.session()
        .execute("CREATE INDEX by_far FOR (n:Far) ON (n.k)")
        .unwrap();
}

/// The bytes the first plan of the candidate allocates, in a fresh session.
fn first_plan_bytes(db: &GrafeoDB) -> (u64, String) {
    let session = db.session();
    let before = BYTES.load(Ordering::Relaxed);
    let plan = session.execute(CANDIDATE).unwrap();
    let after = BYTES.load(Ordering::Relaxed);
    let text = plan.rows()[0][0].as_str().unwrap().to_string();
    (after - before, text)
}

/// The cap: a first plan may grow with the label by no more than this factor over ten times the
/// starts. Listing the label grows it about tenfold.
const GROWTH: f64 = 2.0;

#[test]
fn the_first_plan_does_not_grow_with_the_label() {
    let _one = one_at_a_time();
    let mut hot = Vec::new();
    let mut reopened = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    for starts in [20_000_i64, 200_000] {
        let db = GrafeoDB::new_in_memory();
        fill(&db, starts);
        hot.push(first_plan_bytes(&db).0);
        let path = dir.path().join(format!("sample-{starts}.grafeo"));
        {
            let db = GrafeoDB::open(&path).unwrap();
            fill(&db, starts);
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        reopened.push(first_plan_bytes(&db).0);
        db.close().unwrap();
    }
    for (stage, bytes) in [("hot", &hot), ("reopened", &reopened)] {
        #[allow(clippy::cast_precision_loss, reason = "a ratio of byte counts")]
        let growth = bytes[1] as f64 / bytes[0].max(1) as f64;
        assert!(
            growth <= GROWTH,
            "{stage}: {} bytes over 20,000 starts, {} over 200,000: {growth:.1} times",
            bytes[0],
            bytes[1]
        );
    }
}

/// The sample gives the same plan every time: across fresh sessions, and across opens of one file.
#[test]
fn the_plan_is_the_same_between_runs() {
    let _one = one_at_a_time();
    let db = GrafeoDB::new_in_memory();
    fill(&db, 20_000);
    let plans: Vec<String> = (0..5).map(|_| first_plan_bytes(&db).1).collect();
    assert!(
        plans.windows(2).all(|pair| pair[0] == pair[1]),
        "{plans:#?}"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stable.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        fill(&db, 20_000);
        db.close().unwrap();
    }
    let mut opened = Vec::new();
    for _ in 0..3 {
        let db = GrafeoDB::open(&path).unwrap();
        opened.push(first_plan_bytes(&db).1);
        db.close().unwrap();
    }
    assert!(
        opened.windows(2).all(|pair| pair[0] == pair[1]),
        "{opened:#?}"
    );
}

/// A sample that finds nothing to read never fails the plan: an empty label, a label whose nodes
/// were all deleted, starts with no edges, and nodes only an open transaction holds.
#[test]
fn a_sample_never_fails_planning() {
    let _one = one_at_a_time();
    let query = "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.s, m.k";
    let cases: [(&str, &[&str]); 4] = [
        ("an empty label", &["INSERT (:Far {k: 7})"]),
        (
            "deleted starts",
            &[
                "UNWIND range(1, 2000) AS i INSERT (:Start {s: i})-[:R]->(:Far {k: i})",
                "MATCH (a:Start) DETACH DELETE a",
            ],
        ),
        (
            "starts with no edges",
            &["UNWIND range(1, 2000) AS i INSERT (:Start {s: i}), (:Far {k: i})"],
        ),
        ("one start", &["INSERT (:Start {s: 1})-[:R]->(:Far {k: 7})"]),
    ];
    for (name, statements) in cases {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for statement in statements {
            session.execute(statement).unwrap();
        }
        session
            .execute("CREATE INDEX by_far FOR (n:Far) ON (n.k)")
            .unwrap();
        session
            .execute(&format!("EXPLAIN {query}"))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        session
            .execute(query)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("CREATE INDEX by_far FOR (n:Far) ON (n.k)")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("UNWIND range(1, 2000) AS i INSERT (:Start {s: i})-[:R]->(:Far {k: i})")
        .unwrap();
    let rows = session.execute(query).unwrap();
    assert_eq!(rows.row_count(), 1, "the transaction's own nodes");
    session.rollback().unwrap();
}
