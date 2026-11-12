//! Node rows in the memory account (#751, from #431's `memory_per_entity` bench).
//!
//! The memory limit grants writes by `memory_usage()`'s account, so what nodes hold must be in it.
//! The account and its walk (`RowGroupStore::memory_breakdown`) share their line figures, so a
//! test of one against the other cannot see a part that both leave out. The oracle here is the
//! allocator: this binary counts every byte the process allocates and frees.
//!
//! A node of the row-group store holds about a third of a byte (its label bit and its share of
//! the row group's version info), and the report counts it: the node phase's allocated and
//! reported bytes differ by a fixed few kilobytes at every size, never by an amount per node.
//! (The 8.4 B per node `memory_per_entity` once showed was its own vector of the ids.)
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test node_rows_in_the_account -- --test-threads 1
//! ```

#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this one forwards every call to System"
)]
#![allow(
    deprecated,
    reason = "the walk reads the raw store, which no session shows"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_engine::{Config, GrafeoDB};

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

fn held() -> usize {
    ALLOCATED
        .load(Ordering::Relaxed)
        .wrapping_sub(FREED.load(Ordering::Relaxed))
}

/// What `step` adds: by the allocator, and in the report's total. The report is read before each
/// count, so its own allocations fall outside the difference.
fn grown(db: &GrafeoDB, step: impl FnOnce(&GrafeoDB)) -> (usize, usize) {
    let reported = db.memory_usage().total_bytes;
    let before = held();
    step(db);
    let after = held();
    let now = db.memory_usage().total_bytes;
    (after.saturating_sub(before), now.saturating_sub(reported))
}

/// What the allocator and the report may differ by over a phase, whatever its size: the few
/// kilobytes a store allocates once (a dictionary's first room, the first row group's buffers).
const MARGIN: usize = 64 * 1024;

fn create_nodes(db: &GrafeoDB, count: usize) {
    for _ in 0..count {
        db.create_node(&["Person"]).unwrap();
    }
}

/// The node phase is counted as the allocator holds it: at 10,000, 100,000 and 400,000 nodes the
/// two differ by less than the fixed margin, and the difference does not grow with the nodes.
#[test]
fn the_node_phase_is_counted_as_the_allocator_holds_it() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut gaps = Vec::new();
    for nodes in [10_000, 100_000, 400_000] {
        let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
        let (allocated, reported) = grown(&db, |db| create_nodes(db, nodes));
        assert!(
            allocated.abs_diff(reported) <= MARGIN,
            "{nodes} nodes: {allocated} B allocated, {reported} B reported"
        );
        assert!(reported > 0, "{nodes} nodes are reported");
        gaps.push(allocated.abs_diff(reported));
    }
    assert!(
        gaps[2].abs_diff(gaps[0]) <= MARGIN,
        "the difference does not grow with the nodes: {gaps:?}"
    );
}

/// A load of nodes, then their properties and edges, then the deletion of most of them.
fn mixed(db: &GrafeoDB, nodes: usize) -> Vec<grafeo_common::types::NodeId> {
    let ids: Vec<_> = (0..nodes)
        .map(|i| {
            db.create_node_with_props(
                &["Person"],
                [
                    (
                        "id",
                        grafeo_common::types::Value::Int64(i64::try_from(i).unwrap()),
                    ),
                    ("city", ["Amsterdam", "Prague", "Lyon"][i % 3].into()),
                ],
            )
            .unwrap()
        })
        .collect();
    for i in 0..nodes {
        db.create_edge(ids[i], ids[(i * 7 + 1) % nodes], "KNOWS")
            .unwrap();
    }
    ids
}

fn assert_counted_is_walked(db: &GrafeoDB, what: &str) {
    let store = db.store().unwrap();
    let counted = format!("{:?}", store.counted_breakdown());
    let walked = format!("{:?}", store.memory_breakdown());
    assert_eq!(counted, walked, "{what}: the account equals the walk");
}

/// The account equals its walk, line for line, after a node-only load, a mixed one, and one that
/// deletes most of what it wrote.
#[test]
fn the_account_equals_the_walk_after_node_only_mixed_and_delete_heavy_loads() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
    create_nodes(&db, 50_000);
    assert_counted_is_walked(&db, "node only");

    let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
    let ids = mixed(&db, 20_000);
    assert_counted_is_walked(&db, "mixed");

    for (i, id) in ids.iter().enumerate() {
        if i % 4 != 0 {
            db.detach_delete_node(*id).unwrap();
        }
    }
    db.gc();
    assert_counted_is_walked(&db, "three quarters deleted");
}

/// A node-only load under a small memory limit is refused at the limit, with `GRAFEO-S001` naming
/// it, and the report stays within it: nodes are written against the account like anything else.
/// The nodes come one at a time through the direct API, so no statement's own memory takes the
/// limit first; the limit is half a mebibyte above what an empty database reports.
#[test]
fn a_node_only_load_is_refused_at_a_small_limit() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let empty = GrafeoDB::with_config(Config::in_memory())
        .unwrap()
        .memory_usage()
        .total_bytes;
    let limit = empty + 512 * 1024;
    let db = GrafeoDB::with_config(Config::in_memory().with_memory_limit(limit)).unwrap();
    let mut written = 0_usize;
    let error = loop {
        match db.create_node(&["Person"]) {
            Ok(_) => written += 1,
            Err(error) => break error,
        }
        assert!(written < 50_000_000, "the limit never refused a node");
    };
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{error}");
    assert!(error.to_string().contains("memory limit"), "{error}");
    assert!(
        written >= 100_000,
        "{written} nodes before the refusal: the nodes, not a fixed cost, reached the limit"
    );
    let usage = db.memory_usage();
    assert!(
        usage.total_bytes <= limit,
        "{} B reported under a {limit} B limit",
        usage.total_bytes
    );
}
