//! A spilled sort's finish holds no more than the query memory limit and a
//! chunk or so beyond what the same rows hold without a sort: the merged rows
//! stream through the operators after it (a limit, a filter, an aggregate),
//! and nothing keeps them all. Measured with a counting global allocator: the
//! statement's own count cannot show rows that something outside it holds.
//!
//! Each plan here spills: without a spill path it refuses at the limit. The
//! limit is a top-k whose 10,000 rows outgrow the statement's memory, so it
//! goes on as the spilling sort and a limit; the others take no top-k (a filter
//! or an aggregate follows the sort).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test streamed_finalize_peak
//! ```

#![cfg(all(feature = "gql", feature = "spill"))]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this only forwards to System"
)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CURRENT: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    /// Forwards every call to [`System`] and counts what it hands out.
    pub struct Counting;

    fn grew(by: usize) {
        let now = CURRENT.fetch_add(by, Ordering::Relaxed).wrapping_add(by);
        PEAK.fetch_max(now, Ordering::Relaxed);
    }

    fn shrank(by: usize) {
        CURRENT.fetch_sub(by, Ordering::Relaxed);
    }

    // SAFETY: every method passes its arguments unchanged to `System`, which
    // meets the trait's contract, and returns what `System` returned. The
    // counting touches atomics only and allocates nothing.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller meets `alloc`'s contract, which is `System`'s.
            let block = unsafe { System.alloc(layout) };
            if !block.is_null() {
                grew(layout.size());
            }
            block
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: as in `alloc`.
            let block = unsafe { System.alloc_zeroed(layout) };
            if !block.is_null() {
                grew(layout.size());
            }
            block
        }

        unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
            // SAFETY: `block` was handed out by this allocator, so by
            // `System`, with `layout`.
            unsafe { System.dealloc(block, layout) };
            shrank(layout.size());
        }

        unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: as in `dealloc`, and the caller meets `realloc`'s
            // contract for `new_size`.
            let moved = unsafe { System.realloc(block, layout, new_size) };
            if !moved.is_null() {
                if new_size >= layout.size() {
                    grew(new_size - layout.size());
                } else {
                    shrank(layout.size() - new_size);
                }
            }
            moved
        }
    }

    /// The bytes held now.
    pub fn current() -> usize {
        CURRENT.load(Ordering::Relaxed)
    }

    /// The most bytes held since the last [`reset`].
    pub fn peak() -> usize {
        PEAK.load(Ordering::Relaxed)
    }

    /// Starts a measurement from what is held now.
    pub fn reset() {
        PEAK.store(current(), Ordering::Relaxed);
    }
}

#[global_allocator]
static COUNTING: counting::Counting = counting::Counting;

/// The counters are global: the tests take turns.
static MEASURING: Mutex<()> = Mutex::new(());

fn measuring() -> MutexGuard<'static, ()> {
    MEASURING.lock().unwrap_or_else(PoisonError::into_inner)
}

const MIB: usize = 1024 * 1024;
const PEOPLE: i64 = 40_000;

/// People with a 256-byte biography each: the merged rows hold about 10 MiB
/// of their own (a merged row is read back, so its strings are new).
fn people(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    let cities = ["Amsterdam", "Berlin", "Paris", "Prague", "Utrecht"];
    for i in 0..PEOPLE {
        let city = cities[usize::try_from(i % 5).unwrap()];
        db.create_node_with_props(
            &["Person"],
            [
                ("bio", Value::from(format!("{i:05} {}", "x".repeat(250)))),
                ("age", Value::Int64(i * 7919 % PEOPLE)),
                ("city", Value::from(city)),
            ],
        )
        .unwrap();
    }
    db
}

/// The most `query` held beyond what was held before it, and its rows.
fn peak_of(db: &GrafeoDB, query: &str) -> (usize, Vec<Vec<Value>>) {
    let before = counting::current();
    counting::reset();
    let rows = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec();
    (counting::peak().saturating_sub(before), rows)
}

/// The shapes of a spilled sort followed by a streaming operator, each with
/// the same scan and projection without the sort as its baseline: a limit
/// after an augmenting projection, a filter, an aggregate.
const SHAPES: &[(&str, &str)] = &[
    (
        "MATCH (n:Person) RETURN n.bio ORDER BY n.age LIMIT 10000",
        "MATCH (n:Person) RETURN n.bio LIMIT 10000",
    ),
    (
        "MATCH (n:Person) WITH n.bio AS bio, n.age AS age ORDER BY age WHERE age % 1000 = 0 RETURN bio, age",
        "MATCH (n:Person) WITH n.bio AS bio, n.age AS age WHERE age % 1000 = 0 RETURN bio, age",
    ),
    (
        "MATCH (n:Person) WITH n.city AS c, n.bio AS bio, n.age AS a ORDER BY a RETURN c, count(*) AS k",
        "MATCH (n:Person) WITH n.city AS c, n.bio AS bio, n.age AS a RETURN c, count(*) AS k",
    ),
];

/// Within the limit: each shape holds at most the query memory limit, and
/// three times as much again for the chunk in flight, the spill buffers and
/// the estimate's slack, beyond its baseline. The merged rows reach the
/// operator after the sort a batch at a time, while the merge holds its open
/// runs: what no one counts is the one batch handed on (at most 2,048 rows).
#[test]
fn a_spilled_sort_followed_by_a_streaming_operator_holds_about_the_limit() {
    let _turn = measuring();
    let spill = tempfile::tempdir().unwrap();
    let db = people(
        Config::in_memory()
            .with_query_memory_limit(MIB)
            .with_spill_path(spill.path()),
    );
    let without_a_spill_path = people(Config::in_memory().with_query_memory_limit(MIB));
    let mut over = Vec::new();
    for (query, baseline) in SHAPES {
        assert!(
            without_a_spill_path.execute(query).is_err(),
            "{query}: it spills (without a spill path it refuses)"
        );
        let (base, _) = peak_of(&db, baseline);
        let (peak, rows) = peak_of(&db, query);
        assert!(!rows.is_empty(), "{query}");
        if peak > base + 4 * MIB {
            over.push(format!(
                "{query}: held {} KiB beyond what was held before, its baseline {} KiB",
                peak / 1024,
                base / 1024
            ));
        }
    }
    assert!(over.is_empty(), "{over:#?}");
}
