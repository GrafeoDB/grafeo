//! The change log in the memory account, as the allocator holds it (#679 patch 5).
//!
//! The log's count and its walk (`account_mismatches`) share their figures, so a test of one
//! against the other cannot see a part both leave out. The oracle here is the allocator: this
//! binary counts every byte the process allocates and frees.
//!
//! - **While full,** the log's count follows what its events, their property maps, the entities
//!   table and the commit queue hold: the two differ by a fixed margin, not by an amount per event.
//! - **Emptied,** by a prune or by an eviction, the log holds about nothing and counts about
//!   nothing: the entities table is given back, not kept at its peak while the count reads zero.
//! - **A property map with a removal,** as a transaction's create leaves it when a later write of
//!   the transaction removes one of its properties (the fold into the create), is counted by its
//!   buckets: a removal that leaves a tombstone lowers the map's capacity, not its allocation.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test cdc_log_in_the_account
//! ```

#![cfg(feature = "cdc")]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this one forwards every call to System"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::memory::buffer::MemoryConsumer;
use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::cdc::CdcLog;

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

/// What the allocator and the count may differ by, whatever the log's size: the test's own
/// few allocations and the log's fixed ones.
const MARGIN: usize = 64 * 1024;

/// Records `count` creates of nodes from `first`, one commit each, each with a property map of
/// three entries.
fn fill(log: &CdcLog, first: u64, count: u64) {
    for id in first..first + count {
        let props = HashMap::from([
            ("a".to_string(), Value::Int64(1)),
            ("b".to_string(), Value::Int64(2)),
            ("c".to_string(), Value::Int64(3)),
        ]);
        log.record_create_node(NodeId::new(id), EpochId::new(id), Some(props), None);
    }
}

/// Records `count` creates of nodes from `first`, one commit each, each with a property map of
/// 28 entries (a 32-bucket table at its capacity) from which ten were then removed. A removal
/// leaves a tombstone only where no empty bucket is near, so one removal would leave the
/// capacity as it was for most maps; ten lower it for most.
fn fill_removed(log: &CdcLog, first: u64, count: u64) {
    for id in first..first + count {
        let mut props: HashMap<String, Value> = (0..28)
            .map(|key| (format!("p{key:02}"), Value::Int64(key)))
            .collect();
        for key in 0..10 {
            props.remove(&format!("p{key:02}"));
        }
        log.record_create_node(NodeId::new(id), EpochId::new(id), Some(props), None);
    }
}

fn counted(log: &CdcLog) -> usize {
    log.counted_memory_bytes().0
}

/// 100,000 events: while full, the allocator and the count differ by the margin, and emptied by a
/// prune or an eviction, both read about nothing.
#[test]
fn the_log_is_counted_as_the_allocator_holds_it() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let log = CdcLog::new();
    for evict in [false, true] {
        let base = held();
        fill(&log, 1, 100_000);
        let (allocated, count) = (held().saturating_sub(base), counted(&log));
        assert!(
            allocated.abs_diff(count) <= MARGIN,
            "full (eviction: {evict}): {allocated} B allocated, {count} B counted"
        );
        if evict {
            log.evict(usize::MAX);
        } else {
            log.prune_before(EpochId::new(u64::MAX));
        }
        assert_eq!(log.event_count(), 0);
        let (allocated, count) = (held().saturating_sub(base), counted(&log));
        assert!(
            allocated <= MARGIN && count <= MARGIN,
            "emptied (eviction: {evict}): {allocated} B still allocated, {count} B counted"
        );
    }
}

/// 10,000 events whose property maps lost an entry after they were built: the allocator and the
/// count differ by the margin, as for maps built by inserts alone.
#[test]
fn a_map_with_a_removal_is_counted_by_its_buckets() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let log = CdcLog::new();
    let base = held();
    fill_removed(&log, 1, 10_000);
    let (allocated, count) = (held().saturating_sub(base), counted(&log));
    assert!(
        allocated.abs_diff(count) <= MARGIN,
        "{allocated} B allocated, {count} B counted"
    );
    assert_eq!(log.account_mismatches(), Vec::new());
}
