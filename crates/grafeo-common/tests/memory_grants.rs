//! Memory grants and the refusals of the database's bounds (#751): a grant grows and shrinks
//! with the manager's reservation, its region's count and its own size moving together; growth
//! never runs eviction, so it is safe under any lock, and past the hard limit it is refused with
//! nothing changed; the hard limit is the budget itself (a reservation is refused only past
//! 100% of the memory limit). Eviction runs only from `make_room` (and `try_allocate`), and skips
//! a consumer that cannot evict. Forced reservations pass the limit (replay, load, commit). A
//! refusal carries the figures of its moment and is `GRAFEO-S001`, its message naming the bound,
//! the bytes and the setting that moves it.
//!
//! ```bash
//! cargo test -p grafeo-common --all-features --test main memory_grants::
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use grafeo_common::memory::buffer::{
    BufferManager, ConsumerStats, GRANT_STEP, MemoryConsumer, MemoryGrant, MemoryRegion,
    StorageTier,
};
use grafeo_common::utils::error::{Error, LimitBound, LimitRefusal};

const MIB: usize = 1024 * 1024;

/// The manager's reservation, the region's count and the grant's size, at once.
fn figures(manager: &BufferManager, grant: &MemoryGrant) -> (usize, usize, usize) {
    let region = manager.stats().region_allocated[grant.region().index()];
    (manager.allocated(), region, grant.size())
}

#[test]
fn the_hard_limit_is_the_budget() {
    for budget in [MIB, 64 * MIB, 768 * MIB] {
        let manager = BufferManager::with_budget(budget);
        assert_eq!(manager.hard_limit(), budget, "refused only past 100%");
    }
}

#[test]
fn grow_and_shrink_move_the_reservation_the_region_and_the_size_together() {
    let manager = BufferManager::with_budget(16 * MIB);
    let mut grant = manager.empty_grant(MemoryRegion::ExecutionBuffers);
    assert_eq!(figures(&manager, &grant), (0, 0, 0));

    assert!(grant.try_grow(GRANT_STEP));
    assert_eq!(
        figures(&manager, &grant),
        (GRANT_STEP, GRANT_STEP, GRANT_STEP)
    );
    assert!(grant.try_grow(3 * GRANT_STEP));
    assert_eq!(
        figures(&manager, &grant),
        (4 * GRANT_STEP, 4 * GRANT_STEP, 4 * GRANT_STEP)
    );

    grant.shrink(GRANT_STEP);
    assert_eq!(
        figures(&manager, &grant),
        (3 * GRANT_STEP, 3 * GRANT_STEP, 3 * GRANT_STEP)
    );
    // At most its size is returned.
    grant.shrink(100 * GRANT_STEP);
    assert_eq!(figures(&manager, &grant), (0, 0, 0));

    assert!(grant.try_grow(GRANT_STEP));
    drop(grant);
    assert_eq!(
        manager.allocated(),
        0,
        "a dropped grant returns what it holds"
    );
    assert_eq!(
        manager.stats().region_allocated,
        [0; 4],
        "and its region's count"
    );
}

#[test]
fn a_growth_past_the_hard_limit_is_refused_and_changes_nothing() {
    let manager = BufferManager::with_budget(4 * MIB);
    let mut grant = manager.empty_grant(MemoryRegion::GraphStorage);
    assert!(grant.try_grow(3 * MIB));
    let before = figures(&manager, &grant);
    assert!(!grant.try_grow(MIB + 1), "one byte past the limit");
    assert_eq!(
        figures(&manager, &grant),
        before,
        "a refusal changes nothing"
    );
    assert!(grant.try_grow(MIB), "up to the limit itself");
    assert_eq!(manager.allocated(), 4 * MIB);
    assert!(!grant.try_grow(1));
    // Another grant meets the same limit.
    let mut other = manager.empty_grant(MemoryRegion::IndexBuffers);
    assert!(!other.try_grow(1));
    assert_eq!(figures(&manager, &other), (4 * MIB, 0, 0));
}

/// A consumer that holds a grant of the manager and counts the calls made on it.
struct Holder {
    name: String,
    held: Mutex<Option<MemoryGrant>>,
    evictable: bool,
    evictions: AtomicUsize,
}

impl Holder {
    fn new(manager: &Arc<BufferManager>, name: &str, bytes: usize, evictable: bool) -> Arc<Self> {
        let mut grant = manager.empty_grant(MemoryRegion::GraphStorage);
        grant.grow_forced(bytes);
        Arc::new(Self {
            name: name.to_string(),
            held: Mutex::new(Some(grant)),
            evictable,
            evictions: AtomicUsize::new(0),
        })
    }

    fn evictions(&self) -> usize {
        self.evictions.load(Ordering::SeqCst)
    }
}

impl MemoryConsumer for Holder {
    fn name(&self) -> &str {
        &self.name
    }

    fn memory_usage(&self) -> usize {
        self.held
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, MemoryGrant::size)
    }

    fn eviction_priority(&self) -> u8 {
        10
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::GraphStorage
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        self.evictions.fetch_add(1, Ordering::SeqCst);
        if !self.evictable {
            return 0;
        }
        // Dropping the grant returns its bytes to the manager.
        self.held
            .lock()
            .unwrap()
            .take()
            .map_or(0, |grant| grant.size())
    }

    fn can_evict(&self) -> bool {
        self.evictable
    }

    fn current_tier(&self) -> StorageTier {
        if self.memory_usage() > 0 {
            StorageTier::InMemory
        } else {
            StorageTier::Uninitialized
        }
    }
}

/// A consumer registered with its manager for one test, unregistered when it drops. A test
/// consumer holds a grant of the manager, whose releaser is the manager itself, and the manager
/// holds the registered consumer: without the unregistration each keeps the other alive.
struct Registered {
    manager: Arc<BufferManager>,
    name: String,
}

impl Registered {
    fn new(manager: &Arc<BufferManager>, consumer: Arc<dyn MemoryConsumer>) -> Self {
        let name = consumer.name().to_string();
        manager.register_consumer(consumer);
        Self {
            manager: Arc::clone(manager),
            name,
        }
    }
}

impl Drop for Registered {
    fn drop(&mut self) {
        self.manager.unregister_consumer(&self.name);
    }
}

/// Growth runs no eviction: not past the old 95% line, not past the limit, whatever the
/// consumers hold.
#[test]
fn growth_never_runs_eviction() {
    let manager = BufferManager::with_budget(8 * MIB);
    let holder = Holder::new(&manager, "cache", 6 * MIB, true);
    let _registered = Registered::new(&manager, holder.clone());
    let mut grant = manager.empty_grant(MemoryRegion::ExecutionBuffers);
    assert!(
        grant.try_grow(MIB + MIB / 2),
        "to 7.5 MiB, past the eviction lines"
    );
    assert!(!grant.try_grow(MIB), "past the limit: refused");
    assert_eq!(holder.evictions(), 0, "try_grow never evicts");
    assert_eq!(holder.memory_usage(), 6 * MIB, "the cache was not touched");
}

#[test]
fn make_room_evicts_until_the_bytes_fit() {
    let manager = BufferManager::with_budget(8 * MIB);
    let holder = Holder::new(&manager, "cache", 7 * MIB, true);
    let _registered = Registered::new(&manager, holder.clone());
    let mut grant = manager.empty_grant(MemoryRegion::ExecutionBuffers);
    assert!(!grant.try_grow(2 * MIB));
    assert!(
        manager.make_room(2 * MIB),
        "evicting the cache frees enough"
    );
    assert!(holder.evictions() >= 1);
    assert!(grant.try_grow(2 * MIB), "the bytes fit now");
    assert_eq!(manager.allocated(), 2 * MIB);
}

#[test]
fn make_room_says_when_nothing_frees_enough() {
    let manager = BufferManager::with_budget(8 * MIB);
    let pinned = Holder::new(&manager, "pinned", 7 * MIB, false);
    let _registered = Registered::new(&manager, pinned.clone());
    assert!(!manager.make_room(2 * MIB));
    assert_eq!(
        pinned.evictions(),
        0,
        "a consumer that cannot evict is skipped"
    );
    assert_eq!(pinned.memory_usage(), 7 * MIB);
    assert!(
        manager.make_room(MIB),
        "what already fits needs no eviction"
    );
}

#[test]
fn eviction_skips_a_consumer_that_cannot_evict_and_reaches_one_that_can() {
    let manager = BufferManager::with_budget(8 * MIB);
    let pinned = Holder::new(&manager, "pinned", 3 * MIB, false);
    let cache = Holder::new(&manager, "cache", 4 * MIB, true);
    let _pinned = Registered::new(&manager, pinned.clone());
    let _cache = Registered::new(&manager, cache.clone());
    assert!(manager.make_room(3 * MIB));
    assert_eq!(pinned.evictions(), 0);
    assert_eq!(cache.evictions(), 1);
    assert_eq!(manager.allocated(), 3 * MIB);
    let stats = ConsumerStats::from_consumer(pinned.as_ref());
    assert!(!stats.can_evict);
    assert!(ConsumerStats::from_consumer(cache.as_ref()).can_evict);
}

#[test]
fn a_consumer_cannot_evict_unless_it_says_so() {
    struct Plain;
    impl MemoryConsumer for Plain {
        fn name(&self) -> &str {
            "plain"
        }
        fn memory_usage(&self) -> usize {
            0
        }
        fn eviction_priority(&self) -> u8 {
            0
        }
        fn region(&self) -> MemoryRegion {
            MemoryRegion::GraphStorage
        }
        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }
        fn current_tier(&self) -> StorageTier {
            StorageTier::Uninitialized
        }
    }
    assert!(!Plain.can_evict(), "the default");
}

#[test]
fn forced_reservations_pass_the_limit() {
    let manager = BufferManager::with_budget(MIB);
    let chunk = manager.reserve_forced(2 * MIB, MemoryRegion::GraphStorage);
    assert_eq!(figures(&manager, &chunk), (2 * MIB, 2 * MIB, 2 * MIB));
    let mut grant = manager.empty_grant(MemoryRegion::GraphStorage);
    grant.grow_forced(MIB);
    assert_eq!(manager.allocated(), 3 * MIB);
    assert!(!grant.try_grow(1), "over the limit, growth is refused");
    drop(chunk);
    drop(grant);
    assert_eq!(manager.allocated(), 0);
}

/// A thaw's grant merged into the part's: the sizes add, the reservation and the region stay
/// as they were, and the merged grant returns the whole.
#[test]
fn a_thaw_grant_merges_into_the_parts_grant() {
    let manager = BufferManager::with_budget(16 * MIB);
    let mut part = manager.empty_grant(MemoryRegion::GraphStorage);
    assert!(part.try_grow(2 * MIB));
    let thaw = manager.reserve_forced(3 * MIB, MemoryRegion::GraphStorage);
    assert_eq!(manager.allocated(), 5 * MIB);
    part.merge(thaw);
    assert_eq!(figures(&manager, &part), (5 * MIB, 5 * MIB, 5 * MIB));
    part.shrink(MIB);
    assert_eq!(figures(&manager, &part), (4 * MIB, 4 * MIB, 4 * MIB));
    drop(part);
    assert_eq!(manager.allocated(), 0);
}

#[test]
fn the_grant_step_is_one_mebibyte() {
    assert_eq!(GRANT_STEP, MIB);
}

#[test]
fn a_refusal_carries_the_figures_of_its_moment() {
    let manager = BufferManager::with_budget(4 * MIB);
    let mut grant = manager.empty_grant(MemoryRegion::GraphStorage);
    assert!(grant.try_grow(3 * MIB));
    let refusal = manager.refusal(LimitBound::MemoryLimit, 2 * MIB);
    assert_eq!(
        refusal,
        LimitRefusal::Memory {
            bound: LimitBound::MemoryLimit,
            requested: 2 * MIB as u64,
            used: 3 * MIB as u64,
            limit: 4 * MIB as u64,
        }
    );
    assert_eq!(refusal.bound(), LimitBound::MemoryLimit);
    let query = manager.refusal(LimitBound::QueryMemoryLimit, MIB);
    assert_eq!(query.bound(), LimitBound::QueryMemoryLimit);
}

#[test]
fn each_bound_names_itself_and_the_setting_that_moves_it() {
    let bounds = [
        (
            LimitBound::MemoryLimit,
            "memory limit",
            [
                "Config::with_memory_limit",
                "Python memory_limit",
                "Node memoryLimit",
            ],
        ),
        (
            LimitBound::QueryMemoryLimit,
            "query memory limit",
            [
                "Config::with_query_memory_limit",
                "Python query_memory_limit",
                "Node queryMemoryLimit",
            ],
        ),
        (
            LimitBound::SpillLimit,
            "spill limit",
            [
                "Config::with_spill_limit",
                "Python spill_limit",
                "Node spillLimit",
            ],
        ),
    ];
    for (bound, name, settings) in bounds {
        assert_eq!(bound.name(), name);
        for setting in settings {
            assert!(
                bound.setting().contains(setting),
                "{name}: {}",
                bound.setting()
            );
        }
    }
    assert_eq!(LimitBound::CheckpointRoom.name(), "checkpoint room");
    assert!(
        LimitBound::CheckpointRoom
            .setting()
            .contains("Config::with_spill_path"),
        "the setting that moves the spill volume: {}",
        LimitBound::CheckpointRoom.setting()
    );
}

/// The refusal of each bound as the error a statement returns: `GRAFEO-S001`, the refusal kept
/// inside it, and the message's parts: the bytes, the bound and the setting.
#[test]
fn a_refusal_is_grafeo_s001_with_the_bound_the_bytes_and_the_setting() {
    let cases: [(LimitRefusal, &[&str]); 4] = [
        (
            LimitRefusal::Memory {
                bound: LimitBound::MemoryLimit,
                requested: 2_097_152,
                used: 3_145_728,
                limit: 4_194_304,
            },
            &[
                "the statement needs 2097152 more bytes",
                "3145728 bytes of the memory limit of 4194304 are in use",
                "raise it with Config::with_memory_limit (Python memory_limit, Node memoryLimit)",
            ],
        ),
        (
            LimitRefusal::Memory {
                bound: LimitBound::QueryMemoryLimit,
                requested: 10,
                used: 20,
                limit: 30,
            },
            &[
                "the statement needs 10 more bytes",
                "20 bytes of the query memory limit of 30 are in use",
                "Config::with_query_memory_limit",
            ],
        ),
        (
            LimitRefusal::Spill {
                requested: 100,
                used: 900,
                limit: 950,
            },
            &[
                "spilling 100 more bytes would pass the spill limit of 950 bytes (900 spilled)",
                "raise it with Config::with_spill_limit (Python spill_limit, Node spillLimit)",
            ],
        ),
        (
            LimitRefusal::Room {
                requested: 100,
                free: 1000,
                needed: 950,
                volume: PathBuf::from("spill-volume"),
            },
            &[
                "spilling 100 more bytes would leave 900 bytes free on spill-volume",
                "the database's checkpoint room needs 950 bytes for its next checkpoint",
            ],
        ),
    ];
    for (refusal, parts) in cases {
        let error = Error::from(refusal.clone());
        assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{error}");
        assert_eq!(error.limit_refusal(), Some(&refusal));
        let text = error.to_string();
        for part in parts {
            assert!(text.contains(part), "{part:?} in {text:?}");
        }
    }
    assert_eq!(
        LimitRefusal::Spill {
            requested: 1,
            used: 2,
            limit: 3
        }
        .bound(),
        LimitBound::SpillLimit
    );
    assert_eq!(
        LimitRefusal::Room {
            requested: 1,
            free: 2,
            needed: 3,
            volume: PathBuf::new()
        }
        .bound(),
        LimitBound::CheckpointRoom
    );
}

/// A spill refusal travels through `io::Result` as `StorageFull` and comes back out intact.
#[test]
fn a_spill_refusal_travels_through_io_errors() {
    let refusal = LimitRefusal::Spill {
        requested: 100,
        used: 900,
        limit: 950,
    };
    let io = std::io::Error::new(std::io::ErrorKind::StorageFull, refusal.clone());
    assert_eq!(io.kind(), std::io::ErrorKind::StorageFull);
    let inner = io
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<LimitRefusal>());
    assert_eq!(inner, Some(&refusal));
}

/// An error that is not a refusal has none inside it.
#[test]
fn other_errors_carry_no_refusal() {
    let error = Error::Internal("not a bound".to_string());
    assert_eq!(error.limit_refusal(), None);
}

/// A consumer that frees half of what it holds on each call, or (`lies`) says it freed half
/// and keeps it all.
struct Halver {
    held: Mutex<MemoryGrant>,
    lies: bool,
    calls: AtomicUsize,
}

impl Halver {
    fn new(manager: &Arc<BufferManager>, bytes: usize, lies: bool) -> Arc<Self> {
        let mut grant = manager.empty_grant(MemoryRegion::GraphStorage);
        grant.grow_forced(bytes);
        Arc::new(Self {
            held: Mutex::new(grant),
            lies,
            calls: AtomicUsize::new(0),
        })
    }
}

impl MemoryConsumer for Halver {
    fn name(&self) -> &str {
        if self.lies { "liar" } else { "halver" }
    }

    fn memory_usage(&self) -> usize {
        self.held.lock().unwrap().size()
    }

    fn eviction_priority(&self) -> u8 {
        10
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::GraphStorage
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut held = self.held.lock().unwrap();
        let half = held.size() / 2;
        if !self.lies {
            held.shrink(half);
        }
        half
    }

    fn can_evict(&self) -> bool {
        true
    }

    fn current_tier(&self) -> StorageTier {
        StorageTier::InMemory
    }
}

/// `make_room` keeps asking while each pass lowers the reservation: a consumer that frees half
/// per call is called until the bytes fit, however many halvings that takes (six here, from
/// 8 MiB to 128 KiB on a 9 MiB budget).
#[test]
fn make_room_keeps_evicting_while_each_pass_frees_bytes() {
    let manager = BufferManager::with_budget(9 * MIB);
    let halver = Halver::new(&manager, 8 * MIB, false);
    let _registered = Registered::new(&manager, halver.clone());
    let needed = 9 * MIB - 128 * 1024;
    let made = manager.make_room(needed);
    assert!(
        made,
        "six halvings make the room; the consumer was called {} times",
        halver.calls.load(Ordering::SeqCst)
    );
    assert!(
        halver.calls.load(Ordering::SeqCst) >= 6,
        "called {} times",
        halver.calls.load(Ordering::SeqCst)
    );
    assert!(halver.memory_usage() <= 128 * 1024);
    let mut grant = manager.empty_grant(MemoryRegion::ExecutionBuffers);
    assert!(grant.try_grow(needed), "the bytes fit now");
}

/// A consumer that reports freed bytes but never lowers the reservation ends the attempt:
/// `make_room` returns false, and returns at all.
#[test]
fn make_room_ends_when_freed_bytes_do_not_lower_the_reservation() {
    let manager = BufferManager::with_budget(9 * MIB);
    let liar = Halver::new(&manager, 8 * MIB, true);
    let _registered = Registered::new(&manager, liar.clone());
    let (sender, receiver) = std::sync::mpsc::channel();
    let asking = Arc::clone(&manager);
    let asker = std::thread::spawn(move || {
        let _ = sender.send(asking.make_room(4 * MIB));
    });
    let made = receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("make_room returns");
    asker.join().unwrap();
    assert!(!made, "nothing was freed");
    assert_eq!(manager.allocated(), 8 * MIB);
    assert!(liar.calls.load(Ordering::SeqCst) >= 1);
}
