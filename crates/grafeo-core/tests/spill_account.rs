//! The spill account (#751): one open's spilled bytes against its cap and against the room the
//! database keeps on the spill volume. A reservation is counted before the bytes are written
//! and given back when it drops; the free space is probed for each new file and again after
//! every [`SPILL_PROBE_STEP`] reserved, not on every write.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test main spill_account::
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::utils::error::{LimitBound, LimitRefusal};
use grafeo_core::execution::spill::{
    SPILL_PROBE_STEP, SpillAccount, SpillCap, SpillOwner, SpillRoom,
};

const MIB: u64 = 1 << 20;

/// A room whose probe answers `free` (or an error when `None`), counting its calls, and whose
/// database needs `needed` bytes.
fn room(free: Option<u64>, needed: u64) -> (SpillRoom, Arc<AtomicU64>) {
    let probes = Arc::new(AtomicU64::new(0));
    let counted = Arc::clone(&probes);
    let room = SpillRoom {
        volume: PathBuf::from("spill-volume"),
        probe: Box::new(move |_: &Path| {
            counted.fetch_add(1, Ordering::SeqCst);
            free.ok_or_else(|| std::io::Error::other("the volume cannot be read"))
        }),
        needed: Box::new(move || needed),
    };
    (room, probes)
}

/// The probe runs for each new file and once per step reserved since the last probe, never on
/// the writes in between.
#[test]
fn the_free_space_is_probed_per_file_and_per_step() {
    let (room, probes) = room(Some(1 << 40), 0);
    let account = SpillAccount::new(SpillCap::Unlimited, Some(room));
    let probed = || probes.load(Ordering::SeqCst);

    let mut first = account.reserve(SpillOwner::Sort, MIB, true).unwrap();
    assert_eq!(probed(), 1, "a new file probes");
    for _ in 0..(SPILL_PROBE_STEP / MIB - 2) {
        first.grow(MIB).unwrap();
    }
    assert_eq!(probed(), 1, "writes within one step probe nothing");
    first.grow(2 * MIB).unwrap();
    assert_eq!(
        probed(),
        2,
        "a step reserved since the last probe probes again"
    );

    let _second = account.reserve(SpillOwner::Aggregate, MIB, true).unwrap();
    assert_eq!(probed(), 3, "a second file probes");
}

/// The cap refuses a reservation past it with the spill limit's figures and counts nothing; a
/// dropped reservation gives its bytes back, by owner too.
#[test]
fn the_cap_refuses_past_it_and_a_dropped_reservation_gives_back() {
    let account = SpillAccount::new(SpillCap::Bytes(10 * MIB), None);
    let mut sort = account.reserve(SpillOwner::Sort, 4 * MIB, true).unwrap();
    let aggregate = account
        .reserve(SpillOwner::Aggregate, 4 * MIB, true)
        .unwrap();
    assert_eq!(account.used(), 8 * MIB);
    assert_eq!(account.used_by(SpillOwner::Sort), 4 * MIB);
    assert_eq!(account.used_by(SpillOwner::Aggregate), 4 * MIB);

    let refusal = sort.grow(3 * MIB).expect_err("past the cap");
    assert_eq!(refusal.bound(), LimitBound::SpillLimit);
    assert!(
        matches!(
            refusal,
            LimitRefusal::Spill { requested, used, limit }
                if requested == 3 * MIB && used == 8 * MIB && limit == 10 * MIB
        ),
        "{refusal:?}"
    );
    assert_eq!(account.used(), 8 * MIB, "a refusal counts nothing");
    let new_file = account
        .reserve(SpillOwner::VectorColumns, 3 * MIB, true)
        .expect_err("a new file past the cap");
    assert_eq!(new_file.bound(), LimitBound::SpillLimit);

    sort.trim_to(MIB);
    assert_eq!(
        account.used_by(SpillOwner::Sort),
        MIB,
        "a trim gives back the rest"
    );
    drop(sort);
    drop(aggregate);
    assert_eq!(account.used(), 0);
    assert_eq!(account.used_by(SpillOwner::Sort), 0);
    assert_eq!(account.used_by(SpillOwner::Aggregate), 0);
}

/// A cap of zero spills nothing.
#[test]
fn a_zero_cap_refuses_every_reservation() {
    let account = SpillAccount::new(SpillCap::Bytes(0), None);
    let refusal = account
        .reserve(SpillOwner::Sort, 1, true)
        .expect_err("nothing spills");
    assert_eq!(refusal.bound(), LimitBound::SpillLimit);
    assert_eq!(account.used(), 0);
}

/// A reservation that would leave less free space than the database needs for its next
/// checkpoint is refused with the room's figures, whatever the cap.
#[test]
fn the_room_refuses_a_spill_that_leaves_too_little_free() {
    let (room, _) = room(Some(100 * MIB), 90 * MIB);
    let account = SpillAccount::new(SpillCap::Unlimited, Some(room));
    let mut fits = account.reserve(SpillOwner::Sort, 8 * MIB, true).unwrap();
    let refusal = fits.grow(4 * MIB).expect_err("past the room");
    assert_eq!(refusal.bound(), LimitBound::CheckpointRoom);
    assert!(
        matches!(refusal, LimitRefusal::Room { free, needed, .. } if free <= 100 * MIB && needed == 90 * MIB),
        "{refusal:?}"
    );
    assert_eq!(account.used(), 8 * MIB, "a refusal counts nothing");
}

/// A probe that fails leaves the cap in force: reservations within it pass, one past it is
/// refused, and every later probe that fails changes nothing.
#[test]
fn a_failing_probe_keeps_the_cap() {
    let (room, probes) = room(None, 0);
    let account = SpillAccount::new(SpillCap::Bytes(8 * MIB), Some(room));
    let _first = account.reserve(SpillOwner::Sort, 4 * MIB, true).unwrap();
    let _second = account.reserve(SpillOwner::Sort, 4 * MIB, true).unwrap();
    assert!(probes.load(Ordering::SeqCst) >= 1, "the probe was asked");
    let refusal = account
        .reserve(SpillOwner::Sort, MIB, true)
        .expect_err("past the cap");
    assert_eq!(refusal.bound(), LimitBound::SpillLimit);
}
