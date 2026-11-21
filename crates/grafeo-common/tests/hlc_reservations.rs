//! The hybrid logical clock gives each commit one run of timestamps (#679): `reserve(n)` returns
//! the first of `n` consecutive raw values, each above every timestamp the clock gave before, and
//! `plus` steps through them, carrying past the logical bits into the milliseconds. `now()` is a
//! reservation of one, so more than 65,535 timestamps in one millisecond never repeat (0.6.1 kept
//! the logical counter at 65,535 and repeated it). `starting_after` starts a clock past a floor,
//! as an open starts past the largest timestamp it replayed.
//!
//! ```bash
//! cargo test -p grafeo-common --all-features --test main hlc_reservations::
//! ```

use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use grafeo_common::types::{HlcClock, HlcTimestamp};

fn wall_ms() -> u64 {
    u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
}

fn count(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// A floor an hour ahead of the wall clock: a clock started after it sees the wall clock behind,
/// so every timestamp it gives falls in the same millisecond until the logical bits carry.
fn ahead() -> HlcTimestamp {
    HlcTimestamp::new(wall_ms() + 3_600_000, 0)
}

#[test]
fn plus_steps_through_the_raw_encoding_and_carries() {
    let t = HlcTimestamp::new(5, 65_535);
    assert_eq!(t.plus(0), t);
    assert_eq!(t.plus(1), HlcTimestamp::new(6, 0));
    assert_eq!(
        HlcTimestamp::new(5, 0).plus(65_536),
        HlcTimestamp::new(6, 0)
    );
    assert_eq!(
        HlcTimestamp::new(5, 3).plus(7).as_u64(),
        HlcTimestamp::new(5, 3).as_u64() + 7
    );
}

#[test]
fn a_reservation_is_a_run_above_everything_before_and_after() {
    let clock = HlcClock::new();
    let before = clock.now();
    let first = clock.reserve(count(5));
    assert!(first > before, "{first:?} after {before:?}");
    let after = clock.now();
    assert!(
        after > first.plus(4),
        "{after:?} after the run from {first:?}"
    );
}

#[test]
fn seventy_thousand_timestamps_in_one_millisecond_never_repeat() {
    let floor = ahead();
    let clock = HlcClock::starting_after(floor);
    let mut last = clock.now();
    for _ in 0..70_000 {
        let next = clock.now();
        assert!(next > last, "{next:?} after {last:?}");
        last = next;
    }
    assert!(
        last.physical_ms() > floor.physical_ms(),
        "70,000 timestamps past 65,535 carry into the milliseconds: {last:?}"
    );

    let clock = HlcClock::starting_after(ahead());
    let first = clock.reserve(count(70_000));
    let run: Vec<HlcTimestamp> = (0..70_000).map(|i| first.plus(i)).collect();
    assert!(run.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(clock.now() > first.plus(69_999));
}

#[test]
fn reservations_from_eight_threads_are_disjoint_and_rising() {
    let clock = Arc::new(HlcClock::starting_after(ahead()));
    let runs: Vec<Vec<(u64, u64)>> = (0..8)
        .map(|thread| {
            let clock = Arc::clone(&clock);
            std::thread::spawn(move || {
                (0..2_000_u64)
                    .map(|i| {
                        let n = 1 + (i + thread) % 7;
                        (clock.reserve(count(n)).as_u64(), n)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .map(|handle| handle.join().unwrap())
        .collect();
    for run in &runs {
        assert!(
            run.windows(2)
                .all(|pair| pair[0].0 + pair[0].1 <= pair[1].0),
            "one thread's reservations rise and do not overlap"
        );
    }
    let mut all: Vec<(u64, u64)> = runs.into_iter().flatten().collect();
    all.sort_unstable();
    for pair in all.windows(2) {
        let ((first, n), (next, _)) = (pair[0], pair[1]);
        assert!(
            first + n <= next,
            "the run from {first} of {n} overlaps the one from {next}"
        );
    }
}

#[test]
fn a_clock_started_after_a_floor_gives_only_later_timestamps() {
    for floor in [ahead(), HlcTimestamp::new(wall_ms() + 3_600_000, 65_535)] {
        let clock = HlcClock::starting_after(floor);
        assert!(clock.now() > floor, "now after {floor:?}");
        let clock = HlcClock::starting_after(floor);
        assert!(
            clock.reserve(count(3)) > floor,
            "a reservation after {floor:?}"
        );
    }
    // A floor behind the wall clock leaves the wall clock in charge.
    let past = HlcTimestamp::new(wall_ms() - 3_600_000, 9);
    let clock = HlcClock::starting_after(past);
    assert!(clock.now().physical_ms() + 1_000 >= wall_ms());
}
