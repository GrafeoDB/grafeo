//! The history's retention window, time index and floor, on a clock the
//! tests move.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use grafeo_common::types::{EpochId, Timestamp};
use grafeo_common::utils::error::Error;

use super::{History, HistoryImage};
use crate::config::HistoryRetention;

/// A history on a clock at `start` milliseconds, and the clock.
fn history_at(start: u64) -> (History, Arc<AtomicU64>) {
    let now = Arc::new(AtomicU64::new(start));
    let clock = Arc::clone(&now);
    (
        History::with_clock(Arc::new(move || clock.load(Ordering::SeqCst))),
        now,
    )
}

/// Commits epochs `epochs`, one each `step` milliseconds from the clock.
fn commit(history: &History, now: &AtomicU64, epochs: std::ops::RangeInclusive<u64>, step: u64) {
    for epoch in epochs {
        now.fetch_add(step, Ordering::SeqCst);
        let time = history.commit_time().expect("history is on");
        history.record(EpochId::new(epoch), time);
    }
}

fn not_retained(error: Error) -> String {
    let text = error.to_string();
    assert!(
        text.contains("is not retained"),
        "{text:?} says history is not retained"
    );
    text
}

/// Off, a commit takes no time and records nothing, and there is no
/// retention horizon: the read horizon decides alone.
#[test]
fn off_takes_no_time_and_records_nothing() {
    let (history, _) = history_at(1_000);
    assert_eq!(history.commit_time(), None);
    history.record(EpochId::new(1), 1_000);
    assert_eq!(history.times(), []);
    assert_eq!(history.retention_horizon(), None);
    assert_eq!(history.image(), None);
    assert!(history.epoch_at(Timestamp::from_millis(1_000)).is_err());
}

/// "At T" is the last epoch committed at or before T, at the resolution of
/// a millisecond; a time before the index starts is not retained.
#[test]
fn a_time_reads_at_the_last_epoch_committed_at_or_before_it() {
    let (history, now) = history_at(10_000);
    history.set_retention(HistoryRetention::Forever, 0);
    commit(&history, &now, 1..=4, 100);
    assert_eq!(
        history.times(),
        [
            (0, 10_000),
            (1, 10_100),
            (2, 10_200),
            (3, 10_300),
            (4, 10_400)
        ]
    );
    let at = |millis: i64| history.epoch_at(Timestamp::from_millis(millis));
    assert_eq!(at(10_000).unwrap(), EpochId::new(0), "the anchor");
    assert_eq!(at(10_199).unwrap(), EpochId::new(1));
    assert_eq!(at(10_200).unwrap(), EpochId::new(2), "a commit at T is in");
    assert_eq!(
        at(99_999).unwrap(),
        EpochId::new(4),
        "after the last: the last"
    );
    not_retained(at(9_999).unwrap_err());
}

/// A clock that goes back repeats the last time, so the times follow the
/// epochs and a lookup stays monotone.
#[test]
fn a_clock_that_goes_back_repeats_the_last_time() {
    let (history, now) = history_at(5_000);
    history.set_retention(HistoryRetention::Forever, 0);
    commit(&history, &now, 1..=1, 100);
    now.store(4_000, Ordering::SeqCst);
    let time = history.commit_time().unwrap();
    assert_eq!(time, 5_100, "the last time, not the clock's");
    history.record(EpochId::new(2), time);
    assert_eq!(history.times()[1..], [(1, 5_100), (2, 5_100)]);
    assert_eq!(
        history.epoch_at(Timestamp::from_millis(5_100)).unwrap(),
        EpochId::new(2)
    );
}

/// A window starts at the last epoch committed at or before now minus the
/// window; one that reaches back past every commit starts at the floor, and
/// `Forever` stays at the floor.
#[test]
fn the_retention_horizon_follows_the_window() {
    let (history, now) = history_at(0);
    history.set_retention(HistoryRetention::Duration(Duration::from_millis(250)), 0);
    commit(&history, &now, 1..=8, 100);
    // Now 800: the window reaches back to 550, after epoch 5 at 500.
    assert_eq!(history.retention_horizon(), Some(5));
    now.store(1_000, Ordering::SeqCst);
    assert_eq!(history.retention_horizon(), Some(7), "the window moves on");
    now.store(100_000, Ordering::SeqCst);
    assert_eq!(history.retention_horizon(), Some(8), "past every commit");

    let (history, now) = history_at(0);
    history.set_retention(HistoryRetention::Duration(Duration::from_secs(3_600)), 0);
    commit(&history, &now, 1..=3, 100);
    assert_eq!(
        history.retention_horizon(),
        Some(0),
        "the window holds them all"
    );
    history.collected_at(EpochId::new(2));
    assert_eq!(
        history.retention_horizon(),
        Some(2),
        "never below the floor"
    );

    let (history, _) = history_at(0);
    history.set_retention(HistoryRetention::Forever, 0);
    history.collected_at(EpochId::new(3));
    assert_eq!(history.retention_horizon(), Some(3));
}

/// A collect moves the floor, which never goes back, and drops the times
/// before the newest at or before it; a time whose epoch lies before the
/// floor is refused.
#[test]
fn a_collect_moves_the_floor_and_drops_the_times_before_it() {
    let (history, now) = history_at(0);
    history.set_retention(HistoryRetention::Forever, 0);
    commit(&history, &now, 1..=5, 100);
    history.collected_at(EpochId::new(3));
    assert_eq!(history.floor(), EpochId::new(3));
    assert_eq!(history.times(), [(3, 300), (4, 400), (5, 500)]);
    history.collected_at(EpochId::new(1));
    assert_eq!(
        history.floor(),
        EpochId::new(3),
        "the floor never goes back"
    );
    assert_eq!(
        history.epoch_at(Timestamp::from_millis(350)).unwrap(),
        EpochId::new(3)
    );
    not_retained(history.epoch_at(Timestamp::from_millis(250)).unwrap_err());
    let text = history.before_floor(EpochId::new(2)).to_string();
    assert!(
        text.contains("history before epoch 3 is not retained"),
        "{text}"
    );
}

/// Turned on later, the history starts at the published epoch, now; turned
/// off, the index goes; an image restores the floor and the times.
#[test]
fn turned_on_later_history_starts_at_the_published_epoch() {
    let (history, now) = history_at(7_000);
    history.collected_at(EpochId::new(19));
    history.set_retention(HistoryRetention::Forever, 19);
    assert_eq!(history.times(), [(19, 7_000)]);
    commit(&history, &now, 20..=21, 10);
    let image = history.image().unwrap();
    assert_eq!(
        image,
        HistoryImage {
            floor: 19,
            times: vec![(19, 7_000), (20, 7_010), (21, 7_020)],
        }
    );
    history.set_retention(HistoryRetention::Off, 21);
    assert_eq!((history.times(), history.image()), (Vec::new(), None));

    let (restored, _) = history_at(9_000);
    restored.set_retention(HistoryRetention::Forever, 0);
    restored.restore(image.clone());
    assert_eq!(restored.image(), Some(image));
    assert_eq!(
        restored.next_time(),
        9_000,
        "a restored history goes on from the clock"
    );
}

/// A manager on a clock the test moves, keeping history for 250 ms, with
/// eight commits 100 ms apart (at 100 to 800 ms).
fn manager_with_commits() -> (super::super::TransactionManager, Arc<AtomicU64>) {
    let now = Arc::new(AtomicU64::new(0));
    let clock = Arc::clone(&now);
    let manager = super::super::TransactionManager::with_history_clock(Arc::new(move || {
        clock.load(Ordering::SeqCst)
    }));
    manager.set_history_retention(HistoryRetention::Duration(Duration::from_millis(250)));
    for _ in 0..8 {
        now.fetch_add(100, Ordering::SeqCst);
        let transaction = manager.begin();
        manager
            .start_commit(transaction)
            .unwrap()
            .complete()
            .unwrap();
    }
    (manager, now)
}

/// The collect horizon is the older of the read horizon and the start of
/// the window: the last epoch committed at or before now minus the window.
/// A read at the horizon is pinned, one before it refused; a pinned read
/// and an open transaction hold the horizon back as the window moves on.
#[test]
fn the_collect_horizon_is_the_older_of_the_readers_and_the_window() {
    let (manager, now) = manager_with_commits();
    assert_eq!(manager.current_epoch(), EpochId::new(8));
    // Now 800: the window reaches back to 550, after epoch 5 at 500.
    assert_eq!(manager.collect_horizon(), EpochId::new(5));
    assert_eq!(manager.history_floor(), EpochId::new(5));
    assert_eq!(
        manager.read_horizon(),
        EpochId::new(8),
        "no reader is older"
    );
    let pinned = manager.pin_read_at(EpochId::new(5)).unwrap();
    not_retained(manager.pin_read_at(EpochId::new(4)).unwrap_err());
    let transaction = manager.begin();

    now.store(10_000, Ordering::SeqCst);
    assert_eq!(
        manager.collect_horizon(),
        EpochId::new(5),
        "the pinned read"
    );
    drop(pinned);
    assert_eq!(
        manager.collect_horizon(),
        EpochId::new(8),
        "the open snapshot"
    );
    manager.abort(transaction).unwrap();
    assert_eq!(manager.collect_horizon(), EpochId::new(8));
    let refused = not_retained(manager.epoch_at(Timestamp::from_millis(650)).unwrap_err());
    assert!(
        refused.contains("history before 1970-01-01T00:00:00.800000Z is not retained"),
        "a time whose epoch the window let go: {refused}"
    );
    assert_eq!(
        manager.epoch_at(Timestamp::from_millis(10_000)).unwrap(),
        EpochId::new(8)
    );
}

/// Off, the collect horizon is the read horizon, as without history, and a
/// commit records no time.
#[test]
fn off_the_collect_horizon_is_the_read_horizon() {
    let manager = super::super::TransactionManager::new();
    let transaction = manager.begin();
    manager
        .start_commit(transaction)
        .unwrap()
        .complete()
        .unwrap();
    let reader = manager.begin();
    let transaction = manager.begin();
    manager
        .start_commit(transaction)
        .unwrap()
        .complete()
        .unwrap();
    assert_eq!(manager.collect_horizon(), manager.read_horizon());
    assert_eq!(
        manager.collect_horizon(),
        EpochId::new(1),
        "the open snapshot"
    );
    assert_eq!(manager.history_image(), None);
    manager.abort(reader).unwrap();
    assert_eq!(manager.collect_horizon(), EpochId::new(2));
}
