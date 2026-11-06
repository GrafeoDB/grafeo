//! What a checkpoint keeps of the WAL (#673).
//!
//! A checkpoint covers the log up to its position, so the segments below it
//! could go, but a cursor may still need them: the backup cursor marks where
//! the last backup ended, and the next incremental backup copies the log
//! from there. [`retention_floor`] is the one rule that decides where a
//! checkpoint cuts the log: at the lowest position a cursor needs, but never
//! more than the configured limit below the checkpoint
//! ([`Config::with_wal_retention_limit`](crate::Config::with_wal_retention_limit)),
//! so a backup that stopped cannot fill the disk. A backup that is running
//! pins the log it copies ([`WalRetention::pin`]), which the limit does not
//! cut: a checkpoint during the backup never removes a segment it has not
//! copied yet.

use std::sync::atomic::{AtomicBool, Ordering};

/// Where a cursor stands, for [`retention_floor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CursorPosition {
    /// The cursor needs the log from this position on.
    At(u64),
    /// The cursor exists but cannot be read: it may need the whole log, so
    /// it is held to the limit like a cursor at position 0.
    Unreadable,
    /// A running backup copies the log from this position on: kept whole,
    /// whatever the limit, until the backup ends.
    Pinned(u64),
}

/// Where a checkpoint cuts the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RetentionFloor {
    /// The segments that end at or below this position go.
    pub lsn: u64,
    /// A cursor needed the log below `lsn`: the limit cut it off.
    pub limited: bool,
}

/// Where a checkpoint at `checkpoint_lsn` cuts the WAL: at the lowest
/// position one of `cursors` needs, or `checkpoint_lsn` without one, but no
/// more than `limit` bytes of log below `checkpoint_lsn`, except for a
/// [pinned](CursorPosition::Pinned) position, which is kept whole. A log position is
/// a byte offset into the log, so the distance between two positions is the
/// log bytes between them (segment headers aside).
///
/// The log goes by whole segments, and `segment_starts` are the first
/// positions of the WAL's segments in log order. A segment that reaches past
/// the floor stays, which a cursor the limit leaves whole needs. A cursor the
/// limit cuts off needs none of it, so the floor then moves up to the next
/// segment start (never past a pin or the checkpoint): the log kept stays
/// within the limit.
#[must_use]
pub(super) fn retention_floor(
    checkpoint_lsn: u64,
    cursors: impl IntoIterator<Item = CursorPosition>,
    limit: u64,
    segment_starts: &[u64],
) -> RetentionFloor {
    let mut wanted = checkpoint_lsn;
    let mut pinned = checkpoint_lsn;
    for cursor in cursors {
        match cursor {
            CursorPosition::At(lsn) => wanted = wanted.min(lsn),
            CursorPosition::Unreadable => wanted = 0,
            CursorPosition::Pinned(lsn) => pinned = pinned.min(lsn),
        }
    }
    let mut lsn = wanted.max(checkpoint_lsn.saturating_sub(limit)).min(pinned);
    let limited = wanted < lsn;
    if limited
        && let Some(&start) = segment_starts
            .iter()
            .find(|&&start| start >= lsn && start <= pinned)
    {
        lsn = start;
    }
    RetentionFloor { lsn, limited }
}

/// The retention limit of one database, and the warnings it gave: each is
/// given once per open, not at every checkpoint.
#[derive(Debug)]
pub(crate) struct WalRetention {
    /// The WAL kept beyond a checkpoint for cursors at most, in bytes.
    pub limit: u64,
    limited_warned: AtomicBool,
    unreadable_warned: AtomicBool,
    /// The pins of running backups and how far checkpoints removed the log:
    /// one lock, so a pin is never taken between a checkpoint's floor and
    /// its removal.
    state: parking_lot::Mutex<RetentionState>,
}

/// What [`WalRetention`]'s lock guards.
#[derive(Debug, Default)]
struct RetentionState {
    /// The positions running backups copy the log from.
    pins: Vec<u64>,
    /// The highest floor a checkpoint of this open removed the log below.
    removed_below: u64,
}

/// The log a running backup copies, kept by every checkpoint until this is
/// dropped (see [`WalRetention::pin`]).
#[must_use = "the log is kept only while the pin lives"]
pub(crate) struct RetentionPin<'retention> {
    retention: &'retention WalRetention,
    lsn: u64,
}

impl Drop for RetentionPin<'_> {
    fn drop(&mut self) {
        let mut state = self.retention.state.lock();
        if let Some(index) = state.pins.iter().position(|&lsn| lsn == self.lsn) {
            state.pins.swap_remove(index);
        }
    }
}

impl WalRetention {
    /// The retention of a database with `limit` (see
    /// [`Config::wal_retention_limit`](crate::Config::wal_retention_limit)).
    #[must_use]
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            limit,
            limited_warned: AtomicBool::new(false),
            unreadable_warned: AtomicBool::new(false),
            state: parking_lot::Mutex::new(RetentionState::default()),
        }
    }

    /// Keeps the log from `lsn` on for a backup that copies it, until the
    /// returned pin is dropped (when the backup ends, also on a failure):
    /// checkpoints in between remove none of it, whatever the limit.
    ///
    /// `None` when a checkpoint of this open already removed the log below
    /// a floor above `lsn` (the limit cut the cursor off): that range may
    /// be gone, and only a full backup continues. The pin is taken under
    /// the lock a checkpoint holds from its floor to its removal
    /// ([`release_below_floor`](Self::release_below_floor)), so no
    /// checkpoint runs in between.
    pub(crate) fn pin(&self, lsn: u64) -> Option<RetentionPin<'_>> {
        let mut state = self.state.lock();
        if lsn < state.removed_below {
            return None;
        }
        state.pins.push(lsn);
        Some(RetentionPin {
            retention: self,
            lsn,
        })
    }

    /// Decides where a checkpoint at `checkpoint_lsn` cuts the log
    /// ([`retention_floor`] of `cursor`, the pins of running backups and the
    /// WAL's `segment_starts`), and calls `remove` with the floor, all under
    /// one lock, so a backup never pins log that this removal takes.
    ///
    /// # Errors
    ///
    /// The error of `remove`; the floor is then not recorded.
    pub(super) fn release_below_floor<E>(
        &self,
        checkpoint_lsn: u64,
        cursor: Option<CursorPosition>,
        segment_starts: &[u64],
        remove: impl FnOnce(u64) -> Result<(), E>,
    ) -> Result<RetentionFloor, E> {
        let mut state = self.state.lock();
        let pins = state.pins.iter().copied().map(CursorPosition::Pinned);
        let floor = retention_floor(
            checkpoint_lsn,
            cursor.into_iter().chain(pins),
            self.limit,
            segment_starts,
        );
        remove(floor.lsn)?;
        state.removed_below = state.removed_below.max(floor.lsn);
        Ok(floor)
    }

    /// Whether the limit cut off a cursor's log for the first time in this
    /// open: the caller warns then.
    pub(super) fn first_limited(&self) -> bool {
        !self.limited_warned.swap(true, Ordering::Relaxed)
    }

    /// Whether the cursor was unreadable for the first time in this open.
    pub(super) fn first_unreadable(&self) -> bool {
        !self.unreadable_warned.swap(true, Ordering::Relaxed)
    }
}

impl Default for WalRetention {
    fn default() -> Self {
        Self::new(crate::config::DEFAULT_WAL_RETENTION_LIMIT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn without_a_cursor_the_log_is_cut_at_the_checkpoint() {
        let floor = retention_floor(1988, [], GIB, &[]);
        assert_eq!(
            floor,
            RetentionFloor {
                lsn: 1988,
                limited: false
            }
        );
    }

    #[test]
    fn a_cursor_within_the_limit_keeps_its_log() {
        let floor = retention_floor(1988, [CursorPosition::At(319)], GIB, &[]);
        assert_eq!(
            floor,
            RetentionFloor {
                lsn: 319,
                limited: false
            }
        );
    }

    #[test]
    fn a_cursor_ahead_of_the_checkpoint_keeps_nothing_more() {
        let floor = retention_floor(319, [CursorPosition::At(1988)], GIB, &[]);
        assert_eq!(floor.lsn, 319);
        assert!(!floor.limited);
    }

    #[test]
    fn a_cursor_past_the_limit_is_cut_off_at_the_limit() {
        let floor = retention_floor(3 * GIB, [CursorPosition::At(19)], GIB, &[]);
        assert_eq!(
            floor,
            RetentionFloor {
                lsn: 2 * GIB,
                limited: true
            }
        );
    }

    #[test]
    fn a_cursor_exactly_at_the_limit_is_kept_whole() {
        let floor = retention_floor(3 * GIB, [CursorPosition::At(2 * GIB)], GIB, &[]);
        assert_eq!(floor.lsn, 2 * GIB);
        assert!(!floor.limited);
    }

    #[test]
    fn an_unreadable_cursor_is_held_to_the_limit() {
        let floor = retention_floor(3 * GIB, [CursorPosition::Unreadable], GIB, &[]);
        assert_eq!(
            floor,
            RetentionFloor {
                lsn: 2 * GIB,
                limited: true
            }
        );
        // Below the limit it keeps the whole log, as a cursor at 0 would.
        let floor = retention_floor(88, [CursorPosition::Unreadable], GIB, &[]);
        assert_eq!(floor.lsn, 0);
        assert!(!floor.limited);
    }

    #[test]
    fn the_lowest_of_several_cursors_wins() {
        let floor = retention_floor(
            1988,
            [CursorPosition::At(319), CursorPosition::At(88)],
            GIB,
            &[],
        );
        assert_eq!(floor.lsn, 88);
    }

    #[test]
    fn a_zero_limit_keeps_nothing_beyond_the_checkpoint() {
        let floor = retention_floor(1988, [CursorPosition::At(3)], 0, &[]);
        assert_eq!(floor.lsn, 1988);
        assert!(floor.limited);
    }

    #[test]
    fn a_pin_is_kept_whole_past_the_limit() {
        let floor = retention_floor(
            3 * GIB,
            [CursorPosition::At(19), CursorPosition::Pinned(19)],
            GIB,
            &[],
        );
        assert_eq!(
            floor,
            RetentionFloor {
                lsn: 19,
                limited: false
            }
        );
    }

    #[test]
    fn a_pin_above_the_cursor_does_not_save_the_cursor_from_the_limit() {
        let floor = retention_floor(
            3 * GIB,
            [CursorPosition::At(19), CursorPosition::Pinned(GIB)],
            GIB,
            &[],
        );
        assert_eq!(
            floor,
            RetentionFloor {
                lsn: GIB,
                limited: true
            }
        );
    }

    #[test]
    fn a_cut_off_cursor_loses_the_segment_that_holds_the_cut() {
        // Segments at 0, 0.5, 1.5 and 2.5 GiB, the checkpoint at 3 GiB: the
        // cut at 2 GiB falls inside the segment from 1.5 GiB, which goes too.
        let starts = [0, GIB / 2, 3 * GIB / 2, 5 * GIB / 2, 3 * GIB];
        let floor = retention_floor(3 * GIB, [CursorPosition::At(19)], GIB, &starts);
        assert_eq!(
            floor,
            RetentionFloor {
                lsn: 5 * GIB / 2,
                limited: true
            }
        );
    }

    #[test]
    fn a_cursor_the_limit_leaves_whole_keeps_its_segment() {
        let starts = [0, GIB / 2, 3 * GIB / 2, 5 * GIB / 2, 3 * GIB];
        let floor = retention_floor(3 * GIB, [CursorPosition::At(2 * GIB)], GIB, &starts);
        assert_eq!(floor.lsn, 2 * GIB);
        assert!(!floor.limited);
    }

    #[test]
    fn the_cut_never_moves_past_a_pin() {
        let starts = [0, GIB / 2, 3 * GIB / 2, 5 * GIB / 2, 3 * GIB];
        let floor = retention_floor(
            3 * GIB,
            [CursorPosition::At(19), CursorPosition::Pinned(9 * GIB / 4)],
            GIB,
            &starts,
        );
        assert_eq!(
            floor,
            RetentionFloor {
                lsn: 2 * GIB,
                limited: true
            }
        );
    }

    #[test]
    fn without_a_later_segment_start_the_cut_stays() {
        let floor = retention_floor(3 * GIB, [CursorPosition::At(19)], GIB, &[0, GIB]);
        assert_eq!(floor.lsn, 2 * GIB);
        assert!(floor.limited);
    }

    /// The floor a checkpoint at `checkpoint_lsn` without a cursor finds.
    fn floor_of(retention: &WalRetention, checkpoint_lsn: u64) -> u64 {
        retention
            .release_below_floor(checkpoint_lsn, None, &[], |_| Ok::<(), ()>(()))
            .unwrap()
            .lsn
    }

    #[test]
    fn a_pin_keeps_the_log_until_it_is_dropped() {
        let retention = WalRetention::new(GIB);
        let first = retention.pin(88).unwrap();
        let second = retention.pin(88).unwrap();
        assert_eq!(floor_of(&retention, 1988), 88);
        drop(first);
        assert_eq!(floor_of(&retention, 1988), 88);
        drop(second);
        assert_eq!(floor_of(&retention, 1988), 1988);
    }

    #[test]
    fn a_pin_below_a_removed_floor_is_refused() {
        let retention = WalRetention::new(GIB);
        assert_eq!(floor_of(&retention, 1988), 1988);
        assert!(retention.pin(319).is_none());
        assert!(retention.pin(1988).is_some());
    }

    #[test]
    fn a_failed_removal_records_no_floor() {
        let retention = WalRetention::new(GIB);
        assert!(
            retention
                .release_below_floor(1988, None, &[], |_| Err("disk"))
                .is_err()
        );
        assert!(retention.pin(319).is_some());
    }

    #[test]
    fn each_warning_is_given_once() {
        let retention = WalRetention::new(GIB);
        assert!(retention.first_limited());
        assert!(!retention.first_limited());
        assert!(retention.first_unreadable());
        assert!(!retention.first_unreadable());
    }
}
