//! The database's history: how long it keeps it, when each commit happened,
//! and the oldest epoch a read may still ask for (H2c, the time-travel
//! design).
//!
//! A time-travel read is a read at an epoch in the past, through the same
//! reads at a view as a transaction's snapshot: the stores keep the versions
//! a reader there reads as long as garbage collection does not drop them.
//! So history is one horizon. The collect horizon (see
//! [`TransactionManager::collect_horizon`](super::TransactionManager::collect_horizon))
//! is the older of the read horizon (the oldest snapshot a reader holds) and
//! the start of the retention window, and whatever it reaches becomes the
//! floor: a read before it is refused ("history before ... is not
//! retained"), never answered from versions a collect may have dropped.
//!
//! The retention window is in time, so every commit records when it
//! happened while history is on: the time index maps "at T" to the last
//! epoch committed at or before T. A commit takes its time with its epoch,
//! under the commit lock, so times follow the epochs (a clock that goes back
//! repeats the last time); the WAL group of the commit carries the same
//! time, so a replay after a crash rebuilds the index the live commits
//! built. Times are in milliseconds since the Unix epoch, the resolution of
//! the WAL groups.
//!
//! Off (the default), nothing here runs on a commit but one flag test: no
//! time is taken, nothing is recorded, and the collect horizon is the read
//! horizon, as without the feature.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use grafeo_common::types::{EpochId, Timestamp};
use grafeo_common::utils::error::{Error, Result};
use parking_lot::Mutex;

use crate::config::HistoryRetention;

/// The clock commits take their times from: milliseconds since the Unix
/// epoch.
pub(crate) type Clock = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Milliseconds since the Unix epoch, 0 for a clock set before it.
fn wall_clock() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
        })
}

/// What a checkpoint writes of the history, and a load gives back: the
/// floor, and the commit times from the newest at or before it on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HistoryImage {
    /// The oldest epoch a read may ask for.
    pub floor: u64,
    /// `(epoch, milliseconds)` per commit, epochs ascending, times not
    /// going back.
    pub times: Vec<(u64, u64)>,
}

/// The retention setting and the time index (see the module docs).
pub(crate) struct History {
    /// Whether history is on: the one test a commit pays while it is off.
    on: AtomicBool,
    /// The retention window, while on.
    retention: Mutex<HistoryRetention>,
    /// The commit times, epochs ascending: the newest at or before the floor
    /// first, so a time just after the floor still finds its epoch.
    times: Mutex<VecDeque<(u64, u64)>>,
    /// The last time a commit took, so the next one never goes back.
    last_time: AtomicU64,
    /// The oldest epoch a read may ask for: the newest collect horizon.
    floor: AtomicU64,
    clock: Clock,
}

impl std::fmt::Debug for History {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("History")
            .field("retention", &*self.retention.lock())
            .field("floor", &self.floor.load(Ordering::Relaxed))
            .field("times", &self.times.lock().len())
            .finish_non_exhaustive()
    }
}

impl Default for History {
    fn default() -> Self {
        Self::with_clock(Arc::new(wall_clock))
    }
}

impl History {
    /// History off, its times taken from `clock` once it is on.
    pub(crate) fn with_clock(clock: Clock) -> Self {
        Self {
            on: AtomicBool::new(false),
            retention: Mutex::new(HistoryRetention::Off),
            times: Mutex::new(VecDeque::new()),
            last_time: AtomicU64::new(0),
            floor: AtomicU64::new(0),
            clock,
        }
    }

    /// Whether history is on.
    pub(crate) fn is_on(&self) -> bool {
        self.on.load(Ordering::Acquire)
    }

    /// The retention setting.
    pub(crate) fn retention(&self) -> HistoryRetention {
        *self.retention.lock()
    }

    /// Sets the retention an open starts with, before it loads anything:
    /// the times come from the file and the log, and
    /// [`anchor`](Self::anchor) starts them at the end of the open.
    pub(crate) fn open(&self, retention: HistoryRetention) {
        *self.retention.lock() = retention;
        self.on
            .store(retention != HistoryRetention::Off, Ordering::Release);
    }

    /// Records `published` now when no commit at or after it is recorded:
    /// what a new database and the end of an open do, so a read at a time
    /// from now on finds its epoch.
    pub(crate) fn anchor(&self, published: u64) {
        if !self.is_on() {
            return;
        }
        let newest = self.times.lock().back().map(|(epoch, _)| *epoch);
        if newest.is_none_or(|newest| newest < published) {
            let now = self.next_time();
            self.record(EpochId::new(published), now);
        }
    }

    /// Sets the retention from now on, the published epoch being
    /// `published`: turned on, the index starts at the published epoch, now
    /// (what was collected before stays gone); turned off, it is dropped.
    pub(crate) fn set_retention(&self, retention: HistoryRetention, published: u64) {
        let mut current = self.retention.lock();
        let was_on = *current != HistoryRetention::Off;
        *current = retention;
        let on = retention != HistoryRetention::Off;
        if on && !was_on {
            let now = self.next_time();
            let mut times = self.times.lock();
            if times.back().is_none_or(|(epoch, _)| *epoch < published) {
                times.push_back((published, now));
            }
        } else if !on {
            self.times.lock().clear();
        }
        self.on.store(on, Ordering::Release);
    }

    /// Now on the clock commits take their times from, in milliseconds
    /// since the Unix epoch: what a point in time "ago" counts back from.
    pub(crate) fn now(&self) -> u64 {
        (self.clock)()
    }

    /// The time the next commit takes: now, or the last time when the clock
    /// went back. Called under the commit lock, so times follow the epochs.
    pub(crate) fn next_time(&self) -> u64 {
        let now = (self.clock)();
        self.last_time.fetch_max(now, Ordering::AcqRel).max(now)
    }

    /// The time a commit takes, `None` while history is off.
    pub(crate) fn commit_time(&self) -> Option<u64> {
        self.is_on().then(|| self.next_time())
    }

    /// Records that `epoch` committed at `millis`, a time
    /// [`commit_time`](Self::commit_time) gave or a WAL group holds; an
    /// epoch at or below the newest recorded one changes nothing.
    pub(crate) fn record(&self, epoch: EpochId, millis: u64) {
        if !self.is_on() {
            return;
        }
        let epoch = epoch.as_u64();
        let mut times = self.times.lock();
        let millis = times.back().map_or(millis, |(_, last)| millis.max(*last));
        if times.back().is_none_or(|(newest, _)| *newest < epoch) {
            times.push_back((epoch, millis));
        }
        drop(times);
        self.last_time.fetch_max(millis, Ordering::AcqRel);
    }

    /// The oldest epoch a read may ask for.
    pub(crate) fn floor(&self) -> EpochId {
        EpochId::new(self.floor.load(Ordering::Acquire))
    }

    /// The start of the retention window: `None` while history is off (the
    /// read horizon decides alone); the floor for `Forever`, and for a
    /// window that reaches back past every recorded commit; else the last
    /// epoch committed at or before now minus the window (a reader there
    /// reads the state at that time), never below the floor.
    pub(crate) fn retention_horizon(&self) -> Option<u64> {
        // Off: one flag test, no lock, as a collect without history costs.
        if !self.is_on() {
            return None;
        }
        let retention = *self.retention.lock();
        let floor = self.floor.load(Ordering::Acquire);
        match retention {
            HistoryRetention::Off => None,
            HistoryRetention::Forever => Some(floor),
            HistoryRetention::Duration(window) => {
                let window = u64::try_from(window.as_millis()).unwrap_or(u64::MAX);
                let start = (self.clock)().saturating_sub(window);
                let times = self.times.lock();
                let at = times.partition_point(|(_, millis)| *millis <= start);
                let epoch = at.checked_sub(1).map_or(floor, |index| times[index].0);
                Some(epoch.max(floor))
            }
        }
    }

    /// Moves the floor to `horizon` when it is later, the horizon a collect
    /// runs at, and drops the commit times no read needs any more (those
    /// before the newest at or before the floor). The caller holds the read
    /// pins' lock, so no read pins below the floor meanwhile.
    pub(crate) fn collected_at(&self, horizon: EpochId) {
        let floor = self.floor.fetch_max(horizon.as_u64(), Ordering::AcqRel);
        if self.is_on() && horizon.as_u64() > floor {
            let mut times = self.times.lock();
            let keep_from = times
                .partition_point(|(epoch, _)| *epoch <= horizon.as_u64())
                .saturating_sub(1);
            times.drain(..keep_from);
        }
    }

    /// The error of a read at `epoch`, before the floor.
    pub(crate) fn before_floor(&self, epoch: EpochId) -> Error {
        Error::InvalidValue(format!(
            "history before epoch {} is not retained: a read at epoch {} asks for older \
             history than the database keeps (see its history retention)",
            self.floor().as_u64(),
            epoch.as_u64()
        ))
    }

    /// The epoch a read at `timestamp` reads at: the last one committed at
    /// or before it.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidValue`] while history is off, and for a time before
    /// the oldest retained commit or one whose epoch lies before the floor:
    /// "history before ... is not retained".
    pub(crate) fn epoch_at(&self, timestamp: Timestamp) -> Result<EpochId> {
        if !self.is_on() {
            return Err(Error::InvalidValue(
                "a read at a time needs history: the database keeps none (its history \
                 retention is off)"
                    .to_string(),
            ));
        }
        let millis = u64::try_from(timestamp.as_millis()).unwrap_or(0);
        let times = self.times.lock();
        let at = times.partition_point(|(_, time)| *time <= millis);
        let Some(index) = at.checked_sub(1) else {
            let oldest = times.front().map_or(0, |(_, time)| *time);
            return Err(Error::InvalidValue(format!(
                "history before {} is not retained: a read at {} asks for older history than \
                 the database keeps",
                Timestamp::from_millis(i64::try_from(oldest).unwrap_or(i64::MAX)),
                timestamp
            )));
        };
        let epoch = EpochId::new(times[index].0);
        drop(times);
        if epoch < self.floor() {
            return Err(self.before_floor(epoch));
        }
        Ok(epoch)
    }

    /// What a checkpoint writes, `None` while history is off.
    pub(crate) fn image(&self) -> Option<HistoryImage> {
        if !self.is_on() {
            return None;
        }
        Some(HistoryImage {
            floor: self.floor.load(Ordering::Acquire),
            times: self.times.lock().iter().copied().collect(),
        })
    }

    /// Takes in what a load read: the floor, and the commit times, before
    /// any replayed commit is recorded. The times are kept while history is
    /// on (an open that turned it off drops them).
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn restore(&self, image: HistoryImage) {
        self.floor.fetch_max(image.floor, Ordering::AcqRel);
        if !self.is_on() {
            return;
        }
        let mut times = self.times.lock();
        times.clear();
        times.extend(image.times.iter().copied());
        if let Some((_, last)) = times.back() {
            self.last_time.fetch_max(*last, Ordering::AcqRel);
        }
    }

    /// The commit times, for tests.
    #[cfg(test)]
    pub(crate) fn times(&self) -> Vec<(u64, u64)> {
        self.times.lock().iter().copied().collect()
    }
}

#[cfg(test)]
mod tests;
