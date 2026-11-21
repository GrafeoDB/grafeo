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
    grafeo_common::clock::unix_millis()
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

/// One commit of the time index: its epoch, its time in milliseconds, and the bytes of the
/// labeled property graph's committed state the commits replaced up to and including it (see
/// [`History::record_replacing`]).
type CommitTime = (u64, u64, u64);

/// The retention setting and the time index (see the module docs).
pub(crate) struct History {
    /// Whether history is on: the one test a commit pays while it is off.
    on: AtomicBool,
    /// The retention window, while on.
    retention: Mutex<HistoryRetention>,
    /// How many bytes the commits since the window's start may have replaced, `u64::MAX` for
    /// no limit: past it the start moves forward, whichever of time or bytes comes first.
    limit: AtomicU64,
    /// The commit times, epochs ascending: the newest at or before the floor
    /// first, so a time just after the floor still finds its epoch.
    times: Mutex<VecDeque<CommitTime>>,
    /// The bytes the commits recorded so far replaced.
    replaced: AtomicU64,
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
            limit: AtomicU64::new(u64::MAX),
            times: Mutex::new(VecDeque::new()),
            replaced: AtomicU64::new(0),
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

    /// Whether the time index counts bytes: history is on and a retention limit is set.
    pub(crate) fn counts_bytes(&self) -> bool {
        self.is_on() && self.limit.load(Ordering::Acquire) != u64::MAX
    }

    /// Counts what the store keeps of replaced state at the end of an open (`kept`: the bytes per
    /// replacing epoch, see `RowGroupStore::kept_superseded`) into the time index: each commit's
    /// total holds what the epochs up to it replaced, so the window's byte start reads the loaded
    /// and replayed history as the commits that made it would have left it.
    pub(crate) fn count_kept(&self, kept: Vec<(u64, usize)>) {
        let mut times = self.times.lock();
        // The bytes each commit adds, those of the epochs after the commit before it up to it
        // (the times are in epoch order); the last slot holds the epochs after every commit.
        let mut added = vec![0u64; times.len() + 1];
        for (at, bytes) in kept {
            let index = times.partition_point(|(epoch, _, _)| *epoch < at);
            added[index] = added[index].saturating_add(bytes as u64);
        }
        let mut counted = 0u64;
        for ((_, _, replaced), add) in times.iter_mut().zip(&added) {
            counted = counted.saturating_add(*add);
            *replaced = replaced.saturating_add(counted);
        }
        counted = counted.saturating_add(added[added.len() - 1]);
        self.replaced.fetch_add(counted, Ordering::AcqRel);
    }

    /// Sets how many bytes the commits since the window's start may have replaced (`None`: no
    /// limit): past it the start moves forward, whichever of time or bytes comes first.
    pub(crate) fn set_retention_limit(&self, bytes: Option<u64>) {
        self.limit
            .store(bytes.unwrap_or(u64::MAX), Ordering::Release);
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
        let newest = self.times.lock().back().map(|(epoch, _, _)| *epoch);
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
            let replaced = self.replaced.load(Ordering::Acquire);
            let mut times = self.times.lock();
            if times.back().is_none_or(|(epoch, _, _)| *epoch < published) {
                times.push_back((published, now, replaced));
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
        self.record_replacing(epoch, millis, 0);
    }

    /// [`record`](Self::record) for a commit that replaced `bytes` of the labeled property
    /// graph's committed state, which the window keeps for readers before it: the bytes the
    /// retention limit counts.
    pub(crate) fn record_replacing(&self, epoch: EpochId, millis: u64, bytes: u64) {
        if !self.is_on() {
            return;
        }
        let epoch = epoch.as_u64();
        let mut times = self.times.lock();
        let millis = times
            .back()
            .map_or(millis, |(_, last, _)| millis.max(*last));
        if times.back().is_none_or(|(newest, _, _)| *newest < epoch) {
            let replaced = self
                .replaced
                .fetch_add(bytes, Ordering::AcqRel)
                .saturating_add(bytes);
            times.push_back((epoch, millis, replaced));
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
    /// reads the state at that time), never below the floor. Under a retention limit, no
    /// earlier than the oldest commit after which the commits replaced at most the limit.
    pub(crate) fn retention_horizon(&self) -> Option<u64> {
        // Off: one flag test, no lock, as a collect without history costs.
        if !self.is_on() {
            return None;
        }
        let retention = *self.retention.lock();
        let floor = self.floor.load(Ordering::Acquire);
        let start = match retention {
            HistoryRetention::Off => return None,
            HistoryRetention::Forever => floor,
            HistoryRetention::Duration(window) => {
                let window = u64::try_from(window.as_millis()).unwrap_or(u64::MAX);
                let start = (self.clock)().saturating_sub(window);
                let times = self.times.lock();
                let at = times.partition_point(|(_, millis, _)| *millis <= start);
                let epoch = at.checked_sub(1).map_or(floor, |index| times[index].0);
                epoch.max(floor)
            }
        };
        let limit = self.limit.load(Ordering::Acquire);
        if limit == u64::MAX {
            return Some(start);
        }
        // The bytes replaced since a commit: the newest total less the total at it.
        let times = self.times.lock();
        let newest = times.back().map_or(0, |(_, _, replaced)| *replaced);
        let at = times.partition_point(|(_, _, replaced)| newest - replaced > limit);
        let by_bytes = times.get(at).map_or(start, |(epoch, _, _)| *epoch);
        Some(start.max(by_bytes))
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
                .partition_point(|(epoch, _, _)| *epoch <= horizon.as_u64())
                .saturating_sub(1);
            times.drain(..keep_from);
        }
    }

    /// Starts the history at `published`, now: what a database keeps once a
    /// snapshot arrived (an import, a restore). A snapshot carries no
    /// history, so the floor moves to its epoch, as a load of a file
    /// without a time index starts at its checkpoint, and the commit times
    /// before it go: they name states the snapshot replaced. The caller
    /// holds the read pins' lock, as for [`collected_at`](Self::collected_at).
    pub(crate) fn restart(&self, published: EpochId) {
        self.floor.fetch_max(published.as_u64(), Ordering::AcqRel);
        if !self.is_on() {
            return;
        }
        let now = self.next_time();
        let replaced = self.replaced.load(Ordering::Acquire);
        let mut times = self.times.lock();
        times.clear();
        times.push_back((published.as_u64(), now, replaced));
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
        let at = times.partition_point(|(_, time, _)| *time <= millis);
        let Some(index) = at.checked_sub(1) else {
            let oldest = times.front().map_or(0, |(_, time, _)| *time);
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
            times: self
                .times
                .lock()
                .iter()
                .map(|(epoch, millis, _)| (*epoch, *millis))
                .collect(),
        })
    }

    /// Takes in what a load read: the floor, and the commit times, before
    /// any replayed commit is recorded. The times are kept while history is
    /// on (an open that turned it off drops them).
    pub(crate) fn restore(&self, image: HistoryImage) {
        self.floor.fetch_max(image.floor, Ordering::AcqRel);
        if !self.is_on() {
            return;
        }
        // The bytes of what a load gives back are counted at the end of the open (H4).
        let replaced = self.replaced.load(Ordering::Acquire);
        let mut times = self.times.lock();
        times.clear();
        times.extend(
            image
                .times
                .iter()
                .map(|(epoch, millis)| (*epoch, *millis, replaced)),
        );
        if let Some((_, last, _)) = times.back() {
            self.last_time.fetch_max(*last, Ordering::AcqRel);
        }
    }

    /// Each entry's epoch and the bytes the commits replaced up to it, for tests.
    #[cfg(test)]
    pub(crate) fn totals(&self) -> Vec<(u64, u64)> {
        self.times
            .lock()
            .iter()
            .map(|(epoch, _, replaced)| (*epoch, *replaced))
            .collect()
    }

    /// The commit times, for tests.
    #[cfg(test)]
    pub(crate) fn times(&self) -> Vec<(u64, u64)> {
        self.times
            .lock()
            .iter()
            .map(|(epoch, millis, _)| (*epoch, *millis))
            .collect()
    }
}

#[cfg(test)]
mod tests;
