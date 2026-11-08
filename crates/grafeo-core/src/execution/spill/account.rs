//! The spill account: the bytes one open has spilled, against its cap and
//! the room the database keeps on the spill volume.
//!
//! Every spill reserves its bytes here before it writes them, and a
//! reservation gives them back when it drops (its file is removed). The cap
//! bounds what all of an open's spills hold at once; the room keeps the free
//! space the database needs for its next checkpoint. The free space is read
//! through a closure the engine passes in (this crate has no filesystem
//! dependency): once for each new file, and again after every
//! [`SPILL_PROBE_STEP`] reserved since the last read, never on the writes in
//! between.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::utils::error::LimitRefusal;
use parking_lot::Mutex;

/// Probe the free space again after this much reserved since the last
/// probe: an internal step, not a setting.
pub const SPILL_PROBE_STEP: u64 = 64 << 20;

/// Who spilled, for the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpillOwner {
    /// An external sort's runs.
    Sort,
    /// A spilled aggregate's partitions.
    Aggregate,
    /// Spilled embedding columns of vector indexes.
    VectorColumns,
}

impl SpillOwner {
    /// The owner's slot in the account.
    const fn index(self) -> usize {
        match self {
            Self::Sort => 0,
            Self::Aggregate => 1,
            Self::VectorColumns => 2,
        }
    }
}

/// The cap of one open's spill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillCap {
    /// At most this many bytes spilled at once (0: nothing spills).
    Bytes(u64),
    /// No cap; the room still holds.
    Unlimited,
}

/// The free-space probe and the room the database needs, from the engine.
pub struct SpillRoom {
    /// The spill volume, as the probe reads it (the spill directory).
    pub volume: PathBuf,
    /// The free bytes on the volume at `path`. An error leaves the cap
    /// alone in force; the probe reports it as it sees fit (the engine's
    /// warns once).
    pub probe: Box<dyn Fn(&Path) -> std::io::Result<u64> + Send + Sync>,
    /// The bytes the database needs free on the spill volume now: the
    /// file's length plus one WAL segment on its own volume, one WAL segment
    /// on another, 0 without a WAL.
    pub needed: Box<dyn Fn() -> u64 + Send + Sync>,
}

impl std::fmt::Debug for SpillRoom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpillRoom")
            .field("volume", &self.volume)
            .finish_non_exhaustive()
    }
}

/// What the account knows of the volume's free space.
#[derive(Debug, Default)]
struct ProbeState {
    /// The free bytes at the last probe that answered.
    free: Option<u64>,
    /// The bytes reserved since that probe.
    reserved_since: u64,
}

/// The bytes one open has spilled, its cap and the room it keeps.
#[derive(Debug)]
pub struct SpillAccount {
    used: AtomicU64,
    by_owner: [AtomicU64; 3],
    cap: SpillCap,
    room: Option<SpillRoom>,
    probe: Mutex<ProbeState>,
}

impl SpillAccount {
    /// An account with nothing spilled, capped at `cap`, keeping `room` on
    /// the spill volume (`None`: no probe, the cap alone bounds).
    #[must_use]
    pub fn new(cap: SpillCap, room: Option<SpillRoom>) -> Arc<Self> {
        Arc::new(Self {
            used: AtomicU64::new(0),
            by_owner: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            cap,
            room,
            probe: Mutex::new(ProbeState::default()),
        })
    }

    /// Reserves `bytes` before a write; `new_file` probes the free space
    /// first. The reservation gives them back when it drops.
    ///
    /// # Errors
    ///
    /// [`LimitRefusal::Spill`] past the cap, [`LimitRefusal::Room`] when the
    /// write would leave less free space than the database needs; nothing
    /// is counted then.
    pub fn reserve(
        self: &Arc<Self>,
        owner: SpillOwner,
        bytes: u64,
        new_file: bool,
    ) -> Result<SpillReservation, LimitRefusal> {
        self.take(owner, bytes, new_file)?;
        Ok(SpillReservation {
            account: Arc::clone(self),
            owner,
            bytes,
        })
    }

    /// [`reserve`](Self::reserve) of up to `wanted` bytes and at least
    /// `at_least`: as much of `wanted` as the cap leaves, and only
    /// `at_least` when the room refuses more.
    ///
    /// # Errors
    ///
    /// As [`reserve`](Self::reserve), when not even `at_least` fits.
    pub fn reserve_up_to(
        self: &Arc<Self>,
        owner: SpillOwner,
        wanted: u64,
        at_least: u64,
        new_file: bool,
    ) -> Result<SpillReservation, LimitRefusal> {
        let bytes = self.take_up_to(owner, wanted, at_least, new_file)?;
        Ok(SpillReservation {
            account: Arc::clone(self),
            owner,
            bytes,
        })
    }

    /// The bytes spilled now.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// The bytes `owner` has spilled now.
    #[must_use]
    pub fn used_by(&self, owner: SpillOwner) -> u64 {
        self.by_owner[owner.index()].load(Ordering::Relaxed)
    }

    /// The cap.
    #[must_use]
    pub fn cap(&self) -> SpillCap {
        self.cap
    }

    /// Counts `bytes` for `owner` if the cap and the room allow them.
    fn take(&self, owner: SpillOwner, bytes: u64, new_file: bool) -> Result<(), LimitRefusal> {
        self.take_up_to(owner, bytes, bytes, new_file).map(|_| ())
    }

    /// Counts up to `wanted` bytes for `owner` and at least `at_least`
    /// (`at_least <= wanted`): as much as the cap leaves, then only
    /// `at_least` when the room refuses that. The bytes counted.
    fn take_up_to(
        &self,
        owner: SpillOwner,
        wanted: u64,
        at_least: u64,
        new_file: bool,
    ) -> Result<u64, LimitRefusal> {
        let limit = match self.cap {
            SpillCap::Bytes(limit) => limit,
            SpillCap::Unlimited => u64::MAX,
        };
        let at_least = at_least.min(wanted);
        let mut bytes = at_least;
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                bytes = wanted.min(limit.saturating_sub(used));
                (bytes >= at_least).then_some(used + bytes)
            })
            .map_err(|used| LimitRefusal::Spill {
                requested: at_least,
                used,
                limit,
            })?;
        if let Err(refusal) = self.check_room(bytes, new_file) {
            // A step the room refuses may leave room for what the write
            // needs: the probe of a new file was just taken. When that is
            // refused too, its refusal states what the write needs.
            let refused = if bytes == at_least {
                Some(refusal)
            } else {
                self.check_room(at_least, false).err()
            };
            if let Some(refusal) = refused {
                self.used.fetch_sub(bytes, Ordering::AcqRel);
                return Err(refusal);
            }
            self.used.fetch_sub(bytes - at_least, Ordering::AcqRel);
            bytes = at_least;
        }
        self.by_owner[owner.index()].fetch_add(bytes, Ordering::Relaxed);
        Ok(bytes)
    }

    /// Whether `bytes` more leave the room the database needs: probes for a
    /// new file and once per step, and otherwise counts down from the last
    /// probe. A probe that fails leaves the cap alone in force (the
    /// engine's probe reports it, once).
    fn check_room(&self, bytes: u64, new_file: bool) -> Result<(), LimitRefusal> {
        let Some(room) = &self.room else {
            return Ok(());
        };
        let mut state = self.probe.lock();
        if new_file || state.reserved_since.saturating_add(bytes) >= SPILL_PROBE_STEP {
            match (room.probe)(&room.volume) {
                Ok(free) => {
                    state.free = Some(free);
                    state.reserved_since = 0;
                }
                Err(_) => {
                    // What was read before no longer holds.
                    state.free = None;
                    state.reserved_since = 0;
                }
            }
        }
        if let Some(free) = state.free {
            let available = free.saturating_sub(state.reserved_since);
            let needed = (room.needed)();
            if available.saturating_sub(bytes) < needed {
                return Err(LimitRefusal::Room {
                    requested: bytes,
                    free: available,
                    needed,
                    volume: room.volume.clone(),
                });
            }
        }
        state.reserved_since = state.reserved_since.saturating_add(bytes);
        Ok(())
    }

    /// Gives back `bytes` of `owner`'s. They no longer take room since the
    /// last probe either: what the next reservation sees free grows by them
    /// (at most by what it counted since that probe).
    fn give_back(&self, owner: SpillOwner, bytes: u64) {
        if bytes == 0 {
            return;
        }
        self.used.fetch_sub(bytes, Ordering::AcqRel);
        self.by_owner[owner.index()].fetch_sub(bytes, Ordering::Relaxed);
        if self.room.is_some() {
            let mut state = self.probe.lock();
            state.reserved_since = state.reserved_since.saturating_sub(bytes);
        }
    }
}

/// A reservation of spilled bytes, given back when it drops (its file is
/// removed then).
#[derive(Debug)]
pub struct SpillReservation {
    account: Arc<SpillAccount>,
    owner: SpillOwner,
    bytes: u64,
}

impl SpillReservation {
    /// Reserves `bytes` more.
    ///
    /// # Errors
    ///
    /// As [`SpillAccount::reserve`]; the reservation keeps its size then.
    pub fn grow(&mut self, bytes: u64) -> Result<(), LimitRefusal> {
        self.account.take(self.owner, bytes, false)?;
        self.bytes += bytes;
        Ok(())
    }

    /// Reserves up to `wanted` bytes more and at least `at_least` (see
    /// [`SpillAccount::reserve_up_to`]).
    ///
    /// # Errors
    ///
    /// As [`SpillAccount::reserve`], when not even `at_least` fits; the
    /// reservation keeps its size then.
    pub fn grow_up_to(&mut self, wanted: u64, at_least: u64) -> Result<(), LimitRefusal> {
        self.bytes += self
            .account
            .take_up_to(self.owner, wanted, at_least, false)?;
        Ok(())
    }

    /// Gives back what it holds past `bytes` (a file shorter than reserved).
    pub fn trim_to(&mut self, bytes: u64) {
        if bytes < self.bytes {
            self.account.give_back(self.owner, self.bytes - bytes);
            self.bytes = bytes;
        }
    }

    /// The bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for SpillReservation {
    fn drop(&mut self) {
        self.account.give_back(self.owner, self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reservation_past_u64_is_refused_not_wrapped() {
        let account = SpillAccount::new(SpillCap::Unlimited, None);
        let _held = account
            .reserve(SpillOwner::Sort, u64::MAX - 1, true)
            .unwrap();
        let refusal = account
            .reserve(SpillOwner::Sort, 2, false)
            .expect_err("it would wrap");
        assert!(matches!(refusal, LimitRefusal::Spill { .. }), "{refusal:?}");
    }

    #[test]
    fn a_failed_probe_forgets_the_free_space_it_read_before() {
        let calls = Arc::new(AtomicU64::new(0));
        let counted = Arc::clone(&calls);
        let room = SpillRoom {
            volume: PathBuf::from("lyon"),
            probe: Box::new(move |_: &Path| {
                if counted.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(10)
                } else {
                    Err(std::io::Error::other("gone"))
                }
            }),
            needed: Box::new(|| 5),
        };
        let account = SpillAccount::new(SpillCap::Unlimited, Some(room));
        let _first = account.reserve(SpillOwner::Sort, 4, true).unwrap();
        // The volume no longer answers: the cap alone bounds.
        let _second = account.reserve(SpillOwner::Sort, 100, true).unwrap();
        assert_eq!(account.used(), 104);
    }

    /// When neither the step nor what the write needs leaves the room, the
    /// refusal states what the write needs, not the step.
    #[test]
    fn a_refused_fallback_states_what_the_write_needs() {
        let room = SpillRoom {
            volume: PathBuf::from("rotterdam"),
            probe: Box::new(|_: &Path| Ok(100)),
            needed: Box::new(|| 50),
        };
        let account = SpillAccount::new(SpillCap::Unlimited, Some(room));
        let refusal = account
            .reserve_up_to(SpillOwner::Sort, 1_000, 80, true)
            .expect_err("80 bytes leave 20 of the 50 needed");
        assert!(
            matches!(refusal, LimitRefusal::Room { requested: 80, .. }),
            "{refusal:?}"
        );
        assert_eq!(account.used(), 0, "nothing counted");
    }
}
