//! The WAL v2 writer: groups of frames in segment files.
//!
//! A transaction writes one contiguous group through a [`GroupWriter`]:
//! [`Wal::begin_group`] takes the writer's lock for the whole group, so groups
//! never interleave, and [`GroupWriter::finish`] writes the LAST frame, the
//! group's commit marker, then syncs as the durability mode asks.
//!
//! **Failures** (#498):
//! - A write that fails before the LAST frame is complete cuts the segment
//!   back to where the group started, so the next group lands at the same
//!   log position and nothing of the failed one is ever replayed
//!   ([`GroupError::NotWritten`]). When the cut fails too, the writer is
//!   poisoned: it takes no more groups until a reopen, and the partial group
//!   stays as the torn tail at the end of the log, which the next open cuts.
//! - A sync that fails after the LAST frame was written leaves the outcome
//!   unknown ([`GroupError::OutcomeUnknown`]) and poisons the writer: after a
//!   failed fsync the page cache can no longer be trusted, so only the next
//!   open, reading what the log holds, decides.
//! - Any other failed sync (a background flusher's, a rotation's) poisons the
//!   writer as well.
//!
//! **Segments** start with a FIRST frame and end after a LAST frame: the
//! writer rotates when a group begins in a segment that has reached the
//! configured size, and when [`Wal::rotate`] is called (every checkpoint).
//! Creating a segment writes and syncs its header under a temporary name,
//! renames it to its segment name, switches the writer to it and syncs the
//! directory (Unix), before the first frame: a segment name always has a
//! synced header, and a panic while creating one leaves only a temporary
//! file, which the next writable open removes (#725). Rotation syncs the
//! old segment first, so only the last segment can hold unsynced frames.
//!
//! **Group commit:** [`Wal::sync_until`] makes every group up to a log
//! position durable; one fsync covers every group published before it, and
//! a caller whose group a concurrent sync already covered returns at once.
//! [`GroupWriter::finish_unsynced`] and `sync_until` let a commit pipeline
//! release its commit lock between writing a group and syncing it.
//!
//! **Sync markers:** a group's FIRST frame records how far the log was
//! durable when the group began, which is how a scan tells a hole the disk
//! never had from damage to synced bytes. A sync says nothing in the groups
//! it covered (they all began before it), so after an fsync that covered
//! groups the writer appends a sync marker: an empty group of the system
//! transaction whose prologue records the sync. It costs no fsync of its
//! own, a scan does not return it, and a sync that covers nothing but the
//! last marker writes none.
//!
//! **Frames per key:** each segment has a key of its own, and every frame
//! sealed under it uses a random nonce. A group starts a new segment once
//! the active one holds [`DEFAULT_FRAMES_PER_SEGMENT_KEY`] frames (frames of
//! failed groups included), whatever the segment size.

#![deny(clippy::let_underscore_must_use)]

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use grafeo_common::testing::crash::{maybe_crash, maybe_fail};
use grafeo_common::types::TransactionId;
use parking_lot::{Mutex, MutexGuard};

use super::cipher::{CipherForSalt, frame_aad, seal_payload, segment_cipher};
#[cfg(feature = "encryption")]
use super::cipher::{key_check, new_salt};
use super::error::{GroupError, WalError};
use super::frame::{
    FRAME_HEADER_BYTES, FRAME_PROLOGUE_BYTES, FRAME_TARGET, FrameFlags, FrameHeader,
    MAX_FRAME_PAYLOAD, MAX_FRAME_RECORD_BYTES, SEALED_OVERHEAD,
};
use super::segment::{
    KEY_CHECK_BYTES, SALT_BYTES, SEGMENT_HEADER_BYTES, SegmentHeader, WalDirectory,
    is_unfinished_segment, list_wal_directory, segment_file_name, segment_may_hold_frames,
    segment_temp_file_name, stored_key_check_aad,
};
use super::{DurabilityMode, WalCipher};

/// The segment size the writer rotates at by default.
pub const DEFAULT_SEGMENT_BYTES: u64 = 64 * 1024 * 1024;

/// The frames a segment holds before the next group starts a new segment,
/// by default: with random 96-bit nonces, 2^24 frames under one key keep the
/// chance of a repeated nonce below 2^-49.
pub const DEFAULT_FRAMES_PER_SEGMENT_KEY: u64 = 1 << 24;

/// The frames one segment key seals at most: a group that would pass it
/// fails. One group of this many frames holds terabytes.
const MAX_FRAMES_PER_SEGMENT_KEY: u64 = 1 << 32;

/// How to open a [`Wal`].
#[derive(Clone)]
pub struct WalOptions {
    /// The database the WAL belongs to (from the file header); every segment
    /// header carries it.
    pub database_id: u128,
    /// The log position the writer continues at: the end of the scanned log
    /// after its torn tail was cut, or the checkpoint LSN of the image when
    /// the log ends below it.
    pub start_lsn: u64,
    /// When a finished group is synced.
    pub durability: DurabilityMode,
    /// The size a segment grows to before a new group starts a new one.
    pub segment_bytes: u64,
    /// The plaintext payload size records are packed into frames up to; a
    /// larger record gets a frame of its own. [`FRAME_TARGET`] by default;
    /// smaller only in tests that want many frames.
    pub frame_target_bytes: usize,
    /// The cipher of a segment from its salt, for an encrypted database;
    /// `None` writes plaintext.
    pub cipher_for_salt: Option<CipherForSalt>,
    /// The frames a segment holds before the next group starts a new one
    /// (and with it a new segment key). 2^24 by default
    /// (`DEFAULT_FRAMES_PER_SEGMENT_KEY`); smaller only in tests.
    pub frames_per_segment_key: u64,
}

impl WalOptions {
    /// Options for the WAL of `database_id` continuing at `start_lsn`, with
    /// the default durability, 64 MiB segments, 64 KiB frames and no
    /// encryption.
    #[must_use]
    pub fn new(database_id: u128, start_lsn: u64) -> Self {
        Self {
            database_id,
            start_lsn,
            durability: DurabilityMode::default(),
            segment_bytes: DEFAULT_SEGMENT_BYTES,
            frame_target_bytes: FRAME_TARGET,
            cipher_for_salt: None,
            frames_per_segment_key: DEFAULT_FRAMES_PER_SEGMENT_KEY,
        }
    }
}

impl std::fmt::Debug for WalOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalOptions")
            .field("database_id", &format_args!("{:032x}", self.database_id))
            .field("start_lsn", &self.start_lsn)
            .field("durability", &self.durability)
            .field("segment_bytes", &self.segment_bytes)
            .field("frame_target_bytes", &self.frame_target_bytes)
            .field("encrypted", &self.cipher_for_salt.is_some())
            .field("frames_per_segment_key", &self.frames_per_segment_key)
            .finish()
    }
}

/// Where a finished group sits in the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupEnd {
    /// The log position of the group's FIRST frame.
    pub start_lsn: u64,
    /// The log position right after the group's LAST frame.
    pub end_lsn: u64,
}

/// The segment the writer appends to.
struct ActiveSegment {
    /// The open file; `None` once the WAL is closed ([`Wal::close`]).
    file: Option<Arc<File>>,
    path: PathBuf,
    first_lsn: u64,
    cipher: Option<Arc<WalCipher>>,
}

impl ActiveSegment {
    /// The open file.
    ///
    /// # Errors
    ///
    /// [`WalError::Unavailable`] once the WAL is closed.
    fn open_file(&self) -> Result<&Arc<File>, WalError> {
        self.file.as_ref().ok_or_else(closed)
    }
}

/// The error of a write or sync after [`Wal::close`].
fn closed() -> WalError {
    WalError::Unavailable {
        reason: "the WAL is closed".to_string(),
    }
}

/// What the writer's lock guards: the active segment and the end of the log.
struct WriterState {
    segment: ActiveSegment,
    /// The end of the last complete group.
    end_lsn: u64,
    /// The frames written to the segment so far, at most (the frames of
    /// failed groups count, and so does every frame a reopened segment may
    /// hold).
    frames: u64,
}

/// What a sync needs, kept apart from the writer's lock so a sync never
/// waits for a group being written.
struct SyncTarget {
    /// The open file; `None` once the WAL is closed ([`Wal::close`]).
    file: Option<Arc<File>>,
    path: PathBuf,
    /// The end of the last complete group in `file`.
    end_lsn: u64,
}

/// The write-ahead log of one database: segment files of frame groups.
pub struct Wal {
    dir: PathBuf,
    database_id: u128,
    durability: DurabilityMode,
    segment_bytes: u64,
    frame_target_bytes: usize,
    cipher_for_salt: Option<CipherForSalt>,
    state: Mutex<WriterState>,
    published: Mutex<SyncTarget>,
    sync_lock: Mutex<()>,
    end_lsn: AtomicU64,
    synced_lsn: AtomicU64,
    active_first_lsn: AtomicU64,
    records_since_sync: AtomicU64,
    last_sync: Mutex<Instant>,
    frames_per_segment_key: u64,
    /// Successful fsyncs of [`sync_until`](Self::sync_until).
    sync_count: AtomicU64,
    /// Records of the groups finished since the writer opened.
    records_written: AtomicU64,
    /// A sync covered groups and no group that began after it records it
    /// yet: the next chance to write a sync marker takes it.
    marker_due: AtomicBool,
    /// Where the last sync marker starts and ends (both 0 before the first).
    marker_start_lsn: AtomicU64,
    marker_end_lsn: AtomicU64,
    poisoned: AtomicBool,
    poison_reason: Mutex<Option<String>>,
    /// The writer switched to a new segment whose name may not be durable
    /// yet: a panic came between the switch and the directory sync. The
    /// next group or rotation syncs the directory first, so no frame lands
    /// in a segment a power loss could leave under its temporary name.
    directory_sync_due: AtomicBool,
    /// The log position whose crossing calls the waker
    /// ([`arm_wake`](Self::arm_wake)); `u64::MAX` when none is armed. Set
    /// and compared under the writer's lock.
    wake_at: AtomicU64,
    /// What a crossing of `wake_at` calls, after the writer's lock is
    /// released ([`set_waker`](Self::set_waker)).
    waker: Mutex<Option<Waker>>,
}

/// What [`Wal::set_waker`] calls when a group crosses the armed mark.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

impl std::fmt::Debug for Wal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wal")
            .field("dir", &self.dir)
            .field("database_id", &format_args!("{:032x}", self.database_id))
            .field("end_lsn", &self.end_lsn())
            .field("synced_lsn", &self.synced_lsn())
            .field("poisoned", &self.is_poisoned())
            .finish_non_exhaustive()
    }
}

impl Wal {
    /// Opens the WAL in `dir` for writing at `options.start_lsn`.
    ///
    /// When the newest segment ends exactly at the start, the writer appends
    /// to it (after syncing it, so the log positions it reports as durable
    /// are). When there is no segment, or the newest one ends below the
    /// start (the image covers it), a new segment starts there.
    ///
    /// First it removes the files that are never part of the log: segments
    /// a writer was still creating, and segments proven unused (see
    /// [`WalDirectory`](super::WalDirectory)).
    ///
    /// # Errors
    ///
    /// - [`WalError::Misplaced`] when a segment reaches past the start: scan
    ///   the log and cut its torn tail first.
    /// - [`WalError::SegmentHeader`], [`WalError::UnsupportedSegment`],
    ///   [`WalError::ForeignDatabase`], [`WalError::WrongKey`],
    ///   [`WalError::MissingKey`] or
    ///   [`WalError::NotEncrypted`] for a newest segment the writer cannot
    ///   append to.
    /// - [`WalError::Encryption`] for a cipher without the `encryption`
    ///   feature.
    /// - [`WalError::OccupiedSegment`] when a new segment is to start where a
    ///   file that may hold frames already is.
    /// - [`WalError::TemporaryWithFrames`] for a segment still under its
    ///   temporary name that holds frames (damage; the file is kept).
    /// - [`WalError::Io`] when a file or the directory cannot be read,
    ///   created, removed or synced.
    pub fn open(dir: impl AsRef<Path>, options: WalOptions) -> Result<Self, WalError> {
        let dir = dir.as_ref().to_path_buf();
        #[cfg(not(feature = "encryption"))]
        if options.cipher_for_salt.is_some() {
            return Err(WalError::Encryption {
                reason: "an encrypted WAL needs the encryption feature".to_string(),
            });
        }
        ensure_directory(&dir)?;
        let start = options.start_lsn;
        let listing = list_wal_directory(&dir)?;
        remove_dead_files(&dir, &listing)?;
        let appended = match listing.segments.last() {
            Some((first_lsn, path)) => open_newest(path, *first_lsn, &options)?,
            None => None,
        };
        let segment = match appended {
            Some(segment) => segment,
            None => {
                let segment = create_segment(
                    &dir,
                    start,
                    options.database_id,
                    options.cipher_for_salt.as_ref(),
                )
                .map_err(|failure| failure.error)?;
                maybe_crash("wal:after_segment_rename");
                maybe_crash("wal:before_dir_sync");
                sync_directory(&dir)?;
                segment
            }
        };
        let first_lsn = segment.first_lsn;
        // A segment the writer appends to holds at most one frame per frame
        // header of its bytes.
        let frames = (start - first_lsn) / FRAME_HEADER_BYTES as u64;
        Ok(Self {
            published: Mutex::new(SyncTarget {
                file: segment.file.clone(),
                path: segment.path.clone(),
                end_lsn: start,
            }),
            state: Mutex::new(WriterState {
                segment,
                end_lsn: start,
                frames,
            }),
            dir,
            database_id: options.database_id,
            durability: options.durability,
            segment_bytes: options.segment_bytes,
            frame_target_bytes: options.frame_target_bytes.min(MAX_FRAME_RECORD_BYTES),
            cipher_for_salt: options.cipher_for_salt,
            sync_lock: Mutex::new(()),
            end_lsn: AtomicU64::new(start),
            synced_lsn: AtomicU64::new(start),
            active_first_lsn: AtomicU64::new(first_lsn),
            records_since_sync: AtomicU64::new(0),
            last_sync: Mutex::new(Instant::now()),
            frames_per_segment_key: options.frames_per_segment_key.max(1),
            sync_count: AtomicU64::new(0),
            records_written: AtomicU64::new(0),
            marker_due: AtomicBool::new(false),
            marker_start_lsn: AtomicU64::new(0),
            marker_end_lsn: AtomicU64::new(0),
            poisoned: AtomicBool::new(false),
            poison_reason: Mutex::new(None),
            directory_sync_due: AtomicBool::new(false),
            wake_at: AtomicU64::new(u64::MAX),
            waker: Mutex::new(None),
        })
    }

    /// Sets what a group that crosses the armed mark calls, or none. The
    /// waker runs on the thread that wrote the group, after the writer's
    /// lock is released, so it may take locks of its own; it should only
    /// signal.
    pub fn set_waker(&self, waker: Option<Waker>) {
        *self.waker.lock() = waker;
    }

    /// Arms the mark `lsn`: the first group that ends at or past it takes
    /// the crossing and calls the waker, once; later groups call nothing
    /// until a mark is armed again. Returns `true` when the log already
    /// ends at or past `lsn`: the caller then has the crossing, the mark
    /// stays unarmed, and no waker is called for it.
    ///
    /// Armed under the writer's lock, which a group holds while it compares
    /// its end with the mark, so no crossing falls between the two.
    pub fn arm_wake(&self, lsn: u64) -> bool {
        let state = self.state.lock();
        let past = state.end_lsn >= lsn;
        self.wake_at
            .store(if past { u64::MAX } else { lsn }, Ordering::Release);
        past
    }

    /// Disarms the mark: no group calls the waker.
    pub fn disarm_wake(&self) {
        let _state = self.state.lock();
        self.wake_at.store(u64::MAX, Ordering::Release);
    }

    /// Calls the waker for a crossing (see [`arm_wake`](Self::arm_wake)).
    fn wake(&self) {
        let waker = self.waker.lock().clone();
        if let Some(waker) = waker {
            waker();
        }
    }

    /// Begins the group of `transaction_id`. The group holds the writer's
    /// lock until it is finished or dropped, so groups never interleave.
    ///
    /// A segment that has reached the configured size, or holds the frames
    /// one segment key may seal, is rotated first, so the group starts the
    /// next one.
    ///
    /// # Errors
    ///
    /// [`GroupError::Unavailable`] when the writer is poisoned, and
    /// [`GroupError::NotWritten`] when a rotation fails.
    pub fn begin_group(
        &self,
        transaction_id: TransactionId,
    ) -> Result<GroupWriter<'_>, GroupError> {
        self.check_available().map_err(as_unavailable)?;
        let mut state = self.state.lock();
        self.check_available().map_err(as_unavailable)?;
        self.sync_directory_if_due()
            .map_err(GroupError::NotWritten)?;
        let used = state.end_lsn - state.segment.first_lsn;
        let full = (SEGMENT_HEADER_BYTES as u64).saturating_add(used) >= self.segment_bytes
            || state.frames >= self.frames_per_segment_key;
        if used > 0 && full {
            self.rotate_locked(&mut state)
                .map_err(GroupError::NotWritten)?;
        }
        // This group's prologue records every sync so far: no marker is due
        // for them. (Cleared before the prologue reads the synced LSN, so a
        // sync in between asks for a marker again.)
        self.marker_due.store(false, Ordering::Release);
        Ok(self.group_writer(state, transaction_id.as_u64()))
    }

    /// A group of `transaction_id` at the end of the log, holding the
    /// writer's lock, its prologue filled in.
    fn group_writer<'wal>(
        &'wal self,
        state: MutexGuard<'wal, WriterState>,
        transaction_id: u64,
    ) -> GroupWriter<'wal> {
        let start_lsn = state.end_lsn;
        let mut payload = Vec::with_capacity(
            self.frame_target_bytes
                .min(FRAME_TARGET)
                .saturating_add(FRAME_PROLOGUE_BYTES),
        );
        payload.extend_from_slice(&self.synced_lsn().to_le_bytes());
        GroupWriter {
            wal: self,
            state: Some(state),
            transaction_id,
            start_lsn,
            next_lsn: start_lsn,
            payload,
            pending_records: 0,
            frames_written: 0,
            records: 0,
            touched_file: false,
            failure: None,
            marker: false,
            frame: Vec::new(),
            crossed: false,
        }
    }

    /// Starts a new segment at the end of the log, after syncing the active
    /// one, and returns its first LSN: every group before it is in sealed,
    /// synced segments. An active segment without frames is kept, and its
    /// first LSN returned. Waits for a group being written.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::Unavailable`] when the writer is poisoned, the
    /// error of syncing the active segment or the directory (which poisons
    /// the writer) or of creating the new one, and
    /// [`WalError::OccupiedSegment`] when a file that may hold frames has the
    /// new segment's name (the file is left untouched, and the writer stays
    /// on the active segment).
    pub fn rotate(&self) -> Result<u64, WalError> {
        self.check_available()?;
        let mut state = self.state.lock();
        self.check_available()?;
        self.rotate_locked(&mut state)
    }

    /// Makes every complete group durable (fsync). A failure poisons the
    /// writer: a background flusher's failure stops later writes.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::Unavailable`] when the writer is poisoned, and the
    /// error of the sync.
    pub fn sync(&self) -> Result<(), WalError> {
        self.check_available()?;
        self.sync_until(self.end_lsn())
    }

    /// Makes every group up to `lsn` durable, with one fsync for every group
    /// written so far. Returns at once when an earlier sync covered `lsn`. A
    /// failure poisons the writer. `Ok` means that [`synced_lsn`](Self::synced_lsn)
    /// has reached `lsn`.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::BeyondEnd`] for an `lsn` past [`end_lsn`](Self::end_lsn)
    /// (nothing written reaches it, so a committer must not take it for
    /// durable; the writer stays usable), [`WalError::Unavailable`] when the
    /// writer is poisoned, and the error of the sync.
    pub fn sync_until(&self, lsn: u64) -> Result<(), WalError> {
        if self.synced_lsn() >= lsn {
            return Ok(());
        }
        let end_lsn = self.end_lsn();
        if lsn > end_lsn {
            return Err(WalError::BeyondEnd { lsn, end_lsn });
        }
        let _leader = self.sync_lock.lock();
        if self.synced_lsn() >= lsn {
            return Ok(());
        }
        self.check_available()?;
        let (file, path, target) = {
            let published = self.published.lock();
            (
                published.file.clone().ok_or_else(closed)?,
                published.path.clone(),
                published.end_lsn,
            )
        };
        let pending = self.records_since_sync.load(Ordering::Acquire);
        maybe_crash("wal:before_sync");
        if let Err(error) = sync_file(&file, &path) {
            self.poison(format!("syncing the WAL failed: {error}"));
            return Err(error);
        }
        maybe_crash("wal:after_sync");
        let before = self.synced_lsn.fetch_max(target, Ordering::AcqRel);
        self.sync_count.fetch_add(1, Ordering::AcqRel);
        self.records_since_sync.fetch_sub(pending, Ordering::AcqRel);
        *self.last_sync.lock() = Instant::now();
        // The groups this sync covered began before it, so none of them
        // records it: a marker does, unless the sync covered nothing but
        // the last marker.
        let only_the_marker = before >= self.marker_start_lsn.load(Ordering::Acquire)
            && target <= self.marker_end_lsn.load(Ordering::Acquire);
        if target > before && !only_the_marker {
            self.marker_due.store(true, Ordering::Release);
            // A group being written holds the writer: the marker follows
            // its LAST frame (see `GroupWriter::finish`). A segment whose
            // name is not durable yet takes no marker: the next group syncs
            // the directory, and its prologue records this sync.
            if let Some(state) = self.state.try_lock()
                && !self.directory_sync_due.load(Ordering::Acquire)
            {
                self.group_writer(state, TransactionId::SYSTEM.as_u64())
                    .write_marker_if_due();
            }
        }
        Ok(())
    }

    /// How many fsyncs [`sync_until`](Self::sync_until) (and with it
    /// [`sync`](Self::sync) and [`GroupWriter::finish`]) made so far: with a
    /// group commit, fewer than the groups they made durable.
    #[must_use]
    pub fn sync_count(&self) -> u64 {
        self.sync_count.load(Ordering::Acquire)
    }

    /// Closes the WAL: waits for a group in progress, then releases its files,
    /// so the directory can be removed (on Windows an open file keeps it).
    /// From then on every group and sync fails with
    /// [`WalError::Unavailable`]. The caller syncs first.
    pub fn close(&self) {
        let mut state = self.state.lock();
        self.poison("the WAL is closed".to_string());
        state.segment.file = None;
        self.published.lock().file = None;
    }

    /// How many records the groups finished since the writer opened hold
    /// (sync markers hold none).
    #[must_use]
    pub fn records_written(&self) -> u64 {
        self.records_written.load(Ordering::Acquire)
    }

    /// Deletes the sealed segments that end at or below `lsn`, never the
    /// active one, and returns how many it deleted. They go oldest first, so
    /// a crash in between leaves the newest of them, which a scan from `lsn`
    /// skips and the next call deletes.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::Io`] when the directory cannot be listed or a
    /// segment cannot be deleted.
    pub fn remove_segments_before(&self, lsn: u64) -> Result<usize, WalError> {
        let listing = list_wal_directory(&self.dir)?;
        let active = self.active_first_lsn.load(Ordering::Acquire);
        let mut removed = 0;
        for pair in listing.segments.windows(2) {
            let (first_lsn, path) = &pair[0];
            let end_lsn = pair[1].0;
            if end_lsn <= lsn && *first_lsn != active {
                maybe_crash("wal:before_remove_segment");
                std::fs::remove_file(path).map_err(|source| WalError::io(path, source))?;
                maybe_crash("wal:after_remove_segment");
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// The end of the last complete group: where the next group starts.
    #[must_use]
    pub fn end_lsn(&self) -> u64 {
        self.end_lsn.load(Ordering::Acquire)
    }

    /// The end of the last group known to be durable.
    #[must_use]
    pub fn synced_lsn(&self) -> u64 {
        self.synced_lsn.load(Ordering::Acquire)
    }

    /// Whether an earlier failure stopped the writer: it takes no more
    /// groups until the database is reopened.
    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// The failure that poisoned the writer, if any.
    #[must_use]
    pub fn poison_reason(&self) -> Option<String> {
        self.poison_reason.lock().clone()
    }

    /// The WAL directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The database the WAL belongs to.
    #[must_use]
    pub fn database_id(&self) -> u128 {
        self.database_id
    }

    /// The options to scan this WAL from `from_lsn` on: its database and
    /// its key, without salvage.
    #[must_use]
    pub fn scan_options(&self, from_lsn: u64) -> super::ScanOptions {
        super::ScanOptions {
            cipher_for_salt: self.cipher_for_salt.clone(),
            ..super::ScanOptions::new(self.database_id, from_lsn)
        }
    }

    /// Stops the writer for good: the first reason is kept.
    fn poison(&self, reason: String) {
        let mut stored = self.poison_reason.lock();
        if stored.is_none() {
            *stored = Some(reason);
        }
        self.poisoned.store(true, Ordering::Release);
    }

    /// [`WalError::Unavailable`] once the writer is poisoned.
    fn check_available(&self) -> Result<(), WalError> {
        if self.is_poisoned() {
            return Err(WalError::Unavailable {
                reason: self
                    .poison_reason()
                    .unwrap_or_else(|| "an earlier failure".to_string()),
            });
        }
        Ok(())
    }

    /// Whether a group of `records` records that just finished must be
    /// synced now, as the durability mode asks.
    fn sync_due_after(&self, records: u64) -> bool {
        self.records_written.fetch_add(records, Ordering::AcqRel);
        self.records_since_sync.fetch_add(records, Ordering::AcqRel);
        self.sync_due()
    }

    /// Whether the durability mode asks for a sync of the groups written
    /// since the last one: always in `Sync` mode, once a threshold is
    /// reached in `Batch` mode, never in `Adaptive` (its flusher syncs) and
    /// `NoSync`. A commit pipeline asks after
    /// [`GroupWriter::finish_unsynced`], then syncs with
    /// [`sync_until`](Self::sync_until) outside its own commit lock.
    #[must_use]
    pub fn sync_due(&self) -> bool {
        match self.durability {
            DurabilityMode::Sync => true,
            DurabilityMode::Batch {
                max_delay_ms,
                max_records,
            } => {
                self.records_since_sync.load(Ordering::Acquire) >= max_records
                    || self.last_sync.lock().elapsed() >= Duration::from_millis(max_delay_ms)
            }
            DurabilityMode::Adaptive { .. } | DurabilityMode::NoSync => false,
        }
    }

    /// Seals the active segment (sync) and starts a new one at the end of
    /// the log; keeps an active segment without frames.
    fn rotate_locked(&self, state: &mut WriterState) -> Result<u64, WalError> {
        self.sync_directory_if_due()?;
        let end_lsn = state.end_lsn;
        if end_lsn == state.segment.first_lsn {
            return Ok(end_lsn);
        }
        if let Err(error) = sync_file(state.segment.open_file()?, &state.segment.path) {
            self.poison(format!(
                "syncing a WAL segment before rotating failed: {error}"
            ));
            return Err(error);
        }
        self.synced_lsn.fetch_max(end_lsn, Ordering::AcqRel);
        let segment = match create_segment(
            &self.dir,
            end_lsn,
            self.database_id,
            self.cipher_for_salt.as_ref(),
        ) {
            Ok(segment) => segment,
            Err(failure) => {
                if failure.poisons {
                    self.poison(format!(
                        "creating the WAL segment at {end_lsn} failed: {}",
                        failure.error
                    ));
                }
                return Err(failure.error);
            }
        };
        // The segment has its name: switch to it before anything that can
        // fail or panic, so the writer never tries to create it again
        // (#725).
        self.directory_sync_due.store(true, Ordering::Release);
        *self.published.lock() = SyncTarget {
            file: segment.file.clone(),
            path: segment.path.clone(),
            end_lsn,
        };
        self.active_first_lsn.store(end_lsn, Ordering::Release);
        state.segment = segment;
        state.frames = 0;
        maybe_crash("wal:after_segment_rename");
        // Still under the writer's lock: no group lands in the segment
        // before its name is durable.
        self.sync_directory_if_due()?;
        Ok(end_lsn)
    }

    /// Syncs the WAL directory when the writer switched to a segment whose
    /// name is not durable yet (see `directory_sync_due`). Called under the
    /// writer's lock. A failure poisons the writer.
    fn sync_directory_if_due(&self) -> Result<(), WalError> {
        if !self.directory_sync_due.load(Ordering::Acquire) {
            return Ok(());
        }
        maybe_crash("wal:before_dir_sync");
        if let Err(error) = sync_directory(&self.dir) {
            self.poison(format!(
                "syncing the WAL directory after starting a segment failed: {error}"
            ));
            return Err(error);
        }
        self.directory_sync_due.store(false, Ordering::Release);
        Ok(())
    }
}

/// One transaction's group of frames, written as records are pushed.
///
/// Dropping it before [`finish`](Self::finish) abandons the group: the
/// segment is cut back to where it started.
pub struct GroupWriter<'wal> {
    wal: &'wal Wal,
    /// The writer's lock, held until the group is finished or dropped.
    state: Option<MutexGuard<'wal, WriterState>>,
    transaction_id: u64,
    start_lsn: u64,
    /// The log position of the next frame.
    next_lsn: u64,
    /// The plaintext of the frame being filled.
    payload: Vec<u8>,
    /// Records in `payload`.
    pending_records: usize,
    frames_written: usize,
    records: u64,
    /// A frame write was attempted and the group is not complete: abandoning
    /// it needs a cut back.
    touched_file: bool,
    /// The failure that ended the group.
    failure: Option<String>,
    /// The frame being written is a sync marker's: no injected failure is
    /// spent on it, so a test's failure count means the same with and
    /// without markers.
    marker: bool,
    /// Scratch space for a frame's header and payload.
    frame: Vec<u8>,
    /// A group this writer finished crossed the armed mark: the waker runs
    /// once the writer's lock is released.
    crossed: bool,
}

impl GroupWriter<'_> {
    /// Adds one encoded record. Records are packed into frames of the
    /// configured target size, never split; a record larger than that gets a
    /// frame of its own. A full frame is written as soon as the next record
    /// does not fit.
    ///
    /// # Errors
    ///
    /// [`GroupError::NotWritten`] when the record is over
    /// [`MAX_FRAME_RECORD_BYTES`] or a write fails: the group is cut back and
    /// is over.
    pub fn push(&mut self, record: &[u8]) -> Result<(), GroupError> {
        self.check_alive()?;
        if let Err(error) = check_record_length(record.len()) {
            return Err(self.fail(error));
        }
        if self.pending_records > 0
            && self.payload.len().saturating_add(record.len()) > self.wal.frame_target_bytes
        {
            self.write_pending(FrameFlags::MIDDLE)?;
        }
        self.payload.extend_from_slice(record);
        self.pending_records += 1;
        self.records += 1;
        Ok(())
    }

    /// Writes the LAST frame and syncs as the durability mode asks: always in
    /// `Sync` mode, once a threshold is reached in `Batch` mode.
    ///
    /// # Errors
    ///
    /// [`GroupError::NotWritten`] when the LAST frame was not written (the
    /// group is cut back), and [`GroupError::OutcomeUnknown`] when the sync
    /// after it failed (the writer is poisoned).
    pub fn finish(mut self) -> Result<GroupEnd, GroupError> {
        let end = self.write_last()?;
        let sync_due = self.wal.sync_due_after(self.records);
        let wal = self.wal;
        self.write_marker_if_due();
        self.state = None;
        if sync_due {
            wal.sync_until(end.end_lsn)
                .map_err(GroupError::OutcomeUnknown)?;
        }
        Ok(end)
    }

    /// Writes the LAST frame and releases the writer's lock without syncing,
    /// for a commit pipeline that syncs with [`Wal::sync_until`] after
    /// releasing its own commit lock.
    ///
    /// # Errors
    ///
    /// [`GroupError::NotWritten`] when the LAST frame was not written (the
    /// group is cut back).
    pub fn finish_unsynced(mut self) -> Result<GroupEnd, GroupError> {
        let end = self.write_last()?;
        self.wal
            .records_since_sync
            .fetch_add(self.records, Ordering::AcqRel);
        self.wal
            .records_written
            .fetch_add(self.records, Ordering::AcqRel);
        self.write_marker_if_due();
        self.state = None;
        Ok(end)
    }

    /// Writes a sync marker at the end of the log when one is due: an empty
    /// group of the system transaction whose prologue records how far the
    /// log is durable. Called with the writer's lock and no group in
    /// progress (a new writer, or right after a LAST frame). A marker that
    /// cannot be written is cut back and left out: it is evidence for a
    /// later scan, and no commit waits for it.
    fn write_marker_if_due(&mut self) {
        if self.failure.is_some()
            || self.wal.is_poisoned()
            || !self.wal.marker_due.swap(false, Ordering::AcqRel)
        {
            return;
        }
        let start_lsn = self.next_lsn;
        self.marker = true;
        self.transaction_id = TransactionId::SYSTEM.as_u64();
        self.start_lsn = start_lsn;
        self.frames_written = 0;
        self.pending_records = 0;
        self.payload.clear();
        self.payload
            .extend_from_slice(&self.wal.synced_lsn().to_le_bytes());
        if self.write_last().is_ok() {
            self.wal
                .marker_start_lsn
                .store(start_lsn, Ordering::Release);
            self.wal
                .marker_end_lsn
                .store(self.next_lsn, Ordering::Release);
        }
    }

    /// The log position of the group's FIRST frame.
    #[must_use]
    pub fn start_lsn(&self) -> u64 {
        self.start_lsn
    }

    /// [`GroupError::NotWritten`] once the group has failed.
    fn check_alive(&self) -> Result<(), GroupError> {
        match &self.failure {
            Some(reason) => Err(GroupError::NotWritten(WalError::Unavailable {
                reason: format!("this group already failed: {reason}"),
            })),
            None => Ok(()),
        }
    }

    /// Writes the last frame and publishes the group's end.
    fn write_last(&mut self) -> Result<GroupEnd, GroupError> {
        self.check_alive()?;
        self.write_pending(FrameFlags::LAST)?;
        let end_lsn = self.next_lsn;
        let state = self
            .state
            .as_mut()
            .expect("a group being written holds the writer's lock");
        state.end_lsn = end_lsn;
        self.wal.published.lock().end_lsn = end_lsn;
        self.wal.end_lsn.store(end_lsn, Ordering::Release);
        self.touched_file = false;
        // Under the writer's lock, where `arm_wake` sets the mark: the first
        // group at or past it takes the crossing.
        if end_lsn >= self.wal.wake_at.load(Ordering::Acquire) {
            self.wal.wake_at.store(u64::MAX, Ordering::Release);
            self.crossed = true;
        }
        Ok(GroupEnd {
            start_lsn: self.start_lsn,
            end_lsn,
        })
    }

    /// Writes the pending payload as a frame with `flags` (and FIRST when it
    /// is the group's first frame); ends the group on failure.
    fn write_pending(&mut self, flags: FrameFlags) -> Result<(), GroupError> {
        let flags = if self.frames_written == 0 {
            flags.with(FrameFlags::FIRST)
        } else {
            flags
        };
        match self.write_frame(flags) {
            Ok(()) => Ok(()),
            Err(error) => Err(self.fail(error)),
        }
    }

    fn write_frame(&mut self, flags: FrameFlags) -> Result<(), WalError> {
        // A failed sync elsewhere (a flusher, a concurrent commit) poisons the
        // writer while this group is written: no frame, and above all no
        // LAST frame, follows it.
        self.wal.check_available()?;
        let state = self
            .state
            .as_mut()
            .expect("a group being written holds the writer's lock");
        if state.frames >= MAX_FRAMES_PER_SEGMENT_KEY {
            return Err(WalError::Encryption {
                reason: format!(
                    "the WAL segment {} holds {MAX_FRAMES_PER_SEGMENT_KEY} frames, all its key \
                     may seal: the group is too large for one segment",
                    state.segment.path.display()
                ),
            });
        }
        // Counted before the write: a frame that fails used its nonce too.
        state.frames += 1;
        let segment = &state.segment;
        let sealed;
        let (header, body): (FrameHeader, &[u8]) = match &segment.cipher {
            Some(cipher) => {
                let length = frame_length(self.payload.len().saturating_add(SEALED_OVERHEAD))?;
                let header = FrameHeader::new(length, self.next_lsn, self.transaction_id, flags);
                sealed = seal_payload(
                    cipher,
                    &frame_aad(self.wal.database_id, &header),
                    &self.payload,
                )?;
                if u32::try_from(sealed.len()).ok() != Some(length) {
                    return Err(WalError::Encryption {
                        reason: format!(
                            "a sealed frame is {} bytes, expected {length}",
                            sealed.len()
                        ),
                    });
                }
                (header.with_checksum(&sealed), &sealed)
            }
            None => {
                let length = frame_length(self.payload.len())?;
                let header = FrameHeader::new(length, self.next_lsn, self.transaction_id, flags);
                (header.with_checksum(&self.payload), &self.payload)
            }
        };
        let end_lsn = self
            .next_lsn
            .checked_add(header.frame_bytes())
            .ok_or(WalError::LsnOverflow)?;
        self.touched_file = true;
        if !self.marker {
            maybe_fail("wal:write").map_err(|error| WalError::injected(&segment.path, error))?;
        }
        let mut file = &**segment.open_file()?;
        let written = if body.len() <= FRAME_TARGET {
            // One write for a frame of the usual size.
            self.frame.clear();
            self.frame.reserve_exact(FRAME_HEADER_BYTES + body.len());
            self.frame.extend_from_slice(&header.encode());
            self.frame.extend_from_slice(body);
            file.write_all(&self.frame)
        } else {
            // A large record's frame is not copied once more.
            file.write_all(&header.encode())
                .and_then(|()| file.write_all(body))
        };
        written.map_err(|source| WalError::io(&segment.path, source))?;
        maybe_crash("wal:after_frame");
        self.next_lsn = end_lsn;
        self.frames_written += 1;
        self.payload.clear();
        self.pending_records = 0;
        if self.payload.capacity() > 2 * FRAME_TARGET {
            // A large record's frame: do not keep its memory for the group.
            self.payload.shrink_to(FRAME_TARGET);
        }
        Ok(())
    }

    /// Ends the group: cuts the segment back to the group start, poisoning
    /// the writer when that fails.
    fn fail(&mut self, error: WalError) -> GroupError {
        self.failure = Some(error.to_string());
        if self.touched_file
            && let Err(cut) = self.cut_back()
        {
            self.wal
                .poison(format!("cutting a failed group back failed: {cut}"));
        }
        GroupError::NotWritten(error)
    }

    /// Cuts the segment back to where the group started, so the next group
    /// lands there and nothing of this one is ever replayed.
    fn cut_back(&mut self) -> Result<(), WalError> {
        let state = self
            .state
            .as_ref()
            .expect("a group being written holds the writer's lock");
        let segment = &state.segment;
        let offset = (SEGMENT_HEADER_BYTES as u64) + (self.start_lsn - segment.first_lsn);
        maybe_fail("wal:truncate").map_err(|error| WalError::injected(&segment.path, error))?;
        segment
            .open_file()?
            .set_len(offset)
            .map_err(|source| WalError::io(&segment.path, source))?;
        maybe_crash("wal:after_cut_back");
        (&**segment.open_file()?)
            .seek(SeekFrom::Start(offset))
            .map_err(|source| WalError::io(&segment.path, source))?;
        self.touched_file = false;
        self.next_lsn = self.start_lsn;
        Ok(())
    }
}

impl Drop for GroupWriter<'_> {
    fn drop(&mut self) {
        if self.state.is_some()
            && self.touched_file
            && self.failure.is_none()
            && let Err(error) = self.cut_back()
        {
            self.wal
                .poison(format!("cutting an abandoned group back failed: {error}"));
        }
        // The writer's lock goes first: the waker may take locks of its own.
        self.state = None;
        if self.crossed {
            self.wal.wake();
        }
    }
}

/// [`GroupError::Unavailable`] for a poisoned writer.
fn as_unavailable(error: WalError) -> GroupError {
    match error {
        WalError::Unavailable { reason } => GroupError::Unavailable { reason },
        other => GroupError::NotWritten(other),
    }
}

/// Refuses a record that does not fit in a frame of its own.
fn check_record_length(length: usize) -> Result<(), WalError> {
    if length > MAX_FRAME_RECORD_BYTES {
        return Err(WalError::RecordTooLarge {
            length,
            limit: MAX_FRAME_RECORD_BYTES,
        });
    }
    Ok(())
}

/// The length field of a payload of `length` bytes.
fn frame_length(length: usize) -> Result<u32, WalError> {
    u32::try_from(length)
        .ok()
        .filter(|length| *length <= MAX_FRAME_PAYLOAD)
        .ok_or(WalError::RecordTooLarge {
            length,
            limit: MAX_FRAME_PAYLOAD as usize,
        })
}

/// Opens the newest segment for appending when it ends exactly at the
/// start; `None` when it ends below it.
fn open_newest(
    path: &Path,
    first_lsn: u64,
    options: &WalOptions,
) -> Result<Option<ActiveSegment>, WalError> {
    let start = options.start_lsn;
    let cut_first = |end: u64| WalError::Misplaced {
        reason: format!(
            "the WAL segment {} reaches LSN {end}, past LSN {start} where the writer is to \
             continue: scan the log and cut its torn tail first",
            path.display()
        ),
    };
    if first_lsn > start {
        return Err(cut_first(first_lsn));
    }
    let io = |source| WalError::io(path, source);
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(io)?;
    let length = file.metadata().map_err(io)?.len();
    let mut bytes = Vec::with_capacity(SEGMENT_HEADER_BYTES);
    (&mut file)
        .take(SEGMENT_HEADER_BYTES as u64)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    if is_unfinished_segment(path)? {
        return Err(WalError::Misplaced {
            reason: format!(
                "the WAL segment {} has no complete header (it was cut off while it was \
                 created): scan the log and cut its torn tail first",
                path.display()
            ),
        });
    }
    let header = SegmentHeader::decode(&bytes, path)?;
    check_identity(&header, first_lsn, options.database_id, path)?;
    let end = first_lsn
        .checked_add(length - SEGMENT_HEADER_BYTES as u64)
        .ok_or(WalError::LsnOverflow)?;
    if end > start {
        return Err(cut_first(end));
    }
    if end < start {
        return Ok(None);
    }
    let cipher = segment_cipher(
        options.cipher_for_salt.as_ref(),
        &header,
        &stored_key_check_aad(&bytes),
        path,
    )?;
    // What the segment holds is reported as durable from now on.
    sync_file(&file, path)?;
    file.seek(SeekFrom::End(0)).map_err(io)?;
    Ok(Some(ActiveSegment {
        file: Some(Arc::new(file)),
        path: path.to_path_buf(),
        first_lsn,
        cipher,
    }))
}

/// Checks that a segment header names its file's LSN and the database.
pub(crate) fn check_identity(
    header: &SegmentHeader,
    first_lsn: u64,
    database_id: u128,
    path: &Path,
) -> Result<(), WalError> {
    if header.first_lsn != first_lsn {
        return Err(WalError::SegmentHeader {
            path: path.to_path_buf(),
            reason: format!(
                "the header starts at LSN {}, the file name at LSN {first_lsn}",
                header.first_lsn
            ),
        });
    }
    if header.database_id != database_id {
        return Err(WalError::ForeignDatabase {
            path: path.to_path_buf(),
            found: header.database_id,
            expected: database_id,
        });
    }
    Ok(())
}

/// Why creating a segment failed, and whether the writer must stop.
struct CreateFailure {
    error: WalError,
    /// A sync failed.
    poisons: bool,
}

/// Creates the segment starting at `first_lsn` and gives it its name, so
/// that a segment name in the WAL directory always has a synced header
/// (#725): the header is written and synced under the segment's temporary
/// name ([`segment_temp_file_name`]), then the file is renamed to the
/// segment name. A file already there that holds no complete frame (a
/// leftover of a rotation that panicked before 0.6.2) is replaced; one that
/// may hold frames is left untouched, and creating fails with
/// [`WalError::OccupiedSegment`].
///
/// The directory is not synced: the caller switches to the segment first
/// and syncs the directory next ([`sync_directory`]), before any group
/// lands in it. A failure leaves no segment name behind, and removes the
/// temporary file when it can (a writable open removes one left over).
fn create_segment(
    dir: &Path,
    first_lsn: u64,
    database_id: u128,
    cipher_for_salt: Option<&CipherForSalt>,
) -> Result<ActiveSegment, CreateFailure> {
    let temporary = dir.join(segment_temp_file_name(first_lsn));
    let path = dir.join(segment_file_name(first_lsn));
    // A temporary file never holds frames: one a crash left is cut back.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temporary)
        .map_err(|source| CreateFailure {
            error: WalError::io(&temporary, source),
            poisons: false,
        })?;
    maybe_crash("wal:after_segment_create");
    let given_up = |file: File, error: WalError, poisons: bool| {
        drop(file);
        // Best effort: a writable open removes a temporary file left over.
        let _removed: std::io::Result<()> = std::fs::remove_file(&temporary);
        CreateFailure { error, poisons }
    };
    let cipher =
        match write_and_sync_header(&file, &temporary, first_lsn, database_id, cipher_for_salt) {
            Ok(cipher) => cipher,
            Err((error, sync_failed)) => return Err(given_up(file, error, sync_failed)),
        };
    let occupied = match std::fs::symlink_metadata(&path) {
        Ok(_) => segment_may_hold_frames(&path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(WalError::io(&path, source)),
    };
    match occupied {
        Ok(false) => {}
        Ok(true) => {
            return Err(given_up(file, WalError::OccupiedSegment { path }, false));
        }
        Err(error) => return Err(given_up(file, error, false)),
    }
    if let Err(source) = std::fs::rename(&temporary, &path) {
        return Err(given_up(file, WalError::io(&path, source), false));
    }
    Ok(ActiveSegment {
        file: Some(Arc::new(file)),
        path,
        first_lsn,
        cipher,
    })
}

/// Removes the files of `listing` that are never part of the log: segments
/// a writer was still creating, and segments proven unused (see
/// [`WalDirectory`]), then syncs the directory when it removed any. Only a
/// writable open calls it; a read-only scan leaves them.
///
/// # Errors
///
/// [`WalError::TemporaryWithFrames`] for a segment still under its
/// temporary name that holds frames: no writer leaves one (frames go only
/// into a segment with its name), so it is damage, and nothing is removed.
/// [`WalError::Io`] when a file cannot be read or removed, or the directory
/// cannot be synced.
fn remove_dead_files(dir: &Path, listing: &WalDirectory) -> Result<(), WalError> {
    for path in &listing.temporary {
        if segment_may_hold_frames(path)? {
            return Err(WalError::TemporaryWithFrames { path: path.clone() });
        }
    }
    let dead: Vec<&PathBuf> = listing
        .temporary
        .iter()
        .chain(listing.unused.iter())
        .collect();
    for path in &dead {
        std::fs::remove_file(path).map_err(|source| WalError::io(*path, source))?;
    }
    if dead.is_empty() {
        Ok(())
    } else {
        sync_directory(dir)
    }
}

/// Writes and syncs a new segment's header; on failure, the flag says
/// whether it was the sync that failed.
fn write_and_sync_header(
    file: &File,
    path: &Path,
    first_lsn: u64,
    database_id: u128,
    cipher_for_salt: Option<&CipherForSalt>,
) -> Result<Option<Arc<WalCipher>>, (WalError, bool)> {
    let (header, cipher) =
        new_header(first_lsn, database_id, cipher_for_salt).map_err(|error| (error, false))?;
    maybe_fail("wal:write").map_err(|error| (WalError::injected(path, error), false))?;
    let mut writer = file;
    writer
        .write_all(&header.encode())
        .map_err(|source| (WalError::io(path, source), false))?;
    maybe_crash("wal:after_segment_header");
    sync_file(file, path).map_err(|error| (error, true))?;
    Ok(cipher)
}

/// The header of a new segment, with a fresh salt and its key check when
/// the WAL is encrypted.
fn new_header(
    first_lsn: u64,
    database_id: u128,
    cipher_for_salt: Option<&CipherForSalt>,
) -> Result<(SegmentHeader, Option<Arc<WalCipher>>), WalError> {
    let plain = SegmentHeader {
        encrypted: false,
        database_id,
        first_lsn,
        creation_time_ms: now_ms(),
        salt: [0; SALT_BYTES],
        key_check: [0; KEY_CHECK_BYTES],
    };
    match cipher_for_salt {
        #[cfg(feature = "encryption")]
        Some(cipher_for_salt) => {
            let mut header = SegmentHeader {
                encrypted: true,
                salt: new_salt(),
                ..plain
            };
            let cipher = cipher_for_salt(&header.salt);
            header.key_check = key_check(&cipher, &header.key_check_aad())?;
            Ok((header, Some(Arc::new(cipher))))
        }
        #[cfg(not(feature = "encryption"))]
        Some(_) => Err(WalError::Encryption {
            reason: "an encrypted WAL needs the encryption feature".to_string(),
        }),
        None => Ok((plain, None)),
    }
}

/// Syncs a segment's data (fsync).
fn sync_file(file: &File, path: &Path) -> Result<(), WalError> {
    maybe_fail("wal:sync").map_err(|error| WalError::injected(path, error))?;
    file.sync_data()
        .map_err(|source| WalError::io(path, source))
}

/// Makes the entries of `dir` durable. Windows has no directory handles to
/// sync; its directory changes are metadata-journaled.
pub(crate) fn sync_directory(dir: &Path) -> Result<(), WalError> {
    maybe_fail("wal:dir_sync").map_err(|error| WalError::injected(dir, error))?;
    #[cfg(unix)]
    File::open(dir)
        .and_then(|handle| handle.sync_all())
        .map_err(|source| WalError::io(dir, source))?;
    Ok(())
}

/// Creates the WAL directory when it is missing, and makes its entry in the
/// parent durable.
fn ensure_directory(dir: &Path) -> Result<(), WalError> {
    match std::fs::metadata(dir) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => Err(WalError::Misplaced {
            reason: format!("the WAL path {} is not a directory", dir.display()),
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(dir).map_err(|source| WalError::io(dir, source))?;
            match dir.parent() {
                Some(parent) if !parent.as_os_str().is_empty() => sync_directory(parent),
                _ => Ok(()),
            }
        }
        Err(source) => Err(WalError::io(dir, source)),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;

    use super::super::frame::FRAME_HEADER_BYTES;
    use super::*;

    const DATABASE: u128 = 0x0319_1988_0003_0019_0088_0319_1988_0003;

    fn transaction(id: u64) -> TransactionId {
        TransactionId::new(id)
    }

    /// A test record: a length prefix and the bytes, so payloads split back
    /// into records.
    fn record(text: &str) -> Vec<u8> {
        let mut bytes = u32::try_from(text.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend_from_slice(text.as_bytes());
        bytes
    }

    fn records_of(mut payload: &[u8]) -> Vec<String> {
        let mut records = Vec::new();
        while !payload.is_empty() {
            let length = u32::from_le_bytes(payload[..4].try_into().unwrap());
            let length = usize::try_from(length).unwrap();
            records.push(String::from_utf8(payload[4..4 + length].to_vec()).unwrap());
            payload = &payload[4 + length..];
        }
        records
    }

    /// A frame as the files hold it.
    #[derive(Debug)]
    struct RawFrame {
        segment_first_lsn: u64,
        header: FrameHeader,
        payload: Vec<u8>,
    }

    impl RawFrame {
        /// The records of a plaintext frame, without the FIRST prologue.
        fn records(&self) -> Vec<String> {
            let start = if self.header.flags.is_first() {
                FRAME_PROLOGUE_BYTES
            } else {
                0
            };
            records_of(&self.payload[start..])
        }

        fn prologue(&self) -> u64 {
            assert!(self.header.flags.is_first());
            u64::from_le_bytes(self.payload[..8].try_into().unwrap())
        }
    }

    /// Every frame of every segment in `dir`, checking that each segment's
    /// header names its file and that frames sit at their LSN.
    fn raw_frames(dir: &Path) -> Vec<RawFrame> {
        let mut frames = Vec::new();
        for (first_lsn, path) in list_wal_directory(dir).unwrap().segments {
            let bytes = std::fs::read(&path).unwrap();
            let header = SegmentHeader::decode(&bytes, &path).unwrap();
            assert_eq!(header.first_lsn, first_lsn, "{}", path.display());
            assert_eq!(header.database_id, DATABASE);
            let mut offset = SEGMENT_HEADER_BYTES;
            while offset < bytes.len() {
                let frame_header = FrameHeader::decode(
                    bytes[offset..offset + FRAME_HEADER_BYTES]
                        .try_into()
                        .unwrap(),
                )
                .unwrap();
                let start = offset + FRAME_HEADER_BYTES;
                let end = start + usize::try_from(frame_header.length).unwrap();
                let payload = bytes[start..end].to_vec();
                assert!(frame_header.checksum_matches(&payload));
                assert_eq!(
                    frame_header.lsn,
                    first_lsn + u64::try_from(offset - SEGMENT_HEADER_BYTES).unwrap(),
                    "a frame sits at its LSN"
                );
                frames.push(RawFrame {
                    segment_first_lsn: first_lsn,
                    header: frame_header,
                    payload,
                });
                offset = end;
            }
        }
        frames
    }

    /// The groups the frames form: (transaction id, records), checking that
    /// each group is FIRST, middle frames, LAST with one transaction id.
    fn raw_groups(frames: &[RawFrame]) -> Vec<(u64, Vec<String>)> {
        let mut groups: Vec<(u64, Vec<String>)> = Vec::new();
        let mut open = false;
        for frame in frames {
            let flags = frame.header.flags;
            if flags.is_first() {
                assert!(!open, "a group starts inside another: {frame:?}");
                groups.push((frame.header.transaction_id, Vec::new()));
                open = true;
            } else {
                assert!(open, "a frame outside a group: {frame:?}");
            }
            let group = groups.last_mut().unwrap();
            assert_eq!(
                group.0, frame.header.transaction_id,
                "one transaction per group"
            );
            group.1.extend(frame.records());
            if flags.is_last() {
                open = false;
            }
        }
        assert!(!open, "the last group has no LAST frame");
        groups
    }

    fn options(start_lsn: u64) -> WalOptions {
        WalOptions {
            durability: DurabilityMode::NoSync,
            ..WalOptions::new(DATABASE, start_lsn)
        }
    }

    fn write_group(wal: &Wal, id: u64, texts: &[&str]) -> GroupEnd {
        let mut group = wal.begin_group(transaction(id)).unwrap();
        for text in texts {
            group.push(&record(text)).unwrap();
        }
        group.finish().unwrap()
    }

    /// A waker that counts its calls, and whether the writer's lock was
    /// free when it ran.
    fn counting_waker(wal: &Arc<Wal>) -> (Arc<AtomicU64>, Arc<AtomicBool>) {
        let (calls, locked) = (
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        let weak = Arc::downgrade(wal);
        let (count, saw_lock) = (Arc::clone(&calls), Arc::clone(&locked));
        wal.set_waker(Some(Arc::new(move || {
            if let Some(wal) = weak.upgrade()
                && wal.state.try_lock().is_none()
            {
                saw_lock.store(true, Ordering::SeqCst);
            }
            count.fetch_add(1, Ordering::SeqCst);
        })));
        (calls, locked)
    }

    #[test]
    fn the_first_group_past_the_mark_calls_the_waker_once_without_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(Wal::open(dir.path(), options(0)).unwrap());
        let (calls, locked) = counting_waker(&wal);
        let first = write_group(&wal, 3, &["Alix"]);
        assert!(!wal.arm_wake(first.end_lsn + 1));
        write_group(&wal, 4, &["Gus"]);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the crossing");
        write_group(&wal, 5, &["Vincent"]);
        write_group(&wal, 6, &["Jules"]);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "once per arming");
        assert!(
            !locked.load(Ordering::SeqCst),
            "the lock was released first"
        );

        assert!(!wal.arm_wake(wal.end_lsn() + 1));
        write_group(&wal, 7, &["Mia"]);
        assert_eq!(calls.load(Ordering::SeqCst), 2, "armed again");
    }

    #[test]
    fn a_mark_the_log_already_passed_is_taken_by_the_caller() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(Wal::open(dir.path(), options(0)).unwrap());
        let (calls, _) = counting_waker(&wal);
        let end = write_group(&wal, 3, &["Alix"]);
        assert!(wal.arm_wake(end.end_lsn), "the log ends at the mark");
        assert!(wal.arm_wake(1), "and past it");
        write_group(&wal, 4, &["Gus"]);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "the caller took it");
    }

    #[test]
    fn a_disarmed_mark_and_no_waker_call_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(Wal::open(dir.path(), options(0)).unwrap());
        let (calls, _) = counting_waker(&wal);
        assert!(!wal.arm_wake(1));
        wal.disarm_wake();
        write_group(&wal, 3, &["Alix"]);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "disarmed");

        wal.set_waker(None);
        assert!(!wal.arm_wake(wal.end_lsn() + 1));
        write_group(&wal, 4, &["Gus"]);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no waker");
        assert!(wal.arm_wake(wal.end_lsn()), "the crossing was taken");
    }

    /// An abandoned group ends nothing, so it crosses no mark.
    #[test]
    fn an_abandoned_group_crosses_no_mark() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(Wal::open(dir.path(), options(0)).unwrap());
        let (calls, _) = counting_waker(&wal);
        assert!(!wal.arm_wake(1));
        let mut group = wal.begin_group(transaction(3)).unwrap();
        group.push(&record("Alix")).unwrap();
        drop(group);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        write_group(&wal, 4, &["Gus"]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    fn segment_len(dir: &Path, first_lsn: u64) -> u64 {
        std::fs::metadata(dir.join(segment_file_name(first_lsn)))
            .unwrap()
            .len()
    }

    #[test]
    fn a_small_group_is_one_frame_after_its_prologue() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), options(319)).unwrap();
        assert_eq!(wal.end_lsn(), 319);
        let end = write_group(&wal, 3, &["Alix", "Gus"]);
        let frames = raw_frames(dir.path());
        assert_eq!(frames.len(), 1, "{frames:?}");
        let frame = &frames[0];
        assert_eq!(frame.header.flags, FrameFlags::FIRST_AND_LAST);
        assert_eq!(frame.header.lsn, 319);
        assert_eq!(frame.header.transaction_id, 3);
        assert_eq!(frame.prologue(), 319, "nothing was synced past the start");
        assert_eq!(frame.records(), ["Alix", "Gus"]);
        assert_eq!(
            end,
            GroupEnd {
                start_lsn: 319,
                end_lsn: 319 + frame.header.frame_bytes()
            }
        );
        assert_eq!(wal.end_lsn(), end.end_lsn);
        assert_eq!(
            segment_len(dir.path(), 319),
            SEGMENT_HEADER_BYTES as u64 + frame.header.frame_bytes()
        );
    }

    #[test]
    fn groups_never_interleave_under_eight_writers() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(
            dir.path(),
            WalOptions {
                frame_target_bytes: 48,
                segment_bytes: 4096,
                ..options(0)
            },
        )
        .unwrap();
        let barrier = Barrier::new(8);
        std::thread::scope(|scope| {
            for writer in 0..8u64 {
                let (wal, barrier) = (&wal, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for group in 0..19u64 {
                        let id = writer * 1000 + group;
                        let mut frames = wal.begin_group(transaction(id)).unwrap();
                        for index in 0..(3 + group % 7) {
                            frames
                                .push(&record(&format!("Vincent {writer} {group} {index}")))
                                .unwrap();
                            std::thread::yield_now();
                        }
                        frames.finish().unwrap();
                    }
                });
            }
        });
        let frames = raw_frames(dir.path());
        let groups = raw_groups(&frames);
        assert_eq!(groups.len(), 8 * 19);
        let multi_frame = frames
            .iter()
            .filter(|frame| !frame.header.flags.is_last())
            .count();
        assert!(
            multi_frame >= 8 * 19,
            "every group spans several frames: {multi_frame}"
        );
        for (id, records) in &groups {
            let (writer, group) = (id / 1000, id % 1000);
            let expected: Vec<String> = (0..(3 + group % 7))
                .map(|index| format!("Vincent {writer} {group} {index}"))
                .collect();
            assert_eq!(
                records, &expected,
                "group {id} holds only its own records, in order"
            );
        }
        let segments = list_wal_directory(dir.path()).unwrap().segments.len();
        assert!(segments > 3, "the run rotated: {segments} segments");
        assert_eq!(
            wal.end_lsn(),
            frames
                .iter()
                .map(|frame| frame.header.frame_bytes())
                .sum::<u64>()
        );
    }

    #[test]
    fn rotation_happens_only_between_groups() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(
            dir.path(),
            WalOptions {
                frame_target_bytes: 64,
                segment_bytes: 512,
                ..options(88)
            },
        )
        .unwrap();
        for id in 0..19u64 {
            let texts: Vec<String> = (0..=(id % 9)).map(|n| format!("Mia {id} {n}")).collect();
            let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
            write_group(&wal, id, &texts);
        }
        let frames = raw_frames(dir.path());
        raw_groups(&frames);
        let listing = list_wal_directory(dir.path()).unwrap();
        assert!(listing.segments.len() > 3, "{:?}", listing.segments);
        let mut expected_first = 88;
        for (first_lsn, _) in &listing.segments {
            assert_eq!(*first_lsn, expected_first, "the segments form one chain");
            let in_segment: Vec<&RawFrame> = frames
                .iter()
                .filter(|frame| frame.segment_first_lsn == *first_lsn)
                .collect();
            assert!(
                in_segment.first().unwrap().header.flags.is_first(),
                "segment {first_lsn} starts with a FIRST frame"
            );
            assert!(
                in_segment.last().unwrap().header.flags.is_last(),
                "segment {first_lsn} ends after a LAST frame"
            );
            expected_first += in_segment
                .iter()
                .map(|f| f.header.frame_bytes())
                .sum::<u64>();
        }
        assert_eq!(expected_first, wal.end_lsn());
    }

    #[test]
    fn a_large_record_gets_a_frame_of_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), options(0)).unwrap();
        let large = "Jules".repeat(FRAME_TARGET / 5 + 19);
        write_group(&wal, 3, &["Alix", "Gus", "Mia", &large, "Butch", "Paris"]);
        let frames = raw_frames(dir.path());
        let records: Vec<Vec<String>> = frames.iter().map(RawFrame::records).collect();
        assert_eq!(
            records,
            [
                vec!["Alix".to_string(), "Gus".to_string(), "Mia".to_string()],
                vec![large.clone()],
                vec!["Butch".to_string(), "Paris".to_string()],
            ]
        );
        let flags: Vec<FrameFlags> = frames.iter().map(|frame| frame.header.flags).collect();
        assert_eq!(
            flags,
            [FrameFlags::FIRST, FrameFlags::MIDDLE, FrameFlags::LAST]
        );
        // A large first record shares the FIRST frame only with the prologue.
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), options(0)).unwrap();
        write_group(&wal, 19, &[&large, "Berlin"]);
        let records: Vec<Vec<String>> = raw_frames(dir.path())
            .iter()
            .map(RawFrame::records)
            .collect();
        assert_eq!(records, [vec![large], vec!["Berlin".to_string()]]);
    }

    #[test]
    fn records_fill_a_frame_up_to_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(
            dir.path(),
            WalOptions {
                frame_target_bytes: 8 + 3 * 8,
                ..options(0)
            },
        )
        .unwrap();
        // Each record is 8 bytes: the FIRST frame holds the prologue and three.
        write_group(
            &wal,
            3,
            &["Alix", "Gus1", "Mia2", "Jul3", "Vin4", "Ams5", "Ber6"],
        );
        let counts: Vec<usize> = raw_frames(dir.path())
            .iter()
            .map(|frame| frame.records().len())
            .collect();
        assert_eq!(counts, [3, 4], "a later frame has no prologue");
    }

    #[test]
    fn a_record_over_the_frame_limit_is_refused_by_its_length() {
        assert!(check_record_length(MAX_FRAME_RECORD_BYTES).is_ok());
        let error = check_record_length(MAX_FRAME_RECORD_BYTES + 1).unwrap_err();
        assert!(
            matches!(error, WalError::RecordTooLarge { length, limit }
                if length == MAX_FRAME_RECORD_BYTES + 1 && limit == MAX_FRAME_RECORD_BYTES),
            "{error}"
        );
    }

    #[test]
    fn the_prologue_holds_the_durable_lsn_when_the_group_began() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), options(19)).unwrap();
        let first = write_group(&wal, 1, &["Prague"]);
        assert_eq!(wal.synced_lsn(), 19, "NoSync syncs nothing by itself");
        write_group(&wal, 2, &["Berlin"]);
        wal.sync_until(first.end_lsn).unwrap();
        let second_end = wal.synced_lsn();
        let third = write_group(&wal, 3, &["Paris"]);
        assert!(
            third.start_lsn > second_end,
            "the sync's marker sits between the second group and the third"
        );
        let synced: Vec<u64> = raw_frames(dir.path())
            .iter()
            .map(RawFrame::prologue)
            .collect();
        assert_eq!(
            synced,
            [19, 19, second_end, second_end],
            "a sync covers every group before it, and its marker and the next group say so"
        );
        assert_eq!(wal.synced_lsn(), second_end);
    }

    /// What `synced_lsn` reports per durability mode; that the fsync calls
    /// happen is checked through failure injection (`failures` below).
    #[test]
    fn synced_lsn_follows_the_durability_mode() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(
            dir.path(),
            WalOptions {
                durability: DurabilityMode::Sync,
                ..options(0)
            },
        )
        .unwrap();
        let end = write_group(&wal, 1, &["Alix"]);
        assert_eq!(wal.synced_lsn(), end.end_lsn);

        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(
            dir.path(),
            WalOptions {
                durability: DurabilityMode::Batch {
                    max_delay_ms: 3_600_000,
                    max_records: 3,
                },
                ..options(0)
            },
        )
        .unwrap();
        write_group(&wal, 1, &["Alix"]);
        write_group(&wal, 2, &["Gus"]);
        assert_eq!(wal.synced_lsn(), 0, "below the threshold");
        let end = write_group(&wal, 3, &["Mia"]);
        assert_eq!(wal.synced_lsn(), end.end_lsn, "the third record reaches it");
        write_group(&wal, 4, &["Jules"]);
        assert_eq!(wal.synced_lsn(), end.end_lsn, "the count started over");
    }

    #[test]
    fn finish_unsynced_leaves_the_sync_to_the_caller() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(
            dir.path(),
            WalOptions {
                durability: DurabilityMode::Sync,
                ..options(0)
            },
        )
        .unwrap();
        let mut group = wal.begin_group(transaction(3)).unwrap();
        group.push(&record("Amsterdam")).unwrap();
        let end = group.finish_unsynced().unwrap();
        assert_eq!(wal.synced_lsn(), 0);
        // The writer's lock is free: another group can be written meanwhile.
        let later = write_group(&wal, 19, &["Barcelona"]);
        assert_eq!(
            wal.synced_lsn(),
            later.end_lsn,
            "Sync mode synced the later group"
        );
        wal.sync_until(end.end_lsn).unwrap();
        assert_eq!(wal.synced_lsn(), later.end_lsn, "already covered");
    }

    #[test]
    fn rotate_returns_the_end_and_starts_a_segment_there() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), options(3)).unwrap();
        assert_eq!(wal.rotate().unwrap(), 3, "an empty segment is kept");
        assert_eq!(list_wal_directory(dir.path()).unwrap().segments.len(), 1);
        let end = write_group(&wal, 1, &["Gus"]);
        assert_eq!(wal.rotate().unwrap(), end.end_lsn);
        assert_eq!(
            wal.synced_lsn(),
            end.end_lsn,
            "the sealed segment was synced"
        );
        let lsns: Vec<u64> = list_wal_directory(dir.path())
            .unwrap()
            .segments
            .iter()
            .map(|(lsn, _)| *lsn)
            .collect();
        assert_eq!(lsns, [3, end.end_lsn]);
        assert_eq!(
            segment_len(dir.path(), end.end_lsn),
            SEGMENT_HEADER_BYTES as u64,
            "the new segment holds its header"
        );
        let next = write_group(&wal, 2, &["Mia"]);
        assert_eq!(next.start_lsn, end.end_lsn);
        assert_eq!(raw_groups(&raw_frames(dir.path())).len(), 2);
    }

    #[test]
    fn reopening_appends_to_the_segment_that_ends_at_the_start() {
        let dir = tempfile::tempdir().unwrap();
        let end = {
            let wal = Wal::open(dir.path(), options(0)).unwrap();
            write_group(&wal, 1, &["Alix"])
        };
        {
            let wal = Wal::open(dir.path(), options(end.end_lsn)).unwrap();
            assert_eq!(
                wal.synced_lsn(),
                end.end_lsn,
                "the appended segment was synced at open"
            );
            write_group(&wal, 2, &["Gus"]);
        }
        let listing = list_wal_directory(dir.path()).unwrap();
        assert_eq!(
            listing.segments.len(),
            1,
            "appended: {:?}",
            listing.segments
        );
        let groups = raw_groups(&raw_frames(dir.path()));
        assert_eq!(groups.len(), 2);

        let end = raw_frames(dir.path())
            .iter()
            .map(|frame| frame.header.frame_bytes())
            .sum::<u64>();
        let error = Wal::open(dir.path(), options(end - 1)).unwrap_err();
        assert!(
            matches!(error, WalError::Misplaced { .. }) && error.to_string().contains("cut"),
            "a segment reaching past the start: {error}"
        );
        // A start past the end of the log (the image covers it): a new segment.
        let wal = Wal::open(dir.path(), options(end + 88)).unwrap();
        write_group(&wal, 3, &["Mia"]);
        let lsns: Vec<u64> = list_wal_directory(dir.path())
            .unwrap()
            .segments
            .iter()
            .map(|(lsn, _)| *lsn)
            .collect();
        assert_eq!(lsns, [0, end + 88]);
    }

    #[test]
    fn a_segment_of_another_database_is_not_appended_to() {
        let dir = tempfile::tempdir().unwrap();
        let end = {
            let wal = Wal::open(dir.path(), options(0)).unwrap();
            write_group(&wal, 1, &["Alix"])
        };
        let error = Wal::open(dir.path(), WalOptions::new(DATABASE ^ 1, end.end_lsn)).unwrap_err();
        assert!(
            matches!(error, WalError::ForeignDatabase { found, expected, .. }
                if found == DATABASE && expected == DATABASE ^ 1),
            "{error}"
        );
    }

    #[test]
    fn remove_segments_before_keeps_the_active_and_later_segments() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), options(0)).unwrap();
        write_group(&wal, 1, &["Alix"]);
        let second = wal.rotate().unwrap();
        write_group(&wal, 2, &["Gus"]);
        let third = wal.rotate().unwrap();
        write_group(&wal, 3, &["Mia"]);
        std::fs::write(dir.path().join(super::super::BACKUP_CURSOR_FILE), b"88").unwrap();
        assert_eq!(wal.remove_segments_before(second - 1).unwrap(), 0);
        assert_eq!(wal.remove_segments_before(second).unwrap(), 1);
        assert_eq!(
            wal.remove_segments_before(u64::MAX).unwrap(),
            1,
            "the active stays"
        );
        let listing = list_wal_directory(dir.path()).unwrap();
        let lsns: Vec<u64> = listing.segments.iter().map(|(lsn, _)| *lsn).collect();
        assert_eq!(lsns, [third]);
        assert_eq!(listing.others.len(), 1, "other files are never deleted");
        write_group(&wal, 4, &["Jules"]);
    }

    #[cfg(feature = "testing-crash-injection")]
    mod failures {
        use grafeo_common::testing::crash::with_failure_at;

        use super::*;

        fn small_frames(dir: &Path, durability: DurabilityMode) -> Wal {
            Wal::open(
                dir,
                WalOptions {
                    // Every record of the tests below gets a frame of its own.
                    frame_target_bytes: 16,
                    durability,
                    ..options(0)
                },
            )
            .unwrap()
        }

        const FIVE: [&str; 5] = [
            "Alix 0000",
            "Gus 00000",
            "Mia 00000",
            "Jules 000",
            "Butch 000",
        ];

        #[test]
        fn a_failed_write_truncates_back_and_the_next_group_lands_at_the_same_lsn() {
            // The group writes five frames: fail each frame write in turn.
            for failing_write in 1..=5u64 {
                let dir = tempfile::tempdir().unwrap();
                let wal = small_frames(dir.path(), DurabilityMode::NoSync);
                let first = write_group(&wal, 1, &["Amsterdam"]);
                let outcome = with_failure_at(failing_write, || {
                    let mut group = wal.begin_group(transaction(2))?;
                    for text in FIVE {
                        group.push(&record(text))?;
                    }
                    group.finish()
                });
                let error = outcome.unwrap_err();
                assert!(
                    matches!(error, GroupError::NotWritten(_))
                        && error.to_string().contains("injected failure at: wal:write"),
                    "write {failing_write}: {error}"
                );
                assert!(!wal.is_poisoned(), "a cut that worked keeps the writer");
                assert_eq!(wal.end_lsn(), first.end_lsn);
                assert_eq!(
                    segment_len(dir.path(), 0),
                    SEGMENT_HEADER_BYTES as u64 + first.end_lsn,
                    "write {failing_write}: the segment was cut back to the group start"
                );
                let next = write_group(&wal, 3, &["Berlin"]);
                assert_eq!(
                    next.start_lsn, first.end_lsn,
                    "the next group takes its place"
                );
                let groups = raw_groups(&raw_frames(dir.path()));
                assert_eq!(
                    groups,
                    [
                        (1, vec!["Amsterdam".to_string()]),
                        (3, vec!["Berlin".to_string()])
                    ],
                    "write {failing_write}: nothing of the failed group is left"
                );
            }
        }

        #[test]
        fn a_failed_truncation_poisons_the_writer() {
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), DurabilityMode::NoSync);
            let first = write_group(&wal, 1, &["Prague"]);
            // Two frames are written, then the group is abandoned: the third
            // injection point is the cut back, which fails.
            with_failure_at(3, || {
                let mut group = wal.begin_group(transaction(2)).unwrap();
                for text in &FIVE[..3] {
                    group.push(&record(text)).unwrap();
                }
                drop(group);
            });
            assert!(wal.is_poisoned());
            let reason = wal.poison_reason().unwrap();
            assert!(reason.contains("wal:truncate"), "{reason}");
            let error = wal.begin_group(transaction(3)).map(|_| ()).unwrap_err();
            assert!(matches!(error, GroupError::Unavailable { .. }), "{error}");
            assert!(wal.rotate().is_err() && wal.sync().is_err());
            assert_eq!(wal.end_lsn(), first.end_lsn);
            assert!(
                segment_len(dir.path(), 0) > SEGMENT_HEADER_BYTES as u64 + first.end_lsn,
                "the partial group stays as the torn tail"
            );
        }

        #[test]
        fn a_failed_sync_after_last_reports_outcome_unknown() {
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), DurabilityMode::Sync);
            write_group(&wal, 1, &["Barcelona"]);
            // One frame write, then the sync.
            let outcome = with_failure_at(2, || {
                let mut group = wal.begin_group(transaction(2))?;
                group.push(&record("Vincent"))?;
                group.finish()
            });
            let error = outcome.unwrap_err();
            assert!(
                matches!(error, GroupError::OutcomeUnknown(_))
                    && error.to_string().contains("wal:sync"),
                "{error}"
            );
            assert!(wal.is_poisoned());
            let groups = raw_groups(&raw_frames(dir.path()));
            assert_eq!(
                groups.len(),
                3,
                "the first group, its sync marker, and the group whose sync failed are in \
                 the log: the next open decides"
            );
            let error = wal.begin_group(transaction(3)).map(|_| ()).unwrap_err();
            assert!(matches!(error, GroupError::Unavailable { .. }), "{error}");
        }

        /// Whether finishing a one-frame group of `wal` calls fsync: the
        /// group's frame write is injection point 1, so a failure armed at
        /// point 2 fires only when a sync follows it (and leaves the writer
        /// poisoned, so the caller looks at it once).
        fn finishing_a_group_syncs(wal: &Wal, id: u64) -> bool {
            let outcome = with_failure_at(2, || {
                let mut group = wal.begin_group(transaction(id))?;
                group.push(&record("Vincent"))?;
                group.finish()
            });
            match outcome {
                Ok(_) => false,
                Err(GroupError::OutcomeUnknown(error)) => {
                    assert!(error.to_string().contains("wal:sync"), "{error}");
                    true
                }
                Err(other) => panic!("only the sync may fail here: {other}"),
            }
        }

        #[test]
        fn sync_mode_calls_fsync_for_every_group() {
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), DurabilityMode::Sync);
            assert!(finishing_a_group_syncs(&wal, 1), "the first group");
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), DurabilityMode::Sync);
            write_group(&wal, 1, &["Alix"]);
            write_group(&wal, 2, &["Gus"]);
            assert!(finishing_a_group_syncs(&wal, 3), "a later group too");
        }

        #[test]
        fn batch_mode_calls_fsync_only_at_its_record_threshold() {
            let batch = DurabilityMode::Batch {
                max_delay_ms: 3_600_000,
                max_records: 3,
            };
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), batch);
            assert!(!finishing_a_group_syncs(&wal, 1), "one record");
            assert!(!finishing_a_group_syncs(&wal, 2), "two records");
            assert!(finishing_a_group_syncs(&wal, 3), "the third record");
            // After a sync the count starts over.
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), batch);
            for id in 1..=3 {
                write_group(&wal, id, &["Mia"]);
            }
            assert!(!finishing_a_group_syncs(&wal, 4), "the count started over");
            assert!(!finishing_a_group_syncs(&wal, 5));
            assert!(finishing_a_group_syncs(&wal, 6));
        }

        #[test]
        fn batch_mode_calls_fsync_once_its_delay_has_passed() {
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(
                dir.path(),
                DurabilityMode::Batch {
                    max_delay_ms: 0,
                    max_records: u64::MAX,
                },
            );
            assert!(finishing_a_group_syncs(&wal, 1));
        }

        #[test]
        fn no_sync_mode_and_finish_unsynced_call_no_fsync() {
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), DurabilityMode::NoSync);
            for id in 1..=19 {
                assert!(!finishing_a_group_syncs(&wal, id), "group {id}");
            }
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), DurabilityMode::Sync);
            let outcome = with_failure_at(2, || {
                let mut group = wal.begin_group(transaction(3))?;
                group.push(&record("Amsterdam"))?;
                group.finish_unsynced()
            });
            outcome.unwrap();
            assert!(!wal.is_poisoned(), "no sync was attempted");
        }

        #[test]
        fn a_failed_background_sync_poisons_the_writer() {
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), DurabilityMode::NoSync);
            write_group(&wal, 1, &["Gus"]);
            let error = with_failure_at(1, || wal.sync()).unwrap_err();
            assert!(error.to_string().contains("wal:sync"), "{error}");
            assert!(wal.is_poisoned());
            assert_eq!(wal.synced_lsn(), 0, "nothing became durable");
            assert!(wal.begin_group(transaction(2)).is_err());
        }

        #[test]
        fn a_group_open_when_the_writer_is_poisoned_never_gets_its_last_frame() {
            let dir = tempfile::tempdir().unwrap();
            let wal = small_frames(dir.path(), DurabilityMode::NoSync);
            let first = write_group(&wal, 1, &["Amsterdam"]);
            let mut group = wal.begin_group(transaction(2)).unwrap();
            for text in &FIVE[..3] {
                group.push(&record(text)).unwrap();
            }
            // A background sync fails on another thread meanwhile.
            std::thread::scope(|scope| {
                scope
                    .spawn(|| with_failure_at(1, || wal.sync()).unwrap_err())
                    .join()
                    .unwrap();
            });
            assert!(wal.is_poisoned());
            let error = group.finish().unwrap_err();
            assert!(
                matches!(&error, GroupError::NotWritten(WalError::Unavailable { .. })),
                "{error}"
            );
            assert_eq!(
                segment_len(dir.path(), 0),
                SEGMENT_HEADER_BYTES as u64 + first.end_lsn,
                "the frames already written were cut back"
            );
            assert_eq!(raw_groups(&raw_frames(dir.path())).len(), 1);
        }

        #[test]
        fn a_failed_rotation_keeps_the_old_segment_or_poisons() {
            // Rotation: sync the old segment (1), write the new header (2),
            // sync it (3), sync the directory (4).
            for (failing, poisons) in [(1, true), (2, false), (3, true), (4, true)] {
                let dir = tempfile::tempdir().unwrap();
                let wal = small_frames(dir.path(), DurabilityMode::NoSync);
                let end = write_group(&wal, 1, &["Alix"]);
                let error = with_failure_at(failing, || wal.rotate()).unwrap_err();
                assert!(error.to_string().contains("injected"), "{failing}: {error}");
                assert_eq!(wal.is_poisoned(), poisons, "point {failing}: {error}");
                if !poisons {
                    let lsns: Vec<u64> = list_wal_directory(dir.path())
                        .unwrap()
                        .segments
                        .iter()
                        .map(|(lsn, _)| *lsn)
                        .collect();
                    assert_eq!(lsns, [0], "the half-made segment is removed");
                    let next = write_group(&wal, 2, &["Gus"]);
                    assert_eq!(next.start_lsn, end.end_lsn);
                    assert_eq!(wal.rotate().unwrap(), next.end_lsn);
                }
            }
        }
    }

    use super::super::segment::{is_segment_temp_file_name, segment_temp_file_name};

    /// The files of `dir` whose names end in `.tmp`.
    fn temps(dir: &Path) -> Vec<String> {
        let mut found: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| is_segment_temp_file_name(name))
            .collect();
        found.sort();
        found
    }

    /// What a rotation can find at the next segment's name that holds no complete frame: an empty
    /// file, a torn header, a header only, and a header with a torn frame (#725).
    fn without_a_frame(first_lsn: u64) -> Vec<(&'static str, Vec<u8>)> {
        let other = tempfile::tempdir().unwrap();
        let name = segment_file_name(first_lsn);
        let wal = Wal::open(other.path(), options(first_lsn)).unwrap();
        let header = std::fs::read(other.path().join(&name)).unwrap();
        assert_eq!(
            header.len(),
            SEGMENT_HEADER_BYTES,
            "a new segment holds its header"
        );
        write_group(&wal, 9, &["Vincent"]);
        drop(wal);
        let full = std::fs::read(other.path().join(&name)).unwrap();
        vec![
            ("empty", Vec::new()),
            ("a torn header", header[..SEGMENT_HEADER_BYTES / 2].to_vec()),
            ("a header only", header),
            (
                "a header and a torn frame",
                full[..SEGMENT_HEADER_BYTES + FRAME_HEADER_BYTES + 3].to_vec(),
            ),
        ]
    }

    /// A file at the next segment's name with no complete frame is replaced by the rotation's
    /// rename: the rotation succeeds, the segment holds its header alone, and the next group lands
    /// in it (#725).
    #[test]
    fn a_next_segment_without_a_complete_frame_is_replaced() {
        let probe = tempfile::tempdir().unwrap();
        let end = write_group(&Wal::open(probe.path(), options(0)).unwrap(), 1, &["Alix"]).end_lsn;
        for (leftover, bytes) in without_a_frame(end) {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), options(0)).unwrap();
            assert_eq!(write_group(&wal, 1, &["Alix"]).end_lsn, end);
            std::fs::write(dir.path().join(segment_file_name(end)), &bytes).unwrap();
            let rotated = wal
                .rotate()
                .unwrap_or_else(|error| panic!("{leftover}: {error}"));
            assert_eq!(rotated, end, "{leftover}");
            assert_eq!(
                segment_len(dir.path(), end),
                SEGMENT_HEADER_BYTES as u64,
                "{leftover}: the published segment holds its header alone"
            );
            assert!(temps(dir.path()).is_empty(), "{leftover}: no temp left");
            let next = write_group(&wal, 2, &["Gus"]);
            assert_eq!(next.start_lsn, end, "{leftover}");
            assert!(!wal.is_poisoned(), "{leftover}");
            let groups = raw_groups(&raw_frames(dir.path()));
            assert_eq!(groups.len(), 2, "{leftover}: {groups:?}");
        }
    }

    /// A file at the next segment's name that holds a complete frame, or a first frame header that
    /// does not decode, is never replaced: the rotation fails with `OccupiedSegment` naming it, the
    /// file keeps its bytes, nothing is poisoned (groups go on in the old segment), and once the
    /// file is gone the next rotation succeeds (#725).
    #[test]
    fn an_occupied_next_segment_is_refused_and_kept() {
        let probe = tempfile::tempdir().unwrap();
        let end = write_group(&Wal::open(probe.path(), options(0)).unwrap(), 1, &["Alix"]).end_lsn;
        let other = tempfile::tempdir().unwrap();
        {
            let foreign = Wal::open(other.path(), options(end)).unwrap();
            write_group(&foreign, 9, &["Vincent"]);
        }
        let name = segment_file_name(end);
        let with_a_frame = std::fs::read(other.path().join(&name)).unwrap();
        let mut undecodable = with_a_frame[..SEGMENT_HEADER_BYTES].to_vec();
        undecodable.extend(std::iter::repeat_n(0xFF, FRAME_HEADER_BYTES + 8));
        for (occupant, bytes) in [
            ("a complete frame", with_a_frame),
            ("a first frame header that does not decode", undecodable),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), options(0)).unwrap();
            write_group(&wal, 1, &["Alix"]);
            std::fs::write(dir.path().join(&name), &bytes).unwrap();
            let error = wal.rotate().unwrap_err();
            assert!(
                matches!(&error, WalError::OccupiedSegment { path } if path.ends_with(&name)),
                "{occupant}: {error:?}"
            );
            assert!(error.to_string().contains(&name), "{occupant}: {error}");
            assert_eq!(
                std::fs::read(dir.path().join(&name)).unwrap(),
                bytes,
                "{occupant}: the file is untouched"
            );
            assert!(
                !wal.is_poisoned(),
                "{occupant}: nothing was written, nothing poisoned"
            );
            assert!(temps(dir.path()).is_empty(), "{occupant}: no temp left");
            let more = write_group(&wal, 2, &["Gus"]);
            std::fs::remove_file(dir.path().join(&name)).unwrap();
            assert_eq!(
                wal.rotate()
                    .unwrap_or_else(|error| panic!("{occupant}: the retry: {error}")),
                more.end_lsn,
                "{occupant}"
            );
        }
    }

    /// A 0.6.1 leftover (a blank or header-only file whose name lies strictly inside the previous
    /// segment's frame range) is a proven-unused rotation: a writable open reads the log past it
    /// and removes it (#725).
    #[test]
    fn a_leftover_inside_the_previous_segments_frames_is_removed_on_open() {
        for blank in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let (inside, end) = {
                let wal = Wal::open(dir.path(), options(0)).unwrap();
                let first = write_group(&wal, 1, &["Alix"]);
                let second = write_group(&wal, 2, &["Gus"]);
                (first.end_lsn, second.end_lsn)
            };
            let name = segment_file_name(inside);
            let bytes = if blank {
                Vec::new()
            } else {
                let other = tempfile::tempdir().unwrap();
                drop(Wal::open(other.path(), options(inside)).unwrap());
                std::fs::read(other.path().join(&name)).unwrap()
            };
            std::fs::write(dir.path().join(&name), &bytes).unwrap();
            let listing = list_wal_directory(dir.path()).unwrap();
            assert!(
                format!("{:?}", listing.unused).contains(&name),
                "blank {blank}: listed as unused: {:?}",
                listing.unused
            );
            let wal = Wal::open(dir.path(), options(end))
                .unwrap_or_else(|error| panic!("blank {blank}: {error}"));
            assert!(
                !dir.path().join(&name).exists(),
                "blank {blank}: a writable open removes the leftover"
            );
            write_group(&wal, 3, &["Mia"]);
            assert_eq!(
                raw_groups(&raw_frames(dir.path())).len(),
                3,
                "blank {blank}"
            );
        }
    }

    /// A stale temp file (a segment never published) is ignored by the directory listing and
    /// removed by a writable open (#725).
    #[test]
    fn a_stale_temp_is_ignored_and_removed_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let end = write_group(&Wal::open(dir.path(), options(0)).unwrap(), 1, &["Alix"]).end_lsn;
        let temp = segment_temp_file_name(end);
        assert_eq!(temp, format!("{}.tmp", segment_file_name(end)));
        assert!(is_segment_temp_file_name(&temp));
        std::fs::write(dir.path().join(&temp), b"never published").unwrap();
        let listing = list_wal_directory(dir.path()).unwrap();
        assert_eq!(listing.segments.len(), 1, "the temp is no segment");
        assert!(
            format!("{:?}", listing.temporary).contains(&temp),
            "listed as temporary: {:?}",
            listing.temporary
        );
        drop(Wal::open(dir.path(), options(end)).unwrap());
        assert!(temps(dir.path()).is_empty(), "a writable open removes it");
    }

    /// A temp file that holds frames was never published, yet holds data a writer put there: a
    /// writable open refuses it with `TemporaryWithFrames` naming it, and keeps it (#725).
    #[test]
    fn a_temp_with_frames_is_refused_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let end = write_group(&Wal::open(dir.path(), options(0)).unwrap(), 1, &["Alix"]).end_lsn;
        let other = tempfile::tempdir().unwrap();
        {
            let foreign = Wal::open(other.path(), options(end)).unwrap();
            write_group(&foreign, 9, &["Vincent"]);
        }
        let bytes = std::fs::read(other.path().join(segment_file_name(end))).unwrap();
        let temp = segment_temp_file_name(end);
        std::fs::write(dir.path().join(&temp), &bytes).unwrap();
        let error = Wal::open(dir.path(), options(end)).unwrap_err();
        assert!(
            matches!(&error, WalError::TemporaryWithFrames { path } if path.ends_with(&temp)),
            "{error:?}"
        );
        assert_eq!(
            std::fs::read(dir.path().join(&temp)).unwrap(),
            bytes,
            "the temp is kept"
        );
    }

    #[cfg(feature = "testing-crash-injection")]
    mod rotation_panics {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        use grafeo_common::testing::crash::{disable_crash, enable_crash_at};

        use super::*;

        /// Runs `f` with a panic armed at the `count`th crash point: the point it fired at, if any.
        fn panic_at(count: u64, f: impl FnOnce()) -> Option<String> {
            enable_crash_at(count);
            let outcome = catch_unwind(AssertUnwindSafe(f));
            disable_crash();
            outcome.err().map(|payload| {
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_default()
            })
        }

        /// The crash points a rotation after one group reaches, in order, with their counts.
        fn rotation_points() -> Vec<(u64, String)> {
            let mut points = Vec::new();
            for count in 1..=32 {
                let dir = tempfile::tempdir().unwrap();
                let wal = Wal::open(dir.path(), options(0)).unwrap();
                write_group(&wal, 1, &["Alix"]);
                match panic_at(count, || {
                    let _rotated = wal.rotate();
                }) {
                    Some(message) => points.push((count, message)),
                    None => return points,
                }
            }
            panic!("a rotation ran past 32 crash points: {points:?}")
        }

        /// The rotation reaches the ruled points in the ruled order: the temp created, its header
        /// synced, then (after the rename and the switch) the rename point, then the directory sync.
        #[test]
        fn a_rotation_reaches_its_points_in_order() {
            let fired: Vec<String> = rotation_points()
                .into_iter()
                .map(|(_, message)| message)
                .collect();
            let order = [
                "wal:after_segment_create",
                "wal:after_segment_header",
                "wal:after_segment_rename",
                "wal:before_dir_sync",
            ];
            let positions: Vec<usize> = order
                .iter()
                .map(|point| {
                    fired
                        .iter()
                        .position(|message| message.ends_with(point))
                        .unwrap_or_else(|| panic!("never reached {point}: {fired:?}"))
                })
                .collect();
            assert!(positions.is_sorted(), "{fired:?}");
        }

        /// A panic at each crash point of a rotation, with no reopen: before the rename it leaves
        /// a temp file and the writer at its old segment; after it, the writer on the new one.
        /// Either way the log goes on: groups are written, the next rotation succeeds, and a
        /// writable reopen reads every group and removes any temp (#725).
        #[test]
        fn a_panic_while_rotating_leaves_the_log_going_on() {
            for (count, point) in rotation_points() {
                let dir = tempfile::tempdir().unwrap();
                let wal = Wal::open(dir.path(), options(0)).unwrap();
                let first = write_group(&wal, 1, &["Alix"]);
                let fired = panic_at(count, || {
                    let _rotated = wal.rotate();
                });
                assert_eq!(fired.as_deref(), Some(point.as_str()), "{point}");
                let before_rename = point.ends_with("wal:after_segment_create")
                    || point.ends_with("wal:after_segment_header");
                if before_rename {
                    assert_eq!(temps(dir.path()).len(), 1, "{point}: a temp is left");
                    assert_eq!(
                        list_wal_directory(dir.path()).unwrap().segments.len(),
                        1,
                        "{point}: no new segment name"
                    );
                }
                if point.ends_with("wal:after_segment_rename") {
                    let first_point = panic_at(1, || {
                        write_group(&wal, 7, &["Butch"]);
                    });
                    assert!(
                        first_point
                            .as_deref()
                            .is_some_and(|message| message.ends_with("wal:before_dir_sync")),
                        "{point}: the next write syncs the directory before it lands: {first_point:?}"
                    );
                }
                let second = write_group(&wal, 2, &["Gus"]);
                let rotated = wal
                    .rotate()
                    .unwrap_or_else(|error| panic!("{point}: the next rotation: {error}"));
                assert_eq!(rotated, second.end_lsn, "{point}");
                let third = write_group(&wal, 3, &["Mia"]);
                assert!(!wal.is_poisoned(), "{point}");
                assert!(third.start_lsn > first.end_lsn, "{point}");
                drop(wal);
                drop(Wal::open(dir.path(), options(third.end_lsn)).unwrap());
                assert!(
                    temps(dir.path()).is_empty(),
                    "{point}: the reopen removed the temp"
                );
                assert_eq!(raw_groups(&raw_frames(dir.path())).len(), 3, "{point}");
            }
        }
    }
}
