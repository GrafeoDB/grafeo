//! The WAL v2 scanner: complete groups in log order, torn tails cut, damage
//! refused.
//!
//! [`WalScan::open`] checks every segment header first (the magic, version
//! and checksum, the first LSN against the file name and the database id),
//! the key of every segment the scan reads (a segment that ends at or below
//! the checkpoint is skipped, so its key is not checked), then that the
//! segments from the checkpoint on form one chain of log positions.
//! [`WalScan::next_group`] returns a group only once its LAST frame was
//! found, so a caller never applies part of a transaction. It passes over
//! the writer's sync markers (empty groups that only record how far the log
//! was durable), which count as later groups in the table below.
//! [`WalScan::finish`] says where the log ends and what follows it:
//!
//! | Finding | Where | Result |
//! | --- | --- | --- |
//! | A partial frame, zeros or garbage to the end of the file | last segment | torn tail: cut |
//! | FIRST without LAST at the end of the file | last segment | torn tail: cut at the group start |
//! | A bad frame followed by valid groups | last segment | torn when every later group's `synced_lsn` is at or below the bad frame (never synced: out-of-order writeback); otherwise damage |
//! | A bad frame | sealed segment | damage naming the file, offset and LSN |
//! | Checksum right, authentication wrong | any | damage |
//! | A whole frame at the checkpoint LSN that does not start a group | any | damage: the log does not fit the image |
//! | A FIRST frame whose `synced_lsn` lies past its own position | any | damage |
//! | A whole frame (checksum right) with a flag of a later release | any | refused: the WAL needs a newer version, also with salvage |
//! | A segment header of a later version or with an incompatible flag of a later release | any | refused: the WAL needs a newer version, also with salvage |
//! | A segment shorter than its header, or zeros from its first byte to its last | newest segment | cut off while it was created: removed by the cut |
//! | A header of zeros with other bytes behind it | any | refused: the header is damaged (it is synced before any frame) |
//!
//! With `salvage`, damage and gaps end the scan at the last complete group
//! before them instead, and the cut moves everything after it into a
//! directory of its own, `damaged-<lsn>/` (`damaged-<lsn>-<n>/` when an
//! earlier salvage at the same LSN made that one), never deleting or
//! overwriting it.
//!
//! Memory: a group of up to [`GROUP_BUFFER_BYTES`] is kept from the scan, so
//! it is read once; a larger group is checked frame by frame and read again
//! frame by frame when its payloads are asked for. The scan holds one frame
//! plus that buffer.

#![deny(clippy::let_underscore_must_use)]

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::testing::crash::{maybe_crash, maybe_fail};
use grafeo_common::types::TransactionId;

use super::WalCipher;
use super::cipher::{CipherForSalt, frame_aad, open_payload, segment_cipher};
use super::error::WalError;
use super::frame::{
    FRAME_HEADER_BYTES, FRAME_PROLOGUE_BYTES, FrameFlags, FrameHeader, MAX_FRAME_PAYLOAD,
};
use super::segment::{
    SEGMENT_HEADER_BYTES, SegmentHeader, header_is_blank, is_unfinished_segment,
    list_wal_directory, stored_key_check_aad,
};
use super::writer::{check_identity, sync_directory};

/// The size up to which a group's payloads are kept from the scan.
pub const GROUP_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// How to scan a WAL.
#[derive(Clone)]
pub struct ScanOptions {
    /// The database the WAL must belong to.
    pub database_id: u128,
    /// The checkpoint LSN of the image: the scan starts with the frame at
    /// this position, and segments that end at or below it are skipped.
    pub from_lsn: u64,
    /// The cipher of a segment from its salt, for an encrypted database.
    pub cipher_for_salt: Option<CipherForSalt>,
    /// End at the last complete group before damage or a gap instead of
    /// failing; the cut sets the rest aside.
    pub salvage: bool,
    /// The size up to which a group's payloads are kept from the scan;
    /// [`GROUP_BUFFER_BYTES`] by default.
    pub group_buffer_bytes: usize,
}

impl ScanOptions {
    /// Options for the WAL of `database_id` from `from_lsn` on, without a
    /// key and without salvage.
    #[must_use]
    pub fn new(database_id: u128, from_lsn: u64) -> Self {
        Self {
            database_id,
            from_lsn,
            cipher_for_salt: None,
            salvage: false,
            group_buffer_bytes: GROUP_BUFFER_BYTES,
        }
    }
}

impl std::fmt::Debug for ScanOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanOptions")
            .field("database_id", &format_args!("{:032x}", self.database_id))
            .field("from_lsn", &self.from_lsn)
            .field("encrypted", &self.cipher_for_salt.is_some())
            .field("salvage", &self.salvage)
            .field("group_buffer_bytes", &self.group_buffer_bytes)
            .finish()
    }
}

/// What ends the log after its last complete group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailKind {
    /// Bytes that were never durable: a partial group or frame, or an
    /// unsynced hole before unsynced groups. Cut.
    Torn,
    /// Damage or a gap that `salvage` stopped at. Set aside, then cut.
    Damaged,
}

/// The bytes after the last complete group, and what the cut does with
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tail {
    /// Torn or damaged.
    pub kind: TailKind,
    /// The segment cut at `offset`; `None` when the cut lies before the
    /// first segment (a gap right after the checkpoint).
    pub segment: Option<PathBuf>,
    /// The file offset of the cut in `segment`.
    pub offset: u64,
    /// The log position of the cut: the end of the last complete group.
    pub lsn: u64,
    /// What was found there.
    pub reason: String,
    /// Later segments the cut moves aside (salvage only).
    pub set_aside: Vec<PathBuf>,
}

/// Where a scanned log ends.
#[derive(Debug)]
pub struct ScanEnd {
    /// The end of the last complete group: where a writer continues.
    pub end_lsn: u64,
    /// What follows that end, if anything.
    pub tail: Option<Tail>,
    /// The newest segment, when its header was cut off while it was created:
    /// it holds no frame, and the cut removes it.
    pub unfinished_segment: Option<PathBuf>,
    /// Entries of the WAL directory that are neither segments nor known
    /// files. They are reported and never deleted.
    pub unknown_files: Vec<PathBuf>,
    dir: PathBuf,
}

impl ScanEnd {
    /// Cuts what follows the last complete group, for a read-write open: a
    /// torn tail is truncated, salvaged damage is copied and moved into a
    /// new `damaged-<lsn>/` directory first, and an unfinished newest
    /// segment is removed. Each step is synced; a crash in between leaves a
    /// log that scans to the same end again, and bytes set aside stay where
    /// they were put.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::Io`] when a file cannot be truncated, copied,
    /// moved, removed or synced.
    pub fn cut_torn_tail(&self) -> Result<(), WalError> {
        if let Some(tail) = &self.tail {
            if tail.kind == TailKind::Damaged {
                set_aside(&self.dir, tail)?;
            }
            if let Some(segment) = &tail.segment {
                truncate_segment(segment, tail.offset)?;
            }
        }
        if let Some(path) = &self.unfinished_segment {
            maybe_crash("wal:before_remove_unfinished");
            maybe_fail("wal:truncate").map_err(|error| WalError::injected(path, error))?;
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => return Err(WalError::io(path, source)),
            }
            maybe_crash("wal:after_remove_unfinished");
            sync_directory(&self.dir)?;
        }
        Ok(())
    }

    /// Whether the scan found nothing to cut.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.tail.is_none() && self.unfinished_segment.is_none()
    }
}

/// Truncates the segment at `path` to `offset` bytes and syncs it.
fn truncate_segment(path: &Path, offset: u64) -> Result<(), WalError> {
    let io = |source| WalError::io(path, source);
    maybe_crash("wal:before_cut");
    maybe_fail("wal:truncate").map_err(|error| WalError::injected(path, error))?;
    let file = OpenOptions::new().write(true).open(path).map_err(io)?;
    if file.metadata().map_err(io)?.len() > offset {
        file.set_len(offset).map_err(io)?;
    }
    file.sync_all().map_err(io)?;
    maybe_crash("wal:after_cut");
    Ok(())
}

/// Copies the damaged bytes from the cut on, and moves the later segments,
/// into a new directory of the WAL directory ([`new_damaged_dir`]), so what
/// an earlier salvage set aside is never overwritten.
fn set_aside(dir: &Path, tail: &Tail) -> Result<(), WalError> {
    let target = new_damaged_dir(dir, tail.lsn)?;
    maybe_crash("wal:salvage_dir");
    if let Some(segment) = &tail.segment
        && let Some(name) = segment.file_name()
    {
        let io = |source| WalError::io(segment, source);
        let mut source = File::open(segment).map_err(io)?;
        if source.metadata().map_err(io)?.len() > tail.offset {
            source.seek(SeekFrom::Start(tail.offset)).map_err(io)?;
            let copy_path = target.join(name);
            let copy_io = |error| WalError::io(&copy_path, error);
            let mut copy = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&copy_path)
                .map_err(copy_io)?;
            std::io::copy(&mut source, &mut copy).map_err(copy_io)?;
            copy.sync_all().map_err(copy_io)?;
            maybe_crash("wal:salvage_copy");
        }
    }
    for path in &tail.set_aside {
        let Some(name) = path.file_name() else {
            continue;
        };
        match std::fs::rename(path, target.join(name)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => return Err(WalError::io(path, source)),
        }
        maybe_crash("wal:salvage_move");
    }
    sync_directory(&target)?;
    sync_directory(dir)
}

/// Creates the directory a salvage at `lsn` sets its bytes aside in:
/// `damaged-<lsn>`, or `damaged-<lsn>-<n>` with the first free `n` when an
/// earlier salvage at the same LSN (the log rewritten there and damaged
/// again, or a salvage cut short by a crash) made that one.
fn new_damaged_dir(dir: &Path, lsn: u64) -> Result<PathBuf, WalError> {
    let base = format!("damaged-{lsn:020}");
    for attempt in 0..=u32::MAX {
        let target = if attempt == 0 {
            dir.join(&base)
        } else {
            dir.join(format!("{base}-{attempt}"))
        };
        match std::fs::create_dir(&target) {
            Ok(()) => return Ok(target),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(source) => return Err(WalError::io(&target, source)),
        }
    }
    Err(WalError::io(
        dir.join(&base),
        std::io::Error::other("every name for another set-aside directory is taken"),
    ))
}

/// Whether a non-segment entry of a WAL directory is one the WAL knows.
fn is_known_entry(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "backup.cursor" || name.starts_with("damaged-"))
}

/// A segment the scan reads.
struct ScanSegment {
    path: PathBuf,
    first_lsn: u64,
    /// The end of the segment as its length says.
    end_lsn: u64,
    cipher: Option<Arc<WalCipher>>,
}

/// Where the scan stopped.
struct Ending {
    end_lsn: u64,
    tail: Option<Tail>,
}

/// What reading a frame found.
enum FrameRead {
    /// The segment ends here.
    End,
    /// A whole frame; its plaintext payload is in the buffer.
    Frame(FrameHeader),
    /// A frame that is torn or damaged: the caller classifies it.
    Bad(String),
    /// The checksum is right, the authentication is not.
    Unauthentic,
    /// A whole frame (checksum right) with a flag this build does not know.
    Unsupported(String),
}

/// The reason given for a frame whose checksum is right but whose
/// authentication fails.
const AUTHENTICATION_FAILED: &str = "the checksum is right but authentication failed: the frame \
     was moved, duplicated or changed, or written under another key";

/// Scans the groups of a WAL in log order.
pub struct WalScan {
    dir: PathBuf,
    database_id: u128,
    salvage: bool,
    group_buffer_bytes: usize,
    /// The segments from the checkpoint on, in log order.
    segments: Vec<ScanSegment>,
    /// The segments after a gap that salvage stops before, and the gap.
    beyond_gap: Option<(Vec<PathBuf>, String)>,
    index: usize,
    reader: Option<SegmentReader>,
    /// The log position of the next frame.
    position: u64,
    /// Where the scan starts: the checkpoint LSN, where a group must start.
    start_lsn: u64,
    ending: Option<Ending>,
    unfinished_segment: Option<PathBuf>,
    unknown_files: Vec<PathBuf>,
}

impl WalScan {
    /// Opens the WAL in `dir` for scanning from `options.from_lsn`. A
    /// missing directory is an empty log.
    ///
    /// # Errors
    ///
    /// - [`WalError::SegmentHeader`] for a damaged header (a header of zeros
    ///   with other bytes behind it included) or a first LSN that differs
    ///   from the file name. Also with `salvage`: a damaged header is never
    ///   set aside.
    /// - [`WalError::UnsupportedSegment`] for a header of a later version or
    ///   with an unknown incompatible feature, also with `salvage`.
    /// - [`WalError::ForeignDatabase`] for a segment of another database.
    /// - [`WalError::WrongKey`], [`WalError::MissingKey`] or
    ///   [`WalError::NotEncrypted`] when the key does not fit a segment the
    ///   scan reads; a segment that ends at or below the checkpoint is not
    ///   key-checked.
    /// - [`WalError::Gap`] when the segments from the checkpoint on do not
    ///   form one chain (unless `salvage`).
    /// - [`WalError::Io`] when the directory or a segment cannot be read.
    pub fn open(dir: impl AsRef<Path>, options: ScanOptions) -> Result<Self, WalError> {
        let from_lsn = options.from_lsn;
        let mut scan = Self {
            dir: dir.as_ref().to_path_buf(),
            database_id: options.database_id,
            salvage: options.salvage,
            group_buffer_bytes: options.group_buffer_bytes,
            segments: Vec::new(),
            beyond_gap: None,
            index: 0,
            reader: None,
            position: from_lsn,
            start_lsn: from_lsn,
            ending: None,
            unfinished_segment: None,
            unknown_files: Vec::new(),
        };
        let listing = match list_wal_directory(&scan.dir) {
            Ok(listing) => listing,
            Err(WalError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
                return Ok(scan);
            }
            Err(error) => return Err(error),
        };
        scan.unknown_files = listing
            .others
            .into_iter()
            .filter(|path| !is_known_entry(path))
            .collect();
        let newest = listing.segments.len().checked_sub(1);
        let mut used = Vec::new();
        for (index, (first_lsn, path)) in listing.segments.into_iter().enumerate() {
            let (bytes, length) = read_header_bytes(&path)?;
            if header_is_blank(&bytes) {
                if Some(index) != newest {
                    return Err(WalError::SegmentHeader {
                        path,
                        reason: "the header is missing or all zero, and later segments exist"
                            .to_string(),
                    });
                }
                if is_unfinished_segment(&path)? {
                    scan.unfinished_segment = Some(path);
                    continue;
                }
                return Err(WalError::SegmentHeader {
                    path,
                    reason: format!(
                        "the header is all zero, but bytes that are not follow it in the \
                         {length} bytes of the file: a segment's header is synced before its \
                         first frame, so the header is damaged, not cut off while the segment \
                         was created"
                    ),
                });
            }
            let header = SegmentHeader::decode(&bytes, &path)?;
            check_identity(&header, first_lsn, options.database_id, &path)?;
            let end_lsn = first_lsn
                .checked_add(length - SEGMENT_HEADER_BYTES as u64)
                .ok_or(WalError::LsnOverflow)?;
            if end_lsn > from_lsn || first_lsn >= from_lsn {
                let cipher = segment_cipher(
                    options.cipher_for_salt.as_ref(),
                    &header,
                    &stored_key_check_aad(&bytes),
                    &path,
                )?;
                used.push(ScanSegment {
                    path,
                    first_lsn,
                    end_lsn,
                    cipher,
                });
            }
        }
        if let Some((keep, reason)) = find_gap(&used, from_lsn) {
            if !options.salvage {
                return Err(WalError::Gap { reason });
            }
            let beyond = used.drain(keep..).map(|segment| segment.path).collect();
            scan.beyond_gap = Some((beyond, reason));
        }
        if let Some(first) = used.first() {
            scan.position = from_lsn.max(first.first_lsn);
        }
        scan.start_lsn = scan.position;
        scan.segments = used;
        Ok(scan)
    }

    /// The next complete group, or `None` at the end of the log (see
    /// [`finish`](Self::finish) for what follows it).
    ///
    /// # Errors
    ///
    /// [`WalError::Damaged`] for damage (unless `salvage`),
    /// [`WalError::UnsupportedFrame`] for a whole frame of a later release
    /// (also with `salvage`), and [`WalError::Io`] when a segment cannot be
    /// read.
    pub fn next_group(&mut self) -> Result<Option<GroupFrames>, WalError> {
        loop {
            if self.ending.is_some() {
                return Ok(None);
            }
            if !self.ensure_reader()? {
                self.stop_at_end();
                return Ok(None);
            }
            let group_start = self.position;
            let mut payload = Vec::new();
            match self.read_frame(&mut payload)? {
                FrameRead::End => {
                    // The chain was checked at open: the next segment starts
                    // where this one ends.
                    self.index += 1;
                    self.reader = None;
                }
                FrameRead::Frame(header) if header.flags.is_first() => {
                    // A sync marker: one frame holding only its prologue,
                    // which says how far the log was durable. It is checked
                    // and passed over like a group, and returned to nobody.
                    let marker = header.flags.is_last() && payload.len() == FRAME_PROLOGUE_BYTES;
                    match self.read_group(header, payload)? {
                        Some(_) if marker => {}
                        group => return Ok(group),
                    }
                }
                FrameRead::Frame(header) if group_start == self.start_lsn => {
                    // A whole frame, so never a torn write: the image ends
                    // inside a group of this log, or the log is not the
                    // image's.
                    return self.damage(
                        group_start,
                        group_start,
                        format!(
                            "the checkpoint at LSN {group_start} lands on a frame of transaction \
                             {} that does not start a group: the log does not fit the image",
                            header.transaction_id
                        ),
                    );
                }
                FrameRead::Frame(header) => {
                    return self.bad_frame(
                        group_start,
                        group_start,
                        format!(
                            "a frame of transaction {} that does not start a group sits where a \
                             group must start",
                            header.transaction_id
                        ),
                    );
                }
                FrameRead::Bad(reason) => return self.bad_frame(group_start, group_start, reason),
                FrameRead::Unauthentic => {
                    return self.damage(
                        group_start,
                        group_start,
                        AUTHENTICATION_FAILED.to_string(),
                    );
                }
                FrameRead::Unsupported(reason) => {
                    return Err(self.unsupported(group_start, reason));
                }
            }
        }
    }

    /// The error for a whole frame at `lsn` in the current segment that sets
    /// a flag of a later release.
    fn unsupported(&self, lsn: u64, reason: String) -> WalError {
        WalError::UnsupportedFrame {
            path: self.segments[self.index].path.clone(),
            offset: self.offset_of(lsn),
            lsn,
            reason,
        }
    }

    /// Scans to the end of the log and says where it ends.
    ///
    /// # Errors
    ///
    /// As [`next_group`](Self::next_group).
    pub fn finish(mut self) -> Result<ScanEnd, WalError> {
        while self.next_group()?.is_some() {}
        let ending = self.ending.take().unwrap_or(Ending {
            end_lsn: self.position,
            tail: None,
        });
        Ok(ScanEnd {
            end_lsn: ending.end_lsn,
            tail: ending.tail,
            unfinished_segment: self.unfinished_segment.take(),
            unknown_files: std::mem::take(&mut self.unknown_files),
            dir: self.dir,
        })
    }

    /// Opens the reader of the current segment at the current position;
    /// `false` past the last segment.
    fn ensure_reader(&mut self) -> Result<bool, WalError> {
        if self.reader.is_some() {
            return Ok(true);
        }
        let Some(segment) = self.segments.get(self.index) else {
            return Ok(false);
        };
        let offset = SEGMENT_HEADER_BYTES as u64 + (self.position - segment.first_lsn);
        self.reader = Some(SegmentReader::open(&segment.path, offset)?);
        Ok(true)
    }

    /// The file offset of `lsn` in the current segment.
    fn offset_of(&self, lsn: u64) -> u64 {
        SEGMENT_HEADER_BYTES as u64 + (lsn - self.segments[self.index].first_lsn)
    }

    fn read_frame(&mut self, payload: &mut Vec<u8>) -> Result<FrameRead, WalError> {
        let segment = &self.segments[self.index];
        let reader = self
            .reader
            .as_mut()
            .expect("a frame is read from an open segment");
        read_frame_from(
            reader,
            &segment.path,
            self.position,
            segment.cipher.as_deref(),
            self.database_id,
            payload,
        )
    }

    /// Reads the rest of the group whose FIRST frame was just read.
    fn read_group(
        &mut self,
        first: FrameHeader,
        first_payload: Vec<u8>,
    ) -> Result<Option<GroupFrames>, WalError> {
        let group_start = self.position;
        let group_offset = self.offset_of(group_start);
        let Some(synced_lsn) = prologue(&first_payload) else {
            return self.bad_frame(
                group_start,
                group_start,
                "the first frame of the group has no prologue".to_string(),
            );
        };
        if synced_lsn > group_start {
            // A whole frame that says the log was durable past the start of
            // its own group, which no writer can have known when it began.
            return self.damage(
                group_start,
                group_start,
                format!(
                    "the first frame of the group says the log was synced up to LSN \
                     {synced_lsn}, past the frame's own position"
                ),
            );
        }
        self.position = advance(self.position, &first)?;
        let mut buffered_bytes = first_payload.len();
        let mut large = buffered_bytes > self.group_buffer_bytes;
        let mut buffered = if large {
            Vec::new()
        } else {
            vec![first_payload]
        };
        let mut frame_count = 1;
        let mut last = first;
        while !last.flags.is_last() {
            let frame_lsn = self.position;
            let mut payload = Vec::new();
            let header = match self.read_frame(&mut payload)? {
                FrameRead::Frame(header) => header,
                FrameRead::End => {
                    return self.bad_frame(
                        group_start,
                        frame_lsn,
                        format!(
                            "the segment ends inside the group that starts at LSN {group_start}"
                        ),
                    );
                }
                FrameRead::Bad(reason) => return self.bad_frame(group_start, frame_lsn, reason),
                FrameRead::Unauthentic => {
                    return self.damage(group_start, frame_lsn, AUTHENTICATION_FAILED.to_string());
                }
                FrameRead::Unsupported(reason) => {
                    return Err(self.unsupported(frame_lsn, reason));
                }
            };
            if header.flags.is_first() {
                return self.bad_frame(
                    group_start,
                    frame_lsn,
                    format!("a group starts inside the group that starts at LSN {group_start}"),
                );
            }
            if header.transaction_id != first.transaction_id {
                return self.bad_frame(
                    group_start,
                    frame_lsn,
                    format!(
                        "a frame of transaction {} inside the group of transaction {}",
                        header.transaction_id, first.transaction_id
                    ),
                );
            }
            self.position = advance(self.position, &header)?;
            frame_count += 1;
            if !large {
                buffered_bytes = buffered_bytes.saturating_add(payload.len());
                if buffered_bytes > self.group_buffer_bytes {
                    large = true;
                    buffered = Vec::new();
                } else {
                    buffered.push(payload);
                }
            }
            last = header;
        }
        let source = if large {
            let segment = &self.segments[self.index];
            GroupSource::Reread(Reread {
                path: segment.path.clone(),
                offset: group_offset,
                lsn: group_start,
                cipher: segment.cipher.clone(),
                database_id: self.database_id,
                reader: None,
            })
        } else {
            GroupSource::Buffered(buffered.into_iter())
        };
        Ok(Some(GroupFrames {
            transaction_id: first.transaction_id,
            start_lsn: group_start,
            end_lsn: self.position,
            synced_lsn,
            frame_count,
            returned: 0,
            source,
            current: Vec::new(),
        }))
    }

    /// A bad frame at `bad_lsn` in the group starting at `group_start`: a
    /// torn tail in the last segment unless a later group was written after
    /// it was synced, damage otherwise.
    fn bad_frame(
        &mut self,
        group_start: u64,
        bad_lsn: u64,
        reason: String,
    ) -> Result<Option<GroupFrames>, WalError> {
        let last_segment = self.index + 1 == self.segments.len() && self.beyond_gap.is_none();
        if !last_segment {
            return self.damage(group_start, bad_lsn, reason);
        }
        match self.later_synced_group(bad_lsn)? {
            None => {
                self.stop_with_tail(TailKind::Torn, group_start, reason, Vec::new());
                Ok(None)
            }
            Some((later_lsn, synced_lsn)) => self.damage(
                group_start,
                bad_lsn,
                format!(
                    "{reason}; the group at LSN {later_lsn} began after the log was synced up to \
                     LSN {synced_lsn}, so these bytes were durable"
                ),
            ),
        }
    }

    /// Damage at `bad_lsn`: an error, or with salvage the end of the scan at
    /// `group_start` with the rest set aside.
    fn damage(
        &mut self,
        group_start: u64,
        bad_lsn: u64,
        reason: String,
    ) -> Result<Option<GroupFrames>, WalError> {
        let bad_offset = self.offset_of(bad_lsn);
        if !self.salvage {
            return Err(WalError::Damaged {
                path: self.segments[self.index].path.clone(),
                offset: bad_offset,
                lsn: bad_lsn,
                reason,
            });
        }
        let mut set_aside: Vec<PathBuf> = self.segments[self.index + 1..]
            .iter()
            .map(|segment| segment.path.clone())
            .collect();
        if let Some((beyond, _)) = self.beyond_gap.take() {
            set_aside.extend(beyond);
        }
        self.stop_with_tail(
            TailKind::Damaged,
            group_start,
            format!("offset {bad_offset} (LSN {bad_lsn}): {reason}"),
            set_aside,
        );
        Ok(None)
    }

    /// Ends the scan at `group_start` in the current segment.
    fn stop_with_tail(
        &mut self,
        kind: TailKind,
        group_start: u64,
        reason: String,
        set_aside: Vec<PathBuf>,
    ) {
        let offset = self.offset_of(group_start);
        self.reader = None;
        self.ending = Some(Ending {
            end_lsn: group_start,
            tail: Some(Tail {
                kind,
                segment: Some(self.segments[self.index].path.clone()),
                offset,
                lsn: group_start,
                reason,
                set_aside,
            }),
        });
    }

    /// Ends the scan after the last segment, at a gap when salvage stopped
    /// before one.
    fn stop_at_end(&mut self) {
        let tail = self.beyond_gap.take().map(|(set_aside, reason)| {
            let (segment, offset) = match self.segments.last() {
                Some(segment) => (
                    Some(segment.path.clone()),
                    SEGMENT_HEADER_BYTES as u64 + (self.position - segment.first_lsn),
                ),
                None => (None, 0),
            };
            Tail {
                kind: TailKind::Damaged,
                segment,
                offset,
                lsn: self.position,
                reason,
                set_aside,
            }
        });
        self.ending = Some(Ending {
            end_lsn: self.position,
            tail,
        });
    }

    /// Looks past a bad frame at `bad_lsn` in the last segment for a valid
    /// group whose prologue says the log was synced beyond it: (that group's
    /// LSN, its synced LSN). Reads the rest of the segment a window at a
    /// time.
    ///
    /// A whole frame there with a flag of a later release is refused
    /// ([`WalError::UnsupportedFrame`]): what it says about the sync cannot
    /// be read, and cutting it as a torn tail would drop the later release's
    /// groups.
    fn later_synced_group(&self, bad_lsn: u64) -> Result<Option<(u64, u64)>, WalError> {
        const WINDOW: usize = 1 << 20;
        let segment = &self.segments[self.index];
        let path = &segment.path;
        let io = |source| WalError::io(path, source);
        let mut file = File::open(path).map_err(io)?;
        let length = file.metadata().map_err(io)?.len();
        let header_bytes = FRAME_HEADER_BYTES as u64;
        let mut window = vec![0u8; WINDOW + FRAME_HEADER_BYTES];
        let mut payload = Vec::new();
        let mut candidate = self.offset_of(bad_lsn) + 1;
        while candidate + header_bytes <= length {
            let available = read_at(&mut file, candidate, &mut window).map_err(io)?;
            let mut examined = 0;
            let mut next = None;
            while examined < WINDOW && examined + FRAME_HEADER_BYTES <= available {
                let at = candidate + examined as u64;
                let lsn = segment.first_lsn + (at - SEGMENT_HEADER_BYTES as u64);
                let bytes: &[u8; FRAME_HEADER_BYTES] = window
                    [examined..examined + FRAME_HEADER_BYTES]
                    .try_into()
                    .expect("the slice has the length of a frame header");
                examined += 1;
                // A FIRST frame carries a prologue; a frame with a flag of a
                // later release is looked at too, to refuse it when whole.
                if bytes[8..16] != lsn.to_le_bytes()
                    || FrameFlags::from_bits(bytes[24]).is_some_and(|flags| !flags.is_first())
                {
                    continue;
                }
                let declared = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                if declared > MAX_FRAME_PAYLOAD || at + header_bytes + u64::from(declared) > length
                {
                    continue;
                }
                read_payload_at(&mut file, at + header_bytes, declared, &mut payload)
                    .map_err(io)?;
                if !FrameHeader::stored_checksum_matches(bytes, &payload) {
                    continue;
                }
                let header = match FrameHeader::decode(bytes) {
                    Ok(header) => header,
                    Err(reason) => {
                        return Err(WalError::UnsupportedFrame {
                            path: path.clone(),
                            offset: at,
                            lsn,
                            reason,
                        });
                    }
                };
                let synced = match &segment.cipher {
                    Some(cipher) => {
                        open_payload(cipher, &frame_aad(self.database_id, &header), &payload)
                            .and_then(|plain| prologue(&plain))
                    }
                    None => prologue(&payload),
                };
                let Some(synced) = synced else {
                    continue;
                };
                if synced > bad_lsn {
                    return Ok(Some((lsn, synced)));
                }
                next = Some(at + header.frame_bytes());
                break;
            }
            candidate = next.unwrap_or(candidate + examined as u64);
        }
        Ok(None)
    }
}

/// The first gap in the chain of `segments` from `from_lsn` on: how many
/// segments come before it, and what it is.
fn find_gap(segments: &[ScanSegment], from_lsn: u64) -> Option<(usize, String)> {
    let first = segments.first()?;
    if first.first_lsn > from_lsn {
        return Some((
            0,
            format!(
                "the log after the checkpoint at LSN {from_lsn} is missing: the first segment {} \
                 starts at LSN {}",
                first.path.display(),
                first.first_lsn
            ),
        ));
    }
    segments.windows(2).enumerate().find_map(|(index, pair)| {
        (pair[0].end_lsn != pair[1].first_lsn).then(|| {
            (
                index + 1,
                format!(
                    "the segment {} ends at LSN {}, but the next segment {} starts at LSN {}",
                    pair[0].path.display(),
                    pair[0].end_lsn,
                    pair[1].path.display(),
                    pair[1].first_lsn
                ),
            )
        })
    })
}

/// The log position after `header`'s frame.
fn advance(position: u64, header: &FrameHeader) -> Result<u64, WalError> {
    position
        .checked_add(header.frame_bytes())
        .ok_or(WalError::LsnOverflow)
}

/// The synced LSN at the start of a FIRST frame's plaintext.
fn prologue(payload: &[u8]) -> Option<u64> {
    let bytes: [u8; FRAME_PROLOGUE_BYTES] = payload.get(..FRAME_PROLOGUE_BYTES)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

/// The first bytes of the file at `path` (up to a header) and its length.
fn read_header_bytes(path: &Path) -> Result<(Vec<u8>, u64), WalError> {
    let io = |source| WalError::io(path, source);
    let file = File::open(path).map_err(io)?;
    let length = file.metadata().map_err(io)?.len();
    let mut bytes = Vec::with_capacity(SEGMENT_HEADER_BYTES);
    file.take(SEGMENT_HEADER_BYTES as u64)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    Ok((bytes, length))
}

/// Reads from `offset` until `buffer` is full or the file ends; returns how
/// many bytes were read.
fn read_at(file: &mut File, offset: u64, buffer: &mut [u8]) -> std::io::Result<usize> {
    file.seek(SeekFrom::Start(offset))?;
    let mut filled = 0;
    while filled < buffer.len() {
        match file.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

/// Reads the `length` bytes at `offset` into `payload`; the caller checked
/// that they lie within the file.
fn read_payload_at(
    file: &mut File,
    offset: u64,
    length: u32,
    payload: &mut Vec<u8>,
) -> std::io::Result<()> {
    let length = usize::try_from(length)
        .map_err(|_| std::io::Error::other("a frame length does not fit in memory"))?;
    payload.clear();
    payload.reserve_exact(length);
    payload.resize(length, 0);
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(payload)
}

/// Sequential reads through one segment.
struct SegmentReader {
    reader: BufReader<File>,
    /// Bytes left after the reader's position.
    remaining: u64,
}

impl SegmentReader {
    fn open(path: &Path, offset: u64) -> Result<Self, WalError> {
        let io = |source| WalError::io(path, source);
        let mut file = File::open(path).map_err(io)?;
        let length = file.metadata().map_err(io)?.len();
        file.seek(SeekFrom::Start(offset)).map_err(io)?;
        Ok(Self {
            reader: BufReader::with_capacity(64 * 1024, file),
            remaining: length.saturating_sub(offset),
        })
    }

    fn read_exact(&mut self, buffer: &mut [u8], path: &Path) -> Result<(), WalError> {
        self.reader
            .read_exact(buffer)
            .map_err(|source| WalError::io(path, source))?;
        self.remaining = self.remaining.saturating_sub(buffer.len() as u64);
        Ok(())
    }
}

/// Reads the frame at `lsn` from `reader`: its header is checked (length
/// bound, LSN) before anything is allocated, the payload only once it lies
/// within the segment, then the checksum, then the flags, then the
/// authentication. The flags come after the checksum: a whole frame with a
/// flag of a later release is refused, while a frame whose flags byte is
/// wrong because it is torn or damaged is classified as any bad frame.
fn read_frame_from(
    reader: &mut SegmentReader,
    path: &Path,
    lsn: u64,
    cipher: Option<&WalCipher>,
    database_id: u128,
    payload: &mut Vec<u8>,
) -> Result<FrameRead, WalError> {
    if reader.remaining == 0 {
        return Ok(FrameRead::End);
    }
    if reader.remaining < FRAME_HEADER_BYTES as u64 {
        return Ok(FrameRead::Bad(format!(
            "a partial frame header: {} bytes remain in the segment",
            reader.remaining
        )));
    }
    let mut bytes = [0u8; FRAME_HEADER_BYTES];
    reader.read_exact(&mut bytes, path)?;
    let declared = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if declared > MAX_FRAME_PAYLOAD {
        return Ok(FrameRead::Bad(format!(
            "the frame declares a payload of {declared} bytes, over the limit of \
             {MAX_FRAME_PAYLOAD}"
        )));
    }
    let mut stored_lsn = [0u8; 8];
    stored_lsn.copy_from_slice(&bytes[8..16]);
    let stored_lsn = u64::from_le_bytes(stored_lsn);
    if stored_lsn != lsn {
        return Ok(FrameRead::Bad(format!(
            "the frame header says LSN {stored_lsn}, the frame sits at LSN {lsn}"
        )));
    }
    if u64::from(declared) > reader.remaining {
        return Ok(FrameRead::Bad(format!(
            "the frame declares a payload of {declared} bytes, but {} remain in the segment",
            reader.remaining
        )));
    }
    let length = usize::try_from(declared).map_err(|_| WalError::Damaged {
        path: path.to_path_buf(),
        offset: 0,
        lsn,
        reason: "the frame length does not fit in memory".to_string(),
    })?;
    payload.clear();
    payload.reserve_exact(length);
    payload.resize(length, 0);
    reader.read_exact(payload, path)?;
    if !FrameHeader::stored_checksum_matches(&bytes, payload) {
        return Ok(FrameRead::Bad("checksum mismatch".to_string()));
    }
    let header = match FrameHeader::decode(&bytes) {
        Ok(header) => header,
        Err(reason) => return Ok(FrameRead::Unsupported(reason)),
    };
    if let Some(cipher) = cipher {
        match open_payload(cipher, &frame_aad(database_id, &header), payload) {
            Some(plaintext) => *payload = plaintext,
            None => return Ok(FrameRead::Unauthentic),
        }
    }
    Ok(FrameRead::Frame(header))
}

/// Where a group's payloads come from.
enum GroupSource {
    /// Kept from the scan.
    Buffered(std::vec::IntoIter<Vec<u8>>),
    /// Read again from the segment, one frame at a time.
    Reread(Reread),
}

/// A large group read again from its segment.
struct Reread {
    path: PathBuf,
    /// The file offset of the next frame.
    offset: u64,
    /// The log position of the next frame.
    lsn: u64,
    cipher: Option<Arc<WalCipher>>,
    database_id: u128,
    reader: Option<SegmentReader>,
}

impl Reread {
    /// Reads the next frame of the group into `payload`, checking that it is
    /// still what the scan found.
    fn next_frame(
        &mut self,
        transaction_id: u64,
        first: bool,
        last: bool,
        payload: &mut Vec<u8>,
    ) -> Result<(), WalError> {
        if self.reader.is_none() {
            self.reader = Some(SegmentReader::open(&self.path, self.offset)?);
        }
        let reader = self.reader.as_mut().expect("the reader was just opened");
        let changed = |reason: String| WalError::Damaged {
            path: self.path.clone(),
            offset: self.offset,
            lsn: self.lsn,
            reason: format!("the group changed since it was scanned: {reason}"),
        };
        let header = match read_frame_from(
            reader,
            &self.path,
            self.lsn,
            self.cipher.as_deref(),
            self.database_id,
            payload,
        )? {
            FrameRead::Frame(header) => header,
            FrameRead::End => return Err(changed("the segment ends".to_string())),
            FrameRead::Bad(reason) | FrameRead::Unsupported(reason) => {
                return Err(changed(reason));
            }
            FrameRead::Unauthentic => return Err(changed(AUTHENTICATION_FAILED.to_string())),
        };
        if header.transaction_id != transaction_id
            || header.flags.is_first() != first
            || header.flags.is_last() != last
            || (first && payload.len() < FRAME_PROLOGUE_BYTES)
        {
            return Err(changed("another frame sits at this position".to_string()));
        }
        self.lsn = advance(self.lsn, &header)?;
        self.offset = self.offset.saturating_add(header.frame_bytes());
        Ok(())
    }
}

/// One complete group: its frames' payloads, in order.
pub struct GroupFrames {
    transaction_id: u64,
    start_lsn: u64,
    end_lsn: u64,
    synced_lsn: u64,
    frame_count: usize,
    /// Payloads returned so far.
    returned: usize,
    source: GroupSource,
    /// The payload last returned.
    current: Vec<u8>,
}

impl std::fmt::Debug for GroupFrames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupFrames")
            .field("transaction_id", &self.transaction_id)
            .field("start_lsn", &self.start_lsn)
            .field("end_lsn", &self.end_lsn)
            .field("synced_lsn", &self.synced_lsn)
            .field("frame_count", &self.frame_count)
            .finish_non_exhaustive()
    }
}

impl GroupFrames {
    /// The transaction that wrote the group.
    #[must_use]
    pub fn transaction_id(&self) -> TransactionId {
        TransactionId::new(self.transaction_id)
    }

    /// The log position of the group's FIRST frame.
    #[must_use]
    pub fn start_lsn(&self) -> u64 {
        self.start_lsn
    }

    /// The log position right after the group's LAST frame.
    #[must_use]
    pub fn end_lsn(&self) -> u64 {
        self.end_lsn
    }

    /// The WAL's durable LSN when the group began (its FIRST prologue).
    #[must_use]
    pub fn synced_lsn(&self) -> u64 {
        self.synced_lsn
    }

    /// How many frames the group has.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// The records of the next frame (the FIRST prologue left out), or
    /// `None` after the LAST frame.
    ///
    /// # Errors
    ///
    /// [`WalError::Damaged`] when a large group read again no longer matches
    /// the scan, and [`WalError::Io`] when its segment cannot be read.
    pub fn next_payload(&mut self) -> Result<Option<&[u8]>, WalError> {
        if self.returned == self.frame_count {
            return Ok(None);
        }
        let first = self.returned == 0;
        match &mut self.source {
            GroupSource::Buffered(frames) => match frames.next() {
                Some(payload) => self.current = payload,
                None => return Ok(None),
            },
            GroupSource::Reread(reread) => reread.next_frame(
                self.transaction_id,
                first,
                self.returned + 1 == self.frame_count,
                &mut self.current,
            )?,
        }
        self.returned += 1;
        let skip = if first { FRAME_PROLOGUE_BYTES } else { 0 };
        Ok(Some(&self.current[skip..]))
    }
}

#[cfg(test)]
mod tests {
    use super::super::DurabilityMode;
    use super::super::frame::FrameFlags;
    use super::super::segment::{SALT_BYTES, segment_file_name};
    use super::super::writer::{GroupEnd, Wal, WalOptions};
    use super::*;

    const DATABASE: u128 = 0x1988_0319_0088_0003_0019_1988_0319_0088;

    fn record(text: &str) -> Vec<u8> {
        let mut bytes = u32::try_from(text.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend_from_slice(text.as_bytes());
        bytes
    }

    fn records_of(mut payload: &[u8]) -> Vec<String> {
        let mut records = Vec::new();
        while !payload.is_empty() {
            let length =
                usize::try_from(u32::from_le_bytes(payload[..4].try_into().unwrap())).unwrap();
            records.push(String::from_utf8(payload[4..4 + length].to_vec()).unwrap());
            payload = &payload[4 + length..];
        }
        records
    }

    /// A scanned group: transaction id and records.
    type Group = (u64, Vec<String>);

    fn group(id: u64, texts: &[&str]) -> Group {
        (id, texts.iter().map(|text| (*text).to_string()).collect())
    }

    fn scan_all(dir: &Path, options: ScanOptions) -> Result<(Vec<Group>, ScanEnd), WalError> {
        let mut scan = WalScan::open(dir, options)?;
        let mut groups = Vec::new();
        while let Some(mut frames) = scan.next_group()? {
            let mut records = Vec::new();
            while let Some(payload) = frames.next_payload()? {
                records.extend(records_of(payload));
            }
            groups.push((frames.transaction_id().as_u64(), records));
        }
        Ok((groups, scan.finish()?))
    }

    fn writer_options(start_lsn: u64) -> WalOptions {
        WalOptions {
            durability: DurabilityMode::NoSync,
            frame_target_bytes: 16,
            ..WalOptions::new(DATABASE, start_lsn)
        }
    }

    fn write_group(wal: &Wal, id: u64, texts: &[&str]) -> GroupEnd {
        let mut group = wal.begin_group(TransactionId::new(id)).unwrap();
        for text in texts {
            group.push(&record(text)).unwrap();
        }
        group.finish().unwrap()
    }

    fn segment_path(dir: &Path, first_lsn: u64) -> PathBuf {
        dir.join(segment_file_name(first_lsn))
    }

    /// The file offset of `lsn` in the segment starting at `first_lsn`.
    fn offset(first_lsn: u64, lsn: u64) -> u64 {
        SEGMENT_HEADER_BYTES as u64 + lsn - first_lsn
    }

    fn flip_byte(path: &Path, offset: u64) {
        let mut bytes = std::fs::read(path).unwrap();
        bytes[usize::try_from(offset).unwrap()] ^= 0x5A;
        std::fs::write(path, bytes).unwrap();
    }

    const FIVE: [&str; 5] = [
        "Alix 0000",
        "Gus 00000",
        "Mia 00000",
        "Jules 000",
        "Butch 000",
    ];

    #[test]
    fn a_scan_returns_every_group_with_its_records_across_segments() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), writer_options(3)).unwrap();
        write_group(&wal, 1, &["Alix"]);
        write_group(&wal, 2, &FIVE);
        let second = wal.rotate().unwrap();
        write_group(&wal, 3, &["Amsterdam", "Berlin"]);
        let third = wal.rotate().unwrap();
        let last = write_group(&wal, 4, &FIVE[..2]);
        drop(wal);
        let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 3)).unwrap();
        assert_eq!(
            groups,
            [
                group(1, &["Alix"]),
                group(2, &FIVE),
                group(3, &["Amsterdam", "Berlin"]),
                group(4, &FIVE[..2]),
            ]
        );
        assert_eq!(end.end_lsn, last.end_lsn);
        assert!(end.is_clean(), "{end:?}");
        // From a checkpoint at the second segment: the first is skipped.
        let (groups, _) = scan_all(dir.path(), ScanOptions::new(DATABASE, second)).unwrap();
        assert_eq!(
            groups,
            [group(3, &["Amsterdam", "Berlin"]), group(4, &FIVE[..2])]
        );
        let (groups, _) = scan_all(dir.path(), ScanOptions::new(DATABASE, third)).unwrap();
        assert_eq!(groups, [group(4, &FIVE[..2])]);
        // At the end of the log: nothing to replay.
        let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, last.end_lsn)).unwrap();
        assert!(groups.is_empty(), "{groups:?}");
        assert_eq!(end.end_lsn, last.end_lsn);
    }

    #[test]
    fn a_group_reports_its_position_and_frames() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), writer_options(19)).unwrap();
        let first = write_group(&wal, 88, &FIVE);
        wal.sync().unwrap();
        let second = write_group(&wal, 89, &["Paris"]);
        drop(wal);
        let mut scan = WalScan::open(dir.path(), ScanOptions::new(DATABASE, 19)).unwrap();
        let frames = scan.next_group().unwrap().unwrap();
        assert_eq!(frames.transaction_id(), TransactionId::new(88));
        assert_eq!((frames.start_lsn(), frames.end_lsn()), (19, first.end_lsn));
        assert_eq!(frames.synced_lsn(), 19);
        assert_eq!(
            frames.frame_count(),
            5,
            "a frame per record at a 16-byte target"
        );
        let frames = scan.next_group().unwrap().unwrap();
        assert!(
            second.start_lsn > first.end_lsn,
            "the sync's marker sits between the groups, and the scan passes over it"
        );
        assert_eq!(
            (frames.start_lsn(), frames.end_lsn()),
            (second.start_lsn, second.end_lsn)
        );
        assert_eq!(frames.synced_lsn(), first.end_lsn);
        assert_eq!(frames.frame_count(), 1);
        assert!(scan.next_group().unwrap().is_none());
    }

    #[test]
    fn a_missing_directory_is_an_empty_log() {
        let dir = tempfile::tempdir().unwrap();
        let (groups, end) = scan_all(
            &dir.path().join("absent.wal"),
            ScanOptions::new(DATABASE, 88),
        )
        .unwrap();
        assert!(groups.is_empty(), "{groups:?}");
        assert_eq!(end.end_lsn, 88);
        assert!(end.is_clean());
    }

    /// Cuts the last segment of `source` at every byte of the group after
    /// `complete` and checks that the scan returns the complete groups, cuts
    /// the rest, and that a writer then continues where they end.
    fn check_torn_tail_at_every_byte(
        source: &Path,
        complete: &[Group],
        torn_start: u64,
        torn_end: u64,
        first_lsn: u64,
        writer: impl Fn(u64) -> WalOptions,
        scan: impl Fn() -> ScanOptions,
    ) {
        let path = segment_path(source, first_lsn);
        let bytes = std::fs::read(&path).unwrap();
        let name = path.file_name().unwrap().to_owned();
        assert_eq!(
            u64::try_from(bytes.len()).unwrap(),
            offset(first_lsn, torn_end),
            "the torn group ends the segment"
        );
        for cut in offset(first_lsn, torn_start)..offset(first_lsn, torn_end) {
            let dir = tempfile::tempdir().unwrap();
            for (_, other) in list_wal_directory(source).unwrap().segments {
                if other != path {
                    std::fs::copy(&other, dir.path().join(other.file_name().unwrap())).unwrap();
                }
            }
            let copy = dir.path().join(&name);
            std::fs::write(&copy, &bytes[..usize::try_from(cut).unwrap()]).unwrap();
            let (groups, end) = scan_all(dir.path(), scan()).unwrap();
            assert_eq!(groups, complete, "cut at byte {cut}");
            assert_eq!(end.end_lsn, torn_start, "cut at byte {cut}");
            if cut == offset(first_lsn, torn_start) {
                assert!(end.is_clean(), "cut at the group start: {end:?}");
            } else {
                let tail = end.tail.as_ref().expect("a torn tail");
                assert_eq!(tail.kind, TailKind::Torn, "cut at byte {cut}: {tail:?}");
                assert_eq!(
                    (tail.lsn, tail.offset),
                    (torn_start, offset(first_lsn, torn_start))
                );
            }
            end.cut_torn_tail().unwrap();
            assert_eq!(
                std::fs::metadata(&copy).unwrap().len(),
                offset(first_lsn, torn_start),
                "cut at byte {cut}: the tail is gone"
            );
            let wal = Wal::open(dir.path(), writer(end.end_lsn)).unwrap();
            let next = write_group(&wal, 319, &["Prague"]);
            assert_eq!(next.start_lsn, torn_start);
            drop(wal);
            let (groups, end) = scan_all(dir.path(), scan()).unwrap();
            let mut expected = complete.to_vec();
            expected.push(group(319, &["Prague"]));
            assert_eq!(groups, expected, "cut at byte {cut}: the log continues");
            assert!(end.is_clean());
        }
    }

    #[test]
    fn a_torn_tail_at_every_byte_of_the_last_group_is_cut() {
        let source = tempfile::tempdir().unwrap();
        let wal = Wal::open(source.path(), writer_options(0)).unwrap();
        write_group(&wal, 1, &["Alix"]);
        let rotated = wal.rotate().unwrap();
        let second = write_group(&wal, 2, &["Gus", "Mia"]);
        let torn = write_group(&wal, 3, &FIVE);
        drop(wal);
        check_torn_tail_at_every_byte(
            source.path(),
            &[group(1, &["Alix"]), group(2, &["Gus", "Mia"])],
            second.end_lsn,
            torn.end_lsn,
            rotated,
            writer_options,
            || ScanOptions::new(DATABASE, 0),
        );
    }

    #[test]
    fn zeros_or_garbage_after_the_last_group_are_a_torn_tail() {
        for filler in [vec![0u8; 4096], b"Vincent Vega".repeat(19), vec![0xFF; 3]] {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer_options(0)).unwrap();
            let end = write_group(&wal, 1, &FIVE);
            drop(wal);
            let path = segment_path(dir.path(), 0);
            let mut bytes = std::fs::read(&path).unwrap();
            bytes.extend_from_slice(&filler);
            std::fs::write(&path, bytes).unwrap();
            let (groups, scanned) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
            assert_eq!(groups, [group(1, &FIVE)]);
            assert_eq!(scanned.end_lsn, end.end_lsn);
            assert_eq!(scanned.tail.as_ref().unwrap().kind, TailKind::Torn);
        }
    }

    #[test]
    fn damage_in_a_sealed_segment_names_the_file_and_offset() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), writer_options(0)).unwrap();
        let first = write_group(&wal, 1, &["Alix"]);
        let second = write_group(&wal, 2, &FIVE);
        wal.rotate().unwrap();
        write_group(&wal, 3, &["Berlin"]);
        drop(wal);
        let sealed = segment_path(dir.path(), 0);
        // A byte in the payload of the second group's third frame.
        let frame_lsn = first.end_lsn + (second.end_lsn - first.end_lsn) / 2;
        let mut bytes = std::fs::read(&sealed).unwrap();
        let original = bytes.clone();
        let frame_start = (first.end_lsn..second.end_lsn)
            .find(|lsn| {
                let at = usize::try_from(offset(0, *lsn)).unwrap();
                *lsn >= frame_lsn && bytes[at + 8..at + 16] == lsn.to_le_bytes()
            })
            .unwrap();
        let damaged = offset(0, frame_start) + 30;
        bytes[usize::try_from(damaged).unwrap()] ^= 0x01;
        std::fs::write(&sealed, &bytes).unwrap();
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
        match &error {
            WalError::Damaged {
                path,
                offset: at,
                lsn,
                reason,
            } => {
                assert_eq!(path, &sealed);
                assert_eq!(
                    (*at, *lsn),
                    (offset(0, frame_start), frame_start),
                    "{error}"
                );
                assert!(reason.contains("checksum"), "{reason}");
            }
            other => panic!("expected damage, got {other}"),
        }
        let message = error.to_string();
        assert!(
            message.contains("wal_00000000000000000000.log")
                && message.contains(&format!("offset {}", offset(0, frame_start))),
            "{message}"
        );
        // A damaged frame header is damage too.
        std::fs::write(&sealed, &original).unwrap();
        flip_byte(&sealed, offset(0, frame_start) + 9);
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
        assert!(matches!(error, WalError::Damaged { .. }), "{error}");
        // A sealed segment cut short inside a group no longer reaches the
        // next segment: a gap in the chain, found before any frame is read.
        let cut = usize::try_from(offset(0, frame_start)).unwrap();
        std::fs::write(&sealed, &original[..cut]).unwrap();
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
        assert!(matches!(error, WalError::Gap { .. }), "{error}");
        assert!(
            error.to_string().contains(&format!("LSN {frame_start}")),
            "names where the cut segment ends: {error}"
        );
    }

    /// A log that starts at `start` and, when `later_segment`, rotates once
    /// before the groups under test: returns the first LSN of the segment
    /// they land in. Offsets and LSNs then differ, so a scan that mixes them
    /// up fails here.
    fn start_log(wal: &Wal, start: u64, later_segment: bool) -> u64 {
        if !later_segment {
            return start;
        }
        write_group(wal, 88, &["Amsterdam"]);
        wal.rotate().unwrap()
    }

    #[test]
    fn an_unsynced_hole_followed_by_unsynced_groups_is_a_torn_tail() {
        for (start, later_segment) in [(0, false), (319, true)] {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer_options(start)).unwrap();
            let first_lsn = start_log(&wal, start, later_segment);
            write_group(&wal, 1, &["Alix"]);
            let synced = wal.end_lsn();
            wal.sync().unwrap();
            let hole = write_group(&wal, 2, &FIVE);
            write_group(&wal, 3, &["Gus"]);
            write_group(&wal, 4, &FIVE[..3]);
            drop(wal);
            // Writeback reached the later groups but not the second one.
            let path = segment_path(dir.path(), first_lsn);
            flip_byte(&path, offset(first_lsn, hole.start_lsn) + 40);
            let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, start)).unwrap();
            let expected = if later_segment {
                vec![group(88, &["Amsterdam"]), group(1, &["Alix"])]
            } else {
                vec![group(1, &["Alix"])]
            };
            assert_eq!(groups, expected, "nothing after the hole is replayed");
            // The log is kept up to the hole: the synced group and the
            // marker of its sync.
            let kept = hole.start_lsn;
            assert!(kept > synced);
            assert_eq!(end.end_lsn, kept);
            let tail = end.tail.clone().unwrap();
            assert_eq!(tail.kind, TailKind::Torn, "{tail:?}");
            assert_eq!((tail.lsn, tail.offset), (kept, offset(first_lsn, kept)));
            end.cut_torn_tail().unwrap();
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                offset(first_lsn, kept)
            );
        }
    }

    #[test]
    fn a_hole_before_a_group_synced_after_it_is_damage() {
        for (start, later_segment) in [(0, false), (319, true)] {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer_options(start)).unwrap();
            let first_lsn = start_log(&wal, start, later_segment);
            write_group(&wal, 1, &["Alix"]);
            let hole = write_group(&wal, 2, &FIVE);
            wal.sync().unwrap();
            write_group(&wal, 3, &["Gus"]);
            drop(wal);
            let path = segment_path(dir.path(), first_lsn);
            flip_byte(&path, offset(first_lsn, hole.start_lsn) + 40);
            let error = scan_all(dir.path(), ScanOptions::new(DATABASE, start)).unwrap_err();
            match &error {
                WalError::Damaged {
                    path: at,
                    offset: damaged_at,
                    lsn,
                    reason,
                } => {
                    assert_eq!(at, &path);
                    assert!(*lsn >= hole.start_lsn && *lsn < hole.end_lsn, "{error}");
                    assert_eq!(*damaged_at, offset(first_lsn, *lsn), "{error}");
                    assert!(reason.contains("synced"), "{reason}");
                }
                other => panic!("expected damage, got {other}"),
            }
        }
    }

    /// The checkpoint LSN is where a group starts. A whole frame there that
    /// does not start one is never a torn write: the log does not fit the
    /// image, so it is damage (salvage stops there), never a tail to cut.
    #[test]
    fn a_checkpoint_on_a_frame_inside_a_group_is_damage_not_a_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), writer_options(0)).unwrap();
        let first = write_group(&wal, 1, &["Alix"]);
        let multi = write_group(&wal, 2, &FIVE);
        drop(wal);
        let frames = frame_lsns(&segment_path(dir.path(), 0));
        let inside = frames
            .into_iter()
            .find(|lsn| *lsn > first.end_lsn && *lsn < multi.end_lsn)
            .unwrap();
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, inside)).unwrap_err();
        match &error {
            WalError::Damaged { lsn, reason, .. } => {
                assert_eq!(*lsn, inside);
                assert!(reason.contains("does not start a group"), "{reason}");
            }
            other => panic!("expected damage, got {other}"),
        }
        let salvage = ScanOptions {
            salvage: true,
            ..ScanOptions::new(DATABASE, inside)
        };
        let (groups, end) = scan_all(dir.path(), salvage).unwrap();
        assert!(groups.is_empty(), "{groups:?}");
        let tail = end.tail.unwrap();
        assert_eq!((tail.kind, tail.lsn), (TailKind::Damaged, inside));
        // At a group start the same log scans.
        let (groups, end) =
            scan_all(dir.path(), ScanOptions::new(DATABASE, first.end_lsn)).unwrap();
        assert_eq!(groups, [group(2, &FIVE)]);
        assert!(end.is_clean());
    }

    /// The LSN of every frame of the segment at `path`, in order.
    fn frame_lsns(path: &Path) -> Vec<u64> {
        let bytes = std::fs::read(path).unwrap();
        let mut lsns = Vec::new();
        let mut at = SEGMENT_HEADER_BYTES;
        while at < bytes.len() {
            let header = FrameHeader::decode(bytes[at..at + 25].try_into().unwrap()).unwrap();
            lsns.push(header.lsn);
            at += 25 + usize::try_from(header.length).unwrap();
        }
        lsns
    }

    /// A FIRST frame's prologue holds the durable LSN when its group began,
    /// which is at most the group's own position. A whole frame that claims
    /// more is damage, also in the last segment, never a torn tail.
    #[test]
    fn a_prologue_past_its_own_frame_is_damage() {
        let dir = tempfile::tempdir().unwrap();
        put_segment(
            dir.path(),
            19,
            &[
                (1, FrameFlags::FIRST_AND_LAST, first_payload(19, "Alix")),
                (2, FrameFlags::FIRST_AND_LAST, first_payload(88, "Gus")),
            ],
        );
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 19)).unwrap_err();
        match &error {
            WalError::Damaged { lsn, reason, .. } => {
                assert_eq!(*lsn, 19 + 41);
                assert!(reason.contains("LSN 88"), "{reason}");
            }
            other => panic!("expected damage, got {other}"),
        }
        // Its own position is allowed: everything before it was synced.
        let dir = tempfile::tempdir().unwrap();
        put_segment(
            dir.path(),
            19,
            &[
                (1, FrameFlags::FIRST_AND_LAST, first_payload(19, "Alix")),
                (2, FrameFlags::FIRST_AND_LAST, first_payload(60, "Gus")),
            ],
        );
        let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 19)).unwrap();
        assert_eq!(groups, [group(1, &["Alix"]), group(2, &["Gus"])]);
        assert!(end.is_clean());
    }

    #[test]
    fn a_wal_of_another_database_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), writer_options(0)).unwrap();
        write_group(&wal, 1, &["Alix"]);
        drop(wal);
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE ^ 0xFF, 0)).unwrap_err();
        assert!(
            matches!(error, WalError::ForeignDatabase { found, expected, .. }
                if found == DATABASE && expected == DATABASE ^ 0xFF),
            "{error}"
        );
        let message = error.to_string();
        assert!(
            message.contains("belongs to database")
                && message.contains(&format!("{DATABASE:032x}")),
            "{message}"
        );
    }

    fn three_segments(dir: &Path) -> [u64; 4] {
        let wal = Wal::open(dir, writer_options(0)).unwrap();
        write_group(&wal, 1, &["Alix"]);
        let second = wal.rotate().unwrap();
        write_group(&wal, 2, &["Gus"]);
        let third = wal.rotate().unwrap();
        let end = write_group(&wal, 3, &["Mia"]);
        [0, second, third, end.end_lsn]
    }

    #[test]
    fn a_gap_between_segments_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let [_, second, third, _] = three_segments(dir.path());
        std::fs::remove_file(segment_path(dir.path(), second)).unwrap();
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
        let message = error.to_string();
        assert!(matches!(error, WalError::Gap { .. }), "{message}");
        assert!(
            message.contains(&format!("LSN {second}")) && message.contains(&format!("LSN {third}")),
            "names both positions: {message}"
        );
    }

    #[test]
    fn a_log_that_resumes_after_the_checkpoint_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let [_, second, _, _] = three_segments(dir.path());
        std::fs::remove_file(segment_path(dir.path(), 0)).unwrap();
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 3)).unwrap_err();
        let message = error.to_string();
        assert!(matches!(error, WalError::Gap { .. }), "{message}");
        assert!(
            message.contains(&format!("LSN {second}")) && message.contains("LSN 3"),
            "{message}"
        );
    }

    #[test]
    fn a_segment_from_the_future_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let [.., end] = three_segments(dir.path());
        let future = tempfile::tempdir().unwrap();
        let wal = Wal::open(future.path(), writer_options(end + 1988)).unwrap();
        write_group(&wal, 4, &["Jules"]);
        drop(wal);
        std::fs::copy(
            segment_path(future.path(), end + 1988),
            segment_path(dir.path(), end + 1988),
        )
        .unwrap();
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
        assert!(matches!(error, WalError::Gap { .. }), "{error}");
    }

    #[test]
    fn a_first_lsn_that_differs_from_the_file_name_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let [_, second, third, _] = three_segments(dir.path());
        std::fs::rename(
            segment_path(dir.path(), third),
            segment_path(dir.path(), third + 3),
        )
        .unwrap();
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, second)).unwrap_err();
        let message = error.to_string();
        assert!(matches!(error, WalError::SegmentHeader { .. }), "{message}");
        assert!(
            message.contains(&format!("LSN {third}"))
                && message.contains(&format!("LSN {}", third + 3)),
            "{message}"
        );
    }

    #[test]
    fn a_damaged_segment_header_is_refused_even_below_the_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let [_, _, third, _] = three_segments(dir.path());
        flip_byte(&segment_path(dir.path(), 0), 20);
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, third)).unwrap_err();
        assert!(
            matches!(&error, WalError::SegmentHeader { path, .. } if path == &segment_path(dir.path(), 0)),
            "{error}"
        );
    }

    #[test]
    fn a_segment_cut_off_while_it_was_created_is_removed_by_the_cut() {
        for contents in [Vec::new(), vec![0u8; 128], b"GRAFWAL\0".to_vec()] {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer_options(0)).unwrap();
            let end = write_group(&wal, 1, &["Alix"]);
            drop(wal);
            let unfinished = segment_path(dir.path(), end.end_lsn);
            std::fs::write(&unfinished, &contents).unwrap();
            let (groups, scanned) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
            assert_eq!(groups, [group(1, &["Alix"])]);
            assert_eq!(scanned.end_lsn, end.end_lsn);
            assert_eq!(
                scanned.unfinished_segment.as_deref(),
                Some(unfinished.as_path())
            );
            assert!(scanned.tail.is_none());
            scanned.cut_torn_tail().unwrap();
            assert!(!unfinished.exists());
            let wal = Wal::open(dir.path(), writer_options(end.end_lsn)).unwrap();
            write_group(&wal, 2, &["Gus"]);
        }
        // An unfinished header before a later segment is damage, not a crash
        // during creation.
        let dir = tempfile::tempdir().unwrap();
        let [_, second, ..] = three_segments(dir.path());
        std::fs::write(segment_path(dir.path(), second), [0u8; 19]).unwrap();
        let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
        assert!(matches!(error, WalError::SegmentHeader { .. }), "{error}");
    }

    #[test]
    fn unknown_files_are_reported_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        three_segments(dir.path());
        std::fs::write(dir.path().join("wal_00000001.log"), b"0.5").unwrap();
        std::fs::write(dir.path().join("backup.cursor"), b"88").unwrap();
        std::fs::create_dir(dir.path().join("damaged-00000000000000000003")).unwrap();
        let (_, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
        assert_eq!(end.unknown_files, [dir.path().join("wal_00000001.log")]);
        end.cut_torn_tail().unwrap();
        assert!(dir.path().join("wal_00000001.log").exists());
    }

    #[test]
    fn a_large_group_is_read_again_frame_by_frame() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), writer_options(0)).unwrap();
        let texts: Vec<String> = (0..88).map(|n| format!("Barcelona {n:03}")).collect();
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        write_group(&wal, 1, &["Alix"]);
        write_group(&wal, 2, &texts);
        write_group(&wal, 3, &["Gus"]);
        drop(wal);
        let options = ScanOptions {
            group_buffer_bytes: 256,
            ..ScanOptions::new(DATABASE, 0)
        };
        let (groups, end) = scan_all(dir.path(), options).unwrap();
        assert_eq!(
            groups,
            [group(1, &["Alix"]), group(2, &texts), group(3, &["Gus"])]
        );
        assert!(end.is_clean());
    }

    /// Writes a plaintext segment by hand: frames with the given transaction,
    /// flags and payload, each at its LSN.
    fn put_segment(dir: &Path, first_lsn: u64, frames: &[(u64, FrameFlags, Vec<u8>)]) -> PathBuf {
        let header = SegmentHeader {
            encrypted: false,
            database_id: DATABASE,
            first_lsn,
            creation_time_ms: 1988,
            salt: [0; SALT_BYTES],
            key_check: [0; 28],
        };
        let mut bytes = header.encode().to_vec();
        let mut lsn = first_lsn;
        for (transaction, flags, payload) in frames {
            let frame = FrameHeader::new(
                u32::try_from(payload.len()).unwrap(),
                lsn,
                *transaction,
                *flags,
            )
            .with_checksum(payload);
            bytes.extend_from_slice(&frame.encode());
            bytes.extend_from_slice(payload);
            lsn += frame.frame_bytes();
        }
        let path = segment_path(dir, first_lsn);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn first_payload(synced_lsn: u64, text: &str) -> Vec<u8> {
        let mut payload = synced_lsn.to_le_bytes().to_vec();
        payload.extend_from_slice(&record(text));
        payload
    }

    #[test]
    fn frames_out_of_group_order_end_the_log_or_are_damage() {
        let complete = (1, FrameFlags::FIRST_AND_LAST, first_payload(0, "Alix"));
        let cases: [(&str, Vec<(u64, FrameFlags, Vec<u8>)>); 4] = [
            (
                "a middle frame where a group must start",
                vec![complete.clone(), (2, FrameFlags::MIDDLE, record("Gus"))],
            ),
            (
                "a frame of another transaction inside a group",
                vec![
                    complete.clone(),
                    (2, FrameFlags::FIRST, first_payload(0, "Gus")),
                    (3, FrameFlags::LAST, record("Mia")),
                ],
            ),
            (
                "a group that starts inside another",
                vec![
                    complete.clone(),
                    (2, FrameFlags::FIRST, first_payload(0, "Gus")),
                    (3, FrameFlags::FIRST_AND_LAST, first_payload(0, "Mia")),
                ],
            ),
            (
                "a group without its LAST frame",
                vec![
                    complete.clone(),
                    (2, FrameFlags::FIRST, first_payload(0, "Gus")),
                    (2, FrameFlags::MIDDLE, record("Mia")),
                ],
            ),
        ];
        for (what, frames) in cases {
            // As the last segment: never durable, a torn tail.
            let dir = tempfile::tempdir().unwrap();
            put_segment(dir.path(), 0, &frames);
            let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
            assert_eq!(groups, [group(1, &["Alix"])], "{what}");
            assert_eq!(
                end.tail.map(|tail| tail.kind),
                Some(TailKind::Torn),
                "{what}"
            );
            // As a sealed segment: damage.
            let sealed_end = list_wal_directory(dir.path()).unwrap().segments[0]
                .1
                .metadata()
                .unwrap()
                .len()
                - SEGMENT_HEADER_BYTES as u64;
            put_segment(
                dir.path(),
                sealed_end,
                &[(
                    9,
                    FrameFlags::FIRST_AND_LAST,
                    first_payload(sealed_end, "Paris"),
                )],
            );
            let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
            assert!(matches!(error, WalError::Damaged { .. }), "{what}: {error}");
        }
    }

    #[test]
    fn salvage_stops_at_the_last_intact_group_and_sets_the_rest_aside() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path(), writer_options(0)).unwrap();
        let first = write_group(&wal, 1, &["Alix"]);
        let second = write_group(&wal, 2, &FIVE);
        let later = wal.rotate().unwrap();
        write_group(&wal, 3, &["Berlin"]);
        drop(wal);
        let sealed = segment_path(dir.path(), 0);
        flip_byte(&sealed, offset(0, first.end_lsn) + 30);
        let damaged_bytes = std::fs::read(&sealed).unwrap()
            [usize::try_from(offset(0, first.end_lsn)).unwrap()..]
            .to_vec();
        let options = ScanOptions {
            salvage: true,
            ..ScanOptions::new(DATABASE, 0)
        };
        let (groups, end) = scan_all(dir.path(), options.clone()).unwrap();
        assert_eq!(groups, [group(1, &["Alix"])]);
        assert_eq!(end.end_lsn, first.end_lsn);
        let tail = end.tail.clone().unwrap();
        assert_eq!(tail.kind, TailKind::Damaged);
        assert_eq!(tail.set_aside, [segment_path(dir.path(), later)]);
        assert!(tail.reason.contains("checksum"), "{}", tail.reason);
        end.cut_torn_tail().unwrap();
        let aside = dir.path().join(format!("damaged-{:020}", first.end_lsn));
        assert_eq!(
            std::fs::read(aside.join(segment_file_name(0))).unwrap(),
            damaged_bytes,
            "the damaged bytes are kept"
        );
        assert!(
            aside.join(segment_file_name(later)).exists(),
            "the later segment is kept"
        );
        assert!(second.end_lsn > first.end_lsn);
        // The log now ends at the last intact group and takes writes there.
        let (groups, end) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
        assert_eq!(groups, [group(1, &["Alix"])]);
        assert_eq!(
            end.unknown_files,
            Vec::<PathBuf>::new(),
            "the damaged directory is known"
        );
        let wal = Wal::open(dir.path(), writer_options(end.end_lsn)).unwrap();
        write_group(&wal, 4, &["Prague"]);
        drop(wal);
        let (groups, _) = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap();
        assert_eq!(groups, [group(1, &["Alix"]), group(4, &["Prague"])]);
    }

    #[test]
    fn salvage_stops_before_a_gap() {
        let dir = tempfile::tempdir().unwrap();
        let [_, second, third, _] = three_segments(dir.path());
        std::fs::remove_file(segment_path(dir.path(), second)).unwrap();
        let options = ScanOptions {
            salvage: true,
            ..ScanOptions::new(DATABASE, 0)
        };
        let (groups, end) = scan_all(dir.path(), options).unwrap();
        assert_eq!(groups, [group(1, &["Alix"])]);
        assert_eq!(end.end_lsn, second);
        let tail = end.tail.clone().unwrap();
        assert_eq!(tail.set_aside, [segment_path(dir.path(), third)]);
        end.cut_torn_tail().unwrap();
        let wal = Wal::open(dir.path(), writer_options(end.end_lsn)).unwrap();
        write_group(&wal, 4, &["Prague"]);
    }

    #[cfg(feature = "encryption")]
    mod encrypted {
        use grafeo_common::encryption::KeyChain;

        use super::*;

        fn cipher_for(master: u8) -> CipherForSalt {
            let chain = Arc::new(KeyChain::new([master; 32]));
            Arc::new(move |salt: &[u8; SALT_BYTES]| {
                let mut id = DATABASE.to_le_bytes().to_vec();
                id.extend_from_slice(salt);
                chain.encryptor_for("grafeo-wal", &id)
            })
        }

        fn writer(start_lsn: u64) -> WalOptions {
            WalOptions {
                cipher_for_salt: Some(cipher_for(3)),
                ..writer_options(start_lsn)
            }
        }

        fn scan(master: u8) -> ScanOptions {
            ScanOptions {
                cipher_for_salt: Some(cipher_for(master)),
                ..ScanOptions::new(DATABASE, 0)
            }
        }

        #[test]
        fn an_encrypted_log_scans_back() {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer(0)).unwrap();
            write_group(&wal, 1, &FIVE);
            wal.rotate().unwrap();
            write_group(&wal, 2, &["Amsterdam"]);
            drop(wal);
            let (groups, end) = scan_all(dir.path(), scan(3)).unwrap();
            assert_eq!(groups, [group(1, &FIVE), group(2, &["Amsterdam"])]);
            assert!(end.is_clean());
        }

        #[test]
        fn no_plaintext_in_an_encrypted_segment() {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer(0)).unwrap();
            write_group(&wal, 1, &["Shosanna in Paris"; 19]);
            drop(wal);
            let bytes = std::fs::read(segment_path(dir.path(), 0)).unwrap();
            for needle in [&b"Shosanna"[..], b"Paris"] {
                assert!(
                    !bytes.windows(needle.len()).any(|window| window == needle),
                    "{} appears in the segment",
                    String::from_utf8_lossy(needle)
                );
            }
            let header = SegmentHeader::decode(&bytes, Path::new("segment")).unwrap();
            assert!(header.encrypted);
            assert!(header.salt.iter().any(|&byte| byte != 0));
        }

        #[test]
        fn a_wrong_key_is_reported_as_a_wrong_key() {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer(0)).unwrap();
            write_group(&wal, 1, &["Alix"]);
            drop(wal);
            let error = scan_all(dir.path(), scan(19)).unwrap_err();
            assert!(
                matches!(error, WalError::WrongKey { database_id, .. } if database_id == DATABASE),
                "{error}"
            );
            assert!(error.to_string().contains("wrong key"), "{error}");
            let error = Wal::open(
                dir.path(),
                WalOptions {
                    cipher_for_salt: Some(cipher_for(19)),
                    ..writer_options(wal_end(dir.path()))
                },
            )
            .unwrap_err();
            assert!(matches!(error, WalError::WrongKey { .. }), "{error}");
            let error = scan_all(dir.path(), ScanOptions::new(DATABASE, 0)).unwrap_err();
            assert!(matches!(error, WalError::MissingKey { .. }), "{error}");
        }

        fn wal_end(dir: &Path) -> u64 {
            scan_all(dir, scan(3)).unwrap().1.end_lsn
        }

        #[test]
        fn a_plaintext_segment_in_an_encrypted_database_is_refused() {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer_options(0)).unwrap();
            write_group(&wal, 1, &["Alix"]);
            drop(wal);
            let error = scan_all(dir.path(), scan(3)).unwrap_err();
            assert!(matches!(error, WalError::NotEncrypted { .. }), "{error}");
        }

        #[test]
        fn damage_is_found_without_the_key() {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer(0)).unwrap();
            let first = write_group(&wal, 1, &["Alix"]);
            write_group(&wal, 2, &["Gus"]);
            wal.rotate().unwrap();
            write_group(&wal, 3, &["Mia"]);
            drop(wal);
            let sealed = segment_path(dir.path(), 0);
            let bytes = std::fs::read(&sealed).unwrap();
            let at = usize::try_from(offset(0, first.end_lsn)).unwrap();
            let header = FrameHeader::decode(bytes[at..at + 25].try_into().unwrap()).unwrap();
            let end = at + 25 + usize::try_from(header.length).unwrap();
            assert!(
                header.checksum_matches(&bytes[at + 25..end]),
                "the checksum is over the ciphertext: no key needed"
            );
            flip_byte(&sealed, offset(0, first.end_lsn) + 40);
            let mut damaged = bytes.clone();
            damaged[at + 40] ^= 0x5A;
            assert!(!header.checksum_matches(&damaged[at + 25..end]));
            let error = scan_all(dir.path(), scan(3)).unwrap_err();
            match &error {
                WalError::Damaged { reason, lsn, .. } => {
                    assert_eq!(*lsn, first.end_lsn);
                    assert!(
                        reason.contains("checksum"),
                        "damage, not a key problem: {reason}"
                    );
                }
                other => panic!("expected damage, got {other}"),
            }
        }

        /// The frames of the first segment: (file offset, total length).
        fn frame_spans(bytes: &[u8]) -> Vec<(usize, usize)> {
            let mut spans = Vec::new();
            let mut at = SEGMENT_HEADER_BYTES;
            while at < bytes.len() {
                let header = FrameHeader::decode(bytes[at..at + 25].try_into().unwrap()).unwrap();
                let length = 25 + usize::try_from(header.length).unwrap();
                spans.push((at, length));
                at += length;
            }
            spans
        }

        /// Rewrites the frame header at `at` to LSN `lsn` and `flags`, with a
        /// checksum that matches again: what someone who moves frames can do
        /// without the key.
        fn reseal_header(bytes: &mut [u8], at: usize, lsn: u64, flags: Option<FrameFlags>) {
            let mut header = FrameHeader::decode(bytes[at..at + 25].try_into().unwrap()).unwrap();
            header.lsn = lsn;
            if let Some(flags) = flags {
                header.flags = flags;
            }
            let end = at + 25 + usize::try_from(header.length).unwrap();
            let header = header.with_checksum(&bytes[at + 25..end]);
            bytes[at..at + 25].copy_from_slice(&header.encode());
        }

        #[test]
        fn swapped_or_duplicated_frames_are_refused() {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer(0)).unwrap();
            // Three one-frame groups of the same size, then a sealed end.
            write_group(&wal, 1, &["Alix 1"]);
            write_group(&wal, 1, &["Alix 2"]);
            write_group(&wal, 1, &["Alix 3"]);
            write_group(&wal, 4, &FIVE);
            wal.rotate().unwrap();
            write_group(&wal, 5, &["Prague"]);
            drop(wal);
            let sealed = segment_path(dir.path(), 0);
            let original = std::fs::read(&sealed).unwrap();
            let spans = frame_spans(&original);
            let lsn_at = |at: usize| u64::try_from(at - SEGMENT_HEADER_BYTES).unwrap();
            let (a, length) = spans[0];
            let (b, _) = spans[1];
            assert_eq!(spans[1].1, length, "same-size frames");

            let mut swapped = original.clone();
            swapped[a..a + length].copy_from_slice(&original[b..b + length]);
            swapped[b..b + length].copy_from_slice(&original[a..a + length]);
            reseal_header(&mut swapped, a, lsn_at(a), None);
            reseal_header(&mut swapped, b, lsn_at(b), None);

            let mut duplicated = original.clone();
            duplicated[b..b + length].copy_from_slice(&original[a..a + length]);
            reseal_header(&mut duplicated, b, lsn_at(b), None);

            // The fourth group's first frame marked LAST: the group cut short.
            let (c, _) = spans[3];
            let mut cut_short = original.clone();
            reseal_header(
                &mut cut_short,
                c,
                lsn_at(c),
                Some(FrameFlags::FIRST_AND_LAST),
            );

            for (what, bytes) in [
                ("swapped", swapped),
                ("duplicated", duplicated),
                ("cut at a flag", cut_short),
            ] {
                std::fs::write(&sealed, &bytes).unwrap();
                let error = scan_all(dir.path(), scan(3)).unwrap_err();
                match &error {
                    WalError::Damaged { reason, .. } => {
                        assert!(reason.contains("authentication"), "{what}: {reason}");
                    }
                    other => panic!("{what}: expected damage, got {other}"),
                }
            }
        }

        #[test]
        fn a_torn_encrypted_tail_at_every_byte_is_cut() {
            let source = tempfile::tempdir().unwrap();
            let wal = Wal::open(source.path(), writer(0)).unwrap();
            let first = write_group(&wal, 1, &["Alix"]);
            let torn = write_group(&wal, 2, &FIVE);
            drop(wal);
            check_torn_tail_at_every_byte(
                source.path(),
                &[group(1, &["Alix"])],
                first.end_lsn,
                torn.end_lsn,
                0,
                writer,
                || scan(3),
            );
        }

        /// The same in a later segment of a log that starts at a non-zero
        /// LSN: each segment has a key of its own, and offsets differ from
        /// LSNs.
        #[test]
        fn a_torn_encrypted_tail_in_a_later_segment_is_cut() {
            let source = tempfile::tempdir().unwrap();
            let wal = Wal::open(source.path(), writer(319)).unwrap();
            write_group(&wal, 1, &["Alix"]);
            let rotated = wal.rotate().unwrap();
            let second = write_group(&wal, 2, &["Gus"]);
            let torn = write_group(&wal, 3, &FIVE);
            drop(wal);
            check_torn_tail_at_every_byte(
                source.path(),
                &[group(1, &["Alix"]), group(2, &["Gus"])],
                second.end_lsn,
                torn.end_lsn,
                rotated,
                writer,
                || ScanOptions {
                    from_lsn: 319,
                    ..scan(3)
                },
            );
        }

        /// A segment below the checkpoint is skipped, so its key is never
        /// checked: one written under another key does not fail the open,
        /// while the same segment at or past the checkpoint does.
        #[test]
        fn a_segment_below_the_checkpoint_is_not_key_checked() {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(
                dir.path(),
                WalOptions {
                    cipher_for_salt: Some(cipher_for(19)),
                    ..writer_options(0)
                },
            )
            .unwrap();
            let old = write_group(&wal, 1, &["Alix"]);
            drop(wal);
            let wal = Wal::open(dir.path(), writer(old.end_lsn + 3)).unwrap();
            write_group(&wal, 2, &["Gus"]);
            drop(wal);
            let checkpoint = ScanOptions {
                from_lsn: old.end_lsn + 3,
                ..scan(3)
            };
            let (groups, _) = scan_all(dir.path(), checkpoint).unwrap();
            assert_eq!(groups, [group(2, &["Gus"])]);
            let error = scan_all(dir.path(), scan(3)).unwrap_err();
            assert!(matches!(error, WalError::WrongKey { .. }), "{error}");
        }

        #[test]
        fn an_authentication_failure_in_the_last_segment_is_damage() {
            let dir = tempfile::tempdir().unwrap();
            let wal = Wal::open(dir.path(), writer(0)).unwrap();
            write_group(&wal, 1, &["Alix 1"]);
            write_group(&wal, 2, &["Alix 2"]);
            drop(wal);
            let path = segment_path(dir.path(), 0);
            let mut bytes = std::fs::read(&path).unwrap();
            let spans = frame_spans(&bytes);
            let (b, length) = spans[1];
            let first = bytes[spans[0].0..spans[0].0 + length].to_vec();
            bytes[b..b + length].copy_from_slice(&first);
            reseal_header(
                &mut bytes,
                b,
                u64::try_from(b - SEGMENT_HEADER_BYTES).unwrap(),
                None,
            );
            std::fs::write(&path, &bytes).unwrap();
            let error = scan_all(dir.path(), scan(3)).unwrap_err();
            assert!(
                matches!(&error, WalError::Damaged { reason, .. } if reason.contains("authentication")),
                "a frame that is whole but does not authenticate was never torn: {error}"
            );
        }
    }
}
