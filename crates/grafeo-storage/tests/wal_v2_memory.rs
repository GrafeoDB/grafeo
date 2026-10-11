//! The WAL v2 writer and scanner hold memory that does not grow with a group.
//!
//! The global allocator of this test binary counts the bytes it hands out,
//! the highest count since a reset and the largest single request, and
//! forwards every call to the system allocator. The tests write a group many
//! times larger than the scanner's group buffer and measure:
//!
//! - writing it: the writer streams frames to the segment as records are
//!   pushed, so it holds about one frame;
//! - scanning and reading it: the scanner keeps a group of up to its buffer
//!   size from the scan and reads a larger one again frame by frame, so it
//!   holds that buffer and about one frame;
//! - a frame header that declares more bytes than the limit, or than the
//!   file holds: refused before anything of that size is allocated.
//!
//! The counters are global, so the tests take turns.

#![cfg(feature = "wal")]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this only forwards to System"
)]

use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};

use grafeo_common::types::TransactionId;
use grafeo_storage::wal::{
    DurabilityMode, FRAME_TARGET, FrameFlags, FrameHeader, MAX_FRAME_PAYLOAD, ScanOptions,
    SegmentHeader, TailKind, Wal, WalError, WalOptions, WalScan, segment_file_name,
};

/// Counts the bytes held through the global allocator, the most held since
/// the last [`reset`](counting::reset), and the largest single request.
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CURRENT: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static LARGEST: AtomicUsize = AtomicUsize::new(0);

    /// Forwards every call to [`System`] and counts what it hands out.
    pub struct Counting;

    fn requested(size: usize) {
        LARGEST.fetch_max(size, Ordering::Relaxed);
    }

    fn grew(by: usize) {
        let now = CURRENT.fetch_add(by, Ordering::Relaxed).wrapping_add(by);
        PEAK.fetch_max(now, Ordering::Relaxed);
    }

    fn shrank(by: usize) {
        CURRENT.fetch_sub(by, Ordering::Relaxed);
    }

    // SAFETY: every method passes its arguments unchanged to `System`, which
    // meets the trait's contract, and returns what `System` returned. The
    // counting touches atomics only and allocates nothing.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            requested(layout.size());
            // SAFETY: the caller meets `alloc`'s contract, which is `System`'s.
            let block = unsafe { System.alloc(layout) };
            if !block.is_null() {
                grew(layout.size());
            }
            block
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            requested(layout.size());
            // SAFETY: as in `alloc`.
            let block = unsafe { System.alloc_zeroed(layout) };
            if !block.is_null() {
                grew(layout.size());
            }
            block
        }

        unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
            // SAFETY: `block` was handed out by this allocator, so by
            // `System`, with `layout`.
            unsafe { System.dealloc(block, layout) };
            shrank(layout.size());
        }

        unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            requested(new_size);
            // SAFETY: as in `dealloc`, and the caller meets `realloc`'s
            // contract for `new_size`.
            let moved = unsafe { System.realloc(block, layout, new_size) };
            if !moved.is_null() {
                if new_size >= layout.size() {
                    grew(new_size - layout.size());
                } else {
                    shrank(layout.size() - new_size);
                }
            }
            moved
        }
    }

    /// The bytes held now.
    pub fn current() -> usize {
        CURRENT.load(Ordering::Relaxed)
    }

    /// The most bytes held since the last [`reset`].
    pub fn peak() -> usize {
        PEAK.load(Ordering::Relaxed)
    }

    /// The largest single request since the last [`reset`].
    pub fn largest() -> usize {
        LARGEST.load(Ordering::Relaxed)
    }

    /// Starts a measurement: the peak from what is held now, the largest
    /// request from 0.
    pub fn reset() {
        PEAK.store(current(), Ordering::Relaxed);
        LARGEST.store(0, Ordering::Relaxed);
    }
}

#[global_allocator]
static COUNTING: counting::Counting = counting::Counting;

/// The tests measure one at a time.
static TURN: Mutex<()> = Mutex::new(());

fn take_turn() -> MutexGuard<'static, ()> {
    TURN.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What `run` held above what was held before it, and its largest request.
fn measure<T>(run: impl FnOnce() -> T) -> (T, usize, usize) {
    counting::reset();
    let before = counting::current();
    let value = run();
    (
        value,
        counting::peak().saturating_sub(before),
        counting::largest(),
    )
}

const DATABASE: u128 = 0x3_19_88;
const MIB: usize = 1024 * 1024;
/// The records of the large group: 1 KiB each, 24 MiB in all.
const RECORD_BYTES: usize = 1024;
const RECORDS: usize = 24 * 1024;

fn record(index: usize) -> Vec<u8> {
    let mut bytes = vec![u8::try_from(index % 251).unwrap(); RECORD_BYTES];
    bytes[..8].copy_from_slice(&u64::try_from(index).unwrap().to_le_bytes());
    bytes
}

/// Writes a small group, the 24 MiB group and another small group, and
/// returns what writing the large one held.
fn write_large_log(dir: &Path) -> (usize, usize) {
    let wal = Wal::open(
        dir,
        WalOptions {
            durability: DurabilityMode::NoSync,
            ..WalOptions::new(DATABASE, 0)
        },
    )
    .unwrap();
    let mut small = wal.begin_group(TransactionId::new(1)).unwrap();
    small.push(b"Alix").unwrap();
    small.finish().unwrap();
    let records: Vec<Vec<u8>> = (0..RECORDS).map(record).collect();
    let ((), held, largest) = measure(|| {
        let mut group = wal.begin_group(TransactionId::new(2)).unwrap();
        for bytes in &records {
            group.push(bytes).unwrap();
        }
        group.finish().unwrap();
    });
    let mut small = wal.begin_group(TransactionId::new(3)).unwrap();
    small.push(b"Gus").unwrap();
    small.finish().unwrap();
    (held, largest)
}

/// Scans the log, checking every record of the large group, and returns
/// what the scan held and its largest request.
fn scan_large_log(dir: &Path, group_buffer_bytes: usize) -> (usize, usize) {
    let options = ScanOptions {
        group_buffer_bytes,
        ..ScanOptions::new(DATABASE, 0)
    };
    let (groups, held, largest) = measure(|| {
        let mut scan = WalScan::open(dir, options).unwrap();
        let mut groups = Vec::new();
        while let Some(mut frames) = scan.next_group().unwrap() {
            let large = frames.transaction_id() == TransactionId::new(2);
            let mut records = 0usize;
            let mut bytes = 0usize;
            while let Some(payload) = frames.next_payload().unwrap() {
                for chunk in payload.chunks(RECORD_BYTES) {
                    if large {
                        assert_eq!(chunk, record(records), "record {records} reads back");
                    }
                    records += 1;
                }
                bytes += payload.len();
            }
            groups.push((frames.transaction_id().as_u64(), bytes));
        }
        assert!(scan.finish().unwrap().is_clean());
        groups
    });
    assert_eq!(
        groups,
        [(1, 4), (2, RECORD_BYTES * RECORDS), (3, 3)],
        "every group reads back"
    );
    (held, largest)
}

#[test]
fn memory_stays_at_one_frame_for_a_large_group() {
    let _turn = take_turn();
    let dir = tempfile::tempdir().unwrap();
    let (written, largest) = write_large_log(dir.path());
    assert!(
        written < 3 * FRAME_TARGET,
        "writing a 24 MiB group held {written} bytes, about one 64 KiB frame expected"
    );
    assert!(largest <= FRAME_TARGET + 64, "largest request {largest}");

    // The default buffer of 8 MiB: the group is three times larger.
    let (held, largest) = scan_large_log(dir.path(), 8 * MIB);
    assert!(
        held < 8 * MIB + 4 * FRAME_TARGET,
        "scanning a 24 MiB group held {held} bytes, at most the 8 MiB buffer and a few frames \
         expected"
    );
    assert!(largest <= FRAME_TARGET + 64, "largest request {largest}");

    // A small buffer: the scan holds about one frame.
    let (held, _) = scan_large_log(dir.path(), 256 * 1024);
    assert!(
        held < 256 * 1024 + 4 * FRAME_TARGET,
        "with a 256 KiB buffer the scan held {held} bytes"
    );
}

#[test]
fn a_frame_length_over_the_limit_is_refused_before_allocation() {
    let _turn = take_turn();
    for (declared, last_segment) in [
        (MAX_FRAME_PAYLOAD + 1, false),
        (u32::MAX, false),
        (MAX_FRAME_PAYLOAD, false),
        (MAX_FRAME_PAYLOAD, true),
        (u32::MAX, true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let header = SegmentHeader {
            encrypted: false,
            database_id: DATABASE,
            first_lsn: 0,
            creation_time_ms: 88,
            salt: [0; 32],
            key_check: [0; 28],
        };
        let mut bytes = header.encode().to_vec();
        let frame = FrameHeader::new(declared, 0, 19, FrameFlags::FIRST_AND_LAST);
        bytes.extend_from_slice(&frame.encode());
        bytes.extend_from_slice(&[0x19; 88]);
        std::fs::write(dir.path().join(segment_file_name(0)), &bytes).unwrap();
        if !last_segment {
            // A later segment makes the first one sealed.
            let next = SegmentHeader {
                first_lsn: u64::try_from(bytes.len() - 128).unwrap(),
                ..header
            };
            std::fs::write(
                dir.path().join(segment_file_name(next.first_lsn)),
                next.encode(),
            )
            .unwrap();
        }
        let (outcome, _, largest) = measure(|| {
            let mut scan = WalScan::open(dir.path(), ScanOptions::new(DATABASE, 0))?;
            while scan.next_group()?.is_some() {}
            scan.finish()
        });
        let what = format!(
            "a frame declaring {declared} bytes in a {} segment",
            if last_segment { "last" } else { "sealed" }
        );
        assert!(
            largest < 4 * MIB,
            "{what}: the largest request was {largest} bytes"
        );
        if last_segment {
            let end = outcome.unwrap();
            assert_eq!(
                end.tail.map(|tail| tail.kind),
                Some(TailKind::Torn),
                "{what}"
            );
        } else {
            let error = outcome.unwrap_err();
            assert!(matches!(error, WalError::Damaged { .. }), "{what}: {error}");
        }
    }
}
