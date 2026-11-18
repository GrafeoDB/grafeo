//! What deleting data removes from memory and from the database file (#712).
//!
//! The binary's global allocator zeroes every block it frees and the old block of every
//! `realloc`, and keeps a table of the live blocks: a scan of the live blocks finds every copy of
//! a value Grafeo still holds, and a freed copy is gone. Each case writes values that hold a
//! marker built at run time (no copy of it lives in the test's code or query text: what the test
//! formats it is zeroed again), deletes them, and counts the marker after each step, in memory
//! and in the database's files. The oracle of every row is that count: the marker is found after
//! the write (the positive control) and nowhere before it (the negative control).
//!
//! The guarantee, as it holds or is to hold:
//!
//! - after `compact()` the database's files hold no deleted value;
//! - once the database is dropped, Grafeo holds no value of it in memory, and a reopen of the
//!   compacted file loads none;
//! - after the delete and `compact()`, Grafeo holds no deleted value in memory, except in the
//!   structures that keep values by design: the RDF term dictionary until `close()`, the
//!   change feed while CDC is on, and history while history retention keeps it;
//! - after `close()`, before the drop, Grafeo holds no value of the database in memory.
//!
//! Only Grafeo's own memory is scanned: the host language's copies (a binding's strings, its
//! allocator's free lists) are outside it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test deleted_values_leave_memory
//! ```

#![allow(
    unsafe_code,
    reason = "a global allocator that zeroes freed blocks and lists the live ones"
)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// The allocator that zeroes what it frees and lists what is live.
mod heap {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// The live blocks the table can hold; past three quarters of it the process aborts, as
    /// a full table would lose blocks from the scan.
    const SLOTS: usize = 1 << 20;
    const SLOT_BITS: u32 = 20;

    #[derive(Clone, Copy)]
    struct Block {
        address: usize,
        size: usize,
    }

    const EMPTY: Block = Block {
        address: 0,
        size: 0,
    };

    /// The table of live blocks: static, so keeping it allocates nothing.
    struct Table(UnsafeCell<[Block; SLOTS]>);

    // SAFETY: the table is read and written only through `Locked`, which holds `LOCK`.
    unsafe impl Sync for Table {}

    static TABLE: Table = Table(UnsafeCell::new([EMPTY; SLOTS]));
    static LOCK: AtomicBool = AtomicBool::new(false);
    static LIVE: AtomicUsize = AtomicUsize::new(0);

    /// Holds the table's lock while it lives.
    struct Locked;

    impl Locked {
        fn take() -> Self {
            while LOCK
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                std::hint::spin_loop();
            }
            Self
        }

        /// The table.
        fn table(&mut self) -> &mut [Block; SLOTS] {
            // SAFETY: the lock this guard holds makes this the only reference to the table,
            // for as long as the guard is borrowed.
            unsafe { &mut *TABLE.0.get() }
        }

        fn insert(&mut self, address: usize, size: usize) {
            if LIVE.fetch_add(1, Ordering::Relaxed) > SLOTS / 4 * 3 {
                std::process::abort();
            }
            let table = self.table();
            let mut slot = home(address);
            while table[slot].address != 0 {
                slot = (slot + 1) & (SLOTS - 1);
            }
            table[slot] = Block { address, size };
        }

        fn remove(&mut self, address: usize) {
            let table = self.table();
            let mut slot = home(address);
            loop {
                if table[slot].address == 0 {
                    return;
                }
                if table[slot].address == address {
                    break;
                }
                slot = (slot + 1) & (SLOTS - 1);
            }
            LIVE.fetch_sub(1, Ordering::Relaxed);
            // Backward-shift deletion keeps every block reachable from its home slot.
            let mut hole = slot;
            let mut next = (slot + 1) & (SLOTS - 1);
            while table[next].address != 0 {
                let at = home(table[next].address);
                let stays = if hole <= next {
                    hole < at && at <= next
                } else {
                    hole < at || at <= next
                };
                if !stays {
                    table[hole] = table[next];
                    hole = next;
                }
                next = (next + 1) & (SLOTS - 1);
            }
            table[hole] = EMPTY;
        }
    }

    impl Drop for Locked {
        fn drop(&mut self) {
            LOCK.store(false, Ordering::Release);
        }
    }

    fn home(address: usize) -> usize {
        let hash = ((address >> 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - SLOT_BITS);
        usize::try_from(hash).expect("a slot fits in usize")
    }

    /// The system allocator, zeroing what it frees and listing what is live.
    pub struct Zeroing;

    // SAFETY: every call goes to the system allocator with the caller's layout; a block is
    // zeroed only while it is still allocated, before it is freed.
    unsafe impl GlobalAlloc for Zeroing {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: the caller's contract, passed on.
            let block = unsafe { System.alloc(layout) };
            if !block.is_null() {
                Locked::take().insert(block as usize, layout.size());
            }
            block
        }

        unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
            Locked::take().remove(block as usize);
            // SAFETY: the block is allocated with `layout` (the caller's contract).
            unsafe { zero(block, layout.size()) };
            // SAFETY: the caller's contract, passed on.
            unsafe { System.dealloc(block, layout) };
        }

        unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: the caller's contract: `new_size` is non-zero and fits the alignment.
            let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
            // SAFETY: as `alloc`.
            let moved = unsafe { self.alloc(new_layout) };
            if !moved.is_null() {
                // SAFETY: both blocks are allocated and at least this long, and distinct.
                unsafe {
                    std::ptr::copy_nonoverlapping(block, moved, layout.size().min(new_size));
                }
                // SAFETY: as `dealloc`: the old block is zeroed before it is freed.
                unsafe { self.dealloc(block, layout) };
            }
            moved
        }
    }

    /// Zeroes `size` bytes at `block` with volatile writes. A plain `write_bytes` just before the
    /// free is a dead store the optimizer may drop (LLVM knows glibc's `free` as a free), so in an
    /// optimized build on Linux freed blocks kept their bytes, and a later block that got the same
    /// memory showed them as copies.
    ///
    /// # Safety
    ///
    /// `block` is allocated and at least `size` bytes long.
    unsafe fn zero(block: *mut u8, size: usize) {
        for offset in 0..size {
            // SAFETY: within the block (the caller's contract).
            unsafe { block.add(offset).write_volatile(0) };
        }
    }

    /// The copies of `needle` in the live blocks.
    pub fn copies(needle: &[u8]) -> usize {
        let mut locked = Locked::take();
        let mut found = 0;
        for block in locked.table().iter() {
            if block.address == 0 || block.size < needle.len() {
                continue;
            }
            let start = block.address as *const u8;
            let mut at = 0;
            while at + needle.len() <= block.size {
                // SAFETY: the block is live (no free can run while the lock is held) and
                // `at + needle.len()` is within it; volatile, as another thread may write it.
                let matches = needle
                    .iter()
                    .enumerate()
                    .all(|(i, byte)| unsafe { start.add(at + i).read_volatile() } == *byte);
                if matches {
                    found += 1;
                    at += needle.len();
                } else {
                    at += 1;
                }
            }
        }
        found
    }
}

#[global_allocator]
static GLOBAL: heap::Zeroing = heap::Zeroing;

/// One case at a time: another case's statements would run beside the scans.
static SERIAL: Mutex<()> = Mutex::new(());

static NEXT_MARKER: AtomicUsize = AtomicUsize::new(0);

/// A word no other value holds, on the stack: lowercase, so a text index's terms hold it too.
struct Marker {
    bytes: [u8; 48],
    len: usize,
    /// A vector no other value holds, made from the same run-time numbers.
    #[cfg(feature = "vector-index")]
    vector: [f32; 8],
}

impl Marker {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the clock is past 1970")
            .subsec_nanos();
        let count = NEXT_MARKER.fetch_add(1, Ordering::Relaxed);
        let mut bytes = [0u8; 48];
        let mut len = 0;
        for byte in b"zq"
            .iter()
            .chain(nanos.to_string().as_bytes())
            .chain(b"x")
            .chain(count.to_string().as_bytes())
            .chain(b"vexmarrow")
        {
            bytes[len] = *byte;
            len += 1;
        }
        #[cfg(feature = "vector-index")]
        let vector = {
            let base = 1000.0
                + f32::from(u16::try_from(nanos % 60_000).expect("below 60000")) / 7.0
                + f32::from(u16::try_from(count % 1000).expect("below 1000")) * 101.0;
            let mut vector = [0f32; 8];
            for (step, value) in (0u16..).zip(vector.iter_mut()) {
                *value = base + 0.125 * f32::from(step);
            }
            vector
        };
        Self {
            bytes,
            len,
            #[cfg(feature = "vector-index")]
            vector,
        }
    }

    fn word(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).expect("the marker is ASCII")
    }

    /// The bytes of the marker vector as a vector column and an index hold them.
    #[cfg(feature = "vector-index")]
    fn vector_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (at, value) in self.vector.iter().enumerate() {
            bytes[at * 4..at * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// How the case's database is opened.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Open {
    File,
    InMemory,
    #[cfg(feature = "encryption")]
    Encrypted,
    #[cfg(feature = "cdc")]
    Cdc,
    History,
}

/// What a case searches for: its marker word, or its marker vector's bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Needle {
    Word,
    #[cfg(feature = "vector-index")]
    Vector,
}

struct Case {
    open: Open,
    needle: Needle,
    /// Items written before the case's values, with texts below theirs.
    filler: usize,
    setup: fn(&GrafeoDB),
    write: fn(&GrafeoDB, &Marker),
    delete: fn(&GrafeoDB, &Marker),
    /// The write rolls its values back: it takes the positive control itself, before the
    /// rollback, and nothing holds the values after it.
    rolls_back: bool,
}

impl Case {
    /// The reporter's shape: three items with the marker in a text property.
    fn text(open: Open, filler: usize) -> Self {
        Self {
            open,
            needle: Needle::Word,
            filler,
            setup: |_| {},
            write: write_texts,
            delete: delete_items,
            rolls_back: false,
        }
    }
}

/// The marker's count after each step of a case.
#[derive(Debug, Default)]
struct Counts {
    written: usize,
    deleted: usize,
    compacted: usize,
    files_compacted: usize,
    /// The first 12 bytes of the marker in the files after `compact()`: a cut prefix (a zone
    /// map's bound) is a copy too.
    prefix_in_files_compacted: usize,
    closed: usize,
    files_closed: usize,
    dropped: usize,
    reopened: usize,
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

fn run(db: &GrafeoDB, query: &str, pairs: Vec<(&str, Value)>) {
    db.execute_with_params(query, params(pairs))
        .unwrap_or_else(|error| panic!("{query}: {error}"));
}

/// Runs a statement whose text holds the marker, then zeroes the text.
fn run_literal(run: impl FnOnce(&str), text: String) {
    let mut bytes = text.into_bytes();
    run(std::str::from_utf8(&bytes).expect("UTF-8"));
    bytes.fill(0);
}

/// A text holding the marker, made from bytes that are zeroed again.
fn text_value(item: usize, marker: &Marker) -> Value {
    let mut text = format!(
        "Moment {item}: Alix told Gus about {} at length.",
        marker.word()
    )
    .into_bytes();
    let value = Value::from(std::str::from_utf8(&text).expect("UTF-8"));
    text.fill(0);
    value
}

fn write_texts(db: &GrafeoDB, marker: &Marker) {
    for item in 0..3 {
        run(
            db,
            "INSERT (:Item {id: $id, text: $text})",
            vec![
                ("id", Value::from(format!("M{item}"))),
                ("text", text_value(item, marker)),
            ],
        );
    }
}

fn delete_items(db: &GrafeoDB, _: &Marker) {
    run(
        db,
        "MATCH (n:Item) WHERE n.id IN ['M0', 'M1', 'M2'] DETACH DELETE n",
        vec![],
    );
}

/// Occurrences of `needle` and of its first 12 bytes in the files under `dir`.
fn in_files(dir: &Path, needle: &[u8]) -> (usize, usize) {
    let mut found = (0, 0);
    for entry in std::fs::read_dir(dir).expect("the folder lists").flatten() {
        let path = entry.path();
        if path.is_dir() {
            let (whole, prefix) = in_files(&path, needle);
            found.0 += whole;
            found.1 += prefix;
        } else if let Ok(bytes) = std::fs::read(&path) {
            found.0 += bytes.windows(needle.len()).filter(|w| *w == needle).count();
            found.1 += bytes.windows(12).filter(|w| *w == &needle[..12]).count();
        }
    }
    found
}

fn config(open: Open, path: &Path) -> Config {
    let config = Config::persistent(path);
    match open {
        Open::File | Open::InMemory => config,
        #[cfg(feature = "encryption")]
        Open::Encrypted => config.with_encryption(grafeo_engine::config::EncryptionConfig::new(
            std::sync::Arc::new(grafeo_common::encryption::KeyChain::new([7; 32])),
        )),
        #[cfg(feature = "cdc")]
        Open::Cdc => config.with_cdc(),
        Open::History => {
            config.with_history_retention(grafeo_engine::config::HistoryRetention::Forever)
        }
    }
}

/// Runs `case` and counts its marker after each step.
fn counts(case: &Case) -> Counts {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let marker = Marker::new();
    #[cfg(feature = "vector-index")]
    let vector_bytes = marker.vector_bytes();
    let needle: &[u8] = match case.needle {
        Needle::Word => &marker.bytes[..marker.len],
        #[cfg(feature = "vector-index")]
        Needle::Vector => &vector_bytes,
    };
    let dir = tempfile::tempdir().expect("a temporary folder");
    let path = dir.path().join("deleted.grafeo");
    let open = || match case.open {
        Open::InMemory => GrafeoDB::new_in_memory(),
        other => GrafeoDB::with_config(config(other, &path)).expect("the database opens"),
    };

    let mut db = open();
    (case.setup)(&db);
    for item in 0..case.filler {
        run(
            &db,
            "INSERT (:Item {id: $id, text: $text})",
            vec![
                ("id", Value::from(format!("I{item}"))),
                (
                    "text",
                    Value::from(format!("Item {item}: Gus said something.")),
                ),
            ],
        );
    }
    assert_eq!(heap::copies(needle), 0, "the negative control: no copy yet");
    let mut counts = Counts::default();
    (case.write)(&db, &marker);
    counts.written = heap::copies(needle);
    assert!(
        counts.written > 0 || case.rolls_back,
        "the positive control: the scan finds the written values"
    );
    (case.delete)(&db, &marker);
    counts.deleted = heap::copies(needle);
    db.compact().expect("compact()");
    counts.compacted = heap::copies(needle);
    (counts.files_compacted, counts.prefix_in_files_compacted) = in_files(dir.path(), needle);
    db.close().expect("close()");
    counts.closed = heap::copies(needle);
    counts.files_closed = in_files(dir.path(), needle).0;
    drop(db);
    counts.dropped = heap::copies(needle);
    if case.open != Open::InMemory {
        let db = open();
        counts.reopened = heap::copies(needle);
        drop(db);
    }
    counts
}

/// The values this case deleted are gone from memory after `compact()`.
fn compact_frees(case: &Case) {
    let counts = counts(case);
    assert_eq!(counts.compacted, 0, "after compact(): {counts:?}");
}

// ---------------------------------------------------------------------------------------------
// The file and the drop: hold for every shape.
// ---------------------------------------------------------------------------------------------

/// Every shape this file writes, with the opens that keep nothing by design.
fn every_shape() -> Vec<(&'static str, Case)> {
    let mut shapes = vec![
        ("text property", Case::text(Open::File, 1000)),
        ("in memory", Case::text(Open::InMemory, 100)),
        ("named graph", named_graph()),
        ("list", list()),
        ("map", map()),
        ("edge property", edge_property()),
        ("overwritten", overwritten()),
        ("rolled back", rolled_back()),
        ("literal in the query text", literal_insert()),
        ("property index on the text", property_index()),
    ];
    shapes.push(("the marker first", marker_first()));
    #[cfg(feature = "encryption")]
    shapes.push(("encrypted", Case::text(Open::Encrypted, 100)));
    #[cfg(feature = "text-index")]
    shapes.push(("text index", text_index()));
    #[cfg(feature = "vector-index")]
    {
        shapes.push(("vector index", vector_index(None)));
        shapes.push((
            "scalar-quantized vector index",
            vector_index(Some("scalar")),
        ));
        shapes.push((
            "binary-quantized vector index",
            vector_index(Some("binary")),
        ));
    }
    #[cfg(all(feature = "triple-store", feature = "sparql"))]
    shapes.push(("RDF literal", rdf_literal()));
    #[cfg(feature = "cdc")]
    shapes.push(("CDC on", Case::text(Open::Cdc, 100)));
    shapes
}

/// After `compact()` the database's files hold no deleted value, not even a cut prefix of one
/// (the marker first in its value, so a zone map's prefix would hold part of it).
#[test]
fn the_files_hold_no_deleted_value_after_compact() {
    for (shape, case) in every_shape() {
        let counts = counts(&case);
        assert_eq!(
            (
                counts.files_compacted,
                counts.prefix_in_files_compacted,
                counts.files_closed
            ),
            (0, 0, 0),
            "{shape}: {counts:?}"
        );
    }
}

/// glibc's `free`, interposed in this binary: every call goes on to glibc's own (`__libc_free`,
/// the same function), and the block the guard watches is read first, while it is still
/// allocated. The symbol `free` is a jump in assembly to `watched`, so the optimizer never sees
/// a body for it: the probe's call to `free` through the system allocator stays the call to
/// glibc's free the optimizer knows, and would drop a plain zeroing before, in every profile (a
/// Rust definition of `free` in this crate made the stores live in a single-unit release build).
/// `watched` never allocates, locks or panics: it runs for every free in this binary, the
/// threads' teardown included.
#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod watched_free {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The block the guard watches, or 0.
    pub static WATCHED: AtomicUsize = AtomicUsize::new(0);
    /// The bytes of the watched block that still held the marker byte when it reached `free`;
    /// `usize::MAX` until it does.
    pub static STALE: AtomicUsize = AtomicUsize::new(usize::MAX);
    /// The size of the watched block, and the marker byte it was filled with.
    pub const SIZE: usize = 96;
    pub const MARKER: u8 = b'm';

    unsafe extern "C" {
        fn __libc_free(block: *mut c_void);
    }

    #[cfg(target_arch = "x86_64")]
    std::arch::global_asm!(
        ".globl free",
        ".type free, @function",
        "free:",
        "jmp {watched}",
        watched = sym watched,
    );
    #[cfg(target_arch = "aarch64")]
    std::arch::global_asm!(
        ".globl free",
        ".type free, %function",
        "free:",
        "b {watched}",
        watched = sym watched,
    );

    /// Frees `block` with glibc's `free`, after counting the marker bytes of the watched block.
    ///
    /// # Safety
    ///
    /// `free`'s own: `block` is null or a live block of glibc's allocator.
    unsafe extern "C" fn watched(block: *mut c_void) {
        if !block.is_null() && block as usize == WATCHED.load(Ordering::SeqCst) {
            let bytes = block.cast::<u8>();
            let stale = (0..SIZE)
                // SAFETY: the watched block is SIZE bytes long and not yet freed.
                .filter(|&offset| unsafe { bytes.add(offset).read_volatile() } == MARKER)
                .count();
            STALE.store(stale, Ordering::SeqCst);
            WATCHED.store(0, Ordering::SeqCst);
        }
        // SAFETY: the caller's contract, passed on to glibc's own free.
        unsafe { __libc_free(block) };
    }
}

/// The probe's own guarantee: a block reaches the system allocator's free zeroed. Without it
/// the counts after the drop and the reopen read freed memory that kept its bytes, as an
/// optimized build did when the zeroing was a plain store the optimizer removed before `free`.
/// The guard fills a block with the marker byte, frees it through the global allocator and
/// reads it in glibc's `free`, where the probe hands it over (see `watched_free`): no reuse of
/// the block is needed, so nothing depends on what the allocator does after. Linux with glibc
/// only (x86_64 and aarch64): there the optimizer knows `free`; the Windows heap's free is a call
/// it cannot see into.
#[test]
#[cfg_attr(
    not(all(
        target_os = "linux",
        target_env = "gnu",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )),
    ignore = "the guard interposes glibc's free, on x86_64 and aarch64"
)]
fn the_probe_zeroes_what_it_frees() {
    #[cfg(all(
        target_os = "linux",
        target_env = "gnu",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    {
        use std::alloc::{Layout, alloc, dealloc};
        use std::sync::atomic::Ordering;

        use watched_free::{MARKER, SIZE, STALE, WATCHED};

        let layout = Layout::from_size_align(SIZE, 8).unwrap();
        // SAFETY: a non-zero layout; the block is freed with it below.
        let block = unsafe { alloc(layout) };
        assert!(!block.is_null());
        for offset in 0..SIZE {
            // SAFETY: within the block; volatile, so the fill is not a dead store before the free.
            unsafe { block.add(offset).write_volatile(MARKER) };
        }
        STALE.store(usize::MAX, Ordering::SeqCst);
        WATCHED.store(block as usize, Ordering::SeqCst);
        // SAFETY: allocated above with `layout`.
        unsafe { dealloc(block, layout) };
        let stale = STALE.load(Ordering::SeqCst);
        assert_ne!(
            stale,
            usize::MAX,
            "the block's free went through the watched free"
        );
        assert_eq!(stale, 0, "{stale} bytes of the block reached free unzeroed");
    }
}

/// Once the database is dropped Grafeo holds none of its values, and a reopen of the compacted
/// file loads none.
#[test]
fn no_deleted_value_is_left_once_the_database_is_dropped() {
    for (shape, case) in every_shape() {
        let counts = counts(&case);
        assert_eq!(
            (counts.dropped, counts.reopened),
            (0, 0),
            "{shape}: {counts:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// In memory after compact(): the shapes that free their values.
// ---------------------------------------------------------------------------------------------

/// A larger text written after the delete, so no structure keeps the deleted ones as a bound.
#[test]
fn a_text_with_a_larger_one_written_after_it_is_freed_by_compact() {
    compact_frees(&Case {
        delete: |db, marker| {
            delete_items(db, marker);
            run(
                db,
                "INSERT (:Item {id: 'Z', text: 'zz, Vincent said'})",
                vec![],
            );
        },
        ..Case::text(Open::File, 100)
    });
}

/// An edge property, the graph's only edge.
#[test]
fn an_edge_property_is_freed_by_compact() {
    compact_frees(&edge_property());
}

/// A text in a named graph, the graph's only items.
#[test]
fn a_text_in_a_named_graph_is_freed_by_compact() {
    compact_frees(&named_graph());
}

/// A text replaced by a larger one with SET.
#[test]
fn an_overwritten_text_is_freed_by_compact() {
    compact_frees(&overwritten());
}

/// A property index on the text drops the keys of deleted items (with a larger seed text, so
/// the column's bound keeps none of the case's texts).
#[test]
fn a_property_index_frees_the_keys_of_deleted_items() {
    compact_frees(&property_index());
}

/// A text index drops the terms only the deleted items used (with the larger seed text too).
#[cfg(feature = "text-index")]
#[test]
fn a_text_index_frees_the_terms_of_deleted_items() {
    compact_frees(&text_index());
}

/// A vector index without quantization frees the deleted vectors.
#[cfg(feature = "vector-index")]
#[test]
fn a_vector_index_frees_deleted_vectors() {
    compact_frees(&vector_index(None));
}

/// A binary-quantized vector index frees the deleted vectors and their codes.
#[cfg(feature = "vector-index")]
#[test]
fn a_binary_quantized_vector_index_frees_deleted_vectors() {
    compact_frees(&vector_index(Some("binary")));
}

// ---------------------------------------------------------------------------------------------
// In memory after compact(): the reporter's scenario and its shapes, then the ones that still
// keep a deleted value (expected failures).
// ---------------------------------------------------------------------------------------------

/// The reporter's scenario: the deleted texts are the column's largest, which the column's bound
/// kept until it named rows instead of holding values.
#[test]
fn deleted_texts_are_freed_by_compact() {
    compact_frees(&Case::text(Open::File, 1000));
}

/// The same in a database without a file.
#[test]
fn deleted_texts_are_freed_by_compact_in_memory() {
    compact_frees(&Case::text(Open::InMemory, 100));
}

/// The same in an encrypted database: what it holds in memory is not encrypted.
#[cfg(feature = "encryption")]
#[test]
fn deleted_texts_are_freed_by_compact_when_encrypted() {
    compact_frees(&Case::text(Open::Encrypted, 100));
}

/// A text inside a list.
#[test]
fn a_deleted_list_is_freed_by_compact() {
    compact_frees(&list());
}

/// A text inside a map.
#[test]
fn a_deleted_map_is_freed_by_compact() {
    compact_frees(&map());
}

/// An insert rolled back: never committed, so nothing holds its values once the rollback
/// returns, the column's bound included (it names rows and owns no value).
#[test]
fn a_rolled_back_text_is_freed() {
    let counts = counts(&rolled_back());
    assert_eq!(counts.written, 0, "after the rollback: {counts:?}");
    assert_eq!(counts.compacted, 0, "after compact(): {counts:?}");
}

/// Texts written as literals: the plan cache keeps each statement's text and literals, and
/// `compact()` empties it.
#[test]
fn texts_written_as_literals_are_freed_by_compact() {
    compact_frees(&literal_insert());
}

/// A scalar-quantized index below its training threshold keeps its training samples as ids,
/// so a deleted vector is freed with its entry and never trains the quantizer.
#[cfg(feature = "vector-index")]
#[test]
fn a_scalar_quantized_vector_index_frees_deleted_vectors() {
    compact_frees(&vector_index(Some("scalar")));
}

// ---------------------------------------------------------------------------------------------
// The structures that keep deleted values by design.
// ---------------------------------------------------------------------------------------------

/// The RDF term dictionary keeps a deleted literal while the database is open: present after
/// `compact()`, gone once `close()` releases the store, and not loaded by the reopen (the file
/// holds none).
#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn the_rdf_term_dictionary_keeps_a_deleted_literal_while_open() {
    let counts = counts(&rdf_literal());
    assert!(counts.compacted > 0, "kept after compact(): {counts:?}");
    assert_eq!(counts.closed, 0, "released by close(): {counts:?}");
    assert_eq!(
        (counts.files_compacted, counts.reopened),
        (0, 0),
        "{counts:?}"
    );
}

/// The change feed keeps every deleted value while the database is open with CDC on; it is not
/// written to the file, so a reopen holds none.
#[cfg(feature = "cdc")]
#[test]
fn the_change_feed_keeps_deleted_values_while_open() {
    let counts = counts(&Case::text(Open::Cdc, 100));
    assert_eq!(counts.compacted, counts.written, "{counts:?}");
    assert_eq!(
        (counts.files_compacted, counts.reopened),
        (0, 0),
        "{counts:?}"
    );
}

/// History kept forever keeps every deleted value, in memory and in the file, and a reopen
/// loads it again.
#[test]
fn history_kept_forever_keeps_deleted_values() {
    let counts = counts(&Case::text(Open::History, 100));
    assert_eq!(counts.compacted, counts.written, "{counts:?}");
    assert_eq!(counts.files_closed, counts.written, "{counts:?}");
    assert_eq!(counts.reopened, counts.written, "{counts:?}");
}

// ---------------------------------------------------------------------------------------------
// close().
// ---------------------------------------------------------------------------------------------

/// After `close()`, before the drop, Grafeo holds no value of the database: the store is
/// released, not only persisted. Live values (none deleted) are the oracle.
#[test]
fn close_releases_the_values_before_the_drop() {
    let counts = counts(&Case {
        delete: |_, _| {},
        ..Case::text(Open::File, 100)
    });
    assert_eq!(counts.closed, 0, "after close(): {counts:?}");
}

/// Writes the reporter's texts, keeps what `hold` returns alive across `close()`, and counts
/// the marker after the close, before the holder and the database drop: what a session, a
/// stream or a view handed out before the close holds of the database once it is closed.
fn closed_while_held(hold: fn(&GrafeoDB) -> Box<dyn std::any::Any>) -> usize {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let marker = Marker::new();
    let needle = &marker.bytes[..marker.len];
    let dir = tempfile::tempdir().expect("a temporary folder");
    let path = dir.path().join("held.grafeo");
    let db = GrafeoDB::with_config(
        config(Open::File, &path)
            .with_history_retention(grafeo_engine::config::HistoryRetention::Forever),
    )
    .expect("the database opens");
    write_texts(&db, &marker);
    assert!(
        heap::copies(needle) > 0,
        "the positive control: the scan finds the written values"
    );
    let held = hold(&db);
    db.close().expect("close()");
    let closed = heap::copies(needle);
    drop(held);
    drop(db);
    closed
}

/// A session and the stream it lends (`Session::execute_streaming`, which borrows the session,
/// so it cannot be one of the boxed holders above) held across `close()`: neither keeps a value.
#[test]
fn close_releases_the_values_a_borrowed_stream_held() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let marker = Marker::new();
    let needle = &marker.bytes[..marker.len];
    let dir = tempfile::tempdir().expect("a temporary folder");
    let path = dir.path().join("borrowed.grafeo");
    let db = GrafeoDB::with_config(
        config(Open::File, &path)
            .with_history_retention(grafeo_engine::config::HistoryRetention::Forever),
    )
    .expect("the database opens");
    write_texts(&db, &marker);
    assert!(
        heap::copies(needle) > 0,
        "the positive control: the scan finds the written values"
    );
    let session = db.session();
    let stream = session
        .execute_streaming("MATCH (i:Item) RETURN i.text")
        .expect("the stream");
    db.close().expect("close()");
    let closed = heap::copies(needle);
    drop(stream);
    drop(session);
    drop(db);
    assert_eq!(
        closed, 0,
        "the session and its stream held {closed} copies after close()"
    );
}

/// Dropping the database without `close()`, then the session it handed out last: once the
/// session goes, nothing of the database is left (#712).
#[test]
fn dropping_the_database_then_the_session_last_leaves_nothing() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let marker = Marker::new();
    let needle = &marker.bytes[..marker.len];
    let dir = tempfile::tempdir().expect("a temporary folder");
    let db = GrafeoDB::with_config(config(Open::File, &dir.path().join("dropped.grafeo")))
        .expect("the database opens");
    write_texts(&db, &marker);
    assert!(heap::copies(needle) > 0, "the positive control");
    let session = db.session();
    drop(db);
    drop(session);
    assert_eq!(
        heap::copies(needle),
        0,
        "after the session, the last holder, went"
    );
}

/// A session made before `close()` holds nothing of the database after it.
#[test]
fn close_releases_the_values_a_session_could_reach() {
    assert_eq!(closed_while_held(|db| Box::new(db.session())), 0);
}

/// A stream opened before `close()` and never read holds nothing of the database after it.
#[test]
fn close_releases_the_values_an_unread_stream_could_reach() {
    assert_eq!(
        closed_while_held(|db| {
            let session = db.session();
            let stream = session
                .execute_streaming_owned("MATCH (i:Item) RETURN i.text")
                .expect("the stream opens");
            Box::new((stream, session))
        }),
        0
    );
}

/// A view of an earlier epoch made before `close()`, which sees the texts as they were before
/// they were overwritten (history kept), holds nothing of the database after it.
#[test]
fn close_releases_the_values_a_view_could_reach() {
    assert_eq!(
        closed_while_held(|db| {
            let earlier = db.current_epoch();
            db.execute("MATCH (i:Item) SET i.text = 'overwritten'")
                .expect("the overwrite");
            Box::new(db.as_of(earlier).expect("the view opens"))
        }),
        0
    );
}

// ---------------------------------------------------------------------------------------------
// The shapes.
// ---------------------------------------------------------------------------------------------

fn named_graph() -> Case {
    Case {
        setup: |db| run(db, "CREATE GRAPH g", vec![]),
        write: |db, marker| {
            let session = db.session();
            session.execute("USE GRAPH g").expect("USE GRAPH");
            for item in 0..3 {
                session
                    .execute_with_params(
                        "INSERT (:Item {id: $id, text: $text})",
                        params(vec![
                            ("id", Value::from(format!("M{item}"))),
                            ("text", text_value(item, marker)),
                        ]),
                    )
                    .expect("the insert");
            }
        },
        delete: |db, _| {
            let session = db.session();
            session.execute("USE GRAPH g").expect("USE GRAPH");
            session
                .execute("MATCH (n:Item) WHERE n.id IN ['M0', 'M1', 'M2'] DETACH DELETE n")
                .expect("the delete");
        },
        ..Case::text(Open::File, 0)
    }
}

fn list() -> Case {
    Case {
        write: |db, marker| {
            run(
                db,
                "INSERT (:Item {id: 'M0', tags: [$text, 'Mia']})",
                vec![("text", text_value(0, marker))],
            );
        },
        ..Case::text(Open::File, 100)
    }
}

fn map() -> Case {
    Case {
        write: |db, marker| {
            run(
                db,
                "INSERT (:Item {id: 'M0', note: {said: $text, by: 'Mia'}})",
                vec![("text", text_value(0, marker))],
            );
        },
        ..Case::text(Open::File, 100)
    }
}

fn edge_property() -> Case {
    Case {
        write: |db, marker| {
            run(
                db,
                "INSERT (:Item {id: 'M0'})-[:TOLD {text: $text}]->(:Item {id: 'M1'})",
                vec![("text", text_value(0, marker))],
            );
        },
        ..Case::text(Open::File, 100)
    }
}

fn overwritten() -> Case {
    Case {
        delete: |db, _| {
            run(
                db,
                "MATCH (n:Item) WHERE n.id IN ['M0', 'M1', 'M2'] SET n.text = 'zz, Jules said'",
                vec![],
            );
        },
        ..Case::text(Open::File, 100)
    }
}

fn rolled_back() -> Case {
    Case {
        write: |db, marker| {
            let mut session = db.session();
            session.begin_transaction().expect("BEGIN");
            for item in 0..3 {
                session
                    .execute_with_params(
                        "INSERT (:Item {id: $id, text: $text})",
                        params(vec![
                            ("id", Value::from(format!("M{item}"))),
                            ("text", text_value(item, marker)),
                        ]),
                    )
                    .expect("the insert");
            }
            assert!(
                heap::copies(&marker.bytes[..marker.len]) > 0,
                "the positive control: the scan finds the values before the rollback"
            );
            session.rollback().expect("ROLLBACK");
        },
        delete: |_, _| {},
        rolls_back: true,
        ..Case::text(Open::File, 100)
    }
}

fn literal_insert() -> Case {
    Case {
        write: |db, marker| {
            for item in 0..3 {
                run_literal(
                    |text| {
                        db.execute(text).expect("the insert");
                    },
                    format!(
                        "INSERT (:Item {{id: 'M{item}', text: 'Moment {item}: Alix told Gus about {} at length.'}})",
                        marker.word()
                    ),
                );
            }
        },
        ..Case::text(Open::File, 100)
    }
}

/// A text larger than every other the column holds, so the column's bound keeps none of the
/// case's texts and an index row tests the index alone.
fn larger_seed(db: &GrafeoDB) {
    run(
        db,
        "INSERT (:Item {id: 'seed', text: 'zz, Butch said hello'})",
        vec![],
    );
}

fn property_index() -> Case {
    Case {
        setup: |db| {
            larger_seed(db);
            run(
                db,
                "CREATE INDEX item_text FOR (n:Item) ON (n.text)",
                vec![],
            );
        },
        ..Case::text(Open::File, 100)
    }
}

/// The marker first in each text, and the texts the column's largest: a zone map's cut
/// prefix of the largest would hold the marker's first bytes.
fn marker_first() -> Case {
    Case {
        write: |db, marker| {
            for item in 0..3 {
                let mut text =
                    format!("{} told by Alix, moment {item}.", marker.word()).into_bytes();
                let value = Value::from(std::str::from_utf8(&text).expect("UTF-8"));
                text.fill(0);
                run(
                    db,
                    "INSERT (:Item {id: $id, text: $text})",
                    vec![("id", Value::from(format!("M{item}"))), ("text", value)],
                );
            }
        },
        ..Case::text(Open::File, 300)
    }
}

#[cfg(feature = "text-index")]
fn text_index() -> Case {
    Case {
        setup: |db| {
            larger_seed(db);
            db.create_text_index("Item", "text")
                .expect("the text index");
        },
        ..Case::text(Open::File, 100)
    }
}

#[cfg(feature = "vector-index")]
fn vector_index(quantization: Option<&'static str>) -> Case {
    fn write(db: &GrafeoDB, marker: &Marker) {
        for item in 0..3 {
            run(
                db,
                "INSERT (:Item {id: $id, emb: $emb})",
                vec![
                    ("id", Value::from(format!("M{item}"))),
                    ("emb", Value::Vector(marker.vector.to_vec().into())),
                ],
            );
        }
    }
    fn setup(db: &GrafeoDB, quantization: Option<&str>) {
        for item in 0..100u16 {
            let vector: Vec<f32> = (0..8u16).map(|d| f32::from(item) + f32::from(d)).collect();
            run(
                db,
                "INSERT (:Item {id: $id, emb: $emb})",
                vec![
                    ("id", Value::from(format!("V{item}"))),
                    ("emb", Value::Vector(vector.into())),
                ],
            );
        }
        db.create_vector_index(
            "Item",
            "emb",
            Some(8),
            Some("cosine"),
            None,
            None,
            quantization,
        )
        .expect("the vector index");
    }
    Case {
        needle: Needle::Vector,
        filler: 0,
        setup: match quantization {
            None => |db| setup(db, None),
            Some("scalar") => |db| setup(db, Some("scalar")),
            Some("binary") => |db| setup(db, Some("binary")),
            Some(other) => panic!("no case for {other} quantization"),
        },
        write,
        ..Case::text(Open::File, 0)
    }
}

#[cfg(all(feature = "triple-store", feature = "sparql"))]
fn rdf_literal() -> Case {
    fn data(verb: &str, marker: &Marker) -> String {
        format!(
            "{verb} DATA {{ <http://example.org/alix> <http://example.org/told> \"Alix told Gus about {}.\" }}",
            marker.word()
        )
    }
    Case {
        write: |db, marker| {
            run_literal(
                |text| {
                    db.execute_sparql(text).expect("INSERT DATA");
                },
                data("INSERT", marker),
            );
        },
        delete: |db, marker| {
            run_literal(
                |text| {
                    db.execute_sparql(text).expect("DELETE DATA");
                },
                data("DELETE", marker),
            );
        },
        ..Case::text(Open::File, 0)
    }
}

/// Every public way to get a session, a view or a stream of a database, each held across
/// `close()`: none holds anything of the database after it (#712). A new way to get one is a
/// new row here.
#[test]
#[expect(
    deprecated,
    reason = "session_read_only() is one of the ways a session is handed out until 0.7.0 removes it"
)]
fn close_releases_the_values_whatever_was_handed_out() {
    let holders: Vec<(&str, fn(&GrafeoDB) -> Box<dyn std::any::Any>)> = vec![
        ("session()", |db| Box::new(db.session())),
        ("session_read_only()", |db| Box::new(db.session_read_only())),
        ("session_with_role()", |db| {
            Box::new(db.session_with_role(grafeo_engine::auth::Role::ReadWrite))
        }),
        ("session_with_identity()", |db| {
            Box::new(db.session_with_identity(grafeo_engine::auth::Identity::anonymous()))
        }),
        #[cfg(feature = "cdc")]
        ("session_with_cdc()", |db| {
            Box::new(db.session_with_cdc(true))
        }),
        ("graph(...).session()", |db| {
            db.create_graph("trips").expect("the graph");
            Box::new(
                db.graph("trips")
                    .expect("the handle")
                    .session()
                    .expect("the session"),
            )
        }),
        ("as_of()", |db| {
            Box::new(db.as_of(db.current_epoch()).expect("the view"))
        }),
        ("execute_streaming()", |db| {
            Box::new(
                db.execute_streaming("MATCH (i:Item) RETURN i.text")
                    .expect("the stream"),
            )
        }),
        ("session().execute_streaming_owned()", |db| {
            let session = db.session();
            let stream = session
                .execute_streaming_owned("MATCH (i:Item) RETURN i.text")
                .expect("the stream");
            Box::new((stream, session))
        }),
    ];
    let mut kept = Vec::new();
    for (way, hold) in holders {
        let copies = closed_while_held(hold);
        if copies > 0 {
            kept.push(format!("{way}: {copies} copies"));
        }
    }
    assert!(kept.is_empty(), "held after close(): {}", kept.join(", "));
}

/// What a reader holds; dropped after the count, before the database.
trait Held {}
impl<T> Held for T {}

/// The documented exception to "`close()` releases everything" until 0.7.0 (#712): a snapshot,
/// the store a snapshot shares, and the node and edge iterators, each held across `close()`,
/// keep the graph's values while they live, and nothing is left once they drop. The 0.7.0 fix,
/// which makes them fail with `T007`, turns the first count of each row to zero.
#[test]
fn a_snapshot_or_iterator_held_across_close_keeps_the_values_until_it_drops() {
    type Hold = for<'a> fn(&'a GrafeoDB) -> Box<dyn Held + 'a>;
    let holders: Vec<(&str, Hold)> = vec![
        ("snapshot()", |db| {
            Box::new(db.snapshot().expect("the snapshot"))
        }),
        ("snapshot().shared()", |db| {
            Box::new(db.snapshot().expect("the snapshot").shared())
        }),
        ("projection()", |db| {
            db.create_projection(
                "items",
                grafeo_core::graph::ProjectionSpec::new().with_node_labels(["Item"]),
            )
            .expect("the projection");
            Box::new(db.projection("items").expect("the projection"))
        }),
        ("graph(...).snapshot()", |db| {
            db.create_graph("trips").expect("the graph");
            // The named graph gets a copy of a text: the default graph's go at `close()`.
            let text = db
                .execute("MATCH (i:Item) RETURN i.text LIMIT 1")
                .expect("a text")
                .rows()[0][0]
                .clone();
            db.graph("trips")
                .expect("the handle")
                .session()
                .expect("the session")
                .execute_with_params("INSERT (:Item {text: $text})", params(vec![("text", text)]))
                .expect("the copy");
            Box::new(
                db.graph("trips")
                    .expect("the handle")
                    .snapshot()
                    .expect("the snapshot"),
            )
        }),
        ("as_of().snapshot()", |db| {
            Box::new(db.as_of(db.current_epoch()).expect("the view").snapshot())
        }),
        ("iter_nodes()", |db| {
            Box::new(db.iter_nodes().expect("the iterator"))
        }),
        ("iter_edges()", |db| {
            db.execute("MATCH (i:Item) WITH i LIMIT 1 INSERT (i)-[:NEXT]->(:Item)")
                .expect("an edge");
            Box::new(db.iter_edges().expect("the iterator"))
        }),
    ];
    let mut wrong = Vec::new();
    for (way, hold) in holders {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let marker = Marker::new();
        let needle = &marker.bytes[..marker.len];
        let dir = tempfile::tempdir().expect("a temporary folder");
        let db = GrafeoDB::with_config(config(Open::File, &dir.path().join("held.grafeo")))
            .expect("the database opens");
        write_texts(&db, &marker);
        let held = hold(&db);
        db.close().expect("close()");
        let while_held = heap::copies(needle);
        drop(held);
        let after = heap::copies(needle);
        drop(db);
        if while_held == 0 || after > 0 {
            wrong.push(format!(
                "{way}: {while_held} copies while held, {after} after"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join(", "));
}
