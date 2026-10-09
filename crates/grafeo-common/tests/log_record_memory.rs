//! Log records allocate what they hold, never what a length claims.
//!
//! A WAL frame passed its CRC, but a record inside it can still be damaged
//! (or written by a newer release, or crafted). The reader checks every
//! length against the bytes present before it allocates: a damaged string
//! length, list count or value length in a small payload must cost at most
//! a few small allocations, not the 64 MiB it claims. The writer, for its
//! part, refuses a record over the maximum from the sizes of its values,
//! before it encodes one.
//!
//! The global allocator of this test binary records the largest single
//! request made by the thread that measures, and forwards every call to the
//! system allocator.
//!
//! Not under Miri on a Windows target: std's Windows `System` allocator
//! reads the header of an over-aligned block through the caller's pointer,
//! which Stacked Borrows refuses (the test harness frees such a block), and
//! without a global allocator of its own a program never reaches that code
//! under Miri. CI runs Miri on Linux, where this file runs.

#![cfg(not(all(miri, windows)))]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this only forwards to System"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use grafeo_common::change::DataOp;
use grafeo_common::storage::{LogRecord, MAX_RECORD_BYTES, encoded_len, read_log_records};
use grafeo_common::types::{NodeId, Value};

thread_local! {
    /// Whether this thread is measuring.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    /// The largest single request of this thread while it measures.
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

fn requested(size: usize) {
    // `try_with`: a thread that is shutting down has no locals left.
    let _ = MEASURING.try_with(|measuring| {
        if measuring.get() {
            let _ = LARGEST.try_with(|largest| largest.set(largest.get().max(size)));
        }
    });
}

/// Forwards every call to [`System`] and records the size of each request.
struct Recording;

// SAFETY: every method passes its arguments unchanged to `System`, which
// meets the trait's contract, and returns what `System` returned. The
// recording touches const-initialized thread locals only and allocates
// nothing.
unsafe impl GlobalAlloc for Recording {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        requested(layout.size());
        // SAFETY: the caller meets `alloc`'s contract, which is `System`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        requested(layout.size());
        // SAFETY: as in `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        // SAFETY: `block` was handed out by this allocator, so by `System`,
        // with `layout`.
        unsafe { System.dealloc(block, layout) };
    }

    unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        requested(new_size);
        // SAFETY: as in `dealloc`, and the caller meets `realloc`'s contract
        // for `new_size`.
        unsafe { System.realloc(block, layout, new_size) }
    }
}

#[global_allocator]
static RECORDING: Recording = Recording;

/// The largest single allocation `read` requests on this thread.
fn largest_allocation(read: impl FnOnce()) -> usize {
    LARGEST.with(|largest| largest.set(0));
    MEASURING.with(|measuring| measuring.set(true));
    read();
    MEASURING.with(|measuring| measuring.set(false));
    LARGEST.with(Cell::get)
}

/// A framed, required log record of `kind` holding `payload`.
fn record(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut bytes = vec![kind, 1];
    bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

/// A bincode length or count of 64 MiB: a u32 marker and 4 bytes.
const CLAIM: [u8; 5] = [0xFC, 0, 0, 0, 4];

/// The most a damaged record may allocate at once: its own few bytes, the
/// buffer it is read into and an error message.
const SMALL: usize = 4096;

#[test]
fn a_damaged_length_or_count_allocates_nothing_from_its_claim() {
    let concat = |parts: &[&[u8]]| parts.concat();
    let cases: [(&str, Vec<u8>); 10] = [
        // CreateNode (16): graph Some, its name claiming 64 MiB.
        (
            "a graph name",
            record(16, &concat(&[&[1], &CLAIM, b"Paris"])),
        ),
        // CreateNode: default graph, id 3, a label list of 64 Mi labels.
        (
            "a label count",
            record(16, &concat(&[&[0, 3], &CLAIM, &[6], b"Person"])),
        ),
        // CreateNode: one label claiming 64 MiB.
        (
            "a label",
            record(16, &concat(&[&[0, 3, 1], &CLAIM, b"Person"])),
        ),
        // CreateNode: no labels, a property list of 64 Mi properties.
        (
            "a property count",
            record(16, &concat(&[&[0, 3, 0], &CLAIM, &[1, b'x']])),
        ),
        // CreateNode: one property "x" whose value claims 64 MiB.
        (
            "a value length",
            record(16, &concat(&[&[0, 3, 0, 1, 1, b'x'], &CLAIM, &[2, 3, 0]])),
        ),
        // CreateEdge (18): default graph, id 88 from 3 to 19, a type claiming 64 MiB.
        (
            "an edge type",
            record(18, &concat(&[&[0, 88, 3, 19], &CLAIM, b"KNOWS"])),
        ),
        // SetNodeProperty (20): a key claiming 64 MiB.
        (
            "a property key",
            record(20, &concat(&[&[0, 3], &CLAIM, b"city"])),
        ),
        // InsertTriple (64): default graph, an IRI subject claiming 64 MiB.
        (
            "an RDF term",
            record(64, &concat(&[&[0, 0], &CLAIM, b"ex:Mia"])),
        ),
        // DropRdfGraph (67): a named graph whose name claims 64 MiB.
        (
            "an RDF graph name",
            record(67, &concat(&[&[1], &CLAIM, b"ex:g"])),
        ),
        // DropCatalog (41): a node type (catalog kind 2) whose name claims 64 MiB.
        (
            "a catalog key",
            record(41, &concat(&[&[2], &CLAIM, b"City"])),
        ),
    ];
    for (what, bytes) in cases {
        let mut result = None;
        let largest = largest_allocation(|| {
            result = Some(read_log_records(&bytes, &mut |_| Ok(())));
        });
        let error = match result {
            Some(Err(error)) => error.to_string(),
            other => panic!("{what}: a damaged record was read: {other:?}"),
        };
        assert!(error.contains("log record 0 at byte 0"), "{what}: {error}");
        assert!(
            largest <= SMALL,
            "{what}: the largest allocation took {largest} bytes, more than {SMALL}; the \
             claim of 64 MiB was allocated before its bytes were found"
        );
    }
}

/// The catalog record inside a `PutCatalog` is decoded as the catalog
/// section decodes it: a damaged string length there may claim up to the
/// catalog's decode limit of 16 MiB, and no more.
#[test]
fn a_damaged_catalog_record_allocates_at_most_the_catalogs_decode_limit() {
    const CATALOG_DECODE_LIMIT: usize = 1 << 24;
    // PutCatalog (40): a schema (catalog kind 1) whose name claims 64 MiB,
    // then one whose name claims 8 MiB.
    for claim in [CLAIM, [0xFC, 0, 0, 0x80, 0]] {
        let bytes = record(40, &[&[1u8][..], &claim, b"travel"].concat());
        let mut result = None;
        let largest = largest_allocation(|| {
            result = Some(read_log_records(&bytes, &mut |_| Ok(())));
        });
        assert!(matches!(result, Some(Err(_))), "{result:?}");
        assert!(
            largest <= CATALOG_DECODE_LIMIT,
            "the largest allocation took {largest} bytes"
        );
    }
}

/// A record whose values would take it past the maximum is refused from
/// their sizes: no value is encoded, so the refusal costs no more than the
/// record's own few bytes.
#[test]
#[cfg_attr(
    miri,
    ignore = "a list of 2^20 values takes minutes under Miri; the other tests here run the allocator under Miri"
)]
fn a_record_over_the_maximum_is_refused_before_its_values_are_encoded() {
    // About 1.01 GiB of encoded strings, held as 2^20 references to one
    // string of 1 KiB.
    let line = Value::from("Amsterdam ".repeat(103).as_str());
    let lines = Value::List(Arc::from(vec![line; 1 << 20]));
    assert!(encoded_len(&lines) > MAX_RECORD_BYTES);
    let record = LogRecord::Data {
        graph: None,
        op: DataOp::SetNodeProperty {
            id: NodeId::new(3),
            key: "notes".into(),
            value: lines,
        },
    };
    let mut out = Vec::new();
    let mut result = None;
    let largest = largest_allocation(|| result = Some(record.encode_framed(&mut out)));
    assert!(matches!(result, Some(Err(_))), "{result:?}");
    assert!(
        largest <= SMALL,
        "the largest allocation took {largest} bytes, more than {SMALL}: a value was encoded \
         before the record was refused"
    );
}
