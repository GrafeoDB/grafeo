//! How much a log record decodes into per byte it holds.
//!
//! A label decodes into a name of 8 bytes in its list and, unless it is
//! empty, a header of 16 bytes before its bytes; a property into a pair of
//! 48 and its key's header. That is up to 24 and 64 bytes from as little as
//! one and three bytes: unchecked, a record of empty labels held 13 times
//! its size while its list grew, and one of properties of an empty key and
//! a null 25 times. The list rule keeps a list of labels or properties at 8
//! bytes of slots (a name's header counted in its slot) per byte of its
//! elements plus 1 MiB, twice that while it grows. These tests measure the
//! peak with a counting allocator for the densest lists, one the rule
//! refuses and ones at the rule, and check it against that bound.
//!
//! The global allocator counts, per thread, the bytes held while the
//! thread measures, including the copy a growing list makes (the default
//! `realloc` copies), and forwards every call to the system allocator.

#![cfg(not(miri))]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this only forwards to System"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use grafeo_common::change::{DataOp, Labels};
use grafeo_common::storage::{LogRecord, read_log_records};
use grafeo_common::types::{ArcStr, NodeId, PropertyKey, Value};

/// The header the rule counts in a name's slot: the length and the count a
/// shared string keeps before its bytes.
const NAME_HEADER: usize = 2 * size_of::<usize>();

thread_local! {
    /// Whether this thread is measuring.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    /// The bytes this thread holds from what it allocated while measuring.
    static CURRENT: Cell<usize> = const { Cell::new(0) };
    /// The most bytes this thread held while measuring.
    static PEAK: Cell<usize> = const { Cell::new(0) };
    /// The largest single allocation of this thread while measuring.
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

/// Counts an allocation of `size` bytes, when this thread measures.
fn allocated(size: usize) {
    // `try_with`: a thread that is shutting down has no locals left.
    let measuring = MEASURING.try_with(Cell::get).unwrap_or(false);
    if measuring {
        let now = CURRENT.with(|current| {
            current.set(current.get() + size);
            current.get()
        });
        PEAK.with(|peak| peak.set(peak.get().max(now)));
        LARGEST.with(|largest| largest.set(largest.get().max(size)));
    }
}

/// Counts a deallocation of `size` bytes, when this thread measures.
fn released(size: usize) {
    let measuring = MEASURING.try_with(Cell::get).unwrap_or(false);
    if measuring {
        CURRENT.with(|current| current.set(current.get().saturating_sub(size)));
    }
}

struct Counting;

// SAFETY: every method forwards its arguments unchanged to `System` and
// returns what it returned; the counting touches const-initialized thread
// locals only and allocates nothing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller meets `alloc`'s contract, which is `System`'s.
        let block = unsafe { System.alloc(layout) };
        if !block.is_null() {
            allocated(layout.size());
        }
        block
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        // SAFETY: `block` came from `System` with `layout`.
        unsafe { System.dealloc(block, layout) };
        released(layout.size());
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

const MIB: usize = 1 << 20;

/// The most bytes a list may take in slots once decoded, per byte of its
/// elements, and beyond that.
const PER_BYTE: usize = 8;
const ALLOWANCE: usize = MIB;

/// The slot the rule counts for a property: its place in the list and its
/// key's header.
const PROPERTY_SLOT: usize = 64;

/// A framed, required CreateNode (16) of the default graph, id 3: `labels`
/// labels of `label` and `properties` properties of `key` and a null.
fn create_node(labels: usize, label: &str, properties: usize, key: &str) -> Vec<u8> {
    let mut payload = vec![0u8, 3];
    let count = |count: usize, out: &mut Vec<u8>| {
        out.push(0xFC);
        out.extend_from_slice(&u32::try_from(count).unwrap().to_le_bytes());
    };
    count(labels, &mut payload);
    for _ in 0..labels {
        payload.push(u8::try_from(label.len()).unwrap());
        payload.extend_from_slice(label.as_bytes());
    }
    count(properties, &mut payload);
    for _ in 0..properties {
        payload.push(u8::try_from(key.len()).unwrap());
        payload.extend_from_slice(key.as_bytes());
        payload.extend_from_slice(&[1, 0]); // a value of one byte: a null
    }
    let mut bytes = vec![16u8, 1];
    bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

/// Reads `bytes` while measuring: the outcome and the most bytes held.
fn peak_while_reading(bytes: &[u8]) -> (Result<usize, String>, usize) {
    let (outcome, peak, _) = measure_reading(bytes);
    (outcome, peak)
}

/// Reads `bytes` while measuring: the outcome, the most bytes held and the
/// largest single allocation.
fn measure_reading(bytes: &[u8]) -> (Result<usize, String>, usize, usize) {
    CURRENT.with(|current| current.set(0));
    PEAK.with(|peak| peak.set(0));
    LARGEST.with(|largest| largest.set(0));
    MEASURING.with(|measuring| measuring.set(true));
    let mut count = 0usize;
    let outcome = read_log_records(bytes, &mut |_| {
        count += 1;
        Ok(())
    });
    MEASURING.with(|measuring| measuring.set(false));
    (
        outcome.map(|()| count).map_err(|error| error.to_string()),
        PEAK.with(Cell::get),
        LARGEST.with(Cell::get),
    )
}

/// The bound for a record of `bytes` whose list takes `elements` bytes in
/// it and whose strings take `strings` bytes: the list's slots and, while
/// it grows, their copy; the strings; and the record's payload, read into a
/// buffer that grows with the bytes present (its old and new buffer
/// together at most three times the payload while it grows).
fn bound(bytes: usize, elements: usize, strings: usize) -> usize {
    2 * (PER_BYTE * elements + ALLOWANCE) + strings + 3 * bytes
}

/// A record of 2^20 empty labels in 1 MiB is past the rule: the reader
/// refuses it at the label that breaks it, having held a few times its size,
/// not the 24 MiB its labels would take.
#[test]
fn a_list_of_empty_labels_past_the_rule_is_refused_holding_a_few_times_its_size() {
    let bytes = create_node(1 << 20, "", 0, "");
    let (outcome, peak) = peak_while_reading(&bytes);
    let error = outcome.unwrap_err();
    assert!(error.contains("a list of 65537 labels"), "{error}");
    let factor = peak / bytes.len();
    assert!(
        factor < 8,
        "refusing {} bytes held {peak} bytes: {factor} times its size",
        bytes.len()
    );
}

/// A list grows with its elements, never past what the rule allows for the
/// elements read so far, whatever count the payload claims: 26,215
/// properties of an empty key and a null (three bytes each, the rule breaks
/// at the last) under a claim of 2^20 never take a buffer with room for
/// more properties than the rule's 8 bytes per byte plus 1 MiB holds at 64
/// bytes of slot each (26,214), where doubling the list would make room for
/// 32,768.
#[test]
fn a_list_never_grows_past_what_the_rule_allows_for_its_elements() {
    let mut bytes = create_node(0, "", 26_215, "");
    // Claim 2^20 properties: the count after the default graph, the id and
    // the empty label list.
    bytes[6 + 7..6 + 12].copy_from_slice(&[0xFC, 0, 0, 0x10, 0]);
    let (outcome, _, largest) = measure_reading(&bytes);
    let error = outcome.unwrap_err();
    assert!(error.contains("a list of 26215 properties"), "{error}");
    let allowed = (PER_BYTE * 3 * 26_215 + ALLOWANCE) / PROPERTY_SLOT;
    assert_eq!(allowed, 26_214);
    let rule = allowed * size_of::<(PropertyKey, Value)>();
    assert!(
        largest <= rule,
        "the largest allocation took {largest} bytes, past room for the rule's {allowed} \
         properties ({rule} bytes)"
    );
}

/// The densest lists the rule takes, and some more: labels of two bytes
/// (24 bytes of slot for 3 bytes, the rule's own ratio), labels of one byte
/// at the rule (each a name with a header), and 40,000 properties of a
/// two-byte key and a null (64 for 5). Each decodes within the bound.
#[test]
fn the_densest_lists_the_rule_takes_decode_within_the_bound() {
    for (what, bytes, elements, strings) in [
        (
            "2^19 labels of two bytes",
            create_node(1 << 19, "Ab", 0, ""),
            3 << 19,
            2 << 19,
        ),
        (
            "2^17 labels of one byte, at the rule",
            create_node(1 << 17, "A", 0, ""),
            2 << 17,
            1 << 17,
        ),
        (
            "2^16 empty labels, the allowance",
            create_node(1 << 16, "", 0, ""),
            1 << 16,
            0,
        ),
        (
            "40,000 properties of a two-byte key and a null",
            create_node(0, "", 40_000, "Ab"),
            5 * 40_000,
            2 * 40_000,
        ),
    ] {
        let (outcome, peak) = peak_while_reading(&bytes);
        assert_eq!(outcome, Ok(1), "{what}");
        let limit = bound(bytes.len(), elements, strings);
        assert!(
            peak <= limit,
            "{what}: {} bytes held {peak} bytes, over the bound of {limit}",
            bytes.len()
        );
    }
}

/// The writer refuses what the reader refuses, before it writes a byte:
/// every record written decodes.
#[test]
fn the_writer_refuses_the_list_the_reader_refuses() {
    let record = LogRecord::Data {
        graph: None,
        op: DataOp::CreateNode {
            id: NodeId::new(3),
            labels: Labels::from_elem(ArcStr::new(), 1 << 20),
            properties: vec![(PropertyKey::new(""), Value::Null); 3],
        },
    };
    let mut out = Vec::new();
    let error = record.encode_framed(&mut out).unwrap_err().to_string();
    assert!(error.contains("a list of 65537 labels"), "{error}");
    assert_eq!(out, Vec::<u8>::new(), "a refused record appends nothing");
}

/// The rule counts a label's slot as its place in the list and its name's
/// header: a name decoded is a shared string that takes that header and its
/// bytes, no more, and an empty name takes nothing (it is static).
#[test]
fn a_name_takes_its_header_and_its_bytes_as_the_rule_counts() {
    let long = "Prague".repeat(88);
    for (name, expected) in [
        ("", 0),
        ("A", NAME_HEADER + 1),
        ("Barcelona", NAME_HEADER + 9),
        (long.as_str(), NAME_HEADER + long.len()),
    ] {
        CURRENT.with(|current| current.set(0));
        MEASURING.with(|measuring| measuring.set(true));
        let shared = ArcStr::from(name);
        MEASURING.with(|measuring| measuring.set(false));
        let held = CURRENT.with(Cell::get);
        assert!(
            held <= expected,
            "a name of {} bytes holds {held} bytes, more than {expected}",
            name.len()
        );
        drop(shared);
    }
}
