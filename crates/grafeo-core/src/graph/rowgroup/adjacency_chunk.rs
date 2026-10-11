//! The bytes of an `Adjacency` chunk (`LPG_STORE` version 4, FD6): the
//! adjacency lists of a range of a node row group's rows in one direction,
//! each sorted by (edge type, other node, edge).
//!
//! Layout, little-endian: `flags` (`u8`, zero), three reserved bytes,
//! `row_count` (`u32`), `first_entry` (`u32`: for a piece of one node's
//! list, the index of the first entry it holds), `entry_count` (`u32`, from 1
//! to the row cap), then five bodies: the entries per row (`BitPacked`, one
//! per row), the edge type ids (`BitPacked`, one per entry), each row's first
//! other node (a base `u64`, then `BitPacked` differences to it, one per row
//! with entries), the later entries' other nodes (`BitPacked`, each the
//! zigzag difference to the entry before it in its row), and the edge ids (a
//! base `u64`, then `BitPacked` differences to it, one per entry).
//!
//! A reader refuses every chunk that is not what a writer writes, with
//! `Error::Corruption`: what a chunk decodes into is bounded by the row cap.

use bytes::Bytes;
use grafeo_common::utils::error::{Error, Result};

use super::adjacency::Adjacent;
use crate::codec::BitPackedInts;
use crate::codec::column_chunk::{read_bitpacked_body, write_bitpacked_body};

/// The bytes before the bodies.
const HEADER_LEN: usize = 16;

/// The lists of an adjacency chunk, decoded.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct AdjacencyChunk {
    /// For a piece of one node's list, the index in the list of its first
    /// entry; 0 otherwise.
    pub first_entry: u32,
    /// One list per row of the chunk's range, each sorted.
    pub lists: Vec<Vec<Adjacent>>,
}

fn zigzag(difference: i64) -> u64 {
    (difference.cast_unsigned() << 1) ^ (difference >> 63).cast_unsigned()
}

fn unzigzag(value: u64) -> i64 {
    (value >> 1).cast_signed() ^ -((value & 1).cast_signed())
}

/// Encodes the lists of the rows of a chunk's range (`lists[r]` for row
/// `r`), each sorted by (edge type, other node, edge).
///
/// # Errors
///
/// Returns [`Error::Internal`] when the chunk holds no entry, more entries
/// or rows than `max_entries`, or a list that is not sorted: a writer cuts
/// its chunks so none does.
pub(super) fn encode(lists: &[&[Adjacent]], first_entry: u32, max_entries: u32) -> Result<Vec<u8>> {
    let entry_count: usize = lists.iter().map(|list| list.len()).sum();
    let refuse = |what: String| Error::Internal(format!("cannot write an adjacency chunk: {what}"));
    if entry_count == 0 || entry_count > max_entries as usize || lists.len() > max_entries as usize
    {
        return Err(refuse(format!(
            "{} rows and {entry_count} entries, where a chunk holds 1 to {max_entries} of each",
            lists.len()
        )));
    }
    if let Some(row) = lists
        .iter()
        .position(|list| !list.windows(2).all(|pair| pair[0] < pair[1]))
    {
        return Err(refuse(format!("the list of row {row} is not sorted")));
    }
    let mut degrees = Vec::with_capacity(lists.len());
    let mut types = Vec::with_capacity(entry_count);
    let mut firsts = Vec::new();
    let mut steps = Vec::with_capacity(entry_count);
    let mut edges = Vec::with_capacity(entry_count);
    for list in lists {
        degrees.push(list.len() as u64);
        for (index, adjacent) in list.iter().enumerate() {
            types.push(u64::from(adjacent.edge_type));
            edges.push(adjacent.edge);
            if index == 0 {
                firsts.push(adjacent.other);
            } else {
                steps.push(zigzag(
                    adjacent
                        .other
                        .wrapping_sub(list[index - 1].other)
                        .cast_signed(),
                ));
            }
        }
    }
    let first_base = firsts.iter().copied().min().unwrap_or(0);
    let edge_base = edges.iter().copied().min().unwrap_or(0);
    for first in &mut firsts {
        *first -= first_base;
    }
    for edge in &mut edges {
        *edge -= edge_base;
    }

    let mut out = Vec::with_capacity(HEADER_LEN + entry_count * 4);
    out.extend_from_slice(&[0; 4]);
    let count =
        |value: usize| u32::try_from(value).map_err(|_| refuse(format!("{value} is past a u32")));
    out.extend_from_slice(&count(lists.len())?.to_le_bytes());
    out.extend_from_slice(&first_entry.to_le_bytes());
    out.extend_from_slice(&count(entry_count)?.to_le_bytes());
    write_bitpacked_body(&BitPackedInts::pack(&degrees), &mut out)?;
    write_bitpacked_body(&BitPackedInts::pack(&types), &mut out)?;
    out.extend_from_slice(&first_base.to_le_bytes());
    write_bitpacked_body(&BitPackedInts::pack(&firsts), &mut out)?;
    write_bitpacked_body(&BitPackedInts::pack(&steps), &mut out)?;
    out.extend_from_slice(&edge_base.to_le_bytes());
    write_bitpacked_body(&BitPackedInts::pack(&edges), &mut out)?;
    Ok(out)
}

/// A reader over an adjacency chunk's bytes.
struct Reader<'b> {
    data: &'b Bytes,
    pos: usize,
}

impl Reader<'_> {
    fn refuse(&self, what: impl std::fmt::Display) -> Error {
        Error::corruption(format!("adjacency chunk, byte {}: {what}", self.pos))
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        let bytes = self
            .data
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.refuse(format!("the chunk ends inside {what}")))?;
        self.pos += 4;
        Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        let bytes = self
            .data
            .get(self.pos..self.pos + 8)
            .ok_or_else(|| self.refuse(format!("the chunk ends inside {what}")))?;
        self.pos += 8;
        Ok(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
    }

    /// A `BitPacked` body of exactly `count` values, unpacked.
    fn body(&mut self, count: usize, what: &str) -> Result<Vec<u64>> {
        let start = self.pos;
        let packed = read_bitpacked_body(self.data, &mut self.pos).map_err(|error| {
            self.pos = start;
            self.refuse(format!("{what}: {error}"))
        })?;
        // A writer packs each value inside one word, at the width of the
        // largest, and never at width 0: an empty body alone has no words.
        let bits = usize::from(packed.bits_per_value());
        let words = match (count, bits) {
            (0, 0) => Some(0),
            (_, 1..=64) if count > 0 => Some(count.div_ceil(64 / bits)),
            _ => None,
        };
        if packed.len() != count || words != Some(packed.word_count()) {
            self.pos = start;
            return Err(self.refuse(format!(
                "{what}: {} values of {bits} bits in {} words, where the chunk has {count} values",
                packed.len(),
                packed.word_count()
            )));
        }
        Ok(packed.unpack())
    }
}

/// Decodes an adjacency chunk whose directory entry says it covers
/// `row_count` rows; `max_entries` is the section's row cap.
///
/// # Errors
///
/// Returns [`Error::Corruption`] for anything a writer does not write.
pub(super) fn decode(data: &Bytes, row_count: u32, max_entries: u32) -> Result<AdjacencyChunk> {
    let mut reader = Reader { data, pos: 0 };
    let flags = reader.u32("the flags")?;
    if flags & 0xFF != 0 {
        reader.pos = 0;
        return Err(reader.refuse(format!(
            "flags {:#04x}, where this version writes 0",
            flags & 0xFF
        )));
    }
    let rows = reader.u32("the row count")?;
    if rows != row_count || rows == 0 || rows > max_entries {
        return Err(reader.refuse(format!(
            "{rows} rows, where its directory entry says {row_count} and the cap is {max_entries}"
        )));
    }
    let first_entry = reader.u32("the first entry")?;
    let entry_count = reader.u32("the entry count")?;
    if entry_count == 0 || entry_count > max_entries {
        return Err(reader.refuse(format!(
            "{entry_count} entries, where a chunk holds 1 to {max_entries}"
        )));
    }
    if first_entry != 0 && rows != 1 {
        return Err(reader.refuse(format!(
            "a piece from entry {first_entry} over {rows} rows, where a piece has one"
        )));
    }
    let entries = entry_count as usize;
    let degrees = reader.body(rows as usize, "the entries per row")?;
    if degrees
        .iter()
        .try_fold(0_u64, |sum, degree| sum.checked_add(*degree))
        != Some(u64::from(entry_count))
    {
        return Err(reader.refuse("the entries per row do not add up to the entry count"));
    }
    let with_entries = degrees.iter().filter(|degree| **degree > 0).count();
    let types = reader.body(entries, "the edge types")?;
    let first_base = reader.u64("the base of the first other nodes")?;
    let firsts = reader.body(with_entries, "the first other nodes")?;
    let steps = reader.body(entries - with_entries, "the later other nodes")?;
    let edge_base = reader.u64("the base of the edge ids")?;
    let edges = reader.body(entries, "the edge ids")?;
    if reader.pos != data.len() {
        return Err(reader.refuse(format!(
            "{} bytes after the last body",
            data.len() - reader.pos
        )));
    }

    let mut lists = Vec::with_capacity(rows as usize);
    let (mut entry, mut first, mut step) = (0, 0, 0);
    for (row, degree) in degrees.iter().enumerate() {
        let mut list: Vec<Adjacent> = Vec::with_capacity(usize::try_from(*degree).unwrap_or(0));
        for index in 0..*degree {
            let other = if index == 0 {
                first += 1;
                first_base.checked_add(firsts[first - 1])
            } else {
                step += 1;
                Some(
                    list[list.len() - 1]
                        .other
                        .wrapping_add_signed(unzigzag(steps[step - 1])),
                )
            };
            let adjacent = match (
                other,
                u32::try_from(types[entry]),
                edge_base.checked_add(edges[entry]),
            ) {
                (Some(other), Ok(edge_type), Some(edge)) => Adjacent {
                    edge_type,
                    other,
                    edge,
                },
                _ => {
                    return Err(reader.refuse(format!(
                        "entry {entry} of row {row} holds an id past its range"
                    )));
                }
            };
            if list.last().is_some_and(|last| *last >= adjacent) {
                return Err(reader.refuse(format!(
                    "the list of row {row} is not sorted at entry {index}"
                )));
            }
            list.push(adjacent);
            entry += 1;
        }
        lists.push(list);
    }
    Ok(AdjacencyChunk { first_entry, lists })
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::{AdjacencyChunk, decode, encode, unzigzag, zigzag};
    use crate::graph::rowgroup::adjacency::Adjacent;

    fn edge(edge_type: u32, other: u64, edge: u64) -> Adjacent {
        Adjacent {
            edge_type,
            other,
            edge,
        }
    }

    fn round_trip(lists: &[Vec<Adjacent>], first_entry: u32) -> Vec<u8> {
        let borrowed: Vec<&[Adjacent]> = lists.iter().map(Vec::as_slice).collect();
        let bytes = encode(&borrowed, first_entry, 65_536).unwrap();
        let rows = u32::try_from(lists.len()).unwrap();
        assert_eq!(
            decode(&Bytes::from(bytes.clone()), rows, 65_536).unwrap(),
            AdjacencyChunk {
                first_entry,
                lists: lists.to_vec()
            }
        );
        bytes
    }

    #[test]
    fn zigzag_round_trips_every_sign() {
        for difference in [0, 1, -1, 19, -88, i64::MAX, i64::MIN] {
            assert_eq!(unzigzag(zigzag(difference)), difference);
        }
        assert_eq!((zigzag(0), zigzag(-1), zigzag(1), zigzag(-2)), (0, 1, 2, 3));
    }

    #[test]
    fn lists_round_trip_with_empty_rows_type_changes_and_large_ids() {
        round_trip(
            &[
                vec![edge(0, 7, 100), edge(0, 19, 3), edge(2, 1, 88)],
                vec![],
                vec![edge(1, u64::MAX - 1, 1 << 40)],
                vec![edge(3, 5, 9), edge(3, 5, 10), edge(4, 0, 2)],
            ],
            0,
        );
        // One node's piece, from its 65,536th entry on.
        round_trip(&[vec![edge(0, 1, 1), edge(0, 2, 2)]], 65_536);
    }

    /// The bytes of a small chunk, pinned: two rows, a type change in the
    /// first (the other node goes down, a negative step), bases taken off.
    #[test]
    fn a_small_chunk_has_its_documented_bytes() {
        let bytes = round_trip(
            &[vec![edge(1, 9, 20), edge(2, 3, 21)], vec![edge(1, 5, 23)]],
            0,
        );
        let expected: Vec<u8> = [
            &[0_u8, 0, 0, 0][..],         // flags, reserved
            &2_u32.to_le_bytes(),         // rows
            &0_u32.to_le_bytes(),         // first entry
            &3_u32.to_le_bytes(),         // entries
            &[2, 2, 0, 0, 0, 1, 0, 0, 0], // degrees 2, 1: 2 bits each, 2 values, 1 word
            &0b01_10_u64.to_le_bytes(),
            &[2, 3, 0, 0, 0, 1, 0, 0, 0], // types 1, 2, 1: 2 bits each
            &0b01_10_01_u64.to_le_bytes(),
            &5_u64.to_le_bytes(),         // base of the firsts: 9 and 5
            &[3, 2, 0, 0, 0, 1, 0, 0, 0], // 4, 0: 3 bits each
            &0b000_100_u64.to_le_bytes(),
            &[4, 1, 0, 0, 0, 1, 0, 0, 0], // one step, 3 - 9 = -6: zigzag 11
            &11_u64.to_le_bytes(),
            &20_u64.to_le_bytes(),        // base of the edges
            &[2, 3, 0, 0, 0, 1, 0, 0, 0], // 0, 1, 3: 2 bits each
            &0b11_01_00_u64.to_le_bytes(),
        ]
        .concat();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn a_writer_refuses_what_a_reader_would() {
        let sorted = [edge(0, 1, 1), edge(0, 2, 2)];
        let unsorted = [edge(1, 1, 1), edge(0, 2, 2)];
        assert!(
            encode(&[&[]], 0, 65_536)
                .unwrap_err()
                .to_string()
                .contains("0 entries")
        );
        assert!(
            encode(&[&sorted], 0, 1)
                .unwrap_err()
                .to_string()
                .contains("2 entries")
        );
        assert!(
            encode(&[&unsorted], 0, 65_536)
                .unwrap_err()
                .to_string()
                .contains("not sorted")
        );
        let twice = [edge(0, 1, 1), edge(0, 1, 1)];
        assert!(
            encode(&[&twice], 0, 65_536)
                .unwrap_err()
                .to_string()
                .contains("not sorted")
        );
    }

    #[test]
    fn a_reader_refuses_damaged_chunks_as_corruption() {
        let lists = [vec![edge(1, 9, 20), edge(2, 3, 21)], vec![edge(1, 5, 23)]];
        let good = round_trip(&lists, 0);
        let refused = |bytes: Vec<u8>, rows: u32, cap: u32, what: &str| {
            let error = decode(&Bytes::from(bytes), rows, cap).unwrap_err();
            assert!(
                matches!(error, grafeo_common::utils::error::Error::Corruption(_))
                    && error.to_string().contains(what),
                "{what}: {error}"
            );
        };
        let with = |at: usize, byte: u8| {
            let mut bytes = good.clone();
            bytes[at] = byte;
            bytes
        };
        refused(with(0, 1), 2, 65_536, "flags");
        refused(good.clone(), 3, 65_536, "directory entry says 3");
        refused(good.clone(), 2, 1, "cap is 1");
        refused(with(12, 0), 2, 65_536, "0 entries");
        // A body of width 0 that claims values: the route to a decode bomb.
        refused(with(16, 0), 2, 65_536, "of 0 bits");
        refused(with(12, 4), 2, 65_536, "do not add up");
        refused(with(8, 1), 2, 65_536, "a piece from entry 1 over 2 rows");
        refused(good[..good.len() - 1].to_vec(), 2, 65_536, "edge ids");
        refused([good.clone(), vec![0]].concat(), 2, 65_536, "1 bytes after");
        // A claimed entry count far past the cap, in a chunk of a few bytes.
        let mut bomb = good.clone();
        bomb[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        refused(bomb, 2, 65_536, "entries, where a chunk holds 1 to 65536");
        // The types 1, 2, 1 made 2, 1, 1: the first row's list goes down.
        refused(with(42, 0b01_01_10), 2, 65_536, "not sorted at entry 1");
    }
}
