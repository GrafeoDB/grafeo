//! The column chunk codec of the `.grafeo` format.
//!
//! A column chunk holds the values of one column for a range of rows of a
//! table: only the rows that have a value, in row order, behind a presence
//! bitmap, so a row without a value costs one bit. The chunk's directory
//! entry names its codec and row count; the chunk repeats both, and
//! [`decode_column_chunk`] refuses a chunk that disagrees with its entry.
//!
//! Layout, little-endian:
//!
//! | Field | Size | Meaning |
//! | --- | --- | --- |
//! | codec | u8 | the [`ChunkCodec`], equal to the directory entry's |
//! | flags | u8 | bit 0: presence bitmap, bit 1: zone map, bit 2: epochs; other bits are refused |
//! | reserved | u16 | written 0, ignored |
//! | row count | u32 | equal to the directory entry's, at most 65,536 (the format's row cap) |
//! | value count | u32 | at least 1, at most the row count |
//! | presence bitmap | `ceil(rows / 64)` u64 words | only when the value count is below the row count: bit `r` is set when row `r` has a value, bits past the rows are 0 |
//! | zone map | see below | bounds of the values, for the kinds that have them |
//! | epochs | a bit-packed body | one u64 per value, only when one of them is not 0 |
//! | body | | the values, in the codec's body |
//!
//! The bodies:
//!
//! | Codec | Values | Body |
//! | --- | --- | --- |
//! | 1 `BitPacked` | non-negative `Int64` | bits u8, count u32, word count u32, u64 words |
//! | 2 `Dict` | `String` | entry count u32, (length u32, UTF-8) per entry, code count u32, u32 codes |
//! | 3 `Bitmap` | `Bool` | bit count u32, word count u32, u64 words |
//! | 4 `Float64` | `Float64` | count u32, f64 bits |
//! | 5 `RawI64` | `Int64` | count u32, i64 |
//! | 6 `Float32Vector` | `Vector`s of one dimension count | dimensions u16, component count u32, f32 bits |
//! | 7 `Values` | any | count u32, values in the value codec of `grafeo_common::storage::value_codec` |
//!
//! [`choose_codec`] picks a typed codec only when every value has its kind,
//! so every value round trips exactly: an `Int64` next to a `Float64` stays
//! an `Int64`. The bodies of codecs 1 to 6 are those the compact store's
//! `ColumnCodec` writes after its discriminant, through the same functions.
//!
//! The zone map of an `Int64`, `Float64` (NaN left out, and no zone map when
//! every value is NaN) or `Bool` chunk is its minimum and maximum, two values
//! in the value codec. Every `String` chunk has a [`StringZoneMap`], by the
//! strings' UTF-8 bytes (the order of `str`):
//!
//! | Field | Size | Meaning |
//! | --- | --- | --- |
//! | minimum | length u8, bytes | the smallest string, cut to at most [`STRING_ZONE_PREFIX`] bytes |
//! | maximum | length u8, bytes | the largest string; when longer than [`STRING_ZONE_PREFIX`] bytes, its first ones with the last raised by one |
//! | flags | u8 | bit 0: the minimum was not cut, bit 1: the maximum was not cut; other bits are refused |
//! | shortest, longest | u32 each | the lengths in bytes of the shortest and the longest string |
//!
//! A cut maximum is raised so it stays an upper bound: every string that
//! starts with the cut bytes is below it (UTF-8 holds no `0xFF` byte, so the
//! last byte can always be raised). Other chunks have no zone map. The
//! decoder refuses a zone map other than the one the values give, so a
//! reader can trust it.

use std::cmp::Ordering;
use std::fmt::Display;
use std::sync::Arc;

use arcstr::ArcStr;
use bytes::Bytes;
use grafeo_common::storage::ChunkCaps;
use grafeo_common::storage::value_codec::{decode_value, encode_value, encoded_len};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};

use super::limits::{checked_u16, checked_u32};
use super::{BitPackedInts, BitVector, DictionaryBuilder, DictionaryEncoding};

/// The most bytes of each bound of a [`StringZoneMap`]: longer strings are
/// cut to this many.
pub const STRING_ZONE_PREFIX: usize = 16;

/// The most bytes a zone map takes: a string zone map's two bounds with
/// their length bytes, its flags and its two lengths (the two values of the
/// other kinds take fewer).
const ZONE_MAP_MAX_LEN: usize = 2 * (1 + STRING_ZONE_PREFIX) + 1 + 4 + 4;
/// String zone map flag bit 0: the minimum is the smallest string.
const MIN_EXACT: u8 = 1;
/// String zone map flag bit 1: the maximum is the largest string.
const MAX_EXACT: u8 = 2;

/// The most rows a chunk holds: the format's row cap, which no writer
/// passes ([`ChunkCaps::validate`]). A chunk decodes into one value per row,
/// so this bounds what any chunk decodes into.
const MAX_ROWS: u32 = ChunkCaps::DEFAULT.max_rows;

/// Bytes of the chunk header: codec, flags, reserved, row count, value count.
const HEADER_LEN: usize = 12;
/// The most bytes a codec body takes besides its values' own bounds.
const BODY_COUNTS_LEN: usize = 16;
/// The most bytes epochs take besides 8 per value.
const EPOCHS_COUNTS_LEN: usize = 17;
const FLAG_PRESENCE: u8 = 1;
const FLAG_ZONE_MAP: u8 = 2;
const FLAG_EPOCHS: u8 = 4;
const KNOWN_FLAGS: u8 = FLAG_PRESENCE | FLAG_ZONE_MAP | FLAG_EPOCHS;

/// The codec of a column chunk's body: the byte its directory entry and its
/// header hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChunkCodec {
    /// Non-negative `Int64` values, bit-packed to the width of the largest.
    BitPacked = 1,
    /// `String` values, as a dictionary and one code per value.
    Dict = 2,
    /// `Bool` values, one bit each.
    Bitmap = 3,
    /// `Float64` values, by their bits.
    Float64 = 4,
    /// `Int64` values, 8 bytes each.
    RawI64 = 5,
    /// `Vector` values of one dimension count, by their components' bits.
    Float32Vector = 6,
    /// Values of any kind, each in the lossless value codec.
    Values = 7,
}

impl ChunkCodec {
    /// The byte that names this codec in a directory entry and a chunk header.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        self as u8
    }

    /// The codec `byte` names, or `None` when no codec has that byte.
    #[must_use]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::BitPacked),
            2 => Some(Self::Dict),
            3 => Some(Self::Bitmap),
            4 => Some(Self::Float64),
            5 => Some(Self::RawI64),
            6 => Some(Self::Float32Vector),
            7 => Some(Self::Values),
            _ => None,
        }
    }
}

/// A column chunk, decoded: the rows of `[0, row_count)` that have a value,
/// ascending, with the value's epoch when the chunk carries epochs.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnChunk {
    /// The rows the chunk covers, counted from its first row.
    pub row_count: u32,
    /// Each row that has a value (an offset from the chunk's first row) with
    /// its value, rows ascending.
    pub values: Vec<(u32, Value)>,
    /// One epoch per value, when the chunk carries epochs.
    pub epochs: Option<Vec<u64>>,
    /// The bounds of the values, for the kinds that have them.
    pub zone_map: Option<ZoneMap>,
}

/// The bounds of a chunk's values (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub enum ZoneMap {
    /// The minimum and maximum of `Int64`, `Float64` (NaN left out) or
    /// `Bool` values.
    Values(Value, Value),
    /// The bounds of `String` values.
    Strings(StringZoneMap),
}

/// The bounds of a chunk's strings, by their UTF-8 bytes (the order of
/// `str`): no string of the chunk is below `min` or above `max` (below it
/// when `max` was cut), and each is from `min_len` to `max_len` bytes long.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringZoneMap {
    /// The smallest string, cut to at most [`STRING_ZONE_PREFIX`] bytes.
    pub min: Vec<u8>,
    /// Whether `min` is the smallest string (it was not cut).
    pub min_exact: bool,
    /// The largest string when `max_exact`; else its first
    /// [`STRING_ZONE_PREFIX`] bytes with the last raised by one, which every
    /// string of the chunk is below (these bytes need not be UTF-8).
    pub max: Vec<u8>,
    /// Whether `max` is the largest string (it was not cut).
    pub max_exact: bool,
    /// The length in bytes of the shortest string.
    pub min_len: u32,
    /// The length in bytes of the longest string.
    pub max_len: u32,
}

impl StringZoneMap {
    /// The zone map of `strings`, `None` for no string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] for a string longer than `u32::MAX`
    /// bytes, which no chunk holds.
    fn of<'s>(strings: impl Iterator<Item = &'s str> + Clone) -> Result<Option<Self>> {
        let Some((min, max)) = min_max(strings.clone(), |a: &&str, b: &&str| a.cmp(b)) else {
            return Ok(None);
        };
        let Some((shortest, longest)) = min_max(strings.map(str::len), usize::cmp) else {
            return Ok(None);
        };
        let min_exact = min.len() <= STRING_ZONE_PREFIX;
        let max_exact = max.len() <= STRING_ZONE_PREFIX;
        let mut max = max.as_bytes()[..max.len().min(STRING_ZONE_PREFIX)].to_vec();
        if !max_exact && let Some(last) = max.last_mut() {
            // A UTF-8 byte is at most 0xF4, so it can be raised by one.
            *last += 1;
        }
        Ok(Some(Self {
            min: min.as_bytes()[..min.len().min(STRING_ZONE_PREFIX)].to_vec(),
            min_exact,
            max,
            max_exact,
            min_len: checked_u32(shortest, "string length")?,
            max_len: checked_u32(longest, "string length")?,
        }))
    }

    /// Whether `value` may be one of the chunk's strings: `false` only when
    /// the zone map shows it is not.
    #[must_use]
    pub fn may_hold(&self, value: &str) -> bool {
        let bytes = value.as_bytes();
        let length = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
        let below_max = if self.max_exact {
            bytes <= self.max.as_slice()
        } else {
            bytes < self.max.as_slice()
        };
        (self.min_len..=self.max_len).contains(&length) && bytes >= self.min.as_slice() && below_max
    }
}

/// The codec for these values: a typed codec only when every value has its
/// kind (non-negative `Int64`: `BitPacked`; `Int64`: `RawI64`; `Float64`:
/// `Float64`; `Bool`: `Bitmap`; `String`: `Dict`; vectors of one dimension
/// count from 1 to 65535: `Float32Vector`), else `Values`, which is also the
/// codec for no values.
#[must_use]
pub fn choose_codec<'v>(values: impl Iterator<Item = &'v Value> + Clone) -> ChunkCodec {
    let every = |test: &dyn Fn(&Value) -> bool| values.clone().all(test);
    match values.clone().next() {
        Some(Value::Int64(_)) if every(&|value| matches!(value, Value::Int64(_))) => {
            if every(&|value| matches!(value, Value::Int64(n) if *n >= 0)) {
                ChunkCodec::BitPacked
            } else {
                ChunkCodec::RawI64
            }
        }
        Some(Value::Float64(_)) if every(&|value| matches!(value, Value::Float64(_))) => {
            ChunkCodec::Float64
        }
        Some(Value::Bool(_)) if every(&|value| matches!(value, Value::Bool(_))) => {
            ChunkCodec::Bitmap
        }
        Some(Value::String(_)) if every(&|value| matches!(value, Value::String(_))) => {
            ChunkCodec::Dict
        }
        Some(Value::Vector(first))
            if u16::try_from(first.len()).is_ok_and(|dimensions| dimensions > 0)
                && every(
                    &|value| matches!(value, Value::Vector(other) if other.len() == first.len()),
                ) =>
        {
            ChunkCodec::Float32Vector
        }
        _ => ChunkCodec::Values,
    }
}

/// Encodes rows `values` (offsets below `row_count`, ascending, at least one)
/// with `epochs` (one per value, written only when one is not 0).
///
/// Returns the codec [`choose_codec`] picked, which the chunk's directory
/// entry names, and the chunk's bytes.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when `row_count` is above the format's
/// row cap (65,536), there is no value, a row is not below `row_count` or
/// not above the row before it, `epochs` does not hold one epoch per value,
/// or a value does not fit the format (a string, list or other length past
/// a u32, or nesting past the value codec's limit).
pub fn encode_column_chunk(
    row_count: u32,
    values: &[(u32, Value)],
    epochs: Option<&[u64]>,
) -> Result<(ChunkCodec, Vec<u8>)> {
    check_rows(row_count, values)?;
    if let Some(epochs) = epochs
        && epochs.len() != values.len()
    {
        return Err(Error::Serialization(format!(
            "cannot write a column chunk: {} epochs for {} values",
            epochs.len(),
            values.len()
        )));
    }
    let value_count = checked_u32(values.len(), "column chunk value count")?;
    let epochs = epochs.filter(|epochs| epochs.iter().any(|&epoch| epoch != 0));
    let codec = choose_codec(values.iter().map(|(_, value)| value));
    let zone_map = zone_map(codec, values.iter().map(|(_, value)| value))?;
    let sparse = value_count < row_count;
    let mut flags = 0;
    if sparse {
        flags |= FLAG_PRESENCE;
    }
    if zone_map.is_some() {
        flags |= FLAG_ZONE_MAP;
    }
    if epochs.is_some() {
        flags |= FLAG_EPOCHS;
    }
    let mut out = Vec::new();
    out.push(codec.to_byte());
    out.push(flags);
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&row_count.to_le_bytes());
    out.extend_from_slice(&value_count.to_le_bytes());
    if sparse {
        let mut words = vec![0u64; presence_words(row_count)];
        for &(row, _) in values {
            words[row as usize / 64] |= 1 << (row % 64);
        }
        for word in words {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    if let Some(zone_map) = &zone_map {
        encode_zone_map(zone_map, &mut out)?;
    }
    if let Some(epochs) = epochs {
        write_bitpacked_body(&BitPackedInts::pack(epochs), &mut out)?;
    }
    write_values(codec, values, &mut out)?;
    Ok((codec, out))
}

/// Decodes a chunk whose directory entry says `codec` and `row_count`;
/// refuses any mismatch.
///
/// The decoded values share one copy of the chunk, made once its header
/// checks pass (a chunk held in `Bytes` decodes without a copy through
/// [`decode_column_chunk_bytes`]). Every count is checked against the bytes
/// left before anything it sizes is allocated, and the chunk must be read
/// to its last byte.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the byte offset in the chunk of
/// what is wrong: a codec or row count other than the entry's, more rows
/// than the format's row cap (65,536), an unknown flag bit, no value or more
/// values than rows, a presence bitmap that is missing, needless or does not
/// mark exactly as many rows as there are
/// values (or marks a row past the chunk), a zone map other than the values'
/// minimum and maximum, epochs that are all 0 or not one per value, a body
/// that does not hold the header's number of values of the codec's kind,
/// truncated input, or bytes after the body.
pub fn decode_column_chunk(bytes: &[u8], codec: u8, row_count: u32) -> Result<ColumnChunk> {
    checked_header(bytes, codec, row_count)?;
    decode_column_chunk_bytes(&Bytes::copy_from_slice(bytes), codec, row_count)
}

/// [`decode_column_chunk`] of a chunk held in `Bytes` (as a section source
/// fetches it), read without a copy.
///
/// # Errors
///
/// As [`decode_column_chunk`].
pub fn decode_column_chunk_bytes(data: &Bytes, codec: u8, row_count: u32) -> Result<ColumnChunk> {
    let bytes: &[u8] = data;
    let ChunkHeader {
        codec,
        flags,
        value_count,
        sparse,
    } = checked_header(bytes, codec, row_count)?;
    let mut pos = HEADER_LEN;
    let rows = if sparse {
        Some(read_presence(bytes, &mut pos, row_count, value_count)?)
    } else {
        None
    };
    let zone_start = pos;
    if flags & FLAG_ZONE_MAP != 0 {
        skip_zone_map(codec, bytes, &mut pos)
            .map_err(|error| corrupt(zone_start, format!("zone map: {error}")))?;
    }
    let zone_end = pos;
    let epochs = if flags & FLAG_EPOCHS == 0 {
        None
    } else {
        Some(read_epochs(data, &mut pos, value_count as usize)?)
    };
    let values = read_values(codec, data, &mut pos, value_count as usize)?;
    if pos != bytes.len() {
        return Err(corrupt(
            pos,
            format!("{} bytes after the end of the values", bytes.len() - pos),
        ));
    }
    let zone_map = zone_map(codec, values.iter())?;
    let mut expected = Vec::new();
    if let Some(zone_map) = &zone_map {
        encode_zone_map(zone_map, &mut expected)?;
    }
    if bytes[zone_start..zone_end] != expected[..] {
        return Err(corrupt(
            zone_start,
            format!("the zone map is not the values' bounds {zone_map:?}"),
        ));
    }
    let values = match rows {
        Some(rows) => rows.into_iter().zip(values).collect(),
        None => (0u32..).zip(values).collect(),
    };
    Ok(ColumnChunk {
        row_count,
        values,
        epochs,
        zone_map,
    })
}

/// What a chunk's header says, once [`checked_header`] has checked it.
struct ChunkHeader {
    /// The codec of the directory entry, which the header repeats.
    codec: ChunkCodec,
    /// The flag bits, all known.
    flags: u8,
    /// The number of values: at least one, at most the row count.
    value_count: u32,
    /// Whether some rows have no value, so a presence bitmap follows.
    sparse: bool,
}

/// Checks the header of a chunk whose directory entry says `entry_codec` and
/// `row_count`, reading only its first [`HEADER_LEN`] bytes: the codec and
/// row count against the entry's, the row cap, the flag bits, the value
/// count, and the presence flag against the counts.
fn checked_header(bytes: &[u8], entry_codec: u8, row_count: u32) -> Result<ChunkHeader> {
    let codec = ChunkCodec::from_byte(entry_codec)
        .ok_or_else(|| corrupt(0, format!("unknown codec {entry_codec}")))?;
    let header: [u8; HEADER_LEN] = bytes
        .get(..HEADER_LEN)
        .and_then(|header| header.try_into().ok())
        .ok_or_else(|| {
            corrupt(
                bytes.len(),
                format!(
                    "the chunk ends inside its {HEADER_LEN}-byte header, after {} bytes",
                    bytes.len()
                ),
            )
        })?;
    if header[0] != entry_codec {
        return Err(corrupt(
            0,
            format!(
                "codec {} differs from the directory entry's codec {entry_codec}",
                header[0]
            ),
        ));
    }
    let flags = header[1];
    if flags & !KNOWN_FLAGS != 0 {
        return Err(corrupt(
            1,
            format!("unknown flag bits {:#04x}", flags & !KNOWN_FLAGS),
        ));
    }
    let stored_rows = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if stored_rows != row_count {
        return Err(corrupt(
            4,
            format!("row count {stored_rows} differs from the directory entry's {row_count}"),
        ));
    }
    if row_count > MAX_ROWS {
        return Err(corrupt(
            4,
            format!("{row_count} rows, where a chunk holds at most {MAX_ROWS}"),
        ));
    }
    let value_count = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
    if value_count == 0 {
        return Err(corrupt(8, "no value, where a chunk holds at least one"));
    }
    if value_count > row_count {
        return Err(corrupt(
            8,
            format!("{value_count} values in {row_count} rows"),
        ));
    }
    let sparse = value_count < row_count;
    if (flags & FLAG_PRESENCE != 0) != sparse {
        let what = if sparse {
            format!("{value_count} values in {row_count} rows without a presence bitmap")
        } else {
            "a presence bitmap where every row has a value".to_string()
        };
        return Err(corrupt(1, what));
    }
    Ok(ChunkHeader {
        codec,
        flags,
        value_count,
        sparse,
    })
}

/// An upper bound of what one value adds to a chunk: `encoded_len(value) + 4`.
#[must_use]
pub fn value_bound(value: &Value) -> usize {
    encoded_len(value) + 4
}

/// An upper bound of a chunk's size besides its values' bounds: the header
/// (12), the presence bitmap (`8 * ceil(row_count / 64)`), the zone map
/// (at most 43 bytes, a string zone map's), the codec body's own counts (16),
/// and with epochs `17 + 8 * value_count`.
#[must_use]
pub fn chunk_overhead(row_count: u32, value_count: usize, with_epochs: bool) -> usize {
    let presence = 8 * presence_words(row_count);
    let zone_map = ZONE_MAP_MAX_LEN;
    let epochs = if with_epochs {
        EPOCHS_COUNTS_LEN.saturating_add(value_count.saturating_mul(8))
    } else {
        0
    };
    (HEADER_LEN + presence + zone_map + BODY_COUNTS_LEN).saturating_add(epochs)
}

// The bodies the compact store's `ColumnCodec` writes after its own
// discriminant byte. The readers return the messages that codec reported
// before it shared them; the 8-byte bodies of `Float64` and `RawI64` now
// share one message.

/// Appends a bit-packed body: bits u8, count u32, word count u32, the words.
pub(crate) fn write_bitpacked_body(values: &BitPackedInts, out: &mut Vec<u8>) -> Result<()> {
    out.push(values.bits_per_value());
    put_u32(out, values.len(), "bit-packed value count")?;
    put_u32(out, values.word_count(), "bit-packed word count")?;
    out.extend_from_slice(values.data_bytes().as_ref());
    Ok(())
}

/// Reads a bit-packed body at `*pos`, sharing `data`'s storage.
pub(crate) fn read_bitpacked_body(
    data: &Bytes,
    pos: &mut usize,
) -> std::result::Result<BitPackedInts, &'static str> {
    let bytes = data.as_ref();
    let bits = *bytes.get(*pos).ok_or("truncated bits_per_value")?;
    *pos += 1;
    let count = read_u32(bytes, pos)? as usize;
    let word_count = read_u32(bytes, pos)? as usize;
    let need = word_count
        .checked_mul(8)
        .ok_or("BitPacked word count overflow")?;
    let words = take(data, pos, need).ok_or("truncated BitPacked data")?;
    Ok(BitPackedInts::from_bytes_storage(words, bits, count))
}

/// Appends a dictionary body: entry count u32, (length u32, UTF-8) per entry,
/// code count u32, the u32 codes.
pub(crate) fn write_dict_body(dict: &DictionaryEncoding, out: &mut Vec<u8>) -> Result<()> {
    let entries = dict.dictionary();
    put_u32(out, entries.len(), "dictionary entry count")?;
    for entry in entries.iter() {
        let entry = entry.as_bytes();
        put_u32(out, entry.len(), "dictionary entry length")?;
        out.extend_from_slice(entry);
    }
    put_u32(out, dict.code_count(), "dictionary code count")?;
    out.extend_from_slice(dict.codes_bytes().as_ref());
    Ok(())
}

/// Reads a dictionary body at `*pos`, sharing `data`'s storage for the codes.
pub(crate) fn read_dict_body(
    data: &Bytes,
    pos: &mut usize,
) -> std::result::Result<DictionaryEncoding, &'static str> {
    let bytes = data.as_ref();
    let entry_count = read_u32(bytes, pos)? as usize;
    // Every entry takes at least its 4-byte length.
    let mut entries: Vec<Arc<str>> =
        Vec::with_capacity(entry_count.min(bytes.len().saturating_sub(*pos) / 4));
    for _ in 0..entry_count {
        let len = read_u32(bytes, pos)? as usize;
        let end = end_of(bytes, *pos, len).ok_or("truncated dict string")?;
        let entry = std::str::from_utf8(&bytes[*pos..end]).map_err(|_| "invalid UTF-8 in dict")?;
        entries.push(Arc::from(entry));
        *pos = end;
    }
    let code_count = read_u32(bytes, pos)? as usize;
    let need = code_count.checked_mul(4).ok_or("Dict codes overflow")?;
    let codes = take(data, pos, need).ok_or("truncated Dict codes")?;
    Ok(DictionaryEncoding::from_bytes_storage(
        Arc::from(entries.into_boxed_slice()),
        codes,
        code_count,
    ))
}

/// Appends a bitmap body: bit count u32, word count u32, the words.
pub(crate) fn write_bitmap_body(bits: &BitVector, out: &mut Vec<u8>) -> Result<()> {
    put_u32(out, bits.len(), "bitmap bit count")?;
    put_u32(out, bits.word_count(), "bitmap word count")?;
    out.extend_from_slice(bits.data_bytes());
    Ok(())
}

/// Reads a bitmap body at `*pos`, sharing `data`'s storage.
pub(crate) fn read_bitmap_body(
    data: &Bytes,
    pos: &mut usize,
) -> std::result::Result<BitVector, &'static str> {
    let bytes = data.as_ref();
    let bit_count = read_u32(bytes, pos)? as usize;
    let word_count = read_u32(bytes, pos)? as usize;
    let need = word_count
        .checked_mul(8)
        .ok_or("Bitmap word count overflow")?;
    let words = take(data, pos, need).ok_or("truncated Bitmap data")?;
    Ok(BitVector::from_bytes_storage(words, bit_count))
}

/// Appends a body of 8-byte little-endian values (`Float64` and `RawI64`):
/// count u32, the bytes.
pub(crate) fn write_le_words_body(words: &[u8], out: &mut Vec<u8>) -> Result<()> {
    put_u32(out, words.len() / 8, "8-byte value count")?;
    out.extend_from_slice(words);
    Ok(())
}

/// Reads a body of 8-byte little-endian values at `*pos`: the values' bytes,
/// sharing `data`'s storage.
pub(crate) fn read_le_words_body(
    data: &Bytes,
    pos: &mut usize,
) -> std::result::Result<Bytes, &'static str> {
    let count = read_u32(data.as_ref(), pos)? as usize;
    let need = count.checked_mul(8).ok_or("8-byte value count overflow")?;
    take(data, pos, need).ok_or("truncated 8-byte values")
}

/// Appends a vector body: dimensions u16, component count u32, the
/// components' little-endian bytes.
pub(crate) fn write_f32_vector_body(
    components: &[u8],
    dimensions: u16,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.extend_from_slice(&dimensions.to_le_bytes());
    put_u32(out, components.len() / 4, "vector component count")?;
    out.extend_from_slice(components);
    Ok(())
}

/// Reads a vector body at `*pos`: the dimensions and the components' bytes,
/// sharing `data`'s storage.
pub(crate) fn read_f32_vector_body(
    data: &Bytes,
    pos: &mut usize,
) -> std::result::Result<(u16, Bytes), &'static str> {
    let bytes = data.as_ref();
    let dimensions = read_u16(bytes, pos)?;
    let component_count = read_u32(bytes, pos)? as usize;
    let need = component_count
        .checked_mul(4)
        .ok_or("Float32Vector length overflow")?;
    let components = take(data, pos, need).ok_or("truncated Float32Vector data")?;
    Ok((dimensions, components))
}

fn corrupt(at: usize, what: impl Display) -> Error {
    Error::Serialization(format!("column chunk, byte {at}: {what}"))
}

fn put_u32(out: &mut Vec<u8>, value: usize, what: &str) -> Result<()> {
    out.extend_from_slice(&checked_u32(value, what)?.to_le_bytes());
    Ok(())
}

/// Where `len` bytes from `pos` end, when `bytes` holds them.
fn end_of(bytes: &[u8], pos: usize, len: usize) -> Option<usize> {
    pos.checked_add(len).filter(|&end| end <= bytes.len())
}

fn take(data: &Bytes, pos: &mut usize, len: usize) -> Option<Bytes> {
    let end = end_of(data, *pos, len)?;
    let taken = data.slice(*pos..end);
    *pos = end;
    Some(taken)
}

fn read_u16(bytes: &[u8], pos: &mut usize) -> std::result::Result<u16, &'static str> {
    let end = end_of(bytes, *pos, 2).ok_or("truncated u16")?;
    let value = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]);
    *pos = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> std::result::Result<u32, &'static str> {
    let end = end_of(bytes, *pos, 4).ok_or("truncated u32")?;
    let value = u32::from_le_bytes([
        bytes[*pos],
        bytes[*pos + 1],
        bytes[*pos + 2],
        bytes[*pos + 3],
    ]);
    *pos = end;
    Ok(value)
}

fn presence_words(row_count: u32) -> usize {
    (row_count as usize).div_ceil(64)
}

fn check_rows(row_count: u32, values: &[(u32, Value)]) -> Result<()> {
    if row_count > MAX_ROWS {
        return Err(Error::Serialization(format!(
            "cannot write a column chunk of {row_count} rows: a chunk holds at most {MAX_ROWS}"
        )));
    }
    if values.is_empty() {
        return Err(Error::Serialization(
            "cannot write a column chunk with no value".to_string(),
        ));
    }
    let mut previous: Option<u32> = None;
    for &(row, _) in values {
        if row >= row_count {
            return Err(Error::Serialization(format!(
                "cannot write a column chunk: row {row} is past its {row_count} rows"
            )));
        }
        if previous.is_some_and(|previous| row <= previous) {
            return Err(Error::Serialization(format!(
                "cannot write a column chunk: row {row} does not ascend"
            )));
        }
        previous = Some(row);
    }
    Ok(())
}

/// Reads the presence bitmap at `*pos`: the rows that have a value.
fn read_presence(
    bytes: &[u8],
    pos: &mut usize,
    row_count: u32,
    value_count: u32,
) -> Result<Vec<u32>> {
    let at = *pos;
    let end = presence_words(row_count)
        .checked_mul(8)
        .and_then(|len| end_of(bytes, at, len))
        .ok_or_else(|| {
            corrupt(
                at,
                format!("the presence bitmap of {row_count} rows ends past the chunk"),
            )
        })?;
    let words = || {
        bytes[at..end]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| u64::from_le_bytes(*word))
    };
    let tail = row_count % 64;
    if tail != 0 && words().next_back().is_some_and(|last| last >> tail != 0) {
        return Err(corrupt(
            at,
            format!("the presence bitmap marks a row past row {row_count}"),
        ));
    }
    let marked: u64 = words().map(|word| u64::from(word.count_ones())).sum();
    if marked != u64::from(value_count) {
        return Err(corrupt(
            at,
            format!("the presence bitmap marks {marked} rows for {value_count} values"),
        ));
    }
    let mut rows = Vec::with_capacity(value_count as usize);
    for (mut word, first_row) in words().zip((0u32..).step_by(64)) {
        while word != 0 {
            rows.push(first_row + word.trailing_zeros());
            word &= word - 1;
        }
    }
    *pos = end;
    Ok(rows)
}

/// Checks a bit-packed body of `count` values: 1 to 64 bits each, and the
/// words they take.
fn check_bitpacked(packed: &BitPackedInts, count: usize) -> std::result::Result<(), String> {
    let bits = packed.bits_per_value();
    if bits == 0 || bits > 64 {
        return Err(format!("{bits} bits per value"));
    }
    if packed.len() != count {
        return Err(format!("{} values for {count}", packed.len()));
    }
    let expected = count.div_ceil(64 / usize::from(bits));
    if packed.word_count() != expected {
        return Err(format!(
            "{} words where {count} values of {bits} bits take {expected}",
            packed.word_count()
        ));
    }
    Ok(())
}

/// Reads the epochs at `*pos`: one per value, not all 0.
fn read_epochs(data: &Bytes, pos: &mut usize, value_count: usize) -> Result<Vec<u64>> {
    let at = *pos;
    let packed =
        read_bitpacked_body(data, pos).map_err(|error| corrupt(at, format!("epochs: {error}")))?;
    check_bitpacked(&packed, value_count)
        .map_err(|error| corrupt(at, format!("epochs: {error}")))?;
    let epochs = packed.unpack();
    if epochs.iter().all(|&epoch| epoch == 0) {
        return Err(corrupt(
            at,
            "epochs that are all 0, which the writer leaves out",
        ));
    }
    Ok(epochs)
}

/// Reads the body at `*pos`: exactly `value_count` values of `codec`'s kind.
fn read_values(
    codec: ChunkCodec,
    data: &Bytes,
    pos: &mut usize,
    value_count: usize,
) -> Result<Vec<Value>> {
    let at = *pos;
    let body = |error: &dyn Display| corrupt(at, format!("{codec:?} body: {error}"));
    let count_error = |count: usize| {
        corrupt(
            at,
            format!("{codec:?} body holds {count} values where the header says {value_count}"),
        )
    };
    match codec {
        ChunkCodec::BitPacked => {
            let packed = read_bitpacked_body(data, pos).map_err(|error| body(&error))?;
            if packed.len() != value_count {
                return Err(count_error(packed.len()));
            }
            check_bitpacked(&packed, value_count).map_err(|error| body(&error))?;
            packed
                .unpack()
                .into_iter()
                .map(|raw| {
                    i64::try_from(raw)
                        .map(Value::Int64)
                        .map_err(|_| body(&format!("value {raw} is past the largest Int64")))
                })
                .collect()
        }
        ChunkCodec::Dict => {
            let dict = read_dict_body(data, pos).map_err(|error| body(&error))?;
            if dict.code_count() != value_count {
                return Err(count_error(dict.code_count()));
            }
            let entries: Vec<ArcStr> = dict
                .dictionary()
                .iter()
                .map(|entry| ArcStr::from(&**entry))
                .collect();
            (0..value_count)
                .map(|index| {
                    let code = dict.code_at(index).unwrap_or(u32::MAX);
                    entries
                        .get(code as usize)
                        .map(|entry| Value::String(entry.clone()))
                        .ok_or_else(|| {
                            body(&format!("code {code} past the {} entries", entries.len()))
                        })
                })
                .collect()
        }
        ChunkCodec::Bitmap => {
            let bits = read_bitmap_body(data, pos).map_err(|error| body(&error))?;
            if bits.len() != value_count {
                return Err(count_error(bits.len()));
            }
            let expected = value_count.div_ceil(64);
            if bits.word_count() != expected {
                return Err(body(&format!(
                    "{} words where {value_count} bits take {expected}",
                    bits.word_count()
                )));
            }
            Ok(bits.iter().map(Value::Bool).collect())
        }
        ChunkCodec::Float64 | ChunkCodec::RawI64 => {
            let words = read_le_words_body(data, pos).map_err(|error| body(&error))?;
            if words.len() / 8 != value_count {
                return Err(count_error(words.len() / 8));
            }
            Ok(words
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&word| {
                    if codec == ChunkCodec::Float64 {
                        Value::Float64(f64::from_le_bytes(word))
                    } else {
                        Value::Int64(i64::from_le_bytes(word))
                    }
                })
                .collect())
        }
        ChunkCodec::Float32Vector => {
            let (dimensions, components) =
                read_f32_vector_body(data, pos).map_err(|error| body(&error))?;
            if dimensions == 0 {
                return Err(body(&"vectors of 0 dimensions"));
            }
            let dimensions = usize::from(dimensions);
            let count = components.len() / 4;
            if value_count.checked_mul(dimensions) != Some(count) {
                return Err(body(&format!(
                    "{count} components for {value_count} vectors of {dimensions} dimensions"
                )));
            }
            Ok(components
                .chunks_exact(4 * dimensions)
                .map(|vector| {
                    Value::Vector(
                        vector
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|&component| f32::from_le_bytes(component))
                            .collect(),
                    )
                })
                .collect())
        }
        ChunkCodec::Values => {
            let bytes = data.as_ref();
            let count = read_u32(bytes, pos).map_err(|error| body(&error))? as usize;
            if count != value_count {
                return Err(count_error(count));
            }
            // Every value takes at least its tag byte.
            let mut values = Vec::with_capacity(count.min(bytes.len().saturating_sub(*pos)));
            for _ in 0..count {
                values.push(decode_value(bytes, pos).map_err(|error| body(&error))?);
            }
            Ok(values)
        }
    }
}

/// Appends the body of `codec` holding `values`, whose kinds
/// [`choose_codec`] checked.
fn write_values(codec: ChunkCodec, values: &[(u32, Value)], out: &mut Vec<u8>) -> Result<()> {
    let other_kind = |value: &Value| {
        Error::Internal(format!(
            "a {codec:?} column chunk cannot hold the value {value:?}"
        ))
    };
    match codec {
        ChunkCodec::BitPacked => {
            let ints = values
                .iter()
                .map(|(_, value)| match value {
                    Value::Int64(n) => u64::try_from(*n).map_err(|_| other_kind(value)),
                    _ => Err(other_kind(value)),
                })
                .collect::<Result<Vec<u64>>>()?;
            write_bitpacked_body(&BitPackedInts::pack(&ints), out)
        }
        ChunkCodec::Dict => {
            let mut dict = DictionaryBuilder::with_capacity(values.len(), 0);
            for (_, value) in values {
                let Value::String(string) = value else {
                    return Err(other_kind(value));
                };
                dict.add(string.as_str());
            }
            write_dict_body(&dict.build(), out)
        }
        ChunkCodec::Bitmap => {
            let bools = values
                .iter()
                .map(|(_, value)| value.as_bool().ok_or_else(|| other_kind(value)))
                .collect::<Result<Vec<bool>>>()?;
            write_bitmap_body(&BitVector::from_bools(&bools), out)
        }
        ChunkCodec::Float64 | ChunkCodec::RawI64 => {
            let mut words = Vec::with_capacity(values.len() * 8);
            for (_, value) in values {
                let word = match (codec, value) {
                    (ChunkCodec::Float64, Value::Float64(float)) => float.to_le_bytes(),
                    (ChunkCodec::RawI64, Value::Int64(int)) => int.to_le_bytes(),
                    _ => return Err(other_kind(value)),
                };
                words.extend_from_slice(&word);
            }
            write_le_words_body(&words, out)
        }
        ChunkCodec::Float32Vector => {
            let dimensions = match values.first() {
                Some((_, Value::Vector(first))) => first.len(),
                _ => 0,
            };
            let mut components = Vec::with_capacity(values.len() * dimensions * 4);
            for (_, value) in values {
                match value {
                    Value::Vector(vector) if vector.len() == dimensions => {
                        for component in vector.iter() {
                            components.extend_from_slice(&component.to_le_bytes());
                        }
                    }
                    _ => return Err(other_kind(value)),
                }
            }
            let dimensions = checked_u16(dimensions, "vector dimensions")?;
            write_f32_vector_body(&components, dimensions, out)
        }
        ChunkCodec::Values => {
            put_u32(out, values.len(), "column chunk value count")?;
            for (_, value) in values {
                encode_value(value, out)?;
            }
            Ok(())
        }
    }
}

/// The zone map of `values`, when their codec has one: the minimum and
/// maximum of `Int64`, `Float64` without NaN (by total order, so -0.0 is
/// below 0.0) and `Bool`, and the [`StringZoneMap`] of strings.
///
/// # Errors
///
/// Returns [`Error::Serialization`] for a string longer than `u32::MAX`
/// bytes.
fn zone_map<'v>(
    codec: ChunkCodec,
    values: impl Iterator<Item = &'v Value> + Clone,
) -> Result<Option<ZoneMap>> {
    Ok(match codec {
        ChunkCodec::BitPacked | ChunkCodec::RawI64 => {
            min_max(values.filter_map(Value::as_int64), i64::cmp)
                .map(|(min, max)| ZoneMap::Values(Value::Int64(min), Value::Int64(max)))
        }
        ChunkCodec::Float64 => {
            let floats = values
                .filter_map(Value::as_float64)
                .filter(|float| !float.is_nan());
            min_max(floats, f64::total_cmp)
                .map(|(min, max)| ZoneMap::Values(Value::Float64(min), Value::Float64(max)))
        }
        ChunkCodec::Bitmap => min_max(values.filter_map(Value::as_bool), bool::cmp)
            .map(|(min, max)| ZoneMap::Values(Value::Bool(min), Value::Bool(max))),
        ChunkCodec::Dict => {
            StringZoneMap::of(values.filter_map(Value::as_str))?.map(ZoneMap::Strings)
        }
        ChunkCodec::Float32Vector | ChunkCodec::Values => None,
    })
}

/// Appends `zone_map` in its layout (see the module docs).
///
/// # Errors
///
/// Returns the value codec's error for a value it cannot encode.
fn encode_zone_map(zone_map: &ZoneMap, out: &mut Vec<u8>) -> Result<()> {
    match zone_map {
        ZoneMap::Values(min, max) => {
            encode_value(min, out)?;
            encode_value(max, out)?;
        }
        ZoneMap::Strings(strings) => {
            for bound in [&strings.min, &strings.max] {
                out.push(checked_u8(bound.len())?);
                out.extend_from_slice(bound);
            }
            let mut flags = 0;
            if strings.min_exact {
                flags |= MIN_EXACT;
            }
            if strings.max_exact {
                flags |= MAX_EXACT;
            }
            out.push(flags);
            out.extend_from_slice(&strings.min_len.to_le_bytes());
            out.extend_from_slice(&strings.max_len.to_le_bytes());
        }
    }
    Ok(())
}

/// The length byte of a string zone map bound, at most
/// [`STRING_ZONE_PREFIX`].
fn checked_u8(length: usize) -> Result<u8> {
    u8::try_from(length)
        .ok()
        .filter(|&length| usize::from(length) <= STRING_ZONE_PREFIX)
        .ok_or_else(|| {
            Error::Serialization(format!(
                "a string zone map bound of {length} bytes, past {STRING_ZONE_PREFIX}"
            ))
        })
}

/// Moves `pos` past the zone map of a `codec` chunk at `pos` in `bytes`,
/// refusing a bound longer than [`STRING_ZONE_PREFIX`] and unknown string
/// zone map flags; the caller then compares its bytes with the values'.
fn skip_zone_map(
    codec: ChunkCodec,
    bytes: &[u8],
    pos: &mut usize,
) -> std::result::Result<(), String> {
    if codec != ChunkCodec::Dict {
        for _ in 0..2 {
            decode_value(bytes, pos).map_err(|error| error.to_string())?;
        }
        return Ok(());
    }
    for bound in ["minimum", "maximum"] {
        let length = usize::from(*bytes.get(*pos).ok_or("truncated string zone map")?);
        if length > STRING_ZONE_PREFIX {
            return Err(format!(
                "a {bound} of {length} bytes, past {STRING_ZONE_PREFIX}"
            ));
        }
        *pos = end_of(bytes, *pos + 1, length).ok_or("truncated string zone map")?;
    }
    let flags = *bytes.get(*pos).ok_or("truncated string zone map")?;
    if flags & !(MIN_EXACT | MAX_EXACT) != 0 {
        return Err(format!("unknown string zone map flags {flags:#04x}"));
    }
    *pos = end_of(bytes, *pos + 1, 8).ok_or("truncated string zone map")?;
    Ok(())
}

fn min_max<T: Copy>(
    mut items: impl Iterator<Item = T>,
    compare: impl Fn(&T, &T) -> Ordering,
) -> Option<(T, T)> {
    let first = items.next()?;
    Some(items.fold((first, first), |(min, max), item| {
        (
            if compare(&item, &min).is_lt() {
                item
            } else {
                min
            },
            if compare(&item, &max).is_gt() {
                item
            } else {
                max
            },
        )
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use grafeo_common::storage::ChunkCaps;
    use grafeo_common::types::{
        Date, Duration, PropertyKey, Time, Timestamp, Value, ZonedDatetime,
    };

    use super::*;

    fn list(items: Vec<Value>) -> Value {
        Value::List(Arc::from(items))
    }

    fn map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(Arc::new(
            entries
                .into_iter()
                .map(|(key, value)| (PropertyKey::new(key), value))
                .collect::<BTreeMap<_, _>>(),
        ))
    }

    fn counter(entries: &[(&str, u64)]) -> Arc<HashMap<String, u64>> {
        Arc::new(
            entries
                .iter()
                .map(|(replica, count)| ((*replica).to_string(), *count))
                .collect(),
        )
    }

    fn vector(components: &[f32]) -> Value {
        Value::Vector(Arc::from(components))
    }

    /// One value of every kind, with the edge cases the 0.5.x formats lose
    /// (the list of the value codec's tests in grafeo-common).
    fn every_kind() -> Vec<Value> {
        vec![
            Value::Null,
            Value::Bool(true),
            Value::Int64(-19),
            Value::Int64(i64::MAX),
            Value::Float64(f64::from_bits(0x7FF8_0000_0000_0058)),
            Value::Float64(-0.0),
            Value::from("Amsterdam"),
            Value::from(""),
            Value::Bytes(Arc::from(vec![3u8, 19, 88])),
            Value::Date(Date::from_days(-3)),
            Value::Time(Time::from_nanos(88).unwrap()),
            Value::Time(
                Time::from_nanos(3_600_000_000_019)
                    .unwrap()
                    .with_offset(3600),
            ),
            Value::Timestamp(Timestamp::from_micros(1_696_500_000_123_457)),
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1_696_500_000_123_457),
                7200,
            )),
            Value::Duration(Duration::new(3, 19, 88)),
            list(vec![
                Value::Int64(3),
                list(vec![Value::from("Berlin")]),
                Value::Null,
            ]),
            map(vec![
                ("city", Value::from("Prague")),
                ("stops", list(vec![Value::Int64(19)])),
            ]),
            vector(&[3.0, -19.5, f32::NAN]),
            Value::Path {
                nodes: Arc::from(vec![map(vec![("_id", Value::Int64(3))])]),
                edges: Arc::from(Vec::<Value>::new()),
            },
            Value::GCounter(counter(&[("Alix", 3), ("Gus", 19)])),
            Value::OnCounter {
                pos: counter(&[("Mia", 88)]),
                neg: counter(&[("Jules", 3)]),
            },
            Value::Bool(false),
            Value::Time(Time::from_nanos(88).unwrap().with_offset(0)),
            Value::Path {
                nodes: Arc::from(vec![
                    map(vec![("_id", Value::Int64(3))]),
                    map(vec![("_id", Value::Int64(19))]),
                ]),
                edges: Arc::from(vec![map(vec![("_id", Value::Int64(88))])]),
            },
            list(vec![Value::Null]),
            map(vec![("", Value::Null)]),
            Value::GCounter(counter(&[("", 0)])),
        ]
    }

    fn all_same(a: &[Value], b: &[Value]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(x, y))
    }

    /// Equality that compares floats by their bits, recursively, and times
    /// and zoned datetimes with their offsets.
    fn same(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Float64(x), Value::Float64(y)) => x.to_bits() == y.to_bits(),
            (Value::Vector(x), Value::Vector(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|(p, q)| p.to_bits() == q.to_bits())
            }
            (Value::Time(x), Value::Time(y)) => {
                x.as_nanos() == y.as_nanos() && x.offset_seconds() == y.offset_seconds()
            }
            (Value::ZonedDatetime(x), Value::ZonedDatetime(y)) => {
                x.as_timestamp() == y.as_timestamp() && x.offset_seconds() == y.offset_seconds()
            }
            (Value::List(x), Value::List(y)) => all_same(x, y),
            (Value::Map(x), Value::Map(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|((key_x, value_x), (key_y, value_y))| {
                            key_x == key_y && same(value_x, value_y)
                        })
            }
            (
                Value::Path {
                    nodes: nodes_x,
                    edges: edges_x,
                },
                Value::Path {
                    nodes: nodes_y,
                    edges: edges_y,
                },
            ) => all_same(nodes_x, nodes_y) && all_same(edges_x, edges_y),
            _ => a == b,
        }
    }

    /// `values` on rows 0, 1, 2, ...
    fn dense(values: Vec<Value>) -> Vec<(u32, Value)> {
        values
            .into_iter()
            .zip(0u32..)
            .map(|(v, row)| (row, v))
            .collect()
    }

    fn row_count_of(values: &[(u32, Value)]) -> u32 {
        values.last().unwrap().0 + 1
    }

    /// Encodes and decodes `values`, asserting the rows and values come back
    /// equal by bits; returns the codec, the bytes and the decoded chunk.
    fn round_trip(
        row_count: u32,
        values: &[(u32, Value)],
        epochs: Option<&[u64]>,
    ) -> (ChunkCodec, Vec<u8>, ColumnChunk) {
        let (codec, bytes) = encode_column_chunk(row_count, values, epochs).unwrap();
        let back = decode_column_chunk(&bytes, codec.to_byte(), row_count).unwrap();
        assert_eq!(back.row_count, row_count);
        assert_eq!(back.values.len(), values.len(), "{codec:?}: {back:?}");
        assert!(
            back.values
                .iter()
                .zip(values)
                .all(|((a, x), (b, y))| a == b && same(x, y)),
            "{codec:?}: {values:?} came back as {back:?}"
        );
        (codec, bytes, back)
    }

    fn error_of(bytes: &[u8], codec: ChunkCodec, row_count: u32) -> String {
        decode_column_chunk(bytes, codec.to_byte(), row_count)
            .unwrap_err()
            .to_string()
    }

    /// A chunk built by hand: the header, then `rest`.
    fn crafted(
        codec: ChunkCodec,
        flags: u8,
        row_count: u32,
        value_count: u32,
        rest: &[u8],
    ) -> Vec<u8> {
        let mut bytes = vec![codec.to_byte(), flags, 0, 0];
        bytes.extend_from_slice(&row_count.to_le_bytes());
        bytes.extend_from_slice(&value_count.to_le_bytes());
        bytes.extend_from_slice(rest);
        bytes
    }

    fn encoded(value: &Value) -> Vec<u8> {
        let mut bytes = Vec::new();
        grafeo_common::storage::value_codec::encode_value(value, &mut bytes).unwrap();
        bytes
    }

    #[test]
    fn every_value_kind_round_trips_through_a_column_chunk() {
        // every_kind(), one per row with a gap of 3 rows between values, in one chunk.
        let values: Vec<(u32, Value)> = every_kind()
            .into_iter()
            .enumerate()
            .map(|(i, v)| (u32::try_from(i * 4).unwrap(), v))
            .collect();
        let row_count = values.last().unwrap().0 + 1;
        let (codec, bytes) = encode_column_chunk(row_count, &values, None).unwrap();
        assert_eq!(codec, ChunkCodec::Values);
        let back = decode_column_chunk(&bytes, codec.to_byte(), row_count).unwrap();
        assert_eq!(back.row_count, row_count);
        assert_eq!(back.values.len(), values.len());
        assert!(
            back.values
                .iter()
                .zip(&values)
                .all(|((a, x), (b, y))| a == b && same(x, y)),
            "{back:?}"
        );
        assert_eq!(back.epochs, None);
        assert_eq!(back.zone_map, None, "a Values chunk has no zone map");
    }

    #[test]
    fn int64_and_float64_in_one_column_are_not_coalesced() {
        let values = vec![(0, Value::Int64(3)), (1, Value::Float64(19.5))];
        let (codec, bytes) = encode_column_chunk(2, &values, None).unwrap();
        assert_eq!(codec, ChunkCodec::Values);
        assert_eq!(
            decode_column_chunk(&bytes, codec.to_byte(), 2)
                .unwrap()
                .values,
            values
        );
    }

    #[test]
    fn the_codec_follows_the_values() {
        let pick = |values: Vec<Value>| choose_codec(values.iter());
        let cases = [
            (
                vec![Value::Int64(3), Value::Int64(88)],
                ChunkCodec::BitPacked,
            ),
            (
                vec![Value::Int64(0), Value::Int64(i64::MAX)],
                ChunkCodec::BitPacked,
            ),
            (vec![Value::Int64(3), Value::Int64(-19)], ChunkCodec::RawI64),
            (
                vec![Value::Int64(i64::MIN), Value::Int64(i64::MAX)],
                ChunkCodec::RawI64,
            ),
            (vec![Value::Float64(3.0)], ChunkCodec::Float64),
            (
                vec![
                    Value::Float64(f64::from_bits(0x7FF8_0000_0000_0058)),
                    Value::Float64(-0.0),
                ],
                ChunkCodec::Float64,
            ),
            (
                vec![Value::Bool(true), Value::Bool(false)],
                ChunkCodec::Bitmap,
            ),
            (vec![Value::from("Alix"), Value::from("")], ChunkCodec::Dict),
            (
                vec![Value::from("Gus"), Value::from("Mia"), Value::from("Gus")],
                ChunkCodec::Dict,
            ),
            (
                vec![vector(&[3.0, 19.0]), vector(&[88.0, f32::NAN])],
                ChunkCodec::Float32Vector,
            ),
            (vec![vector(&[-0.0])], ChunkCodec::Float32Vector),
            (
                vec![vector(&[3.0]), vector(&[88.0, 3.0])],
                ChunkCodec::Values,
            ),
            (vec![vector(&[])], ChunkCodec::Values),
            (
                vec![Value::Timestamp(Timestamp::from_micros(3))],
                ChunkCodec::Values,
            ),
            (vec![Value::Null], ChunkCodec::Values),
            (vec![Value::Int64(3), Value::Null], ChunkCodec::Values),
            (
                vec![Value::from("Paris"), Value::Bool(true)],
                ChunkCodec::Values,
            ),
        ];
        for (values, expected) in cases {
            assert_eq!(pick(values.clone()), expected, "{values:?}");
            // each case also round trips through encode and decode, equal by bits
            let rows = dense(values);
            let (codec, _, _) = round_trip(row_count_of(&rows), &rows, None);
            assert_eq!(codec, expected, "encode uses the chosen codec");
        }
        let too_wide = vector(&vec![0.5; 65_536]);
        assert_eq!(pick(vec![too_wide.clone()]), ChunkCodec::Values);
        round_trip(1, &[(0, too_wide)], None);
        let widest = vector(&vec![0.5; 65_535]);
        assert_eq!(pick(vec![widest.clone()]), ChunkCodec::Float32Vector);
        round_trip(1, &[(0, widest)], None);
    }

    #[test]
    fn rows_without_a_value_cost_a_bit() {
        let values = vec![
            (3, Value::Float64(1.88)),
            (19_000, Value::Float64(3.0)),
            (65_535, Value::Float64(-0.0)),
        ];
        let (_, bytes) = encode_column_chunk(65_536, &values, None).unwrap();
        assert!(
            bytes.len() <= 12 + 8 * 1024 + 4 + 3 * 8 + 2 * 9,
            "dense: {} bytes",
            bytes.len()
        );
        round_trip(65_536, &values, None);
    }

    #[test]
    fn zone_maps_hold_the_minimum_and_maximum_of_orderable_values() {
        let zone = |values: Vec<Value>| {
            let rows = dense(values);
            let row_count = row_count_of(&rows);
            let (codec, bytes) = encode_column_chunk(row_count, &rows, None).unwrap();
            decode_column_chunk(&bytes, codec.to_byte(), row_count)
                .unwrap()
                .zone_map
        };
        let values = |min: Value, max: Value| Some(ZoneMap::Values(min, max));
        assert_eq!(
            zone(vec![Value::Int64(19), Value::Int64(-3), Value::Int64(88)]),
            values(Value::Int64(-3), Value::Int64(88))
        );
        assert_eq!(
            zone(vec![Value::Int64(19), Value::Int64(3), Value::Int64(88)]),
            values(Value::Int64(3), Value::Int64(88))
        );
        assert_eq!(
            zone(vec![
                Value::Float64(f64::NAN),
                Value::Float64(3.0),
                Value::Float64(1.88)
            ]),
            values(Value::Float64(1.88), Value::Float64(3.0))
        );
        assert_eq!(
            zone(vec![Value::Float64(f64::NAN), Value::Float64(f64::NAN)]),
            None,
            "every value is NaN"
        );
        assert_eq!(
            zone(vec![Value::Bool(true), Value::Bool(false)]),
            values(Value::Bool(false), Value::Bool(true))
        );
        assert_eq!(
            zone(vec![Value::from("Prague"), Value::from("Amsterdam")]),
            Some(ZoneMap::Strings(StringZoneMap {
                min: b"Amsterdam".to_vec(),
                min_exact: true,
                max: b"Prague".to_vec(),
                max_exact: true,
                min_len: 6,
                max_len: 9,
            })),
            "short strings: exact bounds"
        );
        assert_eq!(
            zone(vec![Value::from("Berlin"), Value::from("x".repeat(65))]),
            Some(ZoneMap::Strings(StringZoneMap {
                min: b"Berlin".to_vec(),
                min_exact: true,
                max: b"xxxxxxxxxxxxxxxy".to_vec(),
                max_exact: false,
                min_len: 6,
                max_len: 65,
            })),
            "a long maximum is cut to 16 bytes and raised"
        );
        assert_eq!(
            zone(vec![Value::from("A".repeat(65)), Value::from("B")]),
            Some(ZoneMap::Strings(StringZoneMap {
                min: b"AAAAAAAAAAAAAAAA".to_vec(),
                min_exact: false,
                max: b"B".to_vec(),
                max_exact: true,
                min_len: 1,
                max_len: 65,
            })),
            "a long minimum is cut to 16 bytes"
        );
        assert_eq!(zone(vec![Value::Date(Date::from_days(3))]), None);
        assert_eq!(zone(vec![vector(&[3.0, 19.0])]), None);
    }

    /// A string zone map cuts on bytes, also inside a character: the cut
    /// maximum need not be UTF-8, and is above every string the chunk holds.
    #[test]
    fn a_cut_maximum_is_raised_inside_a_character() {
        // 21 bytes: 5A, then C3 A9 ten times; the 16th byte is a C3 that
        // starts a character.
        let long = format!("Z{}", "\u{e9}".repeat(10));
        let rows = dense(vec![Value::from(long.as_str()), Value::from("Gus")]);
        let (codec, bytes) = encode_column_chunk(2, &rows, None).unwrap();
        let Some(ZoneMap::Strings(strings)) = decode_column_chunk(&bytes, codec.to_byte(), 2)
            .unwrap()
            .zone_map
        else {
            panic!("a string chunk has a string zone map");
        };
        let mut raised = long.as_bytes()[..16].to_vec();
        raised[15] += 1;
        assert_eq!(strings.max, raised, "5A C3 A9 ... C4");
        assert!(std::str::from_utf8(&strings.max).is_err(), "not UTF-8");
        assert!(!strings.max_exact);
        assert!(strings.may_hold(&long));
    }

    /// The guarantee a reader prunes by: every string of a chunk is within
    /// its zone map, whatever the strings (empty, multi-byte, long, sharing
    /// their first 16 bytes with the maximum), and a string the bounds or
    /// lengths rule out is not.
    #[test]
    fn every_string_of_a_chunk_is_within_its_zone_map() {
        // 16 bytes: a bicycle (F0 9F 9A B2), the highest first character.
        let shared = "\u{1f6b2}abcdefghijkl";
        let strings: Vec<String> = vec![
            String::new(),
            "Amsterdam".into(),
            "\u{c5}rhus".into(),
            "\u{1f6b2} Berlin".into(),
            format!("{shared}Z"),
            format!("{shared}ZZZZ"),
            "Prague".repeat(5),
            shared.to_string(),
        ];
        let rows = dense(
            strings
                .iter()
                .map(|text| Value::from(text.as_str()))
                .collect(),
        );
        let row_count = row_count_of(&rows);
        let (codec, bytes) = encode_column_chunk(row_count, &rows, None).unwrap();
        let Some(ZoneMap::Strings(zone)) = decode_column_chunk(&bytes, codec.to_byte(), row_count)
            .unwrap()
            .zone_map
        else {
            panic!("a string chunk has a string zone map");
        };
        for text in &strings {
            assert!(zone.may_hold(text), "{text:?} is in the chunk");
        }
        assert_eq!((zone.min_len, zone.max_len), (0, 30));
        assert!(
            zone.min_exact && zone.min.is_empty(),
            "the empty string is the minimum"
        );
        // The largest string starts with the shared 16 bytes: a longer one
        // with the same start may be there, one past the raised bound not.
        assert!(!zone.max_exact);
        assert!(zone.may_hold(&format!("{shared}ZZZZZZ")), "same 16 bytes");
        assert!(
            !zone.may_hold("\u{1f6b2}abcdefghijkm"),
            "at the raised bound"
        );
        assert!(!zone.may_hold("\u{1f6b3}"), "above every string");
        assert!(!zone.may_hold(&"A".repeat(31)), "longer than the longest");
    }

    /// The decoder refuses a string zone map other than the strings', a bound
    /// past 16 bytes and an unknown flag bit.
    #[test]
    fn a_string_zone_map_other_than_the_strings_is_refused() {
        let rows = dense(vec![Value::from("Berlin"), Value::from("Prague")]);
        let (codec, bytes) = encode_column_chunk(2, &rows, None).unwrap();
        assert_eq!(codec, ChunkCodec::Dict);
        // header (12), then the minimum's length at 12, "Berlin" at 13..19,
        // the maximum's length at 19, "Prague" at 20..26, the flags at 26.
        assert_eq!(bytes[12], 6);
        assert_eq!(&bytes[13..19], b"Berlin");
        assert_eq!(bytes[26], MIN_EXACT | MAX_EXACT);
        let refused = |at: usize, byte: u8, what: &str| {
            let mut crafted = bytes.clone();
            crafted[at] = byte;
            let error = error_of(&crafted, codec, 2);
            assert!(error.contains("zone map"), "{what}: {error}");
        };
        refused(13, b'A', "another minimum");
        refused(26, MIN_EXACT, "a maximum marked cut");
        refused(26, MIN_EXACT | MAX_EXACT | 4, "an unknown flag bit");
        refused(12, 17, "a minimum past 16 bytes");
        refused(27, 7, "another shortest length");
    }

    /// The decoder rebuilds the zone map and refuses any other, so its order
    /// is part of the format: floats in total order, where -0.0 is below 0.0.
    #[test]
    fn negative_zero_is_below_zero_in_a_float_zone_map() {
        let bits = |zone: Option<ZoneMap>| match zone {
            Some(ZoneMap::Values(Value::Float64(min), Value::Float64(max))) => {
                (min.to_bits(), max.to_bits())
            }
            other => panic!("not a Float64 zone map: {other:?}"),
        };
        for floats in [[0.0, -0.0], [-0.0, 0.0]] {
            let rows = dense(floats.into_iter().map(Value::Float64).collect());
            let (codec, bytes) = encode_column_chunk(2, &rows, None).unwrap();
            let zone_map = decode_column_chunk(&bytes, codec.to_byte(), 2)
                .unwrap()
                .zone_map;
            assert_eq!(
                bits(zone_map),
                ((-0.0f64).to_bits(), 0.0f64.to_bits()),
                "{floats:?}"
            );
        }
        let rows = dense(vec![Value::Float64(0.0), Value::Float64(-0.0)]);
        let (codec, bytes) = encode_column_chunk(2, &rows, None).unwrap();
        // header (12), then the zone map's minimum at 12..21 and maximum at 21..30
        assert_eq!(&bytes[12..21], encoded(&Value::Float64(-0.0)).as_slice());
        let mut both_zero = bytes.clone();
        both_zero[12..21].copy_from_slice(&encoded(&Value::Float64(0.0)));
        let error = error_of(&both_zero, codec, 2);
        assert!(
            error.contains("zone map"),
            "(0.0, 0.0) for -0.0 and 0.0: {error}"
        );
    }

    #[test]
    fn epochs_are_written_only_when_one_is_not_zero() {
        let values = vec![(0, Value::Int64(3)), (2, Value::Int64(19))];
        let (_, plain) = encode_column_chunk(3, &values, Some(&[0, 0])).unwrap();
        assert_eq!(decode_column_chunk(&plain, 1, 3).unwrap().epochs, None);
        let (_, versioned) = encode_column_chunk(3, &values, Some(&[3, 88])).unwrap();
        assert_eq!(
            decode_column_chunk(&versioned, 1, 3).unwrap().epochs,
            Some(vec![3, 88])
        );
        let (_, _, back) = round_trip(3, &values, Some(&[0, u64::MAX]));
        assert_eq!(back.epochs, Some(vec![0, u64::MAX]));
        assert_eq!(
            encode_column_chunk(3, &values, None).unwrap().1,
            plain,
            "all-zero epochs write what no epochs write"
        );
    }

    #[test]
    fn a_corrupt_chunk_is_refused_not_misread() {
        let values = vec![(0, Value::from("Mia")), (5, Value::from("Jules"))];
        let (codec, bytes) = encode_column_chunk(8, &values, None).unwrap();
        assert!(
            decode_column_chunk(&bytes, ChunkCodec::Values.to_byte(), 8).is_err(),
            "codec differs from the entry"
        );
        assert!(
            decode_column_chunk(&bytes, codec.to_byte(), 9).is_err(),
            "row count differs from the entry"
        );
        let mut flag = bytes.clone();
        flag[1] |= 0x80;
        assert!(
            decode_column_chunk(&flag, codec.to_byte(), 8).is_err(),
            "unknown flag bit"
        );
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(
            decode_column_chunk(&longer, codec.to_byte(), 8).is_err(),
            "trailing bytes"
        );
        for cut in 0..bytes.len() {
            assert!(
                decode_column_chunk(&bytes[..cut], codec.to_byte(), 8).is_err(),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn header_mismatches_are_named() {
        let values = vec![(0, Value::from("Mia")), (5, Value::from("Jules"))];
        let (codec, bytes) = encode_column_chunk(8, &values, None).unwrap();
        let error = error_of(&bytes, ChunkCodec::Values, 8);
        assert!(
            error.contains("codec") && error.contains("entry"),
            "{error}"
        );
        let error = decode_column_chunk(&bytes, 0, 8).unwrap_err().to_string();
        assert!(error.contains("unknown codec 0"), "{error}");
        let error = error_of(&bytes, codec, 9);
        assert!(error.contains("row count 8"), "{error}");
        let mut flag = bytes.clone();
        flag[1] |= 0x08;
        let error = error_of(&flag, codec, 8);
        assert!(error.contains("flag"), "{error}");
        let mut longer = bytes.clone();
        longer.extend_from_slice(&[3, 19]);
        let error = error_of(&longer, codec, 8);
        assert!(error.contains("2 bytes after"), "{error}");
        let mut reserved = bytes.clone();
        reserved[2] = 88;
        reserved[3] = 19;
        assert_eq!(
            decode_column_chunk(&reserved, codec.to_byte(), 8)
                .unwrap()
                .values,
            values,
            "the reserved field is ignored"
        );
    }

    #[test]
    fn value_counts_must_fit_the_rows() {
        let null = encoded(&Value::Null);
        let mut body = 1u32.to_le_bytes().to_vec();
        body.extend_from_slice(&null);
        let none = crafted(ChunkCodec::Values, 0, 1, 0, &0u32.to_le_bytes());
        assert!(error_of(&none, ChunkCodec::Values, 1).contains("no value"));
        let more = crafted(ChunkCodec::Values, 0, 1, 2, &body);
        assert!(error_of(&more, ChunkCodec::Values, 1).contains("2 values in 1 row"));
        let empty = crafted(ChunkCodec::Values, 0, 0, 0, &0u32.to_le_bytes());
        assert!(decode_column_chunk(&empty, ChunkCodec::Values.to_byte(), 0).is_err());
        let exact = crafted(ChunkCodec::Values, 0, 1, 1, &body);
        assert_eq!(
            decode_column_chunk(&exact, ChunkCodec::Values.to_byte(), 1)
                .unwrap()
                .values,
            vec![(0, Value::Null)]
        );
    }

    #[test]
    fn presence_bitmaps_are_checked() {
        let null = encoded(&Value::Null);
        let mut body = 1u32.to_le_bytes().to_vec();
        body.extend_from_slice(&null);
        let with_presence = |word: u64| {
            let mut rest = word.to_le_bytes().to_vec();
            rest.extend_from_slice(&body);
            crafted(ChunkCodec::Values, 1, 3, 1, &rest)
        };
        let good = with_presence(0b100);
        assert_eq!(
            decode_column_chunk(&good, ChunkCodec::Values.to_byte(), 3)
                .unwrap()
                .values,
            vec![(2, Value::Null)]
        );
        let past = with_presence(0b1000);
        assert!(
            error_of(&past, ChunkCodec::Values, 3).contains("past row 3"),
            "a bit past the rows"
        );
        let two = with_presence(0b101);
        assert!(
            error_of(&two, ChunkCodec::Values, 3).contains("2 rows for 1 value"),
            "more bits than values"
        );
        let zero = with_presence(0);
        assert!(
            error_of(&zero, ChunkCodec::Values, 3).contains("0 rows for 1 value"),
            "fewer bits than values"
        );
        let missing = crafted(ChunkCodec::Values, 0, 3, 1, &body);
        assert!(
            error_of(&missing, ChunkCodec::Values, 3).contains("presence"),
            "a sparse chunk without its bitmap"
        );
        let mut dense_body = 1u32.to_le_bytes().to_vec();
        dense_body.extend_from_slice(&null);
        let mut rest = 1u64.to_le_bytes().to_vec();
        rest.extend_from_slice(&dense_body);
        let needless = crafted(ChunkCodec::Values, 1, 1, 1, &rest);
        assert!(
            error_of(&needless, ChunkCodec::Values, 1).contains("presence"),
            "a bitmap on a chunk where every row has a value"
        );
    }

    #[test]
    fn a_zone_map_other_than_the_values_minimum_and_maximum_is_refused() {
        let values = dense(vec![Value::Int64(19), Value::Int64(-3), Value::Int64(88)]);
        let (codec, bytes) = encode_column_chunk(3, &values, None).unwrap();
        assert_eq!(codec, ChunkCodec::RawI64);
        // header (12), then the zone map: Int64 tag at 12, the minimum at 13..21
        assert_eq!(&bytes[12..21], encoded(&Value::Int64(-3)).as_slice());
        let mut lower = bytes.clone();
        lower[13..21].copy_from_slice(&(-4i64).to_le_bytes());
        let error = error_of(&lower, codec, 3);
        assert!(error.contains("zone map"), "{error}");
        let mut float = bytes.clone();
        float[12] = 3; // Float64 tag, same bytes
        assert!(error_of(&float, codec, 3).contains("zone map"));
        let mut without = bytes[..12].to_vec();
        without[1] &= !2;
        without.extend_from_slice(&bytes[30..]);
        let error = error_of(&without, codec, 3);
        assert!(error.contains("zone map"), "a missing zone map: {error}");
        let date = dense(vec![Value::Date(Date::from_days(3))]);
        let (codec, bytes) = encode_column_chunk(1, &date, None).unwrap();
        let mut with = bytes[..12].to_vec();
        with[1] |= 2;
        with.extend_from_slice(&encoded(&Value::Date(Date::from_days(3))));
        with.extend_from_slice(&encoded(&Value::Date(Date::from_days(3))));
        with.extend_from_slice(&bytes[12..]);
        let error = error_of(&with, codec, 1);
        assert!(error.contains("zone map"), "a zone map on Values: {error}");
    }

    #[test]
    fn epochs_are_checked() {
        let values = dense(vec![Value::Int64(3), Value::Int64(19)]);
        let mut plain_body = Vec::new();
        write_bitpacked_body(&BitPackedInts::pack(&[3, 19]), &mut plain_body).unwrap();
        let zone = [encoded(&Value::Int64(3)), encoded(&Value::Int64(19))].concat();
        let with_epochs = |epochs: &BitPackedInts| {
            let mut rest = zone.clone();
            write_bitpacked_body(epochs, &mut rest).unwrap();
            rest.extend_from_slice(&plain_body);
            crafted(ChunkCodec::BitPacked, 2 | 4, 2, 2, &rest)
        };
        let good = with_epochs(&BitPackedInts::pack(&[0, 88]));
        assert_eq!(
            decode_column_chunk(&good, 1, 2).unwrap(),
            ColumnChunk {
                row_count: 2,
                values: values.clone(),
                epochs: Some(vec![0, 88]),
                zone_map: Some(ZoneMap::Values(Value::Int64(3), Value::Int64(19))),
            }
        );
        let zeros = with_epochs(&BitPackedInts::pack(&[0, 0]));
        assert!(error_of(&zeros, ChunkCodec::BitPacked, 2).contains("epochs"));
        let short = with_epochs(&BitPackedInts::pack(&[88]));
        assert!(error_of(&short, ChunkCodec::BitPacked, 2).contains("epochs"));
        let no_bits = with_epochs(&BitPackedInts::pack_with_bits(&[0, 0], 0));
        assert!(error_of(&no_bits, ChunkCodec::BitPacked, 2).contains("epochs"));
    }

    #[test]
    fn typed_bodies_are_checked_against_the_header() {
        let body = |write: &dyn Fn(&mut Vec<u8>)| {
            let mut bytes = Vec::new();
            write(&mut bytes);
            bytes
        };
        // a bit-packed value that is no Int64
        let rest = body(&|out| {
            write_bitpacked_body(&BitPackedInts::pack(&[u64::MAX]), out).unwrap();
        });
        let chunk = crafted(ChunkCodec::BitPacked, 0, 1, 1, &rest);
        let error = error_of(&chunk, ChunkCodec::BitPacked, 1);
        assert!(error.contains("18446744073709551615"), "{error}");
        // bit-packed bodies of 0 bits would decode values from no bytes
        let rest = body(&|out| {
            write_bitpacked_body(&BitPackedInts::pack_with_bits(&[0; 19], 0), out).unwrap();
        });
        let chunk = crafted(ChunkCodec::BitPacked, 0, 19, 19, &rest);
        assert!(error_of(&chunk, ChunkCodec::BitPacked, 19).contains("bits"));
        // a bit-packed body whose word count does not fit its values
        let mut rest = body(&|out| {
            write_bitpacked_body(&BitPackedInts::pack(&[3]), out).unwrap();
        });
        rest[5] = 2; // word count 2 for one value
        rest.extend_from_slice(&[0; 8]);
        let chunk = crafted(ChunkCodec::BitPacked, 0, 1, 1, &rest);
        assert!(error_of(&chunk, ChunkCodec::BitPacked, 1).contains("words"));
        // a dictionary code past the entries
        let rest = body(&|out| {
            let dict = DictionaryEncoding::new(Arc::from(vec![Arc::<str>::from("Mia")]), vec![1]);
            write_dict_body(&dict, out).unwrap();
        });
        let chunk = crafted(ChunkCodec::Dict, 0, 1, 1, &rest);
        let error = error_of(&chunk, ChunkCodec::Dict, 1);
        assert!(error.contains("code 1"), "{error}");
        // a body holding another number of values than the header
        let rest = body(&|out| write_le_words_body(&3i64.to_le_bytes(), out).unwrap());
        let chunk = crafted(ChunkCodec::RawI64, 0, 2, 2, &rest);
        assert!(error_of(&chunk, ChunkCodec::RawI64, 2).contains("1 value"));
        let chunk = crafted(ChunkCodec::Float64, 0, 2, 2, &rest);
        assert!(error_of(&chunk, ChunkCodec::Float64, 2).contains("1 value"));
        let rest = body(&|out| {
            write_bitmap_body(&BitVector::from_bools(&[true, false, true]), out).unwrap();
        });
        let chunk = crafted(ChunkCodec::Bitmap, 0, 2, 2, &rest);
        assert!(error_of(&chunk, ChunkCodec::Bitmap, 2).contains("3 values"));
        // a bitmap body whose word count does not fit its bits
        let mut rest = body(&|out| {
            write_bitmap_body(&BitVector::from_bools(&[true]), out).unwrap();
        });
        rest[4] = 2;
        rest.extend_from_slice(&[0; 8]);
        let chunk = crafted(ChunkCodec::Bitmap, 0, 1, 1, &rest);
        assert!(error_of(&chunk, ChunkCodec::Bitmap, 1).contains("words"));
        // vectors without components, and components that do not fill the vectors
        let rest = body(&|out| write_f32_vector_body(&[], 0, out).unwrap());
        let chunk = crafted(ChunkCodec::Float32Vector, 0, 1, 1, &rest);
        assert!(error_of(&chunk, ChunkCodec::Float32Vector, 1).contains("0 dimensions"));
        let components: Vec<u8> = [3.0f32, 19.0, 88.0]
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        let rest = body(&|out| write_f32_vector_body(&components, 2, out).unwrap());
        let chunk = crafted(ChunkCodec::Float32Vector, 0, 1, 1, &rest);
        assert!(error_of(&chunk, ChunkCodec::Float32Vector, 1).contains("3 components"));
    }

    #[test]
    fn a_values_body_holds_the_headers_value_count() {
        let body = |count: u32| {
            let mut body = count.to_le_bytes().to_vec();
            for _ in 0..count {
                body.extend_from_slice(&encoded(&Value::Null));
            }
            body
        };
        let both = crafted(ChunkCodec::Values, 0, 2, 2, &body(2));
        assert_eq!(
            decode_column_chunk(&both, ChunkCodec::Values.to_byte(), 2)
                .unwrap()
                .values,
            vec![(0, Value::Null), (1, Value::Null)]
        );
        let short = crafted(ChunkCodec::Values, 0, 2, 2, &body(1));
        let error = error_of(&short, ChunkCodec::Values, 2);
        assert!(
            error.contains("1 values where the header says 2"),
            "{error}"
        );
    }

    #[test]
    fn a_dict_body_has_one_code_per_value() {
        // the zone map the one value gives, so only the code count can be wrong
        let mia = Value::from("Mia");
        let chunk = |codes: Vec<u32>| {
            let mut rest = Vec::new();
            encode_zone_map(
                &zone_map(ChunkCodec::Dict, [&mia].into_iter())
                    .unwrap()
                    .unwrap(),
                &mut rest,
            )
            .unwrap();
            let dict = DictionaryEncoding::new(Arc::from(vec![Arc::<str>::from("Mia")]), codes);
            write_dict_body(&dict, &mut rest).unwrap();
            crafted(ChunkCodec::Dict, 2, 1, 1, &rest)
        };
        assert_eq!(
            decode_column_chunk(&chunk(vec![0]), ChunkCodec::Dict.to_byte(), 1)
                .unwrap()
                .values,
            vec![(0, mia.clone())]
        );
        let error = error_of(&chunk(vec![0, 0]), ChunkCodec::Dict, 1);
        assert!(
            error.contains("2 values where the header says 1"),
            "{error}"
        );
    }

    #[test]
    fn huge_counts_are_refused_before_allocating() {
        let max = u32::MAX;
        for codec in [
            ChunkCodec::Values,
            ChunkCodec::Dict,
            ChunkCodec::Float64,
            ChunkCodec::RawI64,
            ChunkCodec::Bitmap,
        ] {
            let chunk = crafted(codec, 0, max, max, &max.to_le_bytes());
            assert!(
                decode_column_chunk(&chunk, codec.to_byte(), max).is_err(),
                "{codec:?}"
            );
        }
        let mut rest = vec![1];
        rest.extend_from_slice(&max.to_le_bytes());
        rest.extend_from_slice(&max.to_le_bytes());
        let chunk = crafted(ChunkCodec::BitPacked, 0, max, max, &rest);
        assert!(decode_column_chunk(&chunk, 1, max).is_err());
        let mut rest = 1u16.to_le_bytes().to_vec();
        rest.extend_from_slice(&max.to_le_bytes());
        let chunk = crafted(ChunkCodec::Float32Vector, 0, max, max, &rest);
        assert!(decode_column_chunk(&chunk, 6, max).is_err());
        let sparse = crafted(ChunkCodec::Values, 1, max, 1, &[]);
        assert!(
            decode_column_chunk(&sparse, 7, max).is_err(),
            "a bitmap past the bytes"
        );
    }

    /// A dense `BitPacked` chunk of `rows` rows, each the value 3, built by
    /// hand: a bounded body that decodes into `rows` values.
    fn threes(rows: u32) -> Vec<u8> {
        let mut rest = encoded(&Value::Int64(3));
        rest.extend(encoded(&Value::Int64(3)));
        rest.push(2);
        rest.extend_from_slice(&rows.to_le_bytes());
        let words = rows.div_ceil(32);
        rest.extend_from_slice(&words.to_le_bytes());
        for word in 0..words {
            // 32 values of 2 bits per word; the last word holds the rest.
            let held = if word + 1 < words {
                32
            } else {
                rows - 32 * word
            };
            let bits = if held == 32 {
                u64::MAX
            } else {
                (1u64 << (2 * held)) - 1
            };
            rest.extend_from_slice(&bits.to_le_bytes());
        }
        crafted(ChunkCodec::BitPacked, FLAG_ZONE_MAP, rows, rows, &rest)
    }

    /// A chunk decodes into one value per row, so a row count the writer
    /// never uses would let a small body expand into gigabytes: a chunk of
    /// more rows than the format's cap is refused, however well formed.
    #[test]
    fn a_chunk_of_more_rows_than_the_row_cap_is_refused() {
        let cap = ChunkCaps::DEFAULT.max_rows;
        let decoded = decode_column_chunk(&threes(cap), 1, cap).unwrap();
        assert_eq!(decoded.values.len(), 65_536, "the cap itself is read");
        assert_eq!(decoded.values[65_535], (65_535, Value::Int64(3)));
        let error = error_of(&threes(cap + 1), ChunkCodec::BitPacked, cap + 1);
        assert!(
            error.contains("byte 4") && error.contains("65537 rows") && error.contains("65536"),
            "{error}"
        );
        let values = [(cap, Value::Int64(3))];
        let error = encode_column_chunk(cap + 1, &values, None)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("65537 rows") && error.contains("65536"),
            "the writer refuses what the reader refuses: {error}"
        );
    }

    #[test]
    fn encode_refuses_rows_it_cannot_write() {
        let error = encode_column_chunk(3, &[], None).unwrap_err().to_string();
        assert!(error.contains("no value"), "{error}");
        let repeated = [(1, Value::Int64(3)), (1, Value::Int64(19))];
        assert!(
            encode_column_chunk(3, &repeated, None).is_err(),
            "a repeated row"
        );
        let descending = [(2, Value::Int64(3)), (1, Value::Int64(19))];
        assert!(
            encode_column_chunk(3, &descending, None).is_err(),
            "descending rows"
        );
        let past = [(3, Value::Int64(3))];
        let error = encode_column_chunk(3, &past, None).unwrap_err().to_string();
        assert!(error.contains("row 3"), "{error}");
        let one = [(0, Value::Int64(3))];
        assert!(
            encode_column_chunk(3, &one, Some(&[3, 19])).is_err(),
            "two epochs for one value"
        );
    }

    #[test]
    fn chunks_stay_within_their_bounds() {
        let distinct: Vec<Value> = (0..88).map(|i| Value::from(format!("{i:0>64}"))).collect();
        let cases: Vec<(u32, Vec<(u32, Value)>)> = vec![
            (1, vec![(0, Value::Bool(true))]),
            (65_536, vec![(65_535, Value::Bool(false))]),
            (1, vec![(0, Value::Int64(i64::MAX))]),
            (3, dense(vec![Value::Int64(-19), Value::Int64(i64::MIN)])),
            (
                19,
                vec![(3, Value::Float64(f64::NAN)), (18, Value::Float64(1.88))],
            ),
            (88, dense(distinct)),
            (1, vec![(0, Value::from("x".repeat(65)))]),
            (1, vec![(0, vector(&[3.0]))]),
            (2, dense(vec![vector(&[3.0, 19.0]), vector(&[88.0, 0.5])])),
            (
                u32::try_from(every_kind().len()).unwrap(),
                dense(every_kind()),
            ),
            (1, vec![(0, Value::Null)]),
        ];
        for (row_count, values) in cases {
            for epochs in [None, Some(vec![u64::MAX; values.len()])] {
                let (codec, bytes, _) = round_trip(row_count, &values, epochs.as_deref());
                let bound = chunk_overhead(row_count, values.len(), epochs.is_some())
                    + values.iter().map(|(_, v)| value_bound(v)).sum::<usize>();
                assert!(
                    bytes.len() <= bound,
                    "{codec:?} with epochs {}: {} bytes, bound {bound}",
                    epochs.is_some(),
                    bytes.len()
                );
            }
        }
        assert_eq!(value_bound(&Value::Int64(3)), 9 + 4);
        assert_eq!(
            chunk_overhead(65, 3, false),
            12 + 16 + (2 * (1 + 16) + 1 + 4 + 4) + 16
        );
        assert_eq!(
            chunk_overhead(64, 3, true),
            12 + 8 + (2 * (1 + 16) + 1 + 4 + 4) + 16 + 17 + 24
        );
    }

    #[test]
    fn the_layout_is_fixed() {
        let (codec, bytes) =
            encode_column_chunk(3, &[(0, Value::Int64(3)), (2, Value::Int64(19))], None).unwrap();
        assert_eq!(codec, ChunkCodec::BitPacked);
        #[rustfmt::skip]
        let expected: &[u8] = &[
            1, 3, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, // codec, flags, reserved, rows, values
            5, 0, 0, 0, 0, 0, 0, 0, // presence: rows 0 and 2
            2, 3, 0, 0, 0, 0, 0, 0, 0, // zone map minimum
            2, 19, 0, 0, 0, 0, 0, 0, 0, // zone map maximum
            5, 2, 0, 0, 0, 1, 0, 0, 0, 99, 2, 0, 0, 0, 0, 0, 0, // 5 bits, 2 values, 1 word
        ];
        assert_eq!(bytes, expected);
        let (codec, bytes) = encode_column_chunk(1, &[(0, Value::Null)], None).unwrap();
        assert_eq!(codec, ChunkCodec::Values);
        assert_eq!(bytes, [7, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]);
        let codecs = [
            ChunkCodec::BitPacked,
            ChunkCodec::Dict,
            ChunkCodec::Bitmap,
            ChunkCodec::Float64,
            ChunkCodec::RawI64,
            ChunkCodec::Float32Vector,
            ChunkCodec::Values,
        ];
        for (codec, byte) in codecs.into_iter().zip(1u8..) {
            assert_eq!(codec.to_byte(), byte);
            assert_eq!(ChunkCodec::from_byte(byte), Some(codec));
        }
        assert_eq!(ChunkCodec::from_byte(0), None);
        assert_eq!(ChunkCodec::from_byte(8), None);
    }

    #[test]
    fn the_same_values_give_the_same_bytes() {
        let cities = dense(
            [
                "Prague",
                "Amsterdam",
                "Prague",
                "Berlin",
                "Paris",
                "Amsterdam",
            ]
            .into_iter()
            .map(Value::from)
            .collect(),
        );
        assert_eq!(
            encode_column_chunk(6, &cities, None).unwrap(),
            encode_column_chunk(6, &cities, None).unwrap()
        );
        // Counters are hash maps: build them twice, from fresh maps filled in
        // opposite orders, so their iteration orders differ.
        let replicas = [
            "Alix", "Gus", "Vincent", "Mia", "Jules", "Butch", "Django", "Shosanna", "Hans",
            "Beatrix", "Harm", "Maxence",
        ];
        let counters = |order: &[&str]| {
            let counts = || {
                Arc::new(
                    order
                        .iter()
                        .map(|replica| {
                            let count = 19 * u64::try_from(replica.len()).unwrap();
                            ((*replica).to_string(), count)
                        })
                        .collect::<HashMap<_, _>>(),
                )
            };
            dense(vec![
                Value::GCounter(counts()),
                Value::OnCounter {
                    pos: counts(),
                    neg: counts(),
                },
            ])
        };
        let mut backward = replicas;
        backward.reverse();
        assert_eq!(
            encode_column_chunk(2, &counters(&replicas), None).unwrap(),
            encode_column_chunk(2, &counters(&backward), None).unwrap(),
            "counters filled in another order give the same bytes"
        );
    }
}
