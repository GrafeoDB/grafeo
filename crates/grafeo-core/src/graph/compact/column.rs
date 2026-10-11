//! The property columns of a 0.5.x compacted base, read for migration.
//!
//! A column is one [`ColumnCodec`] body in one of two layouts: v1 holds the
//! body whole ([`ColumnCodec::read_from`]); v3 cuts it into blocks behind a
//! block index, each entry with a zone map, which the reader steps past
//! ([`ColumnCodec::read_blocked`]). Every row has a value: 0.5.x stored a
//! missing property as the column's empty value (`0`, `""`, `false`, #542).
//! Removed in 0.7.0 with the other 0.5.x readers.

use std::sync::Arc;

use arcstr::ArcStr;
use bytes::Bytes;
use grafeo_common::types::Value;

use crate::codec::column_chunk::{
    read_bitmap_body, read_bitpacked_body, read_dict_body, read_f32_vector_body, read_le_words_body,
};
use crate::codec::{BitPackedInts, BitVector, DictionaryEncoding};

/// A column of a 0.5.x compact table, in the codec it was written with.
/// Fixed-width values stay in the section's bytes, little-endian.
#[derive(Debug, Clone)]
pub enum ColumnCodec {
    /// Bit-packed integers, read as `Int64`: the columns whose values are all
    /// at least 0.
    BitPacked(BitPackedInts),
    /// Dictionary-encoded strings.
    Dict(DictionaryEncoding),
    /// Booleans.
    Bitmap(BitVector),
    /// Int8 quantized vectors, `dimensions` bytes per row.
    Int8Vector {
        /// The rows, back to back.
        bytes: Bytes,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
    /// `f64` values.
    Float64(Bytes),
    /// `f32` vectors, `4 * dimensions` bytes per row.
    Float32Vector {
        /// The rows, back to back.
        bytes: Bytes,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
    /// `i64` values: the integer columns with a negative value.
    RawI64(Bytes),
}

impl ColumnCodec {
    /// The value at row `index`, or `None` past the last row.
    ///
    /// A bit-packed value over `i64::MAX` reads as `None`: 0.5.x bit-packed
    /// only columns of values from 0 to `i64::MAX`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value> {
        match self {
            Self::BitPacked(packed) => packed
                .get(index)
                .and_then(|value| i64::try_from(value).ok())
                .map(Value::Int64),
            Self::Dict(dictionary) => dictionary
                .get(index)
                .map(|text| Value::String(ArcStr::from(text))),
            Self::Bitmap(bits) => bits.get(index).map(Value::Bool),
            Self::Int8Vector { bytes, dimensions } => row(bytes, index, usize::from(*dimensions))
                .map(|row| {
                    Value::List(
                        row.iter()
                            .map(|&byte| Value::Int64(i64::from(byte.cast_signed())))
                            .collect(),
                    )
                }),
            Self::Float64(bytes) => row(bytes, index, 8)
                .and_then(|row| row.try_into().ok())
                .map(|row| Value::Float64(f64::from_le_bytes(row))),
            Self::Float32Vector { bytes, dimensions } => {
                row(bytes, index, 4 * usize::from(*dimensions)).map(|row| {
                    Value::Vector(
                        row.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|&component| f32::from_le_bytes(component))
                            .collect(),
                    )
                })
            }
            Self::RawI64(bytes) => row(bytes, index, 8)
                .and_then(|row| row.try_into().ok())
                .map(|row| Value::Int64(i64::from_le_bytes(row))),
        }
    }

    /// Reads a v1 column body at `pos` and moves `pos` past it. Fixed-width
    /// values are slices of `data`.
    ///
    /// # Errors
    ///
    /// Returns an error when the body is cut short, its discriminant is
    /// unknown or it holds what no writer produced: a bit width over 64, too
    /// few words for its values or bits, or a code past the dictionary.
    pub fn read_from(data: &Bytes, pos: &mut usize) -> Result<Self, &'static str> {
        let bytes = data.as_ref();
        let discriminant = *bytes.get(*pos).ok_or("truncated codec discriminant")?;
        *pos += 1;
        match discriminant {
            0 => {
                let packed = read_bitpacked_body(data, pos)?;
                check_packed(&packed)?;
                Ok(Self::BitPacked(packed))
            }
            1 => {
                let dictionary = read_dict_body(data, pos)?;
                check_codes(&dictionary.codes_bytes(), dictionary.dictionary_size())?;
                Ok(Self::Dict(dictionary))
            }
            2 => {
                let bits = read_bitmap_body(data, pos)?;
                if bits.word_count() < bits.len().div_ceil(64) {
                    return Err("Bitmap body has too few words for its bits");
                }
                Ok(Self::Bitmap(bits))
            }
            3 => {
                let dimensions = read_u16_le(bytes, pos)?;
                let len = read_u32_le(bytes, pos)? as usize;
                let end = pos
                    .checked_add(len)
                    .filter(|&end| end <= bytes.len())
                    .ok_or("truncated Int8Vector data")?;
                let rows = data.slice(*pos..end);
                *pos = end;
                Ok(Self::Int8Vector {
                    bytes: rows,
                    dimensions,
                })
            }
            4 => Ok(Self::Float64(read_le_words_body(data, pos)?)),
            5 => {
                let (dimensions, rows) = read_f32_vector_body(data, pos)?;
                Ok(Self::Float32Vector {
                    bytes: rows,
                    dimensions,
                })
            }
            6 => Ok(Self::RawI64(read_le_words_body(data, pos)?)),
            _ => Err("unknown codec discriminant"),
        }
    }

    /// Reads a v3 column body at `pos` and moves `pos` past it. The blocks
    /// are read as one column; dictionary codes and fixed-width values are
    /// slices of `data`.
    ///
    /// # Errors
    ///
    /// Returns an error when the body is cut short, its discriminant is
    /// unknown, its block index does not describe its block bodies back to
    /// back, or it holds what no writer produced: a bit width over 64, a
    /// count the bytes left cannot hold, or a code past the dictionary.
    pub fn read_blocked(data: &Bytes, pos: &mut usize) -> Result<Self, &'static str> {
        let bytes = data.as_ref();
        let discriminant = *bytes.get(*pos).ok_or("truncated codec discriminant")?;
        *pos += 1;
        let codec = match discriminant {
            0 => {
                let bits = *bytes.get(*pos).ok_or("truncated bits_per_value")?;
                *pos += 1;
                let blocks = read_block_index(bytes, pos)?;
                let bodies = bodies(data, pos, &blocks)?;
                Self::BitPacked(read_bitpacked_blocks(&bodies, bits, &blocks)?)
            }
            1 => {
                let dictionary = read_dictionary(bytes, pos)?;
                let blocks = read_block_index(bytes, pos)?;
                let codes = bodies(data, pos, &blocks)?;
                if !codes.len().is_multiple_of(4) {
                    return Err("Dict codes are not whole u32 values");
                }
                check_codes(&codes, dictionary.len())?;
                let count = codes.len() / 4;
                Self::Dict(DictionaryEncoding::from_bytes_storage(
                    dictionary, codes, count,
                ))
            }
            2 => {
                let blocks = read_block_index(bytes, pos)?;
                let bodies = bodies(data, pos, &blocks)?;
                Self::Bitmap(read_bitmap_blocks(&bodies, &blocks)?)
            }
            3 => {
                let dimensions = read_u16_le(bytes, pos)?;
                let blocks = read_block_index(bytes, pos)?;
                Self::Int8Vector {
                    bytes: bodies(data, pos, &blocks)?,
                    dimensions,
                }
            }
            4 => {
                let blocks = read_block_index(bytes, pos)?;
                Self::Float64(bodies(data, pos, &blocks)?)
            }
            5 => {
                let dimensions = read_u16_le(bytes, pos)?;
                let blocks = read_block_index(bytes, pos)?;
                Self::Float32Vector {
                    bytes: bodies(data, pos, &blocks)?,
                    dimensions,
                }
            }
            6 => {
                let blocks = read_block_index(bytes, pos)?;
                Self::RawI64(bodies(data, pos, &blocks)?)
            }
            _ => return Err("unknown codec discriminant"),
        };
        Ok(codec)
    }
}

/// The `width` bytes of row `index`, or `None` past the last row or for a
/// width of 0.
fn row(bytes: &[u8], index: usize, width: usize) -> Option<&[u8]> {
    if width == 0 {
        return None;
    }
    let start = index.checked_mul(width)?;
    bytes.get(start..start.checked_add(width)?)
}

// ── Blocks (v3) ─────────────────────────────────────────────────

/// An entry of a block index: where a block's body starts within the bodies
/// of its column, how long it is and how many rows it holds.
#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    byte_offset: u32,
    byte_len: u32,
    row_count: u32,
}

/// The fewest bytes a block index entry takes: offset, length and rows,
/// then a zone map's null and row counts and two absent bounds.
const BLOCK_META_MIN_BYTES: usize = 12 + 4 + 4 + 1 + 1;

/// Reads a block index at `pos`: the block count, then an entry per block,
/// each followed by a zone map, which this steps past. The count is checked
/// against the bytes left before anything is allocated for it.
fn read_block_index(bytes: &[u8], pos: &mut usize) -> Result<Vec<BlockMeta>, &'static str> {
    let count = read_u32_le(bytes, pos)? as usize;
    if count
        .checked_mul(BLOCK_META_MIN_BYTES)
        .is_none_or(|needed| needed > bytes.len().saturating_sub(*pos))
    {
        return Err("block count exceeds the bytes left");
    }
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        blocks.push(BlockMeta {
            byte_offset: read_u32_le(bytes, pos)?,
            byte_len: read_u32_le(bytes, pos)?,
            row_count: read_u32_le(bytes, pos)?,
        });
        super::zone_map::skip_inline(bytes, pos)?;
    }
    check_back_to_back(&blocks)?;
    Ok(blocks)
}

/// Refuses block bodies that do not follow each other from offset 0 (a gap
/// or an overlap would read the wrong bytes as a block), or that end past
/// the `u32` range their offsets are written in.
fn check_back_to_back(blocks: &[BlockMeta]) -> Result<(), &'static str> {
    let mut expected_offset: u64 = 0;
    for block in blocks {
        if u64::from(block.byte_offset) != expected_offset {
            return Err("non-contiguous block index (gap or overlap)");
        }
        expected_offset += u64::from(block.byte_len);
    }
    if expected_offset > u64::from(u32::MAX) {
        return Err("block bodies exceed u32 range");
    }
    Ok(())
}

/// The bodies of a column's blocks, at `pos`; moves `pos` past them.
fn bodies(data: &Bytes, pos: &mut usize, blocks: &[BlockMeta]) -> Result<Bytes, &'static str> {
    // check_back_to_back keeps the end within u32.
    let len = blocks
        .last()
        .map_or(0, |last| (last.byte_offset + last.byte_len) as usize);
    let end = pos
        .checked_add(len)
        .filter(|&end| end <= data.len())
        .ok_or("column block bodies out of bounds")?;
    let bodies = data.slice(*pos..end);
    *pos = end;
    Ok(bodies)
}

/// The body of `block` within the bodies of its column.
fn block_body<'a>(bodies: &'a [u8], block: &BlockMeta) -> Result<&'a [u8], &'static str> {
    // check_back_to_back keeps offset plus length within u32.
    let start = block.byte_offset as usize;
    bodies
        .get(start..start + block.byte_len as usize)
        .ok_or("column block body out of bounds")
}

/// Reads a block body's `u32` word count and its words, refusing a count the
/// body cannot hold before anything is allocated for it.
fn read_block_words(body: &[u8], too_long: &'static str) -> Result<Vec<u64>, &'static str> {
    let mut pos = 0;
    let count = read_u32_le(body, &mut pos)? as usize;
    let words = count
        .checked_mul(8)
        .and_then(|len| body.get(pos..pos + len))
        .ok_or(too_long)?;
    Ok(words
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&word| u64::from_le_bytes(word))
        .collect())
}

/// Reads the BitPacked blocks of a column into one packed column. A 0-bit
/// column holds only zeros and no words: its rows are counted, and nothing
/// is materialized.
fn read_bitpacked_blocks(
    bodies: &[u8],
    bits: u8,
    blocks: &[BlockMeta],
) -> Result<BitPackedInts, &'static str> {
    check_bits(bits)?;
    if bits == 0 {
        let rows = blocks
            .iter()
            .try_fold(0usize, |rows, block| {
                rows.checked_add(block.row_count as usize)
            })
            .ok_or("BitPacked row count overflow")?;
        return Ok(BitPackedInts::from_raw_parts(Vec::new(), 0, rows));
    }
    let mut values = Vec::new();
    for block in blocks {
        let words = read_block_words(
            block_body(bodies, block)?,
            "BitPacked block words exceed its body",
        )?;
        let packed = BitPackedInts::from_raw_parts(words, bits, block.row_count as usize);
        for row in 0..block.row_count as usize {
            values.push(
                packed
                    .get(row)
                    .ok_or("BitPacked block index out of range")?,
            );
        }
    }
    Ok(BitPackedInts::pack_with_bits(&values, bits))
}

/// Reads the Bitmap blocks of a column into one bit vector.
fn read_bitmap_blocks(bodies: &[u8], blocks: &[BlockMeta]) -> Result<BitVector, &'static str> {
    let mut values = Vec::new();
    for block in blocks {
        let words = read_block_words(
            block_body(bodies, block)?,
            "Bitmap block words exceed its body",
        )?;
        let bits = BitVector::from_raw_parts(words, block.row_count as usize);
        for row in 0..block.row_count as usize {
            values.push(bits.get(row).ok_or("Bitmap block index out of range")?);
        }
    }
    Ok(BitVector::from_bools(&values))
}

/// Reads a v3 dictionary at `pos`: its entry count, then each entry's
/// length and UTF-8 bytes. The count is checked against the bytes left (each
/// entry takes at least its 4-byte length) before anything is allocated for
/// it.
fn read_dictionary(bytes: &[u8], pos: &mut usize) -> Result<Arc<[Arc<str>]>, &'static str> {
    let count = read_u32_le(bytes, pos)? as usize;
    if count > bytes.len().saturating_sub(*pos) / 4 {
        return Err("Dict dictionary count exceeds the bytes left");
    }
    let mut entries: Vec<Arc<str>> = Vec::with_capacity(count);
    for _ in 0..count {
        let len = read_u32_le(bytes, pos)? as usize;
        let text = pos
            .checked_add(len)
            .and_then(|end| bytes.get(*pos..end))
            .ok_or("truncated dict string")?;
        entries.push(Arc::from(
            std::str::from_utf8(text).map_err(|_| "invalid UTF-8 in dict")?,
        ));
        *pos += len;
    }
    Ok(Arc::from(entries))
}

// ── Checks ──────────────────────────────────────────────────────

/// Refuses a BitPacked bit width over 64: reads would divide by zero.
fn check_bits(bits: u8) -> Result<(), &'static str> {
    if bits > 64 {
        return Err("BitPacked bits per value over 64");
    }
    Ok(())
}

/// Refuses a bit-packed body whose reads would divide by zero (a width over
/// 64 bits) or that has too few words for its values (they would read as
/// missing).
fn check_packed(packed: &BitPackedInts) -> Result<(), &'static str> {
    let bits = packed.bits_per_value();
    check_bits(bits)?;
    let words = if bits == 0 {
        0
    } else {
        packed.len().div_ceil(64 / usize::from(bits))
    };
    if packed.word_count() < words {
        return Err("BitPacked body has too few words for its values");
    }
    Ok(())
}

/// Refuses a code past the dictionary, which would read as a missing value.
fn check_codes(codes: &[u8], dictionary_len: usize) -> Result<(), &'static str> {
    let past = codes
        .as_chunks::<4>()
        .0
        .iter()
        .any(|&code| u32::from_le_bytes(code) as usize >= dictionary_len);
    if past {
        return Err("Dict code past the dictionary");
    }
    Ok(())
}

// ── Binary read helpers ─────────────────────────────────────────

fn read_u16_le(data: &[u8], pos: &mut usize) -> Result<u16, &'static str> {
    let bytes = data.get(*pos..*pos + 2).ok_or("truncated u16")?;
    *pos += 2;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    let bytes = data.get(*pos..*pos + 4).ok_or("truncated u32")?;
    *pos += 4;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_v1(bytes: &[u8]) -> Result<ColumnCodec, &'static str> {
        ColumnCodec::read_from(&Bytes::copy_from_slice(bytes), &mut 0)
    }

    // ── v1 bodies cut short or unknown ────────────────────────────

    #[test]
    fn test_read_from_truncated_discriminant() {
        assert_eq!(read_v1(&[]).unwrap_err(), "truncated codec discriminant");
    }

    #[test]
    fn test_read_from_unknown_discriminant() {
        // Discriminants 0 to 6 are codecs; 7 and up are not.
        for discriminant in [7u8, 42, 99] {
            assert_eq!(
                read_v1(&[discriminant]).unwrap_err(),
                "unknown codec discriminant"
            );
        }
    }

    #[test]
    fn test_read_from_truncated_bits_per_value() {
        // Discriminant 0 (BitPacked) with no following byte for bits_per_value.
        assert_eq!(read_v1(&[0]).unwrap_err(), "truncated bits_per_value");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_count() {
        // Discriminant + bits byte, but only 2 of the 4 bytes of the count.
        assert_eq!(read_v1(&[0, 4, 0, 0]).unwrap_err(), "truncated u32");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_words() {
        // Discriminant + bits + count=1 + data_len=2, but no u64 data words.
        let mut buf = vec![0u8, 4];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        assert_eq!(read_v1(&buf).unwrap_err(), "truncated BitPacked data");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_word() {
        // Discriminant + bits + count=1 + data_len=1, then only 3 bytes of
        // the 8-byte word.
        let mut buf = vec![0u8, 4];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8, 0, 0]);
        assert_eq!(read_v1(&buf).unwrap_err(), "truncated BitPacked data");
    }

    #[test]
    fn test_read_from_truncated_dict_string() {
        // Discriminant=1 (Dict), dict_len=1, slen=5, but only 3 bytes of string.
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&5u32.to_le_bytes());
        buf.extend_from_slice(b"abc");
        assert_eq!(read_v1(&buf).unwrap_err(), "truncated dict string");
    }

    #[test]
    fn test_read_from_invalid_utf8_in_dict() {
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&[0xFF, 0xFE]);
        assert_eq!(read_v1(&buf).unwrap_err(), "invalid UTF-8 in dict");
    }

    #[test]
    fn test_read_from_truncated_bitmap_words() {
        // Discriminant=2 (Bitmap), bit_len=64, data_len=1, but no u64 data.
        let mut buf = vec![2u8];
        buf.extend_from_slice(&64u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        assert_eq!(read_v1(&buf).unwrap_err(), "truncated Bitmap data");
    }

    #[test]
    fn test_read_from_truncated_int8_vector_dimensions() {
        // Discriminant=3 (Int8Vector), only 1 byte of the 2-byte dimensions.
        assert_eq!(read_v1(&[3, 0]).unwrap_err(), "truncated u16");
    }

    #[test]
    fn test_read_from_truncated_int8_vector_data() {
        // Discriminant=3, dimensions=2, data_len=4, but only 2 data bytes.
        let mut buf = vec![3u8];
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&[10u8, 20]);
        assert_eq!(read_v1(&buf).unwrap_err(), "truncated Int8Vector data");
    }

    // ── Values ─────────────────────────────────────────────────────

    /// Fixed-width columns read their rows from the section's bytes, and a
    /// row past the last, or a cut-short last row, reads as `None`.
    #[test]
    fn fixed_width_columns_read_their_rows() {
        let mut floats = Vec::new();
        for value in [1.88f64, -0.0] {
            floats.extend_from_slice(&value.to_le_bytes());
        }
        let column = ColumnCodec::Float64(Bytes::from(floats));
        assert_eq!(column.get(0), Some(Value::Float64(1.88)));
        assert_eq!(column.get(2), None);

        let mut integers = Vec::new();
        for value in [-19i64, 88] {
            integers.extend_from_slice(&value.to_le_bytes());
        }
        integers.extend_from_slice(&[1, 2, 3]);
        let column = ColumnCodec::RawI64(Bytes::from(integers));
        assert_eq!(column.get(0), Some(Value::Int64(-19)));
        assert_eq!(column.get(1), Some(Value::Int64(88)));
        assert_eq!(column.get(2), None, "a row cut short");

        let column = ColumnCodec::Int8Vector {
            bytes: Bytes::from(vec![3u8, 0xED, 88, 0]),
            dimensions: 2,
        };
        assert_eq!(
            column.get(0),
            Some(Value::List(Arc::from([Value::Int64(3), Value::Int64(-19)])))
        );
        assert_eq!(column.get(2), None);

        let mut components = Vec::new();
        for value in [3.0f32, -19.5, 88.0, 0.25] {
            components.extend_from_slice(&value.to_le_bytes());
        }
        let column = ColumnCodec::Float32Vector {
            bytes: Bytes::from(components),
            dimensions: 2,
        };
        assert_eq!(
            column.get(1),
            Some(Value::Vector(Arc::from([88.0f32, 0.25])))
        );
        assert_eq!(column.get(2), None);

        let column = ColumnCodec::Int8Vector {
            bytes: Bytes::from(vec![3u8]),
            dimensions: 0,
        };
        assert_eq!(column.get(0), None, "no dimensions, no rows");
    }

    /// A bit-packed value past `i64::MAX`, which no 0.5.x build wrote, reads
    /// as missing rather than as a negative number.
    #[test]
    fn a_bit_packed_value_past_i64_max_reads_as_missing() {
        let column = ColumnCodec::BitPacked(BitPackedInts::pack(&[88, u64::MAX]));
        assert_eq!(column.get(0), Some(Value::Int64(88)));
        assert_eq!(column.get(1), None);
    }

    // ── Crafted bodies ──────────────────────────────────────────────

    /// Bytes built from little-endian fields.
    #[derive(Default)]
    struct Body(Vec<u8>);

    impl Body {
        fn byte(mut self, value: u8) -> Self {
            self.0.push(value);
            self
        }

        fn u32(mut self, value: u32) -> Self {
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }

        fn u64(mut self, value: u64) -> Self {
            self.0.extend_from_slice(&value.to_le_bytes());
            self
        }

        fn text(mut self, value: &str) -> Self {
            self = self.u32(u32::try_from(value.len()).unwrap());
            self.0.extend_from_slice(value.as_bytes());
            self
        }

        /// A block index entry, with a zone map without bounds.
        fn block(self, offset: u32, len: u32, rows: u32) -> Self {
            self.u32(offset)
                .u32(len)
                .u32(rows)
                .u32(0)
                .u32(rows)
                .byte(0)
                .byte(0)
        }

        fn read_v1(self) -> Result<ColumnCodec, &'static str> {
            ColumnCodec::read_from(&Bytes::from(self.0), &mut 0)
        }

        fn read_blocked(self) -> Result<ColumnCodec, &'static str> {
            ColumnCodec::read_blocked(&Bytes::from(self.0), &mut 0)
        }
    }

    /// A v1 BitPacked body of `count` values of `bits` bits in `words` words.
    fn bitpacked_v1(bits: u8, count: u32, words: u32) -> Body {
        let mut body = Body::default().byte(0).byte(bits).u32(count).u32(words);
        for _ in 0..words {
            body = body.u64(0);
        }
        body
    }

    /// v1 bodies no writer produces are refused: a bit width over 64 (whose
    /// reads would divide by zero), too few words for the values, a
    /// dictionary count the bytes left cannot hold (which would ask for
    /// gigabytes), a code past the dictionary (which would read as missing)
    /// and a bitmap with too few words.
    #[test]
    fn crafted_v1_column_bodies_are_refused() {
        let nine = bitpacked_v1(8, 9, 2).read_v1().unwrap();
        assert_eq!((nine.get(8), nine.get(9)), (Some(Value::Int64(0)), None));
        let dictionary = Body::default()
            .byte(1)
            .u32(1)
            .text("Alix")
            .u32(2)
            .u32(0)
            .u32(0);
        assert_eq!(
            dictionary.read_v1().unwrap().get(1),
            Some(Value::from("Alix"))
        );

        for (case, body, expected) in [
            ("a bit width over 64", bitpacked_v1(65, 1, 2), "bits"),
            ("too few words", bitpacked_v1(8, 9, 1), "words"),
            (
                // The shared body reader sizes nothing from the count: the
                // first missing entry ends the read.
                "a dictionary count",
                Body::default().byte(1).u32(u32::MAX),
                "truncated",
            ),
            (
                "a code past the dictionary",
                Body::default()
                    .byte(1)
                    .u32(1)
                    .text("Alix")
                    .u32(2)
                    .u32(0)
                    .u32(3),
                "past the dictionary",
            ),
            (
                "a bitmap with too few words",
                Body::default().byte(2).u32(65).u32(1).u64(19),
                "words",
            ),
        ] {
            let error = body.read_v1().unwrap_err();
            assert!(error.contains(expected), "{case}: {error}");
        }
    }

    /// v3 bodies no writer produces are refused the same way, and a count
    /// read from the file is checked against the bytes left before anything
    /// is allocated for it.
    #[test]
    fn crafted_v3_column_bodies_are_refused() {
        let valid = Body::default()
            .byte(1)
            .u32(1)
            .text("Gus")
            .u32(1)
            .block(0, 8, 2)
            .u32(0)
            .u32(0);
        assert_eq!(
            valid.read_blocked().unwrap().get(1),
            Some(Value::from("Gus"))
        );

        for (case, body, expected) in [
            (
                "a bit width over 64",
                Body::default()
                    .byte(0)
                    .byte(65)
                    .u32(1)
                    .block(0, 12, 1)
                    .u32(1)
                    .u64(3),
                "bits",
            ),
            (
                "a word count",
                Body::default()
                    .byte(0)
                    .byte(8)
                    .u32(1)
                    .block(0, 4, 1)
                    .u32(u32::MAX),
                "words",
            ),
            (
                "a dictionary count",
                Body::default().byte(1).u32(u32::MAX),
                "dictionary",
            ),
            (
                "a code past the dictionary",
                Body::default()
                    .byte(1)
                    .u32(1)
                    .text("Gus")
                    .u32(1)
                    .block(0, 8, 2)
                    .u32(0)
                    .u32(7),
                "past the dictionary",
            ),
            (
                "codes cut short",
                Body::default()
                    .byte(1)
                    .u32(1)
                    .text("Gus")
                    .u32(1)
                    .block(0, 6, 1)
                    .u32(0)
                    .byte(0)
                    .byte(0),
                "codes",
            ),
            (
                "a bitmap word count",
                Body::default().byte(2).u32(1).block(0, 4, 1).u32(u32::MAX),
                "words",
            ),
            (
                "a block count",
                Body::default().byte(6).u32(u32::MAX),
                "block",
            ),
            (
                "a gap between blocks",
                Body::default()
                    .byte(6)
                    .u32(2)
                    .block(0, 8, 1)
                    .block(9, 8, 1)
                    .u64(19)
                    .byte(0)
                    .u64(88),
                "non-contiguous",
            ),
        ] {
            let error = body.read_blocked().unwrap_err();
            assert!(error.contains(expected), "{case}: {error}");
        }
    }

    /// The blocks of a column read as one column, in block order, and the
    /// reader stops after the last body.
    #[test]
    fn blocks_read_as_one_column() {
        let mut body = Body::default()
            .byte(6)
            .u32(2)
            .block(0, 16, 2)
            .block(16, 8, 1);
        for value in [-19i64, 88, 3] {
            body = body.u64(value.cast_unsigned());
        }
        let whole = body.byte(0xAB).0;
        let mut pos = 0;
        let column = ColumnCodec::read_blocked(&Bytes::from(whole.clone()), &mut pos).unwrap();
        let values: Vec<_> = (0..4).map(|row| column.get(row)).collect();
        assert_eq!(
            values,
            [
                Some(Value::Int64(-19)),
                Some(Value::Int64(88)),
                Some(Value::Int64(3)),
                None
            ]
        );
        assert_eq!(pos, whole.len() - 1, "the byte after is left");
    }

    /// A 0-bit column holds only zeros and no words: it is read without
    /// materializing its values, so a block may claim 4 billion rows.
    #[test]
    fn a_zero_bit_column_is_read_without_materializing_its_values() {
        let column = Body::default()
            .byte(0)
            .byte(0)
            .u32(1)
            .block(0, 4, u32::MAX)
            .u32(0)
            .read_blocked()
            .unwrap();
        let last = usize::try_from(u32::MAX).unwrap() - 1;
        assert_eq!(column.get(last), Some(Value::Int64(0)));
        assert_eq!(column.get(last + 1), None);
        assert_eq!(column.get(88), Some(Value::Int64(0)));
    }

    /// A 0-bit column whose block bodies end past the bytes is refused like
    /// any other width, before the reader moves past them: it must not read
    /// as a column and leave the position beyond the input.
    #[test]
    fn a_truncated_zero_bit_column_is_refused() {
        let whole = Body::default()
            .byte(0)
            .byte(0)
            .u32(1)
            .block(0, 4, 3)
            .u32(0)
            .0;
        let mut pos = 0;
        let column = ColumnCodec::read_blocked(&Bytes::from(whole.clone()), &mut pos).unwrap();
        assert_eq!(
            (column.get(2), column.get(3)),
            (Some(Value::Int64(0)), None),
            "the whole column"
        );
        assert_eq!(pos, whole.len(), "read to its last byte");
        for cut in 1..=4 {
            let data = Bytes::from(whole[..whole.len() - cut].to_vec());
            let mut pos = 0;
            let error = ColumnCodec::read_blocked(&data, &mut pos)
                .err()
                .unwrap_or_else(|| {
                    panic!("{cut} bytes cut: read, position {pos} of {}", data.len())
                });
            assert!(error.contains("out of bounds"), "{cut} cut: {error}");
        }
    }
}
