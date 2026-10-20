//! Point reads over fixed-size bodies of already validated column chunks.

use bytes::Bytes;
use grafeo_common::types::{Decimal, Value};
use grafeo_common::utils::error::Result;

use super::{
    BitPackedInts, BitVector, ChunkCodec, FLAG_EPOCHS, FLAG_ZONE_MAP, HEADER_LEN, check_bitpacked,
    checked_header, corrupt, presence_words, read_bitmap_body, read_bitpacked_body,
    read_le_words_body, skip_zone_map, take,
};

#[derive(Debug)]
enum Body {
    Packed(BitPackedInts),
    Bitmap(BitVector),
    Float64(Bytes),
    Int64(Bytes),
    Decimal64 { words: Bytes, scale: u8 },
}

#[derive(Debug)]
struct Presence {
    words: Bytes,
    /// Number of present rows before each 64-row word, without a sentinel.
    before: Box<[u32]>,
}

/// A shared encoded body and, for sparse chunks, a small row-rank index.
///
/// This is not a replacement for the full column-chunk validator.
#[derive(Debug)]
pub(crate) struct PointView {
    row_count: u32,
    presence: Option<Presence>,
    body: Body,
}

impl PointView {
    /// Builds a view after `decode_column_chunk_bytes` has successfully
    /// validated these exact bytes against this codec and row count.
    ///
    /// The full decode remains mandatory: it checks value ranges, zone-map
    /// agreement and epoch contents. This constructor only reads the layout
    /// needed by point access and checks its structural bounds. Unsupported
    /// codecs return `None` and keep the caller's existing decode path.
    ///
    /// # Errors
    ///
    /// Returns corruption when the supplied layout contradicts its header
    /// or does not fit the bytes. This does not perform full validation.
    pub(crate) fn from_validated(data: &Bytes, codec: u8, row_count: u32) -> Result<Option<Self>> {
        let header = checked_header(data, codec, row_count)?;
        if matches!(
            header.codec,
            ChunkCodec::Dict | ChunkCodec::Values | ChunkCodec::Float32Vector
        ) {
            return Ok(None);
        }
        let mut pos = HEADER_LEN;
        let presence = if header.sparse {
            let at = pos;
            let words = take(data, &mut pos, presence_words(row_count) * 8)
                .ok_or_else(|| corrupt(at, "truncated presence bitmap"))?;
            let mut before = Vec::with_capacity(presence_words(row_count));
            let mut marked = 0u32;
            for word in words.as_chunks::<8>().0 {
                before.push(marked);
                marked += u64::from_le_bytes(*word).count_ones();
            }
            let tail = row_count % 64;
            if tail != 0
                && words
                    .as_chunks::<8>()
                    .0
                    .last()
                    .is_some_and(|last| u64::from_le_bytes(*last) >> tail != 0)
            {
                return Err(corrupt(at, "presence bitmap marks a row past the chunk"));
            }
            if marked != header.value_count {
                return Err(corrupt(at, "presence bitmap disagrees with value count"));
            }
            Some(Presence {
                words,
                before: before.into_boxed_slice(),
            })
        } else {
            None
        };
        if header.flags & FLAG_ZONE_MAP != 0 {
            let at = pos;
            skip_zone_map(header.codec, data, &mut pos).map_err(|error| corrupt(at, error))?;
        }
        let count = header.value_count as usize;
        if header.flags & FLAG_EPOCHS != 0 {
            let at = pos;
            let epochs = read_bitpacked_body(data, &mut pos).map_err(|error| corrupt(at, error))?;
            check_bitpacked(&epochs, count).map_err(|error| corrupt(at, error))?;
        }
        let at = pos;
        let body = match header.codec {
            ChunkCodec::BitPacked => {
                let packed =
                    read_bitpacked_body(data, &mut pos).map_err(|error| corrupt(at, error))?;
                check_bitpacked(&packed, count).map_err(|error| corrupt(at, error))?;
                Body::Packed(packed)
            }
            ChunkCodec::Bitmap => {
                let bits = read_bitmap_body(data, &mut pos).map_err(|error| corrupt(at, error))?;
                if bits.len() != count || bits.word_count() != count.div_ceil(64) {
                    return Err(corrupt(at, "bitmap body disagrees with value count"));
                }
                Body::Bitmap(bits)
            }
            ChunkCodec::Float64 | ChunkCodec::RawI64 | ChunkCodec::Decimal64 => {
                let scale = if header.codec == ChunkCodec::Decimal64 {
                    let scale = *data
                        .get(pos)
                        .ok_or_else(|| corrupt(pos, "truncated decimal scale"))?;
                    pos += 1;
                    Decimal::new(0, scale).map_err(|error| corrupt(at, error))?;
                    scale
                } else {
                    0
                };
                let words =
                    read_le_words_body(data, &mut pos).map_err(|error| corrupt(at, error))?;
                if words.len() / 8 != count {
                    return Err(corrupt(at, "fixed-size body disagrees with value count"));
                }
                match header.codec {
                    ChunkCodec::Float64 => Body::Float64(words),
                    ChunkCodec::RawI64 => Body::Int64(words),
                    _ => Body::Decimal64 { words, scale },
                }
            }
            ChunkCodec::Dict | ChunkCodec::Values | ChunkCodec::Float32Vector => return Ok(None),
        };
        if pos != data.len() {
            return Err(corrupt(pos, "bytes after the end of the values"));
        }
        Ok(Some(Self {
            row_count,
            presence,
            body,
        }))
    }

    /// Reads a row offset; absent rows and offsets outside the chunk return `None`.
    pub(crate) fn get(&self, row: u32) -> Option<Value> {
        if row >= self.row_count {
            return None;
        }
        let index = match &self.presence {
            None => row as usize,
            Some(presence) => {
                let word_index = row as usize / 64;
                let word = u64::from_le_bytes(*presence.words.as_chunks::<8>().0.get(word_index)?);
                let bit = 1u64 << (row % 64);
                if word & bit == 0 {
                    return None;
                }
                (presence.before.get(word_index)? + (word & (bit - 1)).count_ones()) as usize
            }
        };
        let word = |bytes: &Bytes| bytes.as_chunks::<8>().0.get(index).copied();
        match &self.body {
            Body::Packed(packed) => i64::try_from(packed.get(index)?).ok().map(Value::Int64),
            Body::Bitmap(bits) => bits.get(index).map(Value::Bool),
            Body::Float64(bytes) => Some(Value::Float64(f64::from_le_bytes(word(bytes)?))),
            Body::Int64(bytes) => Some(Value::Int64(i64::from_le_bytes(word(bytes)?))),
            Body::Decimal64 { words, scale } => {
                Decimal::new(i128::from(i64::from_le_bytes(word(words)?)), *scale)
                    .ok()
                    .map(Value::Decimal)
            }
        }
    }

    /// Owned rank-index bytes, excluding shared encoded backing and inline fields.
    /// Dense chunks allocate no index; the largest sparse index is 4 KiB.
    pub(crate) fn index_heap_bytes(&self) -> usize {
        self.presence
            .as_ref()
            .map_or(0, |presence| std::mem::size_of_val(&*presence.before))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::column_chunk::{decode_column_chunk_bytes, encode_column_chunk};

    fn parity(rows: u32, values: &[(u32, Value)], epochs: Option<&[u64]>) -> PointView {
        let (codec, encoded) = encode_column_chunk(rows, values, epochs).unwrap();
        let data = Bytes::from(encoded);
        let decoded = decode_column_chunk_bytes(&data, codec.to_byte(), rows).unwrap();
        let view = PointView::from_validated(&data, codec.to_byte(), rows)
            .unwrap()
            .expect("a supported fixed-size codec");
        // The view owns shared backing even after the input handle is dropped.
        drop(data);
        let mut expected = decoded.values.iter().peekable();
        for row in 0..rows {
            let want = if expected.peek().is_some_and(|(offset, _)| *offset == row) {
                expected.next().map(|(_, value)| value.clone())
            } else {
                None
            };
            match (view.get(row), want) {
                (Some(Value::Float64(actual)), Some(Value::Float64(expected))) => {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "row {row}");
                }
                (Some(Value::Decimal(actual)), Some(Value::Decimal(expected))) => {
                    assert_eq!(actual.mantissa(), expected.mantissa(), "row {row}");
                    assert_eq!(actual.scale(), expected.scale(), "row {row}");
                }
                (actual, expected) => assert_eq!(actual, expected, "row {row}"),
            }
        }
        assert_eq!(view.get(rows), None);
        assert_eq!(view.get(u32::MAX), None);
        view
    }

    #[test]
    fn point_sparse_word_boundaries_and_epoch_offsets() {
        let values: Vec<_> = [0, 1, 62, 63, 64, 65, 127, 128, 65_534, 65_535]
            .into_iter()
            .map(|row| (row, Value::Int64(i64::from(row))))
            .collect();
        let epochs: Vec<_> = (0..values.len()).map(|i| i as u64 + 1).collect();
        let view = parity(65_536, &values, Some(&epochs));
        assert_eq!(view.index_heap_bytes(), 4_096);
    }

    #[test]
    fn point_fixed_codecs_match_full_decoder() {
        let cases = [
            vec![Value::Int64(0), Value::Int64(i64::MAX)],
            vec![Value::Int64(i64::MIN), Value::Int64(-1)],
            vec![Value::Bool(false), Value::Bool(true)],
            vec![Value::Float64(-0.0), Value::Float64(0.0)],
            vec![
                Value::Float64(f64::from_bits(0x7ff8_0000_0000_0123)),
                Value::Float64(f64::NEG_INFINITY),
            ],
            vec![
                Value::Decimal(Decimal::new(i128::from(i64::MIN), 17).unwrap()),
                Value::Decimal(Decimal::new(i128::from(i64::MAX), 17).unwrap()),
            ],
        ];
        for values in cases {
            let dense = [(0, values[0].clone()), (1, values[1].clone())];
            assert_eq!(parity(2, &dense, None).index_heap_bytes(), 0);
            let sparse = [(63, values[0].clone()), (65, values[1].clone())];
            assert_eq!(parity(67, &sparse, Some(&[0, 128])).index_heap_bytes(), 8);
        }
        assert_eq!(
            parity(1, &[(0, Value::Int64(0))], None).index_heap_bytes(),
            0
        );
    }

    #[test]
    fn point_random_sparse_rows_match_full_decoder() {
        let mut seed = 17u64;
        let mut values = Vec::new();
        for row in 0..1_025 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            if seed & 3 != 0 {
                values.push((row, Value::Int64(i64::try_from(seed >> 1).unwrap())));
            }
        }
        parity(1_025, &values, None);
    }

    #[test]
    fn point_complex_codecs_keep_the_decode_fallback() {
        for value in [
            Value::from("dictionary"),
            Value::Null,
            Value::Vector(vec![1.0f32, 2.0].into()),
        ] {
            let (codec, encoded) = encode_column_chunk(1, &[(0, value)], None).unwrap();
            let data = Bytes::from(encoded);
            decode_column_chunk_bytes(&data, codec.to_byte(), 1).unwrap();
            assert!(
                PointView::from_validated(&data, codec.to_byte(), 1)
                    .unwrap()
                    .is_none()
            );
        }
    }
}
