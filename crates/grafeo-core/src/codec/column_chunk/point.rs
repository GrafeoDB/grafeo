//! Point reads over fixed-size bodies of already validated column chunks:
//! the integer, boolean, float and decimal bodies, and the vector body, whose
//! vectors in one chunk all have the same number of components.

use std::sync::Arc;

use bytes::Bytes;
use grafeo_common::types::{Decimal, Value};
use grafeo_common::utils::error::Result;

use super::{
    BitPackedInts, BitVector, ChunkCodec, ChunkHeader, FLAG_EPOCHS, FLAG_ZONE_MAP, HEADER_LEN,
    check_bitpacked, checked_header, corrupt, presence_words, read_bitmap_body,
    read_bitpacked_body, read_f32_vector_body, read_le_words_body, read_u32, read_zone_map_body,
    take,
};

#[derive(Debug)]
enum Body {
    Packed(BitPackedInts),
    Bitmap(BitVector),
    Float64(Bytes),
    Int64(Bytes),
    Decimal64 {
        words: Bytes,
        scale: u8,
    },
    /// The components of every vector, `dimensions` little-endian `f32`s
    /// each, the vectors in row order.
    Vector {
        components: Bytes,
        dimensions: usize,
    },
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

/// The bits of `rows`, at most 64, whose key is among `targets`, sorted.
pub(crate) fn bits_among<T: Copy, K: Copy + Ord>(
    rows: &[T],
    targets: &[K],
    key: impl Fn(T) -> K,
) -> u64 {
    fn bits_where<T: Copy>(rows: &[T], test: impl Fn(T) -> bool) -> u64 {
        let mut bits = 0;
        for (bit, value) in rows.iter().enumerate() {
            bits |= u64::from(test(*value)) << bit;
        }
        bits
    }
    match targets {
        [] => 0,
        [target] => bits_where(rows, |value| key(value) == *target),
        _ => bits_where(rows, |value| targets.binary_search(&key(value)).is_ok()),
    }
}

/// `bits`, a bit per value in order, as a bit per row of the rows `present` names.
fn at_present_rows(mut bits: u64, present: u64) -> u64 {
    let mut rows = 0;
    let mut left = present;
    while bits != 0 && left != 0 {
        let row = left.isolate_lowest_one();
        if bits & 1 != 0 {
            rows |= row;
        }
        bits >>= 1;
        left &= left - 1;
    }
    rows
}

/// Reads the presence bitmap at `*pos`, moving `*pos` past it, and past the zone map and epochs
/// after it, to the body.
fn presence_up_to_body(
    data: &Bytes,
    header: &ChunkHeader,
    row_count: u32,
    pos: &mut usize,
) -> Result<Option<Presence>> {
    let presence = if header.sparse {
        let at = *pos;
        let words = take(data, pos, presence_words(row_count) * 8)
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
        let at = *pos;
        read_zone_map_body(header.codec, data, pos).map_err(|error| corrupt(at, error))?;
    }
    if header.flags & FLAG_EPOCHS != 0 {
        let at = *pos;
        let epochs = read_bitpacked_body(data, pos).map_err(|error| corrupt(at, error))?;
        check_bitpacked(&epochs, header.value_count as usize)
            .map_err(|error| corrupt(at, error))?;
    }
    Ok(presence)
}

/// The rows of word `word` of a chunk of `row_count` rows that have a value.
fn present_in_word(presence: Option<&Presence>, row_count: u32, word: usize) -> u64 {
    match presence {
        Some(presence) => presence
            .words
            .as_chunks::<8>()
            .0
            .get(word)
            .map_or(0, |bits| u64::from_le_bytes(*bits)),
        None => {
            let rows = (row_count as usize).saturating_sub(word * 64);
            if rows >= 64 {
                u64::MAX
            } else {
                (1 << rows) - 1
            }
        }
    }
}

/// Calls `found` with the offset of each row `scope` names (by the word of rows) whose value's key
/// is among `targets`, comparing only words with such a row; returns how many rows it compared.
fn find_values<T: Copy, K: Copy + Ord>(
    presence: Option<&Presence>,
    row_count: u32,
    values: &[T],
    targets: &[K],
    key: impl Fn(T) -> K,
    scope: impl Fn(usize) -> u64,
    mut found: impl FnMut(u32),
) -> usize {
    let mut compared = 0;
    let mut at = 0;
    for (word, first) in (0..presence_words(row_count)).zip((0_u32..).step_by(64)) {
        let present = present_in_word(presence, row_count, word);
        let count = present.count_ones() as usize;
        let asked = scope(word) & present;
        if asked != 0 {
            compared += asked.count_ones() as usize;
            let rows = values.get(at..at + count).unwrap_or_default();
            let mut hits = bits_among(rows, targets, &key);
            if presence.is_some() {
                hits = at_present_rows(hits, present);
            }
            hits &= asked;
            while hits != 0 {
                let bit = hits.trailing_zeros();
                hits &= hits - 1;
                found(first + bit);
            }
        }
        at += count;
    }
    compared
}

impl PointView {
    /// Whether a chunk of `codec` gets a view: its values are of one fixed
    /// size, so a row's value is found without reading the others.
    pub(crate) fn supports(codec: ChunkCodec) -> bool {
        !matches!(codec, ChunkCodec::Dict | ChunkCodec::Values)
    }

    /// Builds a view after `decode_column_chunk_bytes` has successfully
    /// validated these exact bytes against this codec and row count (or
    /// over bytes equal to what `encode_column_chunk` wrote, which decode).
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
        if !Self::supports(header.codec) {
            return Ok(None);
        }
        let mut pos = HEADER_LEN;
        let presence = presence_up_to_body(data, &header, row_count, &mut pos)?;
        let count = header.value_count as usize;
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
            ChunkCodec::Float32Vector => {
                let (dimensions, components) =
                    read_f32_vector_body(data, &mut pos).map_err(|error| corrupt(at, error))?;
                let dimensions = usize::from(dimensions);
                if dimensions == 0
                    || count
                        .checked_mul(dimensions)
                        .and_then(|components| components.checked_mul(4))
                        != Some(components.len())
                {
                    return Err(corrupt(at, "vector body disagrees with value count"));
                }
                Body::Vector {
                    components,
                    dimensions,
                }
            }
            ChunkCodec::Dict | ChunkCodec::Values => return Ok(None),
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

    /// Whether the row at offset `row` has a value, read from the presence
    /// bits alone; offsets outside the chunk have none.
    pub(crate) fn has(&self, row: u32) -> bool {
        self.index_of(row).is_some()
    }

    /// The index among the values of the row at offset `row`, when it has
    /// one.
    fn index_of(&self, row: u32) -> Option<usize> {
        if row >= self.row_count {
            return None;
        }
        Some(match &self.presence {
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
        })
    }

    /// How many rows compared in place, or decoded for a body with no such compare, cost what one
    /// read of [`get`](Self::get) does, as measured: a read of fewer rows than the chunk's over
    /// this goes through `get`. The other bodies have no measure of their own.
    pub(crate) fn rows_per_point_read(&self) -> usize {
        match (&self.body, &self.presence) {
            (Body::Packed(_) | Body::Int64(_), None) => 6,
            (Body::Packed(_) | Body::Int64(_), Some(_)) => 10,
            (Body::Float64(_), _) => 20,
            _ => 16,
        }
    }

    /// Reads a row offset; absent rows and offsets outside the chunk return `None`.
    #[inline]
    pub(crate) fn get(&self, row: u32) -> Option<Value> {
        let index = self.index_of(row)?;
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
            Body::Vector {
                components,
                dimensions,
            } => {
                let width = dimensions * 4;
                let start = index.checked_mul(width)?;
                let vector: Arc<[f32]> = components
                    .get(start..start.checked_add(width)?)?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|component| f32::from_le_bytes(*component))
                    .collect();
                Some(Value::Vector(vector))
            }
        }
    }

    /// Finds the rows of `scope` (by the word of rows) holding one of `integers` or of the bits
    /// `floats`, both sorted, with no `Value` built, calling `found` with each one's offset;
    /// returns how many rows it compared, or `None` for a body of another kind.
    pub(crate) fn find_numbers(
        &self,
        integers: &[i64],
        floats: &[u64],
        scope: impl Fn(usize) -> u64,
        found: impl FnMut(u32),
    ) -> Option<usize> {
        let presence = self.presence.as_ref();
        let rows = self.row_count;
        match &self.body {
            Body::Int64(words) => Some(find_values(
                presence,
                rows,
                words.as_chunks::<8>().0,
                integers,
                i64::from_le_bytes,
                scope,
                found,
            )),
            Body::Float64(words) => Some(find_values(
                presence,
                rows,
                words.as_chunks::<8>().0,
                floats,
                u64::from_le_bytes,
                scope,
                found,
            )),
            Body::Packed(packed) => {
                // A packed value is not negative, nor past the largest `Int64` (the load's check).
                let targets: Vec<u64> = integers
                    .iter()
                    .filter_map(|target| u64::try_from(*target).ok())
                    .collect();
                let scoped = (0..presence_words(rows)).any(|word| scope(word) != 0);
                if targets.is_empty() || !scoped {
                    return Some(0);
                }
                Some(find_values(
                    presence,
                    rows,
                    &packed.unpack(),
                    &targets,
                    |value| value,
                    scope,
                    found,
                ))
            }
            _ => None,
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

/// A dictionary chunk, already validated, read in place: its entries are asked once, with no string
/// built, and its codes compared with those of the entries found.
#[derive(Debug)]
pub(crate) struct DictView {
    row_count: u32,
    presence: Option<Presence>,
    /// The codes, a little-endian u32 per value.
    codes: Bytes,
    /// The codes of the entries found, ascending.
    found: Vec<u32>,
}

impl DictView {
    /// The view of a chunk `decode_column_chunk_bytes` has validated, asking `finds` of each of its
    /// entries, by its UTF-8 bytes; `None` for another codec.
    ///
    /// # Errors
    ///
    /// Returns corruption when the layout contradicts the header or does not fit the bytes.
    pub(crate) fn read_validated(
        data: &Bytes,
        codec: u8,
        row_count: u32,
        mut finds: impl FnMut(&[u8]) -> bool,
    ) -> Result<Option<Self>> {
        let header = checked_header(data, codec, row_count)?;
        if header.codec != ChunkCodec::Dict {
            return Ok(None);
        }
        let mut pos = HEADER_LEN;
        let presence = presence_up_to_body(data, &header, row_count, &mut pos)?;
        let at = pos;
        let bad = |error: &'static str| corrupt(at, error);
        let count = read_u32(data, &mut pos).map_err(bad)?;
        let mut found = Vec::new();
        // A walk over every entry to reach the codes, so a test per entry is what it costs.
        let mut rest = data.get(pos..).unwrap_or_default();
        for code in 0..count {
            let (length, after) = rest
                .split_first_chunk::<4>()
                .ok_or_else(|| bad("truncated dictionary entry"))?;
            let (entry, after) = after
                .split_at_checked(u32::from_le_bytes(*length) as usize)
                .ok_or_else(|| bad("truncated dictionary entry"))?;
            rest = after;
            if finds(entry) {
                found.push(code);
            }
        }
        pos = data.len() - rest.len();
        let codes_count = read_u32(data, &mut pos).map_err(bad)? as usize;
        if codes_count != header.value_count as usize {
            return Err(bad("dictionary codes disagree with value count"));
        }
        let codes = take(data, &mut pos, codes_count * 4)
            .ok_or_else(|| bad("truncated dictionary codes"))?;
        Ok(Some(Self {
            row_count,
            presence,
            codes,
            found,
        }))
    }

    /// Whether no entry was found, so no row can be.
    pub(crate) fn finds_nothing(&self) -> bool {
        self.found.is_empty()
    }

    /// Calls `found` with the offset of each row of `scope` (by the word of rows) whose entry was
    /// found; returns how many rows it compared.
    pub(crate) fn find_codes(&self, scope: impl Fn(usize) -> u64, found: impl FnMut(u32)) -> usize {
        find_values(
            self.presence.as_ref(),
            self.row_count,
            self.codes.as_chunks::<4>().0,
            &self.found,
            u32::from_le_bytes,
            scope,
            found,
        )
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
                (Some(Value::Vector(actual)), Some(Value::Vector(expected))) => {
                    let bits =
                        |vector: &[f32]| vector.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&actual), bits(&expected), "row {row}");
                }
                (actual, expected) => assert_eq!(actual, expected, "row {row}"),
            }
        }
        assert_eq!(view.get(rows), None);
        assert_eq!(view.get(u32::MAX), None);
        for row in 0..rows {
            assert_eq!(view.has(row), view.get(row).is_some(), "row {row}");
        }
        assert!(!view.has(rows) && !view.has(u32::MAX));
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

    fn vector(components: &[f32]) -> Value {
        Value::Vector(components.into())
    }

    /// A vector chunk reads each row's vector from its own components, dense
    /// or sparse, across presence words, with the bits of every component
    /// kept (negative zero, NaN payloads, infinities).
    #[test]
    fn point_vectors_match_full_decoder() {
        let odd = [
            -0.0,
            f32::from_bits(0x7fc0_0123),
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE,
        ];
        let dense = [(0, vector(&odd)), (1, vector(&[3.0, 19.0, 88.0, 1.5]))];
        let view = parity(2, &dense, None);
        assert_eq!(view.index_heap_bytes(), 0);
        let sparse: Vec<_> = [0, 63, 64, 127, 2_047]
            .into_iter()
            .map(|row| (row, vector(&[f32::from(u16::try_from(row).unwrap()), -1.0])))
            .collect();
        parity(2_048, &sparse, Some(&[3, 19, 88, 0, 1]));
        let wide: Vec<f32> = (0..1_536u16).map(f32::from).collect();
        parity(3, &[(1, vector(&wide))], None);
        parity(1, &[(0, vector(&[0.25]))], None);
    }

    /// A vector body whose component count disagrees with its vectors is
    /// refused, not read past.
    #[test]
    fn point_vectors_refuse_a_short_body() {
        let (codec, mut encoded) = encode_column_chunk(
            2,
            &[(0, vector(&[1.0, 2.0])), (1, vector(&[3.0, 4.0]))],
            None,
        )
        .unwrap();
        // The component count sits before the last 16 bytes of components;
        // claim one more component than the body holds.
        let at = encoded.len() - 16 - 4;
        encoded[at..at + 4].copy_from_slice(&5u32.to_le_bytes());
        encoded.extend_from_slice(&0f32.to_le_bytes());
        let data = Bytes::from(encoded);
        assert!(PointView::from_validated(&data, codec.to_byte(), 2).is_err());
        assert!(PointView::supports(ChunkCodec::Float32Vector));
        assert!(!PointView::supports(ChunkCodec::Dict));
        assert!(!PointView::supports(ChunkCodec::Values));
    }

    #[test]
    fn point_complex_codecs_keep_the_decode_fallback() {
        for value in [
            Value::from("dictionary"),
            Value::Null,
            Value::List(vec![vector(&[1.0, 2.0])].into()),
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
