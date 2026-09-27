//! Column codecs for CompactStore.
//!
//! Wraps Grafeo's existing storage primitives into a unified enum with
//! random access and `Value` decoding. CompactStore owns these types:
//! the underlying primitives are not modified.

use std::sync::Arc;

use arcstr::ArcStr;
use bytes::{Bytes, BytesMut};
use grafeo_common::types::Value;

use crate::codec::{BitPackedInts, BitVector, BlockEntry, DictionaryEncoding};

// ── Phase 3a: Bytes-backed read helpers ──────────────────────────────
//
// Fixed-width codec variants (RawI64, Float64, Int8Vector, Float32Vector)
// store their raw bytes as `bytes::Bytes` rather than `Vec<T>`. This is
// the long-term storage abstraction for the entire columnar layer
// (revised D7): a heap-owned column and a mmap-backed column have the
// same type; the `Bytes` constructor decides. Phase 3c adds the mmap
// constructor (`Bytes::from_owner`).
//
// Read helpers use safe `from_le_bytes` (no `unsafe` required). On x86,
// modern compilers fold `try_into().unwrap()` + `from_le_bytes` into a
// single `mov` for aligned reads; unaligned reads cost nothing extra.
// Phase 3 endianness is locked to little-endian on disk; readers
// validate this contract at the section header, not per-element.

#[inline]
fn read_le_i64(bytes: &Bytes, byte_idx: usize) -> Option<i64> {
    let end = byte_idx.checked_add(8)?;
    let chunk: [u8; 8] = bytes.get(byte_idx..end)?.try_into().ok()?;
    Some(i64::from_le_bytes(chunk))
}

#[inline]
fn read_le_f64(bytes: &Bytes, byte_idx: usize) -> Option<f64> {
    let end = byte_idx.checked_add(8)?;
    let chunk: [u8; 8] = bytes.get(byte_idx..end)?.try_into().ok()?;
    Some(f64::from_le_bytes(chunk))
}

#[inline]
fn read_le_f32(bytes: &Bytes, byte_idx: usize) -> Option<f32> {
    let end = byte_idx.checked_add(4)?;
    let chunk: [u8; 4] = bytes.get(byte_idx..end)?.try_into().ok()?;
    Some(f32::from_le_bytes(chunk))
}

#[inline]
fn read_i8(bytes: &Bytes, byte_idx: usize) -> Option<i8> {
    bytes.get(byte_idx).copied().map(u8::cast_signed)
}

fn vec_to_bytes_i64(values: &[i64]) -> Bytes {
    let mut buf = BytesMut::with_capacity(values.len() * 8);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.freeze()
}

fn vec_to_bytes_f64(values: &[f64]) -> Bytes {
    let mut buf = BytesMut::with_capacity(values.len() * 8);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.freeze()
}

fn vec_to_bytes_f32(values: &[f32]) -> Bytes {
    let mut buf = BytesMut::with_capacity(values.len() * 4);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.freeze()
}

fn vec_to_bytes_i8(values: &[i8]) -> Bytes {
    // i8 and u8 have identical 1-byte layout; the cast is exact.
    let mut buf = BytesMut::with_capacity(values.len());
    for &v in values {
        buf.extend_from_slice(&[v.cast_unsigned()]);
    }
    buf.freeze()
}

/// Two-variant backing for `i64` columns. `Inline(Vec<i64>)` is used
/// when the column is built in RAM; `Mapped(Bytes)` carries an mmap
/// slice. Scans branch once to use the slice fast path when available.
#[derive(Debug, Clone)]
pub enum I64Store {
    /// In-memory `Vec<i64>` backing (built fresh in RAM).
    Inline(Vec<i64>),
    /// Refcounted byte buffer backing (mmap or pre-encoded bytes).
    Mapped(Bytes),
}

impl I64Store {
    /// Returns the number of `i64` elements stored.
    #[inline]
    #[must_use]
    pub fn len_elements(&self) -> usize {
        match self {
            Self::Inline(v) => v.len(),
            Self::Mapped(b) => b.len() / 8,
        }
    }

    /// Returns the underlying slice when the data is held inline, or
    /// `None` when it is mmap-backed (no native slice available).
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> Option<&[i64]> {
        match self {
            Self::Inline(v) => Some(v.as_slice()),
            Self::Mapped(_) => None,
        }
    }

    /// Returns the value at `idx`, or `None` if `idx` is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<i64> {
        match self {
            Self::Inline(v) => v.get(idx).copied(),
            Self::Mapped(b) => read_le_i64(b, idx.checked_mul(8)?),
        }
    }

    /// Materialises the contents as a [`Bytes`] buffer.
    ///
    /// `Mapped` returns a cheap refcount clone; `Inline` allocates and
    /// encodes to little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Bytes {
        match self {
            Self::Inline(v) => vec_to_bytes_i64(v),
            Self::Mapped(b) => b.clone(),
        }
    }

    /// Returns the byte length of the encoded data.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Inline(v) => v.len() * 8,
            Self::Mapped(b) => b.len(),
        }
    }
}

/// Two-variant backing for `f64` columns. `Inline(Vec<f64>)` is used
/// when the column is built in RAM; `Mapped(Bytes)` carries an mmap
/// slice. Scans branch once to use the slice fast path when available.
#[derive(Debug, Clone)]
pub enum F64Store {
    /// In-memory `Vec<f64>` backing (built fresh in RAM).
    Inline(Vec<f64>),
    /// Refcounted byte buffer backing (mmap or pre-encoded bytes).
    Mapped(Bytes),
}

impl F64Store {
    /// Returns the number of `f64` elements stored.
    #[inline]
    #[must_use]
    pub fn len_elements(&self) -> usize {
        match self {
            Self::Inline(v) => v.len(),
            Self::Mapped(b) => b.len() / 8,
        }
    }

    /// Returns the underlying slice when the data is held inline, or
    /// `None` when it is mmap-backed (no native slice available).
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> Option<&[f64]> {
        match self {
            Self::Inline(v) => Some(v.as_slice()),
            Self::Mapped(_) => None,
        }
    }

    /// Returns the value at `idx`, or `None` if `idx` is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<f64> {
        match self {
            Self::Inline(v) => v.get(idx).copied(),
            Self::Mapped(b) => read_le_f64(b, idx.checked_mul(8)?),
        }
    }

    /// Materialises the contents as a [`Bytes`] buffer.
    ///
    /// `Mapped` returns a cheap refcount clone; `Inline` allocates and
    /// encodes to little-endian bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Bytes {
        match self {
            Self::Inline(v) => vec_to_bytes_f64(v),
            Self::Mapped(b) => b.clone(),
        }
    }

    /// Returns the byte length of the encoded data.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::Inline(v) => v.len() * 8,
            Self::Mapped(b) => b.len(),
        }
    }
}

/// A single column of data backed by one of Grafeo's storage codecs.
///
/// Each variant wraps an existing primitive via composition: the
/// primitives themselves are never modified. Use [`get`](Self::get) for
/// `Value`-typed access and the specialised accessors when you know the
/// underlying codec.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ColumnCodec {
    /// Fixed-width bit-packed unsigned integers.
    ///
    /// Used for `Value::Int64` columns whose values are all `>= 0`. Preserves
    /// the `Int64` type on decode via `v as i64`.
    BitPacked(BitPackedInts),
    /// Dictionary-encoded strings.
    Dict(DictionaryEncoding),
    /// Null/boolean bitmap.
    Bitmap(BitVector),
    /// Int8 quantized vectors (flat array with stride). Bytes-backed
    /// (Phase 3a): each row occupies `dimensions` consecutive bytes.
    Int8Vector {
        /// Flat byte array; `len() / dimensions` logical rows.
        bytes: Bytes,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
    /// Native IEEE 754 double-precision floats. Two-variant backing
    /// (Phase 3d): [`F64Store::Inline`] keeps a `Vec<f64>` for
    /// in-memory builds (preserves native slice access); [`F64Store::Mapped`]
    /// holds LE `f64` bytes from mmap or pre-encoded buffers.
    Float64(F64Store),
    /// Float32 vectors (flat array with stride), for embedding / vector search.
    /// Bytes-backed (Phase 3a): LE `f32` components, `4 * dimensions`
    /// bytes per row.
    Float32Vector {
        /// Flat byte array; `len() / (4 * dimensions)` logical rows.
        bytes: Bytes,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
    /// Native signed 64-bit integers.
    ///
    /// Used for `Value::Int64` columns when at least one value is negative.
    /// `BitPacked` can't represent negatives correctly in its ordered
    /// operations (`find_eq`, `find_in_range`, zone maps) because it operates
    /// on `u64`; `RawI64` stores the values natively to preserve both the
    /// GQL type and signed ordering semantics on round-trip. Two-variant
    /// backing (Phase 3d): [`I64Store::Inline`] keeps a `Vec<i64>` for
    /// in-memory builds (preserves native slice access); [`I64Store::Mapped`]
    /// holds LE `i64` bytes from mmap or pre-encoded buffers.
    RawI64(I64Store),
}

impl ColumnCodec {
    // ── Phase 3a: Bytes-backed constructors ─────────────────────────

    /// Constructs a [`RawI64`](Self::RawI64) column from a `Vec<i64>`
    /// (in-memory path).
    ///
    /// The values are kept as a native `Vec<i64>` so scans can take the
    /// slice fast path via [`as_raw_i64_slice`](Self::as_raw_i64_slice).
    /// Use [`raw_i64_from_bytes`](Self::raw_i64_from_bytes) when loading
    /// from a mmap-backed [`Bytes`] buffer.
    #[must_use]
    pub fn raw_i64(values: Vec<i64>) -> Self {
        Self::RawI64(I64Store::Inline(values))
    }

    /// Constructs a [`RawI64`](Self::RawI64) column from pre-encoded
    /// little-endian `i64` bytes (mmap / on-disk load path).
    #[must_use]
    pub fn raw_i64_from_bytes(bytes: Bytes) -> Self {
        Self::RawI64(I64Store::Mapped(bytes))
    }

    /// Constructs a [`Float64`](Self::Float64) column from a `Vec<f64>`
    /// (in-memory path).
    ///
    /// The values are kept as a native `Vec<f64>` so scans can take the
    /// slice fast path via [`as_float64_slice`](Self::as_float64_slice).
    /// Use [`float64_from_bytes`](Self::float64_from_bytes) when loading
    /// from a mmap-backed [`Bytes`] buffer.
    #[must_use]
    pub fn float64(values: Vec<f64>) -> Self {
        Self::Float64(F64Store::Inline(values))
    }

    /// Constructs a [`Float64`](Self::Float64) column from pre-encoded
    /// little-endian `f64` bytes (mmap / on-disk load path).
    #[must_use]
    pub fn float64_from_bytes(bytes: Bytes) -> Self {
        Self::Float64(F64Store::Mapped(bytes))
    }

    /// Returns the underlying `i64` slice for a [`RawI64`](Self::RawI64)
    /// column when it lives in RAM.
    ///
    /// Returns `None` when the column is mmap-backed
    /// ([`I64Store::Mapped`]) or when this codec is not `RawI64`.
    #[must_use]
    #[inline]
    pub fn as_raw_i64_slice(&self) -> Option<&[i64]> {
        match self {
            Self::RawI64(s) => s.as_slice(),
            _ => None,
        }
    }

    /// Returns the underlying `f64` slice for a [`Float64`](Self::Float64)
    /// column when it lives in RAM.
    ///
    /// Returns `None` when the column is mmap-backed
    /// ([`F64Store::Mapped`]) or when this codec is not `Float64`.
    #[must_use]
    #[inline]
    pub fn as_float64_slice(&self) -> Option<&[f64]> {
        match self {
            Self::Float64(s) => s.as_slice(),
            _ => None,
        }
    }

    /// Constructs an [`Int8Vector`](Self::Int8Vector) column from a flat
    /// `Vec<i8>` and a per-vector dimension count.
    ///
    /// `data.len()` must be a multiple of `dimensions` (or 0 when
    /// `dimensions == 0`); the codec doesn't enforce this on construction
    /// to match the previous tuple-style API, but `len()` and `get` use
    /// integer division so a misaligned tail is silently truncated.
    #[must_use]
    pub fn int8_vector(data: Vec<i8>, dimensions: u16) -> Self {
        Self::Int8Vector {
            bytes: vec_to_bytes_i8(&data),
            dimensions,
        }
    }

    /// Constructs a [`Float32Vector`](Self::Float32Vector) column from a
    /// flat `Vec<f32>` and a per-vector dimension count. Each component
    /// is stored as 4 LE bytes.
    #[must_use]
    pub fn float32_vector(data: Vec<f32>, dimensions: u16) -> Self {
        Self::Float32Vector {
            bytes: vec_to_bytes_f32(&data),
            dimensions,
        }
    }

    /// Decodes the value at `index` into a [`Value`].
    ///
    /// - [`BitPacked`](Self::BitPacked) → `Value::Int64(v as i64)`
    /// - [`Dict`](Self::Dict) → `Value::String(ArcStr::from(s))`
    /// - [`Bitmap`](Self::Bitmap) → `Value::Bool(b)`
    /// - [`Int8Vector`](Self::Int8Vector) → `Value::List(...)` of `Int64` values
    /// - [`Float64`](Self::Float64) → `Value::Float64(f)`
    /// - [`RawI64`](Self::RawI64) → `Value::Int64(n)`
    ///
    /// Returns `None` when `index` is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value> {
        match self {
            // The builder validates all values <= i64::MAX, so this cast is lossless.
            // reason: values validated <= i64::MAX during build
            Self::BitPacked(bp) => bp.get(index).map(|v| {
                // reason: values validated <= i64::MAX during build
                #[allow(clippy::cast_possible_wrap)]
                let val = Value::Int64(v as i64);
                val
            }),
            Self::Dict(dict) => dict.get(index).map(|s| Value::String(ArcStr::from(s))),
            Self::Bitmap(bv) => bv.get(index).map(Value::Bool),
            Self::Int8Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > bytes.len() {
                    return None;
                }
                let values: Vec<Value> = (start..end)
                    .map(|i| Value::Int64(read_i8(bytes, i).unwrap_or(0) as i64))
                    .collect();
                Some(Value::List(Arc::from(values)))
            }
            Self::Float64(store) => store.get(index).map(Value::Float64),
            Self::RawI64(store) => store.get(index).map(Value::Int64),
            Self::Float32Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start_byte = index.checked_mul(dims)?.checked_mul(4)?;
                let end_byte = start_byte.checked_add(dims.checked_mul(4)?)?;
                if end_byte > bytes.len() {
                    return None;
                }
                let values: Vec<f32> = (0..dims)
                    .map(|d| read_le_f32(bytes, start_byte + d * 4).unwrap_or(0.0))
                    .collect();
                Some(Value::Vector(Arc::from(values.as_slice())))
            }
        }
    }

    /// Returns the raw `u64` stored at `index` (useful for FK columns).
    ///
    /// Only meaningful for [`BitPacked`](Self::BitPacked) columns; all other
    /// variants return `None`.
    #[inline]
    #[must_use]
    pub fn get_raw_u64(&self, index: usize) -> Option<u64> {
        match self {
            Self::BitPacked(bp) => bp.get(index),
            _ => None,
        }
    }

    /// Returns a slice over the int8 vector at `index`.
    ///
    /// Only meaningful for [`Int8Vector`](Self::Int8Vector) columns; all other
    /// variants return `None`.
    #[must_use]
    pub fn get_int8_vector(&self, index: usize) -> Option<&[i8]> {
        match self {
            Self::Int8Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > bytes.len() {
                    return None;
                }
                let u8_slice: &[u8] = &bytes[start..end];
                // SAFETY: `i8` and `u8` have identical layout (both are 1
                // byte, no padding, no niche). Reinterpreting `&[u8]` as
                // `&[i8]` is valid: the slice metadata (ptr + len) is the
                // same; only the element type changes for the caller.
                // This is the same operation `bytemuck::cast_slice` would
                // perform; we inline it to avoid a dependency.
                #[allow(unsafe_code)]
                let i8_slice: &[i8] = unsafe {
                    std::slice::from_raw_parts(u8_slice.as_ptr().cast::<i8>(), u8_slice.len())
                };
                Some(i8_slice)
            }
            _ => None,
        }
    }

    /// Returns the number of logical values in this column.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::BitPacked(bp) => bp.len(),
            Self::Dict(dict) => dict.len(),
            Self::Bitmap(bv) => bv.len(),
            Self::Int8Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                bytes.len().checked_div(dims).unwrap_or(0)
            }
            Self::Float64(store) => store.len_elements(),
            Self::Float32Vector { bytes, dimensions } => {
                let dims = *dimensions as usize;
                bytes.len().checked_div(dims * 4).unwrap_or(0)
            }
            Self::RawI64(store) => store.len_elements(),
        }
    }

    /// Returns `true` if the column contains no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of logical blocks in this column.
    ///
    /// Empty columns report `1` (a single zero-row block) so downstream
    /// serializers and iterators can treat block emission uniformly.
    /// Non-empty columns return `ceil(len / DEFAULT_BLOCK_ROWS)`.
    #[must_use]
    pub fn block_count(&self) -> usize {
        if self.is_empty() {
            return 1;
        }
        let block_rows = crate::codec::DEFAULT_BLOCK_ROWS as usize;
        self.len().div_ceil(block_rows)
    }

    /// Descriptor for the block at index `i`, or `None` if `i` is out of
    /// range.
    ///
    /// Block `i` covers rows `[i * DEFAULT_BLOCK_ROWS, min((i+1) *
    /// DEFAULT_BLOCK_ROWS, len()))`. Empty columns return a single
    /// zero-row block at index 0.
    ///
    /// Phase 2c will fill in per-block statistics (`min`, `max`,
    /// `null_count`, optional `bloom`).
    #[must_use]
    pub fn block_at(&self, i: usize) -> Option<BlockEntry> {
        if i >= self.block_count() {
            return None;
        }
        let block_rows = crate::codec::DEFAULT_BLOCK_ROWS as usize;
        let start = i * block_rows;
        let end = (start + block_rows).min(self.len());
        // reason: block lengths fit in u32 since DEFAULT_BLOCK_ROWS itself is u32.
        let row_count = u32::try_from(end - start).ok()?;
        Some(BlockEntry::new(row_count))
    }

    /// Iterator over this column's block descriptors.
    ///
    /// Phase 2a yields exactly one block (covering all rows). The
    /// invariant that each block's `row_count`s sum to `self.len()` will
    /// remain true through Phase 2b's multi-block layout.
    pub fn block_iter(&self) -> impl Iterator<Item = BlockEntry> + '_ {
        (0..self.block_count()).filter_map(move |i| self.block_at(i))
    }

    /// Returns the offsets of all rows whose value equals `target`.
    ///
    /// Operates in the codec's native domain to avoid per-row `Value`
    /// allocation:
    /// - [`BitPacked`](Self::BitPacked): compares raw `u64` values via
    ///   [`BitPackedInts::get`]
    /// - [`Dict`](Self::Dict): resolves the target string to a dictionary
    ///   code once, then scans integer codes via
    ///   [`DictionaryEncoding::filter_by_code`]
    /// - [`Bitmap`](Self::Bitmap): checks bits directly
    ///
    /// Falls back to [`get`](Self::get)-based comparison for type mismatches.
    pub fn find_eq(&self, target: &Value) -> Vec<usize> {
        match (self, target) {
            (Self::BitPacked(bp), &Value::Int64(v)) => {
                if v < 0 {
                    return Vec::new();
                }
                // reason: v >= 0 checked above
                #[allow(clippy::cast_sign_loss)]
                let target_u64 = v as u64;
                // `scan_eq` hoists the WordStore discriminant + bit-packing
                // constants out of the per-row loop. The previous
                // `(0..len).filter(|i| bp.get(i) == ...)` form forced LLVM
                // to re-check the variant tag and re-derive `values_per_word`
                // / `mask` per iteration, which CodSpeed callgrind counted
                // as real instruction cost (Tasks 1-8 follow-up).
                bp.scan_eq(target_u64)
            }
            (Self::Dict(dict), Value::String(s)) => match dict.encode(s.as_str()) {
                Some(code) => dict.filter_by_code(|c| c == code),
                None => Vec::new(),
            },
            (Self::Bitmap(bv), &Value::Bool(target_bool)) => (0..bv.len())
                .filter(|&i| bv.get(i) == Some(target_bool))
                .collect(),
            (Self::Float64(store), &Value::Float64(target)) => match store.as_slice() {
                Some(slice) => slice
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| v == target)
                    .map(|(i, _)| i)
                    .collect(),
                None => (0..store.len_elements())
                    .filter(|&i| store.get(i) == Some(target))
                    .collect(),
            },
            (Self::RawI64(store), &Value::Int64(target)) => match store.as_slice() {
                Some(slice) => slice
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| v == target)
                    .map(|(i, _)| i)
                    .collect(),
                None => (0..store.len_elements())
                    .filter(|&i| store.get(i) == Some(target))
                    .collect(),
            },
            _ => (0..self.len())
                .filter(|&i| self.get(i).as_ref() == Some(target))
                .collect(),
        }
    }

    /// Returns the offsets of all rows whose value falls within the given range.
    ///
    /// Like [`find_eq`](Self::find_eq), operates in the codec's native domain
    /// to avoid per-row `Value` allocation for integer columns.
    pub fn find_in_range(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<usize> {
        if let Self::BitPacked(bp) = self {
            let min_u64 = match min {
                // reason: v >= 0 guard ensures no sign loss
                #[allow(clippy::cast_sign_loss)]
                Some(&Value::Int64(v)) if v >= 0 => Some(v as u64),
                Some(&Value::Int64(_)) => Some(0),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };
            let max_u64 = match max {
                // reason: v >= 0 guard ensures no sign loss
                #[allow(clippy::cast_sign_loss)]
                Some(&Value::Int64(v)) if v >= 0 => Some(v as u64),
                Some(&Value::Int64(v)) if v < 0 => return Vec::new(),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };

            return (0..bp.len())
                .filter(|&i| {
                    if let Some(v) = bp.get(i) {
                        let above_min = match min_u64 {
                            Some(lo) if min_inclusive => v >= lo,
                            Some(lo) => v > lo,
                            None => true,
                        };
                        let below_max = match max_u64 {
                            Some(hi) if max_inclusive => v <= hi,
                            Some(hi) => v < hi,
                            None => true,
                        };
                        above_min && below_max
                    } else {
                        false
                    }
                })
                .collect();
        }

        if let Self::RawI64(store) = self {
            // Native i64 comparison: signed ordering semantics for free.
            let min_i64 = match min {
                Some(&Value::Int64(v)) => Some(v),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };
            let max_i64 = match max {
                Some(&Value::Int64(v)) => Some(v),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };

            let pred = |v: i64| {
                let above_min = match min_i64 {
                    Some(lo) if min_inclusive => v >= lo,
                    Some(lo) => v > lo,
                    None => true,
                };
                let below_max = match max_i64 {
                    Some(hi) if max_inclusive => v <= hi,
                    Some(hi) => v < hi,
                    None => true,
                };
                above_min && below_max
            };

            return match store.as_slice() {
                Some(slice) => slice
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| pred(v).then_some(i))
                    .collect(),
                None => (0..store.len_elements())
                    .filter_map(|i| store.get(i).and_then(|v| pred(v).then_some(i)))
                    .collect(),
            };
        }

        self.find_in_range_fallback(min, max, min_inclusive, max_inclusive)
    }

    /// Per-row range predicate: returns `true` iff row `i` decodes to a
    /// non-null value that satisfies `[min, max]` with the given
    /// inclusivity. Shared by `find_in_range_fallback` and `range_iter`.
    /// Rows with `None` decoded value (nulls, NaN floats, dictionary
    /// nulls) and rows whose value type is incomparable with the bounds
    /// are excluded.
    #[inline]
    fn matches_range(
        &self,
        i: usize,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> bool {
        use super::zone_map::compare_values;

        let Some(v) = self.get(i) else {
            return false;
        };
        if let Some(min_val) = min {
            match compare_values(&v, min_val) {
                Some(std::cmp::Ordering::Less) => return false,
                Some(std::cmp::Ordering::Equal) if !min_inclusive => return false,
                None => return false,
                _ => {}
            }
        }
        if let Some(max_val) = max {
            match compare_values(&v, max_val) {
                Some(std::cmp::Ordering::Greater) => return false,
                Some(std::cmp::Ordering::Equal) if !max_inclusive => return false,
                None => return false,
                _ => {}
            }
        }
        true
    }

    /// Fallback range scan via per-row `Value` decode.
    fn find_in_range_fallback(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<usize> {
        (0..self.len())
            .filter(|&i| self.matches_range(i, min, max, min_inclusive, max_inclusive))
            .collect()
    }

    /// Lazy iterator over row offsets matching a range predicate.
    ///
    /// Walks per-block zone maps to skip blocks whose stats prove no
    /// match, then evaluates the predicate per row within matching
    /// blocks. Yields offsets in ascending order.
    ///
    /// `block_zone_maps` should come from
    /// [`NodeTable::block_zone_maps_for`](super::node_table::NodeTable::block_zone_maps_for).
    /// When `None`, the iterator scans every block (no skip pruning) but
    /// still yields the same correct result. When `Some(slice)`, the
    /// slice length must equal [`block_count`](Self::block_count); a
    /// shape mismatch falls back to a full scan.
    ///
    /// The predicate semantics match
    /// [`find_in_range`](Self::find_in_range): rows whose value compares
    /// `Less` than `min` (or `Greater` than `max`) are excluded, and
    /// `min_inclusive`/`max_inclusive` control boundary behaviour.
    /// Rows with `None` decoded value (nulls, NaN floats) are excluded.
    pub fn range_iter<'a>(
        &'a self,
        block_zone_maps: Option<&'a [super::zone_map::ZoneMap]>,
        min: Option<&'a Value>,
        max: Option<&'a Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Box<dyn Iterator<Item = usize> + 'a> {
        use crate::graph::lpg::CompareOp;

        let block_count = self.block_count();
        let block_rows = crate::codec::DEFAULT_BLOCK_ROWS as usize;
        let total_len = self.len();

        // Validate zone-maps shape; mismatched slices act as "absent".
        let zone_maps_ok = block_zone_maps.is_some_and(|zm| zm.len() == block_count);
        let zone_maps = if zone_maps_ok { block_zone_maps } else { None };

        let blocks = (0..block_count).filter_map(move |block_idx| {
            let start = block_idx * block_rows;
            // Empty columns report block_count == 1 with start == 0 ==
            // total_len; the resulting empty range yields nothing below.
            let end = start.saturating_add(block_rows).min(total_len);

            if let Some(zms) = zone_maps {
                let zm = &zms[block_idx];
                // Range matches the block iff:
                //   - if min exists: block.max >= min  (Ge)  / >  (Gt)
                //   - if max exists: block.min <= max  (Le)  / <  (Lt)
                if let Some(min_val) = min {
                    let op = if min_inclusive {
                        CompareOp::Ge
                    } else {
                        CompareOp::Gt
                    };
                    if !zm.might_match(op, min_val) {
                        return None;
                    }
                }
                if let Some(max_val) = max {
                    let op = if max_inclusive {
                        CompareOp::Le
                    } else {
                        CompareOp::Lt
                    };
                    if !zm.might_match(op, max_val) {
                        return None;
                    }
                }
            }

            Some((start, end))
        });

        Box::new(blocks.flat_map(move |(start, end)| {
            (start..end)
                .filter(move |&i| self.matches_range(i, min, max, min_inclusive, max_inclusive))
        }))
    }

    // ── Serialization (for CompactStoreSection) ───────────────────

    /// Serializes this codec to a byte buffer.
    ///
    /// Format: `[discriminant: u8][codec-specific data]`
    ///
    /// # Errors
    ///
    /// Fails if a length or count does not fit the format's `u32` fields.
    pub fn write_to(&self, buf: &mut Vec<u8>) -> grafeo_common::utils::error::Result<()> {
        match self {
            Self::BitPacked(bp) => {
                buf.push(0); // discriminant
                buf.push(bp.bits_per_value());
                write_usize_as_u32(buf, bp.len())?;
                write_usize_as_u32(buf, bp.word_count())?;
                buf.extend_from_slice(bp.data_bytes().as_ref());
            }
            Self::Dict(dict) => {
                buf.push(1); // discriminant
                let dict_entries = dict.dictionary();
                write_usize_as_u32(buf, dict_entries.len())?;
                for entry in dict_entries.iter() {
                    let s = entry.as_ref().as_bytes();
                    write_usize_as_u32(buf, s.len())?;
                    buf.extend_from_slice(s);
                }
                write_usize_as_u32(buf, dict.code_count())?;
                buf.extend_from_slice(dict.codes_bytes().as_ref());
            }
            Self::Bitmap(bv) => {
                buf.push(2); // discriminant
                write_usize_as_u32(buf, bv.len())?;
                write_usize_as_u32(buf, bv.word_count())?;
                buf.extend_from_slice(bv.data_bytes());
            }
            Self::Int8Vector { bytes, dimensions } => {
                buf.push(3); // discriminant
                buf.extend_from_slice(&dimensions.to_le_bytes());
                write_usize_as_u32(buf, bytes.len())?;
                buf.extend_from_slice(bytes);
            }
            Self::Float64(store) => {
                buf.push(4); // discriminant
                let body = store.to_bytes();
                write_usize_as_u32(buf, body.len() / 8)?;
                buf.extend_from_slice(&body);
            }
            Self::Float32Vector { bytes, dimensions } => {
                buf.push(5); // discriminant
                buf.extend_from_slice(&dimensions.to_le_bytes());
                let dims_bytes = (*dimensions as usize) * 4;
                let total_components = bytes.len().checked_div(4).unwrap_or(0);
                write_usize_as_u32(buf, total_components)?;
                // Block-pad: ensure rows align to dims_bytes for read_from.
                let _ = dims_bytes;
                buf.extend_from_slice(bytes);
            }
            Self::RawI64(store) => {
                buf.push(6); // discriminant
                let body = store.to_bytes();
                write_usize_as_u32(buf, body.len() / 8)?;
                buf.extend_from_slice(&body);
            }
        }
        Ok(())
    }

    /// Deserializes a codec from a byte buffer at the given offset.
    ///
    /// Returns the decoded codec and advances `pos` past the consumed bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is truncated or the discriminant is unknown.
    pub fn read_from(data: &Bytes, pos: &mut usize) -> Result<Self, &'static str> {
        // Phase 3c: take `&Bytes` so storage construction can be zero-copy
        // via `data.slice(range)` when `data` wraps a mmap (Phase 3c
        // `Bytes::from_owner`). Scalar helpers below take `&[u8]`; we
        // pass the underlying view.
        let bytes = data.as_ref();
        let discriminant = *bytes.get(*pos).ok_or("truncated codec discriminant")?;
        *pos += 1;

        match discriminant {
            0 => {
                // BitPacked: contiguous LE u64 words → zero-copy slice.
                let bits = *bytes.get(*pos).ok_or("truncated bits_per_value")?;
                *pos += 1;
                let count = read_u32_le(bytes, pos)? as usize;
                let word_count = read_u32_le(bytes, pos)? as usize;
                let need = word_count
                    .checked_mul(8)
                    .ok_or("BitPacked word count overflow")?;
                if *pos + need > bytes.len() {
                    return Err("truncated BitPacked data");
                }
                let storage = data.slice(*pos..*pos + need);
                *pos += need;
                Ok(Self::BitPacked(BitPackedInts::from_bytes_storage(
                    storage, bits, count,
                )))
            }
            1 => {
                // Dict: dict header on heap; codes contiguous LE u32 → slice.
                let dict_len = read_u32_le(bytes, pos)? as usize;
                let mut entries: Vec<Arc<str>> = Vec::with_capacity(dict_len);
                for _ in 0..dict_len {
                    let slen = read_u32_le(bytes, pos)? as usize;
                    if *pos + slen > bytes.len() {
                        return Err("truncated dict string");
                    }
                    let s = std::str::from_utf8(&bytes[*pos..*pos + slen])
                        .map_err(|_| "invalid UTF-8 in dict")?;
                    entries.push(Arc::from(s));
                    *pos += slen;
                }
                let codes_len = read_u32_le(bytes, pos)? as usize;
                let need = codes_len.checked_mul(4).ok_or("Dict codes overflow")?;
                if *pos + need > bytes.len() {
                    return Err("truncated Dict codes");
                }
                let codes_bytes = data.slice(*pos..*pos + need);
                *pos += need;
                Ok(Self::Dict(DictionaryEncoding::from_bytes_storage(
                    Arc::from(entries.into_boxed_slice()),
                    codes_bytes,
                    codes_len,
                )))
            }
            2 => {
                // Bitmap: contiguous LE u64 words → zero-copy slice.
                let bit_len = read_u32_le(bytes, pos)? as usize;
                let word_count = read_u32_le(bytes, pos)? as usize;
                let need = word_count
                    .checked_mul(8)
                    .ok_or("Bitmap word count overflow")?;
                if *pos + need > bytes.len() {
                    return Err("truncated Bitmap data");
                }
                let storage = data.slice(*pos..*pos + need);
                *pos += need;
                Ok(Self::Bitmap(BitVector::from_bytes_storage(
                    storage, bit_len,
                )))
            }
            3 => {
                // Int8Vector
                let dimensions = read_u16_le(bytes, pos)?;
                let data_len = read_u32_le(bytes, pos)? as usize;
                if *pos + data_len > bytes.len() {
                    return Err("truncated Int8Vector data");
                }
                let storage = data.slice(*pos..*pos + data_len);
                *pos += data_len;
                Ok(Self::Int8Vector {
                    bytes: storage,
                    dimensions,
                })
            }
            4 => {
                // Float64: count of f64 values; storage is 8 bytes each.
                let count = read_u32_le(bytes, pos)? as usize;
                let byte_need = count.checked_mul(8).ok_or("Float64 length overflow")?;
                if *pos + byte_need > bytes.len() {
                    return Err("truncated Float64 data");
                }
                let storage = data.slice(*pos..*pos + byte_need);
                *pos += byte_need;
                Ok(Self::Float64(F64Store::Mapped(storage)))
            }
            5 => {
                // Float32Vector: total component count (rows * dims).
                let dimensions = read_u16_le(bytes, pos)?;
                let component_count = read_u32_le(bytes, pos)? as usize;
                let byte_need = component_count
                    .checked_mul(4)
                    .ok_or("Float32Vector length overflow")?;
                if *pos + byte_need > bytes.len() {
                    return Err("truncated Float32Vector data");
                }
                let storage = data.slice(*pos..*pos + byte_need);
                *pos += byte_need;
                Ok(Self::Float32Vector {
                    bytes: storage,
                    dimensions,
                })
            }
            6 => {
                // RawI64: count of i64 values; storage is 8 bytes each.
                let count = read_u32_le(bytes, pos)? as usize;
                let byte_need = count.checked_mul(8).ok_or("RawI64 length overflow")?;
                if *pos + byte_need > bytes.len() {
                    return Err("truncated RawI64 data");
                }
                let storage = data.slice(*pos..*pos + byte_need);
                *pos += byte_need;
                Ok(Self::RawI64(I64Store::Mapped(storage)))
            }
            _ => Err("unknown codec discriminant"),
        }
    }

    /// Serializes this codec to v2 format with a per-block index.
    ///
    /// Layout:
    /// ```text
    /// [disc:u8]
    /// [global_params...]                  // codec-specific (empty for RawI64/Bitmap/Float64)
    /// [block_count:u32]
    /// [block_index]                       // block_count x [byte_offset:u32, byte_len:u32, row_count:u32]
    /// [block_data]                        // concatenated per-block bodies
    /// ```
    ///
    /// Block bodies carry just the row data for their range; codec
    /// parameters that apply to the whole column (bit width, dictionary,
    /// vector dimensions) live in `global_params`. Phase 2c will extend
    /// the per-block index entry with stats fields by bumping section
    /// version to 3.
    ///
    /// # Errors
    ///
    /// Fails if the block count, a block's row count or the column's total
    /// byte size does not fit the format's `u32` fields (a column over
    /// 4 GiB, for example large vectors).
    pub fn write_to_v2(&self, buf: &mut Vec<u8>) -> grafeo_common::utils::error::Result<()> {
        let (metas, bodies) = self.emit_blocked_codec(buf)?;
        write_block_index_and_bodies(buf, &metas, &bodies)
    }

    /// Serializes this codec to v3 format: like v2, but each block index
    /// entry is followed by an inline per-block `ZoneMap` for skip
    /// pruning.
    ///
    /// `stats_hint` may pass pre-computed per-block stats from
    /// `NodeTable::block_zone_maps_for`; when `None` (or wrong shape),
    /// per-block stats are computed inline during write.
    ///
    /// # Errors
    ///
    /// Same conditions as [`write_to_v2`](Self::write_to_v2).
    pub fn write_to_v3(
        &self,
        buf: &mut Vec<u8>,
        stats_hint: Option<&[super::zone_map::ZoneMap]>,
    ) -> grafeo_common::utils::error::Result<()> {
        let (metas, bodies) = self.emit_blocked_codec(buf)?;
        let computed;
        let stats: &[super::zone_map::ZoneMap] = match stats_hint {
            Some(hint) if hint.len() == metas.len() => hint,
            _ => {
                computed = super::zone_map::compute_block_zone_maps(self);
                &computed
            }
        };
        write_block_index_and_bodies_with_stats(buf, &metas, &bodies, stats)
    }

    /// Pushes the discriminant + global params for this codec into
    /// `buf`, then collects per-block bodies and metadata. Shared by
    /// `write_to_v2` and `write_to_v3`.
    fn emit_blocked_codec(
        &self,
        buf: &mut Vec<u8>,
    ) -> grafeo_common::utils::error::Result<(Vec<BlockMeta>, Vec<u8>)> {
        let block_count = self.block_count();
        let block_rows = crate::codec::DEFAULT_BLOCK_ROWS as usize;
        let mut bodies: Vec<u8> = Vec::new();
        let mut metas: Vec<BlockMeta> = Vec::with_capacity(block_count);

        match self {
            Self::BitPacked(bp) => {
                buf.push(0);
                buf.push(bp.bits_per_value());
                let bits_per_value = bp.bits_per_value();
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(bp.len());
                    let row_count =
                        crate::codec::limits::checked_u32(end - start, "compact block row count")?;
                    let byte_offset = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block offset",
                    )?;
                    let row_values: Vec<u64> = (start..end)
                        .map(|j| bp.get(j).expect("row in range"))
                        .collect();
                    let block_packed =
                        crate::codec::BitPackedInts::pack_with_bits(&row_values, bits_per_value);
                    write_usize_as_u32(&mut bodies, block_packed.word_count())?;
                    bodies.extend_from_slice(block_packed.data_bytes().as_ref());
                    let byte_len = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block end",
                    )? - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Dict(dict) => {
                buf.push(1);
                let entries = dict.dictionary();
                write_usize_as_u32(buf, entries.len())?;
                for entry in entries.iter() {
                    let s = entry.as_ref().as_bytes();
                    write_usize_as_u32(buf, s.len())?;
                    buf.extend_from_slice(s);
                }
                let codes_bytes = dict.codes_bytes();
                let total_codes = dict.code_count();
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(total_codes);
                    let row_count =
                        crate::codec::limits::checked_u32(end - start, "compact block row count")?;
                    let byte_offset = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block offset",
                    )?;
                    bodies.extend_from_slice(&codes_bytes[start * 4..end * 4]);
                    let byte_len = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block end",
                    )? - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Bitmap(bv) => {
                buf.push(2);
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(bv.len());
                    let row_count =
                        crate::codec::limits::checked_u32(end - start, "compact block row count")?;
                    let byte_offset = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block offset",
                    )?;
                    let bits: Vec<bool> = (start..end)
                        .map(|j| bv.get(j).expect("row in range"))
                        .collect();
                    let block_bv = crate::codec::BitVector::from_bools(&bits);
                    write_usize_as_u32(&mut bodies, block_bv.word_count())?;
                    bodies.extend_from_slice(block_bv.data_bytes());
                    let byte_len = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block end",
                    )? - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Int8Vector { bytes, dimensions } => {
                buf.push(3);
                buf.extend_from_slice(&dimensions.to_le_bytes());
                let dims = *dimensions as usize;
                let row_count_total = bytes.len().checked_div(dims).unwrap_or(0);
                for i in 0..block_count {
                    let start_row = i * block_rows;
                    let end_row = (start_row + block_rows).min(row_count_total);
                    let row_count = crate::codec::limits::checked_u32(
                        end_row - start_row,
                        "compact block row count",
                    )?;
                    let byte_offset = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block offset",
                    )?;
                    if dims > 0 {
                        let start = start_row * dims;
                        let end = end_row * dims;
                        bodies.extend_from_slice(&bytes[start..end]);
                    }
                    let byte_len = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block end",
                    )? - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Float64(store) => {
                buf.push(4);
                let body = store.to_bytes();
                let total_rows = body.len() / 8;
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(total_rows);
                    let row_count =
                        crate::codec::limits::checked_u32(end - start, "compact block row count")?;
                    let byte_offset = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block offset",
                    )?;
                    bodies.extend_from_slice(&body[start * 8..end * 8]);
                    let byte_len = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block end",
                    )? - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::Float32Vector { bytes, dimensions } => {
                buf.push(5);
                buf.extend_from_slice(&dimensions.to_le_bytes());
                let dims = *dimensions as usize;
                let row_byte_size = dims.checked_mul(4).unwrap_or(0);
                // Match `len()` exactly: a zero-dimension column has no
                // logical rows, regardless of any (spurious) bytes. Using
                // `.max(1)` here would emit nonzero per-block row counts
                // while no body bytes are written, leaving the block index
                // out of sync with `len()` and the empty-column block
                // count.
                let row_count_total = bytes.len().checked_div(row_byte_size).unwrap_or(0);
                for i in 0..block_count {
                    let start_row = i * block_rows;
                    let end_row = (start_row + block_rows).min(row_count_total);
                    let row_count = crate::codec::limits::checked_u32(
                        end_row - start_row,
                        "compact block row count",
                    )?;
                    let byte_offset = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block offset",
                    )?;
                    if row_byte_size > 0 {
                        let start = start_row * row_byte_size;
                        let end = end_row * row_byte_size;
                        bodies.extend_from_slice(&bytes[start..end]);
                    }
                    let byte_len = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block end",
                    )? - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
            Self::RawI64(store) => {
                buf.push(6);
                let body = store.to_bytes();
                let total_rows = body.len() / 8;
                for i in 0..block_count {
                    let start = i * block_rows;
                    let end = (start + block_rows).min(total_rows);
                    let row_count =
                        crate::codec::limits::checked_u32(end - start, "compact block row count")?;
                    let byte_offset = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block offset",
                    )?;
                    bodies.extend_from_slice(&body[start * 8..end * 8]);
                    let byte_len = crate::codec::limits::checked_u32(
                        bodies.len(),
                        "compact column block end",
                    )? - byte_offset;
                    metas.push(BlockMeta {
                        byte_offset,
                        byte_len,
                        row_count,
                    });
                }
            }
        }

        Ok((metas, bodies))
    }

    /// Deserializes a codec from v2 format.
    ///
    /// Inverse of [`write_to_v2`](Self::write_to_v2). The reader is
    /// strict: it requires the block index and concatenated bodies to
    /// match exactly, and refuses unknown discriminants.
    ///
    /// # Errors
    ///
    /// Returns a static-string error on truncation, unknown discriminant,
    /// or block-index inconsistency (offset + len out of bounds).
    pub fn read_from_v2(data: &Bytes, pos: &mut usize) -> Result<Self, &'static str> {
        // Phase 3c: same `&Bytes` switch as `read_from`. Fixed-width and
        // Dict bodies are contiguous on disk (writer appends per-block in
        // order), so the whole bodies region can be sliced as one
        // refcounted view (zero-copy on the mmap path).
        //
        // BitPacked and Bitmap blocks carry inline `word_count` prefixes
        // and are bit-packed independently per block, so they cannot
        // be zero-copy under the v2/v3 format and continue to materialize
        // a `Vec<u64>` per load. A future format revision could lift this.
        let bytes = data.as_ref();
        let discriminant = *bytes.get(*pos).ok_or("truncated codec discriminant")?;
        *pos += 1;
        match discriminant {
            0 => {
                // BitPacked: per-block packing → materialize-on-load.
                let bits = *bytes.get(*pos).ok_or("truncated bits_per_value")?;
                *pos += 1;
                let (metas, bodies_start) = read_block_index(bytes, pos)?;
                let mut all_values: Vec<u64> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let body_end = body_start + meta.byte_len as usize;
                    if body_end > bytes.len() {
                        return Err("BitPacked block body out of bounds");
                    }
                    let mut bp = body_start;
                    let word_count = read_u32_le(bytes, &mut bp)? as usize;
                    let mut words = Vec::with_capacity(word_count);
                    for _ in 0..word_count {
                        words.push(read_u64_le(bytes, &mut bp)?);
                    }
                    let block_bp = crate::codec::BitPackedInts::from_raw_parts(
                        words,
                        bits,
                        meta.row_count as usize,
                    );
                    for j in 0..meta.row_count as usize {
                        all_values.push(
                            block_bp
                                .get(j)
                                .ok_or("BitPacked block index out of range")?,
                        );
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::BitPacked(
                    crate::codec::BitPackedInts::pack_with_bits(&all_values, bits),
                ))
            }
            1 => {
                // Dict: global dictionary header on heap; codes contiguous → slice.
                let dict_len = read_u32_le(bytes, pos)? as usize;
                let mut entries: Vec<Arc<str>> = Vec::with_capacity(dict_len);
                for _ in 0..dict_len {
                    let slen = read_u32_le(bytes, pos)? as usize;
                    if *pos + slen > bytes.len() {
                        return Err("truncated dict string");
                    }
                    let s = std::str::from_utf8(&bytes[*pos..*pos + slen])
                        .map_err(|_| "invalid UTF-8 in dict")?;
                    entries.push(Arc::from(s));
                    *pos += slen;
                }
                let (metas, bodies_start) = read_block_index(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("Dict v2 bodies out of bounds");
                }
                let codes_bytes = data.slice(bodies_start..bodies_start + total);
                let code_count = total / 4;
                *pos = bodies_start + total;
                Ok(Self::Dict(DictionaryEncoding::from_bytes_storage(
                    Arc::from(entries.into_boxed_slice()),
                    codes_bytes,
                    code_count,
                )))
            }
            2 => {
                // Bitmap: per-block packing → materialize-on-load.
                let (metas, bodies_start) = read_block_index(bytes, pos)?;
                let mut all_bits: Vec<bool> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let mut bp = body_start;
                    let word_count = read_u32_le(bytes, &mut bp)? as usize;
                    let mut words = Vec::with_capacity(word_count);
                    for _ in 0..word_count {
                        words.push(read_u64_le(bytes, &mut bp)?);
                    }
                    let block_bv =
                        crate::codec::BitVector::from_raw_parts(words, meta.row_count as usize);
                    for j in 0..meta.row_count as usize {
                        all_bits.push(block_bv.get(j).ok_or("Bitmap block index out of range")?);
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok(Self::Bitmap(crate::codec::BitVector::from_bools(&all_bits)))
            }
            3 => {
                // Int8Vector: contiguous bytes → zero-copy slice.
                let dimensions = read_u16_le(bytes, pos)?;
                let (metas, bodies_start) = read_block_index(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("Int8Vector v2 bodies out of bounds");
                }
                let storage = data.slice(bodies_start..bodies_start + total);
                *pos = bodies_start + total;
                Ok(Self::Int8Vector {
                    bytes: storage,
                    dimensions,
                })
            }
            4 => {
                // Float64: contiguous LE u64 bytes → zero-copy slice.
                let (metas, bodies_start) = read_block_index(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("Float64 v2 bodies out of bounds");
                }
                let storage = data.slice(bodies_start..bodies_start + total);
                *pos = bodies_start + total;
                Ok(Self::Float64(F64Store::Mapped(storage)))
            }
            5 => {
                // Float32Vector: contiguous LE bytes → zero-copy slice.
                let dimensions = read_u16_le(bytes, pos)?;
                let (metas, bodies_start) = read_block_index(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("Float32Vector v2 bodies out of bounds");
                }
                let storage = data.slice(bodies_start..bodies_start + total);
                *pos = bodies_start + total;
                Ok(Self::Float32Vector {
                    bytes: storage,
                    dimensions,
                })
            }
            6 => {
                // RawI64: contiguous LE i64 bytes → zero-copy slice.
                let (metas, bodies_start) = read_block_index(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("RawI64 v2 bodies out of bounds");
                }
                let storage = data.slice(bodies_start..bodies_start + total);
                *pos = bodies_start + total;
                Ok(Self::RawI64(I64Store::Mapped(storage)))
            }
            _ => Err("unknown codec discriminant"),
        }
    }

    /// Deserializes a codec from v3 format, returning the codec and
    /// per-block zone maps for skip pruning.
    ///
    /// The body of each block uses the same layout as v2; only the
    /// block index is extended with an inline `ZoneMap` per entry.
    ///
    /// # Errors
    ///
    /// Returns a static-string error on truncation, unknown
    /// discriminant, or block-index inconsistency.
    pub fn read_from_v3(
        data: &Bytes,
        pos: &mut usize,
    ) -> Result<(Self, Vec<super::zone_map::ZoneMap>), &'static str> {
        // Phase 3c: same `&Bytes` shape as `read_from_v2`. v3 differs
        // only in that the block index carries inline per-block stats.
        let bytes = data.as_ref();
        let discriminant = *bytes.get(*pos).ok_or("truncated codec discriminant")?;
        *pos += 1;
        match discriminant {
            0 => {
                let bits = *bytes.get(*pos).ok_or("truncated bits_per_value")?;
                *pos += 1;
                let (metas, stats, bodies_start) = read_block_index_v3(bytes, pos)?;
                let mut all_values: Vec<u64> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let body_end = body_start + meta.byte_len as usize;
                    if body_end > bytes.len() {
                        return Err("BitPacked block body out of bounds");
                    }
                    let mut bp = body_start;
                    let word_count = read_u32_le(bytes, &mut bp)? as usize;
                    let mut words = Vec::with_capacity(word_count);
                    for _ in 0..word_count {
                        words.push(read_u64_le(bytes, &mut bp)?);
                    }
                    let block_bp = crate::codec::BitPackedInts::from_raw_parts(
                        words,
                        bits,
                        meta.row_count as usize,
                    );
                    for j in 0..meta.row_count as usize {
                        all_values.push(
                            block_bp
                                .get(j)
                                .ok_or("BitPacked block index out of range")?,
                        );
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((
                    Self::BitPacked(crate::codec::BitPackedInts::pack_with_bits(
                        &all_values,
                        bits,
                    )),
                    stats,
                ))
            }
            1 => {
                let dict_len = read_u32_le(bytes, pos)? as usize;
                let mut entries: Vec<Arc<str>> = Vec::with_capacity(dict_len);
                for _ in 0..dict_len {
                    let slen = read_u32_le(bytes, pos)? as usize;
                    if *pos + slen > bytes.len() {
                        return Err("truncated dict string");
                    }
                    let s = std::str::from_utf8(&bytes[*pos..*pos + slen])
                        .map_err(|_| "invalid UTF-8 in dict")?;
                    entries.push(Arc::from(s));
                    *pos += slen;
                }
                let (metas, stats, bodies_start) = read_block_index_v3(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("Dict v3 bodies out of bounds");
                }
                let codes_bytes = data.slice(bodies_start..bodies_start + total);
                let code_count = total / 4;
                *pos = bodies_start + total;
                Ok((
                    Self::Dict(DictionaryEncoding::from_bytes_storage(
                        Arc::from(entries.into_boxed_slice()),
                        codes_bytes,
                        code_count,
                    )),
                    stats,
                ))
            }
            2 => {
                let (metas, stats, bodies_start) = read_block_index_v3(bytes, pos)?;
                let mut all_bits: Vec<bool> = Vec::new();
                for meta in &metas {
                    let body_start = bodies_start + meta.byte_offset as usize;
                    let mut bp = body_start;
                    let word_count = read_u32_le(bytes, &mut bp)? as usize;
                    let mut words = Vec::with_capacity(word_count);
                    for _ in 0..word_count {
                        words.push(read_u64_le(bytes, &mut bp)?);
                    }
                    let block_bv =
                        crate::codec::BitVector::from_raw_parts(words, meta.row_count as usize);
                    for j in 0..meta.row_count as usize {
                        all_bits.push(block_bv.get(j).ok_or("Bitmap block index out of range")?);
                    }
                }
                *pos = bodies_start + total_bodies_len(&metas);
                Ok((
                    Self::Bitmap(crate::codec::BitVector::from_bools(&all_bits)),
                    stats,
                ))
            }
            3 => {
                let dimensions = read_u16_le(bytes, pos)?;
                let (metas, stats, bodies_start) = read_block_index_v3(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("Int8Vector v3 bodies out of bounds");
                }
                let storage = data.slice(bodies_start..bodies_start + total);
                *pos = bodies_start + total;
                Ok((
                    Self::Int8Vector {
                        bytes: storage,
                        dimensions,
                    },
                    stats,
                ))
            }
            4 => {
                let (metas, stats, bodies_start) = read_block_index_v3(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("Float64 v3 bodies out of bounds");
                }
                let storage = data.slice(bodies_start..bodies_start + total);
                *pos = bodies_start + total;
                Ok((Self::Float64(F64Store::Mapped(storage)), stats))
            }
            5 => {
                let dimensions = read_u16_le(bytes, pos)?;
                let (metas, stats, bodies_start) = read_block_index_v3(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("Float32Vector v3 bodies out of bounds");
                }
                let storage = data.slice(bodies_start..bodies_start + total);
                *pos = bodies_start + total;
                Ok((
                    Self::Float32Vector {
                        bytes: storage,
                        dimensions,
                    },
                    stats,
                ))
            }
            6 => {
                let (metas, stats, bodies_start) = read_block_index_v3(bytes, pos)?;
                let total = total_bodies_len(&metas);
                if bodies_start + total > bytes.len() {
                    return Err("RawI64 v3 bodies out of bounds");
                }
                let storage = data.slice(bodies_start..bodies_start + total);
                *pos = bodies_start + total;
                Ok((Self::RawI64(I64Store::Mapped(storage)), stats))
            }
            _ => Err("unknown codec discriminant"),
        }
    }

    /// Returns an estimate of heap memory used by this column in bytes.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::BitPacked(bp) => bp.data_bytes().len(),
            Self::Dict(d) => {
                let codes_bytes = d.code_count() * 4;
                let dict_bytes: usize = d.dictionary().iter().map(|s| s.len()).sum();
                codes_bytes + dict_bytes
            }
            Self::Bitmap(bv) => bv.data_bytes().len(),
            Self::Int8Vector { bytes, .. } => bytes.len(),
            Self::Float64(store) => store.byte_len(),
            Self::Float32Vector { bytes, .. } => bytes.len(),
            Self::RawI64(store) => store.byte_len(),
        }
    }
}

// ── v2 block-index helpers ──────────────────────────────────────

/// Per-block metadata in the v2 column index.
///
/// Phase 2c will extend this with stats fields by bumping the section
/// version to 3; the current 12-byte layout pins the v2 format.
#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    byte_offset: u32,
    byte_len: u32,
    row_count: u32,
}

const BLOCK_META_BYTES: usize = 12;

/// Total byte length of all bodies described by `metas`.
fn total_bodies_len(metas: &[BlockMeta]) -> usize {
    metas
        .last()
        .map_or(0, |m| (m.byte_offset + m.byte_len) as usize)
}

/// Writes the v2 block index followed by the concatenated block bodies.
fn write_block_index_and_bodies(
    buf: &mut Vec<u8>,
    metas: &[BlockMeta],
    bodies: &[u8],
) -> grafeo_common::utils::error::Result<()> {
    write_usize_as_u32(buf, metas.len())?;
    for meta in metas {
        buf.extend_from_slice(&meta.byte_offset.to_le_bytes());
        buf.extend_from_slice(&meta.byte_len.to_le_bytes());
        buf.extend_from_slice(&meta.row_count.to_le_bytes());
    }
    buf.extend_from_slice(bodies);
    Ok(())
}

/// Writes the v3 block index (v2 layout + inline per-block ZoneMap)
/// followed by the concatenated block bodies.
fn write_block_index_and_bodies_with_stats(
    buf: &mut Vec<u8>,
    metas: &[BlockMeta],
    bodies: &[u8],
    stats: &[super::zone_map::ZoneMap],
) -> grafeo_common::utils::error::Result<()> {
    debug_assert_eq!(metas.len(), stats.len(), "stats must align with metas");
    write_usize_as_u32(buf, metas.len())?;
    for (meta, zm) in metas.iter().zip(stats.iter()) {
        buf.extend_from_slice(&meta.byte_offset.to_le_bytes());
        buf.extend_from_slice(&meta.byte_len.to_le_bytes());
        buf.extend_from_slice(&meta.row_count.to_le_bytes());
        zm.write_inline(buf);
    }
    buf.extend_from_slice(bodies);
    Ok(())
}

/// Reads the v2 block index and returns the parsed metas and the byte
/// position where the concatenated bodies start.
fn read_block_index(data: &[u8], pos: &mut usize) -> Result<(Vec<BlockMeta>, usize), &'static str> {
    let block_count = read_u32_le(data, pos)? as usize;
    let index_bytes = block_count
        .checked_mul(BLOCK_META_BYTES)
        .ok_or("block index overflow")?;
    if *pos + index_bytes > data.len() {
        return Err("truncated block index");
    }
    let mut metas = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let byte_offset = read_u32_le(data, pos)?;
        let byte_len = read_u32_le(data, pos)?;
        let row_count = read_u32_le(data, pos)?;
        metas.push(BlockMeta {
            byte_offset,
            byte_len,
            row_count,
        });
    }
    validate_block_metas(&metas)?;
    let bodies_start = *pos;
    Ok((metas, bodies_start))
}

/// Validates that block metas describe a tightly-packed, gap-free,
/// non-overlapping body region.
///
/// Zero-copy readers slice `bodies_start..bodies_start + total_bodies_len`
/// and rely on the block layout being contiguous from offset 0; without
/// this check, malformed input could decode into off-by-one or
/// overlapping body slices and produce silent corruption rather than an
/// error.
fn validate_block_metas(metas: &[BlockMeta]) -> Result<(), &'static str> {
    let mut expected_offset: u64 = 0;
    for meta in metas {
        if u64::from(meta.byte_offset) != expected_offset {
            return Err("non-contiguous block index (gap or overlap)");
        }
        expected_offset = expected_offset
            .checked_add(u64::from(meta.byte_len))
            .ok_or("block byte_len overflow")?;
    }
    // Cap the total at u32::MAX so callers using `total_bodies_len() as
    // u32` (writers, future readers) do not silently truncate. The on-disk
    // BlockMeta uses u32 fields, so overflow here means malformed input.
    if expected_offset > u64::from(u32::MAX) {
        return Err("block bodies exceed u32 range");
    }
    Ok(())
}

/// Reads the v3 block index (v2 layout + inline per-block ZoneMap) and
/// returns the parsed metas, per-block stats, and the byte position
/// where the concatenated bodies start.
fn read_block_index_v3(
    data: &[u8],
    pos: &mut usize,
) -> Result<(Vec<BlockMeta>, Vec<super::zone_map::ZoneMap>, usize), &'static str> {
    let block_count = read_u32_le(data, pos)? as usize;
    let mut metas = Vec::with_capacity(block_count);
    let mut stats = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let byte_offset = read_u32_le(data, pos)?;
        let byte_len = read_u32_le(data, pos)?;
        let row_count = read_u32_le(data, pos)?;
        let zm = super::zone_map::ZoneMap::read_inline(data, pos)?;
        metas.push(BlockMeta {
            byte_offset,
            byte_len,
            row_count,
        });
        stats.push(zm);
    }
    validate_block_metas(&metas)?;
    let bodies_start = *pos;
    Ok((metas, stats, bodies_start))
}

// ── Binary read helpers ─────────────────────────────────────────

/// Writes a usize as u32 LE, failing if it does not fit (data over 4 GiB).
fn write_usize_as_u32(buf: &mut Vec<u8>, v: usize) -> grafeo_common::utils::error::Result<()> {
    let n = crate::codec::limits::checked_u32(v, "compact store column size")?;
    buf.extend_from_slice(&n.to_le_bytes());
    Ok(())
}

fn read_u16_le(data: &[u8], pos: &mut usize) -> Result<u16, &'static str> {
    if *pos + 2 > data.len() {
        return Err("truncated u16");
    }
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated u32");
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_u64_le(data: &[u8], pos: &mut usize) -> Result<u64, &'static str> {
    if *pos + 8 > data.len() {
        return Err("truncated u64");
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

#[cfg(test)]
// reason: test values are small known constants
#[allow(clippy::cast_possible_wrap)]
mod tests {
    use super::*;
    use crate::codec::{BitPackedInts, BitVector, DictionaryBuilder};

    #[test]
    fn test_bitpacked_round_trip() {
        // 4-bit values (max = 15)
        let values = vec![0u64, 5, 10, 15, 3, 7];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);

        assert_eq!(col.len(), 6);
        assert!(!col.is_empty());

        for (i, &expected) in values.iter().enumerate() {
            let v = col.get(i).unwrap();
            assert_eq!(v, Value::Int64(expected as i64));
        }
    }

    #[test]
    fn test_dict_round_trip() {
        let mut builder = DictionaryBuilder::new();
        builder.add("alpha");
        builder.add("beta");
        builder.add("alpha");
        let dict = builder.build();

        let col = ColumnCodec::Dict(dict);
        assert_eq!(col.len(), 3);

        assert_eq!(col.get(0), Some(Value::String(ArcStr::from("alpha"))));
        assert_eq!(col.get(1), Some(Value::String(ArcStr::from("beta"))));
        assert_eq!(col.get(2), Some(Value::String(ArcStr::from("alpha"))));
    }

    #[test]
    fn test_bitmap_round_trip() {
        let bools = vec![true, false, true, true, false];
        let bv = BitVector::from_bools(&bools);
        let col = ColumnCodec::Bitmap(bv);

        assert_eq!(col.len(), 5);
        assert_eq!(col.get(0), Some(Value::Bool(true)));
        assert_eq!(col.get(1), Some(Value::Bool(false)));
        assert_eq!(col.get(2), Some(Value::Bool(true)));
        assert_eq!(col.get(3), Some(Value::Bool(true)));
        assert_eq!(col.get(4), Some(Value::Bool(false)));
    }

    #[test]
    fn test_int8_vector_round_trip() {
        // 2 vectors of dimension 3
        let data = vec![1i8, 2, 3, -4, -5, -6];
        let col = ColumnCodec::int8_vector(data, 3);

        assert_eq!(col.len(), 2);

        let v0 = col.get(0).unwrap();
        let expected0: Vec<Value> = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        assert_eq!(v0, Value::List(Arc::from(expected0)));

        let v1 = col.get(1).unwrap();
        let expected1: Vec<Value> = vec![Value::Int64(-4), Value::Int64(-5), Value::Int64(-6)];
        assert_eq!(v1, Value::List(Arc::from(expected1)));
    }

    #[test]
    fn test_get_raw_u64_on_bitpacked() {
        let values = vec![100u64, 200, 300];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);

        assert_eq!(col.get_raw_u64(0), Some(100));
        assert_eq!(col.get_raw_u64(1), Some(200));
        assert_eq!(col.get_raw_u64(2), Some(300));
        assert_eq!(col.get_raw_u64(3), None);

        // Non-BitPacked variant returns None.
        let bv = BitVector::from_bools(&[true]);
        let bm_col = ColumnCodec::Bitmap(bv);
        assert_eq!(bm_col.get_raw_u64(0), None);
    }

    #[test]
    fn test_get_int8_vector_slice() {
        let data = vec![10i8, 20, 30, 40, 50, 60];
        let col = ColumnCodec::int8_vector(data, 3);

        assert_eq!(col.get_int8_vector(0), Some(&[10i8, 20, 30][..]));
        assert_eq!(col.get_int8_vector(1), Some(&[40i8, 50, 60][..]));
        assert_eq!(col.get_int8_vector(2), None);

        // Non-Int8Vector variant returns None.
        let bp = BitPackedInts::pack(&[1u64]);
        let bp_col = ColumnCodec::BitPacked(bp);
        assert_eq!(bp_col.get_int8_vector(0), None);
    }

    #[test]
    fn test_out_of_bounds_returns_none() {
        let bp = BitPackedInts::pack(&[1u64, 2, 3]);
        let col = ColumnCodec::BitPacked(bp);
        assert_eq!(col.get(999), None);
        assert_eq!(col.get_raw_u64(999), None);

        let bv = BitVector::from_bools(&[true]);
        let bm = ColumnCodec::Bitmap(bv);
        assert_eq!(bm.get(5), None);

        let mut builder = DictionaryBuilder::new();
        builder.add("x");
        let dict = builder.build();
        let dc = ColumnCodec::Dict(dict);
        assert_eq!(dc.get(10), None);

        let vec_col = ColumnCodec::int8_vector(vec![1, 2], 2);
        assert_eq!(vec_col.get(1), None);
        assert_eq!(vec_col.get_int8_vector(1), None);
    }

    // -----------------------------------------------------------------------
    // find_eq tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_eq_bitpacked() {
        let values = vec![0u64, 5, 10, 5, 3, 5];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);

        assert_eq!(col.find_eq(&Value::Int64(5)), vec![1, 3, 5]);
        assert_eq!(col.find_eq(&Value::Int64(0)), vec![0]);
        assert_eq!(col.find_eq(&Value::Int64(99)), Vec::<usize>::new());
        // Negative target: BitPacked stores unsigned values, no matches.
        assert_eq!(col.find_eq(&Value::Int64(-1)), Vec::<usize>::new());
    }

    #[test]
    fn test_find_eq_dict() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Vincent", "Jules", "Vincent", "Mia", "Jules"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        assert_eq!(col.find_eq(&Value::String("Vincent".into())), vec![0, 2]);
        assert_eq!(col.find_eq(&Value::String("Mia".into())), vec![3]);
        assert_eq!(
            col.find_eq(&Value::String("Butch".into())),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_find_eq_bitmap() {
        let bools = vec![true, false, true, true, false];
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&bools));

        assert_eq!(col.find_eq(&Value::Bool(true)), vec![0, 2, 3]);
        assert_eq!(col.find_eq(&Value::Bool(false)), vec![1, 4]);
    }

    #[test]
    fn test_find_eq_type_mismatch_uses_fallback() {
        let values = vec![1u64, 2, 3];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // String target on BitPacked column: type mismatch, falls back.
        assert_eq!(
            col.find_eq(&Value::String("hello".into())),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_find_eq_int8_vector_uses_fallback() {
        // Int8Vector has no specialised find_eq path, so it uses the fallback.
        let data = vec![1i8, 2, 3, 4, 5, 6];
        let col = ColumnCodec::int8_vector(data, 3);
        let target_vec: Vec<Value> = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        let target = Value::List(Arc::from(target_vec));
        let matches = col.find_eq(&target);
        assert_eq!(matches, vec![0]);
    }

    // -----------------------------------------------------------------------
    // Int8Vector zero-dimension edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_int8_vector_zero_dimensions_get() {
        let col = ColumnCodec::int8_vector(vec![1, 2, 3], 0);
        // Zero dimensions: get() should return None.
        assert_eq!(col.get(0), None);
    }

    #[test]
    fn test_int8_vector_zero_dimensions_get_int8_vector() {
        let col = ColumnCodec::int8_vector(vec![1, 2, 3], 0);
        // Zero dimensions: get_int8_vector() should return None.
        assert_eq!(col.get_int8_vector(0), None);
    }

    #[test]
    fn test_int8_vector_zero_dimensions_len_and_is_empty() {
        let col = ColumnCodec::int8_vector(vec![1, 2, 3], 0);
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
    }

    // -----------------------------------------------------------------------
    // heap_bytes tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_heap_bytes_bitpacked() {
        let values = vec![0u64, 5, 10, 15];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);
        // Should report nonzero heap usage.
        assert!(col.heap_bytes() > 0);
    }

    #[test]
    fn test_heap_bytes_dict() {
        let mut builder = DictionaryBuilder::new();
        builder.add("Amsterdam");
        builder.add("Berlin");
        builder.add("Paris");
        let dict = builder.build();
        let col = ColumnCodec::Dict(dict);
        assert!(col.heap_bytes() > 0);
    }

    #[test]
    fn test_heap_bytes_bitmap() {
        let bools = vec![true, false, true, true, false];
        let bv = BitVector::from_bools(&bools);
        let col = ColumnCodec::Bitmap(bv);
        assert!(col.heap_bytes() > 0);
    }

    #[test]
    fn test_heap_bytes_int8_vector() {
        let data = vec![1i8, 2, 3, 4, 5, 6];
        let col = ColumnCodec::int8_vector(data, 3);
        // Heap usage equals data length (1 byte per i8).
        assert_eq!(col.heap_bytes(), 6);
    }

    // -----------------------------------------------------------------------
    // find_in_range tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_in_range_bitpacked_inclusive() {
        // values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let values: Vec<u64> = (0..10).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // [3, 6] inclusive
        let result = col.find_in_range(Some(&Value::Int64(3)), Some(&Value::Int64(6)), true, true);
        assert_eq!(result, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_find_in_range_bitpacked_exclusive() {
        let values: Vec<u64> = (0..10).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // (3, 6) exclusive
        let result =
            col.find_in_range(Some(&Value::Int64(3)), Some(&Value::Int64(6)), false, false);
        assert_eq!(result, vec![4, 5]);
    }

    #[test]
    fn test_find_in_range_bitpacked_open_ended() {
        let values: Vec<u64> = (0..10).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // > 7 (no upper bound)
        let result = col.find_in_range(Some(&Value::Int64(7)), None, false, false);
        assert_eq!(result, vec![8, 9]);

        // <= 2 (no lower bound)
        let result = col.find_in_range(None, Some(&Value::Int64(2)), false, true);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_find_in_range_fallback_for_dict() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Amsterdam", "Berlin", "Paris", "Prague"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        // String range ["Berlin", "Prague"] inclusive: Berlin, Paris, Prague
        let result = col.find_in_range(
            Some(&Value::String("Berlin".into())),
            Some(&Value::String("Prague".into())),
            true,
            true,
        );
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_find_in_range_negative_max() {
        // All values are unsigned (>= 0), so a negative max should yield no results.
        let values: Vec<u64> = (0..10).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        let result = col.find_in_range(None, Some(&Value::Int64(-1)), false, true);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_in_range_negative_min() {
        // Negative min is clamped to 0 internally: all values (0..10) should pass.
        let values: Vec<u64> = (0..5).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        let result = col.find_in_range(Some(&Value::Int64(-10)), None, true, true);
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_find_in_range_type_mismatch_uses_fallback() {
        let values = vec![1u64, 2, 3];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // String bounds on a BitPacked column: type mismatch, falls back.
        let result = col.find_in_range(
            Some(&Value::String("a".into())),
            Some(&Value::String("z".into())),
            true,
            true,
        );
        // Fallback uses compare_values which returns None for Int vs String,
        // so no rows match.
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_in_range_int8_vector_uses_fallback() {
        let data = vec![1i8, 2, 3, 4, 5, 6];
        let col = ColumnCodec::int8_vector(data, 3);

        // Int8Vector is not BitPacked, so it goes through the fallback path.
        // Range scan on list values uses compare_values, which returns None
        // for lists, so nothing matches.
        let result = col.find_in_range(Some(&Value::Int64(0)), Some(&Value::Int64(10)), true, true);
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_out_of_bounds_all_codecs() {
        // BitPacked
        let bp = BitPackedInts::pack(&[1u64, 2, 3]);
        let col = ColumnCodec::BitPacked(bp);
        assert_eq!(col.get(3), None);

        // Dict
        let mut builder = DictionaryBuilder::new();
        builder.add("Alix");
        let col = ColumnCodec::Dict(builder.build());
        assert_eq!(col.get(1), None);

        // Bitmap
        let bv = BitVector::from_bools(&[true]);
        let col = ColumnCodec::Bitmap(bv);
        assert_eq!(col.get(1), None);

        // Int8Vector
        let col = ColumnCodec::int8_vector(vec![1, 2, 3], 3);
        assert_eq!(col.get(1), None);
        assert_eq!(col.get_int8_vector(1), None);
    }

    // -----------------------------------------------------------------------
    // Large Int8Vector column: serde round-trip and random access at scale
    // -----------------------------------------------------------------------

    /// Build a 100-vector Int8Vector column of dim 384, serialize it via
    /// `write_to`, deserialize via `read_from`, and verify bit-exact roundtrip
    /// at representative indices. Covers the Int8Vector branches in `get`,
    /// `get_int8_vector`, and both `write_to` / `read_from` (discriminant 3).
    #[test]
    fn test_column_int8_vector_roundtrip() {
        let dims: u16 = 384;
        let rows = 100usize;
        // Deterministic values: row r, dim d -> ((r * 7 + d) mod 251) - 120.
        // reason: intentional modular wrap to produce i8 values across the range
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let data: Vec<i8> = (0..rows * dims as usize)
            .map(|idx| (((idx * 7) % 251) as i64 - 120) as i8)
            .collect();
        let col = ColumnCodec::int8_vector(data.clone(), dims);
        assert_eq!(col.len(), rows);

        // Serialize and deserialize.
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len(), "read_from should consume the full buffer");
        assert_eq!(decoded.len(), rows);

        // Check representative indices after the round-trip.
        for &row in &[0usize, 1, 50, 99] {
            let decoded_slice = decoded.get_int8_vector(row).unwrap();
            let start = row * dims as usize;
            assert_eq!(decoded_slice, &data[start..start + dims as usize]);
            // Also verify the Value::List decoding path is consistent.
            let decoded_value = decoded.get(row).unwrap();
            if let Value::List(items) = decoded_value {
                assert_eq!(items.len(), dims as usize);
                assert_eq!(items[0], Value::Int64(i64::from(decoded_slice[0])));
            } else {
                panic!("expected Value::List for Int8Vector element");
            }
        }
    }

    /// Index-out-of-bounds and zero-dimensions edge cases for `Int8Vector`
    /// exercised together. Covers the early-return guards in both `get`
    /// (lines 62-70) and `get_int8_vector` (lines 100-110).
    #[test]
    fn test_column_vector_oob_and_zero_dim() {
        // OOB: 2 vectors of dim 3, index 5 is well past the end.
        let col = ColumnCodec::int8_vector(vec![1i8, 2, 3, 4, 5, 6], 3);
        assert_eq!(col.len(), 2);
        assert!(col.get(2).is_none());
        assert!(col.get(5).is_none());
        assert!(col.get_int8_vector(2).is_none());
        assert!(col.get_int8_vector(5).is_none());

        // Zero-dim column: len == 0 by construction and no element is accessible.
        let zero = ColumnCodec::int8_vector(Vec::new(), 0);
        assert_eq!(zero.len(), 0);
        assert!(zero.is_empty());
        assert!(zero.get(0).is_none());
        assert!(zero.get_int8_vector(0).is_none());
    }

    /// A `Dict` column queried with `Int64` bounds: both bounds are a type
    /// mismatch, so `find_in_range` takes the fallback path. Inside the
    /// fallback, `compare_values(String, Int64)` returns `None` for every row,
    /// so the result is empty. Covers the non-BitPacked entry into
    /// `find_in_range_fallback` with a type-mismatched range.
    #[test]
    fn test_find_in_range_incompatible_types() {
        let mut builder = DictionaryBuilder::new();
        for city in ["Amsterdam", "Berlin", "Paris", "Prague", "Barcelona"] {
            builder.add(city);
        }
        let col = ColumnCodec::Dict(builder.build());

        let result =
            col.find_in_range(Some(&Value::Int64(0)), Some(&Value::Int64(100)), true, true);
        assert!(
            result.is_empty(),
            "Int64 bounds on a Dict column should yield no matches"
        );
    }

    /// A buffer that is truncated mid-codec must return an error, never panic.
    /// Serializes a valid BitPacked column, then truncates the buffer at each
    /// interior byte and verifies `read_from` reports an error.
    #[test]
    fn test_column_serde_truncated_buffer() {
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&[1u64, 2, 3, 4, 5]));
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        assert!(buf.len() > 4);

        // Truncate to zero: missing discriminant.
        let mut pos = 0;
        assert!(ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&[]), &mut pos).is_err());

        // Truncate after the discriminant but before bits_per_value.
        let mut pos = 0;
        assert!(
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf[..1]), &mut pos).is_err()
        );

        // Truncate mid-header (before the row count u32 is complete).
        let mut pos = 0;
        assert!(
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf[..3]), &mut pos).is_err()
        );

        // Truncate mid-payload (drop the last byte of the packed u64 words).
        let mut pos = 0;
        assert!(
            ColumnCodec::read_from(
                &bytes::Bytes::copy_from_slice(&buf[..buf.len() - 1]),
                &mut pos
            )
            .is_err()
        );

        // Unknown discriminant: must return an error, not panic.
        let mut pos = 0;
        assert!(
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&[0xFFu8]), &mut pos).is_err()
        );

        // Truncated Int8Vector payload (dimensions OK, length promises more
        // bytes than exist). Build a minimal header: discriminant=3, dims=2,
        // data_len=4, but only provide 2 bytes of payload.
        let mut bad = vec![3u8];
        bad.extend_from_slice(&2u16.to_le_bytes());
        bad.extend_from_slice(&4u32.to_le_bytes());
        bad.extend_from_slice(&[0u8, 0u8]);
        let mut pos = 0;
        assert!(ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&bad), &mut pos).is_err());
    }

    // -----------------------------------------------------------------------
    // Serde round-trip tests: write_to + read_from
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_read_round_trip_bitpacked() {
        let bp = BitPackedInts::pack(&[3u64, 7, 12, 5]);
        let col = ColumnCodec::BitPacked(bp);

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len(), "read should consume entire buffer");

        assert_eq!(decoded.len(), 4);
        for i in 0..4 {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_read_round_trip_dict() {
        let mut b = DictionaryBuilder::new();
        for s in ["Amsterdam", "Berlin", "Amsterdam", "Paris"] {
            b.add(s);
        }
        let col = ColumnCodec::Dict(b.build());

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), 4);
        for i in 0..4 {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_read_round_trip_bitmap() {
        let bv = BitVector::from_bools(&[true, false, true, true, false, false, true]);
        let col = ColumnCodec::Bitmap(bv);

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), 7);
        for i in 0..7 {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_read_round_trip_int8_vector() {
        let data: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
        let col = ColumnCodec::int8_vector(data, 4);

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.get_int8_vector(0), Some(&[1i8, -2, 3, -4][..]));
        assert_eq!(decoded.get_int8_vector(1), Some(&[5i8, -6, 7, -8][..]));
    }

    // -----------------------------------------------------------------------
    // Serde error cases: truncation and unknown discriminants
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_from_empty_buffer_errors() {
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&[]), &mut pos).unwrap_err();
        assert_eq!(err, "truncated codec discriminant");
    }

    #[test]
    fn test_read_from_unknown_discriminant_errors() {
        // Discriminant values 0..=3 are valid; 99 is unknown.
        let buf = vec![99u8];
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "unknown codec discriminant");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_bits() {
        // Discriminant 0 (BitPacked) but no bits_per_value byte following.
        let buf = vec![0u8];
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated bits_per_value");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_count() {
        // Discriminant + bits byte, but truncated before u32 count.
        let buf = vec![0u8, 4, 0, 0]; // only 2 of 4 bytes for u32 count
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated u32");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_words() {
        // Discriminant + bits + count + data_len=2 but no u64 data words.
        let mut buf = vec![0u8, 4];
        buf.extend_from_slice(&1u32.to_le_bytes()); // count=1
        buf.extend_from_slice(&2u32.to_le_bytes()); // data_len=2
        // no u64 words
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated BitPacked data");
    }

    #[test]
    fn test_read_from_truncated_dict_string() {
        // Discriminant=1 (Dict), dict_len=1, slen=5, but only 3 bytes of string.
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len=1
        buf.extend_from_slice(&5u32.to_le_bytes()); // slen=5
        buf.extend_from_slice(b"abc"); // only 3 bytes, need 5
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated dict string");
    }

    #[test]
    fn test_read_from_invalid_utf8_in_dict() {
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len=1
        buf.extend_from_slice(&2u32.to_le_bytes()); // slen=2
        buf.extend_from_slice(&[0xFF, 0xFE]); // invalid UTF-8
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "invalid UTF-8 in dict");
    }

    #[test]
    fn test_read_from_truncated_bitmap_words() {
        // Discriminant=2 (Bitmap), bit_len=64, data_len=1, but no u64 data.
        let mut buf = vec![2u8];
        buf.extend_from_slice(&64u32.to_le_bytes()); // bit_len
        buf.extend_from_slice(&1u32.to_le_bytes()); // data_len
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated Bitmap data");
    }

    #[test]
    fn test_read_from_truncated_int8_vector_dimensions() {
        // Discriminant=3 (Int8Vector), only 1 byte of the 2-byte dimensions.
        let buf = vec![3u8, 0];
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated u16");
    }

    #[test]
    fn test_read_from_truncated_int8_vector_data() {
        // Discriminant=3, dimensions=2, data_len=4, but only 2 data bytes.
        let mut buf = vec![3u8];
        buf.extend_from_slice(&2u16.to_le_bytes()); // dimensions=2
        buf.extend_from_slice(&4u32.to_le_bytes()); // data_len=4
        buf.extend_from_slice(&[10u8, 20]); // only 2 of 4 bytes
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated Int8Vector data");
    }

    // -----------------------------------------------------------------------
    // Empty (zero-length) columns: all codec variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_bitpacked_round_trip() {
        let bp = BitPackedInts::pack(&[]);
        let col = ColumnCodec::BitPacked(bp);
        assert!(col.is_empty());
        assert_eq!(col.len(), 0);

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_empty_dict_round_trip() {
        let builder = DictionaryBuilder::new();
        let dict = builder.build();
        let col = ColumnCodec::Dict(dict);
        assert!(col.is_empty());

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_empty_bitmap_round_trip() {
        let bv = BitVector::from_bools(&[]);
        let col = ColumnCodec::Bitmap(bv);
        assert!(col.is_empty());

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_empty_int8_vector_round_trip() {
        let col = ColumnCodec::int8_vector(Vec::new(), 4);
        assert!(col.is_empty());

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_empty_string_in_dict() {
        let mut b = DictionaryBuilder::new();
        b.add("");
        b.add("Alix");
        b.add("");
        let col = ColumnCodec::Dict(b.build());

        assert_eq!(col.get(0), Some(Value::String(ArcStr::from(""))));
        assert_eq!(col.get(1), Some(Value::String(ArcStr::from("Alix"))));
        assert_eq!(col.get(2), Some(Value::String(ArcStr::from(""))));

        // Round-trip to exercise the len=0 branch of write/read dict string.
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap();
        assert_eq!(decoded.get(0), Some(Value::String(ArcStr::from(""))));
        assert_eq!(decoded.get(2), Some(Value::String(ArcStr::from(""))));
    }

    // -----------------------------------------------------------------------
    // find_eq / find_in_range boundary exactness
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_in_range_exact_boundaries_inclusive_vs_exclusive() {
        // values: 10, 20, 30, 40, 50
        let values = vec![10u64, 20, 30, 40, 50];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // [20, 40] inclusive on both ends, boundary values included.
        let inclusive =
            col.find_in_range(Some(&Value::Int64(20)), Some(&Value::Int64(40)), true, true);
        assert_eq!(inclusive, vec![1, 2, 3]);

        // (20, 40) exclusive on both ends, boundaries excluded.
        let exclusive = col.find_in_range(
            Some(&Value::Int64(20)),
            Some(&Value::Int64(40)),
            false,
            false,
        );
        assert_eq!(exclusive, vec![2]);

        // [20, 40) inclusive-exclusive mix.
        let mixed_a = col.find_in_range(
            Some(&Value::Int64(20)),
            Some(&Value::Int64(40)),
            true,
            false,
        );
        assert_eq!(mixed_a, vec![1, 2]);

        // (20, 40] exclusive-inclusive mix.
        let mixed_b = col.find_in_range(
            Some(&Value::Int64(20)),
            Some(&Value::Int64(40)),
            false,
            true,
        );
        assert_eq!(mixed_b, vec![2, 3]);
    }

    #[test]
    fn test_find_in_range_bitpacked_fallback_on_float_min() {
        // Float min triggers the fallback path on BitPacked. The fallback
        // still knows how to compare Int64 <-> Float64, so values >= 2.5
        // out of [1, 2, 3] match.
        let values = vec![1u64, 2, 3];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        let result = col.find_in_range(Some(&Value::Float64(2.5)), None, true, true);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_find_in_range_bitpacked_fallback_on_float_max() {
        let values = vec![1u64, 2, 3];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        // Float max also routes through the fallback, comparing Int <-> Float.
        let result = col.find_in_range(None, Some(&Value::Float64(2.5)), true, true);
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_find_in_range_open_both_ends_returns_all() {
        let values = vec![1u64, 2, 3, 4, 5];
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));

        let all = col.find_in_range(None, None, true, true);
        assert_eq!(all, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_find_in_range_fallback_dict_exclusive() {
        let mut b = DictionaryBuilder::new();
        for name in ["Amsterdam", "Berlin", "Paris", "Prague"] {
            b.add(name);
        }
        let col = ColumnCodec::Dict(b.build());

        // Exclusive on lower bound: "Berlin" excluded.
        let result = col.find_in_range(Some(&Value::String("Berlin".into())), None, false, true);
        assert_eq!(result, vec![2, 3]); // Paris, Prague

        // Exclusive on upper bound: "Prague" excluded.
        let result = col.find_in_range(None, Some(&Value::String("Prague".into())), true, false);
        assert_eq!(result, vec![0, 1, 2]); // Amsterdam, Berlin, Paris
    }

    #[test]
    fn test_find_in_range_fallback_mismatch_returns_none_for_row() {
        // Target uses None from compare_values by mixing codec types.
        // Bitmap column + Int64 range -> compare returns None -> rows excluded.
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&[true, false, true]));
        let result = col.find_in_range(Some(&Value::Int64(0)), Some(&Value::Int64(5)), true, true);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // get_raw_u64 / get_int8_vector on all non-matching variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_raw_u64_returns_none_for_all_non_bitpacked() {
        let mut b = DictionaryBuilder::new();
        b.add("x");
        assert_eq!(ColumnCodec::Dict(b.build()).get_raw_u64(0), None);

        assert_eq!(ColumnCodec::int8_vector(vec![1i8], 1).get_raw_u64(0), None);
    }

    #[test]
    fn test_get_int8_vector_returns_none_for_all_non_vector() {
        let bp = BitPackedInts::pack(&[1u64]);
        assert_eq!(ColumnCodec::BitPacked(bp).get_int8_vector(0), None);

        let mut b = DictionaryBuilder::new();
        b.add("x");
        assert_eq!(ColumnCodec::Dict(b.build()).get_int8_vector(0), None);

        let bv = BitVector::from_bools(&[true]);
        assert_eq!(ColumnCodec::Bitmap(bv).get_int8_vector(0), None);
    }

    // -----------------------------------------------------------------------
    // heap_bytes on empty columns
    // -----------------------------------------------------------------------

    #[test]
    fn test_heap_bytes_empty_columns() {
        let bp = BitPackedInts::pack(&[]);
        assert_eq!(ColumnCodec::BitPacked(bp).heap_bytes(), 0);

        let builder = DictionaryBuilder::new();
        assert_eq!(ColumnCodec::Dict(builder.build()).heap_bytes(), 0);

        let col = ColumnCodec::int8_vector(Vec::new(), 4);
        assert_eq!(col.heap_bytes(), 0);
    }

    // -----------------------------------------------------------------------
    // find_eq for Dict where target string is not in the dictionary
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_eq_dict_target_not_in_dictionary() {
        let mut b = DictionaryBuilder::new();
        b.add("Amsterdam");
        b.add("Berlin");
        let col = ColumnCodec::Dict(b.build());

        // Target "Prague" does not exist in the dictionary, so encode returns None.
        let result = col.find_eq(&Value::String(ArcStr::from("Prague")));
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Accessor coverage for non-matching variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_raw_u64_on_dict_and_int8_vector_returns_none() {
        // Dict variant: get_raw_u64 is meaningless.
        let mut builder = DictionaryBuilder::new();
        builder.add("Vincent");
        let dict_col = ColumnCodec::Dict(builder.build());
        assert_eq!(dict_col.get_raw_u64(0), None);

        // Int8Vector variant: get_raw_u64 is meaningless.
        let vec_col = ColumnCodec::int8_vector(vec![1i8, 2, 3], 3);
        assert_eq!(vec_col.get_raw_u64(0), None);
    }

    #[test]
    fn test_get_int8_vector_on_dict_and_bitmap_returns_none() {
        // Dict variant: not an Int8Vector.
        let mut builder = DictionaryBuilder::new();
        builder.add("Jules");
        let dict_col = ColumnCodec::Dict(builder.build());
        assert_eq!(dict_col.get_int8_vector(0), None);

        // Bitmap variant: not an Int8Vector.
        let bm_col = ColumnCodec::Bitmap(BitVector::from_bools(&[true, false]));
        assert_eq!(bm_col.get_int8_vector(0), None);
    }

    // -----------------------------------------------------------------------
    // find_in_range fallback branches: Dict exclusive/open-ended, Equal with
    // !inclusive, and uncomparable (None) values.
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_in_range_dict_exclusive_bounds() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Amsterdam", "Berlin", "Paris", "Prague"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        // (Amsterdam, Prague) exclusive should yield Berlin and Paris.
        let result = col.find_in_range(
            Some(&Value::String("Amsterdam".into())),
            Some(&Value::String("Prague".into())),
            false,
            false,
        );
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_find_in_range_dict_open_bounds() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Amsterdam", "Berlin", "Paris", "Prague"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        // No lower bound, upper inclusive = Berlin.
        let result = col.find_in_range(None, Some(&Value::String("Berlin".into())), true, true);
        assert_eq!(result, vec![0, 1]);

        // Lower inclusive = Paris, no upper bound.
        let result = col.find_in_range(Some(&Value::String("Paris".into())), None, true, true);
        assert_eq!(result, vec![2, 3]);
    }

    #[test]
    fn test_find_in_range_fallback_uncomparable_skips_rows() {
        // Int8Vector rows compare to None against any Value, so the fallback
        // filters them out when any bound is supplied.
        let data = vec![1i8, 2, 3];
        let col = ColumnCodec::int8_vector(data, 3);
        let min = Value::Int64(0);
        let max = Value::Int64(10);

        // min alone: the None (Uncomparable) branch returns false.
        let result = col.find_in_range(Some(&min), None, true, true);
        assert!(result.is_empty());

        // max alone: same story for the max arm.
        let result = col.find_in_range(None, Some(&max), true, true);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // Serialization round-trips for each codec variant
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_to_read_from_bitpacked_round_trip() {
        let values = vec![0u64, 5, 10, 15, 3, 7];
        let bp = BitPackedInts::pack(&values);
        let col = ColumnCodec::BitPacked(bp);

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_to_read_from_dict_round_trip() {
        let mut builder = DictionaryBuilder::new();
        for name in ["Vincent", "Jules", "Vincent", "Mia"] {
            builder.add(name);
        }
        let col = ColumnCodec::Dict(builder.build());

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_to_read_from_bitmap_round_trip() {
        let bools = vec![true, false, true, true, false, false, true];
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&bools));

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_to_read_from_int8_vector_round_trip() {
        // 3 vectors of dimension 4, mixing positive and negative values.
        let data = vec![1i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12];
        let col = ColumnCodec::int8_vector(data, 4);

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get_int8_vector(i), col.get_int8_vector(i));
        }
    }

    // -----------------------------------------------------------------------
    // read_from error paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_from_truncated_discriminant() {
        let data: &[u8] = &[];
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(data), &mut pos).unwrap_err();
        assert_eq!(err, "truncated codec discriminant");
    }

    #[test]
    fn test_read_from_unknown_discriminant() {
        let data: &[u8] = &[42];
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(data), &mut pos).unwrap_err();
        assert_eq!(err, "unknown codec discriminant");
    }

    #[test]
    fn test_read_from_truncated_bits_per_value() {
        // Discriminant 0 (BitPacked) with no following byte for bits_per_value.
        let data: &[u8] = &[0];
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(data), &mut pos).unwrap_err();
        assert_eq!(err, "truncated bits_per_value");
    }

    #[test]
    fn test_read_from_truncated_bitpacked_word() {
        // Discriminant 0 + bits_per_value=4 + count=1 + data_len=1, then
        // truncated 8-byte word.
        let mut buf = vec![0u8, 4];
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0u8, 0, 0]); // only 3 bytes of the 8-byte word
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated BitPacked data");
    }

    #[test]
    fn test_read_from_dict_truncated_string() {
        // Discriminant 1 (Dict) + dict_len=1 + slen=5, but no string bytes follow.
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len = 1
        buf.extend_from_slice(&5u32.to_le_bytes()); // slen = 5
        // Only add 2 of the 5 needed bytes.
        buf.extend_from_slice(b"ab");
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated dict string");
    }

    #[test]
    fn test_read_from_dict_invalid_utf8() {
        // Discriminant 1 + dict_len=1 + slen=2 + invalid UTF-8 bytes.
        let mut buf = vec![1u8];
        buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len = 1
        buf.extend_from_slice(&2u32.to_le_bytes()); // slen = 2
        buf.extend_from_slice(&[0xFFu8, 0xFE]); // invalid UTF-8
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "invalid UTF-8 in dict");
    }

    #[test]
    fn test_read_from_int8_vector_truncated_data() {
        // Discriminant 3 + dimensions=2 + data_len=6, but only 3 bytes follow.
        let mut buf = vec![3u8];
        buf.extend_from_slice(&2u16.to_le_bytes()); // dimensions = 2
        buf.extend_from_slice(&6u32.to_le_bytes()); // data_len = 6
        buf.extend_from_slice(&[1u8, 2, 3]);
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated Int8Vector data");
    }

    #[test]
    fn test_read_from_int8_vector_truncated_dimensions() {
        // Discriminant 3 + only 1 byte (u16 needs 2).
        let buf = vec![3u8, 0];
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated u16");
    }

    #[test]
    fn test_read_from_bitmap_truncated() {
        // Discriminant 2 + truncated u32 (bit_len).
        let buf = vec![2u8, 0, 0];
        let mut pos = 0;
        let err =
            ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos).unwrap_err();
        assert_eq!(err, "truncated u32");
    }

    // -----------------------------------------------------------------------
    // Empty-column round-trips
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_to_read_from_empty_bitpacked() {
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&[]));
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert_eq!(decoded.len(), 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_write_to_read_from_empty_bitmap() {
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&[]));
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_write_to_read_from_empty_int8_vector() {
        let col = ColumnCodec::int8_vector(Vec::new(), 4);
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert_eq!(decoded.len(), 0);
    }

    // -----------------------------------------------------------------------
    // RawI64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_raw_i64_get_decodes_as_int64() {
        let col = ColumnCodec::raw_i64(vec![-100, 0, 42, i64::MIN, i64::MAX]);
        assert_eq!(col.len(), 5);
        assert_eq!(col.get(0), Some(Value::Int64(-100)));
        assert_eq!(col.get(1), Some(Value::Int64(0)));
        assert_eq!(col.get(2), Some(Value::Int64(42)));
        assert_eq!(col.get(3), Some(Value::Int64(i64::MIN)));
        assert_eq!(col.get(4), Some(Value::Int64(i64::MAX)));
        assert_eq!(col.get(5), None);
    }

    #[test]
    fn test_raw_i64_find_eq() {
        let col = ColumnCodec::raw_i64(vec![-50, 10, -50, 20, 0, -50]);
        assert_eq!(col.find_eq(&Value::Int64(-50)), vec![0, 2, 5]);
        assert_eq!(col.find_eq(&Value::Int64(10)), vec![1]);
        assert_eq!(col.find_eq(&Value::Int64(0)), vec![4]);
        assert_eq!(col.find_eq(&Value::Int64(999)), Vec::<usize>::new());
        // Cross-type matches are not supported: a Float64 target should
        // not match any i64 row via the fast path. Fallback may coerce,
        // but for now the equality predicate is type-strict.
        assert_eq!(col.find_eq(&Value::Float64(10.0)), Vec::<usize>::new());
    }

    #[test]
    fn test_raw_i64_find_in_range_signed_ordering() {
        // Values span the sign boundary; native i64 ordering must apply.
        let col = ColumnCodec::raw_i64(vec![-10, -5, 0, 5, 10, -100, 100]);

        // [-5, 5] inclusive
        let result = col.find_in_range(Some(&Value::Int64(-5)), Some(&Value::Int64(5)), true, true);
        assert_eq!(result, vec![1, 2, 3]);

        // (-5, 5) exclusive
        let result = col.find_in_range(
            Some(&Value::Int64(-5)),
            Some(&Value::Int64(5)),
            false,
            false,
        );
        assert_eq!(result, vec![2]);

        // < 0 (no lower bound)
        let result = col.find_in_range(None, Some(&Value::Int64(0)), false, false);
        assert_eq!(result, vec![0, 1, 5]);

        // >= 10 (no upper bound)
        let result = col.find_in_range(Some(&Value::Int64(10)), None, true, true);
        assert_eq!(result, vec![4, 6]);
    }

    #[test]
    fn test_write_to_read_from_raw_i64_round_trip() {
        let col = ColumnCodec::raw_i64(vec![-42, 0, 1, i64::MIN, i64::MAX, -1_000_000_000]);

        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();

        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert_eq!(pos, buf.len());
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i));
        }
    }

    #[test]
    fn test_write_to_read_from_empty_raw_i64() {
        let col = ColumnCodec::raw_i64(Vec::new());
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let decoded = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("decode should succeed");
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn test_raw_i64_heap_bytes() {
        let col = ColumnCodec::raw_i64(vec![-1, 2, -3]);
        assert_eq!(col.heap_bytes(), 3 * std::mem::size_of::<i64>());

        let empty = ColumnCodec::raw_i64(Vec::new());
        assert_eq!(empty.heap_bytes(), 0);
    }

    // ── Phase 2a: Block API ────────────────────────────────────────────
    //
    // Phase 2a treats every column as a single block (block_count == 1).
    // Phase 2b will introduce multi-block layouts. These tests pin the
    // contract so the API is stable before the format changes.

    #[test]
    fn alix_block_count_is_one_for_every_codec() {
        // BitPacked
        let bp = ColumnCodec::BitPacked(BitPackedInts::pack(&[1, 2, 3]));
        assert_eq!(bp.block_count(), 1);

        // Dict
        let mut b = DictionaryBuilder::new();
        b.add("x");
        let dict = ColumnCodec::Dict(b.build());
        assert_eq!(dict.block_count(), 1);

        // Bitmap
        let bm = ColumnCodec::Bitmap(BitVector::from_bools(&[true, false]));
        assert_eq!(bm.block_count(), 1);

        // Int8Vector
        let iv = ColumnCodec::int8_vector(vec![1i8, 2, 3, 4], 2);
        assert_eq!(iv.block_count(), 1);

        // Float64
        let f64 = ColumnCodec::float64(vec![1.0, 2.0]);
        assert_eq!(f64.block_count(), 1);

        // Float32Vector
        let fv = ColumnCodec::float32_vector(vec![1.0f32, 2.0, 3.0, 4.0], 2);
        assert_eq!(fv.block_count(), 1);

        // RawI64
        let r64 = ColumnCodec::raw_i64(vec![-1, 2, -3]);
        assert_eq!(r64.block_count(), 1);
    }

    #[test]
    fn gus_block_at_zero_carries_full_row_count() {
        let bp = ColumnCodec::BitPacked(BitPackedInts::pack(&[1, 2, 3, 4, 5]));
        let entry = bp.block_at(0).expect("block 0 exists");
        assert_eq!(entry.row_count, 5);

        let r64 = ColumnCodec::raw_i64(vec![-1, 2, -3, 4]);
        let entry = r64.block_at(0).expect("block 0 exists");
        assert_eq!(entry.row_count, 4);
    }

    #[test]
    fn vincent_empty_column_has_one_zero_row_block() {
        // Phase 2a convention: empty columns still report block_count == 1
        // so downstream serializers can treat block emission uniformly.
        let empty = ColumnCodec::raw_i64(Vec::new());
        assert_eq!(empty.block_count(), 1);
        let entry = empty.block_at(0).expect("zero-row block at 0");
        assert_eq!(entry.row_count, 0);
    }

    #[test]
    fn jules_block_at_out_of_bounds_returns_none() {
        let bp = ColumnCodec::BitPacked(BitPackedInts::pack(&[1, 2, 3]));
        assert!(bp.block_at(1).is_none());
        assert!(bp.block_at(usize::MAX).is_none());
    }

    #[test]
    fn mia_block_iter_yields_block_count_entries() {
        let r64 = ColumnCodec::raw_i64(vec![-1, 2, -3, 4]);
        let entries: Vec<_> = r64.block_iter().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_count, 4);
    }

    #[test]
    fn butch_block_row_counts_sum_to_column_len() {
        // Phase 2a: trivially true (one block, row_count == len).
        // Phase 2b: this contract is the migration test — multi-block
        // serializers must preserve total row count.
        for col in [
            ColumnCodec::BitPacked(BitPackedInts::pack(&[1, 2, 3, 4, 5, 6, 7])),
            ColumnCodec::float64(vec![1.0, 2.0, 3.0]),
            ColumnCodec::raw_i64(vec![-1, 2, -3, 4, -5]),
        ] {
            let total: u32 = col.block_iter().map(|b| b.row_count).sum();
            assert_eq!(total as usize, col.len());
        }
    }

    // ── Phase 2b: v2 multi-block on-disk format ───────────────────────
    //
    // v2 layout per column (after the section header):
    //   [disc:u8][global_params...]
    //   [block_count:u32]
    //   [block_index: block_count x (byte_offset:u32, byte_len:u32, row_count:u32)]
    //   [block_data: concatenated per-block bodies]
    //
    // The runtime API (block_count(), block_at(), len(), get(...)) is
    // unchanged after a round-trip — only the on-disk layout differs.

    fn assert_round_trip_v2_equals(col: &ColumnCodec) {
        let mut buf = Vec::new();
        col.write_to_v2(&mut buf).unwrap();
        let mut pos = 0;
        let recovered = ColumnCodec::read_from_v2(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("v2 round-trip");
        assert_eq!(pos, buf.len(), "v2 reader should consume entire buffer");
        assert_eq!(recovered.len(), col.len(), "len after v2 round-trip");
        for i in 0..col.len() {
            assert_eq!(
                recovered.get(i),
                col.get(i),
                "value at row {i} after v2 round-trip"
            );
        }
    }

    #[test]
    fn django_v2_round_trip_raw_i64_single_block() {
        let col = ColumnCodec::raw_i64(vec![-1, 2, -3, 4, -5]);
        assert_eq!(col.block_count(), 1);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_raw_i64_multi_block() {
        // 2049 rows: 1024 + 1024 + 1 → 3 blocks at default block size.
        // reason: i is bounded by 2049 which fits in i64.
        #[allow(clippy::cast_possible_wrap)]
        let values: Vec<i64> = (0..2049i64).map(|i| i - 1024).collect();
        let col = ColumnCodec::raw_i64(values);
        assert_eq!(col.block_count(), 3);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_bitpacked_multi_block() {
        let values: Vec<u64> = (0..2500u64).map(|i| i % 16).collect();
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&values));
        assert!(col.block_count() >= 2, "expect multi-block at 2500 rows");
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_dict_multi_block() {
        let mut b = DictionaryBuilder::new();
        for i in 0..1500u32 {
            b.add(if i % 3 == 0 {
                "alpha"
            } else if i % 3 == 1 {
                "beta"
            } else {
                "gamma"
            });
        }
        let col = ColumnCodec::Dict(b.build());
        assert_eq!(col.block_count(), 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_bitmap_multi_block() {
        let bools: Vec<bool> = (0..1100u32).map(|i| i % 2 == 0).collect();
        let col = ColumnCodec::Bitmap(BitVector::from_bools(&bools));
        assert!(col.block_count() >= 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_float64_multi_block() {
        let vals: Vec<f64> = (0..1100u32).map(|i| f64::from(i) * 0.5).collect();
        let col = ColumnCodec::float64(vals);
        assert!(col.block_count() >= 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_int8_vector_multi_block() {
        // 1100 vectors of dim 4 = 4400 i8 entries.
        // reason: known small integer literals
        #[allow(clippy::cast_possible_wrap)]
        let data: Vec<i8> = (0..4400u32).map(|i| (i % 200) as i8).collect();
        let col = ColumnCodec::int8_vector(data, 4);
        assert!(col.block_count() >= 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn django_v2_round_trip_float32_vector_multi_block() {
        // 1100 vectors of dim 4 = 4400 f32 entries.
        let data: Vec<f32> = (0..4400u32).map(|i| i as f32 * 0.5).collect();
        let col = ColumnCodec::float32_vector(data, 4);
        assert!(col.block_count() >= 2);
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn shosanna_v2_round_trip_empty_column() {
        // Empty columns should still serialize/deserialize cleanly with v2.
        let col = ColumnCodec::raw_i64(Vec::new());
        assert_round_trip_v2_equals(&col);
    }

    #[test]
    fn beatrix_v2_block_index_with_gap_is_rejected() {
        // Build a v2 buffer for a 2-block RawI64 column, then poke the
        // second block's byte_offset to leave a 16-byte gap. The reader
        // must reject this rather than zero-copy slice into the gap and
        // misinterpret bodies.
        let col =
            ColumnCodec::raw_i64((0..(crate::codec::DEFAULT_BLOCK_ROWS as i64) + 4).collect());
        assert!(col.block_count() >= 2, "need a multi-block column");
        let mut buf = Vec::new();
        col.write_to_v2(&mut buf).unwrap();

        // Layout of the block index for RawI64:
        //   [discriminant:1][block_count:4][meta0:12][meta1:12]...
        // discriminant is at index 0, block_count at index 1..5,
        // first meta at 5..17, second meta at 17..29. Patch the second
        // meta's byte_offset (the first u32 of meta1) to skip past where
        // it should start.
        let original_offset = u32::from_le_bytes(buf[17..21].try_into().unwrap());
        let bumped = original_offset + 16;
        buf[17..21].copy_from_slice(&bumped.to_le_bytes());

        let mut pos = 0;
        let result = ColumnCodec::read_from_v2(&bytes::Bytes::copy_from_slice(&buf), &mut pos);
        assert!(
            result.is_err(),
            "reader must reject block index with a gap, got {result:?}"
        );
    }

    #[test]
    fn shosanna_v2_block_index_with_overlap_is_rejected() {
        // Build a v2 buffer for a 2-block Float64 column, then make the
        // second block's byte_offset overlap the first by setting it to 0.
        let col = ColumnCodec::float64(
            (0..(crate::codec::DEFAULT_BLOCK_ROWS as i64) + 8)
                .map(|i| i as f64)
                .collect(),
        );
        assert!(col.block_count() >= 2);
        let mut buf = Vec::new();
        col.write_to_v2(&mut buf).unwrap();

        // Float64 has the same per-block index layout as RawI64; second
        // meta's byte_offset is at buf[17..21]. Force overlap.
        buf[17..21].copy_from_slice(&0u32.to_le_bytes());

        let mut pos = 0;
        let result = ColumnCodec::read_from_v2(&bytes::Bytes::copy_from_slice(&buf), &mut pos);
        assert!(
            result.is_err(),
            "reader must reject overlapping block index, got {result:?}"
        );
    }

    #[test]
    fn vincent_zero_dimension_float32_vector_v2_round_trip() {
        // Zero-dimension Float32Vector must serialize as an empty column
        // (no body bytes, block index reports zero rows). The writer used
        // to .max(1) the row stride and emit `bytes.len()` rows, leaving
        // the block index inconsistent with `len()`.
        let col = ColumnCodec::float32_vector(Vec::new(), 0);
        assert_eq!(col.len(), 0);
        assert_round_trip_v2_equals(&col);

        let mut buf = Vec::new();
        col.write_to_v2(&mut buf).unwrap();
        let mut pos = 0;
        let recovered = ColumnCodec::read_from_v2(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("v2 round-trip");
        assert_eq!(recovered.len(), 0);
    }

    #[test]
    fn hans_v1_and_v2_produce_different_bytes() {
        // Sanity: the v1 (flat) and v2 (block-indexed) layouts differ
        // even for a small single-block column. This pins the format
        // separation so a future "did we forget to switch?" regression
        // is caught.
        let col = ColumnCodec::raw_i64(vec![1, 2, 3, 4, 5]);
        let mut v1 = Vec::new();
        col.write_to(&mut v1).unwrap();
        let mut v2 = Vec::new();
        col.write_to_v2(&mut v2).unwrap();
        assert_ne!(v1, v2, "v1 and v2 layouts must differ");
    }

    #[test]
    fn beatrix_v1_round_trip_still_works() {
        // v1 (flat) reader/writer must keep working for one release as
        // the section reader's compat path. We round-trip via the
        // unchanged write_to / read_from pair.
        let col = ColumnCodec::raw_i64(vec![-1, 2, -3, 4, -5]);
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let recovered = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("v1 round-trip");
        assert_eq!(recovered.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(recovered.get(i), col.get(i));
        }
    }

    // ── v3 round-trips for fixed-width bytes-backed codecs ────────────
    //
    // v3 extends v2's block index with an inline ZoneMap per block. The
    // body layout is identical to v2, so RawI64 / Float64 fixed-width
    // bodies are zero-copy slices of the original buffer. These tests
    // pin the v3 reader paths for those two codecs (Task 8 reworked
    // them) so a future regression in the writer/reader pair is caught
    // element-wise rather than only via the cross-variant find_eq
    // smoke test.

    #[test]
    fn test_column_codec_raw_i64_v3_round_trip() {
        // Mix of negative, positive, and edge values across 200 rows.
        let values: Vec<i64> = (0..200i64).map(|i| (i * 7 - 100) % 991).collect();
        let col = ColumnCodec::raw_i64(values);

        let mut buf = Vec::new();
        col.write_to_v3(&mut buf, None).unwrap();

        let mut pos = 0;
        let (decoded, _stats) =
            ColumnCodec::read_from_v3(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
                .expect("v3 round-trip");
        assert_eq!(pos, buf.len(), "v3 reader should consume entire buffer");
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i), "value at row {i}");
        }

        // Pick a target known to occur in the generated sequence and
        // verify find_eq agrees on both forms.
        let target = col.get(42).expect("row 42 exists");
        assert_eq!(decoded.find_eq(&target), col.find_eq(&target));
        assert!(!col.find_eq(&target).is_empty());
    }

    #[test]
    fn test_column_codec_float64_v3_round_trip() {
        let values: Vec<f64> = (0..200u32)
            .map(|i| f64::from(i) * std::f64::consts::PI)
            .collect();
        let col = ColumnCodec::float64(values);

        let mut buf = Vec::new();
        col.write_to_v3(&mut buf, None).unwrap();

        let mut pos = 0;
        let (decoded, _stats) =
            ColumnCodec::read_from_v3(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
                .expect("v3 round-trip");
        assert_eq!(pos, buf.len(), "v3 reader should consume entire buffer");
        assert_eq!(decoded.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(decoded.get(i), col.get(i), "value at row {i}");
        }

        // Row 0 is 0.0 * PI == 0.0; check find_eq agreement on a value
        // we know is present.
        let target = Value::Float64(0.0);
        assert_eq!(decoded.find_eq(&target), col.find_eq(&target));
        assert!(!col.find_eq(&target).is_empty());
    }

    // ── Phase 3a: Bytes-backed fixed-width codecs ─────────────────────

    #[test]
    fn test_raw_i64_constructor_round_trip() {
        let values = vec![-100i64, -1, 0, 1, 100, i64::MIN, i64::MAX];
        let col = ColumnCodec::raw_i64(values.clone());

        assert_eq!(col.len(), values.len());
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(col.get(i), Some(Value::Int64(expected)));
        }
        assert_eq!(col.get(values.len()), None);
    }

    #[test]
    fn test_float64_constructor_round_trip() {
        let values = vec![-1.5_f64, 0.0, 100.25, f64::MIN, f64::MAX];
        let col = ColumnCodec::float64(values.clone());

        assert_eq!(col.len(), values.len());
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(col.get(i), Some(Value::Float64(expected)));
        }
    }

    #[test]
    fn test_int8_vector_constructor_round_trip() {
        let col = ColumnCodec::int8_vector(vec![1i8, 2, 3, -4, -5, -6], 3);

        assert_eq!(col.len(), 2);
        let v0 = col.get(0).unwrap();
        let expected0: Vec<Value> = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        assert_eq!(v0, Value::List(Arc::from(expected0)));

        // get_int8_vector slice access
        assert_eq!(col.get_int8_vector(0), Some(&[1i8, 2, 3][..]));
        assert_eq!(col.get_int8_vector(1), Some(&[-4i8, -5, -6][..]));
    }

    #[test]
    fn test_float32_vector_constructor_round_trip() {
        let col = ColumnCodec::float32_vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 3);

        assert_eq!(col.len(), 2);
        match col.get(0) {
            Some(Value::Vector(v)) => {
                assert_eq!(&*v, &[1.0_f32, 2.0, 3.0]);
            }
            other => panic!("expected Vector, got {other:?}"),
        }
    }

    #[test]
    fn test_bytes_backed_zero_copy_clone() {
        // Cloning a ColumnCodec should be cheap: the underlying Bytes
        // is refcounted, not deep-copied. Verify by building a large
        // RawI64 column and asserting clone is fast and shares storage
        // (smoke test via len equivalence; full reference-counting is
        // tested by `bytes` itself).
        let big: Vec<i64> = (0..10_000).collect();
        let col = ColumnCodec::raw_i64(big);
        let cloned = col.clone();
        assert_eq!(col.len(), cloned.len());
        for i in (0..10_000).step_by(1024) {
            assert_eq!(col.get(i), cloned.get(i));
        }
    }

    #[test]
    fn test_raw_i64_v1_round_trip_with_bytes_storage() {
        let col = ColumnCodec::raw_i64(vec![-7, 0, 7, 42]);
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let recovered = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("v1 round-trip");
        assert_eq!(recovered.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(recovered.get(i), col.get(i));
        }
    }

    #[test]
    fn test_float64_v1_round_trip_with_bytes_storage() {
        let col = ColumnCodec::float64(vec![-2.5, 0.0, 1.0, std::f64::consts::PI]);
        let mut buf = Vec::new();
        col.write_to(&mut buf).unwrap();
        let mut pos = 0;
        let recovered = ColumnCodec::read_from(&bytes::Bytes::copy_from_slice(&buf), &mut pos)
            .expect("v1 round-trip");
        assert_eq!(recovered.len(), col.len());
        for i in 0..col.len() {
            assert_eq!(recovered.get(i), col.get(i));
        }
    }

    // ── Phase 4a: range_iter (lazy block-skip iterator) ───────────────
    //
    // `range_iter` walks per-block zone maps (Phase 2c/2d) and skips
    // blocks whose stats prove no match, then evaluates the predicate
    // per row within matching blocks. These tests pin equivalence with
    // the existing eager `find_in_range`, plus pruning behavior on
    // multi-block columns.

    use crate::graph::compact::zone_map::compute_block_zone_maps;

    fn raw_i64_seq(n: i64) -> ColumnCodec {
        ColumnCodec::raw_i64((0..n).collect())
    }

    #[test]
    fn alix_range_iter_matches_find_in_range_full_scan() {
        let col = raw_i64_seq(50);
        let zm = compute_block_zone_maps(&col);
        let min = Value::Int64(10);
        let max = Value::Int64(20);

        let from_iter: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), Some(&max), true, true)
            .collect();
        let eager = col.find_in_range(Some(&min), Some(&max), true, true);

        assert_eq!(from_iter, eager);
    }

    #[test]
    fn gus_range_iter_skips_disjoint_blocks() {
        // 3 blocks at DEFAULT_BLOCK_ROWS == 1024:
        //   block 0: rows 0..1024,    values 0..1024
        //   block 1: rows 1024..2048, values 1024..2048
        //   block 2: rows 2048..3072, values 2048..3072
        // Query [1500, 1700] hits block 1 only; blocks 0 and 2 must be
        // skipped (their zone maps prove disjointedness).
        let col = raw_i64_seq(3072);
        let zm = compute_block_zone_maps(&col);
        let min = Value::Int64(1500);
        let max = Value::Int64(1700);

        let from_iter: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), Some(&max), true, true)
            .collect();

        let expected: Vec<usize> = (1500..=1700).collect();
        assert_eq!(from_iter, expected);
    }

    #[test]
    fn vincent_range_iter_open_min_bound() {
        let col = raw_i64_seq(50);
        let zm = compute_block_zone_maps(&col);
        let max = Value::Int64(10);
        let result: Vec<usize> = col
            .range_iter(Some(&zm), None, Some(&max), false, true)
            .collect();
        let expected: Vec<usize> = (0..=10).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn jules_range_iter_open_max_bound() {
        let col = raw_i64_seq(20);
        let zm = compute_block_zone_maps(&col);
        let min = Value::Int64(15);
        let result: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), None, true, false)
            .collect();
        let expected: Vec<usize> = (15..20).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn mia_range_iter_no_zone_maps_falls_back_to_full_scan() {
        // When block zone maps are unavailable, range_iter must still
        // produce correct results (just without skip pruning).
        let col = raw_i64_seq(100);
        let min = Value::Int64(40);
        let max = Value::Int64(60);
        let result: Vec<usize> = col
            .range_iter(None, Some(&min), Some(&max), true, true)
            .collect();
        let expected = col.find_in_range(Some(&min), Some(&max), true, true);
        assert_eq!(result, expected);
    }

    #[test]
    fn butch_range_iter_empty_column_yields_nothing() {
        let col = raw_i64_seq(0);
        let zm = compute_block_zone_maps(&col);
        let min = Value::Int64(0);
        let result: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), None, true, false)
            .collect();
        assert!(result.is_empty());
    }

    #[test]
    fn shosanna_range_iter_string_column() {
        let mut b = DictionaryBuilder::new();
        for s in ["amsterdam", "berlin", "paris", "prague", "barcelona"] {
            b.add(s);
        }
        let col = ColumnCodec::Dict(b.build());
        let zm = compute_block_zone_maps(&col);
        let min = Value::from("b");
        let max = Value::from("c");
        let from_iter: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), Some(&max), true, true)
            .collect();
        let eager = col.find_in_range(Some(&min), Some(&max), true, true);
        assert_eq!(from_iter, eager);
    }

    #[test]
    fn hans_range_iter_bitpacked_negative_min_bound() {
        // BitPacked stores u64; a negative min bound must not crash and
        // must match `find_in_range` semantics.
        let col = ColumnCodec::BitPacked(BitPackedInts::pack(&(0..20u64).collect::<Vec<_>>()));
        let zm = compute_block_zone_maps(&col);
        let min = Value::Int64(-5);
        let max = Value::Int64(10);
        let from_iter: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), Some(&max), true, true)
            .collect();
        let eager = col.find_in_range(Some(&min), Some(&max), true, true);
        assert_eq!(from_iter, eager);
    }

    #[test]
    fn beatrix_range_iter_exclusive_bounds() {
        let col = raw_i64_seq(50);
        let zm = compute_block_zone_maps(&col);
        let min = Value::Int64(10);
        let max = Value::Int64(20);
        let result: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), Some(&max), false, false)
            .collect();
        let expected: Vec<usize> = (11..20).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn django_range_iter_float64_with_nan_in_column() {
        // NaN in a Float64 column is not orderable and must never appear
        // in any range query result.
        let col = ColumnCodec::float64(vec![1.0, f64::NAN, 2.0, 3.0]);
        let zm = compute_block_zone_maps(&col);
        let min = Value::Float64(0.5);
        let max = Value::Float64(4.0);
        let from_iter: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), Some(&max), true, true)
            .collect();
        let eager = col.find_in_range(Some(&min), Some(&max), true, true);
        assert_eq!(from_iter, eager);
        assert!(
            !from_iter.contains(&1),
            "NaN row offset 1 must not appear in range result"
        );
    }

    #[test]
    fn tarantino_range_iter_yields_offsets_in_ascending_order() {
        // Iterator order matters for downstream chunking; rows must be
        // emitted in increasing offset order.
        let col = raw_i64_seq(2048);
        let zm = compute_block_zone_maps(&col);
        let min = Value::Int64(500);
        let max = Value::Int64(1500);
        let result: Vec<usize> = col
            .range_iter(Some(&zm), Some(&min), Some(&max), true, true)
            .collect();
        let mut sorted = result.clone();
        sorted.sort_unstable();
        assert_eq!(result, sorted, "iterator output must be sorted ascending");
    }
}
