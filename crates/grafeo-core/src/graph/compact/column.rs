//! Column codecs for CompactStore.
//!
//! Wraps Grafeo's existing storage primitives into a unified enum with
//! random access and `Value` decoding. CompactStore owns these types —
//! the underlying primitives are not modified.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::Value;

use crate::storage::{BitPackedInts, BitVector, DictionaryEncoding};

/// A single column of data backed by one of Grafeo's storage codecs.
///
/// Each variant wraps an existing primitive via composition — the
/// primitives themselves are never modified. Use [`get`](Self::get) for
/// `Value`-typed access and the specialised accessors when you know the
/// underlying codec.
#[derive(Debug, Clone)]
pub enum ColumnCodec {
    /// Fixed-width bit-packed unsigned integers.
    BitPacked(BitPackedInts),
    /// Dictionary-encoded strings.
    Dict(DictionaryEncoding),
    /// Null/boolean bitmap.
    Bitmap(BitVector),
    /// Int8 quantized vectors (flat array with stride).
    Int8Vector {
        /// Flat array of int8 components.
        data: Vec<i8>,
        /// Number of dimensions per vector.
        dimensions: u16,
    },
}

impl ColumnCodec {
    /// Decodes the value at `index` into a [`Value`].
    ///
    /// - [`BitPacked`](Self::BitPacked) → `Value::Int64(v as i64)`
    /// - [`Dict`](Self::Dict) → `Value::String(ArcStr::from(s))`
    /// - [`Bitmap`](Self::Bitmap) → `Value::Bool(b)`
    /// - [`Int8Vector`](Self::Int8Vector) → `Value::List(...)` of `Int64` values
    ///
    /// Returns `None` when `index` is out of bounds.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Value> {
        match self {
            // Safety: the builder validates bits <= 63, so all stored values
            // fit in i63 and the `as i64` cast is lossless.
            Self::BitPacked(bp) => bp.get(index).map(|v| Value::Int64(v as i64)),
            Self::Dict(dict) => dict.get(index).map(|s| Value::String(ArcStr::from(s))),
            Self::Bitmap(bv) => bv.get(index).map(Value::Bool),
            Self::Int8Vector { data, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > data.len() {
                    return None;
                }
                let values: Vec<Value> = data[start..end]
                    .iter()
                    .map(|&v| Value::Int64(v as i64))
                    .collect();
                Some(Value::List(Arc::from(values)))
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
            Self::Int8Vector { data, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 {
                    return None;
                }
                let start = index.checked_mul(dims)?;
                let end = start.checked_add(dims)?;
                if end > data.len() {
                    return None;
                }
                Some(&data[start..end])
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
            Self::Int8Vector { data, dimensions } => {
                let dims = *dimensions as usize;
                if dims == 0 { 0 } else { data.len() / dims }
            }
        }
    }

    /// Returns `true` if the column contains no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an estimate of heap memory used by this column in bytes.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::BitPacked(bp) => bp.data().len() * std::mem::size_of::<u64>(),
            Self::Dict(d) => {
                let codes_bytes = d.codes().len() * std::mem::size_of::<u32>();
                let dict_bytes: usize = d.dictionary().iter().map(|s| s.len()).sum();
                codes_bytes + dict_bytes
            }
            Self::Bitmap(bv) => bv.data().len() * std::mem::size_of::<u64>(),
            Self::Int8Vector { data, .. } => data.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{BitPackedInts, BitVector, DictionaryBuilder};

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
        let col = ColumnCodec::Int8Vector {
            data,
            dimensions: 3,
        };

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
        let col = ColumnCodec::Int8Vector {
            data,
            dimensions: 3,
        };

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

        let vec_col = ColumnCodec::Int8Vector {
            data: vec![1, 2],
            dimensions: 2,
        };
        assert_eq!(vec_col.get(1), None);
        assert_eq!(vec_col.get_int8_vector(1), None);
    }
}
