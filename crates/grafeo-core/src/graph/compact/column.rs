//! Column codecs for CompactStore.
//!
//! Wraps Grafeo's existing storage primitives into a unified enum with
//! random access and `Value` decoding. CompactStore owns these types:
//! the underlying primitives are not modified.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::Value;

use crate::codec::{BitPackedInts, BitVector, DictionaryEncoding};

/// A single column of data backed by one of Grafeo's storage codecs.
///
/// Each variant wraps an existing primitive via composition: the
/// primitives themselves are never modified. Use [`get`](Self::get) for
/// `Value`-typed access and the specialised accessors when you know the
/// underlying codec.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
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
            // The builder validates all values <= i64::MAX, so this cast is lossless.
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
                let target_u64 = v as u64;
                (0..bp.len())
                    .filter(|&i| bp.get(i) == Some(target_u64))
                    .collect()
            }
            (Self::Dict(dict), Value::String(s)) => match dict.encode(s.as_str()) {
                Some(code) => dict.filter_by_code(|c| c == code),
                None => Vec::new(),
            },
            (Self::Bitmap(bv), &Value::Bool(target_bool)) => (0..bv.len())
                .filter(|&i| bv.get(i) == Some(target_bool))
                .collect(),
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
                Some(&Value::Int64(v)) if v >= 0 => Some(v as u64),
                Some(&Value::Int64(_)) => Some(0),
                None => None,
                _ => return self.find_in_range_fallback(min, max, min_inclusive, max_inclusive),
            };
            let max_u64 = match max {
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

        self.find_in_range_fallback(min, max, min_inclusive, max_inclusive)
    }

    /// Fallback range scan via per-row `Value` decode.
    fn find_in_range_fallback(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<usize> {
        use super::zone_map::compare_values;

        (0..self.len())
            .filter(|&i| {
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
            })
            .collect()
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
}
