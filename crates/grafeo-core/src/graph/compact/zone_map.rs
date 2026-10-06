//! Per-column zone maps for predicate pushdown.
//!
//! Each column tracks min/max/null_count statistics. The query engine uses
//! these to skip entire tables when a predicate cannot match.

use std::cmp::Ordering;

use crate::graph::lpg::CompareOp;
use grafeo_common::types::Value;

/// Per-column min/max statistics for skip pruning.
///
/// A zone map tracks the range of values in a column so the query engine can
/// eliminate entire tables without scanning rows. If [`might_match`](Self::might_match)
/// returns `false`, the predicate is guaranteed to have zero matching rows.
#[derive(Debug, Clone, Default)]
pub struct ZoneMap {
    /// Minimum value in the column, or `None` if the column has no non-null values.
    pub min: Option<Value>,
    /// Maximum value in the column, or `None` if the column has no non-null values.
    pub max: Option<Value>,
    /// Number of null values in the column.
    pub null_count: usize,
    /// Total number of rows in the column.
    pub row_count: usize,
}

impl ZoneMap {
    /// Creates a new empty zone map with no statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the predicate `column <op> value` might match any row.
    ///
    /// This is a conservative check: returning `true` does not guarantee a match,
    /// but returning `false` guarantees there are no matches. When min/max are
    /// unavailable (all nulls, or incomparable types), this returns `true` to
    /// avoid false negatives.
    #[must_use]
    pub fn might_match(&self, op: CompareOp, value: &Value) -> bool {
        let (Some(min), Some(max)) = (&self.min, &self.max) else {
            // No statistics available, cannot rule anything out.
            return true;
        };

        match op {
            // column == value: possible only if min <= value <= max
            CompareOp::Eq => {
                let ge_min = compare_values(value, min).map_or(true, |ord| ord != Ordering::Less);
                let le_max =
                    compare_values(value, max).map_or(true, |ord| ord != Ordering::Greater);
                ge_min && le_max
            }
            // column != value: impossible only if min == max == value (and no nulls)
            CompareOp::Ne => {
                if self.null_count > 0 {
                    return true;
                }
                let all_same = compare_values(min, max).is_some_and(|ord| ord == Ordering::Equal);
                let eq_value = min == value;
                !(all_same && eq_value)
            }
            // column < value: possible if min < value
            CompareOp::Lt => compare_values(min, value).map_or(true, |ord| ord == Ordering::Less),
            // column <= value: possible if min <= value
            CompareOp::Le => {
                compare_values(min, value).map_or(true, |ord| ord != Ordering::Greater)
            }
            // column > value: possible if max > value
            CompareOp::Gt => {
                compare_values(max, value).map_or(true, |ord| ord == Ordering::Greater)
            }
            // column >= value: possible if max >= value
            CompareOp::Ge => compare_values(max, value).map_or(true, |ord| ord != Ordering::Less),
        }
    }
}

impl ZoneMap {
    /// Serializes this zone map into `buf` using the v3 inline layout:
    /// `[null_count:u32][row_count:u32][min:tagged][max:tagged]`.
    ///
    /// The tagged-value encoding supports `Int64`, `Bool`, and `String`
    /// today; `Float64` and other variants serialize as absent (tag 0)
    /// to match the existing `write_optional_value` helper used for
    /// per-column zone maps. Per-block Float64 stats are tracked in
    /// memory (see [`compute_block_zone_maps`]) and survive in-process
    /// queries; persistence support is a follow-on.
    pub fn write_inline(&self, buf: &mut Vec<u8>) {
        let null_count = u32::try_from(self.null_count).unwrap_or(u32::MAX);
        let row_count = u32::try_from(self.row_count).unwrap_or(u32::MAX);
        buf.extend_from_slice(&null_count.to_le_bytes());
        buf.extend_from_slice(&row_count.to_le_bytes());
        write_inline_value(buf, &self.min);
        write_inline_value(buf, &self.max);
    }

    /// Deserializes a zone map from the inline layout written by
    /// [`write_inline`](Self::write_inline). Advances `pos` past the
    /// consumed bytes.
    ///
    /// # Errors
    ///
    /// Returns a static-string error on truncation or invalid tag.
    pub fn read_inline(data: &[u8], pos: &mut usize) -> Result<Self, &'static str> {
        let null_count = read_inline_u32(data, pos)? as usize;
        let row_count = read_inline_u32(data, pos)? as usize;
        let min = read_inline_value(data, pos)?;
        let max = read_inline_value(data, pos)?;
        Ok(Self {
            min,
            max,
            null_count,
            row_count,
        })
    }
}

fn write_inline_value(buf: &mut Vec<u8>, v: &Option<Value>) {
    match v {
        None => buf.push(0),
        Some(Value::Int64(n)) => {
            buf.push(1);
            buf.extend_from_slice(&n.to_le_bytes());
        }
        Some(Value::Bool(b)) => {
            buf.push(2);
            buf.push(u8::from(*b));
        }
        Some(Value::String(s)) => {
            let bytes = s.as_str().as_bytes();
            // Strings whose byte length exceeds u32::MAX cannot be encoded
            // in the inline format. Writing a saturated length while still
            // emitting the full bytes would corrupt the trailing values; emit
            // tag 0 (absent) instead so the column simply loses its zone-map
            // stat for this side of the range.
            if let Ok(len) = u32::try_from(bytes.len()) {
                buf.push(3);
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(bytes);
            } else {
                buf.push(0);
            }
        }
        Some(_) => buf.push(0),
    }
}

fn read_inline_value(data: &[u8], pos: &mut usize) -> Result<Option<Value>, &'static str> {
    let tag = *data.get(*pos).ok_or("truncated zone map value tag")?;
    *pos += 1;
    match tag {
        0 => Ok(None),
        1 => {
            if *pos + 8 > data.len() {
                return Err("truncated zone map Int64");
            }
            let bytes: [u8; 8] = data[*pos..*pos + 8].try_into().expect("8 bytes guaranteed");
            *pos += 8;
            Ok(Some(Value::Int64(i64::from_le_bytes(bytes))))
        }
        2 => {
            let b = *data.get(*pos).ok_or("truncated zone map Bool")?;
            *pos += 1;
            Ok(Some(Value::Bool(b != 0)))
        }
        3 => {
            let len = read_inline_u32(data, pos)? as usize;
            if *pos + len > data.len() {
                return Err("truncated zone map String");
            }
            let s = std::str::from_utf8(&data[*pos..*pos + len])
                .map_err(|_| "invalid UTF-8 in zone map String")?;
            *pos += len;
            Ok(Some(Value::String(arcstr::ArcStr::from(s))))
        }
        _ => Err("unknown zone map value tag"),
    }
}

fn read_inline_u32(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated zone map u32");
    }
    let bytes: [u8; 4] = data[*pos..*pos + 4].try_into().expect("4 bytes guaranteed");
    *pos += 4;
    Ok(u32::from_le_bytes(bytes))
}

/// Computes the zone map of a whole column: min and max over the rows that
/// have a value, the other rows counted as nulls.
#[must_use]
pub fn compute_zone_map(column: &super::column::CompactColumn) -> ZoneMap {
    let mut zm = ZoneMap {
        row_count: column.len(),
        ..ZoneMap::default()
    };
    for row in 0..column.len() {
        match column.get(row) {
            Some(value) => update_block_min_max(&mut zm, &value),
            None => zm.null_count += 1,
        }
    }
    zm
}

/// Computes per-block zone maps for a column.
///
/// Walks each block's row range, tracking min/max/null_count for the
/// orderable scalar types (`Int64`, `Float64`, `String`, `Bool`). A row
/// without a value counts as a null. Vector and list columns produce zone
/// maps with `None` min/max but still track row and null counts.
///
/// Empty columns produce a single zero-row block (matching
/// [`ColumnCodec::block_count`](super::column::ColumnCodec::block_count)).
#[must_use]
pub fn compute_block_zone_maps(column: &super::column::CompactColumn) -> Vec<ZoneMap> {
    block_zone_maps(column.block_count(), column.len(), |row| column.get(row))
}

/// [`compute_block_zone_maps`] for a bare codec, where every row has a value.
#[must_use]
pub(super) fn compute_codec_block_zone_maps(codec: &super::column::ColumnCodec) -> Vec<ZoneMap> {
    block_zone_maps(codec.block_count(), codec.len(), |row| codec.get(row))
}

fn block_zone_maps(
    block_count: usize,
    len: usize,
    get: impl Fn(usize) -> Option<Value>,
) -> Vec<ZoneMap> {
    let block_rows = crate::codec::DEFAULT_BLOCK_ROWS as usize;
    let mut result = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let start = i * block_rows;
        let end = (start + block_rows).min(len);
        let mut zm = ZoneMap {
            row_count: end - start,
            ..ZoneMap::default()
        };
        for j in start..end {
            match get(j) {
                Some(value) => update_block_min_max(&mut zm, &value),
                None => zm.null_count += 1,
            }
        }
        result.push(zm);
    }
    result
}

fn update_block_min_max(zm: &mut ZoneMap, value: &Value) {
    // Skip non-orderable types (lists, vectors, etc.); their min/max
    // remain `None` so the planner falls back to scanning the block.
    if !is_orderable(value) {
        return;
    }
    let smaller = match &zm.min {
        Some(current) => compare_values(value, current) == Some(Ordering::Less),
        None => true,
    };
    if smaller {
        zm.min = Some(value.clone());
    }
    let larger = match &zm.max {
        Some(current) => compare_values(value, current) == Some(Ordering::Greater),
        None => true,
    };
    if larger {
        zm.max = Some(value.clone());
    }
}

fn is_orderable(value: &Value) -> bool {
    matches!(
        value,
        Value::Int64(_) | Value::Float64(_) | Value::String(_) | Value::Bool(_)
    )
}

/// Compares two values for ordering.
///
/// Returns `None` for incomparable types (different type families).
pub(super) fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: builds a zone map with Int64 min/max.
    fn int_zone(min: i64, max: i64, null_count: usize, row_count: usize) -> ZoneMap {
        ZoneMap {
            min: Some(Value::Int64(min)),
            max: Some(Value::Int64(max)),
            null_count,
            row_count,
        }
    }

    #[test]
    fn test_eq_in_range() {
        let zm = int_zone(10, 50, 0, 100);
        assert!(zm.might_match(CompareOp::Eq, &Value::Int64(25)));
        assert!(zm.might_match(CompareOp::Eq, &Value::Int64(10)));
        assert!(zm.might_match(CompareOp::Eq, &Value::Int64(50)));
    }

    #[test]
    fn test_eq_out_of_range() {
        let zm = int_zone(10, 50, 0, 100);
        assert!(!zm.might_match(CompareOp::Eq, &Value::Int64(5)));
        assert!(!zm.might_match(CompareOp::Eq, &Value::Int64(51)));
    }

    #[test]
    fn test_lt() {
        let zm = int_zone(10, 50, 0, 100);
        // column < 20: min(10) < 20, so possible
        assert!(zm.might_match(CompareOp::Lt, &Value::Int64(20)));
        // column < 10: min(10) < 10 is false, so impossible
        assert!(!zm.might_match(CompareOp::Lt, &Value::Int64(10)));
        // column < 5: min(10) < 5 is false
        assert!(!zm.might_match(CompareOp::Lt, &Value::Int64(5)));
    }

    #[test]
    fn test_le() {
        let zm = int_zone(10, 50, 0, 100);
        // column <= 10: min(10) <= 10, so possible
        assert!(zm.might_match(CompareOp::Le, &Value::Int64(10)));
        // column <= 9: min(10) <= 9 is false
        assert!(!zm.might_match(CompareOp::Le, &Value::Int64(9)));
    }

    #[test]
    fn test_gt() {
        let zm = int_zone(10, 50, 0, 100);
        // column > 40: max(50) > 40, so possible
        assert!(zm.might_match(CompareOp::Gt, &Value::Int64(40)));
        // column > 50: max(50) > 50 is false
        assert!(!zm.might_match(CompareOp::Gt, &Value::Int64(50)));
        // column > 60: max(50) > 60 is false
        assert!(!zm.might_match(CompareOp::Gt, &Value::Int64(60)));
    }

    #[test]
    fn test_ge() {
        let zm = int_zone(10, 50, 0, 100);
        // column >= 50: max(50) >= 50, so possible
        assert!(zm.might_match(CompareOp::Ge, &Value::Int64(50)));
        // column >= 51: max(50) >= 51 is false
        assert!(!zm.might_match(CompareOp::Ge, &Value::Int64(51)));
    }

    #[test]
    fn test_ne() {
        let zm = int_zone(10, 50, 0, 100);
        // Range has spread, so Ne always matches.
        assert!(zm.might_match(CompareOp::Ne, &Value::Int64(10)));
        assert!(zm.might_match(CompareOp::Ne, &Value::Int64(25)));

        // Single-value range, no nulls: Ne with that value is impossible.
        let single = int_zone(42, 42, 0, 10);
        assert!(!single.might_match(CompareOp::Ne, &Value::Int64(42)));
        assert!(single.might_match(CompareOp::Ne, &Value::Int64(43)));
    }

    #[test]
    fn test_ne_with_nulls() {
        // If there are nulls, Ne is always conservatively true.
        let zm = int_zone(42, 42, 5, 10);
        assert!(zm.might_match(CompareOp::Ne, &Value::Int64(42)));
    }

    #[test]
    fn test_empty_zone_map() {
        let zm = ZoneMap::new();
        // No stats: must return true for all predicates (conservative).
        assert!(zm.might_match(CompareOp::Eq, &Value::Int64(1)));
        assert!(zm.might_match(CompareOp::Ne, &Value::Int64(1)));
        assert!(zm.might_match(CompareOp::Lt, &Value::Int64(1)));
        assert!(zm.might_match(CompareOp::Le, &Value::Int64(1)));
        assert!(zm.might_match(CompareOp::Gt, &Value::Int64(1)));
        assert!(zm.might_match(CompareOp::Ge, &Value::Int64(1)));
    }

    #[test]
    fn test_string_zone_map() {
        let zm = ZoneMap {
            min: Some(Value::from("apple")),
            max: Some(Value::from("grape")),
            null_count: 0,
            row_count: 50,
        };

        assert!(zm.might_match(CompareOp::Eq, &Value::from("banana")));
        assert!(!zm.might_match(CompareOp::Eq, &Value::from("zebra")));
        assert!(zm.might_match(CompareOp::Lt, &Value::from("banana")));
        assert!(!zm.might_match(CompareOp::Gt, &Value::from("zebra")));
    }

    #[test]
    fn test_incomparable_types_are_conservative() {
        let zm = int_zone(10, 50, 0, 100);
        // Comparing Int64 zone map against a String value: types are incomparable,
        // so we must conservatively return true.
        assert!(zm.might_match(CompareOp::Eq, &Value::from("hello")));
        assert!(zm.might_match(CompareOp::Lt, &Value::from("hello")));
    }

    #[test]
    fn test_default() {
        let zm = ZoneMap::default();
        assert!(zm.min.is_none());
        assert!(zm.max.is_none());
        assert_eq!(zm.null_count, 0);
        assert_eq!(zm.row_count, 0);
    }

    #[test]
    fn test_inline_value_round_trip_keeps_following_field_aligned() {
        // The string min plus an Int64 max must round-trip exactly: this
        // pins down the contract that `write_inline_value`/`read_inline_value`
        // never desynchronize the cursor (saturating an oversize length
        // while still emitting the bytes used to corrupt the next field).
        let zm = ZoneMap {
            min: Some(Value::from("apple")),
            max: Some(Value::Int64(99)),
            null_count: 1,
            row_count: 12,
        };
        let mut buf = Vec::new();
        zm.write_inline(&mut buf);
        let mut pos = 0;
        let recovered = ZoneMap::read_inline(&buf, &mut pos).expect("round-trip");
        assert_eq!(pos, buf.len(), "reader must consume the entire buffer");
        assert_eq!(recovered.min, zm.min);
        assert_eq!(recovered.max, zm.max);
        assert_eq!(recovered.null_count, zm.null_count);
        assert_eq!(recovered.row_count, zm.row_count);
    }
}
