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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
}
