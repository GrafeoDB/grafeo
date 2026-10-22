//! The type of a typed property, as node and edge types declare it.
//!
//! [`PropertyDataType`] is what type DDL declares a property to hold
//! (`price DECIMAL(6, 2)`, `tags LIST<STRING>`): the catalog checks and
//! converts values against it, and the value type predicate `IS TYPED` checks
//! values against the same types (see [`ValueType`](super::ValueType)), so
//! both read one set of names.

use std::fmt;

use crate::storage::catalog_record::MAX_LIST_LEVELS_PER_RECORD;
use crate::storage::value_codec::MAX_PROPERTY_VALUE_DEPTH;
use crate::types::{Decimal, DecimalType, Value};

/// A property type name nests more `LIST<...>` levels than `limit`
/// ([`PropertyDataType::MAX_LIST_DEPTH`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropertyTypeTooDeep {
    /// The most levels a property type nests.
    pub limit: usize,
}

impl fmt::Display for PropertyTypeTooDeep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "A property type nests at most {} LIST<...> levels",
            self.limit
        )
    }
}

impl std::error::Error for PropertyTypeTooDeep {}

/// Data type for a typed property in a node or edge type definition.
///
/// Exhaustive, so the catalog's serializers name every type: a new variant
/// fails their matches until they write it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PropertyDataType {
    /// UTF-8 string.
    String,
    /// 64-bit signed integer.
    Int64,
    /// 64-bit floating point.
    Float64,
    /// Boolean.
    Bool,
    /// Calendar date.
    Date,
    /// Time of day.
    Time,
    /// Timestamp (date + time).
    Timestamp,
    /// Duration / interval.
    Duration,
    /// Ordered list of values (untyped).
    List,
    /// Typed list: `LIST<element_type>` (ISO sec 4.16.9).
    ListTyped(Box<PropertyDataType>),
    /// Key-value map.
    Map,
    /// Raw bytes.
    Bytes,
    /// Node reference type (ISO sec 4.15.1).
    Node,
    /// Edge reference type (ISO sec 4.15.1).
    Edge,
    /// Any type (no enforcement).
    Any,
    // The 0.5.x catalog layouts (catalog section v1, snapshot v4) are bincode
    // of this enum, which numbers the variants by position: new variants go
    // here, after the existing ones.
    /// Datetime with a fixed UTC offset (`ZONED DATETIME`), matching
    /// [`Value::ZonedDatetime`].
    ZonedDatetime,
    /// Datetime without a time zone (`LOCAL DATETIME`), matching
    /// [`Value::Timestamp`], which `local_datetime()` returns.
    LocalDatetime,
    /// Exact decimal: `DECIMAL`, any decimal with its own scale (`None`), or
    /// `DECIMAL(p, s)`, matching [`Value::Decimal`].
    Decimal(Option<DecimalType>),
}

impl PropertyDataType {
    /// The most `LIST<...>` levels a property type nests: as deep as a
    /// property value may be ([`MAX_PROPERTY_VALUE_DEPTH`], a list of scalars
    /// being 1 deep), and as deep as the catalog records store.
    pub const MAX_LIST_DEPTH: usize = MAX_PROPERTY_VALUE_DEPTH;

    /// The most `LIST<...>` levels the property types of one node or edge
    /// type nest in all: as many as its catalog record holds
    /// ([`MAX_LIST_LEVELS_PER_RECORD`], 32,768, so 256 properties of the
    /// deepest type).
    pub const MAX_LIST_LEVELS_PER_TYPE: usize = MAX_LIST_LEVELS_PER_RECORD;

    /// The `LIST<...>` levels around the type inside them, counted, not
    /// recursed into.
    #[must_use]
    pub fn list_levels(&self) -> usize {
        let mut levels = 0;
        let mut current = self;
        while let Self::ListTyped(inner) = current {
            levels += 1;
            current = inner;
        }
        levels
    }

    /// Parses a type name string (case-insensitive) into a `PropertyDataType`.
    ///
    /// Reads every spelling [`Display`](std::fmt::Display) writes, including
    /// `ZONED DATETIME`, `LOCAL DATETIME` and nested `LIST<...>`, so the type
    /// names of WAL records and `SHOW` output read back as the same type.
    ///
    /// Unknown names are [`Any`](Self::Any): type DDL refuses them (the
    /// parsers check each name against `PROPERTY_TYPE_NAMES` of
    /// `grafeo_adapters::query::schema`),
    /// so only a WAL record written by 0.5.x, which logged a type as it was
    /// written, holds one, and 0.5.x read it as `ANY` too.
    ///
    /// # Errors
    ///
    /// [`PropertyTypeTooDeep`] when the name nests more than
    /// [`MAX_LIST_DEPTH`](Self::MAX_LIST_DEPTH) `LIST<...>` levels. The levels
    /// are counted, not recursed into, so a name of any depth is refused
    /// without a stack overflow.
    pub fn from_type_name(name: &str) -> Result<Self, PropertyTypeTooDeep> {
        let upper = name.to_uppercase();
        let mut element = upper.as_str();
        let mut levels = 0;
        while let Some(inner) = element
            .strip_prefix("LIST<")
            .and_then(|s| s.strip_suffix('>'))
        {
            levels += 1;
            if levels > Self::MAX_LIST_DEPTH {
                return Err(PropertyTypeTooDeep {
                    limit: Self::MAX_LIST_DEPTH,
                });
            }
            element = inner;
        }
        let mut data_type = Self::from_element_name(element).unwrap_or(Self::Any);
        for _ in 0..levels {
            data_type = Self::ListTyped(Box::new(data_type));
        }
        Ok(data_type)
    }

    /// The type `name` names, as [`from_type_name`](Self::from_type_name)
    /// reads it, or `None` when a name in it is no type (where
    /// `from_type_name` reads `ANY`) or it nests too deep.
    #[must_use]
    pub fn from_known_type_name(name: &str) -> Option<Self> {
        let upper = name.to_uppercase();
        let mut element = upper.as_str();
        let mut levels = 0;
        while let Some(inner) = element
            .strip_prefix("LIST<")
            .and_then(|s| s.strip_suffix('>'))
        {
            levels += 1;
            element = inner;
        }
        if levels > Self::MAX_LIST_DEPTH {
            return None;
        }
        let mut data_type = Self::from_element_name(element)?;
        for _ in 0..levels {
            data_type = Self::ListTyped(Box::new(data_type));
        }
        Some(data_type)
    }

    /// The type a name without `LIST<...>` around it names (in capitals), or
    /// `None` for a name that is no type.
    fn from_element_name(upper: &str) -> Option<Self> {
        if let Some(decimal) = DecimalType::from_name(upper) {
            return Some(Self::Decimal(decimal));
        }
        Some(match upper {
            "STRING" | "VARCHAR" | "TEXT" => Self::String,
            "INT" | "INT64" | "INTEGER" | "BIGINT" => Self::Int64,
            "FLOAT" | "FLOAT64" | "DOUBLE" | "REAL" => Self::Float64,
            "BOOL" | "BOOLEAN" => Self::Bool,
            "DATE" => Self::Date,
            "TIME" => Self::Time,
            "TIMESTAMP" | "DATETIME" => Self::Timestamp,
            "ZONED DATETIME" | "ZONED_DATETIME" | "ZONEDDATETIME" => Self::ZonedDatetime,
            "LOCAL DATETIME" | "LOCAL_DATETIME" | "LOCALDATETIME" => Self::LocalDatetime,
            "DURATION" | "INTERVAL" => Self::Duration,
            "LIST" | "ARRAY" => Self::List,
            "MAP" | "RECORD" => Self::Map,
            "BYTES" | "BINARY" | "BLOB" => Self::Bytes,
            "NODE" => Self::Node,
            "EDGE" | "RELATIONSHIP" => Self::Edge,
            "ANY" => Self::Any,
            // A name only an older WAL holds (see `from_type_name`).
            _ => return None,
        })
    }

    /// Checks whether a value conforms to this type.
    #[must_use]
    pub fn matches(&self, value: &Value) -> bool {
        match (self, value) {
            (Self::Any, _) | (_, Value::Null) => true,
            (Self::String, Value::String(_)) => true,
            (Self::Int64, Value::Int64(_)) => true,
            (Self::Float64, Value::Float64(_)) => true,
            (Self::Bool, Value::Bool(_)) => true,
            (Self::Date, Value::Date(_)) => true,
            (Self::Time, Value::Time(_)) => true,
            (Self::Timestamp | Self::LocalDatetime, Value::Timestamp(_)) => true,
            (Self::ZonedDatetime, Value::ZonedDatetime(_)) => true,
            (Self::Decimal(None), Value::Decimal(_)) => true,
            (Self::Decimal(Some(decimal_type)), Value::Decimal(decimal)) => {
                decimal_type.holds(decimal)
            }
            (Self::Duration, Value::Duration(_)) => true,
            (Self::List, Value::List(_)) => true,
            (Self::ListTyped(elem_type), Value::List(items)) => {
                items.iter().all(|item| elem_type.matches(item))
            }
            (Self::Map, Value::Map(_)) => true,
            (Self::Bytes, Value::Bytes(_)) => true,
            // Node/Edge reference types match Map values (graph elements are
            // represented as maps with _id, _labels/_type, and properties)
            (Self::Node | Self::Edge, Value::Map(_)) => true,
            _ => false,
        }
    }

    /// The value a property of this type stores for `value` when `value` is
    /// not of the type but converts to it on assignment without losing
    /// anything, as ISO/IEC 39075 converts between numeric types and SQL's
    /// store assignment does:
    ///
    /// - an `INT64` to `FLOAT64`, when the float is exactly the integer (every
    ///   integer up to 2^53 in magnitude, and the larger ones a float holds);
    /// - a decimal to `FLOAT64`, as the nearest float: ISO/IEC 39075 assigns an
    ///   exact number to an approximate type that way (`1.5` is stored as the
    ///   float `1.5`);
    /// - an integer, a float (by its shortest text) or a decimal to `DECIMAL`,
    ///   and to `DECIMAL(p, s)` rounded to `s` digits after the point, half
    ///   away from zero, as SQL's store assignment rounds; not one with more
    ///   than `p - s` digits before it, nor a NaN or an infinity;
    /// - a zoned datetime to its instant, for `TIMESTAMP` (`DATETIME`), which
    ///   compares with zoned values as an instant in UTC;
    /// - the items of a `LIST<...>`, each by these rules.
    ///
    /// A `LOCAL DATETIME` and a `ZONED DATETIME` take no other datetime: a
    /// local datetime has no offset to be an instant by.
    ///
    /// `None` when `value` is of the type already, or does not convert.
    #[must_use]
    pub fn convert(&self, value: &Value) -> Option<Value> {
        match (self, value) {
            (Self::Float64, Value::Int64(int)) => exact_float(*int).map(Value::Float64),
            (Self::Float64, Value::Decimal(decimal)) => Some(Value::Float64(decimal.to_f64())),
            (Self::Decimal(decimal_type), _) if !self.matches(value) => {
                let decimal = match value {
                    Value::Decimal(decimal) => *decimal,
                    Value::Int64(int) => Decimal::from_i64(*int),
                    Value::Float64(float) => Decimal::from_f64(*float).ok()?,
                    _ => return None,
                };
                match decimal_type {
                    None => Some(Value::Decimal(decimal)),
                    Some(decimal_type) => decimal_type.fit(&decimal).ok().map(Value::Decimal),
                }
            }
            (Self::Timestamp, Value::ZonedDatetime(zoned)) => {
                Some(Value::Timestamp(zoned.as_timestamp()))
            }
            (Self::ListTyped(item_type), Value::List(items)) if !self.matches(value) => items
                .iter()
                .map(|item| {
                    if item_type.matches(item) {
                        Some(item.clone())
                    } else {
                        item_type.convert(item)
                    }
                })
                .collect::<Option<Vec<Value>>>()
                .map(|items| Value::List(items.into())),
            _ => None,
        }
    }
}

/// `int` as a float, when the float is exactly `int`.
pub fn exact_float(int: i64) -> Option<f64> {
    // 2^63: the one float an i64 rounds to that is out of the i64 range
    // (i64::MAX rounds up to it).
    const BEYOND_I64: f64 = 9_223_372_036_854_775_808.0;
    #[allow(
        clippy::cast_precision_loss,
        reason = "the cast back below checks that the float is exact"
    )]
    let float = int as f64;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "`float` came from an i64 and is below 2^63, checked first, and a whole \
                  number (a float from an integer is), so the cast back is exact"
    )]
    let exact = float < BEYOND_I64 && float as i64 == int;
    exact.then_some(float)
}

impl fmt::Display for PropertyDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "STRING"),
            Self::Int64 => write!(f, "INT64"),
            Self::Float64 => write!(f, "FLOAT64"),
            Self::Bool => write!(f, "BOOLEAN"),
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
            Self::Duration => write!(f, "DURATION"),
            Self::List => write!(f, "LIST"),
            Self::ListTyped(elem) => write!(f, "LIST<{elem}>"),
            Self::Map => write!(f, "MAP"),
            Self::Bytes => write!(f, "BYTES"),
            Self::Node => write!(f, "NODE"),
            Self::Edge => write!(f, "EDGE"),
            Self::Any => write!(f, "ANY"),
            Self::ZonedDatetime => write!(f, "ZONED DATETIME"),
            Self::LocalDatetime => write!(f, "LOCAL DATETIME"),
            Self::Decimal(None) => write!(f, "DECIMAL"),
            Self::Decimal(Some(decimal_type)) => write!(f, "{decimal_type}"),
        }
    }
}
