//! Property value types for graph elements.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use super::Timestamp;

/// A property key (interned string identifier).
///
/// Property keys are commonly used strings that benefit from interning.
/// This uses an `Arc<str>` for cheap cloning and comparison.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PropertyKey(Arc<str>);

impl PropertyKey {
    /// Creates a new property key from a string.
    #[must_use]
    pub fn new(s: impl Into<Arc<str>>) -> Self {
        Self(s.into())
    }

    /// Returns the string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for PropertyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PropertyKey({:?})", self.0)
    }
}

impl fmt::Display for PropertyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for PropertyKey {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for PropertyKey {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl AsRef<str> for PropertyKey {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// A dynamically-typed property value.
///
/// This enum represents all possible value types that can be stored as
/// properties on nodes and edges. It supports the GQL type system.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Null/missing value
    Null,

    /// Boolean value
    Bool(bool),

    /// 64-bit signed integer
    Int64(i64),

    /// 64-bit floating point
    Float64(f64),

    /// UTF-8 string (uses Arc for cheap cloning)
    String(Arc<str>),

    /// Binary data
    Bytes(Arc<[u8]>),

    /// Timestamp with timezone
    Timestamp(Timestamp),

    /// Ordered list of values
    List(Arc<[Value]>),

    /// Key-value map (uses BTreeMap for deterministic ordering)
    Map(Arc<BTreeMap<PropertyKey, Value>>),
}

impl Value {
    /// Returns `true` if this value is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Returns the boolean value if this is a Bool, otherwise None.
    #[inline]
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the integer value if this is an Int64, otherwise None.
    #[inline]
    #[must_use]
    pub const fn as_int64(&self) -> Option<i64> {
        match self {
            Value::Int64(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the float value if this is a Float64, otherwise None.
    #[inline]
    #[must_use]
    pub const fn as_float64(&self) -> Option<f64> {
        match self {
            Value::Float64(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the string value if this is a String, otherwise None.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the bytes value if this is Bytes, otherwise None.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// Returns the timestamp value if this is a Timestamp, otherwise None.
    #[inline]
    #[must_use]
    pub const fn as_timestamp(&self) -> Option<Timestamp> {
        match self {
            Value::Timestamp(t) => Some(*t),
            _ => None,
        }
    }

    /// Returns the list value if this is a List, otherwise None.
    #[inline]
    #[must_use]
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) => Some(l),
            _ => None,
        }
    }

    /// Returns the map value if this is a Map, otherwise None.
    #[inline]
    #[must_use]
    pub fn as_map(&self) -> Option<&BTreeMap<PropertyKey, Value>> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Returns the type name of this value.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "NULL",
            Value::Bool(_) => "BOOL",
            Value::Int64(_) => "INT64",
            Value::Float64(_) => "FLOAT64",
            Value::String(_) => "STRING",
            Value::Bytes(_) => "BYTES",
            Value::Timestamp(_) => "TIMESTAMP",
            Value::List(_) => "LIST",
            Value::Map(_) => "MAP",
        }
    }

    /// Serializes this value to bytes.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("Value serialization should not fail")
    }

    /// Deserializes a value from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes do not represent a valid Value.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        let (value, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(value)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "Null"),
            Value::Bool(b) => write!(f, "Bool({b})"),
            Value::Int64(i) => write!(f, "Int64({i})"),
            Value::Float64(fl) => write!(f, "Float64({fl})"),
            Value::String(s) => write!(f, "String({s:?})"),
            Value::Bytes(b) => write!(f, "Bytes([{}; {} bytes])", b.first().unwrap_or(&0), b.len()),
            Value::Timestamp(t) => write!(f, "Timestamp({t:?})"),
            Value::List(l) => write!(f, "List({l:?})"),
            Value::Map(m) => write!(f, "Map({m:?})"),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "NULL"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int64(i) => write!(f, "{i}"),
            Value::Float64(fl) => write!(f, "{fl}"),
            Value::String(s) => write!(f, "{s:?}"),
            Value::Bytes(b) => write!(f, "<bytes: {} bytes>", b.len()),
            Value::Timestamp(t) => write!(f, "{t}"),
            Value::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

// Convenient From implementations
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int64(i)
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int64(i64::from(i))
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float64(f)
    }
}

impl From<f32> for Value {
    fn from(f: f32) -> Self {
        Value::Float64(f64::from(f))
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.into())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s.into())
    }
}

impl From<Arc<str>> for Value {
    fn from(s: Arc<str>) -> Self {
        Value::String(s)
    }
}

impl From<Vec<u8>> for Value {
    fn from(b: Vec<u8>) -> Self {
        Value::Bytes(b.into())
    }
}

impl From<&[u8]> for Value {
    fn from(b: &[u8]) -> Self {
        Value::Bytes(b.into())
    }
}

impl From<Timestamp> for Value {
    fn from(t: Timestamp) -> Self {
        Value::Timestamp(t)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::List(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => Value::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type_checks() {
        assert!(Value::Null.is_null());
        assert!(!Value::Bool(true).is_null());

        assert_eq!(Value::Bool(true).as_bool(), Some(true));
        assert_eq!(Value::Bool(false).as_bool(), Some(false));
        assert_eq!(Value::Int64(42).as_bool(), None);

        assert_eq!(Value::Int64(42).as_int64(), Some(42));
        assert_eq!(Value::String("test".into()).as_int64(), None);

        assert_eq!(Value::Float64(3.14).as_float64(), Some(3.14));
        assert_eq!(Value::String("hello".into()).as_str(), Some("hello"));
    }

    #[test]
    fn test_value_from_conversions() {
        let v: Value = true.into();
        assert_eq!(v.as_bool(), Some(true));

        let v: Value = 42i64.into();
        assert_eq!(v.as_int64(), Some(42));

        let v: Value = 3.14f64.into();
        assert_eq!(v.as_float64(), Some(3.14));

        let v: Value = "hello".into();
        assert_eq!(v.as_str(), Some("hello"));

        let v: Value = vec![1u8, 2, 3].into();
        assert_eq!(v.as_bytes(), Some(&[1u8, 2, 3][..]));
    }

    #[test]
    fn test_value_serialization_roundtrip() {
        let values = vec![
            Value::Null,
            Value::Bool(true),
            Value::Int64(i64::MAX),
            Value::Float64(std::f64::consts::PI),
            Value::String("hello world".into()),
            Value::Bytes(vec![0, 1, 2, 255].into()),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
        ];

        for v in values {
            let bytes = v.serialize();
            let decoded = Value::deserialize(&bytes).unwrap();
            assert_eq!(v, decoded);
        }
    }

    #[test]
    fn test_property_key() {
        let key = PropertyKey::new("name");
        assert_eq!(key.as_str(), "name");

        let key2: PropertyKey = "age".into();
        assert_eq!(key2.as_str(), "age");

        // Keys should be comparable
        assert!(key < key2); // "name" > "age" alphabetically, but we're using Ord
    }

    #[test]
    fn test_value_type_name() {
        assert_eq!(Value::Null.type_name(), "NULL");
        assert_eq!(Value::Bool(true).type_name(), "BOOL");
        assert_eq!(Value::Int64(0).type_name(), "INT64");
        assert_eq!(Value::Float64(0.0).type_name(), "FLOAT64");
        assert_eq!(Value::String("".into()).type_name(), "STRING");
        assert_eq!(Value::Bytes(vec![].into()).type_name(), "BYTES");
        assert_eq!(Value::List(vec![].into()).type_name(), "LIST");
        assert_eq!(Value::Map(BTreeMap::new().into()).type_name(), "MAP");
    }
}
