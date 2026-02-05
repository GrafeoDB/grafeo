//! Property values and keys for nodes and edges.
//!
//! [`Value`] is the dynamic type that can hold any property value - strings,
//! numbers, lists, maps, etc. [`PropertyKey`] is an interned string for
//! efficient property lookups.

use arcstr::ArcStr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::Timestamp;

/// An interned property name - cheap to clone and compare.
///
/// Property names like "name", "age", "created_at" get used repeatedly, so
/// we intern them with `ArcStr`. You can create these from strings directly.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PropertyKey(ArcStr);

impl PropertyKey {
    /// Creates a new property key from a string.
    #[must_use]
    pub fn new(s: impl Into<ArcStr>) -> Self {
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
/// Nodes and edges can have properties of various types - this enum holds
/// them all. Follows the GQL type system, so you can store nulls, booleans,
/// numbers, strings, timestamps, lists, and maps.
///
/// # Examples
///
/// ```
/// use grafeo_common::types::Value;
///
/// let name = Value::from("Alice");
/// let age = Value::from(30i64);
/// let active = Value::from(true);
///
/// // Check types
/// assert!(name.as_str().is_some());
/// assert_eq!(age.as_int64(), Some(30));
/// ```
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

    /// UTF-8 string (uses ArcStr for cheap cloning)
    String(ArcStr),

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

impl From<ArcStr> for Value {
    fn from(s: ArcStr) -> Self {
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

/// A hashable wrapper around [`Value`] for use in hash-based indexes.
///
/// `Value` itself cannot implement `Hash` because it contains `f64` (which has
/// NaN issues). This wrapper converts floats to their bit representation for
/// hashing, allowing values to be used as keys in hash maps and sets.
///
/// # Note on Float Equality
///
/// Two `HashableValue`s containing `f64` are considered equal if they have
/// identical bit representations. This means `NaN == NaN` (same bits) and
/// positive/negative zero are considered different.
#[derive(Clone, Debug)]
pub struct HashableValue(pub Value);

/// An orderable wrapper around [`Value`] for use in B-tree indexes and range queries.
///
/// `Value` itself cannot implement `Ord` because `f64` doesn't implement `Ord`
/// (due to NaN). This wrapper provides total ordering for comparable value types,
/// enabling use in `BTreeMap`, `BTreeSet`, and range queries.
///
/// # Supported Types
///
/// - `Int64` - standard integer ordering
/// - `Float64` - total ordering (NaN treated as greater than all other values)
/// - `String` - lexicographic ordering
/// - `Bool` - false < true
/// - `Timestamp` - chronological ordering
///
/// Other types (`Null`, `Bytes`, `List`, `Map`) return `None` from `try_from`.
///
/// # Examples
///
/// ```
/// use grafeo_common::types::{OrderableValue, Value};
/// use std::collections::BTreeSet;
///
/// let mut set = BTreeSet::new();
/// set.insert(OrderableValue::try_from(&Value::Int64(30)).unwrap());
/// set.insert(OrderableValue::try_from(&Value::Int64(10)).unwrap());
/// set.insert(OrderableValue::try_from(&Value::Int64(20)).unwrap());
///
/// // Iterates in sorted order: 10, 20, 30
/// let values: Vec<_> = set.iter().map(|v| v.as_i64().unwrap()).collect();
/// assert_eq!(values, vec![10, 20, 30]);
/// ```
#[derive(Clone, Debug)]
pub enum OrderableValue {
    /// 64-bit signed integer
    Int64(i64),
    /// 64-bit floating point with total ordering (NaN > everything)
    Float64(OrderedFloat64),
    /// UTF-8 string
    String(ArcStr),
    /// Boolean value (false < true)
    Bool(bool),
    /// Timestamp (microseconds since epoch)
    Timestamp(Timestamp),
}

/// A wrapper around `f64` that implements `Ord` with total ordering.
///
/// NaN values are treated as greater than all other values (including infinity).
/// Negative zero is considered equal to positive zero.
#[derive(Clone, Copy, Debug)]
pub struct OrderedFloat64(pub f64);

impl OrderedFloat64 {
    /// Creates a new ordered float.
    #[must_use]
    pub const fn new(f: f64) -> Self {
        Self(f)
    }

    /// Returns the inner f64 value.
    #[must_use]
    pub const fn get(&self) -> f64 {
        self.0
    }
}

impl PartialEq for OrderedFloat64 {
    fn eq(&self, other: &Self) -> bool {
        // Handle NaN: NaN equals NaN for consistency with Ord
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => true,
            (true, false) | (false, true) => false,
            (false, false) => self.0 == other.0,
        }
    }
}

impl Eq for OrderedFloat64 {}

impl PartialOrd for OrderedFloat64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Handle NaN: NaN is greater than everything (including itself for consistency)
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => {
                // Normal comparison for non-NaN values
                self.0
                    .partial_cmp(&other.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    }
}

impl Hash for OrderedFloat64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for OrderedFloat64 {
    fn from(f: f64) -> Self {
        Self(f)
    }
}

impl OrderableValue {
    /// Attempts to create an `OrderableValue` from a `Value`.
    ///
    /// Returns `None` for types that don't have a natural ordering
    /// (`Null`, `Bytes`, `List`, `Map`).
    #[must_use]
    pub fn try_from(value: &Value) -> Option<Self> {
        match value {
            Value::Int64(i) => Some(Self::Int64(*i)),
            Value::Float64(f) => Some(Self::Float64(OrderedFloat64(*f))),
            Value::String(s) => Some(Self::String(s.clone())),
            Value::Bool(b) => Some(Self::Bool(*b)),
            Value::Timestamp(t) => Some(Self::Timestamp(*t)),
            Value::Null | Value::Bytes(_) | Value::List(_) | Value::Map(_) => None,
        }
    }

    /// Converts this `OrderableValue` back to a `Value`.
    #[must_use]
    pub fn into_value(self) -> Value {
        match self {
            Self::Int64(i) => Value::Int64(i),
            Self::Float64(f) => Value::Float64(f.0),
            Self::String(s) => Value::String(s),
            Self::Bool(b) => Value::Bool(b),
            Self::Timestamp(t) => Value::Timestamp(t),
        }
    }

    /// Returns the value as an i64, if it's an Int64.
    #[must_use]
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int64(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the value as an f64, if it's a Float64.
    #[must_use]
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float64(f) => Some(f.0),
            _ => None,
        }
    }

    /// Returns the value as a string slice, if it's a String.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

impl PartialEq for OrderableValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int64(a), Self::Int64(b)) => a == b,
            (Self::Float64(a), Self::Float64(b)) => a == b,
            (Self::String(a), Self::String(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Timestamp(a), Self::Timestamp(b)) => a == b,
            // Cross-type numeric comparison
            (Self::Int64(a), Self::Float64(b)) => (*a as f64) == b.0,
            (Self::Float64(a), Self::Int64(b)) => a.0 == (*b as f64),
            _ => false,
        }
    }
}

impl Eq for OrderableValue {}

impl PartialOrd for OrderableValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderableValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Int64(a), Self::Int64(b)) => a.cmp(b),
            (Self::Float64(a), Self::Float64(b)) => a.cmp(b),
            (Self::String(a), Self::String(b)) => a.cmp(b),
            (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
            (Self::Timestamp(a), Self::Timestamp(b)) => a.cmp(b),
            // Cross-type numeric comparison
            (Self::Int64(a), Self::Float64(b)) => OrderedFloat64(*a as f64).cmp(b),
            (Self::Float64(a), Self::Int64(b)) => a.cmp(&OrderedFloat64(*b as f64)),
            // Different types: order by type ordinal for consistency
            // Order: Bool < Int64 < Float64 < String < Timestamp
            _ => self.type_ordinal().cmp(&other.type_ordinal()),
        }
    }
}

impl OrderableValue {
    /// Returns a numeric ordinal for consistent cross-type ordering.
    const fn type_ordinal(&self) -> u8 {
        match self {
            Self::Bool(_) => 0,
            Self::Int64(_) => 1,
            Self::Float64(_) => 2,
            Self::String(_) => 3,
            Self::Timestamp(_) => 4,
        }
    }
}

impl Hash for OrderableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Int64(i) => i.hash(state),
            Self::Float64(f) => f.hash(state),
            Self::String(s) => s.hash(state),
            Self::Bool(b) => b.hash(state),
            Self::Timestamp(t) => t.hash(state),
        }
    }
}

impl HashableValue {
    /// Creates a new hashable value from a value.
    #[must_use]
    pub fn new(value: Value) -> Self {
        Self(value)
    }

    /// Returns a reference to the inner value.
    #[must_use]
    pub fn inner(&self) -> &Value {
        &self.0
    }

    /// Consumes the wrapper and returns the inner value.
    #[must_use]
    pub fn into_inner(self) -> Value {
        self.0
    }
}

impl Hash for HashableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the discriminant first
        std::mem::discriminant(&self.0).hash(state);

        match &self.0 {
            Value::Null => {}
            Value::Bool(b) => b.hash(state),
            Value::Int64(i) => i.hash(state),
            Value::Float64(f) => {
                // Use bit representation for hashing floats
                f.to_bits().hash(state);
            }
            Value::String(s) => s.hash(state),
            Value::Bytes(b) => b.hash(state),
            Value::Timestamp(t) => t.hash(state),
            Value::List(l) => {
                l.len().hash(state);
                for v in l.iter() {
                    HashableValue(v.clone()).hash(state);
                }
            }
            Value::Map(m) => {
                m.len().hash(state);
                for (k, v) in m.iter() {
                    k.hash(state);
                    HashableValue(v.clone()).hash(state);
                }
            }
        }
    }
}

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Value::Float64(a), Value::Float64(b)) => {
                // Compare by bits for consistent hash/eq behavior
                a.to_bits() == b.to_bits()
            }
            (Value::List(a), Value::List(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| HashableValue(x.clone()) == HashableValue(y.clone()))
            }
            (Value::Map(a), Value::Map(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter().all(|(k, v)| {
                    b.get(k)
                        .is_some_and(|bv| HashableValue(v.clone()) == HashableValue(bv.clone()))
                })
            }
            // For other types, use normal Value equality
            _ => self.0 == other.0,
        }
    }
}

impl Eq for HashableValue {}

impl From<Value> for HashableValue {
    fn from(value: Value) -> Self {
        Self(value)
    }
}

impl From<HashableValue> for Value {
    fn from(hv: HashableValue) -> Self {
        hv.0
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

        assert_eq!(Value::Float64(1.234).as_float64(), Some(1.234));
        assert_eq!(Value::String("hello".into()).as_str(), Some("hello"));
    }

    #[test]
    fn test_value_from_conversions() {
        let v: Value = true.into();
        assert_eq!(v.as_bool(), Some(true));

        let v: Value = 42i64.into();
        assert_eq!(v.as_int64(), Some(42));

        let v: Value = 1.234f64.into();
        assert_eq!(v.as_float64(), Some(1.234));

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

        // Keys should be comparable ("age" < "name" alphabetically)
        assert!(key2 < key);
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

    #[test]
    fn test_hashable_value_basic() {
        use std::collections::HashMap;

        let mut map: HashMap<HashableValue, i32> = HashMap::new();

        // Test various value types as keys
        map.insert(HashableValue::new(Value::Int64(42)), 1);
        map.insert(HashableValue::new(Value::String("test".into())), 2);
        map.insert(HashableValue::new(Value::Bool(true)), 3);
        map.insert(HashableValue::new(Value::Float64(3.14)), 4);

        assert_eq!(map.get(&HashableValue::new(Value::Int64(42))), Some(&1));
        assert_eq!(
            map.get(&HashableValue::new(Value::String("test".into()))),
            Some(&2)
        );
        assert_eq!(map.get(&HashableValue::new(Value::Bool(true))), Some(&3));
        assert_eq!(map.get(&HashableValue::new(Value::Float64(3.14))), Some(&4));
    }

    #[test]
    fn test_hashable_value_float_edge_cases() {
        use std::collections::HashMap;

        let mut map: HashMap<HashableValue, i32> = HashMap::new();

        // NaN should be hashable and equal to itself (same bits)
        let nan = f64::NAN;
        map.insert(HashableValue::new(Value::Float64(nan)), 1);
        assert_eq!(map.get(&HashableValue::new(Value::Float64(nan))), Some(&1));

        // Positive and negative zero have different bits
        let pos_zero = 0.0f64;
        let neg_zero = -0.0f64;
        map.insert(HashableValue::new(Value::Float64(pos_zero)), 2);
        map.insert(HashableValue::new(Value::Float64(neg_zero)), 3);
        assert_eq!(
            map.get(&HashableValue::new(Value::Float64(pos_zero))),
            Some(&2)
        );
        assert_eq!(
            map.get(&HashableValue::new(Value::Float64(neg_zero))),
            Some(&3)
        );
    }

    #[test]
    fn test_hashable_value_equality() {
        let v1 = HashableValue::new(Value::Int64(42));
        let v2 = HashableValue::new(Value::Int64(42));
        let v3 = HashableValue::new(Value::Int64(43));

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_hashable_value_inner() {
        let hv = HashableValue::new(Value::String("hello".into()));
        assert_eq!(hv.inner().as_str(), Some("hello"));

        let v = hv.into_inner();
        assert_eq!(v.as_str(), Some("hello"));
    }

    #[test]
    fn test_hashable_value_conversions() {
        let v = Value::Int64(42);
        let hv: HashableValue = v.clone().into();
        let v2: Value = hv.into();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_orderable_value_try_from() {
        // Supported types
        assert!(OrderableValue::try_from(&Value::Int64(42)).is_some());
        assert!(OrderableValue::try_from(&Value::Float64(3.14)).is_some());
        assert!(OrderableValue::try_from(&Value::String("test".into())).is_some());
        assert!(OrderableValue::try_from(&Value::Bool(true)).is_some());
        assert!(OrderableValue::try_from(&Value::Timestamp(Timestamp::from_secs(1000))).is_some());

        // Unsupported types
        assert!(OrderableValue::try_from(&Value::Null).is_none());
        assert!(OrderableValue::try_from(&Value::Bytes(vec![1, 2, 3].into())).is_none());
        assert!(OrderableValue::try_from(&Value::List(vec![].into())).is_none());
        assert!(OrderableValue::try_from(&Value::Map(BTreeMap::new().into())).is_none());
    }

    #[test]
    fn test_orderable_value_ordering() {
        use std::collections::BTreeSet;

        // Test integer ordering
        let mut set = BTreeSet::new();
        set.insert(OrderableValue::try_from(&Value::Int64(30)).unwrap());
        set.insert(OrderableValue::try_from(&Value::Int64(10)).unwrap());
        set.insert(OrderableValue::try_from(&Value::Int64(20)).unwrap());

        let values: Vec<_> = set.iter().filter_map(|v| v.as_i64()).collect();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn test_orderable_value_float_ordering() {
        let v1 = OrderableValue::try_from(&Value::Float64(1.0)).unwrap();
        let v2 = OrderableValue::try_from(&Value::Float64(2.0)).unwrap();
        let v_nan = OrderableValue::try_from(&Value::Float64(f64::NAN)).unwrap();
        let v_inf = OrderableValue::try_from(&Value::Float64(f64::INFINITY)).unwrap();

        assert!(v1 < v2);
        assert!(v2 < v_inf);
        assert!(v_inf < v_nan); // NaN is greater than everything
        assert!(v_nan == v_nan); // NaN equals itself for total ordering
    }

    #[test]
    fn test_orderable_value_string_ordering() {
        let a = OrderableValue::try_from(&Value::String("apple".into())).unwrap();
        let b = OrderableValue::try_from(&Value::String("banana".into())).unwrap();
        let c = OrderableValue::try_from(&Value::String("cherry".into())).unwrap();

        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_orderable_value_into_value() {
        let original = Value::Int64(42);
        let orderable = OrderableValue::try_from(&original).unwrap();
        let back = orderable.into_value();
        assert_eq!(original, back);

        let original = Value::Float64(3.14);
        let orderable = OrderableValue::try_from(&original).unwrap();
        let back = orderable.into_value();
        assert_eq!(original, back);

        let original = Value::String("test".into());
        let orderable = OrderableValue::try_from(&original).unwrap();
        let back = orderable.into_value();
        assert_eq!(original, back);
    }

    #[test]
    fn test_orderable_value_cross_type_numeric() {
        // Int64 and Float64 should be comparable
        let i = OrderableValue::try_from(&Value::Int64(10)).unwrap();
        let f = OrderableValue::try_from(&Value::Float64(10.0)).unwrap();

        // They should compare as equal
        assert_eq!(i, f);

        let f2 = OrderableValue::try_from(&Value::Float64(10.5)).unwrap();
        assert!(i < f2);
    }

    #[test]
    fn test_ordered_float64_nan_handling() {
        let nan1 = OrderedFloat64::new(f64::NAN);
        let nan2 = OrderedFloat64::new(f64::NAN);
        let inf = OrderedFloat64::new(f64::INFINITY);
        let neg_inf = OrderedFloat64::new(f64::NEG_INFINITY);
        let zero = OrderedFloat64::new(0.0);

        // NaN equals itself
        assert_eq!(nan1, nan2);

        // Ordering: -inf < 0 < inf < nan
        assert!(neg_inf < zero);
        assert!(zero < inf);
        assert!(inf < nan1);
    }
}
