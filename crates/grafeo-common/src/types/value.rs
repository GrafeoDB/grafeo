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

use super::{Date, Duration, Time, Timestamp, ZonedDatetime};

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

impl std::borrow::Borrow<str> for PropertyKey {
    fn borrow(&self) -> &str {
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
/// let name = Value::from("Alix");
/// let age = Value::from(30i64);
/// let active = Value::from(true);
///
/// // Check types
/// assert!(name.as_str().is_some());
/// assert_eq!(age.as_int64(), Some(30));
/// ```
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
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

    /// Calendar date (days since 1970-01-01)
    Date(Date),

    /// Time of day with optional UTC offset
    Time(Time),

    /// ISO 8601 duration (months, days, nanos)
    Duration(Duration),

    /// Datetime with a fixed UTC offset
    ZonedDatetime(ZonedDatetime),

    /// Ordered list of values
    List(Arc<[Value]>),

    /// Key-value map (uses BTreeMap for deterministic ordering)
    Map(Arc<BTreeMap<PropertyKey, Value>>),

    /// Fixed-size vector of 32-bit floats for embeddings.
    ///
    /// Uses f32 for 4x compression vs f64. Arc for cheap cloning.
    /// Dimension is implicit from length. Common dimensions: 384, 768, 1536.
    Vector(Arc<[f32]>),

    /// Graph path: alternating sequence of nodes and edges.
    ///
    /// Nodes and edges are stored as lists of values (typically node/edge maps
    /// with `_id`, `_labels`/`_type`, and properties). The invariant is that
    /// `edges.len() == nodes.len() - 1` for a valid path.
    Path {
        /// Nodes along the path, from source to target.
        nodes: Arc<[Value]>,
        /// Edges along the path, connecting consecutive nodes.
        edges: Arc<[Value]>,
    },

    /// Grow-only counter (GCounter) for conflict-free distributed counting.
    ///
    /// Stores per-replica contribution counts. The logical value is the sum
    /// of all entries. Merge is per-replica max, making it commutative and
    /// idempotent.
    ///
    /// Key: replica ID string. Value: that replica's running total (grows
    /// monotonically).
    GCounter(Arc<std::collections::HashMap<String, u64>>),

    /// Positive-negative counter (ON-Counter) for distributed increment and
    /// decrement.
    ///
    /// Two grow-only maps track positive and negative contributions
    /// independently. The logical value is `sum(pos) − sum(neg)`. Merge is
    /// per-replica max on each map independently.
    OnCounter {
        /// Per-replica positive contributions (increments).
        pos: Arc<std::collections::HashMap<String, u64>>,
        /// Per-replica negative contributions (decrements, stored as positive
        /// magnitudes).
        neg: Arc<std::collections::HashMap<String, u64>>,
    },
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

    /// Returns the date value if this is a Date, otherwise None.
    #[inline]
    #[must_use]
    pub const fn as_date(&self) -> Option<Date> {
        match self {
            Value::Date(d) => Some(*d),
            _ => None,
        }
    }

    /// Returns the time value if this is a Time, otherwise None.
    #[inline]
    #[must_use]
    pub const fn as_time(&self) -> Option<Time> {
        match self {
            Value::Time(t) => Some(*t),
            _ => None,
        }
    }

    /// Returns the duration value if this is a Duration, otherwise None.
    #[inline]
    #[must_use]
    pub const fn as_duration(&self) -> Option<Duration> {
        match self {
            Value::Duration(d) => Some(*d),
            _ => None,
        }
    }

    /// Returns the zoned datetime value if this is a ZonedDatetime, otherwise None.
    #[inline]
    #[must_use]
    pub const fn as_zoned_datetime(&self) -> Option<ZonedDatetime> {
        match self {
            Value::ZonedDatetime(zdt) => Some(*zdt),
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

    /// Returns the vector if this is a Vector, otherwise None.
    #[inline]
    #[must_use]
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Value::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the path components if this is a Path, otherwise None.
    #[inline]
    #[must_use]
    pub fn as_path(&self) -> Option<(&[Value], &[Value])> {
        match self {
            Value::Path { nodes, edges } => Some((nodes, edges)),
            _ => None,
        }
    }

    /// Returns true if this is a vector type.
    #[inline]
    #[must_use]
    pub const fn is_vector(&self) -> bool {
        matches!(self, Value::Vector(_))
    }

    /// Returns the vector dimensions if this is a Vector.
    #[inline]
    #[must_use]
    pub fn vector_dimensions(&self) -> Option<usize> {
        match self {
            Value::Vector(v) => Some(v.len()),
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
            Value::Date(_) => "DATE",
            Value::Time(_) => "TIME",
            Value::Duration(_) => "DURATION",
            Value::ZonedDatetime(_) => "ZONED DATETIME",
            Value::List(_) => "LIST",
            Value::Map(_) => "MAP",
            Value::Vector(_) => "VECTOR",
            Value::Path { .. } => "PATH",
            Value::GCounter(_) => "GCOUNTER",
            Value::OnCounter { .. } => "PNCOUNTER",
        }
    }

    /// Serializes this value to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the value cannot be encoded (e.g. deeply nested structures).
    pub fn serialize(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
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

    /// Returns an estimate of the heap memory used by this value in bytes.
    ///
    /// This is a best-effort approximation used for enforcing property size
    /// limits. Fixed-size types return 0 (only the enum discriminant, which
    /// is already accounted for on the stack).
    #[must_use]
    pub fn estimated_size_bytes(&self) -> usize {
        match self {
            Value::Null
            | Value::Bool(_)
            | Value::Int64(_)
            | Value::Float64(_)
            | Value::Timestamp(_)
            | Value::Date(_)
            | Value::Time(_)
            | Value::Duration(_)
            | Value::ZonedDatetime(_) => 0,
            Value::String(s) => s.len(),
            Value::Bytes(b) => b.len(),
            Value::Vector(v) => v.len() * size_of::<f32>(),
            Value::List(items) => {
                items.iter().map(Value::estimated_size_bytes).sum::<usize>()
                    + items.len() * size_of::<Value>()
            }
            Value::Map(m) => m
                .iter()
                .map(|(k, v)| k.as_ref().len() + v.estimated_size_bytes() + size_of::<Value>())
                .sum(),
            Value::Path { nodes, edges } => {
                let n: usize = nodes.iter().map(Value::estimated_size_bytes).sum::<usize>()
                    + nodes.len() * size_of::<Value>();
                let e: usize = edges.iter().map(Value::estimated_size_bytes).sum::<usize>()
                    + edges.len() * size_of::<Value>();
                n + e
            }
            Value::GCounter(m) => m.keys().map(|k| k.len() + size_of::<u64>()).sum(),
            Value::OnCounter { pos, neg } => {
                let p: usize = pos.keys().map(|k| k.len() + size_of::<u64>()).sum();
                let n: usize = neg.keys().map(|k| k.len() + size_of::<u64>()).sum();
                p + n
            }
        }
    }

    /// String / bytes payload bytes only (subset of [`Self::estimated_size_bytes`]).
    ///
    /// Used by LPG residency accounting to attribute ArcStr/bytes separately
    /// from map-slot and compressed-column bytes. Nested lists/maps recurse.
    #[must_use]
    pub fn string_payload_bytes(&self) -> usize {
        match self {
            Value::String(s) => s.len(),
            Value::Bytes(b) => b.len(),
            Value::List(items) => items.iter().map(Value::string_payload_bytes).sum(),
            Value::Map(m) => m
                .iter()
                .map(|(k, v)| k.as_ref().len() + v.string_payload_bytes())
                .sum(),
            Value::Path { nodes, edges } => {
                nodes.iter().map(Value::string_payload_bytes).sum::<usize>()
                    + edges.iter().map(Value::string_payload_bytes).sum::<usize>()
            }
            _ => 0,
        }
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
            Value::Date(d) => write!(f, "Date({d})"),
            Value::Time(t) => write!(f, "Time({t})"),
            Value::Duration(d) => write!(f, "Duration({d})"),
            Value::ZonedDatetime(zdt) => write!(f, "ZonedDatetime({zdt})"),
            Value::List(l) => write!(f, "List({l:?})"),
            Value::Map(m) => write!(f, "Map({m:?})"),
            Value::Vector(v) => write!(
                f,
                "Vector([{}; {} dims])",
                v.first().unwrap_or(&0.0),
                v.len()
            ),
            Value::Path { nodes, edges } => {
                write!(f, "Path({} nodes, {} edges)", nodes.len(), edges.len())
            }
            Value::GCounter(counts) => {
                let total: u64 = counts.values().sum();
                write!(f, "GCounter(total={total}, replicas={})", counts.len())
            }
            Value::OnCounter { pos, neg } => {
                let pos_sum: u128 = pos.values().copied().map(u128::from).sum();
                let neg_sum: u128 = neg.values().copied().map(u128::from).sum();
                if pos_sum >= neg_sum {
                    write!(f, "OnCounter(net={})", pos_sum - neg_sum)
                } else {
                    write!(f, "OnCounter(net=-{})", neg_sum - pos_sum)
                }
            }
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
            Value::Date(d) => write!(f, "{d}"),
            Value::Time(t) => write!(f, "{t}"),
            Value::Duration(d) => write!(f, "{d}"),
            Value::ZonedDatetime(zdt) => write!(f, "{zdt}"),
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
            Value::Vector(v) => {
                write!(f, "vector([")?;
                let show_count = v.len().min(3);
                for (i, val) in v.iter().take(show_count).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val}")?;
                }
                if v.len() > 3 {
                    write!(f, ", ... ({} dims)", v.len())?;
                }
                write!(f, "])")
            }
            Value::Path { nodes, edges } => {
                // Display path as alternating node-edge-node sequence
                write!(f, "<")?;
                for (i, node) in nodes.iter().enumerate() {
                    if i > 0
                        && let Some(edge) = edges.get(i - 1)
                    {
                        write!(f, "-[{edge}]-")?;
                    }
                    write!(f, "({node})")?;
                }
                write!(f, ">")
            }
            Value::GCounter(counts) => {
                let total: u64 = counts.values().sum();
                write!(f, "GCounter({total})")
            }
            Value::OnCounter { pos, neg } => {
                let pos_sum: u128 = pos.values().copied().map(u128::from).sum();
                let neg_sum: u128 = neg.values().copied().map(u128::from).sum();
                if pos_sum >= neg_sum {
                    write!(f, "OnCounter({})", pos_sum - neg_sum)
                } else {
                    write!(f, "OnCounter(-{})", neg_sum - pos_sum)
                }
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

impl From<Date> for Value {
    fn from(d: Date) -> Self {
        Value::Date(d)
    }
}

impl From<Time> for Value {
    fn from(t: Time) -> Self {
        Value::Time(t)
    }
}

impl From<Duration> for Value {
    fn from(d: Duration) -> Self {
        Value::Duration(d)
    }
}

impl From<ZonedDatetime> for Value {
    fn from(zdt: ZonedDatetime) -> Self {
        Value::ZonedDatetime(zdt)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::List(v.into_iter().map(Into::into).collect())
    }
}

impl From<&[f32]> for Value {
    fn from(v: &[f32]) -> Self {
        Value::Vector(v.into())
    }
}

impl From<Arc<[f32]>> for Value {
    fn from(v: Arc<[f32]>) -> Self {
        Value::Vector(v)
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
/// Other types (`Null`, `Bytes`, `List`, `Map`) return `Err(())` from `try_from`.
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
#[non_exhaustive]
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
    /// Calendar date (days since epoch)
    Date(Date),
    /// Time of day with optional offset
    Time(Time),
    /// Datetime with a fixed UTC offset
    ZonedDatetime(ZonedDatetime),
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

impl TryFrom<&Value> for OrderableValue {
    type Error = ();

    /// Attempts to create an `OrderableValue` from a `Value`.
    ///
    /// Returns `Err(())` for types that don't have a natural ordering
    /// (`Null`, `Bytes`, `List`, `Map`, `Vector`).
    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        match value {
            Value::Int64(i) => Ok(Self::Int64(*i)),
            Value::Float64(f) => Ok(Self::Float64(OrderedFloat64(*f))),
            Value::String(s) => Ok(Self::String(s.clone())),
            Value::Bool(b) => Ok(Self::Bool(*b)),
            Value::Timestamp(t) => Ok(Self::Timestamp(*t)),
            Value::Date(d) => Ok(Self::Date(*d)),
            Value::Time(t) => Ok(Self::Time(*t)),
            Value::ZonedDatetime(zdt) => Ok(Self::ZonedDatetime(*zdt)),
            Value::Null
            | Value::Bytes(_)
            | Value::Duration(_)
            | Value::List(_)
            | Value::Map(_)
            | Value::Vector(_)
            | Value::Path { .. }
            | Value::GCounter(_)
            | Value::OnCounter { .. } => Err(()),
        }
    }
}

impl OrderableValue {
    /// Converts this `OrderableValue` back to a `Value`.
    #[must_use]
    pub fn into_value(self) -> Value {
        match self {
            Self::Int64(i) => Value::Int64(i),
            Self::Float64(f) => Value::Float64(f.0),
            Self::String(s) => Value::String(s),
            Self::Bool(b) => Value::Bool(b),
            Self::Timestamp(t) => Value::Timestamp(t),
            Self::Date(d) => Value::Date(d),
            Self::Time(t) => Value::Time(t),
            Self::ZonedDatetime(zdt) => Value::ZonedDatetime(zdt),
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
            (Self::Date(a), Self::Date(b)) => a == b,
            (Self::Time(a), Self::Time(b)) => a == b,
            (Self::ZonedDatetime(a), Self::ZonedDatetime(b)) => a == b,
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
            (Self::Date(a), Self::Date(b)) => a.cmp(b),
            (Self::Time(a), Self::Time(b)) => a.cmp(b),
            (Self::ZonedDatetime(a), Self::ZonedDatetime(b)) => a.cmp(b),
            // Cross-type numeric comparison
            (Self::Int64(a), Self::Float64(b)) => OrderedFloat64(*a as f64).cmp(b),
            (Self::Float64(a), Self::Int64(b)) => a.cmp(&OrderedFloat64(*b as f64)),
            // Different types: order by type ordinal for consistency
            // Order: Bool < Int64 < Float64 < String < Timestamp < Date < Time < ZonedDatetime
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
            Self::Date(_) => 5,
            Self::Time(_) => 6,
            Self::ZonedDatetime(_) => 7,
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
            Self::Date(d) => d.hash(state),
            Self::Time(t) => t.hash(state),
            Self::ZonedDatetime(zdt) => zdt.hash(state),
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

/// Hashes a `Value` by reference without cloning nested values.
fn hash_value<H: Hasher>(value: &Value, state: &mut H) {
    std::mem::discriminant(value).hash(state);

    match value {
        Value::Null => {}
        Value::Bool(b) => b.hash(state),
        Value::Int64(i) => i.hash(state),
        Value::Float64(f) => f.to_bits().hash(state),
        Value::String(s) => s.hash(state),
        Value::Bytes(b) => b.hash(state),
        Value::Timestamp(t) => t.hash(state),
        Value::Date(d) => d.hash(state),
        Value::Time(t) => t.hash(state),
        Value::Duration(d) => d.hash(state),
        Value::ZonedDatetime(zdt) => zdt.hash(state),
        Value::List(l) => {
            l.len().hash(state);
            for v in l.iter() {
                hash_value(v, state);
            }
        }
        Value::Map(m) => {
            m.len().hash(state);
            for (k, v) in m.iter() {
                k.hash(state);
                hash_value(v, state);
            }
        }
        Value::Vector(v) => {
            v.len().hash(state);
            for &f in v.iter() {
                f.to_bits().hash(state);
            }
        }
        Value::Path { nodes, edges } => {
            nodes.len().hash(state);
            for v in nodes.iter() {
                hash_value(v, state);
            }
            edges.len().hash(state);
            for v in edges.iter() {
                hash_value(v, state);
            }
        }
        Value::GCounter(counts) => {
            // Sort keys for deterministic hash order.
            let mut pairs: Vec<_> = counts.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            pairs.len().hash(state);
            for (k, v) in pairs {
                k.hash(state);
                v.hash(state);
            }
        }
        Value::OnCounter { pos, neg } => {
            let mut pos_pairs: Vec<_> = pos.iter().collect();
            pos_pairs.sort_by_key(|(k, _)| k.as_str());
            pos_pairs.len().hash(state);
            for (k, v) in pos_pairs {
                k.hash(state);
                v.hash(state);
            }
            let mut neg_pairs: Vec<_> = neg.iter().collect();
            neg_pairs.sort_by_key(|(k, _)| k.as_str());
            neg_pairs.len().hash(state);
            for (k, v) in neg_pairs {
                k.hash(state);
                v.hash(state);
            }
        }
    }
}

impl Hash for HashableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_value(&self.0, state);
    }
}

/// Compares two `Value`s for hashable equality by reference (bit-equal floats).
fn values_hash_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Float64(a), Value::Float64(b)) => a.to_bits() == b.to_bits(),
        (Value::List(a), Value::List(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_hash_eq(x, y))
        }
        (Value::Map(a), Value::Map(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|bv| values_hash_eq(v, bv)))
        }
        (Value::Vector(a), Value::Vector(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
        }
        (
            Value::Path {
                nodes: an,
                edges: ae,
            },
            Value::Path {
                nodes: bn,
                edges: be,
            },
        ) => {
            an.len() == bn.len()
                && ae.len() == be.len()
                && an.iter().zip(bn.iter()).all(|(x, y)| values_hash_eq(x, y))
                && ae.iter().zip(be.iter()).all(|(x, y)| values_hash_eq(x, y))
        }
        _ => a == b,
    }
}

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        values_hash_eq(&self.0, &other.0)
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
            Value::Int64(i64::MIN),
            Value::Int64(0),
            Value::Float64(std::f64::consts::PI),
            Value::String("hello world".into()),
            Value::Bytes(vec![0, 1, 2, 255].into()),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
        ];

        for v in values {
            let bytes = v.serialize().unwrap();
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
        assert_eq!(
            Value::Date(Date::from_ymd(2024, 1, 15).unwrap()).type_name(),
            "DATE"
        );
        assert_eq!(
            Value::Time(Time::from_hms(12, 0, 0).unwrap()).type_name(),
            "TIME"
        );
        assert_eq!(Value::Duration(Duration::default()).type_name(), "DURATION");
        assert_eq!(Value::List(vec![].into()).type_name(), "LIST");
        assert_eq!(Value::Map(BTreeMap::new().into()).type_name(), "MAP");
        assert_eq!(Value::Vector(vec![].into()).type_name(), "VECTOR");
    }

    #[test]
    fn test_value_vector() {
        // Create vector directly (Vec<f32>.into() would create List due to generic impl)
        let v = Value::Vector(vec![0.1f32, 0.2, 0.3].into());
        assert!(v.is_vector());
        assert_eq!(v.vector_dimensions(), Some(3));
        assert_eq!(v.as_vector(), Some(&[0.1f32, 0.2, 0.3][..]));

        // From slice
        let slice: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let v2: Value = slice.into();
        assert!(v2.is_vector());
        assert_eq!(v2.vector_dimensions(), Some(4));

        // From Arc<[f32]>
        let arc: Arc<[f32]> = vec![5.0f32, 6.0].into();
        let v3: Value = arc.into();
        assert!(v3.is_vector());
        assert_eq!(v3.vector_dimensions(), Some(2));

        // Non-vector returns None
        assert!(!Value::Int64(42).is_vector());
        assert_eq!(Value::Int64(42).as_vector(), None);
        assert_eq!(Value::Int64(42).vector_dimensions(), None);
    }

    #[test]
    fn test_hashable_value_vector() {
        use std::collections::HashMap;

        let mut map: HashMap<HashableValue, i32> = HashMap::new();

        let v1 = HashableValue::new(Value::Vector(vec![0.1f32, 0.2, 0.3].into()));
        let v2 = HashableValue::new(Value::Vector(vec![0.1f32, 0.2, 0.3].into()));
        let v3 = HashableValue::new(Value::Vector(vec![0.4f32, 0.5, 0.6].into()));

        map.insert(v1.clone(), 1);

        // Same vector should hash to same bucket
        assert_eq!(map.get(&v2), Some(&1));

        // Different vector should not match
        assert_eq!(map.get(&v3), None);

        // v1 and v2 should be equal
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_orderable_value_vector_unsupported() {
        // Vectors don't have a natural ordering, so try_from should return Err
        let v = Value::Vector(vec![0.1f32, 0.2, 0.3].into());
        assert!(OrderableValue::try_from(&v).is_err());
    }

    #[test]
    fn test_hashable_value_basic() {
        use std::collections::HashMap;

        let mut map: HashMap<HashableValue, i32> = HashMap::new();

        // Test various value types as keys
        map.insert(HashableValue::new(Value::Int64(42)), 1);
        map.insert(HashableValue::new(Value::String("test".into())), 2);
        map.insert(HashableValue::new(Value::Bool(true)), 3);
        map.insert(HashableValue::new(Value::Float64(std::f64::consts::PI)), 4);

        assert_eq!(map.get(&HashableValue::new(Value::Int64(42))), Some(&1));
        assert_eq!(
            map.get(&HashableValue::new(Value::String("test".into()))),
            Some(&2)
        );
        assert_eq!(map.get(&HashableValue::new(Value::Bool(true))), Some(&3));
        assert_eq!(
            map.get(&HashableValue::new(Value::Float64(std::f64::consts::PI))),
            Some(&4)
        );
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
        assert!(OrderableValue::try_from(&Value::Int64(42)).is_ok());
        assert!(OrderableValue::try_from(&Value::Float64(std::f64::consts::PI)).is_ok());
        assert!(OrderableValue::try_from(&Value::String("test".into())).is_ok());
        assert!(OrderableValue::try_from(&Value::Bool(true)).is_ok());
        assert!(OrderableValue::try_from(&Value::Timestamp(Timestamp::from_secs(1000))).is_ok());
        assert!(
            OrderableValue::try_from(&Value::Date(Date::from_ymd(2024, 1, 15).unwrap())).is_ok()
        );
        assert!(OrderableValue::try_from(&Value::Time(Time::from_hms(12, 0, 0).unwrap())).is_ok());

        // Unsupported types
        assert!(OrderableValue::try_from(&Value::Null).is_err());
        assert!(OrderableValue::try_from(&Value::Bytes(vec![1, 2, 3].into())).is_err());
        assert!(OrderableValue::try_from(&Value::Duration(Duration::default())).is_err());
        assert!(OrderableValue::try_from(&Value::List(vec![].into())).is_err());
        assert!(OrderableValue::try_from(&Value::Map(BTreeMap::new().into())).is_err());
    }

    #[test]
    fn test_orderable_value_ordering() {
        use std::collections::BTreeSet;

        // Test integer ordering including i64::MIN and i64::MAX
        let mut set = BTreeSet::new();
        set.insert(OrderableValue::try_from(&Value::Int64(30)).unwrap());
        set.insert(OrderableValue::try_from(&Value::Int64(10)).unwrap());
        set.insert(OrderableValue::try_from(&Value::Int64(20)).unwrap());
        set.insert(OrderableValue::try_from(&Value::Int64(i64::MIN)).unwrap());
        set.insert(OrderableValue::try_from(&Value::Int64(i64::MAX)).unwrap());

        let values: Vec<_> = set.iter().filter_map(|v| v.as_i64()).collect();
        assert_eq!(values, vec![i64::MIN, 10, 20, 30, i64::MAX]);
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

        let original = Value::Float64(std::f64::consts::PI);
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

    #[test]
    fn test_value_temporal_accessors() {
        let date = Date::from_ymd(2024, 3, 15).unwrap();
        let time = Time::from_hms(14, 30, 0).unwrap();
        let dur = Duration::from_months(3);

        let vd = Value::Date(date);
        let vt = Value::Time(time);
        let vr = Value::Duration(dur);

        assert_eq!(vd.as_date(), Some(date));
        assert_eq!(vt.as_time(), Some(time));
        assert_eq!(vr.as_duration(), Some(dur));

        // Wrong type returns None
        assert_eq!(vd.as_time(), None);
        assert_eq!(vt.as_date(), None);
        assert_eq!(vd.as_duration(), None);
    }

    #[test]
    fn test_value_temporal_from_conversions() {
        let date = Date::from_ymd(2024, 1, 15).unwrap();
        let v: Value = date.into();
        assert_eq!(v.as_date(), Some(date));

        let time = Time::from_hms(10, 30, 0).unwrap();
        let v: Value = time.into();
        assert_eq!(v.as_time(), Some(time));

        let dur = Duration::from_days(7);
        let v: Value = dur.into();
        assert_eq!(v.as_duration(), Some(dur));
    }

    #[test]
    fn test_value_temporal_display() {
        let v = Value::Date(Date::from_ymd(2024, 3, 15).unwrap());
        assert_eq!(format!("{v}"), "2024-03-15");

        let v = Value::Time(Time::from_hms(14, 30, 0).unwrap());
        assert_eq!(format!("{v}"), "14:30:00");

        let v = Value::Duration(Duration::from_days(7));
        assert_eq!(format!("{v}"), "P7D");
    }

    #[test]
    fn test_value_temporal_serialization_roundtrip() {
        let values = vec![
            Value::Date(Date::from_ymd(2024, 6, 15).unwrap()),
            Value::Time(Time::from_hms(23, 59, 59).unwrap()),
            Value::Duration(Duration::new(1, 2, 3_000_000_000)),
        ];

        for v in values {
            let bytes = v.serialize().unwrap();
            let decoded = Value::deserialize(&bytes).unwrap();
            assert_eq!(v, decoded);
        }
    }

    #[test]
    fn test_orderable_value_date_ordering() {
        let d1 =
            OrderableValue::try_from(&Value::Date(Date::from_ymd(2024, 1, 1).unwrap())).unwrap();
        let d2 =
            OrderableValue::try_from(&Value::Date(Date::from_ymd(2024, 6, 15).unwrap())).unwrap();
        assert!(d1 < d2);

        let back = d1.into_value();
        assert_eq!(back.as_date(), Some(Date::from_ymd(2024, 1, 1).unwrap()));
    }

    #[test]
    fn test_hashable_value_temporal() {
        use std::collections::HashMap;

        let mut map: HashMap<HashableValue, i32> = HashMap::new();

        let date_val = Value::Date(Date::from_ymd(2024, 3, 15).unwrap());
        map.insert(HashableValue::new(date_val.clone()), 1);
        assert_eq!(map.get(&HashableValue::new(date_val)), Some(&1));

        let time_val = Value::Time(Time::from_hms(12, 0, 0).unwrap());
        map.insert(HashableValue::new(time_val.clone()), 2);
        assert_eq!(map.get(&HashableValue::new(time_val)), Some(&2));

        let dur_val = Value::Duration(Duration::from_months(6));
        map.insert(HashableValue::new(dur_val.clone()), 3);
        assert_eq!(map.get(&HashableValue::new(dur_val)), Some(&3));
    }

    // --- T2-15: Value::Path tests ---

    #[test]
    fn test_value_path_construction_and_equality() {
        let path1 = Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into(),
            edges: vec![Value::String("KNOWS".into()), Value::String("LIKES".into())].into(),
        };
        let path2 = Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into(),
            edges: vec![Value::String("KNOWS".into()), Value::String("LIKES".into())].into(),
        };
        let path3 = Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
            edges: vec![Value::String("KNOWS".into())].into(),
        };

        assert_eq!(path1, path2, "Identical paths should be equal");
        assert_ne!(path1, path3, "Different paths should not be equal");
    }

    #[test]
    fn test_value_path_type_name() {
        let path = Value::Path {
            nodes: vec![Value::Int64(1)].into(),
            edges: vec![].into(),
        };
        assert_eq!(path.type_name(), "PATH");
    }

    #[test]
    fn test_value_path_serialization_roundtrip() {
        let path = Value::Path {
            nodes: vec![
                Value::String("node_a".into()),
                Value::String("node_b".into()),
            ]
            .into(),
            edges: vec![Value::String("CONNECTS".into())].into(),
        };

        let bytes = path.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(path, decoded);
    }

    #[test]
    fn test_value_path_hashing() {
        use std::collections::HashMap;

        let path1 = Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
            edges: vec![Value::String("KNOWS".into())].into(),
        };
        let path2 = Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
            edges: vec![Value::String("KNOWS".into())].into(),
        };

        let mut map: HashMap<HashableValue, i32> = HashMap::new();
        map.insert(HashableValue::new(path1), 42);
        assert_eq!(map.get(&HashableValue::new(path2)), Some(&42));
    }

    // --- T2-16: Collection nesting tests ---

    #[test]
    fn test_nested_list_serialization_roundtrip() {
        let nested = Value::List(
            vec![
                Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
                Value::List(vec![Value::Int64(3), Value::Int64(4)].into()),
            ]
            .into(),
        );

        let bytes = nested.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(nested, decoded);
    }

    #[test]
    fn test_map_with_entries_serialization_roundtrip() {
        let mut entries = BTreeMap::new();
        entries.insert(PropertyKey::new("name"), Value::String("Alix".into()));
        entries.insert(PropertyKey::new("age"), Value::Int64(30));
        entries.insert(PropertyKey::new("active"), Value::Bool(true));

        let map = Value::Map(entries.into());

        let bytes = map.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(map, decoded);
    }

    #[test]
    fn test_mixed_type_list_serialization_roundtrip() {
        let mixed = Value::List(
            vec![
                Value::Int64(1),
                Value::String("hello".into()),
                Value::Null,
                Value::Bool(false),
                Value::Float64(3.125),
            ]
            .into(),
        );

        let bytes = mixed.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(mixed, decoded);
    }

    #[test]
    fn test_map_with_nested_list() {
        let mut entries = BTreeMap::new();
        entries.insert(
            PropertyKey::new("tags"),
            Value::List(vec![Value::String("a".into()), Value::String("b".into())].into()),
        );
        entries.insert(PropertyKey::new("count"), Value::Int64(2));

        let map = Value::Map(entries.into());

        let bytes = map.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(map, decoded);
    }

    #[test]
    fn test_list_with_nested_map() {
        let mut inner_map = BTreeMap::new();
        inner_map.insert(PropertyKey::new("key"), Value::String("val".into()));

        let list = Value::List(vec![Value::Map(inner_map.into()), Value::Int64(42)].into());

        let bytes = list.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(list, decoded);
    }

    #[test]
    fn test_property_key_borrow_str() {
        use std::collections::HashMap;
        let mut map: HashMap<PropertyKey, i32> = HashMap::new();
        map.insert(PropertyKey::new("name"), 42);
        map.insert(PropertyKey::new("age"), 30);
        assert_eq!(map.get("name"), Some(&42));
        assert_eq!(map.get("age"), Some(&30));
        assert_eq!(map.get("missing"), None);
        assert!(map.contains_key("name"));
        assert!(!map.contains_key("nope"));
    }

    #[test]
    fn test_gcounter_type_name() {
        let v = Value::GCounter(Arc::new(std::collections::HashMap::new()));
        assert_eq!(v.type_name(), "GCOUNTER");
    }

    #[test]
    fn test_oncounter_type_name() {
        let v = Value::OnCounter {
            pos: Arc::new(std::collections::HashMap::new()),
            neg: Arc::new(std::collections::HashMap::new()),
        };
        assert_eq!(v.type_name(), "PNCOUNTER");
    }

    #[test]
    fn test_gcounter_display() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("node-a".to_string(), 10u64);
        counts.insert("node-b".to_string(), 5u64);
        let v = Value::GCounter(Arc::new(counts));
        assert_eq!(format!("{v}"), "GCounter(15)");
    }

    #[test]
    fn test_gcounter_debug() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("node-a".to_string(), 3u64);
        let v = Value::GCounter(Arc::new(counts));
        let debug = format!("{v:?}");
        assert!(debug.contains("GCounter"));
        assert!(debug.contains("total=3"));
    }

    #[test]
    fn test_oncounter_display() {
        let mut pos = std::collections::HashMap::new();
        pos.insert("node-a".to_string(), 10u64);
        pos.insert("node-b".to_string(), 3u64);
        let mut neg = std::collections::HashMap::new();
        neg.insert("node-a".to_string(), 4u64);
        let v = Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        };
        // pos_sum = 13, neg_sum = 4, net = 9
        assert_eq!(format!("{v}"), "OnCounter(9)");
    }

    #[test]
    fn test_oncounter_debug() {
        let v = Value::OnCounter {
            pos: Arc::new(std::collections::HashMap::new()),
            neg: Arc::new(std::collections::HashMap::new()),
        };
        let debug = format!("{v:?}");
        assert!(debug.contains("OnCounter"));
        assert!(debug.contains("net=0"));
    }

    #[test]
    fn test_gcounter_hash_is_insertion_order_independent() {
        use std::hash::{Hash, Hasher};
        let mut counts1 = std::collections::HashMap::new();
        counts1.insert("b".to_string(), 2u64);
        counts1.insert("a".to_string(), 1u64);
        let mut counts2 = std::collections::HashMap::new();
        counts2.insert("a".to_string(), 1u64);
        counts2.insert("b".to_string(), 2u64);
        let v1 = HashableValue(Value::GCounter(Arc::new(counts1)));
        let v2 = HashableValue(Value::GCounter(Arc::new(counts2)));
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        v1.hash(&mut h1);
        v2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_oncounter_hash_is_insertion_order_independent() {
        use std::hash::{Hash, Hasher};
        let mut pos1 = std::collections::HashMap::new();
        pos1.insert("b".to_string(), 5u64);
        pos1.insert("a".to_string(), 3u64);
        let mut pos2 = std::collections::HashMap::new();
        pos2.insert("a".to_string(), 3u64);
        pos2.insert("b".to_string(), 5u64);
        let neg = Arc::new(std::collections::HashMap::new());
        let v1 = HashableValue(Value::OnCounter {
            pos: Arc::new(pos1),
            neg: neg.clone(),
        });
        let v2 = HashableValue(Value::OnCounter {
            pos: Arc::new(pos2),
            neg: neg.clone(),
        });
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        v1.hash(&mut h1);
        v2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_gcounter_serialize_roundtrip() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("replica-1".to_string(), 42u64);
        counts.insert("replica-2".to_string(), 7u64);
        let v = Value::GCounter(Arc::new(counts));
        let bytes = v.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn test_oncounter_serialize_roundtrip() {
        let mut pos = std::collections::HashMap::new();
        pos.insert("node-a".to_string(), 10u64);
        let mut neg = std::collections::HashMap::new();
        neg.insert("node-a".to_string(), 3u64);
        let v = Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        };
        let bytes = v.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn test_gcounter_empty_display() {
        let v = Value::GCounter(Arc::new(std::collections::HashMap::new()));
        assert_eq!(format!("{v}"), "GCounter(0)");
    }

    #[test]
    fn test_oncounter_zero_net() {
        let mut pos = std::collections::HashMap::new();
        pos.insert("r".to_string(), 5u64);
        let mut neg = std::collections::HashMap::new();
        neg.insert("r".to_string(), 5u64);
        let v = Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        };
        assert_eq!(format!("{v}"), "OnCounter(0)");
    }

    #[test]
    fn test_estimated_size_bytes_fixed_types() {
        assert_eq!(Value::Null.estimated_size_bytes(), 0);
        assert_eq!(Value::Bool(true).estimated_size_bytes(), 0);
        assert_eq!(Value::Int64(42).estimated_size_bytes(), 0);
        assert_eq!(Value::Float64(3.125).estimated_size_bytes(), 0);
    }

    #[test]
    fn test_estimated_size_bytes_string() {
        let v = Value::from("hello");
        assert_eq!(v.estimated_size_bytes(), 5);
    }

    #[test]
    fn test_estimated_size_bytes_bytes() {
        let v = Value::Bytes(Arc::from(vec![0u8; 100].as_slice()));
        assert_eq!(v.estimated_size_bytes(), 100);
    }

    #[test]
    fn test_estimated_size_bytes_vector() {
        let v = Value::Vector(Arc::from(vec![1.0f32; 384].as_slice()));
        assert_eq!(v.estimated_size_bytes(), 384 * 4);
    }

    #[test]
    fn test_estimated_size_bytes_list() {
        let v = Value::List(Arc::from(vec![Value::from("abc"), Value::Int64(1)]));
        assert!(v.estimated_size_bytes() >= 3);
    }

    // --- Accessor tests for temporal and counter types ---

    #[test]
    fn test_as_date_matching() {
        let date = Date::from_ymd(2024, 6, 15).unwrap();
        let v = Value::Date(date);
        assert_eq!(v.as_date(), Some(date));
    }

    #[test]
    fn test_as_date_non_matching() {
        let v = Value::Int64(42);
        assert_eq!(v.as_date(), None);
    }

    #[test]
    fn test_as_time_matching() {
        let time = Time::from_hms(14, 30, 0).unwrap();
        let v = Value::Time(time);
        assert_eq!(v.as_time(), Some(time));
    }

    #[test]
    fn test_as_time_non_matching() {
        let v = Value::String("x".into());
        assert_eq!(v.as_time(), None);
    }

    #[test]
    fn test_as_duration_matching() {
        let dur = Duration::new(1, 2, 3_000_000_000);
        let v = Value::Duration(dur);
        assert_eq!(v.as_duration(), Some(dur));
    }

    #[test]
    fn test_as_duration_non_matching() {
        let v = Value::Bool(true);
        assert_eq!(v.as_duration(), None);
    }

    #[test]
    fn test_as_zoned_datetime_matching() {
        let zdt = ZonedDatetime::parse("2024-06-15T10:30:00+01:00").unwrap();
        let v = Value::ZonedDatetime(zdt);
        assert_eq!(v.as_zoned_datetime(), Some(zdt));
    }

    #[test]
    fn test_as_zoned_datetime_non_matching() {
        let v = Value::Null;
        assert_eq!(v.as_zoned_datetime(), None);
    }

    #[test]
    fn test_as_zoned_datetime_from_conversion() {
        let zdt = ZonedDatetime::parse("2025-01-01T00:00:00+02:00").unwrap();
        let v: Value = zdt.into();
        assert_eq!(v.as_zoned_datetime(), Some(zdt));
        assert_eq!(v.type_name(), "ZONED DATETIME");
    }

    #[test]
    fn test_as_date_wrong_temporal_type() {
        // A Time value should not be returned by as_date
        let time = Time::from_hms(12, 0, 0).unwrap();
        let v = Value::Time(time);
        assert_eq!(v.as_date(), None);
    }

    #[test]
    fn test_as_time_wrong_temporal_type() {
        // A Date value should not be returned by as_time
        let date = Date::from_ymd(2024, 3, 15).unwrap();
        let v = Value::Date(date);
        assert_eq!(v.as_time(), None);
    }

    #[test]
    fn test_as_duration_wrong_temporal_type() {
        // A Timestamp should not be returned by as_duration
        let ts = Timestamp::from_secs(1_000_000);
        let v = Value::Timestamp(ts);
        assert_eq!(v.as_duration(), None);
    }

    #[test]
    fn test_as_zoned_datetime_wrong_temporal_type() {
        // A plain Timestamp should not be returned by as_zoned_datetime
        let ts = Timestamp::from_secs(1_000_000);
        let v = Value::Timestamp(ts);
        assert_eq!(v.as_zoned_datetime(), None);
    }

    #[test]
    fn test_zoned_datetime_serialization_roundtrip() {
        let zdt = ZonedDatetime::parse("2024-12-25T18:00:00+05:30").unwrap();
        let v = Value::ZonedDatetime(zdt);
        let bytes = v.serialize().unwrap();
        let decoded = Value::deserialize(&bytes).unwrap();
        assert_eq!(v, decoded);
        assert_eq!(decoded.as_zoned_datetime(), Some(zdt));
    }

    #[test]
    fn test_zoned_datetime_type_name() {
        let zdt = ZonedDatetime::parse("2024-06-15T10:30:00+02:00").unwrap();
        let v = Value::ZonedDatetime(zdt);
        assert_eq!(v.type_name(), "ZONED DATETIME");
    }

    #[test]
    fn test_zoned_datetime_display() {
        let zdt = ZonedDatetime::parse("2024-06-15T10:30:00+02:00").unwrap();
        let v = Value::ZonedDatetime(zdt);
        let displayed = format!("{v}");
        assert!(
            displayed.contains("2024-06-15"),
            "Display should contain the date"
        );
        assert!(
            displayed.contains("10:30:00"),
            "Display should contain the time"
        );
    }
}
