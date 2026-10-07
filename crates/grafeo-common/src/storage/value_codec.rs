//! The lossless value codec of the `.grafeo` format.
//!
//! [`encode_value`] writes one [`Value`] as a tag byte followed by its
//! fields, little-endian, and [`decode_value`] reads it back. Every kind round
//! trips exactly: floats by their bits (NaN payloads and -0.0 included), times
//! and zoned datetimes with their offsets, counters with every replica. The
//! tags are the format and never change:
//!
//! | Tag | Kind | Fields |
//! | --- | --- | --- |
//! | 0 | Null | none |
//! | 1 | Bool | u8, 0 or 1 |
//! | 2 | Int64 | i64 |
//! | 3 | Float64 | u64 bits |
//! | 4 | String | length u32, UTF-8 |
//! | 5 | Bytes | length u32, bytes |
//! | 6 | Date | i32 days since 1970-01-01 |
//! | 7 | Time | u64 nanoseconds since midnight, u8 has offset (0 or 1), i32 offset seconds (0 without one) |
//! | 8 | Timestamp | i64 microseconds since the epoch |
//! | 9 | ZonedDatetime | i64 UTC microseconds, i32 offset seconds |
//! | 10 | Duration | i64 months, i64 days, i64 nanoseconds |
//! | 11 | List | count u32, values |
//! | 12 | Map | count u32, (key as length u32 and UTF-8, value) per entry |
//! | 13 | Vector | dimensions u32, u32 bits per component |
//! | 14 | Path | node count u32, values, edge count u32, values |
//! | 15 | GCounter | count u32, (replica as length u32 and UTF-8, u64) per entry, sorted by replica |
//! | 16 | OnCounter | the positive entries, then the negative entries, each as in 15 |
//!
//! The decoder checks every length and count against the bytes left before
//! it allocates, and refuses an unknown tag, a field the type does not accept,
//! map keys or counter replicas that are not strictly increasing, and nesting
//! deeper than [`MAX_VALUE_DEPTH`], so each value has exactly one encoding.
//! The encoder refuses what the decoder would refuse.
//!
//! This is not the string-table codec of the LPG block format in grafeo-core,
//! which has functions of the same names.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use arcstr::ArcStr;

use crate::types::{Date, Duration, PropertyKey, Time, Timestamp, Value, ZonedDatetime};
use crate::utils::error::{Error, Result};

/// Deepest nesting of lists, maps and paths a decode accepts.
///
/// A list of scalars is 1 deep, a list holding that list 2 deep.
/// [`encode_value`] refuses a value nested deeper. Two levels above
/// [`MAX_PROPERTY_VALUE_DEPTH`], so the history of a property (a list of
/// `[epoch, value]` lists) of any value a write accepts encodes.
pub const MAX_VALUE_DEPTH: usize = MAX_PROPERTY_VALUE_DEPTH + 2;

/// Deepest nesting of lists, maps and paths a property value may have when
/// it is written: 128, the nesting the GQL parser accepts. See
/// [`nests_too_deep`].
pub const MAX_PROPERTY_VALUE_DEPTH: usize = 128;

const TAG_NULL: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_INT64: u8 = 2;
const TAG_FLOAT64: u8 = 3;
const TAG_STRING: u8 = 4;
const TAG_BYTES: u8 = 5;
const TAG_DATE: u8 = 6;
const TAG_TIME: u8 = 7;
const TAG_TIMESTAMP: u8 = 8;
const TAG_ZONED_DATETIME: u8 = 9;
const TAG_DURATION: u8 = 10;
const TAG_LIST: u8 = 11;
const TAG_MAP: u8 = 12;
const TAG_VECTOR: u8 = 13;
const TAG_PATH: u8 = 14;
const TAG_GCOUNTER: u8 = 15;
const TAG_ON_COUNTER: u8 = 16;

/// The fewest bytes a map entry takes: a key length and a value tag.
const MIN_MAP_ENTRY_BYTES: usize = 5;
/// The fewest bytes a counter entry takes: a replica length and a count.
const MIN_COUNTER_ENTRY_BYTES: usize = 12;

/// Appends the encoding of `value`. Every `Value` kind round trips exactly
/// (floats by their bits).
///
/// # Errors
///
/// Returns [`Error::Serialization`] when `value` nests lists, maps and paths
/// deeper than [`MAX_VALUE_DEPTH`], or holds a string, byte string, list,
/// map, vector, path or counter whose length does not fit a u32. `out` is
/// then left as it was.
pub fn encode_value(value: &Value, out: &mut Vec<u8>) -> Result<()> {
    let start = out.len();
    let result = encode_nested(value, out, 0);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// Encodes `value`, which sits inside `depth` lists, maps and paths.
fn encode_nested(value: &Value, out: &mut Vec<u8>, depth: usize) -> Result<()> {
    match value {
        Value::Null => out.push(TAG_NULL),
        Value::Bool(flag) => {
            out.push(TAG_BOOL);
            out.push(u8::from(*flag));
        }
        Value::Int64(integer) => {
            out.push(TAG_INT64);
            out.extend_from_slice(&integer.to_le_bytes());
        }
        Value::Float64(float) => {
            out.push(TAG_FLOAT64);
            out.extend_from_slice(&float.to_bits().to_le_bytes());
        }
        Value::String(string) => {
            out.push(TAG_STRING);
            put_str(string, "string", out)?;
        }
        Value::Bytes(bytes) => {
            out.push(TAG_BYTES);
            put_length(bytes.len(), "byte string", out)?;
            out.extend_from_slice(bytes);
        }
        Value::Date(date) => {
            out.push(TAG_DATE);
            out.extend_from_slice(&date.as_days().to_le_bytes());
        }
        Value::Time(time) => {
            out.push(TAG_TIME);
            out.extend_from_slice(&time.as_nanos().to_le_bytes());
            let (has_offset, offset) = match time.offset_seconds() {
                Some(offset) => (1, offset),
                None => (0, 0),
            };
            out.push(has_offset);
            out.extend_from_slice(&offset.to_le_bytes());
        }
        Value::Timestamp(timestamp) => {
            out.push(TAG_TIMESTAMP);
            out.extend_from_slice(&timestamp.as_micros().to_le_bytes());
        }
        Value::ZonedDatetime(zoned) => {
            out.push(TAG_ZONED_DATETIME);
            out.extend_from_slice(&zoned.as_timestamp().as_micros().to_le_bytes());
            out.extend_from_slice(&zoned.offset_seconds().to_le_bytes());
        }
        Value::Duration(duration) => {
            out.push(TAG_DURATION);
            out.extend_from_slice(&duration.months().to_le_bytes());
            out.extend_from_slice(&duration.days().to_le_bytes());
            out.extend_from_slice(&duration.nanos().to_le_bytes());
        }
        Value::List(items) => {
            let depth = one_deeper(depth)?;
            out.push(TAG_LIST);
            put_values(items, "list", out, depth)?;
        }
        Value::Map(map) => {
            let depth = one_deeper(depth)?;
            out.push(TAG_MAP);
            put_length(map.len(), "map", out)?;
            for (key, item) in map.iter() {
                put_str(key.as_str(), "map key", out)?;
                encode_nested(item, out, depth)?;
            }
        }
        Value::Vector(components) => {
            out.push(TAG_VECTOR);
            put_length(components.len(), "vector", out)?;
            for component in components.iter() {
                out.extend_from_slice(&component.to_bits().to_le_bytes());
            }
        }
        Value::Path { nodes, edges } => {
            let depth = one_deeper(depth)?;
            out.push(TAG_PATH);
            put_values(nodes, "path node list", out, depth)?;
            put_values(edges, "path edge list", out, depth)?;
        }
        Value::GCounter(counts) => {
            out.push(TAG_GCOUNTER);
            put_counter(counts, out)?;
        }
        Value::OnCounter { pos, neg } => {
            out.push(TAG_ON_COUNTER);
            put_counter(pos, out)?;
            put_counter(neg, out)?;
        }
    }
    Ok(())
}

/// The depth inside one more list, map or path, refused past
/// [`MAX_VALUE_DEPTH`].
fn one_deeper(depth: usize) -> Result<usize> {
    let depth = depth + 1;
    if depth > MAX_VALUE_DEPTH {
        return Err(Error::Serialization(format!(
            "value codec: cannot encode a value nested deeper than {MAX_VALUE_DEPTH} lists, \
             maps and paths"
        )));
    }
    Ok(depth)
}

fn put_length(length: usize, what: &str, out: &mut Vec<u8>) -> Result<()> {
    let length = u32::try_from(length).map_err(|_| {
        Error::Serialization(format!(
            "value codec: cannot encode a {what} of length {length}: it does not fit a u32"
        ))
    })?;
    out.extend_from_slice(&length.to_le_bytes());
    Ok(())
}

fn put_str(string: &str, what: &str, out: &mut Vec<u8>) -> Result<()> {
    put_length(string.len(), what, out)?;
    out.extend_from_slice(string.as_bytes());
    Ok(())
}

fn put_values(values: &[Value], what: &str, out: &mut Vec<u8>, depth: usize) -> Result<()> {
    put_length(values.len(), what, out)?;
    for value in values {
        encode_nested(value, out, depth)?;
    }
    Ok(())
}

/// Writes the entries sorted by replica, so equal counters give equal bytes
/// whatever order their map iterates in.
fn put_counter(counts: &HashMap<String, u64>, out: &mut Vec<u8>) -> Result<()> {
    let mut entries: Vec<(&String, &u64)> = counts.iter().collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    put_length(entries.len(), "counter", out)?;
    for (replica, count) in entries {
        put_str(replica, "counter replica", out)?;
        out.extend_from_slice(&count.to_le_bytes());
    }
    Ok(())
}

/// Whether `value` nests lists, maps and paths deeper than
/// [`MAX_PROPERTY_VALUE_DEPTH`], the deepest property value a write accepts.
///
/// Writes refuse such a value before it reaches a store, so a checkpoint
/// never meets one. Looks at most `MAX_PROPERTY_VALUE_DEPTH + 1` levels down,
/// so a value nested far deeper costs no deeper recursion.
#[must_use]
pub fn nests_too_deep(value: &Value) -> bool {
    deeper_than(value, MAX_PROPERTY_VALUE_DEPTH)
}

/// Whether `value` nests lists, maps and paths more than `room` levels deep.
fn deeper_than(value: &Value, room: usize) -> bool {
    let inner = |item: &Value| deeper_than(item, room - 1);
    match value {
        Value::List(items) => room == 0 || items.iter().any(inner),
        Value::Map(map) => room == 0 || map.values().any(inner),
        Value::Path { nodes, edges } => room == 0 || nodes.iter().chain(edges.iter()).any(inner),
        _ => false,
    }
}

/// The number of bytes `encode_value` appends for `value`.
///
/// For a value [`encode_value`] refuses, the size its encoding would have.
#[must_use]
pub fn encoded_len(value: &Value) -> usize {
    1 + match value {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Int64(_) | Value::Float64(_) | Value::Timestamp(_) => 8,
        Value::String(string) => 4 + string.len(),
        Value::Bytes(bytes) => 4 + bytes.len(),
        Value::Date(_) => 4,
        Value::Time(_) => 8 + 1 + 4,
        Value::ZonedDatetime(_) => 8 + 4,
        Value::Duration(_) => 3 * 8,
        Value::List(items) => values_len(items),
        Value::Map(map) => {
            4 + map
                .iter()
                .map(|(key, item)| 4 + key.as_str().len() + encoded_len(item))
                .sum::<usize>()
        }
        Value::Vector(components) => 4 + 4 * components.len(),
        Value::Path { nodes, edges } => values_len(nodes) + values_len(edges),
        Value::GCounter(counts) => counter_len(counts),
        Value::OnCounter { pos, neg } => counter_len(pos) + counter_len(neg),
    }
}

fn values_len(values: &[Value]) -> usize {
    4 + values.iter().map(encoded_len).sum::<usize>()
}

fn counter_len(counts: &HashMap<String, u64>) -> usize {
    4 + counts
        .keys()
        .map(|replica| 4 + replica.len() + 8)
        .sum::<usize>()
}

/// Decodes one value at `*pos` and advances it.
///
/// Lengths and counts are checked against the bytes left before anything is
/// allocated. What the format does not allow is an error, never a default
/// value; dates and offsets are taken as their types accept them (any i32).
/// Lists, maps, paths and counters grow as their entries decode, so memory
/// stays proportional to the input whatever counts a damaged input claims.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the byte offset in `data` of what
/// is wrong, and leaves `*pos` as it was:
///
/// - an unknown tag;
/// - a time of day of a full day or more;
/// - a bool or time offset flag other than 0 or 1;
/// - a time without an offset whose offset field is not 0;
/// - a string, map key or counter replica that is not UTF-8;
/// - a count or length past the bytes left, or input that ends inside the
///   value;
/// - nesting deeper than [`MAX_VALUE_DEPTH`];
/// - map keys or counter replicas that are not strictly increasing (a repeat
///   is one).
pub fn decode_value(data: &[u8], pos: &mut usize) -> Result<Value> {
    let mut decoder = Decoder {
        data,
        pos: *pos,
        depth: 0,
    };
    let value = decoder.value()?;
    *pos = decoder.pos;
    Ok(value)
}

/// Reads values from `data`, starting at `pos`, inside `depth` lists, maps
/// and paths.
struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Decoder<'a> {
    fn error(at: usize, message: impl fmt::Display) -> Error {
        Error::Serialization(format!("value codec, byte {at}: {message}"))
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// The next `n` bytes, after checking that `n` bytes are left; an error
    /// names `what` and the offset `at` where it starts.
    fn take(&mut self, n: usize, what: &str, at: usize) -> Result<&'a [u8]> {
        let remaining = self.remaining();
        if n > remaining {
            return Err(Self::error(
                at,
                format_args!("{what} needs {n} bytes, only {remaining} left"),
            ));
        }
        let bytes = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        let at = self.pos;
        let mut array = [0; N];
        array.copy_from_slice(self.take(N, what, at)?);
        Ok(array)
    }

    fn read_u8(&mut self, what: &str) -> Result<u8> {
        let [byte] = self.array(what)?;
        Ok(byte)
    }

    fn read_u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array(what)?))
    }

    fn read_i32(&mut self, what: &str) -> Result<i32> {
        Ok(i32::from_le_bytes(self.array(what)?))
    }

    fn read_u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array(what)?))
    }

    fn read_i64(&mut self, what: &str) -> Result<i64> {
        Ok(i64::from_le_bytes(self.array(what)?))
    }

    /// A u8 that must be 0 or 1.
    fn read_flag(&mut self, what: &str) -> Result<bool> {
        let at = self.pos;
        match self.read_u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(Self::error(
                at,
                format_args!("{what} is {other}, expected 0 or 1"),
            )),
        }
    }

    /// A u32 length or count.
    fn read_length(&mut self, what: &str) -> Result<usize> {
        let at = self.pos;
        let length = self.read_u32(what)?;
        usize::try_from(length).map_err(|_| {
            Self::error(
                at,
                format_args!("{what} of length {length} does not fit this platform"),
            )
        })
    }

    /// A count of entries that take at least `min_entry_bytes` each, refused
    /// when the bytes left cannot hold that many.
    fn read_count(&mut self, what: &str, min_entry_bytes: usize) -> Result<usize> {
        let at = self.pos;
        let count = self.read_length(what)?;
        let remaining = self.remaining();
        if count
            .checked_mul(min_entry_bytes)
            .is_none_or(|needed| needed > remaining)
        {
            return Err(Self::error(
                at,
                format_args!(
                    "{what} of {count} entries needs at least {min_entry_bytes} bytes each, \
                     only {remaining} left"
                ),
            ));
        }
        Ok(count)
    }

    /// A length-prefixed UTF-8 string.
    fn read_str(&mut self, what: &str) -> Result<&'a str> {
        let at = self.pos;
        let length = self.read_length(what)?;
        let bytes = self.take(length, what, at)?;
        std::str::from_utf8(bytes)
            .map_err(|error| Self::error(at, format_args!("{what} is not UTF-8 ({error})")))
    }

    /// Enters a list, map or path whose tag is at `at`.
    fn enter(&mut self, at: usize) -> Result<()> {
        if self.depth >= MAX_VALUE_DEPTH {
            return Err(Self::error(
                at,
                format_args!("a value nested deeper than {MAX_VALUE_DEPTH} lists, maps and paths"),
            ));
        }
        self.depth += 1;
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    fn value(&mut self) -> Result<Value> {
        let at = self.pos;
        let value = match self.read_u8("value tag")? {
            TAG_NULL => Value::Null,
            TAG_BOOL => Value::Bool(self.read_flag("bool")?),
            TAG_INT64 => Value::Int64(self.read_i64("integer")?),
            TAG_FLOAT64 => Value::Float64(f64::from_bits(self.read_u64("float")?)),
            TAG_STRING => Value::String(ArcStr::from(self.read_str("string")?)),
            TAG_BYTES => {
                let bytes_at = self.pos;
                let length = self.read_length("byte string")?;
                Value::Bytes(Arc::from(self.take(length, "byte string", bytes_at)?))
            }
            TAG_DATE => Value::Date(Date::from_days(self.read_i32("date")?)),
            TAG_TIME => Value::Time(self.time()?),
            TAG_TIMESTAMP => Value::Timestamp(Timestamp::from_micros(self.read_i64("timestamp")?)),
            TAG_ZONED_DATETIME => {
                let micros = self.read_i64("zoned datetime")?;
                let offset = self.read_i32("zoned datetime offset")?;
                Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                    Timestamp::from_micros(micros),
                    offset,
                ))
            }
            TAG_DURATION => {
                let months = self.read_i64("duration months")?;
                let days = self.read_i64("duration days")?;
                let nanos = self.read_i64("duration nanoseconds")?;
                Value::Duration(Duration::new(months, days, nanos))
            }
            TAG_LIST => {
                self.enter(at)?;
                let items = self.values("list")?;
                self.leave();
                Value::List(items)
            }
            TAG_MAP => {
                self.enter(at)?;
                let map = self.map()?;
                self.leave();
                Value::Map(Arc::new(map))
            }
            TAG_VECTOR => {
                let dimensions = self.read_count("vector", 4)?;
                let mut components = Vec::with_capacity(dimensions);
                for _ in 0..dimensions {
                    components.push(f32::from_bits(self.read_u32("vector component")?));
                }
                Value::Vector(components.into())
            }
            TAG_PATH => {
                self.enter(at)?;
                let nodes = self.values("path node list")?;
                let edges = self.values("path edge list")?;
                self.leave();
                Value::Path { nodes, edges }
            }
            TAG_GCOUNTER => Value::GCounter(Arc::new(self.counter("counter")?)),
            TAG_ON_COUNTER => {
                let pos = self.counter("positive counter")?;
                let neg = self.counter("negative counter")?;
                Value::OnCounter {
                    pos: Arc::new(pos),
                    neg: Arc::new(neg),
                }
            }
            other => return Err(Self::error(at, format_args!("unknown value tag {other}"))),
        };
        Ok(value)
    }

    /// A time: nanoseconds within a day, an offset flag and an offset that is
    /// 0 when the flag is.
    fn time(&mut self) -> Result<Time> {
        let at = self.pos;
        let nanos = self.read_u64("time")?;
        let time = Time::from_nanos(nanos).ok_or_else(|| {
            Self::error(
                at,
                format_args!("a time of {nanos} nanoseconds is not within a day"),
            )
        })?;
        let flag_at = self.pos;
        let has_offset = self.read_flag("time offset flag")?;
        let offset = self.read_i32("time offset")?;
        if has_offset {
            Ok(time.with_offset(offset))
        } else if offset == 0 {
            Ok(time)
        } else {
            Err(Self::error(
                flag_at,
                format_args!("a time without an offset holds offset {offset}"),
            ))
        }
    }

    fn values(&mut self, what: &str) -> Result<Arc<[Value]>> {
        let count = self.read_count(what, 1)?;
        let mut values = Vec::new();
        for _ in 0..count {
            values.push(self.value()?);
        }
        Ok(values.into())
    }

    /// Refuses `key`, read at `at`, unless it comes strictly after `previous`
    /// in `str` order, which is the order of `PropertyKey` (a `BTreeMap`
    /// writes its keys in it) and of `String` (counters are written sorted).
    /// A repeated key is one that does not.
    fn check_increasing(what: &str, previous: Option<&str>, key: &str, at: usize) -> Result<()> {
        match previous {
            Some(previous) if key <= previous => Err(Self::error(
                at,
                format_args!(
                    "{what} {key:?} does not come after {previous:?}: they are stored \
                     strictly increasing"
                ),
            )),
            _ => Ok(()),
        }
    }

    fn map(&mut self) -> Result<BTreeMap<PropertyKey, Value>> {
        let count = self.read_count("map", MIN_MAP_ENTRY_BYTES)?;
        let mut map = BTreeMap::new();
        let mut previous = None;
        for _ in 0..count {
            let at = self.pos;
            let key = self.read_str("map key")?;
            Self::check_increasing("map key", previous, key, at)?;
            previous = Some(key);
            let value = self.value()?;
            map.insert(PropertyKey::new(key), value);
        }
        Ok(map)
    }

    fn counter(&mut self, what: &str) -> Result<HashMap<String, u64>> {
        let count = self.read_count(what, MIN_COUNTER_ENTRY_BYTES)?;
        let mut counts = HashMap::new();
        let mut previous = None;
        for _ in 0..count {
            let at = self.pos;
            let replica = self.read_str("counter replica")?;
            Self::check_increasing("counter replica", previous, replica, at)?;
            previous = Some(replica);
            counts.insert(replica.to_owned(), self.read_u64("counter value")?);
        }
        Ok(counts)
    }
}

/// serde adapter: an `Option<Value>` stored as `Option<bytes>` of
/// `encode_value`.
///
/// Use it as `#[serde(with = "grafeo_common::storage::value_codec::serde_option_value")]`
/// on an `Option<Value>` field, so the value keeps its exact kind and bits
/// in formats that would otherwise lose them.
///
/// It needs a format with native byte strings, such as bincode, which the
/// catalog uses: the value is written with `serialize_bytes` and read back
/// through `deserialize_byte_buf`, so a format that writes bytes as a
/// sequence of numbers (JSON) refuses it on read.
pub mod serde_option_value {
    use std::fmt;

    use serde::de::{self, Deserializer, Visitor};
    use serde::ser::{Error as _, Serialize, Serializer};

    use super::{decode_value, encode_value};
    use crate::types::Value;

    /// Writes `None` as none and `Some(value)` as the bytes of
    /// [`encode_value`].
    ///
    /// # Errors
    ///
    /// Returns the serializer's error, or a custom one when [`encode_value`]
    /// refuses the value.
    pub fn serialize<S: Serializer>(
        value: &Option<Value>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        match value {
            None => serializer.serialize_none(),
            Some(value) => {
                let mut bytes = Vec::new();
                encode_value(value, &mut bytes).map_err(S::Error::custom)?;
                serializer.serialize_some(&EncodedValue(&bytes))
            }
        }
    }

    /// Reads what [`serialize`] wrote.
    ///
    /// # Errors
    ///
    /// Returns the deserializer's error, or a custom one when
    /// [`decode_value`] refuses the bytes or they hold
    /// more than one value.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Option<Value>, D::Error> {
        deserializer.deserialize_option(OptionVisitor)
    }

    /// The bytes of one encoded value, serialized as bytes, not as a sequence.
    struct EncodedValue<'a>(&'a [u8]);

    impl Serialize for EncodedValue<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
            serializer.serialize_bytes(self.0)
        }
    }

    struct OptionVisitor;

    impl<'de> Visitor<'de> for OptionVisitor {
        type Value = Option<Value>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an optional encoded value")
        }

        fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> std::result::Result<Self::Value, D::Error> {
            deserializer
                .deserialize_byte_buf(EncodedValueVisitor)
                .map(Some)
        }
    }

    struct EncodedValueVisitor;

    impl Visitor<'_> for EncodedValueVisitor {
        type Value = Value;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("the bytes of one encoded value")
        }

        fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> std::result::Result<Value, E> {
            let mut pos = 0;
            let value = decode_value(bytes, &mut pos).map_err(E::custom)?;
            if pos != bytes.len() {
                return Err(E::custom(format_args!(
                    "value codec: {} trailing bytes after a value of {pos} bytes",
                    bytes.len() - pos
                )));
            }
            Ok(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use super::*;
    use crate::types::{Date, Duration, PropertyKey, Time, Timestamp, Value, ZonedDatetime};

    fn list(items: Vec<Value>) -> Value {
        Value::List(Arc::from(items))
    }

    fn map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(Arc::new(
            entries
                .into_iter()
                .map(|(key, value)| (PropertyKey::new(key), value))
                .collect::<BTreeMap<_, _>>(),
        ))
    }

    fn counter(entries: &[(&str, u64)]) -> Arc<HashMap<String, u64>> {
        Arc::new(
            entries
                .iter()
                .map(|(replica, count)| ((*replica).to_string(), *count))
                .collect(),
        )
    }

    /// One value of every kind, with the edge cases the 0.5.x formats lose.
    fn every_kind() -> Vec<Value> {
        vec![
            Value::Null,
            Value::Bool(true),
            Value::Int64(-19),
            Value::Int64(i64::MAX),
            Value::Float64(f64::from_bits(0x7FF8_0000_0000_0058)),
            Value::Float64(-0.0),
            Value::from("Amsterdam"),
            Value::from(""),
            Value::Bytes(Arc::from(vec![3u8, 19, 88])),
            Value::Date(Date::from_days(-3)),
            Value::Time(Time::from_nanos(88).unwrap()),
            Value::Time(
                Time::from_nanos(3_600_000_000_019)
                    .unwrap()
                    .with_offset(3600),
            ),
            Value::Timestamp(Timestamp::from_micros(1_696_500_000_123_457)),
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1_696_500_000_123_457),
                7200,
            )),
            Value::Duration(Duration::new(3, 19, 88)),
            list(vec![
                Value::Int64(3),
                list(vec![Value::from("Berlin")]),
                Value::Null,
            ]),
            map(vec![
                ("city", Value::from("Prague")),
                ("stops", list(vec![Value::Int64(19)])),
            ]),
            Value::Vector(Arc::from(vec![3.0f32, -19.5, f32::NAN])),
            Value::Path {
                nodes: Arc::from(vec![map(vec![("_id", Value::Int64(3))])]),
                edges: Arc::from(Vec::<Value>::new()),
            },
            Value::GCounter(counter(&[("Alix", 3), ("Gus", 19)])),
            Value::OnCounter {
                pos: counter(&[("Mia", 88)]),
                neg: counter(&[("Jules", 3)]),
            },
            Value::Bool(false),
            Value::Time(Time::from_nanos(88).unwrap().with_offset(0)),
            Value::Path {
                nodes: Arc::from(vec![
                    map(vec![("_id", Value::Int64(3))]),
                    map(vec![("_id", Value::Int64(19))]),
                ]),
                edges: Arc::from(vec![map(vec![("_id", Value::Int64(88))])]),
            },
            list(vec![Value::Null]),
            map(vec![("", Value::Null)]),
            Value::GCounter(counter(&[("", 0)])),
        ]
    }

    fn all_same(a: &[Value], b: &[Value]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(x, y))
    }

    /// Equality that compares floats by their bits, recursively.
    ///
    /// `Value`'s own equality says NaN is not NaN, -0.0 is 0.0, and a time or
    /// zoned datetime equals another at the same instant whatever its offset.
    fn same(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Float64(x), Value::Float64(y)) => x.to_bits() == y.to_bits(),
            (Value::Vector(x), Value::Vector(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|(p, q)| p.to_bits() == q.to_bits())
            }
            (Value::Time(x), Value::Time(y)) => {
                x.as_nanos() == y.as_nanos() && x.offset_seconds() == y.offset_seconds()
            }
            (Value::ZonedDatetime(x), Value::ZonedDatetime(y)) => {
                x.as_timestamp() == y.as_timestamp() && x.offset_seconds() == y.offset_seconds()
            }
            (Value::List(x), Value::List(y)) => all_same(x, y),
            (Value::Map(x), Value::Map(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(y.iter())
                        .all(|((key_x, value_x), (key_y, value_y))| {
                            key_x == key_y && same(value_x, value_y)
                        })
            }
            (
                Value::Path {
                    nodes: nodes_x,
                    edges: edges_x,
                },
                Value::Path {
                    nodes: nodes_y,
                    edges: edges_y,
                },
            ) => all_same(nodes_x, nodes_y) && all_same(edges_x, edges_y),
            _ => a == b,
        }
    }

    fn encoded(value: &Value) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_value(value, &mut bytes).unwrap();
        bytes
    }

    /// Lists, maps and paths in turn, `depth` containers deep, a null inside.
    fn nested(depth: usize) -> Value {
        let mut value = Value::Null;
        for level in 0..depth {
            value = match level % 3 {
                0 => list(vec![value]),
                1 => map(vec![("next", value)]),
                _ => Value::Path {
                    nodes: Arc::from(vec![value]),
                    edges: Arc::from(Vec::<Value>::new()),
                },
            };
        }
        value
    }

    #[test]
    fn every_value_kind_round_trips_bit_for_bit() {
        for value in every_kind() {
            let mut bytes = Vec::new();
            encode_value(&value, &mut bytes).unwrap();
            assert_eq!(encoded_len(&value), bytes.len(), "{value:?}");
            let mut pos = 0;
            let back = decode_value(&bytes, &mut pos).unwrap();
            assert!(same(&value, &back), "{value:?} came back as {back:?}");
            assert_eq!(pos, bytes.len(), "{value:?}: the whole encoding is read");
        }
    }

    #[test]
    fn values_decode_one_after_another_from_a_shared_buffer() {
        let mut bytes = vec![3, 19, 88];
        for value in every_kind() {
            encode_value(&value, &mut bytes).unwrap();
        }
        let mut pos = 3;
        for value in every_kind() {
            let back = decode_value(&bytes, &mut pos).unwrap();
            assert!(same(&value, &back), "{value:?} came back as {back:?}");
        }
        assert_eq!(pos, bytes.len(), "every value is read, nothing more");
    }

    #[test]
    fn the_tags_are_fixed() {
        assert_eq!(encoded(&Value::Int64(3)), [2, 3, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            encoded(&Value::from("Gus")),
            [4, 3, 0, 0, 0, b'G', b'u', b's']
        );
        assert_eq!(
            encoded(&Value::Timestamp(Timestamp::from_micros(1)))[..2],
            [8, 1]
        );
        assert_eq!(
            encoded(&Value::GCounter(Default::default())),
            [15, 0, 0, 0, 0]
        );
        let tags: Vec<u8> = every_kind().iter().map(|value| encoded(value)[0]).collect();
        assert_eq!(
            tags,
            [
                0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 7, 14, 11,
                12, 15
            ],
            "one tag per kind, in the order of every_kind"
        );
    }

    #[test]
    fn the_field_layouts_are_fixed() {
        let kinds = every_kind();
        let mut time = vec![7];
        time.extend_from_slice(&3_600_000_000_019u64.to_le_bytes());
        time.push(1);
        time.extend_from_slice(&3600i32.to_le_bytes());
        assert_eq!(encoded(&kinds[11]), time, "time: nanos, has_offset, offset");

        let mut zoned = vec![9];
        zoned.extend_from_slice(&1_696_500_000_123_457i64.to_le_bytes());
        zoned.extend_from_slice(&7200i32.to_le_bytes());
        assert_eq!(
            encoded(&kinds[13]),
            zoned,
            "zoned datetime: UTC micros, offset"
        );

        let mut duration = vec![10];
        for part in [3i64, 19, 88] {
            duration.extend_from_slice(&part.to_le_bytes());
        }
        assert_eq!(
            encoded(&kinds[14]),
            duration,
            "duration: months, days, nanos"
        );

        let mut vector = vec![13, 3, 0, 0, 0];
        for component in [3.0f32, -19.5, f32::NAN] {
            vector.extend_from_slice(&component.to_bits().to_le_bytes());
        }
        assert_eq!(encoded(&kinds[17]), vector, "vector: dims, f32 bits");

        let mut path = vec![
            14, 1, 0, 0, 0, 12, 1, 0, 0, 0, 3, 0, 0, 0, b'_', b'i', b'd', 2,
        ];
        path.extend_from_slice(&3i64.to_le_bytes());
        path.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(encoded(&kinds[18]), path, "path: nodes, then edges");

        let mut on_counter = vec![16, 1, 0, 0, 0, 3, 0, 0, 0, b'M', b'i', b'a'];
        on_counter.extend_from_slice(&88u64.to_le_bytes());
        on_counter.extend_from_slice(&[1, 0, 0, 0, 5, 0, 0, 0, b'J', b'u', b'l', b'e', b's']);
        on_counter.extend_from_slice(&3u64.to_le_bytes());
        assert_eq!(
            encoded(&kinds[20]),
            on_counter,
            "on counter: positive, then negative"
        );
    }

    #[test]
    fn counters_encode_the_same_whatever_the_insertion_order() {
        let replicas = [
            "Alix",
            "Gus",
            "Vincent",
            "Mia",
            "Jules",
            "Amsterdam",
            "Berlin",
            "Paris",
            "Prague",
            "Alix3",
            "Gus19",
            "Mia88",
        ];
        let entries: Vec<(&str, u64)> = replicas
            .iter()
            .copied()
            .zip((0u64..).map(|count| count * 3))
            .collect();
        let mut forward = HashMap::new();
        for (replica, count) in &entries {
            forward.insert((*replica).to_string(), *count);
        }
        let mut backward = HashMap::new();
        for (replica, count) in entries.iter().rev() {
            backward.insert((*replica).to_string(), *count);
        }
        let forward = encoded(&Value::GCounter(Arc::new(forward)));
        let backward = encoded(&Value::GCounter(Arc::new(backward)));
        assert_eq!(
            forward, backward,
            "insertion order must not change the bytes"
        );

        let mut sorted = entries;
        sorted.sort_unstable();
        let mut expected = vec![15];
        expected.extend_from_slice(&12u32.to_le_bytes());
        for (replica, count) in sorted {
            expected.extend_from_slice(&u32::try_from(replica.len()).unwrap().to_le_bytes());
            expected.extend_from_slice(replica.as_bytes());
            expected.extend_from_slice(&count.to_le_bytes());
        }
        assert_eq!(forward, expected, "entries are sorted by replica");
    }

    #[test]
    fn truncated_or_unknown_input_is_refused() {
        for value in every_kind() {
            let mut bytes = Vec::new();
            encode_value(&value, &mut bytes).unwrap();
            for cut in 0..bytes.len() {
                assert!(
                    decode_value(&bytes[..cut], &mut 0).is_err(),
                    "{value:?} cut at {cut}"
                );
            }
        }
        let error = decode_value(&[250], &mut 0).unwrap_err().to_string();
        assert!(error.contains("250"), "{error}");
        let huge_list = [11, 0xFF, 0xFF, 0xFF, 0xFF, 0];
        assert!(
            decode_value(&huge_list, &mut 0).is_err(),
            "a count past the bytes left, before allocating"
        );
        let mut deep = [11u8, 1, 0, 0, 0].repeat(MAX_VALUE_DEPTH + 1);
        deep.push(0);
        assert!(
            decode_value(&deep, &mut 0)
                .unwrap_err()
                .to_string()
                .contains("deeper")
        );
    }

    #[test]
    fn huge_counts_of_every_kind_are_refused_before_allocating() {
        let huge = [0xFF, 0xFF, 0xFF, 0xFF];
        for tag in [4u8, 5, 11, 12, 13, 14, 15, 16] {
            let mut bytes = vec![tag];
            bytes.extend_from_slice(&huge);
            bytes.extend_from_slice(&[0; 19]);
            let error = decode_value(&bytes, &mut 0).unwrap_err().to_string();
            assert!(error.contains("byte 1:"), "tag {tag}: {error}");
        }
    }

    #[test]
    fn a_failed_decode_leaves_the_position_alone() {
        let mut bytes = vec![3, 19, 88];
        bytes.extend_from_slice(&encoded(&Value::from("Paris"))[..6]);
        let mut pos = 3;
        assert!(decode_value(&bytes, &mut pos).is_err());
        assert_eq!(pos, 3, "the position of the value that failed");
    }

    #[test]
    fn an_invalid_time_is_refused_not_turned_into_midnight() {
        let mut bytes = vec![7];
        bytes.extend_from_slice(&86_400_000_000_000u64.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0, 0]);
        let error = decode_value(&bytes, &mut 0).unwrap_err().to_string();
        assert!(
            error.contains("byte 1:") && error.contains("not within a day"),
            "{error}"
        );
    }

    #[test]
    fn malformed_fields_are_refused_with_their_offset() {
        let refused = |bytes: &[u8], at: usize| {
            let error = decode_value(bytes, &mut 0).unwrap_err();
            assert!(matches!(error, Error::Serialization(_)), "{error:?}");
            let error = error.to_string();
            assert!(error.contains(&format!("byte {at}:")), "{bytes:?}: {error}");
        };
        refused(&[1, 2], 1);
        let mut time = vec![7];
        time.extend_from_slice(&88u64.to_le_bytes());
        refused(&[&time[..], &[2, 0, 0, 0, 0]].concat(), 9);
        refused(&[&time[..], &[0, 16, 14, 0, 0]].concat(), 9);
        refused(&[4, 2, 0, 0, 0, 0xC3, 0x28], 1);
        refused(&[12, 1, 0, 0, 0, 1, 0, 0, 0, 0xFF, 0], 5);
    }

    #[test]
    fn repeated_keys_and_replicas_are_refused() {
        let mut repeated_key = vec![12, 2, 0, 0, 0];
        for _ in 0..2 {
            repeated_key.extend_from_slice(&[4, 0, 0, 0]);
            repeated_key.extend_from_slice(b"city");
            repeated_key.push(0);
        }
        let error = decode_value(&repeated_key, &mut 0).unwrap_err().to_string();
        assert!(
            error.contains("city") && error.contains("byte 14:"),
            "{error}"
        );

        let mut repeated_replica = vec![15, 2, 0, 0, 0];
        for count in [3u64, 19] {
            repeated_replica.extend_from_slice(&[4, 0, 0, 0]);
            repeated_replica.extend_from_slice(b"Alix");
            repeated_replica.extend_from_slice(&count.to_le_bytes());
        }
        let error = decode_value(&repeated_replica, &mut 0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Alix") && error.contains("byte 21:"),
            "{error}"
        );
    }

    #[test]
    fn keys_and_replicas_out_of_order_are_refused() {
        let map_of = |keys: [&str; 2]| {
            let mut bytes = vec![12, 2, 0, 0, 0];
            for key in keys {
                bytes.extend_from_slice(&u32::try_from(key.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(key.as_bytes());
                bytes.push(0);
            }
            bytes
        };
        assert!(
            decode_value(&map_of(["city", "stops"]), &mut 0).is_ok(),
            "increasing keys decode"
        );
        let error = decode_value(&map_of(["stops", "city"]), &mut 0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("\"city\"") && error.contains("\"stops\"") && error.contains("byte 15:"),
            "{error}"
        );

        let counter_of = |replicas: [&str; 2]| {
            let mut bytes = vec![15, 2, 0, 0, 0];
            for replica in replicas {
                bytes.extend_from_slice(&u32::try_from(replica.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(replica.as_bytes());
                bytes.extend_from_slice(&88u64.to_le_bytes());
            }
            bytes
        };
        assert!(
            decode_value(&counter_of(["Alix", "Gus"]), &mut 0).is_ok(),
            "increasing replicas decode"
        );
        let error = decode_value(&counter_of(["Gus", "Alix"]), &mut 0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("\"Alix\"") && error.contains("\"Gus\"") && error.contains("byte 20:"),
            "{error}"
        );

        let on_counter_of = |negative: [&str; 2]| {
            let mut bytes = vec![16, 1, 0, 0, 0, 4, 0, 0, 0];
            bytes.extend_from_slice(b"Alix");
            bytes.extend_from_slice(&3u64.to_le_bytes());
            bytes.extend_from_slice(&counter_of(negative)[1..]);
            bytes
        };
        assert!(
            decode_value(&on_counter_of(["Alix", "Gus"]), &mut 0).is_ok(),
            "increasing negative replicas decode"
        );
        let error = decode_value(&on_counter_of(["Gus", "Alix"]), &mut 0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("\"Alix\"") && error.contains("\"Gus\"") && error.contains("byte 40:"),
            "the second negative replica: {error}"
        );
    }

    #[test]
    fn containers_side_by_side_do_not_count_as_nesting() {
        let path = || Value::Path {
            nodes: Arc::from(vec![
                map(vec![("city", Value::from("Paris"))]),
                map(vec![("city", Value::from("Prague"))]),
            ]),
            edges: Arc::from(vec![list(vec![Value::Int64(19)])]),
        };
        let wide = list((0..=MAX_VALUE_DEPTH).map(|_| path()).collect());
        let bytes = encoded(&wide);
        assert_eq!(encoded_len(&wide), bytes.len());
        let mut pos = 0;
        let back = decode_value(&bytes, &mut pos).unwrap();
        assert!(
            same(&wide, &back),
            "a list of {} paths is 3 deep",
            MAX_VALUE_DEPTH + 1
        );
        assert_eq!(pos, bytes.len());
    }

    #[test]
    fn encoding_refuses_what_decoding_would_refuse() {
        let deepest = nested(MAX_VALUE_DEPTH);
        let bytes = encoded(&deepest);
        assert_eq!(encoded_len(&deepest), bytes.len());
        let back = decode_value(&bytes, &mut 0).unwrap();
        assert!(same(&deepest, &back), "{MAX_VALUE_DEPTH} deep round trips");

        let mut out = vec![3, 19, 88];
        let error = encode_value(&nested(MAX_VALUE_DEPTH + 1), &mut out).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let error = error.to_string();
        assert!(error.contains("deeper"), "{error}");
        assert_eq!(out, [3, 19, 88], "a refused value appends nothing");

        let one_deeper = [&[11u8, 1, 0, 0, 0][..], &bytes].concat();
        let error = decode_value(&one_deeper, &mut 0).unwrap_err().to_string();
        assert!(error.contains("deeper"), "{error}");
    }

    /// The write limit leaves two levels below the codec's: a history value
    /// (a list of `[epoch, value]` lists) of any value a write accepts still
    /// encodes and decodes.
    #[test]
    fn nests_too_deep_holds_values_two_levels_below_the_codec_limit() {
        assert_eq!(
            MAX_PROPERTY_VALUE_DEPTH, 128,
            "the GQL parser's nesting limit"
        );
        assert_eq!(MAX_VALUE_DEPTH, MAX_PROPERTY_VALUE_DEPTH + 2);
        for depth in [
            0,
            1,
            3,
            MAX_PROPERTY_VALUE_DEPTH,
            MAX_PROPERTY_VALUE_DEPTH + 1,
            MAX_VALUE_DEPTH,
            MAX_VALUE_DEPTH + 1,
            300,
        ] {
            let value = nested(depth);
            assert_eq!(
                nests_too_deep(&value),
                depth > MAX_PROPERTY_VALUE_DEPTH,
                "{depth} deep: refused by writes"
            );
            assert_eq!(
                encode_value(&value, &mut Vec::new()).is_err(),
                depth > MAX_VALUE_DEPTH,
                "{depth} deep: refused by the encoder"
            );
        }
        let deepest = nested(MAX_PROPERTY_VALUE_DEPTH);
        let version = list(vec![Value::Int64(88), deepest]);
        let history = list(vec![version]);
        let bytes = encoded(&history);
        let back = decode_value(&bytes, &mut 0).unwrap();
        assert!(
            same(&history, &back),
            "a history value of the deepest value"
        );

        // Width does not count, and each container kind adds a level.
        let wide = list((0..=MAX_VALUE_DEPTH).map(|_| nested(3)).collect());
        assert!(!nests_too_deep(&wide), "a list of shallow values");
        let in_map = map(vec![("Mia", nested(MAX_PROPERTY_VALUE_DEPTH))]);
        assert!(nests_too_deep(&in_map), "a map around the deepest value");
        let in_path = Value::Path {
            nodes: Arc::from(vec![Value::Int64(3)]),
            edges: Arc::from(vec![nested(MAX_PROPERTY_VALUE_DEPTH)]),
        };
        assert!(nests_too_deep(&in_path), "a path around the deepest value");
        assert!(!nests_too_deep(&Value::Vector(Arc::from(vec![
            3.0f32, 19.0
        ]))));
    }

    /// Every `Time` is within a day, the decoder's condition, so the encoder
    /// writes no time the decoder refuses: a stored value (the WAL, the 0.5.x
    /// readers) holding a time outside a day is refused when serde reads it.
    #[test]
    fn a_stored_value_with_a_time_outside_a_day_is_refused_by_serde() {
        #[derive(serde::Serialize)]
        struct RawTime {
            nanos: u64,
            offset: Option<i32>,
        }
        let config = bincode::config::standard();
        let stored = |nanos: u64| {
            let valid = Value::Time(Time::from_nanos(88).unwrap());
            let variant = bincode::serde::encode_to_vec(&valid, config).unwrap()[0];
            let mut bytes = vec![variant];
            let raw = RawTime {
                nanos,
                offset: None,
            };
            bytes.extend(bincode::serde::encode_to_vec(&raw, config).unwrap());
            bytes
        };
        let (back, _): (Value, _) = bincode::serde::decode_from_slice(&stored(88), config).unwrap();
        assert!(
            same(&back, &Value::Time(Time::from_nanos(88).unwrap())),
            "the raw form is a stored time: {back:?}"
        );
        let result: std::result::Result<(Value, usize), _> =
            bincode::serde::decode_from_slice(&stored(86_400_000_000_000), config);
        let error = result
            .expect_err("a stored time of a full day is refused")
            .to_string();
        assert!(error.contains("not within a day"), "{error}");
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct Property {
        #[serde(with = "serde_option_value")]
        default_value: Option<Value>,
    }

    #[test]
    fn default_values_pass_through_serde() {
        for default_value in [None, Some(every_kind()[13].clone())] {
            let config = bincode::config::standard();
            let bytes = bincode::serde::encode_to_vec(
                &Property {
                    default_value: default_value.clone(),
                },
                config,
            )
            .unwrap();
            let (back, _): (Property, _) =
                bincode::serde::decode_from_slice(&bytes, config).unwrap();
            assert!(
                match (&default_value, &back.default_value) {
                    (None, None) => true,
                    (Some(a), Some(b)) => same(a, b),
                    _ => false,
                },
                "{default_value:?} came back as {:?}",
                back.default_value
            );
        }
    }

    #[test]
    fn a_default_value_with_trailing_bytes_is_refused() {
        #[derive(serde::Serialize)]
        struct RawProperty {
            default_value: Option<Vec<u8>>,
        }
        let config = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(
            &RawProperty {
                default_value: Some(vec![0, 88]),
            },
            config,
        )
        .unwrap();
        let result: std::result::Result<(Property, usize), _> =
            bincode::serde::decode_from_slice(&bytes, config);
        let error = result
            .err()
            .expect("a null followed by a stray byte")
            .to_string();
        assert!(error.contains("trailing"), "{error}");
    }
}
