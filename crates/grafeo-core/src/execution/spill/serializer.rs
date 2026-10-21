//! Binary serialization for Values without serde overhead.
//!
//! This module provides efficient serialization for spilling operator state
//! to disk. The format is designed for:
//! - Minimal overhead (no schema, direct binary encoding)
//! - Fast serialization/deserialization
//! - Compact representation

use arcstr::ArcStr;
use grafeo_common::types::Value;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::sync::Arc;

use crate::execution::memory::{
    GroupedMemoryOwner, checked_grouped_bytes, grouped_hash_table_bytes, grouped_io_error,
    grouped_memory_limit,
};
use crate::execution::operators::OperatorError;
use crate::execution::operators::accumulator::{
    DECIMAL_SCRATCH_BYTES, representative_decimal, representative_f32, representative_float,
    representative_time,
};

// Type tags for Value variants
const TAG_NULL: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_INT64: u8 = 2;
const TAG_FLOAT64: u8 = 3;
const TAG_STRING: u8 = 4;
const TAG_BYTES: u8 = 5;
const TAG_TIMESTAMP: u8 = 6;
const TAG_LIST: u8 = 7;
const TAG_MAP: u8 = 8;
const TAG_VECTOR: u8 = 9;
const TAG_DATE: u8 = 10;
const TAG_TIME: u8 = 11;
const TAG_DURATION: u8 = 12;
const TAG_PATH: u8 = 13;
const TAG_ZONED_DATETIME: u8 = 14;
const TAG_GCOUNTER: u8 = 15;
const TAG_PNCOUNTER: u8 = 16;
const TAG_DECIMAL: u8 = 17;

fn invalid_capacity() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "invalid spill value capacity",
    )
}

fn decoded_length(value: u64) -> std::io::Result<usize> {
    usize::try_from(value).map_err(|_| invalid_capacity())
}

fn decoded_vec<T>(length: usize) -> std::io::Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| invalid_capacity())?;
    Ok(values)
}

fn decoded_bytes(length: usize) -> std::io::Result<Vec<u8>> {
    let mut bytes = decoded_vec(length)?;
    bytes.resize(length, 0);
    Ok(bytes)
}

/// Serializes a Value to bytes.
///
/// Returns the number of bytes written.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn serialize_value<W: Write + ?Sized>(value: &Value, w: &mut W) -> std::io::Result<usize> {
    match value {
        Value::Null => {
            w.write_all(&[TAG_NULL])?;
            Ok(1)
        }
        Value::Bool(b) => {
            w.write_all(&[TAG_BOOL, u8::from(*b)])?;
            Ok(2)
        }
        Value::Int64(i) => {
            w.write_all(&[TAG_INT64])?;
            w.write_all(&i.to_le_bytes())?;
            Ok(9)
        }
        Value::Float64(f) => {
            w.write_all(&[TAG_FLOAT64])?;
            w.write_all(&f.to_le_bytes())?;
            Ok(9)
        }
        Value::String(s) => {
            w.write_all(&[TAG_STRING])?;
            let bytes = s.as_bytes();
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)?;
            Ok(1 + 8 + bytes.len())
        }
        Value::Bytes(b) => {
            w.write_all(&[TAG_BYTES])?;
            w.write_all(&(b.len() as u64).to_le_bytes())?;
            w.write_all(b)?;
            Ok(1 + 8 + b.len())
        }
        Value::Timestamp(t) => {
            w.write_all(&[TAG_TIMESTAMP])?;
            // Timestamp is internally an i64 (microseconds since epoch)
            let micros = t.as_micros();
            w.write_all(&micros.to_le_bytes())?;
            Ok(9)
        }
        Value::List(items) => {
            w.write_all(&[TAG_LIST])?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            let mut total = 1 + 8;
            for item in items.iter() {
                total += serialize_value(item, w)?;
            }
            Ok(total)
        }
        Value::Map(map) => {
            w.write_all(&[TAG_MAP])?;
            w.write_all(&(map.len() as u64).to_le_bytes())?;
            let mut total = 1 + 8;
            for (key, val) in map.iter() {
                // Serialize key as string
                let key_bytes = key.as_str().as_bytes();
                w.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
                w.write_all(key_bytes)?;
                total += 8 + key_bytes.len();
                // Serialize value
                total += serialize_value(val, w)?;
            }
            Ok(total)
        }
        Value::Vector(v) => {
            w.write_all(&[TAG_VECTOR])?;
            w.write_all(&(v.len() as u64).to_le_bytes())?;
            for &f in v.iter() {
                w.write_all(&f.to_le_bytes())?;
            }
            Ok(1 + 8 + v.len() * 4)
        }
        Value::Date(d) => {
            w.write_all(&[TAG_DATE])?;
            w.write_all(&d.as_days().to_le_bytes())?;
            Ok(5)
        }
        Value::Time(t) => {
            w.write_all(&[TAG_TIME])?;
            w.write_all(&t.as_nanos().to_le_bytes())?;
            let offset = t.offset_seconds().unwrap_or(i32::MIN);
            w.write_all(&offset.to_le_bytes())?;
            Ok(13)
        }
        Value::Duration(d) => {
            w.write_all(&[TAG_DURATION])?;
            w.write_all(&d.months().to_le_bytes())?;
            w.write_all(&d.days().to_le_bytes())?;
            w.write_all(&d.nanos().to_le_bytes())?;
            Ok(25)
        }
        Value::ZonedDatetime(zdt) => {
            w.write_all(&[TAG_ZONED_DATETIME])?;
            w.write_all(&zdt.as_timestamp().as_micros().to_le_bytes())?;
            w.write_all(&zdt.offset_seconds().to_le_bytes())?;
            Ok(13)
        }
        Value::Path { nodes, edges } => {
            w.write_all(&[TAG_PATH])?;
            w.write_all(&(nodes.len() as u64).to_le_bytes())?;
            let mut total = 1 + 8;
            for node in nodes.iter() {
                total += serialize_value(node, w)?;
            }
            w.write_all(&(edges.len() as u64).to_le_bytes())?;
            total += 8;
            for edge in edges.iter() {
                total += serialize_value(edge, w)?;
            }
            Ok(total)
        }
        Value::GCounter(counts) => {
            w.write_all(&[TAG_GCOUNTER])?;
            w.write_all(&(counts.len() as u64).to_le_bytes())?;
            let mut total = 1 + 8;
            for (k, v) in counts.iter() {
                let key_bytes = k.as_bytes();
                w.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
                w.write_all(key_bytes)?;
                w.write_all(&v.to_le_bytes())?;
                total += 8 + key_bytes.len() + 8;
            }
            Ok(total)
        }
        Value::OnCounter { pos, neg } => {
            w.write_all(&[TAG_PNCOUNTER])?;
            let mut total = 1;
            for map in [pos.as_ref(), neg.as_ref()] {
                w.write_all(&(map.len() as u64).to_le_bytes())?;
                total += 8;
                for (k, v) in map {
                    let key_bytes = k.as_bytes();
                    w.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
                    w.write_all(key_bytes)?;
                    w.write_all(&v.to_le_bytes())?;
                    total += 8 + key_bytes.len() + 8;
                }
            }
            Ok(total)
        }
        // Tag, scale, then the mantissa as 16 little-endian bytes.
        Value::Decimal(d) => {
            w.write_all(&[TAG_DECIMAL, d.scale()])?;
            w.write_all(&d.mantissa().to_le_bytes())?;
            Ok(2 + 16)
        }
        // `Value` is `#[non_exhaustive]`: a kind added without a tag here
        // fails the spill instead of coming back as null.
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("the spill format has no tag for this kind of value: {other:?}"),
        )),
    }
}

/// Deserializes a Value from bytes.
///
/// # Errors
///
/// Returns an error if reading fails or the format is invalid.
pub fn deserialize_value<R: Read + ?Sized>(r: &mut R) -> std::io::Result<Value> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;

    match tag[0] {
        TAG_NULL => Ok(Value::Null),
        TAG_BOOL => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(Value::Bool(buf[0] != 0))
        }
        TAG_INT64 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(Value::Int64(i64::from_le_bytes(buf)))
        }
        TAG_FLOAT64 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(Value::Float64(f64::from_le_bytes(buf)))
        }
        TAG_STRING => {
            let mut len_buf = [0u8; 8];
            r.read_exact(&mut len_buf)?;
            let len = decoded_length(u64::from_le_bytes(len_buf))?;
            let mut str_buf = decoded_bytes(len)?;
            r.read_exact(&mut str_buf)?;
            let s = String::from_utf8(str_buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            Ok(Value::String(ArcStr::from(s)))
        }
        TAG_BYTES => {
            let mut len_buf = [0u8; 8];
            r.read_exact(&mut len_buf)?;
            let len = decoded_length(u64::from_le_bytes(len_buf))?;
            let mut bytes_buf = decoded_bytes(len)?;
            r.read_exact(&mut bytes_buf)?;
            Ok(Value::Bytes(Arc::from(bytes_buf)))
        }
        TAG_TIMESTAMP => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            let micros = i64::from_le_bytes(buf);
            Ok(Value::Timestamp(
                grafeo_common::types::Timestamp::from_micros(micros),
            ))
        }
        TAG_LIST => {
            let mut len_buf = [0u8; 8];
            r.read_exact(&mut len_buf)?;
            let len = decoded_length(u64::from_le_bytes(len_buf))?;
            let mut items = decoded_vec(len)?;
            for _ in 0..len {
                items.push(deserialize_value(r)?);
            }
            Ok(Value::List(Arc::from(items)))
        }
        TAG_MAP => {
            let mut len_buf = [0u8; 8];
            r.read_exact(&mut len_buf)?;
            let len = decoded_length(u64::from_le_bytes(len_buf))?;
            let mut map = BTreeMap::new();
            for _ in 0..len {
                // Read key
                let mut key_len_buf = [0u8; 8];
                r.read_exact(&mut key_len_buf)?;
                let key_len = decoded_length(u64::from_le_bytes(key_len_buf))?;
                let mut key_buf = decoded_bytes(key_len)?;
                r.read_exact(&mut key_buf)?;
                let key_str = String::from_utf8(key_buf).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                })?;
                // Read value
                let val = deserialize_value(r)?;
                map.insert(grafeo_common::types::PropertyKey::new(key_str), val);
            }
            Ok(Value::Map(Arc::new(map)))
        }
        TAG_VECTOR => {
            let mut len_buf = [0u8; 8];
            r.read_exact(&mut len_buf)?;
            let len = decoded_length(u64::from_le_bytes(len_buf))?;
            let mut floats = decoded_vec(len)?;
            let mut buf = [0u8; 4];
            for _ in 0..len {
                r.read_exact(&mut buf)?;
                floats.push(f32::from_le_bytes(buf));
            }
            Ok(Value::Vector(Arc::from(floats)))
        }
        TAG_DATE => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            Ok(Value::Date(grafeo_common::types::Date::from_days(
                i32::from_le_bytes(buf),
            )))
        }
        TAG_TIME => {
            let mut nanos_buf = [0u8; 8];
            r.read_exact(&mut nanos_buf)?;
            let nanos = u64::from_le_bytes(nanos_buf);
            let mut offset_buf = [0u8; 4];
            r.read_exact(&mut offset_buf)?;
            let offset = i32::from_le_bytes(offset_buf);
            let time = grafeo_common::types::Time::from_nanos(nanos).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid time nanos")
            })?;
            if offset == i32::MIN {
                Ok(Value::Time(time))
            } else {
                Ok(Value::Time(time.with_offset(offset)))
            }
        }
        TAG_DURATION => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            let months = i64::from_le_bytes(buf);
            r.read_exact(&mut buf)?;
            let days = i64::from_le_bytes(buf);
            r.read_exact(&mut buf)?;
            let nanos = i64::from_le_bytes(buf);
            Ok(Value::Duration(grafeo_common::types::Duration::new(
                months, days, nanos,
            )))
        }
        TAG_ZONED_DATETIME => {
            let mut micros_buf = [0u8; 8];
            r.read_exact(&mut micros_buf)?;
            let micros = i64::from_le_bytes(micros_buf);
            let mut offset_buf = [0u8; 4];
            r.read_exact(&mut offset_buf)?;
            let offset = i32::from_le_bytes(offset_buf);
            Ok(Value::ZonedDatetime(
                grafeo_common::types::ZonedDatetime::from_timestamp_offset(
                    grafeo_common::types::Timestamp::from_micros(micros),
                    offset,
                ),
            ))
        }
        TAG_PATH => {
            let mut len_buf = [0u8; 8];
            r.read_exact(&mut len_buf)?;
            let nodes_len = decoded_length(u64::from_le_bytes(len_buf))?;
            let mut nodes = decoded_vec(nodes_len)?;
            for _ in 0..nodes_len {
                nodes.push(deserialize_value(r)?);
            }
            r.read_exact(&mut len_buf)?;
            let edges_len = decoded_length(u64::from_le_bytes(len_buf))?;
            let mut edges = decoded_vec(edges_len)?;
            for _ in 0..edges_len {
                edges.push(deserialize_value(r)?);
            }
            Ok(Value::Path {
                nodes: Arc::from(nodes),
                edges: Arc::from(edges),
            })
        }
        TAG_GCOUNTER => {
            let mut u64_buf = [0u8; 8];
            r.read_exact(&mut u64_buf)?;
            let count = decoded_length(u64::from_le_bytes(u64_buf))?;
            let mut map = std::collections::HashMap::new();
            map.try_reserve(count).map_err(|_| invalid_capacity())?;
            for _ in 0..count {
                r.read_exact(&mut u64_buf)?;
                let key_len = decoded_length(u64::from_le_bytes(u64_buf))?;
                let mut key_buf = decoded_bytes(key_len)?;
                r.read_exact(&mut key_buf)?;
                let key = String::from_utf8(key_buf).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                })?;
                r.read_exact(&mut u64_buf)?;
                let value = u64::from_le_bytes(u64_buf);
                map.insert(key, value);
            }
            Ok(Value::GCounter(Arc::new(map)))
        }
        TAG_PNCOUNTER => {
            let mut u64_buf = [0u8; 8];
            let mut maps = [
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
            ];
            for map in &mut maps {
                r.read_exact(&mut u64_buf)?;
                let count = decoded_length(u64::from_le_bytes(u64_buf))?;
                map.try_reserve(count).map_err(|_| invalid_capacity())?;
                for _ in 0..count {
                    r.read_exact(&mut u64_buf)?;
                    let key_len = decoded_length(u64::from_le_bytes(u64_buf))?;
                    let mut key_buf = decoded_bytes(key_len)?;
                    r.read_exact(&mut key_buf)?;
                    let key = String::from_utf8(key_buf).map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                    })?;
                    r.read_exact(&mut u64_buf)?;
                    let value = u64::from_le_bytes(u64_buf);
                    map.insert(key, value);
                }
            }
            let [pos, neg] = maps;
            Ok(Value::OnCounter {
                pos: Arc::new(pos),
                neg: Arc::new(neg),
            })
        }
        TAG_DECIMAL => {
            let mut scale = [0u8; 1];
            r.read_exact(&mut scale)?;
            let mut mantissa = [0u8; 16];
            r.read_exact(&mut mantissa)?;
            grafeo_common::types::Decimal::new(i128::from_le_bytes(mantissa), scale[0])
                .map(Value::Decimal)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown value tag: {}", tag[0]),
        )),
    }
}

// The admitted decoder is deliberately separate from the legacy entrypoint:
// spill bytes stay identical, while grouped callers own every allocation.
struct Admission<'a> {
    owner: &'a mut GroupedMemoryOwner,
    original: usize,
    keep: bool,
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        if !self.keep {
            // All decoding/traversal locals have already dropped. Shrinking a
            // live grant cannot request additional manager capacity.
            let _ = self.owner.try_resize(self.original);
        }
    }
}

fn with_admission<T>(
    owner: &mut GroupedMemoryOwner,
    operation: impl FnOnce(&mut GroupedMemoryOwner) -> Result<T, OperatorError>,
) -> Result<T, OperatorError> {
    let original = owner.size();
    let mut admission = Admission {
        owner,
        original,
        keep: false,
    };
    let result = operation(admission.owner);
    admission.keep = result.is_ok();
    result
}

fn release_bytes(owner: &mut GroupedMemoryOwner, bytes: usize) -> Result<(), OperatorError> {
    let size = owner
        .size()
        .checked_sub(bytes)
        .ok_or_else(|| OperatorError::Internal("spill decoder released unowned bytes".into()))?;
    owner.try_resize(size)
}

fn checked_add(left: usize, right: usize) -> Result<usize, OperatorError> {
    left.checked_add(right).ok_or_else(grouped_memory_limit)
}

fn array_bytes<T>(count: usize) -> Result<usize, OperatorError> {
    std::alloc::Layout::array::<T>(count)
        .map(|layout| layout.size())
        .map_err(|_| grouped_io_error(invalid_capacity()))
}

fn arc_bytes<T>(count: usize) -> Result<usize, OperatorError> {
    // Rust 1.99 ArcInner is repr(C): two atomic usize counts followed by
    // aligned data. Layout::extend accounts for header/data and tail padding.
    let header =
        std::alloc::Layout::array::<usize>(2).map_err(|_| grouped_io_error(invalid_capacity()))?;
    let data =
        std::alloc::Layout::array::<T>(count).map_err(|_| grouped_io_error(invalid_capacity()))?;
    header
        .extend(data)
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| grouped_io_error(invalid_capacity()))
}

fn arcstr_bytes(length: usize) -> Result<usize, OperatorError> {
    // arcstr 1.2 ThinInner: repr(C, align(8)), two usize fields, then bytes.
    let bytes = checked_add(2 * std::mem::size_of::<usize>(), length)?;
    std::alloc::Layout::from_size_align(bytes, 8)
        .map(|layout| layout.pad_to_align().size())
        .map_err(|_| grouped_io_error(invalid_capacity()))
}

fn btree_node_bytes() -> Result<usize, OperatorError> {
    // Rust 1.99 btree/node.rs has capacity 11. A leaf stores a parent pointer,
    // two u16 fields, eleven keys and eleven values; an internal node adds
    // twelve pointers. Bound each repr(C) field's alignment padding, including
    // the internal-node tail. Allocated populated nodes never outnumber keys.
    type Key = grafeo_common::types::PropertyKey;
    let alignment = std::mem::align_of::<Key>()
        .max(std::mem::align_of::<Value>())
        .max(std::mem::align_of::<usize>())
        .max(2);
    let entries = checked_grouped_bytes(
        11,
        checked_add(std::mem::size_of::<Key>(), std::mem::size_of::<Value>())?,
    )?;
    let links = checked_grouped_bytes(13, std::mem::size_of::<usize>())?;
    checked_add(
        checked_add(entries, links)?,
        checked_add(4, checked_grouped_bytes(7, alignment - 1)?)?,
    )
}

fn btree_bytes(length: usize) -> Result<usize, OperatorError> {
    checked_grouped_bytes(length, btree_node_bytes()?)
}

fn btree_insert_scratch(length: usize) -> Result<usize, OperatorError> {
    // Splitting/promoting a node can temporarily create an empty sibling and
    // root before they join the populated-node bound. Neither survives as
    // extra retained storage; account them separately during construction.
    if length == 0 {
        Ok(0)
    } else {
        checked_grouped_bytes(2, btree_node_bytes()?)
    }
}

fn admitted_vec<T>(length: usize, owner: &mut GroupedMemoryOwner) -> Result<Vec<T>, OperatorError> {
    let bytes = array_bytes::<T>(length)?;
    owner.try_grow(bytes)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| grouped_memory_limit())?;
    Ok(values)
}

fn push_frame<T>(
    frames: &mut Vec<T>,
    frame: T,
    owner: &mut GroupedMemoryOwner,
) -> Result<(), OperatorError> {
    if frames.len() == frames.capacity() {
        let old_bytes = array_bytes::<T>(frames.capacity())?;
        let capacity = frames
            .capacity()
            .checked_mul(2)
            .ok_or_else(grouped_memory_limit)?
            .max(1);
        let new_bytes = array_bytes::<T>(capacity)?;
        // A realloc may retain the old buffer while creating the new buffer.
        owner.try_grow(new_bytes)?;
        frames
            .try_reserve_exact(capacity - frames.len())
            .map_err(|_| grouped_memory_limit())?;
        release_bytes(owner, old_bytes)?;
    }
    frames.push(frame);
    Ok(())
}

fn wire_bytes<const N: usize>(r: &mut dyn Read) -> Result<[u8; N], OperatorError> {
    let mut bytes = [0; N];
    r.read_exact(&mut bytes).map_err(grouped_io_error)?;
    Ok(bytes)
}

fn wire_len(r: &mut dyn Read) -> Result<usize, OperatorError> {
    decoded_length(u64::from_le_bytes(wire_bytes(r)?)).map_err(grouped_io_error)
}

fn admitted_string(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<String, OperatorError> {
    let length = wire_len(r)?;
    let mut bytes = admitted_vec::<u8>(length, owner)?;
    bytes.resize(length, 0);
    r.read_exact(&mut bytes).map_err(grouped_io_error)?;
    String::from_utf8(bytes).map_err(|_| {
        grouped_io_error(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "spill string is not UTF-8",
        ))
    })
}

fn admitted_arcstr(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<ArcStr, OperatorError> {
    let text = admitted_string(r, owner)?;
    owner.try_grow(arcstr_bytes(text.len())?)?;
    let value = ArcStr::try_alloc(&text).ok_or_else(grouped_memory_limit)?;
    let capacity = text.capacity();
    drop(text);
    release_bytes(owner, capacity)?;
    Ok(value)
}

fn canonical_scalar(value: Value, owner: &mut GroupedMemoryOwner) -> Result<Value, OperatorError> {
    Ok(match value {
        Value::Float64(value) => representative_float(value),
        Value::Decimal(value) => {
            if value.floor() == value
                && let Some(integer) = value.trunc_to_i64()
            {
                return Ok(Value::Int64(integer));
            }
            let original = owner.size();
            let admission = Admission {
                owner,
                original,
                keep: false,
            };
            admission.owner.try_grow(DECIMAL_SCRATCH_BYTES)?;
            representative_decimal(value)
        }
        Value::Time(value) => Value::Time(representative_time(value)),
        Value::ZonedDatetime(value) => Value::ZonedDatetime(
            grafeo_common::types::ZonedDatetime::from_timestamp_offset(value.as_timestamp(), 0),
        ),
        other => other,
    })
}

fn counter_admitted(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<Arc<std::collections::HashMap<String, u64>>, OperatorError> {
    let length = wire_len(r)?;
    owner.try_grow(grouped_hash_table_bytes::<(String, u64)>(length)?)?;
    owner.try_grow(arc_bytes::<std::collections::HashMap<String, u64>>(1)?)?;
    let mut counts = std::collections::HashMap::new();
    counts
        .try_reserve(length)
        .map_err(|_| grouped_memory_limit())?;
    for _ in 0..length {
        let key = admitted_string(r, owner)?;
        let value = u64::from_le_bytes(wire_bytes(r)?);
        counts.insert(key, value);
    }
    Ok(Arc::new(counts))
}

enum ArrayKind {
    List,
    PathNodes,
    PathEdges(Arc<[Value]>),
}

enum DecodeFrame {
    Values {
        values: Vec<Value>,
        remaining: usize,
        kind: ArrayKind,
    },
    Map {
        values: BTreeMap<grafeo_common::types::PropertyKey, Value>,
        remaining: usize,
        key: grafeo_common::types::PropertyKey,
        insertion_scratch: usize,
    },
}

fn array_storage(
    length: usize,
    owner: &mut GroupedMemoryOwner,
) -> Result<Vec<Value>, OperatorError> {
    owner.try_grow(arc_bytes::<Value>(length)?)?;
    admitted_vec(length, owner)
}

fn finish_array(
    values: Vec<Value>,
    owner: &mut GroupedMemoryOwner,
) -> Result<Arc<[Value]>, OperatorError> {
    let scratch = array_bytes::<Value>(values.capacity())?;
    let values = Arc::from(values);
    release_bytes(owner, scratch)?;
    Ok(values)
}

/// Decodes the existing Value bytes with a caller-owned admission reservation.
/// Heap storage stays charged on success; refusal/malformed input restores the
/// original reservation after all temporary containers have dropped.
pub(crate) fn deserialize_value_admitted(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<Value, OperatorError> {
    with_admission(owner, |owner| decode_value(r, owner, false))
}

/// Same-byte decoder for an already representative DISTINCT identity.
pub(crate) fn deserialize_value_representative_admitted(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
) -> Result<Value, OperatorError> {
    with_admission(owner, |owner| decode_value(r, owner, true))
}

fn decode_value(
    r: &mut dyn Read,
    owner: &mut GroupedMemoryOwner,
    normalize: bool,
) -> Result<Value, OperatorError> {
    let mut frames = Vec::<DecodeFrame>::new();
    loop {
        let tag = wire_bytes::<1>(r)?[0];
        let mut value = match tag {
            TAG_NULL | TAG_BOOL | TAG_INT64 | TAG_FLOAT64 | TAG_TIMESTAMP | TAG_DATE | TAG_TIME
            | TAG_DURATION | TAG_ZONED_DATETIME | TAG_DECIMAL => {
                // This branch contains only fixed-size, allocation-free scalar
                // variants; legacy validation/representation remains identical.
                let mut scalar = std::io::Cursor::new([tag]).chain(&mut *r);
                let value = deserialize_value(&mut scalar).map_err(grouped_io_error)?;
                if normalize {
                    canonical_scalar(value, owner)?
                } else {
                    value
                }
            }
            TAG_STRING => Value::String(admitted_arcstr(r, owner)?),
            TAG_BYTES => {
                let length = wire_len(r)?;
                owner.try_grow(arc_bytes::<u8>(length)?)?;
                let mut bytes = admitted_vec::<u8>(length, owner)?;
                bytes.resize(length, 0);
                r.read_exact(&mut bytes).map_err(grouped_io_error)?;
                let scratch = bytes.capacity();
                let bytes = Arc::from(bytes);
                release_bytes(owner, scratch)?;
                Value::Bytes(bytes)
            }
            TAG_VECTOR => {
                let length = wire_len(r)?;
                owner.try_grow(arc_bytes::<f32>(length)?)?;
                let mut values = admitted_vec::<f32>(length, owner)?;
                for _ in 0..length {
                    let value = f32::from_le_bytes(wire_bytes(r)?);
                    values.push(if normalize {
                        representative_f32(value)
                    } else {
                        value
                    });
                }
                let scratch = array_bytes::<f32>(values.capacity())?;
                let values = Arc::from(values);
                release_bytes(owner, scratch)?;
                Value::Vector(values)
            }
            TAG_GCOUNTER => Value::GCounter(counter_admitted(r, owner)?),
            TAG_PNCOUNTER => Value::OnCounter {
                pos: counter_admitted(r, owner)?,
                neg: counter_admitted(r, owner)?,
            },
            TAG_LIST | TAG_PATH => {
                let length = wire_len(r)?;
                let values = array_storage(length, owner)?;
                let kind = if tag == TAG_LIST {
                    ArrayKind::List
                } else {
                    ArrayKind::PathNodes
                };
                if length != 0 {
                    push_frame(
                        &mut frames,
                        DecodeFrame::Values {
                            values,
                            remaining: length,
                            kind,
                        },
                        owner,
                    )?;
                    continue;
                }
                let values = finish_array(values, owner)?;
                if tag == TAG_LIST {
                    Value::List(values)
                } else {
                    let length = wire_len(r)?;
                    let edges = array_storage(length, owner)?;
                    if length != 0 {
                        push_frame(
                            &mut frames,
                            DecodeFrame::Values {
                                values: edges,
                                remaining: length,
                                kind: ArrayKind::PathEdges(values),
                            },
                            owner,
                        )?;
                        continue;
                    }
                    Value::Path {
                        nodes: values,
                        edges: finish_array(edges, owner)?,
                    }
                }
            }
            TAG_MAP => {
                let length = wire_len(r)?;
                owner.try_grow(checked_add(
                    btree_bytes(length)?,
                    btree_insert_scratch(length)?,
                )?)?;
                owner.try_grow(arc_bytes::<
                    BTreeMap<grafeo_common::types::PropertyKey, Value>,
                >(1)?)?;
                let values = BTreeMap::new();
                if length != 0 {
                    let key = grafeo_common::types::PropertyKey::new(admitted_arcstr(r, owner)?);
                    push_frame(
                        &mut frames,
                        DecodeFrame::Map {
                            values,
                            remaining: length,
                            key,
                            insertion_scratch: btree_insert_scratch(length)?,
                        },
                        owner,
                    )?;
                    continue;
                }
                Value::Map(Arc::new(values))
            }
            _ => {
                return Err(grouped_io_error(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unknown spill value tag",
                )));
            }
        };
        loop {
            let Some(frame) = frames.pop() else {
                let scratch = array_bytes::<DecodeFrame>(frames.capacity())?;
                drop(frames);
                release_bytes(owner, scratch)?;
                return Ok(value);
            };
            match frame {
                DecodeFrame::Values {
                    mut values,
                    remaining,
                    kind,
                } => {
                    values.push(value);
                    if remaining > 1 {
                        // The stack capacity is already owned: popping and
                        // replacing this frame cannot grow the frame buffer.
                        frames.push(DecodeFrame::Values {
                            values,
                            remaining: remaining - 1,
                            kind,
                        });
                        break;
                    }
                    let values = finish_array(values, owner)?;
                    value = match kind {
                        ArrayKind::List => Value::List(values),
                        ArrayKind::PathEdges(nodes) => Value::Path {
                            nodes,
                            edges: values,
                        },
                        ArrayKind::PathNodes => {
                            let length = wire_len(r)?;
                            let edges = array_storage(length, owner)?;
                            if length != 0 {
                                frames.push(DecodeFrame::Values {
                                    values: edges,
                                    remaining: length,
                                    kind: ArrayKind::PathEdges(values),
                                });
                                break;
                            }
                            Value::Path {
                                nodes: values,
                                edges: finish_array(edges, owner)?,
                            }
                        }
                    };
                }
                DecodeFrame::Map {
                    mut values,
                    remaining,
                    key,
                    insertion_scratch,
                } => {
                    values.insert(key, value);
                    if remaining > 1 {
                        let key =
                            grafeo_common::types::PropertyKey::new(admitted_arcstr(r, owner)?);
                        frames.push(DecodeFrame::Map {
                            values,
                            remaining: remaining - 1,
                            key,
                            insertion_scratch,
                        });
                        break;
                    }
                    value = Value::Map(Arc::new(values));
                    release_bytes(owner, insertion_scratch)?;
                }
            }
        }
    }
}

// A depth-first iterator keeps scratch proportional to nesting depth, not to
// the number of list/map entries. Every frame capacity is admitted before grow.
enum ValueChildren<'a> {
    Slice(std::slice::Iter<'a, Value>),
    Map(std::collections::btree_map::Values<'a, grafeo_common::types::PropertyKey, Value>),
    Path {
        nodes: std::slice::Iter<'a, Value>,
        edges: std::slice::Iter<'a, Value>,
    },
}

impl<'a> ValueChildren<'a> {
    fn next(&mut self) -> Option<&'a Value> {
        match self {
            Self::Slice(values) => values.next(),
            Self::Map(values) => values.next(),
            Self::Path { nodes, edges } => nodes.next().or_else(|| edges.next()),
        }
    }
}

fn counter_heap(counts: &std::collections::HashMap<String, u64>) -> Result<usize, OperatorError> {
    let mut size = arc_bytes::<std::collections::HashMap<String, u64>>(1)?;
    size = checked_add(
        size,
        grouped_hash_table_bytes::<(String, u64)>(counts.capacity())?,
    )?;
    for key in counts.keys() {
        size = checked_add(size, key.capacity())?;
    }
    Ok(size)
}

fn immediate_heap(value: &Value) -> Result<usize, OperatorError> {
    Ok(match value {
        Value::Null
        | Value::Bool(_)
        | Value::Int64(_)
        | Value::Float64(_)
        | Value::Timestamp(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Duration(_)
        | Value::ZonedDatetime(_)
        | Value::Decimal(_) => 0,
        Value::String(value) => arcstr_bytes(value.len())?,
        Value::Bytes(value) => arc_bytes::<u8>(value.len())?,
        Value::Vector(value) => arc_bytes::<f32>(value.len())?,
        Value::List(value) => arc_bytes::<Value>(value.len())?,
        Value::Path { nodes, edges } => checked_add(
            arc_bytes::<Value>(nodes.len())?,
            arc_bytes::<Value>(edges.len())?,
        )?,
        Value::Map(values) => {
            let mut size = checked_add(
                btree_bytes(values.len())?,
                arc_bytes::<BTreeMap<grafeo_common::types::PropertyKey, Value>>(1)?,
            )?;
            for key in values.keys() {
                size = checked_add(size, arcstr_bytes(key.as_str().len())?)?;
            }
            size
        }
        Value::GCounter(counts) => counter_heap(counts)?,
        Value::OnCounter { pos, neg } => checked_add(counter_heap(pos)?, counter_heap(neg)?)?,
        _ => {
            return Err(OperatorError::InvalidValue(
                "unsupported grouped value variant".into(),
            ));
        }
    })
}

fn measure_value(
    value: &Value,
    owner: &mut GroupedMemoryOwner,
    hash_scratch: bool,
) -> Result<usize, OperatorError> {
    let original = owner.size();
    let admission = Admission {
        owner,
        original,
        keep: false,
    };
    let mut frames = Vec::<ValueChildren<'_>>::new();
    let mut current = Some(value);
    let mut total = 0;
    while let Some(value) = current {
        let size = if hash_scratch {
            let replicas = match value {
                Value::GCounter(counts) => counts.len(),
                Value::OnCounter { pos, neg } => pos.len().max(neg.len()),
                _ => 0,
            };
            // common HashableValue creates a reference vector and stable-sort
            // workspace (at most another vector's byte size) for a counter.
            checked_grouped_bytes(replicas, 2 * std::mem::size_of::<(&String, &u64)>())?
        } else {
            immediate_heap(value)?
        };
        total = checked_add(total, size)?;
        let children = match value {
            Value::List(values) => Some(ValueChildren::Slice(values.iter())),
            Value::Map(values) => Some(ValueChildren::Map(values.values())),
            Value::Path { nodes, edges } => Some(ValueChildren::Path {
                nodes: nodes.iter(),
                edges: edges.iter(),
            }),
            _ => None,
        };
        if let Some(mut children) = children
            && let Some(next) = children.next()
        {
            push_frame(&mut frames, children, admission.owner)?;
            current = Some(next);
            continue;
        }
        current = loop {
            match frames.last_mut() {
                Some(children) => match children.next() {
                    Some(next) => break Some(next),
                    None => {
                        frames.pop();
                    }
                },
                None => break None,
            }
        };
    }
    // frames drops before admission restores scratch, including on errors.
    drop(frames);
    drop(admission);
    Ok(total)
}

/// Conservative retained HEAP bound, excluding the enclosing Value slot.
/// Traversal workspace is admitted and released before this function returns.
pub(crate) fn value_retained_bytes(
    value: &Value,
    owner: &mut GroupedMemoryOwner,
) -> Result<usize, OperatorError> {
    measure_value(value, owner, false)
}

/// Temporary counter-identity hashing workspace, separate from retained bytes.
pub(crate) fn value_hash_scratch_bytes(
    value: &Value,
    owner: &mut GroupedMemoryOwner,
) -> Result<usize, OperatorError> {
    measure_value(value, owner, true)
}

enum CloneKind<'a> {
    List,
    PathNodes(&'a [Value]),
    PathEdges(Arc<[Value]>),
}

enum CloneFrame<'a> {
    Values {
        values: Vec<Value>,
        children: std::slice::Iter<'a, Value>,
        kind: CloneKind<'a>,
    },
    Map {
        values: BTreeMap<grafeo_common::types::PropertyKey, Value>,
        children: std::collections::btree_map::Iter<'a, grafeo_common::types::PropertyKey, Value>,
        key: grafeo_common::types::PropertyKey,
    },
}

fn copy_vec<T>(length: usize) -> Result<Vec<T>, OperatorError> {
    // Storage was admitted by the complete output/copy bound before this
    // traversal starts; only frame storage needs incremental admission here.
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| grouped_memory_limit())?;
    Ok(values)
}

/// Creates a canonical identity without recursive normalization or unowned
/// container copies. Shared Arc payloads count as retained by the new owner.
pub(crate) fn value_representative_admitted(
    value: &Value,
    owner: &mut GroupedMemoryOwner,
) -> Result<Value, OperatorError> {
    if matches!(
        value,
        Value::Null
            | Value::Bool(_)
            | Value::Int64(_)
            | Value::Float64(_)
            | Value::Timestamp(_)
            | Value::Date(_)
            | Value::Time(_)
            | Value::Duration(_)
            | Value::ZonedDatetime(_)
            | Value::Decimal(_)
    ) {
        return canonical_scalar(value.clone(), owner);
    }
    with_admission(owner, |owner| {
        let original = owner.size();
        let retained = value_retained_bytes(value, owner)?;
        // Output containers and their temporary Vec backing can coexist at
        // Arc conversion. Each is bounded by the full retained heap bound.
        owner.try_grow(retained)?;
        owner.try_grow(retained)?;
        let mut frames = Vec::<CloneFrame<'_>>::new();
        let mut current = value;
        let result = 'normalize: loop {
            let mut result = match current {
                Value::List(children)
                | Value::Path {
                    nodes: children, ..
                } => {
                    let values = copy_vec(children.len())?;
                    let kind = match current {
                        Value::Path { edges, .. } => CloneKind::PathNodes(edges),
                        _ => CloneKind::List,
                    };
                    let mut children = children.iter();
                    if let Some(next) = children.next() {
                        push_frame(
                            &mut frames,
                            CloneFrame::Values {
                                values,
                                children,
                                kind,
                            },
                            owner,
                        )?;
                        current = next;
                        continue;
                    }
                    match kind {
                        CloneKind::List => Value::List(Arc::from(values)),
                        CloneKind::PathNodes(edges) => {
                            let nodes = Arc::from(values);
                            let values = copy_vec(edges.len())?;
                            let mut children = edges.iter();
                            if let Some(next) = children.next() {
                                push_frame(
                                    &mut frames,
                                    CloneFrame::Values {
                                        values,
                                        children,
                                        kind: CloneKind::PathEdges(nodes),
                                    },
                                    owner,
                                )?;
                                current = next;
                                continue;
                            }
                            Value::Path {
                                nodes,
                                edges: Arc::from(values),
                            }
                        }
                        CloneKind::PathEdges(_) => unreachable!("initial clone kind is not edges"),
                    }
                }
                Value::Map(map) => {
                    let values = BTreeMap::new();
                    let mut children = map.iter();
                    if let Some((key, next)) = children.next() {
                        push_frame(
                            &mut frames,
                            CloneFrame::Map {
                                values,
                                children,
                                key: key.clone(),
                            },
                            owner,
                        )?;
                        current = next;
                        continue;
                    }
                    Value::Map(Arc::new(values))
                }
                Value::Vector(values) => {
                    let mut output = copy_vec(values.len())?;
                    output.extend(values.iter().map(|&value| representative_f32(value)));
                    Value::Vector(Arc::from(output))
                }
                value => canonical_scalar(value.clone(), owner)?,
            };
            loop {
                let Some(frame) = frames.pop() else {
                    break 'normalize result;
                };
                match frame {
                    CloneFrame::Values {
                        mut values,
                        mut children,
                        kind,
                    } => {
                        values.push(result);
                        if let Some(next) = children.next() {
                            frames.push(CloneFrame::Values {
                                values,
                                children,
                                kind,
                            });
                            current = next;
                            continue 'normalize;
                        }
                        let values = Arc::from(values);
                        result = match kind {
                            CloneKind::List => Value::List(values),
                            CloneKind::PathEdges(nodes) => Value::Path {
                                nodes,
                                edges: values,
                            },
                            CloneKind::PathNodes(edges) => {
                                let mut children = edges.iter();
                                let output = copy_vec(edges.len())?;
                                if let Some(next) = children.next() {
                                    frames.push(CloneFrame::Values {
                                        values: output,
                                        children,
                                        kind: CloneKind::PathEdges(values),
                                    });
                                    current = next;
                                    continue 'normalize;
                                }
                                Value::Path {
                                    nodes: values,
                                    edges: Arc::from(output),
                                }
                            }
                        };
                    }
                    CloneFrame::Map {
                        mut values,
                        mut children,
                        key,
                    } => {
                        values.insert(key, result);
                        if let Some((key, next)) = children.next() {
                            frames.push(CloneFrame::Map {
                                values,
                                children,
                                key: key.clone(),
                            });
                            current = next;
                            continue 'normalize;
                        }
                        result = Value::Map(Arc::new(values));
                    }
                }
            }
        };
        // Release copy/frame workspace only after their buffers have dropped.
        drop(frames);
        owner.try_resize(checked_add(original, retained)?)?;
        Ok(result)
    })
}

/// Serializes a row (slice of Values) to bytes.
///
/// Format: `[num_columns: u64][value1][value2]...`
///
/// Returns the number of bytes written.
///
/// # Errors
///
/// Returns an error if writing fails.
pub fn serialize_row<W: Write + ?Sized>(row: &[Value], w: &mut W) -> std::io::Result<usize> {
    w.write_all(&(row.len() as u64).to_le_bytes())?;
    let mut total = 8;
    for value in row {
        total += serialize_value(value, w)?;
    }
    Ok(total)
}

/// Deserializes a row from bytes.
///
/// # Arguments
///
/// * `r` - Reader to read from
/// * `expected_columns` - Expected number of columns (for validation, 0 to skip)
///
/// # Errors
///
/// Returns an error if reading fails or column count mismatches.
pub fn deserialize_row<R: Read + ?Sized>(
    r: &mut R,
    expected_columns: usize,
) -> std::io::Result<Vec<Value>> {
    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)?;
    // reason: column count is bounded by practical query sizes
    #[allow(clippy::cast_possible_truncation)]
    let num_columns = decoded_length(u64::from_le_bytes(len_buf))?;

    if expected_columns > 0 && num_columns != expected_columns {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Column count mismatch: expected {}, got {}",
                expected_columns, num_columns
            ),
        ));
    }

    let mut row = Vec::with_capacity(num_columns);
    for _ in 0..num_columns {
        row.push(deserialize_value(r)?);
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcstr::ArcStr;
    use std::io::Cursor;

    fn admitted_owner(budget: usize) -> GroupedMemoryOwner {
        use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
        let manager = BufferManager::new(BufferManagerConfig {
            hard_limit_fraction: 1.0,
            ..BufferManagerConfig::with_budget(budget)
        });
        GroupedMemoryOwner::try_new(manager, 0).unwrap()
    }

    #[test]
    fn admitted_decimals_keep_current_bytes_and_scale() {
        use grafeo_common::types::Decimal;
        for text in ["1.50", "-0.001", "99999999999999999999999999999999999999"] {
            let value = Value::Decimal(Decimal::parse(text).unwrap());
            let mut bytes = Vec::new();
            serialize_value(&value, &mut bytes).unwrap();
            assert_eq!((bytes[0], bytes.len()), (17, 18));
            let mut owner = admitted_owner(64);
            let decoded = deserialize_value_admitted(&mut bytes.as_slice(), &mut owner).unwrap();
            let Value::Decimal(decimal) = &decoded else {
                panic!("a decimal is preserved");
            };
            assert_eq!(decimal.to_string(), text);
            assert_eq!(value_retained_bytes(&decoded, &mut owner).unwrap(), 0);
            assert_eq!(owner.size(), 0, "a decimal has no retained heap");
            let mut encoded = Vec::new();
            serialize_value(&decoded, &mut encoded).unwrap();
            assert_eq!(encoded, bytes);
        }
    }

    #[test]
    fn admitted_decimal_normalization_reserves_temporary_workspace() {
        use grafeo_common::types::Decimal;
        let value = Value::Decimal(Decimal::parse("1.50").unwrap());
        let mut bytes = Vec::new();
        serialize_value(&value, &mut bytes).unwrap();
        let mut owner = admitted_owner(64);
        owner.try_grow(16).unwrap();
        assert!(matches!(
            deserialize_value_representative_admitted(&mut bytes.as_slice(), &mut owner),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(owner.size(), 16);
        assert!(matches!(
            value_representative_admitted(&value, &mut owner),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(owner.size(), 16);
        assert_eq!(owner.manager().allocated(), 16);

        let mut owner = admitted_owner(4096);
        assert_eq!(
            deserialize_value_representative_admitted(&mut bytes.as_slice(), &mut owner).unwrap(),
            Value::Float64(1.5)
        );
        assert_eq!(owner.size(), 0, "normalization releases its scratch");
        let whole = Value::Decimal(Decimal::parse("3.00").unwrap());
        let mut small_owner = admitted_owner(1);
        assert_eq!(
            value_representative_admitted(&whole, &mut small_owner).unwrap(),
            Value::Int64(3)
        );
        assert_eq!(small_owner.size(), 0);
    }

    #[test]
    fn admitted_nested_decimal_normalization_matches_scalar_identity() {
        use grafeo_common::types::Decimal;
        let value = Value::List(
            vec![
                Value::Decimal(Decimal::parse("3.00").unwrap()),
                Value::Decimal(Decimal::parse("-0.1250").unwrap()),
            ]
            .into(),
        );
        let expected = Value::List(vec![Value::Int64(3), Value::Float64(-0.125)].into());
        let mut bytes = Vec::new();
        serialize_value(&value, &mut bytes).unwrap();
        let mut owner = admitted_owner(4096);
        assert_eq!(
            deserialize_value_representative_admitted(&mut bytes.as_slice(), &mut owner).unwrap(),
            expected
        );
        let mut copy_owner = admitted_owner(4096);
        assert_eq!(
            value_representative_admitted(&value, &mut copy_owner).unwrap(),
            expected
        );
    }

    #[test]
    fn admitted_invalid_decimal_restores_original_reservation() {
        let mut bytes = vec![TAG_DECIMAL, u8::MAX];
        bytes.extend(1i128.to_le_bytes());
        let mut owner = admitted_owner(64);
        owner.try_grow(16).unwrap();
        assert!(deserialize_value_admitted(&mut bytes.as_slice(), &mut owner).is_err());
        assert_eq!(owner.size(), 16);
        assert_eq!(owner.manager().allocated(), 16);
    }

    #[test]
    fn admitted_nested_length_refusal_restores_owner() {
        let mut bytes = vec![TAG_LIST];
        bytes.extend(1u64.to_le_bytes());
        bytes.push(TAG_VECTOR);
        bytes.extend(u64::MAX.to_le_bytes());
        let mut owner = admitted_owner(4096);
        assert!(deserialize_value_admitted(&mut bytes.as_slice(), &mut owner).is_err());
        assert_eq!(owner.size(), 0);
        assert_eq!(owner.manager().allocated(), 0);
    }

    #[test]
    fn admitted_normalization_refusal_preserves_original_value_and_owner() {
        let value = Value::List(vec![Value::Float64(3.0); 8192].into());
        let mut owner = admitted_owner(4096);
        assert!(matches!(
            value_representative_admitted(&value, &mut owner),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(owner.size(), 0);
        assert!(matches!(&value, Value::List(items) if items.len() == 8192));
    }

    #[test]
    fn admitted_decode_and_normalization_keep_nested_identity() {
        let value =
            Value::List(vec![Value::Float64(3.0), Value::Vector(vec![-0.0, 2.0].into())].into());
        let mut bytes = Vec::new();
        serialize_value(&value, &mut bytes).unwrap();
        let mut owner = admitted_owner(1024 * 1024);
        let decoded =
            deserialize_value_representative_admitted(&mut bytes.as_slice(), &mut owner).unwrap();
        assert!(matches!(&decoded, Value::List(items) if matches!(items[0], Value::Int64(3))));
        let measured = value_retained_bytes(&decoded, &mut owner).unwrap();
        assert!(owner.size() >= measured);
        let mut other_owner = admitted_owner(1024 * 1024);
        let representative = value_representative_admitted(&value, &mut other_owner).unwrap();
        assert_eq!(representative, decoded);
    }

    fn roundtrip_value(value: Value) -> Value {
        let mut buf = Vec::new();
        serialize_value(&value, &mut buf).unwrap();
        let mut cursor = Cursor::new(buf);
        deserialize_value(&mut cursor).unwrap()
    }

    /// Every kind of value comes back as it went in (compared by its Debug
    /// form, so a NaN equals itself): a kind without a tag must not spill as
    /// null.
    #[test]
    fn every_kind_of_value_round_trips() {
        use grafeo_common::types::{
            Date, Decimal, Duration, PropertyKey, Time, Timestamp, ZonedDatetime,
        };
        use std::collections::HashMap;
        let counter = |pairs: &[(&str, u64)]| -> Arc<HashMap<String, u64>> {
            Arc::new(pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect())
        };
        let list = |items: Vec<Value>| Value::List(Arc::from(items));
        let map = Value::Map(Arc::new(
            [(PropertyKey::new("city"), Value::from("Prague"))]
                .into_iter()
                .collect(),
        ));
        for value in [
            Value::Null,
            Value::Bool(true),
            Value::Int64(i64::MIN),
            Value::Float64(f64::NAN),
            Value::Float64(-0.0),
            Value::from("Amsterdam"),
            Value::Bytes(Arc::from(vec![3u8, 19, 88])),
            Value::Timestamp(Timestamp::from_micros(1_696_500_000_123_457)),
            Value::Date(Date::from_days(-3)),
            Value::Time(Time::from_nanos(88).unwrap().with_offset(3600)),
            Value::Duration(Duration::new(3, 19, 88)),
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1_696_500_000_123_457),
                7200,
            )),
            list(vec![Value::Int64(3), list(vec![Value::from("Berlin")])]),
            map.clone(),
            Value::Vector(Arc::from(vec![3.0f32, -19.5])),
            Value::Path {
                nodes: Arc::from(vec![map.clone()]),
                edges: Arc::from(Vec::<Value>::new()),
            },
            Value::GCounter(counter(&[("Alix", 3)])),
            Value::OnCounter {
                pos: counter(&[("Mia", 88)]),
                neg: counter(&[("Jules", 3)]),
            },
            Value::Decimal(Decimal::parse("1.50").unwrap()),
        ] {
            assert_eq!(
                format!("{:?}", roundtrip_value(value.clone())),
                format!("{value:?}")
            );

            let mut bytes = Vec::new();
            serialize_value(&value, &mut bytes).unwrap();
            let mut owner = admitted_owner(1024 * 1024);
            owner.try_grow(32).unwrap();
            let manager = Arc::clone(owner.manager());
            let decoded = deserialize_value_admitted(&mut bytes.as_slice(), &mut owner).unwrap();
            assert_eq!(format!("{decoded:?}"), format!("{value:?}"));
            let retained = value_retained_bytes(&decoded, &mut owner).unwrap();
            assert!(owner.size() >= 32 + retained, "{value:?}");
            assert_eq!(manager.allocated(), owner.size());
            let mut encoded = Vec::new();
            serialize_value(&decoded, &mut encoded).unwrap();
            assert_eq!(encoded, bytes, "admission preserves the value's bytes");
            drop(decoded);
            owner.try_resize(32).unwrap();
            assert_eq!(manager.allocated(), 32);
            drop(owner);
            assert_eq!(manager.allocated(), 0);
        }
    }

    /// A spilled decimal comes back with its value and its scale.
    #[test]
    fn test_serialize_decimal() {
        for text in ["1.50", "-0.001", "99999999999999999999999999999999999999"] {
            let decimal = grafeo_common::types::Decimal::parse(text).unwrap();
            match roundtrip_value(Value::Decimal(decimal)) {
                Value::Decimal(back) => assert_eq!(back.to_string(), text),
                other => panic!("{text} came back as {other:?}"),
            }
        }
    }

    #[test]
    fn test_serialize_null() {
        let result = roundtrip_value(Value::Null);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_serialize_bool() {
        assert_eq!(roundtrip_value(Value::Bool(true)), Value::Bool(true));
        assert_eq!(roundtrip_value(Value::Bool(false)), Value::Bool(false));
    }

    #[test]
    fn test_serialize_int64() {
        assert_eq!(roundtrip_value(Value::Int64(0)), Value::Int64(0));
        assert_eq!(
            roundtrip_value(Value::Int64(i64::MAX)),
            Value::Int64(i64::MAX)
        );
        assert_eq!(
            roundtrip_value(Value::Int64(i64::MIN)),
            Value::Int64(i64::MIN)
        );
        assert_eq!(roundtrip_value(Value::Int64(-42)), Value::Int64(-42));
    }

    #[test]
    fn test_serialize_float64() {
        assert_eq!(roundtrip_value(Value::Float64(0.0)), Value::Float64(0.0));
        assert_eq!(
            roundtrip_value(Value::Float64(std::f64::consts::PI)),
            Value::Float64(std::f64::consts::PI)
        );
        // Note: NaN != NaN, so we test differently
        let nan_result = roundtrip_value(Value::Float64(f64::NAN));
        assert!(matches!(nan_result, Value::Float64(f) if f.is_nan()));
    }

    #[test]
    fn test_serialize_string() {
        let result = roundtrip_value(Value::String(ArcStr::from("hello world")));
        assert_eq!(result.as_str(), Some("hello world"));

        // Empty string
        let result = roundtrip_value(Value::String(ArcStr::from("")));
        assert_eq!(result.as_str(), Some(""));

        // Unicode
        let result = roundtrip_value(Value::String(ArcStr::from("héllo 世界 🌍")));
        assert_eq!(result.as_str(), Some("héllo 世界 🌍"));
    }

    #[test]
    fn test_serialize_bytes() {
        let data = vec![0u8, 1, 2, 255, 128];
        let result = roundtrip_value(Value::Bytes(Arc::from(data.clone())));
        assert_eq!(result.as_bytes(), Some(&data[..]));

        // Empty bytes
        let result = roundtrip_value(Value::Bytes(Arc::from(vec![])));
        assert_eq!(result.as_bytes(), Some(&[][..]));
    }

    #[test]
    fn test_serialize_timestamp() {
        let ts = grafeo_common::types::Timestamp::from_micros(1234567890);
        let result = roundtrip_value(Value::Timestamp(ts));
        assert_eq!(result.as_timestamp(), Some(ts));
    }

    #[test]
    fn test_serialize_list() {
        let list = Value::List(Arc::from(vec![
            Value::Int64(1),
            Value::String(ArcStr::from("two")),
            Value::Bool(true),
        ]));
        let result = roundtrip_value(list.clone());
        assert_eq!(result, list);

        // Nested list
        let nested = Value::List(Arc::from(vec![
            Value::List(Arc::from(vec![Value::Int64(1), Value::Int64(2)])),
            Value::List(Arc::from(vec![Value::Int64(3)])),
        ]));
        let result = roundtrip_value(nested.clone());
        assert_eq!(result, nested);

        // Empty list
        let empty = Value::List(Arc::from(vec![]));
        let result = roundtrip_value(empty.clone());
        assert_eq!(result, empty);
    }

    #[test]
    fn test_serialize_map() {
        let mut map = BTreeMap::new();
        map.insert(
            grafeo_common::types::PropertyKey::new("name"),
            Value::String(ArcStr::from("Alix")),
        );
        map.insert(
            grafeo_common::types::PropertyKey::new("age"),
            Value::Int64(30),
        );

        let value = Value::Map(Arc::new(map));
        let result = roundtrip_value(value.clone());
        assert_eq!(result, value);
    }

    #[test]
    fn test_serialize_row() {
        let row = vec![
            Value::Int64(1),
            Value::String(ArcStr::from("test")),
            Value::Bool(true),
            Value::Null,
        ];

        let mut buf = Vec::new();
        serialize_row(&row, &mut buf).unwrap();

        let mut cursor = Cursor::new(buf);
        let result = deserialize_row(&mut cursor, 4).unwrap();
        assert_eq!(result, row);
    }

    #[test]
    fn test_serialize_row_column_count_check() {
        let row = vec![Value::Int64(1), Value::Int64(2)];

        let mut buf = Vec::new();
        serialize_row(&row, &mut buf).unwrap();

        // Wrong expected column count
        let mut cursor = Cursor::new(buf.clone());
        let result = deserialize_row(&mut cursor, 3);
        assert!(result.is_err());

        // Skip check with 0
        let mut cursor = Cursor::new(buf);
        let result = deserialize_row(&mut cursor, 0).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_serialize_multiple_rows() {
        let rows = vec![
            vec![Value::Int64(1), Value::String(ArcStr::from("a"))],
            vec![Value::Int64(2), Value::String(ArcStr::from("b"))],
            vec![Value::Int64(3), Value::String(ArcStr::from("c"))],
        ];

        let mut buf = Vec::new();
        for row in &rows {
            serialize_row(row, &mut buf).unwrap();
        }

        let mut cursor = Cursor::new(buf);
        for expected in &rows {
            let result = deserialize_row(&mut cursor, 2).unwrap();
            assert_eq!(&result, expected);
        }
    }

    #[test]
    fn test_serialize_gcounter_roundtrip() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("replica-1".to_string(), 42u64);
        counts.insert("replica-2".to_string(), 17u64);
        let v = Value::GCounter(Arc::new(counts));
        let result = roundtrip_value(v.clone());
        assert_eq!(result, v);
    }

    #[test]
    fn test_serialize_gcounter_empty() {
        let v = Value::GCounter(Arc::new(std::collections::HashMap::new()));
        let result = roundtrip_value(v.clone());
        assert_eq!(result, v);
    }

    #[test]
    fn test_serialize_pncounter_roundtrip() {
        let mut pos = std::collections::HashMap::new();
        pos.insert("node-a".to_string(), 10u64);
        pos.insert("node-b".to_string(), 5u64);
        let mut neg = std::collections::HashMap::new();
        neg.insert("node-a".to_string(), 3u64);
        let v = Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        };
        let result = roundtrip_value(v.clone());
        assert_eq!(result, v);
    }

    #[test]
    fn test_serialization_size() {
        // Verify expected sizes
        let mut buf = Vec::new();

        // Null: 1 byte (tag only)
        serialize_value(&Value::Null, &mut buf).unwrap();
        assert_eq!(buf.len(), 1);
        buf.clear();

        // Bool: 2 bytes (tag + value)
        serialize_value(&Value::Bool(true), &mut buf).unwrap();
        assert_eq!(buf.len(), 2);
        buf.clear();

        // Int64: 9 bytes (tag + 8)
        serialize_value(&Value::Int64(42), &mut buf).unwrap();
        assert_eq!(buf.len(), 9);
        buf.clear();

        // String "hi": 11 bytes (tag + 8 length + 2)
        serialize_value(&Value::String(ArcStr::from("hi")), &mut buf).unwrap();
        assert_eq!(buf.len(), 11);
    }

    #[test]
    fn admitted_complex_payload_errors_restore_existing_reservation() {
        use grafeo_common::memory::buffer::BufferManager;
        use grafeo_common::types::PropertyKey;
        use grafeo_common::utils::error::Error;

        struct ObservedRead<'a> {
            remaining: &'a [u8],
            manager: &'a BufferManager,
            peak: usize,
        }

        impl Read for ObservedRead<'_> {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                self.peak = self.peak.max(self.manager.allocated());
                self.remaining.read(output)
            }
        }

        let map = Value::Map(Arc::new(BTreeMap::from([
            (
                PropertyKey::new("first"),
                Value::List(vec![Value::from("inside"), Value::Int64(7)].into()),
            ),
            (PropertyKey::new("last"), Value::from("payload")),
        ])));
        for value in [
            Value::from("retained payload"),
            map.clone(),
            Value::Path {
                nodes: vec![map, Value::from("last node")].into(),
                edges: vec![Value::from("edge payload")].into(),
            },
        ] {
            let mut bytes = Vec::new();
            serialize_value(&value, &mut bytes).unwrap();
            let mut measuring = admitted_owner(1024 * 1024);
            let retained = value_retained_bytes(&value, &mut measuring).unwrap();
            assert!(retained > 0);
            assert_eq!(measuring.size(), 0);
            drop(measuring);

            let mut owner = admitted_owner(32 + retained - 1);
            owner.try_grow(32).unwrap();
            let manager = Arc::clone(owner.manager());
            assert!(matches!(
                deserialize_value_admitted(&mut bytes.as_slice(), &mut owner),
                Err(OperatorError::LimitExceeded(_))
            ));
            assert_eq!(owner.size(), 32);
            assert_eq!(manager.allocated(), 32);
            drop(owner);
            assert_eq!(manager.allocated(), 0);

            let mut owner = admitted_owner(1024 * 1024);
            owner.try_grow(32).unwrap();
            let manager = Arc::clone(owner.manager());
            let mut reader = ObservedRead {
                remaining: &bytes[..bytes.len() - 1],
                manager: &manager,
                peak: 32,
            };
            let error = deserialize_value_admitted(&mut reader, &mut owner).unwrap_err();
            assert!(matches!(
                error,
                OperatorError::Wrapped(ref error)
                    if matches!(error.as_ref(), Error::Io(error)
                        if error.kind() == std::io::ErrorKind::UnexpectedEof)
            ));
            assert!(reader.peak > 32, "the payload must start decoding");
            assert_eq!(reader.remaining, &[] as &[u8]);
            assert_eq!(owner.size(), 32);
            assert_eq!(manager.allocated(), 32);
            drop(owner);
            assert_eq!(manager.allocated(), 0);
        }
    }
}
