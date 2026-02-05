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
            let len = u64::from_le_bytes(len_buf) as usize;
            let mut str_buf = vec![0u8; len];
            r.read_exact(&mut str_buf)?;
            let s = String::from_utf8(str_buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            Ok(Value::String(ArcStr::from(s)))
        }
        TAG_BYTES => {
            let mut len_buf = [0u8; 8];
            r.read_exact(&mut len_buf)?;
            let len = u64::from_le_bytes(len_buf) as usize;
            let mut bytes_buf = vec![0u8; len];
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
            let len = u64::from_le_bytes(len_buf) as usize;
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(deserialize_value(r)?);
            }
            Ok(Value::List(Arc::from(items)))
        }
        TAG_MAP => {
            let mut len_buf = [0u8; 8];
            r.read_exact(&mut len_buf)?;
            let len = u64::from_le_bytes(len_buf) as usize;
            let mut map = BTreeMap::new();
            for _ in 0..len {
                // Read key
                let mut key_len_buf = [0u8; 8];
                r.read_exact(&mut key_len_buf)?;
                let key_len = u64::from_le_bytes(key_len_buf) as usize;
                let mut key_buf = vec![0u8; key_len];
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
            let len = u64::from_le_bytes(len_buf) as usize;
            let mut floats = Vec::with_capacity(len);
            let mut buf = [0u8; 4];
            for _ in 0..len {
                r.read_exact(&mut buf)?;
                floats.push(f32::from_le_bytes(buf));
            }
            Ok(Value::Vector(Arc::from(floats)))
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown value tag: {}", tag[0]),
        )),
    }
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
    let num_columns = u64::from_le_bytes(len_buf) as usize;

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

    fn roundtrip_value(value: Value) -> Value {
        let mut buf = Vec::new();
        serialize_value(&value, &mut buf).unwrap();
        let mut cursor = Cursor::new(buf);
        deserialize_value(&mut cursor).unwrap()
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
            Value::String(ArcStr::from("Alice")),
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
}
