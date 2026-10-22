//! Bidirectional conversion between `serde_json::Value` and `grafeo Value`.
//!
//! Used by the Node.js and C bindings for parameter parsing and result
//! serialization. The C binding adds a thin wrapper for its `$timestamp_us`
//! convention on top of [`value_to_json`].

use std::collections::BTreeMap;
use std::sync::Arc;

use grafeo_common::types::{PropertyKey, Value};

/// Convert a `serde_json::Value` to a Grafeo [`Value`].
///
/// Numbers are parsed as `Int64` when they fit, otherwise `Float64`.
/// Objects with a `$timestamp_us` key are decoded as timestamps.
pub fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float64(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => match s.as_str() {
            "Infinity" => Value::Float64(f64::INFINITY),
            "-Infinity" => Value::Float64(f64::NEG_INFINITY),
            "NaN" => Value::Float64(f64::NAN),
            _ => Value::String(s.as_str().into()),
        },
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_value).collect();
            Value::List(items.into())
        }
        serde_json::Value::Object(obj) => {
            // Check for special $timestamp_us encoding (used by C binding).
            if let Some(ts) = obj.get("$timestamp_us").and_then(serde_json::Value::as_i64) {
                return Value::Timestamp(grafeo_common::types::Timestamp::from_micros(ts));
            }
            // Check for $date encoding
            if let Some(s) = obj.get("$date").and_then(serde_json::Value::as_str)
                && let Some(d) = grafeo_common::types::Date::parse(s)
            {
                return Value::Date(d);
            }
            // Check for $time encoding
            if let Some(s) = obj.get("$time").and_then(serde_json::Value::as_str)
                && let Some(t) = grafeo_common::types::Time::parse(s)
            {
                return Value::Time(t);
            }
            // Check for $duration encoding
            if let Some(s) = obj.get("$duration").and_then(serde_json::Value::as_str)
                && let Some(d) = grafeo_common::types::Duration::parse(s)
            {
                return Value::Duration(d);
            }
            // Check for $zoned_datetime encoding
            if let Some(s) = obj
                .get("$zoned_datetime")
                .and_then(serde_json::Value::as_str)
                && let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(s)
            {
                return Value::ZonedDatetime(zdt);
            }
            let mut map = BTreeMap::new();
            for (k, v) in obj {
                map.insert(PropertyKey::new(k.clone()), json_to_value(v));
            }
            Value::Map(Arc::new(map))
        }
    }
}

/// Convert a Grafeo [`Value`] to a `serde_json::Value`.
///
/// Timestamps are encoded as `{ "$timestamp_us": <micros> }`.
/// Bytes are encoded as a JSON array of integers for lossless roundtrip.
/// Vectors are encoded as a JSON array of floats.
pub fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int64(i) => serde_json::json!(*i),
        Value::Float64(f) => {
            // JSON does not support Infinity/NaN, encode as strings
            if f.is_infinite() {
                serde_json::Value::String(if f.is_sign_positive() {
                    "Infinity".to_string()
                } else {
                    "-Infinity".to_string()
                })
            } else if f.is_nan() {
                serde_json::Value::String("NaN".to_string())
            } else {
                serde_json::json!(*f)
            }
        }
        Value::String(s) => serde_json::Value::String(s.to_string()),
        Value::Bytes(b) => {
            let arr: Vec<serde_json::Value> =
                b.iter().map(|&byte| serde_json::json!(byte)).collect();
            serde_json::Value::Array(arr)
        }
        Value::Timestamp(ts) => serde_json::json!({ "$timestamp_us": ts.as_micros() }),
        Value::Date(d) => serde_json::json!({ "$date": d.to_string() }),
        Value::Time(t) => serde_json::json!({ "$time": t.to_string() }),
        Value::Duration(d) => serde_json::json!({ "$duration": d.to_string() }),
        Value::ZonedDatetime(zdt) => serde_json::json!({ "$zoned_datetime": zdt.to_string() }),
        Value::List(items) => serde_json::Value::Array(items.iter().map(value_to_json).collect()),
        Value::Map(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.as_str().to_string(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Vector(vec) => {
            serde_json::Value::Array(vec.iter().map(|&f| serde_json::json!(f)).collect())
        }
        Value::Path { nodes, edges } => {
            serde_json::json!({
                "$path": {
                    "nodes": nodes.iter().map(value_to_json).collect::<Vec<_>>(),
                    "edges": edges.iter().map(value_to_json).collect::<Vec<_>>(),
                }
            })
        }
        Value::GCounter(counts) => {
            let obj: serde_json::Map<String, serde_json::Value> = counts
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::json!(v)))
                .collect();
            serde_json::json!({ "$gcounter": serde_json::Value::Object(obj) })
        }
        Value::OnCounter { pos, neg } => {
            let pos_obj: serde_json::Map<String, serde_json::Value> = pos
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::json!(v)))
                .collect();
            let neg_obj: serde_json::Map<String, serde_json::Value> = neg
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::json!(v)))
                .collect();
            serde_json::json!({
                "$pncounter": {
                    "pos": serde_json::Value::Object(pos_obj),
                    "neg": serde_json::Value::Object(neg_obj),
                }
            })
        }
        _ => serde_json::Value::String(v.to_string()),
    }
}

/// Convert a JSON params object to a `HashMap<String, Value>`.
///
/// Returns `None` if the input is `None`. Returns an error string if the
/// input is not a JSON object.
///
/// # Errors
///
/// Returns `Err` if `params` is `Some` but the contained value is not a JSON object.
pub fn json_params_to_map(
    params: Option<&serde_json::Value>,
) -> Result<Option<std::collections::HashMap<String, Value>>, String> {
    let Some(params) = params else {
        return Ok(None);
    };
    let Some(obj) = params.as_object() else {
        return Err("params must be an object".into());
    };
    let mut map = std::collections::HashMap::with_capacity(obj.len());
    for (key, value) in obj {
        map.insert(key.clone(), json_to_value(value));
    }
    Ok(Some(map))
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::Timestamp;

    use super::*;

    #[test]
    fn roundtrip_primitives() {
        assert_eq!(json_to_value(&serde_json::json!(null)), Value::Null);
        assert_eq!(json_to_value(&serde_json::json!(true)), Value::Bool(true));
        assert_eq!(json_to_value(&serde_json::json!(42)), Value::Int64(42));
        assert_eq!(json_to_value(&serde_json::json!(1.5)), Value::Float64(1.5));
        assert_eq!(
            json_to_value(&serde_json::json!("hello")),
            Value::String("hello".into())
        );
    }

    #[test]
    fn roundtrip_timestamp() {
        let ts = Value::Timestamp(Timestamp::from_micros(1_000_000));
        let json = value_to_json(&ts);
        let back = json_to_value(&json);
        assert_eq!(ts, back);
    }

    #[test]
    fn roundtrip_nested() {
        let json = serde_json::json!({"a": [1, 2, 3], "b": {"c": true}});
        let val = json_to_value(&json);
        let back = value_to_json(&val);
        assert_eq!(json, back);
    }

    #[test]
    fn params_conversion() {
        let params = serde_json::json!({"name": "Alix", "age": 30});
        let map = json_params_to_map(Some(&params)).unwrap().unwrap();
        assert_eq!(map.get("name"), Some(&Value::String("Alix".into())));
        assert_eq!(map.get("age"), Some(&Value::Int64(30)));
    }

    #[test]
    fn params_none() {
        assert!(json_params_to_map(None).unwrap().is_none());
    }

    #[test]
    fn params_not_object() {
        let params = serde_json::json!([1, 2, 3]);
        assert!(json_params_to_map(Some(&params)).is_err());
    }

    #[test]
    fn roundtrip_bytes() {
        let v = Value::Bytes(Arc::new([1u8, 2, 3, 255]));
        let json = value_to_json(&v);
        // Bytes encode as array of integers
        assert!(json.is_array());
        assert_eq!(json[0], serde_json::json!(1u8));
        assert_eq!(json[3], serde_json::json!(255u8));
    }

    #[test]
    fn roundtrip_vector() {
        let v = Value::Vector(Arc::new([0.1f32, 0.2, 0.3]));
        let json = value_to_json(&v);
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 3);
    }

    #[test]
    fn roundtrip_path() {
        let v = Value::Path {
            nodes: Arc::new([Value::Int64(1), Value::Int64(2)]),
            edges: Arc::new([Value::String("e".into())]),
        };
        let json = value_to_json(&v);
        assert!(json.get("$path").is_some());
    }

    #[test]
    fn roundtrip_date() {
        let d = grafeo_common::types::Date::parse("2024-01-15").unwrap();
        let v = Value::Date(d);
        let json = value_to_json(&v);
        let back = json_to_value(&json);
        assert_eq!(v, back);
    }

    #[test]
    fn roundtrip_time() {
        let t = grafeo_common::types::Time::parse("14:30:00").unwrap();
        let v = Value::Time(t);
        let json = value_to_json(&v);
        let back = json_to_value(&json);
        assert_eq!(v, back);
    }

    #[test]
    fn roundtrip_duration() {
        let d = grafeo_common::types::Duration::parse("PT1H30M").unwrap();
        let v = Value::Duration(d);
        let json = value_to_json(&v);
        let back = json_to_value(&json);
        assert_eq!(v, back);
    }

    #[test]
    fn roundtrip_zoned_datetime() {
        let zdt = grafeo_common::types::ZonedDatetime::parse("2024-01-15T14:30:00+01:00").unwrap();
        let v = Value::ZonedDatetime(zdt);
        let json = value_to_json(&v);
        let back = json_to_value(&json);
        assert_eq!(v, back);
    }

    #[test]
    fn value_to_json_gcounter() {
        let mut counts = std::collections::HashMap::new();
        counts.insert("node-a".to_string(), 10u64);
        counts.insert("node-b".to_string(), 5u64);
        let v = Value::GCounter(Arc::new(counts));
        let json = value_to_json(&v);
        let gcounter = json.get("$gcounter").expect("should have $gcounter key");
        assert!(gcounter.is_object());
        assert_eq!(gcounter["node-a"], serde_json::json!(10u64));
        assert_eq!(gcounter["node-b"], serde_json::json!(5u64));
    }

    #[test]
    fn value_to_json_oncounter() {
        let mut pos = std::collections::HashMap::new();
        pos.insert("r1".to_string(), 8u64);
        let mut neg = std::collections::HashMap::new();
        neg.insert("r1".to_string(), 3u64);
        let v = Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        };
        let json = value_to_json(&v);
        let on = json.get("$pncounter").expect("should have $pncounter key");
        assert_eq!(on["pos"]["r1"], serde_json::json!(8u64));
        assert_eq!(on["neg"]["r1"], serde_json::json!(3u64));
    }

    #[test]
    fn json_to_value_invalid_date_falls_through_to_map() {
        // $date with non-parseable string falls through to plain Map
        let json = serde_json::json!({ "$date": "not-a-date" });
        let v = json_to_value(&json);
        assert!(matches!(v, Value::Map(_)));
    }

    #[test]
    fn json_float_becomes_float64() {
        // f64 value that does not fit in i64
        let json = serde_json::json!(1.5e300f64);
        let v = json_to_value(&json);
        assert!(matches!(v, Value::Float64(_)));
    }
}
