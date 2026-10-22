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
/// A number follows [`Value::from_number_text`] on its source text (the
/// workspace reads JSON with `arbitrary_precision`): an integer is an `Int64`
/// when it fits 64 bits, else an exact `Decimal` up to 38 digits; any other
/// number is a `Float64`. A tagged object (see [`tagged_value`]) is the value
/// it stands for.
///
/// # Errors
///
/// An integer with more than 38 digits or a float past the `Float64` range.
pub fn json_to_value(v: &serde_json::Value) -> Result<Value, String> {
    Ok(match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            Value::from_number_text(n.as_str()).map_err(|error| error.to_string())?
        }
        serde_json::Value::String(s) => match s.as_str() {
            "Infinity" => Value::Float64(f64::INFINITY),
            "-Infinity" => Value::Float64(f64::NEG_INFINITY),
            "NaN" => Value::Float64(f64::NAN),
            _ => Value::String(s.as_str().into()),
        },
        serde_json::Value::Array(arr) => {
            let items = arr
                .iter()
                .map(json_to_value)
                .collect::<Result<Vec<_>, _>>()?;
            Value::List(items.into())
        }
        serde_json::Value::Object(obj) => {
            let mut map = BTreeMap::new();
            for (k, v) in obj {
                map.insert(PropertyKey::new(k.clone()), json_to_value(v)?);
            }
            tagged_value(&map).unwrap_or_else(|| Value::Map(Arc::new(map)))
        }
    })
}

/// The value a tagged map stands for, the encoding JSON and JavaScript use
/// for the types they have no literal for: `{ "$timestamp_us": 1700000000000000 }`
/// (used by the C binding), `{ "$date": "2024-01-15" }`, `{ "$time": ... }`,
/// `{ "$duration": ... }`, `{ "$zoned_datetime": ... }`, and
/// `{ "$decimal": "1.50" }` as the map's only key (with other keys it is a
/// map). `None` when the map is none of these, or its text does not parse.
#[must_use]
pub fn tagged_value(map: &BTreeMap<PropertyKey, Value>) -> Option<Value> {
    let text = |key: &str| map.get(&PropertyKey::new(key)).and_then(Value::as_str);
    if let Some(Value::Int64(micros)) = map.get(&PropertyKey::new("$timestamp_us")) {
        return Some(Value::Timestamp(
            grafeo_common::types::Timestamp::from_micros(*micros),
        ));
    }
    if let Some(d) = text("$date").and_then(grafeo_common::types::Date::parse) {
        return Some(Value::Date(d));
    }
    if let Some(t) = text("$time").and_then(grafeo_common::types::Time::parse) {
        return Some(Value::Time(t));
    }
    if let Some(d) = text("$duration").and_then(grafeo_common::types::Duration::parse) {
        return Some(Value::Duration(d));
    }
    if map.len() == 1
        && let Some(Ok(d)) = text("$decimal").map(grafeo_common::types::Decimal::parse)
    {
        return Some(Value::Decimal(d));
    }
    if let Some(zdt) = text("$zoned_datetime").and_then(grafeo_common::types::ZonedDatetime::parse)
    {
        return Some(Value::ZonedDatetime(zdt));
    }
    None
}

/// Convert a Grafeo [`Value`] to a `serde_json::Value`.
///
/// Timestamps are encoded as `{ "$timestamp_us": <micros> }`, decimals as
/// `{ "$decimal": "1.50" }` (the text with its scale).
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
        Value::Decimal(d) => serde_json::json!({ "$decimal": d.to_string() }),
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
/// Returns `Err` if `params` is `Some` but the contained value is not a JSON
/// object, or a parameter has no value (see [`json_to_value`]); the error
/// names the parameter.
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
        let value = json_to_value(value).map_err(|error| format!("parameter '{key}': {error}"))?;
        map.insert(key.clone(), value);
    }
    Ok(Some(map))
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{Decimal, Timestamp};

    use super::*;

    /// [`super::json_to_value`] for a value that has one.
    fn json_to_value(v: &serde_json::Value) -> Value {
        super::json_to_value(v).unwrap()
    }

    /// JSON text read as the C binding and LOAD DATA read it.
    fn parsed(text: &str) -> serde_json::Value {
        serde_json::from_str(text).unwrap()
    }

    /// An integer past 64 bits is an exact decimal up to 38 digits, read from
    /// its source text; past 38 digits it is an error naming the parameter.
    #[test]
    fn integers_past_64_bits_are_exact_decimals() {
        let decimal = |text: &str| Value::Decimal(Decimal::parse(text).unwrap());
        let largest = "9".repeat(38);
        for text in [
            "9223372036854775808",
            "18446744073709551616",
            "-9223372036854775809",
            largest.as_str(),
        ] {
            assert_eq!(json_to_value(&parsed(text)), decimal(text), "{text}");
        }
        assert_eq!(
            json_to_value(&parsed("9223372036854775807")),
            Value::Int64(i64::MAX)
        );
        let nested = json_to_value(&parsed(
            r#"{"m": {"big": 18446744073709551616}, "l": [3, 9223372036854775808]}"#,
        ));
        let Value::Map(map) = nested else {
            panic!("{nested:?}")
        };
        assert_eq!(
            map.get(&PropertyKey::new("l")),
            Some(&Value::List(
                vec![Value::Int64(3), decimal("9223372036854775808")].into()
            ))
        );
        let too_wide = format!("1{}", "0".repeat(38));
        let error = json_params_to_map(Some(&parsed(&format!(
            r#"{{"x": 1, "big": [{too_wide}]}}"#
        ))))
        .unwrap_err();
        assert_eq!(
            error,
            format!("parameter 'big': the integer {too_wide} has more than 38 digits")
        );
        assert!(super::json_to_value(&parsed("1e400")).is_err());
    }

    /// A float keeps its value whatever its text: `1.50` is the float 1.5,
    /// and a value written back is the same JSON number.
    #[test]
    fn floats_keep_their_value_and_type() {
        for (text, float) in [
            ("1.50", 1.5),
            ("-0.25", -0.25),
            ("1e300", 1e300),
            ("2.0", 2.0),
        ] {
            assert_eq!(
                json_to_value(&parsed(text)),
                Value::Float64(float),
                "{text}"
            );
        }
        assert_eq!(value_to_json(&Value::Float64(1.5)), parsed("1.5"));
        assert_eq!(value_to_json(&Value::Int64(7)), parsed("7"));
    }

    /// A decimal travels as `{"$decimal": "1.50"}` both ways and keeps its
    /// scale; a tag that is not a decimal stays a map.
    #[test]
    fn decimals_round_trip_as_tagged_text() {
        let decimal = Value::Decimal(grafeo_common::types::Decimal::parse("-1.50").unwrap());
        let json = value_to_json(&decimal);
        assert_eq!(json, serde_json::json!({ "$decimal": "-1.50" }));
        match json_to_value(&json) {
            Value::Decimal(back) => assert_eq!(back.to_string(), "-1.50"),
            other => panic!("came back as {other:?}"),
        }
        let not_a_decimal = json_to_value(&serde_json::json!({ "$decimal": "1.2.3" }));
        assert!(matches!(not_a_decimal, Value::Map(_)), "{not_a_decimal:?}");
        // With other keys the object is a map, all of its keys kept.
        let map = json_to_value(&serde_json::json!({ "$decimal": "1", "other": 2 }));
        assert!(matches!(&map, Value::Map(m) if m.len() == 2), "{map:?}");
        // A plain JSON number stays a float.
        assert_eq!(json_to_value(&serde_json::json!(1.5)), Value::Float64(1.5));
    }

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
