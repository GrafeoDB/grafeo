//! Value conversions between JavaScript and Grafeo for the WASM bindings.

use std::collections::BTreeMap;
use std::sync::Arc;

use grafeo_common::types::{PropertyKey, Timestamp, Value};
use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

/// Converts a JavaScript value to a Grafeo [`Value`], as the Node.js binding
/// does: a whole number below 2^53 is an `Int64` and any other number a
/// `Float64`; a `BigInt` follows the integer rule at any width (an `Int64`
/// when it fits 64 bits, else an exact `Decimal` up to 38 digits); an array
/// is a list, a `Date` a timestamp, a `Float32Array` a vector, a `Uint8Array`
/// bytes, a `Map` or a plain object a map, and a tagged object
/// (`{ $decimal: "1.50" }`, `{ $date: "2024-01-15" }`, ...) the value it
/// stands for.
///
/// # Errors
///
/// A `BigInt` with more than 38 digits, or a value of no Grafeo type (a
/// function, a symbol).
pub fn js_to_value(value: &JsValue) -> Result<Value, String> {
    if value.is_null() || value.is_undefined() {
        return Ok(Value::Null);
    }
    if let Some(b) = value.as_bool() {
        return Ok(Value::Bool(b));
    }
    if let Some(n) = value.as_f64() {
        return Ok(if n.fract() == 0.0 && n.abs() < (1i64 << 53) as f64 {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the check above keeps n a whole number in (-2^53, 2^53), which an i64 holds exactly"
            )]
            Value::Int64(n as i64)
        } else {
            Value::Float64(n)
        });
    }
    if let Some(s) = value.as_string() {
        return Ok(Value::String(s.into()));
    }
    if value.is_bigint() {
        let text = String::from(
            js_sys::BigInt::from(value.clone())
                .to_string(10)
                .map_err(|_| "a BigInt without a decimal text".to_string())?,
        );
        return Value::from_number_text(&text).map_err(|error| error.to_string());
    }
    if Array::is_array(value) {
        let items = Array::from(value)
            .iter()
            .map(|item| js_to_value(&item))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Value::List(items.into()));
    }
    if let Some(date) = value.dyn_ref::<js_sys::Date>() {
        let millis = date.get_time();
        if millis.is_nan() {
            return Err("an invalid Date".to_string());
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "ECMAScript bounds a valid Date to 8.64e15 ms from the epoch, so its microseconds fit an i64"
        )]
        let micros = (millis * 1000.0) as i64;
        return Ok(Value::Timestamp(Timestamp::from_micros(micros)));
    }
    if let Some(vector) = value.dyn_ref::<Float32Array>() {
        return Ok(Value::Vector(vector.to_vec().into()));
    }
    if let Some(bytes) = value.dyn_ref::<Uint8Array>() {
        return Ok(Value::Bytes(bytes.to_vec().into()));
    }
    let Some(entries) = js_entries(value) else {
        return Err(format!(
            "Unsupported JavaScript type: {}",
            value.js_typeof().as_string().unwrap_or_default()
        ));
    };
    let mut map = BTreeMap::new();
    for (key, item) in entries? {
        let item = js_to_value(&item).map_err(|error| format!("key '{key}': {error}"))?;
        map.insert(PropertyKey::new(key), item);
    }
    Ok(grafeo_bindings_common::json::tagged_value(&map)
        .unwrap_or_else(|| Value::Map(Arc::new(map))))
}

/// The entries of a `Map` or a plain object, key first; `None` for any other
/// value (an array, a function, a primitive).
///
/// # Errors
///
/// A key that is not a string, or a `Map` that cannot be read.
pub fn js_entries(value: &JsValue) -> Option<Result<Vec<(String, JsValue)>, String>> {
    let pairs: Vec<Array> = if let Some(map) = value.dyn_ref::<js_sys::Map>() {
        match map
            .entries()
            .into_iter()
            .map(|entry| entry.map(|pair| Array::from(&pair)))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(pairs) => pairs,
            Err(_) => return Some(Err("a Map that cannot be read".to_string())),
        }
    } else if value.is_object() && !value.is_function() && !Array::is_array(value) {
        Object::entries(value.unchecked_ref::<Object>())
            .iter()
            .map(|pair| Array::from(&pair))
            .collect()
    } else {
        return None;
    };
    Some(
        pairs
            .into_iter()
            .map(|pair| match pair.get(0).as_string() {
                Some(key) => Ok((key, pair.get(1))),
                None => Err("a map key must be a string".to_string()),
            })
            .collect(),
    )
}

/// Converts a Grafeo [`Value`] to a JavaScript value.
pub fn value_to_js(value: &Value) -> JsValue {
    match value {
        Value::Null => JsValue::NULL,
        Value::Bool(b) => JsValue::from_bool(*b),
        Value::Int64(n) => {
            if *n > -(1i64 << 53) && *n < (1i64 << 53) {
                JsValue::from_f64(*n as f64)
            } else {
                js_sys::BigInt::from(*n).into()
            }
        }
        Value::Float64(f) => JsValue::from_f64(*f),
        Value::String(s) => JsValue::from_str(s),
        Value::Bytes(b) => {
            let arr = Uint8Array::new_with_length(b.len() as u32);
            arr.copy_from(b);
            arr.into()
        }
        Value::Timestamp(ts) => JsValue::from_str(&ts.to_string()),
        Value::Date(d) => JsValue::from_str(&d.to_string()),
        Value::Time(t) => JsValue::from_str(&t.to_string()),
        Value::Duration(d) => JsValue::from_str(&d.to_string()),
        Value::ZonedDatetime(zdt) => JsValue::from_str(&zdt.to_string()),
        Value::List(items) => {
            let arr = Array::new_with_length(items.len() as u32);
            for (i, item) in items.iter().enumerate() {
                arr.set(i as u32, value_to_js(item));
            }
            arr.into()
        }
        Value::Map(map) => {
            let obj = Object::new();
            for (key, val) in map.iter() {
                let _ = Reflect::set(&obj, &JsValue::from_str(key.as_str()), &value_to_js(val));
            }
            obj.into()
        }
        Value::Vector(v) => {
            let arr = Float32Array::new_with_length(v.len() as u32);
            arr.copy_from(v);
            arr.into()
        }
        Value::Path { nodes, edges } => {
            let obj = Object::new();
            let nodes_arr = Array::new_with_length(nodes.len() as u32);
            for (i, node) in nodes.iter().enumerate() {
                nodes_arr.set(i as u32, value_to_js(node));
            }
            let edges_arr = Array::new_with_length(edges.len() as u32);
            for (i, edge) in edges.iter().enumerate() {
                edges_arr.set(i as u32, value_to_js(edge));
            }
            let _ = Reflect::set(&obj, &JsValue::from_str("nodes"), &nodes_arr.into());
            let _ = Reflect::set(&obj, &JsValue::from_str("edges"), &edges_arr.into());
            let _ = Reflect::set(
                &obj,
                &JsValue::from_str("_type"),
                &JsValue::from_str("path"),
            );
            obj.into()
        }
        Value::GCounter(counts) => {
            let obj = Object::new();
            for (replica, count) in counts.iter() {
                let _ = Reflect::set(
                    &obj,
                    &JsValue::from_str(replica),
                    &JsValue::from_f64(*count as f64),
                );
            }
            let wrapper = Object::new();
            let _ = Reflect::set(&wrapper, &JsValue::from_str("$gcounter"), &obj.into());
            let _ = Reflect::set(
                &wrapper,
                &JsValue::from_str("$value"),
                &JsValue::from_f64(counts.values().copied().map(|v| v as f64).sum()),
            );
            wrapper.into()
        }
        Value::OnCounter { pos, neg } => {
            let pos_sum: u128 = pos.values().copied().map(u128::from).sum();
            let neg_sum: u128 = neg.values().copied().map(u128::from).sum();
            let net = if pos_sum >= neg_sum {
                (pos_sum - neg_sum) as f64
            } else {
                -((neg_sum - pos_sum) as f64)
            };
            let wrapper = Object::new();
            let _ = Reflect::set(
                &wrapper,
                &JsValue::from_str("$pncounter"),
                &JsValue::from_str("pncounter"),
            );
            let _ = Reflect::set(
                &wrapper,
                &JsValue::from_str("$value"),
                &JsValue::from_f64(net),
            );
            wrapper.into()
        }
        // `{ $decimal: "1.50" }`: the text with its scale, as parameters take it.
        Value::Decimal(d) => {
            let wrapper = Object::new();
            let _ = Reflect::set(
                &wrapper,
                &JsValue::from_str("$decimal"),
                &JsValue::from_str(&d.to_string()),
            );
            wrapper.into()
        }
        _ => JsValue::from_str(&value.to_string()),
    }
}

/// Converts a row of values to a JavaScript object with column names as keys.
pub fn row_to_js_object(columns: &[String], row: &[Value]) -> JsValue {
    let obj = Object::new();
    for (col, val) in columns.iter().zip(row.iter()) {
        let _ = Reflect::set(&obj, &JsValue::from_str(col), &value_to_js(val));
    }
    obj.into()
}
