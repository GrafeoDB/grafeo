//! Rust-to-JavaScript value conversions for WASM bindings.

use grafeo_common::types::Value;
use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

/// Converts a Grafeo [`Value`] to a JavaScript value.
pub fn value_to_js(value: &Value) -> JsValue {
    match value {
        Value::Null => JsValue::NULL,
        Value::Bool(b) => JsValue::from_bool(*b),
        Value::Int64(n) => JsValue::from_f64(*n as f64),
        Value::Float64(f) => JsValue::from_f64(*f),
        Value::String(s) => JsValue::from_str(s),
        Value::Bytes(b) => {
            let arr = Uint8Array::new_with_length(b.len() as u32);
            arr.copy_from(b);
            arr.into()
        }
        Value::Timestamp(ts) => JsValue::from_str(&ts.to_string()),
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
