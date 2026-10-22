//! Rust-to-JavaScript value conversions for WASM bindings.

use grafeo_common::types::Value;
use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

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
