//! Converts between JavaScript and Grafeo value types.
//!
//! | JavaScript type  | Grafeo type   | Notes                          |
//! | ---------------- | ------------- | ------------------------------ |
//! | `null/undefined` | `Null`        |                                |
//! | `boolean`        | `Bool`        |                                |
//! | `number`         | `Int64/Float64` | Integer if no fractional part |
//! | `string`         | `String`      |                                |
//! | `Array`          | `List`        | Elements converted recursively |
//! | `Object`         | `Map`         | Keys must be strings           |
//! | `Buffer`         | `Bytes`       |                                |
//! | `Date`           | `Timestamp`   | Millisecond precision          |
//! | `BigInt`         | `Int64`       |                                |
//! | `Float32Array`   | `Vector`      |                                |

use std::collections::BTreeMap;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::{
    JsBigInt, JsBuffer, JsDate, JsObject, JsString, JsTypedArray, JsUnknown, NapiRaw, NapiValue,
    ValueType,
};

use grafeo_common::types::{PropertyKey, Timestamp, Value};

/// Converts a JavaScript value to a Grafeo Value.
pub fn js_to_value(env: &Env, val: JsUnknown) -> Result<Value> {
    #![allow(clippy::trivially_copy_pass_by_ref)] // Env refs are conventional in napi
    let value_type = val.get_type()?;
    match value_type {
        ValueType::Null | ValueType::Undefined => Ok(Value::Null),
        ValueType::Boolean => {
            let b = val.coerce_to_bool()?.get_value()?;
            Ok(Value::Bool(b))
        }
        ValueType::Number => {
            let n: f64 = val.coerce_to_number()?.get_double()?;
            // If the number is an integer within safe range, store as Int64
            if n.fract() == 0.0 && n.abs() < (1i64 << 53) as f64 {
                Ok(Value::Int64(n as i64))
            } else {
                Ok(Value::Float64(n))
            }
        }
        ValueType::String => {
            let s = val.coerce_to_string()?.into_utf8()?.into_owned()?;
            Ok(Value::String(s.into()))
        }
        ValueType::BigInt => {
            let mut bigint: JsBigInt = unsafe { val.cast() };
            let (_, words, _) = bigint.get_u128()?;
            // Truncate to i64 range
            Ok(Value::Int64(words as i64))
        }
        ValueType::Object => {
            let obj: JsObject = unsafe { val.cast() };
            js_object_to_value(env, &obj)
        }
        _ => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Unsupported JavaScript type: {:?}", value_type),
        )),
    }
}

/// Converts a JavaScript object (Array, Buffer, Date, or plain object) to a Grafeo Value.
fn js_object_to_value(env: &Env, obj: &JsObject) -> Result<Value> {
    if obj.is_array()? {
        let len = obj.get_array_length()?;
        let mut items = Vec::with_capacity(len as usize);
        for i in 0..len {
            let elem: JsUnknown = obj.get_element(i)?;
            items.push(js_to_value(env, elem)?);
        }
        return Ok(Value::List(items.into()));
    }

    if obj.is_buffer()? {
        let buf: JsBuffer = unsafe { JsUnknown::from_raw_unchecked(env.raw(), obj.raw()).cast() };
        let data = buf.into_value()?;
        return Ok(Value::Bytes(data.to_vec().into()));
    }

    if obj.is_date()? {
        let date: JsDate = unsafe { JsUnknown::from_raw_unchecked(env.raw(), obj.raw()).cast() };
        let ms = date.value_of()?;
        let micros = (ms * 1000.0) as i64;
        return Ok(Value::Timestamp(Timestamp::from_micros(micros)));
    }

    // Check for TypedArray (Float32Array for vectors)
    if obj.is_typedarray()? {
        let typedarray: JsTypedArray =
            unsafe { JsUnknown::from_raw_unchecked(env.raw(), obj.raw()).cast() };
        let ta_value = typedarray.into_value()?;
        if ta_value.typedarray_type == TypedArrayType::Float32 {
            let len = ta_value.length;
            // Read elements through the arraybuffer to avoid private field access
            let arraybuf_value = ta_value.arraybuffer.into_value()?;
            let byte_offset = ta_value.byte_offset;
            let bytes = &arraybuf_value[byte_offset..byte_offset + len * 4];
            let mut vec = Vec::with_capacity(len);
            for chunk in bytes.chunks_exact(4) {
                vec.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            return Ok(Value::Vector(vec.into()));
        }
    }

    // Plain object -> Map
    let keys = obj.get_property_names()?;
    let len = keys.get_array_length()?;
    let mut map = BTreeMap::new();
    for i in 0..len {
        let key: JsString = keys.get_element(i)?;
        let key_str = key.into_utf8()?.into_owned()?;
        let value: JsUnknown = obj.get_named_property(&key_str)?;
        map.insert(PropertyKey::new(key_str), js_to_value(env, value)?);
    }
    Ok(Value::Map(Arc::new(map)))
}

/// Converts a Grafeo Value to a JavaScript value.
#[allow(clippy::trivially_copy_pass_by_ref)] // Env refs are conventional in napi
pub fn value_to_js(env: &Env, value: &Value) -> Result<JsUnknown> {
    match value {
        Value::Null => Ok(env.get_null()?.into_unknown()),
        Value::Bool(b) => Ok(env.get_boolean(*b)?.into_unknown()),
        Value::Int64(i) => {
            // Use number for safe integer range, BigInt for larger values
            if *i > -(1i64 << 53) && *i < (1i64 << 53) {
                Ok(env.create_int64(*i)?.into_unknown())
            } else {
                Ok(env.create_bigint_from_i64(*i)?.into_unknown()?)
            }
        }
        Value::Float64(f) => Ok(env.create_double(*f)?.into_unknown()),
        Value::String(s) => Ok(env.create_string(s.as_ref())?.into_unknown()),
        Value::List(items) => {
            let mut arr = env.create_array_with_length(items.len())?;
            for (i, item) in items.iter().enumerate() {
                arr.set_element(i as u32, value_to_js(env, item)?)?;
            }
            Ok(arr.into_unknown())
        }
        Value::Map(map) => {
            let mut obj = env.create_object()?;
            for (key, val) in map.as_ref() {
                obj.set_named_property(key.as_str(), value_to_js(env, val)?)?;
            }
            Ok(obj.into_unknown())
        }
        Value::Bytes(bytes) => Ok(env.create_buffer_with_data(bytes.to_vec())?.into_unknown()),
        Value::Timestamp(ts) => {
            let ms = ts.as_micros() as f64 / 1000.0;
            Ok(env.create_date(ms)?.into_unknown())
        }
        Value::Vector(v) => {
            // Return as Float32Array for zero-copy efficiency
            let mut data = vec![0f32; v.len()];
            data.copy_from_slice(v);
            let buf = unsafe {
                env.create_arraybuffer_with_borrowed_data(
                    data.as_mut_ptr().cast(),
                    data.len() * std::mem::size_of::<f32>(),
                    data,
                    |_data, _hint| {},
                )?
            };
            let float32 = buf
                .value
                .into_typedarray(TypedArrayType::Float32, 0, v.len())?;
            Ok(float32.into_unknown())
        }
    }
}
