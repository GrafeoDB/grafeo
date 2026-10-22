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
use std::ffi::CString;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::{JsDate, JsString, JsValue, ValueType, sys};

use grafeo_common::types::{PropertyKey, Timestamp, Value};

/// Converts a JavaScript value to a Grafeo Value.
pub fn js_to_value(env: &Env, val: Unknown<'_>) -> Result<Value> {
    #![allow(clippy::trivially_copy_pass_by_ref)] // Env refs are conventional in napi
    let value_type = val.get_type()?;
    match value_type {
        ValueType::Null | ValueType::Undefined => Ok(Value::Null),
        ValueType::Boolean => {
            let b = val.coerce_to_bool()?;
            Ok(Value::Bool(b))
        }
        ValueType::Number => {
            let n: f64 = val.coerce_to_number()?.get_double()?;
            // If the number is an integer within safe range, store as Int64
            if n.fract() == 0.0 && n.abs() < (1i64 << 53) as f64 {
                // reason: Value is a whole number in (-2^53, 2^53), fits in i64
                #[allow(clippy::cast_possible_truncation)]
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
            // SAFETY: type was checked as BigInt by the match arm, so cast is valid
            let bigint: BigInt = unsafe { val.cast()? };
            if bigint.words.len() > 1 {
                return Err(napi::Error::new(
                    napi::Status::InvalidArg,
                    "BigInt value too large for i64",
                ));
            }
            let word = if bigint.words.is_empty() {
                0u64
            } else {
                bigint.words[0]
            };
            let signed = if bigint.sign_bit {
                if word == i64::MIN.unsigned_abs() {
                    i64::MIN
                } else if let Ok(v) = i64::try_from(word) {
                    -v
                } else {
                    return Err(napi::Error::new(
                        napi::Status::InvalidArg,
                        "BigInt value too large for i64",
                    ));
                }
            } else {
                i64::try_from(word).map_err(|_| {
                    napi::Error::new(napi::Status::InvalidArg, "BigInt value too large for i64")
                })?
            };
            Ok(Value::Int64(signed))
        }
        ValueType::Object => {
            // SAFETY: type was checked as Object by the match arm, so cast is valid
            let obj: Object<'_> = unsafe { val.cast()? };
            js_object_to_value(env, &obj)
        }
        _ => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Unsupported JavaScript type: {:?}", value_type),
        )),
    }
}

/// Converts a JavaScript object (Array, Buffer, Date, or plain object) to a Grafeo Value.
fn js_object_to_value(env: &Env, obj: &Object<'_>) -> Result<Value> {
    if obj.is_array()? {
        let len = obj.get_array_length()?;
        let mut items = Vec::with_capacity(len as usize);
        for i in 0..len {
            let elem: Unknown<'_> = obj.get_element(i)?;
            items.push(js_to_value(env, elem)?);
        }
        return Ok(Value::List(items.into()));
    }

    if obj.is_buffer()? {
        // SAFETY: env and obj are valid napi values within this callback scope
        let unknown = unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()) };
        // SAFETY: obj.is_buffer() returned true, so casting to Buffer is valid
        let buf: Buffer = unsafe { unknown.cast()? };
        return Ok(Value::Bytes(buf.to_vec().into()));
    }

    if obj.is_date()? {
        // SAFETY: obj.is_date() returned true, and env/obj are valid in this scope
        let date: JsDate = unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
        let ms = date.value_of()?;
        // reason: JS Date.valueOf() returns ms since epoch; *1000 gives microseconds, fits in i64
        #[allow(clippy::cast_possible_truncation)]
        let micros = (ms * 1000.0) as i64;
        return Ok(Value::Timestamp(Timestamp::from_micros(micros)));
    }

    // Check for TypedArray (Float32Array for vectors)
    if obj.is_typedarray()? {
        // SAFETY: obj.is_typedarray() returned true, and env/obj are valid in this scope
        let ta: TypedArray<'_> =
            unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
        if ta.typed_array_type == TypedArrayType::Float32 {
            // SAFETY: typed array type was verified as Float32, and env/obj are valid
            let f32arr: Float32Array =
                unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
            return Ok(Value::Vector(f32arr.to_vec().into()));
        }
    }

    // Plain object -> Map
    let keys = obj.get_property_names()?;
    let len = keys.get_array_length()?;
    let mut map = BTreeMap::new();
    for i in 0..len {
        let key: JsString = keys.get_element(i)?;
        let key_str = key.into_utf8()?.into_owned()?;
        let value: Unknown<'_> = obj.get_named_property(&key_str)?;
        map.insert(PropertyKey::new(key_str), js_to_value(env, value)?);
    }
    Ok(Value::Map(Arc::new(map)))
}

/// Helper to check napi_status and convert to Result.
pub(crate) fn check_napi(status: sys::napi_status) -> Result<()> {
    if status == sys::Status::napi_ok {
        Ok(())
    } else {
        Err(napi::Error::new(
            napi::Status::GenericFailure,
            format!("napi call failed with status: {status:?}"),
        ))
    }
}

/// Converts a Grafeo Value to a raw napi value (no lifetime constraints).
///
/// This uses the raw napi C API to avoid lifetime issues when returning
/// JS values from `#[napi]` methods where `env` is taken by value.
pub fn value_to_napi(env: sys::napi_env, value: &Value) -> Result<sys::napi_value> {
    match value {
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Null => unsafe { <Null as ToNapiValue>::to_napi_value(env, Null) },
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Bool(b) => unsafe { <bool as ToNapiValue>::to_napi_value(env, *b) },
        Value::Int64(i) => {
            // Use number for safe integer range, BigInt for larger values
            if *i > -(1i64 << 53) && *i < (1i64 << 53) {
                // SAFETY: env is a valid napi_env passed by the caller
                unsafe { <i64 as ToNapiValue>::to_napi_value(env, *i) }
            } else {
                // SAFETY: env is a valid napi_env passed by the caller
                unsafe {
                    <BigInt as ToNapiValue>::to_napi_value(
                        env,
                        BigInt {
                            sign_bit: *i < 0,
                            words: vec![i.unsigned_abs()],
                        },
                    )
                }
            }
        }
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Float64(f) => unsafe { <f64 as ToNapiValue>::to_napi_value(env, *f) },
        // SAFETY: env is a valid napi_env passed by the caller
        Value::String(s) => unsafe { <&str as ToNapiValue>::to_napi_value(env, s.as_ref()) },
        Value::List(items) => {
            let mut arr = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_array_with_length writes to our out-pointer
            check_napi(unsafe {
                sys::napi_create_array_with_length(env, items.len(), &raw mut arr)
            })?;
            for (i, item) in items.iter().enumerate() {
                let val = value_to_napi(env, item)?;
                // SAFETY: env, arr, and val are valid napi values
                // reason: JS arrays are limited to 2^32-1 elements, so index fits u32
                #[allow(clippy::cast_possible_truncation)]
                check_napi(unsafe { sys::napi_set_element(env, arr, i as u32, val) })?;
            }
            Ok(arr)
        }
        Value::Map(map) => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
            for (key, val) in map.as_ref() {
                let key_cstr = CString::new(key.as_str())
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?;
                let napi_val = value_to_napi(env, val)?;
                // SAFETY: env, obj, key_cstr, and napi_val are all valid
                check_napi(unsafe {
                    sys::napi_set_named_property(env, obj, key_cstr.as_ptr(), napi_val)
                })?;
            }
            Ok(obj)
        }
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Bytes(bytes) => unsafe {
            <Buffer as ToNapiValue>::to_napi_value(env, Buffer::from(bytes.to_vec()))
        },
        Value::Timestamp(ts) => {
            let ms = ts.as_micros() as f64 / 1000.0;
            let env_wrapper = Env::from_raw(env);
            Ok(env_wrapper.create_date(ms)?.raw())
        }
        Value::Date(d) => {
            let s = d.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
        Value::Time(t) => {
            let s = t.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
        Value::Duration(d) => {
            let s = d.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
        Value::ZonedDatetime(zdt) => {
            let s = zdt.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
        // SAFETY: env is a valid napi_env passed by the caller
        Value::Vector(v) => unsafe {
            <Float32Array as ToNapiValue>::to_napi_value(env, Float32Array::new(v.to_vec()))
        },
        Value::Path { nodes, edges } => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;

            // Create nodes array
            let mut nodes_arr = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_array_with_length writes to our out-pointer
            check_napi(unsafe {
                sys::napi_create_array_with_length(env, nodes.len(), &raw mut nodes_arr)
            })?;
            for (i, node) in nodes.iter().enumerate() {
                let val = value_to_napi(env, node)?;
                // SAFETY: env, nodes_arr, and val are valid napi values
                // reason: JS arrays are limited to 2^32-1 elements
                #[allow(clippy::cast_possible_truncation)]
                check_napi(unsafe { sys::napi_set_element(env, nodes_arr, i as u32, val) })?;
            }

            // Create edges array
            let mut edges_arr = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_array_with_length writes to our out-pointer
            check_napi(unsafe {
                sys::napi_create_array_with_length(env, edges.len(), &raw mut edges_arr)
            })?;
            for (i, edge) in edges.iter().enumerate() {
                let val = value_to_napi(env, edge)?;
                // SAFETY: env, edges_arr, and val are valid napi values
                // reason: JS arrays are limited to 2^32-1 elements
                #[allow(clippy::cast_possible_truncation)]
                check_napi(unsafe { sys::napi_set_element(env, edges_arr, i as u32, val) })?;
            }

            let nodes_key = CString::new("nodes").expect("static string has no null bytes");
            let edges_key = CString::new("edges").expect("static string has no null bytes");
            // SAFETY: env, obj, and the key/value pointers are all valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, nodes_key.as_ptr(), nodes_arr)
            })?;
            // SAFETY: env, obj, and the key/value pointers are all valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, edges_key.as_ptr(), edges_arr)
            })?;

            Ok(obj)
        }
        Value::GCounter(counts) => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
            let mut replicas = std::ptr::null_mut();
            check_napi(unsafe { sys::napi_create_object(env, &raw mut replicas) })?;
            let mut total: u64 = 0;
            for (replica, count) in counts.iter() {
                total += count;
                let mut val = std::ptr::null_mut();
                let count_f64 = *count as f64;
                // SAFETY: env is valid; napi_create_double writes to our out-pointer
                check_napi(unsafe { sys::napi_create_double(env, count_f64, &raw mut val) })?;
                let key = CString::new(replica.as_str())
                    .map_err(|e| napi::Error::from_reason(e.to_string()))?;
                // SAFETY: env, replicas, key, and val are valid
                check_napi(unsafe {
                    sys::napi_set_named_property(env, replicas, key.as_ptr(), val)
                })?;
            }
            let gcounter_key = CString::new("$gcounter").expect("static string has no null bytes");
            // SAFETY: env, obj, and replicas are valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, gcounter_key.as_ptr(), replicas)
            })?;
            let mut total_val = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_double writes to our out-pointer
            check_napi(unsafe { sys::napi_create_double(env, total as f64, &raw mut total_val) })?;
            let value_key = CString::new("$value").expect("static string has no null bytes");
            // SAFETY: env, obj, and total_val are valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, value_key.as_ptr(), total_val)
            })?;
            Ok(obj)
        }
        Value::OnCounter { pos, neg } => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
            let pos_sum: u128 = pos.values().copied().map(u128::from).sum();
            let neg_sum: u128 = neg.values().copied().map(u128::from).sum();
            let net = if pos_sum >= neg_sum {
                (pos_sum - neg_sum) as f64
            } else {
                -((neg_sum - pos_sum) as f64)
            };
            let pncounter_key =
                CString::new("$pncounter").expect("static string has no null bytes");
            let mut true_val = std::ptr::null_mut();
            // SAFETY: env is valid; napi_get_boolean writes to our out-pointer
            check_napi(unsafe { sys::napi_get_boolean(env, true, &raw mut true_val) })?;
            // SAFETY: env, obj, and true_val are valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, pncounter_key.as_ptr(), true_val)
            })?;
            let mut net_val = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_double writes to our out-pointer
            check_napi(unsafe { sys::napi_create_double(env, net, &raw mut net_val) })?;
            let value_key = CString::new("$value").expect("static string has no null bytes");
            // SAFETY: env, obj, and net_val are valid
            check_napi(unsafe {
                sys::napi_set_named_property(env, obj, value_key.as_ptr(), net_val)
            })?;
            Ok(obj)
        }
        _ => {
            let s = value.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            unsafe { <&str as ToNapiValue>::to_napi_value(env, &s) }
        }
    }
}

/// Converts a Grafeo Value to a JavaScript Unknown value.
///
/// Uses `value_to_napi` internally and wraps the result as `Unknown`.
/// The lifetime is unconstrained (from `from_raw_unchecked`), so this is
/// safe to call from `#[napi]` methods where `env` is taken by value.
pub fn value_to_js(env: sys::napi_env, value: &Value) -> Result<Unknown<'_>> {
    let raw = value_to_napi(env, value)?;
    // SAFETY: env and raw are valid napi values produced by value_to_napi
    Ok(unsafe { Unknown::from_raw_unchecked(env, raw) })
}
