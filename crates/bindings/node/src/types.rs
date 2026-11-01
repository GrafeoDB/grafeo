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
//! | `BigInt`         | `Int64/Decimal` | Past 64 bits an exact `Decimal`, past 38 digits an error |
//! | `Float32Array`   | `Vector`      |                                |
//! | `Float64Array`   | `Vector`      | Each element the nearest 32-bit float |
//! | RDF/JS term      | `RdfTerm`     | `{ termType, value, language, datatype, equals }`, as results give it; read by `termType` and `value` |

use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi::{JsDate, JsString, JsValue, ValueType, sys};

use grafeo_common::types::rdf::{Term, UnsupportedTermKind};
use grafeo_common::types::{PropertyKey, Timestamp, Value};

/// The entries of a JavaScript object (not an array), each converted by
/// [`js_to_value`]; a conversion error names the entry as `{what} 'key'`, and
/// a value that is no such object gives `not_an_object`.
fn object_entries(
    env: Env,
    value: Unknown<'_>,
    what: &str,
    not_an_object: &str,
) -> Result<Vec<(String, Value)>> {
    let refuse = || napi::Error::new(napi::Status::InvalidArg, not_an_object);
    if value.get_type()? != ValueType::Object {
        return Err(refuse());
    }
    // SAFETY: the type was checked as Object above
    let object: Object<'_> = unsafe { value.cast()? };
    if object.is_array()? {
        return Err(refuse());
    }
    let keys = object.get_property_names()?;
    let len = keys.get_array_length()?;
    let mut entries = Vec::with_capacity(len as usize);
    for i in 0..len {
        let key: JsString = keys.get_element(i)?;
        let key = key.into_utf8()?.into_owned()?;
        let value: Unknown<'_> = object.get_named_property(&key)?;
        let value = js_to_named_value(env, value, what, &key)?;
        entries.push((key, value));
    }
    Ok(entries)
}

/// The JavaScript value of an argument, from napi.
///
/// # Safety
///
/// `env` and `napi_val` come from napi for this argument.
unsafe fn argument<'env>(env: sys::napi_env, napi_val: sys::napi_value) -> (Env, Unknown<'env>) {
    let env = Env::from_raw(env);
    // SAFETY: the caller passes what napi gave for this argument
    let value = unsafe { Unknown::from_raw_unchecked(env.raw(), napi_val) };
    (env, value)
}

/// Query parameters, converted from the JavaScript object by [`js_to_value`]
/// when the call is made, as a property is: a `BigInt` follows the integer
/// rule at any width and a whole number is an INT64 (a JSON step would turn
/// a `BigInt` past 64 bits into a string and 5000000000 into a float). An
/// error names the parameter; it is kept until [`Params::into_map`], so an
/// async call rejects its promise rather than throwing.
pub struct Params(std::result::Result<HashMap<String, Value>, String>);

impl Params {
    /// The parameters, or the error that converting them gave.
    pub fn into_map(self) -> Result<HashMap<String, Value>> {
        self.0
            .map_err(|message| crate::error::NodeGrafeoError::InvalidArgument(message).into())
    }
}

impl TypeName for Params {
    fn type_name() -> &'static str {
        "Record<string, unknown>"
    }

    fn value_type() -> ValueType {
        ValueType::Object
    }
}

impl FromNapiValue for Params {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        // SAFETY: napi calls this with the argument's env and value
        let (env, value) = unsafe { argument(env, napi_val) };
        let params = object_entries(env, value, "parameter", "params must be an object");
        Ok(Self(
            params
                .map(|entries| entries.into_iter().collect())
                .map_err(|error| error.reason),
        ))
    }
}

/// An object of properties (a batch or upsert row, a set of filters, an
/// edge's properties), converted as [`Params`] are: by [`js_to_value`] when
/// the call is made, an error naming the property and kept until
/// [`Properties::into_entries`].
pub struct Properties(std::result::Result<Vec<(String, Value)>, String>);

impl Properties {
    /// The properties, or the error that converting them gave.
    pub fn into_entries(self) -> Result<Vec<(String, Value)>> {
        self.0
            .map_err(|message| crate::error::NodeGrafeoError::InvalidArgument(message).into())
    }

    /// Rows of properties, an error naming the row (`rows[2]: ...`).
    pub fn rows(rows: Vec<Self>) -> Result<Vec<HashMap<PropertyKey, Value>>> {
        rows.into_iter()
            .enumerate()
            .map(|(i, row)| match row.0 {
                Ok(entries) => Ok(entries
                    .into_iter()
                    .map(|(key, value)| (PropertyKey::new(key), value))
                    .collect()),
                Err(message) => Err(crate::error::NodeGrafeoError::InvalidArgument(format!(
                    "rows[{i}]: {message}"
                ))
                .into()),
            })
            .collect()
    }
}

impl TypeName for Properties {
    fn type_name() -> &'static str {
        "Record<string, unknown>"
    }

    fn value_type() -> ValueType {
        ValueType::Object
    }
}

impl ValidateNapiValue for Properties {}

impl FromNapiValue for Properties {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        // SAFETY: napi calls this with the argument's env and value
        let (env, value) = unsafe { argument(env, napi_val) };
        Ok(Self(
            object_entries(env, value, "property", "properties must be an object")
                .map_err(|error| error.reason),
        ))
    }
}

/// [`js_to_value`] for a named value: a conversion error names it, as
/// `property 'v'`.
pub fn js_to_named_value(env: Env, val: Unknown<'_>, what: &str, name: &str) -> Result<Value> {
    js_to_value(&env, val).map_err(|error| {
        napi::Error::new(error.status, format!("{what} '{name}': {}", error.reason))
    })
}

/// Converts a JavaScript value to a Grafeo Value.
pub fn js_to_value(env: &Env, val: Unknown<'_>) -> Result<Value> {
    #![allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "napi passes Env by reference to every conversion"
    )]
    js_to_value_at(env, val, 0)
}

/// [`js_to_value`] for a value inside `depth` arrays and objects: one
/// nested deeper than a stored value is refused before its conversion
/// recurses further (a deep one would overflow the stack).
fn js_to_value_at(env: &Env, val: Unknown<'_>, depth: usize) -> Result<Value> {
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
            // An INT64 when it fits 64 bits, else an exact DECIMAL up to 38
            // digits: two words hold every one of them as an i128.
            let wide = match bigint.words.as_slice() {
                [] => Some(0),
                [low] => Some(i128::from(*low)),
                [low, high] => i128::try_from(u128::from(*low) | (u128::from(*high) << 64)).ok(),
                _ => None,
            };
            let wide = wide.map(|magnitude| {
                if bigint.sign_bit {
                    -magnitude
                } else {
                    magnitude
                }
            });
            wide.and_then(|integer| Value::from_wide_integer(integer).ok())
                .ok_or_else(|| {
                    let text = val
                        .coerce_to_string()
                        .and_then(|text| text.into_utf8())
                        .and_then(|text| text.into_owned())
                        .unwrap_or_default();
                    napi::Error::new(
                        napi::Status::InvalidArg,
                        format!("the integer {text} has more than 38 digits"),
                    )
                })
        }
        ValueType::Object => {
            // SAFETY: type was checked as Object by the match arm, so cast is valid
            let obj: Object<'_> = unsafe { val.cast()? };
            js_object_to_value(env, &obj, depth)
        }
        _ => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("Unsupported JavaScript type: {:?}", value_type),
        )),
    }
}

/// The error of an array or object nested deeper than a stored value.
fn too_deep() -> napi::Error {
    napi::Error::new(
        napi::Status::InvalidArg,
        format!(
            "the value nests arrays and objects more than {} levels deep",
            grafeo_common::storage::MAX_PROPERTY_VALUE_DEPTH
        ),
    )
}

/// Converts a JavaScript object (Array, Buffer, Date, or plain object),
/// inside `depth` arrays and objects, to a Grafeo Value.
fn js_object_to_value(env: &Env, obj: &Object<'_>, depth: usize) -> Result<Value> {
    if obj.is_array()? {
        if depth >= grafeo_common::storage::MAX_PROPERTY_VALUE_DEPTH {
            return Err(too_deep());
        }
        let len = obj.get_array_length()?;
        let mut items = Vec::with_capacity(len as usize);
        for i in 0..len {
            let elem: Unknown<'_> = obj.get_element(i)?;
            items.push(js_to_value_at(env, elem, depth + 1)?);
        }
        return Ok(Value::List(items.into()));
    }

    // A Float32Array or Float64Array is a vector. This comes before the
    // buffer check: Node-API reports every typed array as a buffer.
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
        if ta.typed_array_type == TypedArrayType::Float64 {
            // SAFETY: typed array type was verified as Float64, and env/obj are valid
            let f64arr: Float64Array =
                unsafe { Unknown::from_raw_unchecked(env.raw(), obj.raw()).cast()? };
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a vector holds 32-bit floats: each element is the nearest one, as vector() makes it"
            )]
            let elements: Vec<f32> = f64arr.iter().map(|x| *x as f32).collect();
            return Ok(Value::Vector(elements.into()));
        }
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

    // An RDF/JS term, as a SPARQL result gives it or another library makes
    // it, is that term.
    if let Some(term) = rdfjs_term(obj)? {
        return Ok(Value::RdfTerm(Arc::new(term)));
    }

    // Plain object -> Map
    if depth >= grafeo_common::storage::MAX_PROPERTY_VALUE_DEPTH {
        return Err(too_deep());
    }
    let keys = obj.get_property_names()?;
    let len = keys.get_array_length()?;
    let mut map = BTreeMap::new();
    for i in 0..len {
        let key: JsString = keys.get_element(i)?;
        let key_str = key.into_utf8()?.into_owned()?;
        let value: Unknown<'_> = obj.get_named_property(&key_str)?;
        map.insert(
            PropertyKey::new(key_str),
            js_to_value_at(env, value, depth + 1)?,
        );
    }
    // A tagged object (`{ $decimal: "1.50" }` alone, `{ $date: "2024-01-15" }`,
    // ...) is the value it stands for, as in the JSON the other bindings send.
    Ok(grafeo_bindings_common::json::tagged_value(&map)
        .unwrap_or_else(|| Value::Map(Arc::new(map))))
}

/// The RDF term an object in the RDF/JS data model stands for, read by its
/// `termType` (`NamedNode`, `BlankNode` or `Literal`) and `value`, and for
/// a literal its `language` and the `value` of its `datatype`. Properties
/// are read as JavaScript reads them, getters included, and other keys are
/// ignored: the terms of other libraries (N3.js, rdf-data-factory) read too.
/// `None` for an object of another shape.
fn rdfjs_term(obj: &Object<'_>) -> Result<Option<Term>> {
    let Some(term_type) = string_property(obj, "termType")? else {
        return Ok(None);
    };
    if !matches!(term_type.as_str(), "NamedNode" | "BlankNode" | "Literal") {
        return Ok(None);
    }
    let Some(value) = string_property(obj, "value")? else {
        return Ok(None);
    };
    let (language, datatype) = if term_type == "Literal" {
        let datatype: Unknown<'_> = obj.get_named_property("datatype")?;
        let datatype = if datatype.get_type()? == ValueType::Object {
            // SAFETY: the type was checked as Object above
            let datatype: Object<'_> = unsafe { datatype.cast()? };
            string_property(&datatype, "value")?
        } else {
            None
        };
        (string_property(obj, "language")?, datatype)
    } else {
        (None, None)
    };
    Ok(grafeo_bindings_common::json::rdfjs_term(
        &term_type,
        &value,
        language.as_deref(),
        datatype.as_deref(),
    ))
}

/// The string an object holds under `key`; `None` when it holds another
/// type or nothing.
fn string_property(obj: &Object<'_>, key: &str) -> Result<Option<String>> {
    let value: Unknown<'_> = obj.get_named_property(key)?;
    if value.get_type()? != ValueType::String {
        return Ok(None);
    }
    Ok(Some(value.coerce_to_string()?.into_utf8()?.into_owned()?))
}

/// The RDF term a raw JavaScript value stands for (see [`rdfjs_term`]);
/// `None` for a value that is no object or no term.
fn term_of(env: sys::napi_env, value: sys::napi_value) -> Result<Option<Term>> {
    if value.is_null() {
        return Ok(None);
    }
    // SAFETY: env and value come from napi for this call
    let value = unsafe { Unknown::from_raw_unchecked(env, value) };
    if value.get_type()? != ValueType::Object {
        return Ok(None);
    }
    // SAFETY: the type was checked as Object above
    let obj: Object<'_> = unsafe { value.cast()? };
    rdfjs_term(&obj)
}

/// `term.equals(other)` of the RDF/JS data model: true when `other` is a
/// term of the same `termType` and `value`, and for a literal the same
/// language (without case) and datatype; false for `null`, `undefined` and
/// any other value.
///
/// # Safety
///
/// napi calls it with the call's env and callback info.
unsafe extern "C" fn term_equals(
    env: sys::napi_env,
    info: sys::napi_callback_info,
) -> sys::napi_value {
    let equal = || -> Result<bool> {
        let mut argc = 1_usize;
        let mut argv = [std::ptr::null_mut(); 1];
        let mut this = std::ptr::null_mut();
        // SAFETY: napi passes a valid env and info; the out-pointers hold one argument
        check_napi(unsafe {
            sys::napi_get_cb_info(
                env,
                info,
                &raw mut argc,
                argv.as_mut_ptr(),
                &raw mut this,
                std::ptr::null_mut(),
            )
        })?;
        if argc == 0 {
            return Ok(false);
        }
        let Some(own) = term_of(env, this)? else {
            return Ok(false);
        };
        Ok(term_of(env, argv[0])? == Some(own))
    };
    let mut result = std::ptr::null_mut();
    match equal() {
        Ok(equal) => {
            // SAFETY: env is valid; napi_get_boolean writes to our out-pointer
            if unsafe { sys::napi_get_boolean(env, equal, &raw mut result) } != sys::Status::napi_ok
            {
                return std::ptr::null_mut();
            }
            result
        }
        Err(error) => {
            // A getter that threw left its exception pending: it propagates.
            if let Ok(message) = CString::new(error.reason) {
                // SAFETY: env is valid and the message a C string
                unsafe { sys::napi_throw_error(env, std::ptr::null(), message.as_ptr()) };
            }
            std::ptr::null_mut()
        }
    }
}

/// The `equals` function every term object shares, made once per
/// environment and kept in its instance data.
struct TermEquals(sys::napi_ref);

/// The shared `equals` function of [`TermEquals`].
fn equals_function(env: sys::napi_env) -> Result<sys::napi_value> {
    let wrapper = Env::from_raw(env);
    let mut function = std::ptr::null_mut();
    if let Some(TermEquals(reference)) = wrapper.get_instance_data::<TermEquals>()? {
        // SAFETY: the reference was made in this env and is never deleted
        check_napi(unsafe { sys::napi_get_reference_value(env, *reference, &raw mut function) })?;
        return Ok(function);
    }
    // SAFETY: env is valid; the name is a C string of 6 bytes and the
    // callback has napi's signature
    check_napi(unsafe {
        sys::napi_create_function(
            env,
            c"equals".as_ptr(),
            6,
            Some(term_equals),
            std::ptr::null_mut(),
            &raw mut function,
        )
    })?;
    let mut reference = std::ptr::null_mut();
    // SAFETY: env and function are valid; the reference keeps the function alive
    check_napi(unsafe { sys::napi_create_reference(env, function, 1, &raw mut reference) })?;
    wrapper.set_instance_data(TermEquals(reference), (), |_| {})?;
    Ok(function)
}

/// Gives a term object its `equals` method, not enumerable, so that the
/// object's keys, its JSON and a spread copy hold the term's fields only.
fn add_equals(env: sys::napi_env, obj: sys::napi_value) -> Result<()> {
    let descriptor = sys::napi_property_descriptor {
        utf8name: c"equals".as_ptr(),
        name: std::ptr::null_mut(),
        method: None,
        getter: None,
        setter: None,
        value: equals_function(env)?,
        attributes: sys::PropertyAttributes::default,
        data: std::ptr::null_mut(),
    };
    // SAFETY: env and obj are valid; the descriptor lives through the call
    check_napi(unsafe { sys::napi_define_properties(env, obj, 1, &raw const descriptor) })
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
        // `{ $decimal: "1.50" }`: the text with its scale, as parameters take it.
        Value::Decimal(d) => {
            let mut obj = std::ptr::null_mut();
            // SAFETY: env is valid; napi_create_object writes to our out-pointer
            check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
            let text = d.to_string();
            // SAFETY: env is a valid napi_env passed by the caller
            let text_val = unsafe { <&str as ToNapiValue>::to_napi_value(env, &text) }?;
            let key = CString::new("$decimal").expect("static string has no null bytes");
            // SAFETY: env, obj, and text_val are valid
            check_napi(unsafe { sys::napi_set_named_property(env, obj, key.as_ptr(), text_val) })?;
            Ok(obj)
        }
        Value::RdfTerm(term) => term_to_napi(env, term),
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

/// A JavaScript object with the string properties `entries`.
fn string_object(env: sys::napi_env, entries: &[(&str, &str)]) -> Result<sys::napi_value> {
    let mut obj = std::ptr::null_mut();
    // SAFETY: env is valid; napi_create_object writes to our out-pointer
    check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
    for (key, text) in entries {
        // SAFETY: env is a valid napi_env passed by the caller
        let text_val = unsafe { <&str as ToNapiValue>::to_napi_value(env, text) }?;
        let key = CString::new(*key).expect("static string has no null bytes");
        // SAFETY: env, obj, and text_val are valid
        check_napi(unsafe { sys::napi_set_named_property(env, obj, key.as_ptr(), text_val) })?;
    }
    Ok(obj)
}

/// An RDF term as the RDF/JS data model writes it, a plain object:
/// `{ termType: "NamedNode", value }`, `{ termType: "BlankNode", value }`
/// (the label without `_:`), or `{ termType: "Literal", value, language,
/// datatype: { termType: "NamedNode", value } }`, `language` `""` when the
/// literal has none and the datatype `rdf:langString` when it has one. Each
/// (the datatype too) has the data model's `equals(other)` method.
///
/// # Errors
///
/// For a term of a kind this binding does not know.
fn term_to_napi(env: sys::napi_env, term: &Term) -> Result<sys::napi_value> {
    let obj = match term {
        Term::Iri(iri) => {
            string_object(env, &[("termType", "NamedNode"), ("value", iri.as_str())])?
        }
        Term::BlankNode(node) => {
            string_object(env, &[("termType", "BlankNode"), ("value", node.id())])?
        }
        Term::Literal(literal) => {
            let obj = string_object(
                env,
                &[
                    ("termType", "Literal"),
                    ("value", literal.value()),
                    ("language", literal.language().unwrap_or("")),
                ],
            )?;
            let datatype = string_object(
                env,
                &[("termType", "NamedNode"), ("value", literal.datatype())],
            )?;
            add_equals(env, datatype)?;
            let key = CString::new("datatype").expect("static string has no null bytes");
            // SAFETY: env, obj, and datatype are valid
            check_napi(unsafe { sys::napi_set_named_property(env, obj, key.as_ptr(), datatype) })?;
            obj
        }
        other => {
            return Err(napi::Error::new(
                napi::Status::GenericFailure,
                UnsupportedTermKind::of(other).to_string(),
            ));
        }
    };
    add_equals(env, obj)?;
    Ok(obj)
}

/// A value converted off the JavaScript thread and made into JavaScript on
/// it, for the rows of `executeStream` and the properties of change events:
/// the JSON form of `grafeo_bindings_common::json` (`{ $date: ... }`,
/// `{ $decimal: ... }`), with each RDF term as the RDF/JS term object that
/// `execute` gives.
pub enum JsonWithTerms {
    /// A value without RDF terms, as JSON.
    Json(serde_json::Value),
    /// An RDF term.
    Term(Arc<Term>),
    /// A list.
    List(Vec<JsonWithTerms>),
    /// An object, its entries in order.
    Object(Vec<(String, JsonWithTerms)>),
}

impl JsonWithTerms {
    /// The form of `value`.
    ///
    /// # Errors
    ///
    /// For an RDF term of a kind this binding does not know.
    pub fn of(value: &Value) -> Result<Self> {
        let list = |items: &[Value]| -> Result<Self> {
            Ok(Self::List(
                items.iter().map(Self::of).collect::<Result<_>>()?,
            ))
        };
        Ok(match value {
            Value::RdfTerm(term) => Self::Term(Arc::clone(term)),
            Value::List(items) => list(items)?,
            Value::Map(map) => Self::Object(
                map.iter()
                    .map(|(key, value)| Ok((key.as_str().to_string(), Self::of(value)?)))
                    .collect::<Result<_>>()?,
            ),
            Value::Path { nodes, edges } => Self::Object(vec![(
                "$path".to_string(),
                Self::Object(vec![
                    ("nodes".to_string(), list(nodes)?),
                    ("edges".to_string(), list(edges)?),
                ]),
            )]),
            other => Self::Json(grafeo_bindings_common::json::value_to_json(other).map_err(
                |error| napi::Error::new(napi::Status::GenericFailure, error.to_string()),
            )?),
        })
    }
}

impl TypeName for JsonWithTerms {
    fn type_name() -> &'static str {
        "any"
    }

    fn value_type() -> ValueType {
        ValueType::Unknown
    }
}

impl ToNapiValue for JsonWithTerms {
    unsafe fn to_napi_value(env: sys::napi_env, val: Self) -> Result<sys::napi_value> {
        match val {
            // SAFETY: napi calls this with a valid env
            Self::Json(json) => unsafe { serde_json::Value::to_napi_value(env, json) },
            Self::Term(term) => term_to_napi(env, &term),
            Self::List(items) => {
                let mut arr = std::ptr::null_mut();
                // SAFETY: env is valid; napi_create_array_with_length writes to our out-pointer
                check_napi(unsafe {
                    sys::napi_create_array_with_length(env, items.len(), &raw mut arr)
                })?;
                for (i, item) in items.into_iter().enumerate() {
                    let index = u32::try_from(i).map_err(|_| {
                        napi::Error::from_reason("a list longer than a JavaScript array holds")
                    })?;
                    // SAFETY: napi calls this with a valid env
                    let item = unsafe { Self::to_napi_value(env, item) }?;
                    // SAFETY: env, arr, and item are valid napi values
                    check_napi(unsafe { sys::napi_set_element(env, arr, index, item) })?;
                }
                Ok(arr)
            }
            Self::Object(entries) => {
                let mut obj = std::ptr::null_mut();
                // SAFETY: env is valid; napi_create_object writes to our out-pointer
                check_napi(unsafe { sys::napi_create_object(env, &raw mut obj) })?;
                for (key, value) in entries {
                    let key =
                        CString::new(key).map_err(|e| napi::Error::from_reason(e.to_string()))?;
                    // SAFETY: napi calls this with a valid env
                    let value = unsafe { Self::to_napi_value(env, value) }?;
                    // SAFETY: env, obj, key, and value are valid
                    check_napi(unsafe {
                        sys::napi_set_named_property(env, obj, key.as_ptr(), value)
                    })?;
                }
                Ok(obj)
            }
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
