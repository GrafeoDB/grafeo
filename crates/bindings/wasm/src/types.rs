//! Value conversions between JavaScript and Grafeo for the WASM bindings.

use std::collections::BTreeMap;
use std::sync::Arc;

use grafeo_common::types::rdf::{Term, UnsupportedTermKind};
use grafeo_common::types::{PropertyKey, Timestamp, Value};
use js_sys::{Array, Float32Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

/// Converts a JavaScript value to a Grafeo [`Value`], as the Node.js binding
/// does: a whole number below 2^53 is an `Int64` and any other number a
/// `Float64`; a `BigInt` follows the integer rule at any width (an `Int64`
/// when it fits 64 bits, else an exact `Decimal` up to 38 digits); an array
/// is a list, a `Date` a timestamp, a `Float32Array` a vector, a `Uint8Array`
/// bytes, an RDF/JS term (see [`rdfjs_term`]) that term, a `Map` or a plain
/// object a map, and a tagged object (`{ $decimal: "1.50" }`,
/// `{ $date: "2024-01-15" }`, ...) the value it stands for.
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
    // An RDF/JS term, as a SPARQL result gives it or another library makes
    // it, is that term.
    if let Some(term) = rdfjs_term(value)? {
        return Ok(Value::RdfTerm(Arc::new(term)));
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

/// `object[key]`, undefined when the object has no such field.
pub fn js_field(object: &JsValue, key: &str) -> JsValue {
    Reflect::get(object, &key.into()).unwrap_or(JsValue::UNDEFINED)
}

/// The properties of a JavaScript object (a plain object or a `Map`), each
/// converted by [`js_to_value`]; undefined or null has none.
///
/// # Errors
///
/// A value that is no object, or a property that has no Grafeo value; the
/// error names the property.
pub fn js_properties(value: &JsValue) -> Result<Vec<(String, Value)>, String> {
    if value.is_undefined() || value.is_null() {
        return Ok(Vec::new());
    }
    js_entries(value)
        .ok_or_else(|| "properties must be an object".to_string())??
        .into_iter()
        .map(|(key, item)| {
            let item = js_to_value(&item).map_err(|error| format!("property '{key}': {error}"))?;
            Ok((key, item))
        })
        .collect()
}

/// Rows of properties from an array of objects, each by [`js_properties`].
///
/// # Errors
///
/// A value that is no array, or a row that is no object or has a property
/// with no Grafeo value; the error names the row (`rows[2]: ...`).
pub fn js_rows(value: &JsValue) -> Result<Vec<Vec<(String, Value)>>, String> {
    if !Array::is_array(value) {
        return Err("rows must be an array of objects".to_string());
    }
    Array::from(value)
        .iter()
        .enumerate()
        .map(|(i, row)| {
            if row.is_undefined() || row.is_null() {
                return Err(format!("rows[{i}]: properties must be an object"));
            }
            js_properties(&row).map_err(|error| format!("rows[{i}]: {error}"))
        })
        .collect()
}

/// Converts a Grafeo [`Value`] to a JavaScript value.
///
/// # Errors
///
/// A JS `Error` for an RDF term of a kind this binding does not know, also
/// inside a list, a map or a path.
pub fn value_to_js(value: &Value) -> Result<JsValue, JsValue> {
    Ok(match value {
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
                arr.set(i as u32, value_to_js(item)?);
            }
            arr.into()
        }
        Value::Map(map) => {
            let obj = Object::new();
            for (key, val) in map.iter() {
                let _ = Reflect::set(&obj, &JsValue::from_str(key.as_str()), &value_to_js(val)?);
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
                nodes_arr.set(i as u32, value_to_js(node)?);
            }
            let edges_arr = Array::new_with_length(edges.len() as u32);
            for (i, edge) in edges.iter().enumerate() {
                edges_arr.set(i as u32, value_to_js(edge)?);
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
        Value::RdfTerm(term) => term_to_js(term)
            .map_err(|error| crate::database_error(grafeo_common::utils::Error::from(error)))?,
        _ => JsValue::from_str(&value.to_string()),
    })
}

/// The RDF term a JavaScript value of the RDF/JS data model stands for, read
/// by its `termType` (`NamedNode`, `BlankNode` or `Literal`) and `value`,
/// and for a literal its `language` and the `value` of its `datatype`.
/// Properties are read as JavaScript reads them, getters included, and other
/// keys are ignored: the terms of other libraries (N3.js, rdf-data-factory)
/// read too. `None` for a value of another shape.
///
/// # Errors
///
/// When reading a property throws.
pub fn rdfjs_term(value: &JsValue) -> Result<Option<Term>, String> {
    if !value.is_object() {
        return Ok(None);
    }
    let text = |object: &JsValue, key: &str| {
        Reflect::get(object, &JsValue::from_str(key))
            .map(|value| value.as_string())
            .map_err(|_| format!("reading '{key}' of an RDF term threw"))
    };
    let Some(term_type) = text(value, "termType")? else {
        return Ok(None);
    };
    if !matches!(term_type.as_str(), "NamedNode" | "BlankNode" | "Literal") {
        return Ok(None);
    }
    let Some(term_value) = text(value, "value")? else {
        return Ok(None);
    };
    let (language, datatype) = if term_type == "Literal" {
        let datatype = Reflect::get(value, &JsValue::from_str("datatype"))
            .map_err(|_| "reading 'datatype' of an RDF term threw".to_string())?;
        let datatype = if datatype.is_object() {
            text(&datatype, "value")?
        } else {
            None
        };
        (text(value, "language")?, datatype)
    } else {
        (None, None)
    };
    Ok(grafeo_bindings_common::json::rdfjs_term(
        &term_type,
        &term_value,
        language.as_deref(),
        datatype.as_deref(),
    ))
}

thread_local! {
    /// The function behind every term's `equals(other)`, made once: it gets
    /// the term itself as its first argument (see [`add_equals`]).
    static TERM_EQUALS: js_sys::Function = {
        let closure = Closure::<dyn Fn(JsValue, JsValue) -> bool>::new(
            |this: JsValue, other: JsValue| {
                matches!(
                    (rdfjs_term(&this), rdfjs_term(&other)),
                    (Ok(Some(own)), Ok(Some(other))) if own == other
                )
            },
        );
        let function = closure.as_ref().unchecked_ref::<js_sys::Function>().clone();
        // The function lives as long as the module: every term may call it.
        closure.forget();
        function
    };
}

/// Gives a term object the RDF/JS data model's `equals(other)`: true when
/// `other` is a term of the same `termType` and `value`, and for a literal
/// the same language (without case) and datatype; false for `null`,
/// `undefined` and any other value. The method is not enumerable, so the
/// object's keys, its JSON and a spread copy hold the term's fields only.
fn add_equals(obj: &Object) {
    let equals = TERM_EQUALS.with(|equals| equals.bind1(&JsValue::UNDEFINED, obj));
    let descriptor = Object::new();
    let _ = Reflect::set(&descriptor, &JsValue::from_str("value"), &equals);
    Object::define_property(obj, &JsValue::from_str("equals"), &descriptor);
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
/// [`UnsupportedTermKind`] for a term of a kind this binding does not know.
fn term_to_js(term: &Term) -> Result<JsValue, UnsupportedTermKind> {
    let object = |entries: &[(&str, JsValue)]| {
        let obj = Object::new();
        for (key, value) in entries {
            let _ = Reflect::set(&obj, &JsValue::from_str(key), value);
        }
        add_equals(&obj);
        JsValue::from(obj)
    };
    let named = |iri: &str| {
        object(&[
            ("termType", JsValue::from_str("NamedNode")),
            ("value", JsValue::from_str(iri)),
        ])
    };
    Ok(match term {
        Term::Iri(iri) => named(iri.as_str()),
        Term::BlankNode(node) => object(&[
            ("termType", JsValue::from_str("BlankNode")),
            ("value", JsValue::from_str(node.id())),
        ]),
        Term::Literal(literal) => object(&[
            ("termType", JsValue::from_str("Literal")),
            ("value", JsValue::from_str(literal.value())),
            (
                "language",
                JsValue::from_str(literal.language().unwrap_or("")),
            ),
            ("datatype", named(literal.datatype())),
        ]),
        other => return Err(UnsupportedTermKind::of(other)),
    })
}

/// Converts a row of values to a JavaScript object with column names as keys.
///
/// # Errors
///
/// As [`value_to_js`].
pub fn row_to_js_object(columns: &[String], row: &[Value]) -> Result<JsValue, JsValue> {
    let obj = Object::new();
    for (col, val) in columns.iter().zip(row.iter()) {
        let _ = Reflect::set(&obj, &JsValue::from_str(col), &value_to_js(val)?);
    }
    Ok(obj.into())
}
