//! The JSON Lines format that `grafeo data dump` writes and `grafeo data load`
//! reads.
//!
//! Each line holds one record, a node or an edge:
//!
//! ```text
//! {"type":"node","id":3,"labels":["Person"],"properties":{"name":{"String":"Alix"}}}
//! {"type":"edge","id":0,"source":3,"target":19,"edge_type":"KNOWS","properties":{"since":{"Int64":2019}}}
//! ```
//!
//! A node's `id` names the node in the file only: edges name their endpoints
//! by it, and a load gives every node a new ID. A node without an `id` loads,
//! but no edge can name it. An edge's `id` is not read.
//!
//! A property value is plain JSON or tagged with its type. Plain JSON reads
//! as a boolean, an `INT64` (an integer), a `FLOAT64` (a number with a
//! fraction or an exponent), a string, a list or a map; a null is no
//! property. A tagged value is an object with one key, the name of the type
//! as [`Value`]'s variants are named (`{"Int64": 3}`, `{"Date": 19000}`), and
//! is the form a dump writes, since plain JSON has no dates, durations,
//! bytes or vectors. Inside a tagged list, map or path every value is tagged
//! too, and a null is `"Null"`. JSON has no NaN or infinities: a dump writes
//! them as the strings `"NaN"`, `"Infinity"` and `"-Infinity"` under their
//! tag. A map whose only key is the name of a type reads as that type: write
//! such a map tagged (`{"Map": {"Date": {"String": "today"}}}`).

use std::collections::BTreeMap;
use std::sync::Arc;

use grafeo_common::types::{PropertyKey, Value};
use serde_json::{Map, Number, Value as Json};

/// One record of the file.
pub(super) enum Record {
    /// A node, with the ID edges name it by in the file, if it has one.
    Node {
        /// The node's ID in the file.
        id: Option<u64>,
        /// The node's labels.
        labels: Vec<String>,
        /// The node's properties.
        properties: Vec<(PropertyKey, Value)>,
    },
    /// An edge between the nodes the file names `source` and `target`.
    Edge {
        /// The file ID of the source node.
        source: u64,
        /// The file ID of the target node.
        target: u64,
        /// The edge's type.
        edge_type: String,
        /// The edge's properties.
        properties: Vec<(PropertyKey, Value)>,
    },
}

/// Reads the record on one line.
///
/// # Errors
///
/// Returns what is wrong with the record (without its line number).
pub(super) fn read_record(line: &str) -> Result<Record, String> {
    let json: Json = serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let Json::Object(mut record) = json else {
        return Err(format!("a record is a JSON object, not {}", kind(&json)));
    };
    match record.remove("type") {
        Some(Json::String(record_type)) if record_type == "node" => {
            let id = match record.remove("id") {
                None | Some(Json::Null) => None,
                Some(id) => Some(
                    id.as_u64()
                        .ok_or_else(|| format!("node id {id} is not a non-negative integer"))?,
                ),
            };
            Ok(Record::Node {
                id,
                labels: read_labels(record.remove("labels"))?,
                properties: read_properties(record.remove("properties"))?,
            })
        }
        Some(Json::String(record_type)) if record_type == "edge" => {
            let source = read_endpoint(record.remove("source"), "source")?;
            let target = read_endpoint(record.remove("target"), "target")?;
            let edge_type = match record.remove("edge_type") {
                Some(Json::String(edge_type)) => edge_type,
                Some(other) => return Err(format!("edge_type is {}, not text", kind(&other))),
                None => return Err("the edge has no edge_type".to_string()),
            };
            Ok(Record::Edge {
                source,
                target,
                edge_type,
                properties: read_properties(record.remove("properties"))?,
            })
        }
        Some(other) => Err(format!(
            "unknown record type {other}: expected \"node\" or \"edge\""
        )),
        None => Err("the record has no type: expected \"node\" or \"edge\"".to_string()),
    }
}

/// Reads a node's `labels`: a list of text, or nothing.
fn read_labels(labels: Option<Json>) -> Result<Vec<String>, String> {
    match labels {
        None | Some(Json::Null) => Ok(Vec::new()),
        Some(Json::Array(labels)) => labels
            .into_iter()
            .map(|label| match label {
                Json::String(label) => Ok(label),
                other => Err(format!("a label is text, not {}", kind(&other))),
            })
            .collect(),
        Some(other) => Err(format!("labels are a list of text, not {}", kind(&other))),
    }
}

/// Reads an edge's `source` or `target`: a node ID of the file.
fn read_endpoint(endpoint: Option<Json>, role: &str) -> Result<u64, String> {
    match endpoint {
        Some(endpoint) => endpoint.as_u64().ok_or_else(|| {
            format!("edge {role} {endpoint} is not a node id (a non-negative integer)")
        }),
        None => Err(format!("the edge has no {role}")),
    }
}

/// Reads `properties`: an object, or nothing.
fn read_properties(properties: Option<Json>) -> Result<Vec<(PropertyKey, Value)>, String> {
    match properties {
        None | Some(Json::Null) => Ok(Vec::new()),
        Some(Json::Object(properties)) => properties
            .into_iter()
            .map(|(key, value)| match read_value(value) {
                Ok(value) => Ok((PropertyKey::new(key), value)),
                Err(reason) => Err(format!("property {key:?}: {reason}")),
            })
            .collect(),
        Some(other) => Err(format!(
            "properties are a JSON object, not {}",
            kind(&other)
        )),
    }
}

/// The names of [`Value`]'s variants that tag a value (the names serde gives
/// them). A null is not among them: tagged, it is the string `"Null"`.
const TAGS: [&str; 16] = [
    "Bool",
    "Int64",
    "Float64",
    "String",
    "Bytes",
    "Timestamp",
    "Date",
    "Time",
    "Duration",
    "ZonedDatetime",
    "List",
    "Map",
    "Vector",
    "Path",
    "GCounter",
    "OnCounter",
];

/// Splits a tagged value into its tag and payload, or gives the object back
/// when it is not tagged.
fn split_tag(object: Map<String, Json>) -> Result<(String, Json), Map<String, Json>> {
    let is_tagged = object.len() == 1 && object.keys().all(|key| TAGS.contains(&key.as_str()));
    if !is_tagged {
        return Err(object);
    }
    object.into_iter().next().ok_or_else(Map::new)
}

/// Reads a property value: plain JSON or tagged (see the module docs).
fn read_value(json: Json) -> Result<Value, String> {
    match json {
        Json::Null => Ok(Value::Null),
        Json::Bool(value) => Ok(Value::Bool(value)),
        Json::Number(number) => read_number(&number),
        Json::String(text) => Ok(Value::String(text.into())),
        Json::Array(items) => items
            .into_iter()
            .map(read_value)
            .collect::<Result<Vec<_>, _>>()
            .map(|items| Value::List(items.into())),
        Json::Object(object) => match split_tag(object) {
            Ok((tag, payload)) => read_tagged(&tag, payload),
            Err(entries) => entries
                .into_iter()
                .map(|(key, value)| Ok((PropertyKey::new(key), read_value(value)?)))
                .collect::<Result<BTreeMap<_, _>, String>>()
                .map(|entries| Value::Map(Arc::new(entries))),
        },
    }
}

/// Reads a plain JSON number: an `INT64` when it is an integer, a `FLOAT64`
/// when it has a fraction or an exponent.
fn read_number(number: &Number) -> Result<Value, String> {
    if let Some(integer) = number.as_i64() {
        Ok(Value::Int64(integer))
    } else if number.is_u64() {
        Err(format!("{number} is beyond the range of INT64"))
    } else {
        number
            .as_f64()
            .map(Value::Float64)
            .ok_or_else(|| format!("{number} is not a number"))
    }
}

/// Reads a value inside a tagged list, map or path: tagged as well, with
/// `"Null"` (or a JSON null) for a null.
fn read_tagged_value(json: Json) -> Result<Value, String> {
    match json {
        Json::Null => Ok(Value::Null),
        Json::String(text) if text == "Null" => Ok(Value::Null),
        Json::Object(object) => match split_tag(object) {
            Ok((tag, payload)) => read_tagged(&tag, payload),
            Err(_) => Err("an object without a type tag inside a tagged value".to_string()),
        },
        other => Err(format!(
            "{} inside a tagged value, where a tagged value such as {{\"Int64\": 3}} belongs",
            kind(&other)
        )),
    }
}

/// Reads the payload of a value tagged `tag`.
fn read_tagged(tag: &str, payload: Json) -> Result<Value, String> {
    match (tag, payload) {
        ("Float64", payload) => read_float(payload).map(Value::Float64),
        ("List", Json::Array(items)) => items
            .into_iter()
            .map(read_tagged_value)
            .collect::<Result<Vec<_>, _>>()
            .map(|items| Value::List(items.into())),
        ("Map", Json::Object(entries)) => entries
            .into_iter()
            .map(|(key, value)| Ok((PropertyKey::new(key), read_tagged_value(value)?)))
            .collect::<Result<BTreeMap<_, _>, String>>()
            .map(|entries| Value::Map(Arc::new(entries))),
        ("Vector", Json::Array(items)) => items
            .into_iter()
            .map(read_float32)
            .collect::<Result<Vec<_>, _>>()
            .map(|items| Value::Vector(items.into())),
        ("Path", Json::Object(mut path)) => {
            let mut part = |name: &str| match path.remove(name) {
                Some(Json::Array(items)) => items
                    .into_iter()
                    .map(read_tagged_value)
                    .collect::<Result<Vec<_>, _>>(),
                _ => Err(format!("a Path holds a list of {name}")),
            };
            let nodes = part("nodes")?;
            let edges = part("edges")?;
            Ok(Value::Path {
                nodes: nodes.into(),
                edges: edges.into(),
            })
        }
        ("List" | "Map" | "Vector" | "Path", other) => {
            Err(format!("a tagged {tag} does not hold {}", kind(&other)))
        }
        // The other types read as serde writes them.
        (tag, payload) => {
            let tagged = Json::Object(Map::from_iter([(tag.to_string(), payload)]));
            serde_json::from_value::<Value>(tagged)
                .map_err(|e| format!("not a valid {tag} value: {e}"))
        }
    }
}

/// The strings a dump writes for the floats JSON has no number for.
const NAN: &str = "NaN";
const INFINITY: &str = "Infinity";
const NEG_INFINITY: &str = "-Infinity";

/// Reads a tagged `FLOAT64`: a number, or one of the strings for NaN and the
/// infinities.
fn read_float(json: Json) -> Result<f64, String> {
    match json {
        Json::Number(number) => number
            .as_f64()
            .ok_or_else(|| format!("{number} is not a number")),
        Json::String(text) => match text.as_str() {
            NAN => Ok(f64::NAN),
            INFINITY => Ok(f64::INFINITY),
            NEG_INFINITY => Ok(f64::NEG_INFINITY),
            _ => Err(format!("{text:?} is not a FLOAT64")),
        },
        Json::Null => Err("a FLOAT64 that is null: a dump before 0.6.0 wrote NaN and \
             the infinities as null, and which one it was is lost"
            .to_string()),
        other => Err(format!("a FLOAT64 is a number, not {}", kind(&other))),
    }
}

/// Reads an element of a tagged vector: a number within the range of a
/// 32-bit float, or one of the strings for NaN and the infinities.
fn read_float32(json: Json) -> Result<f32, String> {
    match json {
        Json::String(text) => match text.as_str() {
            NAN => Ok(f32::NAN),
            INFINITY => Ok(f32::INFINITY),
            NEG_INFINITY => Ok(f32::NEG_INFINITY),
            _ => Err(format!("{text:?} is not a vector element")),
        },
        Json::Number(number) => {
            let narrow: f32 = serde_json::from_value(Json::Number(number.clone()))
                .map_err(|e| format!("{number} is not a vector element: {e}"))?;
            if narrow.is_infinite() {
                return Err(format!("{number} is beyond the range of a vector element"));
            }
            Ok(narrow)
        }
        other => Err(format!(
            "a vector element is a number, not {}",
            kind(&other)
        )),
    }
}

/// Writes the properties of a node or edge as a dump stores them, each value
/// as [`write_value`] writes it.
///
/// # Errors
///
/// Returns an error naming the property whose value cannot be written.
pub(super) fn write_properties(properties: &BTreeMap<PropertyKey, Value>) -> Result<Json, String> {
    properties
        .iter()
        .map(|(key, value)| match write_value(value) {
            Ok(value) => Ok((key.as_str().to_string(), value)),
            Err(reason) => Err(format!("property {:?}: {reason}", key.as_str())),
        })
        .collect::<Result<Map<_, _>, _>>()
        .map(Json::Object)
}

/// Writes a value as a dump stores it: tagged, as serde writes [`Value`],
/// except that NaN and the infinities, for which JSON has no number, are
/// written as strings under their tag (serde would write a null and lose the
/// value).
///
/// # Errors
///
/// Returns an error for a type this format does not know yet, rather than
/// writing something a load would read as another type.
pub(super) fn write_value(value: &Value) -> Result<Json, String> {
    Ok(match value {
        Value::Float64(float) => tag("Float64", write_float(*float)),
        Value::List(items) => tag(
            "List",
            Json::Array(items.iter().map(write_value).collect::<Result<_, _>>()?),
        ),
        Value::Map(entries) => tag(
            "Map",
            Json::Object(
                entries
                    .iter()
                    .map(|(key, value)| Ok((key.as_str().to_string(), write_value(value)?)))
                    .collect::<Result<_, String>>()?,
            ),
        ),
        Value::Vector(items) => tag(
            "Vector",
            Json::Array(items.iter().map(|item| write_float32(*item)).collect()),
        ),
        Value::Path { nodes, edges } => {
            let nodes = nodes.iter().map(write_value).collect::<Result<_, _>>()?;
            let edges = edges.iter().map(write_value).collect::<Result<_, _>>()?;
            tag(
                "Path",
                Json::Object(Map::from_iter([
                    ("nodes".to_string(), Json::Array(nodes)),
                    ("edges".to_string(), Json::Array(edges)),
                ])),
            )
        }
        Value::Null
        | Value::Bool(_)
        | Value::Int64(_)
        | Value::String(_)
        | Value::Bytes(_)
        | Value::Timestamp(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Duration(_)
        | Value::ZonedDatetime(_)
        | Value::GCounter(_)
        | Value::OnCounter { .. } => {
            serde_json::to_value(value).map_err(|e| format!("cannot write the value: {e}"))?
        }
        other => {
            return Err(format!(
                "data dump cannot write a {} value yet",
                other.type_name()
            ));
        }
    })
}

/// A value tagged `name`.
fn tag(name: &str, payload: Json) -> Json {
    Json::Object(Map::from_iter([(name.to_string(), payload)]))
}

/// A `FLOAT64` as a JSON number, or as a string when JSON has no number for
/// it.
fn write_float(float: f64) -> Json {
    Number::from_f64(float)
        .map_or_else(|| Json::String(non_finite(float).to_string()), Json::Number)
}

/// A vector element as a JSON number, or as a string when JSON has no number
/// for it.
fn write_float32(float: f32) -> Json {
    if float.is_finite() {
        Json::from(float)
    } else {
        Json::String(non_finite(f64::from(float)).to_string())
    }
}

/// The string for a float that is NaN or infinite.
fn non_finite(float: f64) -> &'static str {
    if float.is_nan() {
        NAN
    } else if float.is_sign_positive() {
        INFINITY
    } else {
        NEG_INFINITY
    }
}

/// What a JSON value is, for messages.
fn kind(json: &Json) -> &'static str {
    match json {
        Json::Null => "null",
        Json::Bool(_) => "a boolean",
        Json::Number(_) => "a number",
        Json::String(_) => "text",
        Json::Array(_) => "a list",
        Json::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};
    use serde_json::json;

    use super::*;

    fn list(items: Vec<Value>) -> Value {
        Value::List(items.into())
    }

    fn map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(Arc::new(
            entries
                .into_iter()
                .map(|(key, value)| (PropertyKey::new(key), value))
                .collect(),
        ))
    }

    fn counts(entries: &[(&str, u64)]) -> Arc<HashMap<String, u64>> {
        Arc::new(
            entries
                .iter()
                .map(|(replica, count)| ((*replica).to_string(), *count))
                .collect(),
        )
    }

    /// A value of every type `Value` has, with nulls and floats JSON has no
    /// number for inside containers (NaN is checked on its own, as it equals
    /// nothing).
    fn every_value() -> Vec<Value> {
        let node = map(vec![("name", Value::from("Alix"))]);
        vec![
            Value::Bool(false),
            Value::Int64(-3),
            Value::Int64(i64::MAX),
            Value::Float64(19.88),
            Value::Float64(3.0),
            Value::Float64(-0.0),
            Value::Float64(f64::INFINITY),
            Value::Float64(f64::NEG_INFINITY),
            Value::Float64(f64::MIN_POSITIVE),
            Value::from("Mia \"Butch\" \\ \n caf\u{e9}"),
            Value::from("Null"),
            Value::from(""),
            Value::Bytes(vec![0_u8, 3, 19, 88, 255].into()),
            Value::Timestamp(Timestamp::from_micros(-19)),
            Value::Date(Date::parse("1988-03-19").expect("a date")),
            Value::Time(Time::parse("23:19:03.088").expect("a time")),
            Value::Duration(Duration::parse("P1Y3M19DT8H").expect("a duration")),
            Value::ZonedDatetime(
                ZonedDatetime::parse("2026-10-09T08:19:03-03:00").expect("a datetime"),
            ),
            list(vec![]),
            list(vec![
                Value::Null,
                Value::Int64(3),
                Value::Float64(f64::INFINITY),
                list(vec![Value::from("nested"), Value::Null]),
                map(vec![("Int64", Value::from("a key named like a type"))]),
            ]),
            map(vec![]),
            map(vec![
                ("Date", Value::from("a key named like a type")),
                ("none", Value::Null),
                ("far", Value::Float64(f64::NEG_INFINITY)),
            ]),
            Value::Vector(vec![0.1_f32, -19.88, f32::MAX, f32::MIN_POSITIVE].into()),
            Value::Vector(vec![f32::INFINITY, f32::NEG_INFINITY].into()),
            Value::Path {
                nodes: vec![node.clone(), node].into(),
                edges: vec![map(vec![("since", Value::Int64(2019))])].into(),
            },
            Value::GCounter(counts(&[("amsterdam", 3), ("berlin", 19)])),
            Value::OnCounter {
                pos: counts(&[("paris", 88)]),
                neg: counts(&[("prague", 3)]),
            },
        ]
    }

    /// Reads back what [`write_value`] wrote, through the text of a line.
    fn round_trip(value: &Value) -> Value {
        let text = serde_json::to_string(&write_value(value).expect("write")).unwrap();
        read_value(serde_json::from_str(&text).unwrap())
            .unwrap_or_else(|reason| panic!("{text} does not read back: {reason}"))
    }

    #[test]
    fn every_value_reads_back_as_written() {
        for value in every_value() {
            assert_eq!(round_trip(&value), value);
        }
    }

    #[test]
    fn nan_reads_back_as_nan() {
        let Value::Float64(float) = round_trip(&Value::Float64(f64::NAN)) else {
            panic!("not a FLOAT64");
        };
        assert!(float.is_nan());
        let Value::Vector(vector) = round_trip(&Value::Vector(vec![f32::NAN].into())) else {
            panic!("not a VECTOR");
        };
        assert!(vector[0].is_nan());
    }

    /// A dump writes what 0.5.44 wrote, so files written by both read alike;
    /// only the floats JSON has no number for differ.
    #[test]
    fn a_dump_writes_values_as_0_5_44_did_except_nan_and_infinities() {
        for (value, written) in [
            (Value::Int64(3), json!({"Int64": 3})),
            (Value::from("Gus"), json!({"String": "Gus"})),
            (Value::Float64(1.5), json!({"Float64": 1.5})),
            (
                list(vec![Value::Null, Value::Bool(true)]),
                json!({"List": ["Null", {"Bool": true}]}),
            ),
            (
                map(vec![("zip", Value::Int64(19))]),
                json!({"Map": {"zip": {"Int64": 19}}}),
            ),
            (
                Value::Vector(vec![3.0_f32].into()),
                json!({"Vector": [3.0]}),
            ),
            (Value::Float64(f64::NAN), json!({"Float64": "NaN"})),
            (
                Value::Float64(f64::NEG_INFINITY),
                json!({"Float64": "-Infinity"}),
            ),
            (
                Value::Vector(vec![f32::INFINITY].into()),
                json!({"Vector": ["Infinity"]}),
            ),
        ] {
            assert_eq!(write_value(&value).unwrap(), written, "{value:?}");
        }
        // 0.5.44 wrote every value as serde does, which writes a null for
        // each float JSON has no number for.
        for value in every_value() {
            let serde = serde_json::to_value(&value).unwrap();
            if !serde.to_string().contains("null") {
                assert_eq!(write_value(&value).unwrap(), serde, "{value:?}");
            }
        }
    }

    /// Plain JSON reads by its JSON type: an integer is an INT64, a number
    /// with a fraction or an exponent a FLOAT64, an object a map unless it is
    /// a single key naming a type.
    #[test]
    fn plain_json_reads_by_its_json_type() {
        for (json, value) in [
            (json!(3), Value::Int64(3)),
            (json!(-19), Value::Int64(-19)),
            (json!(3.0), Value::Float64(3.0)),
            (json!(1e3), Value::Float64(1000.0)),
            (json!("Amsterdam"), Value::from("Amsterdam")),
            (json!("Null"), Value::from("Null")),
            (json!(true), Value::Bool(true)),
            (json!(null), Value::Null),
            (
                json!([1, "a", null, [2.5]]),
                list(vec![
                    Value::Int64(1),
                    Value::from("a"),
                    Value::Null,
                    list(vec![Value::Float64(2.5)]),
                ]),
            ),
            (
                json!({"city": "Berlin", "zip": {"Int64": 19}}),
                map(vec![
                    ("city", Value::from("Berlin")),
                    ("zip", Value::Int64(19)),
                ]),
            ),
            (
                json!({"Int64": 3, "Bool": true}),
                map(vec![
                    ("Bool", Value::Bool(true)),
                    ("Int64", Value::Int64(3)),
                ]),
            ),
            (
                json!({"Integer": 3}),
                map(vec![("Integer", Value::Int64(3))]),
            ),
            (json!({"String": "Paris"}), Value::from("Paris")),
            (
                json!({"Date": 6652}),
                Value::Date(Date::parse("1988-03-19").expect("a date")),
            ),
        ] {
            assert_eq!(read_value(json.clone()), Ok(value), "{json}");
        }
    }

    /// A value that does not read is an error, never a null or a string.
    #[test]
    fn a_tagged_value_that_does_not_read_is_an_error() {
        for json in [
            json!(18_446_744_073_709_551_615_u64),
            json!({"Int64": 1.5}),
            json!({"Int64": "3"}),
            json!({"Bool": 1}),
            json!({"String": 3}),
            json!({"Float64": null}),
            json!({"Float64": "nan"}),
            json!({"Float64": [1.0]}),
            json!({"List": {"a": 1}}),
            json!({"List": [3]}),
            json!({"List": [{"name": "Gus"}]}),
            json!({"Map": {"zip": 19}}),
            json!({"Map": [1]}),
            json!({"Vector": [1e39]}),
            json!({"Vector": ["x"]}),
            json!({"Vector": 3}),
            json!({"Path": {"nodes": []}}),
            json!({"Path": {"nodes": [], "edges": [3]}}),
            json!({"Date": "someday"}),
            json!({"Bytes": [256]}),
            json!({"GCounter": {"amsterdam": -3}}),
            json!([{"Int64": "x"}]),
            json!({"address": {"Int64": "x"}}),
        ] {
            assert!(read_value(json.clone()).is_err(), "{json} read");
        }
    }

    #[test]
    fn a_record_reads_its_fields_and_ignores_others() {
        let Ok(Record::Node {
            id,
            labels,
            properties,
        }) = read_record(r#"{"type":"node","labels":null,"properties":null,"note":"extra"}"#)
        else {
            panic!("not a node");
        };
        assert_eq!((id, labels.len(), properties.len()), (None, 0, 0));

        let Ok(Record::Edge {
            source,
            target,
            edge_type,
            properties,
        }) = read_record(
            r#"{"type":"edge","id":"not read","source":3,"target":19,"edge_type":"","properties":{"since":2019}}"#,
        )
        else {
            panic!("not an edge");
        };
        assert_eq!((source, target, edge_type.as_str()), (3, 19, ""));
        assert_eq!(
            properties,
            vec![(PropertyKey::new("since"), Value::Int64(2019))]
        );

        for (line, reason) in [
            ("[]", "a record is a JSON object"),
            (r#"{"type":3}"#, "unknown record type 3"),
            (r#"{"type":"node","id":1.5}"#, "node id 1.5"),
            (
                r#"{"type":"edge","source":3,"target":19,"edge_type":3}"#,
                "edge_type is a number",
            ),
            (
                r#"{"type":"node","properties":{"age":{"Int64":"x"}}}"#,
                "property \"age\"",
            ),
        ] {
            let Err(error) = read_record(line) else {
                panic!("{line} read");
            };
            assert!(error.contains(reason), "{line}: {error}");
        }
    }

    /// A type the format does not know yet is an error when it is written,
    /// not a map a load reads back: the tags it reads are the variants serde
    /// writes.
    #[test]
    fn the_tags_are_the_variants_serde_writes() {
        for value in every_value() {
            let serde = serde_json::to_value(&value).unwrap();
            if let Json::Object(object) = serde {
                let tag = object.keys().next().expect("a tag");
                assert!(TAGS.contains(&tag.as_str()), "{tag} is not a known tag");
            }
        }
        assert_eq!(serde_json::to_value(Value::Null).unwrap(), json!("Null"));
    }
}
