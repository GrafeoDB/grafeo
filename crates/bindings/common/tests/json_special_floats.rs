//! Special floats in the JSON form of values (0.6.1): JSON has no NaN or
//! infinities, so they travel as a tagged object, `{"$float": "NaN"}`,
//! `{"$float": "Infinity"}` and `{"$float": "-Infinity"}`, both ways; the
//! strings "NaN", "Infinity" and "-Infinity" stay strings (they used to become
//! floats). `{"$float": 3}` or `{"$float": "3"}` forces a float where a client
//! writes a whole float as an integer. As with `$decimal`, the tag counts
//! only as the map's only key, and a text that is no float leaves the map a
//! map.
//!
//! ```bash
//! cargo test -p grafeo-bindings-common --test json_special_floats
//! ```

use grafeo_bindings_common::json::{json_to_value, value_to_json};
use grafeo_common::types::Value;
use serde_json::json;

fn float(value: &Value) -> f64 {
    match value {
        Value::Float64(f) => *f,
        other => panic!("not a float: {other:?}"),
    }
}

#[test]
fn the_strings_nan_and_infinity_stay_strings() {
    for text in ["NaN", "Infinity", "-Infinity"] {
        assert_eq!(
            json_to_value(&json!(text)).unwrap(),
            Value::from(text),
            "{text}"
        );
    }
    // Inside a list and a map too.
    let list = json_to_value(&json!(["NaN", 1])).unwrap();
    assert_eq!(
        list,
        Value::List(vec![Value::from("NaN"), Value::Int64(1)].into())
    );
}

#[test]
fn a_float_tag_alone_is_its_float() {
    assert!(float(&json_to_value(&json!({ "$float": "NaN" })).unwrap()).is_nan());
    assert_eq!(
        float(&json_to_value(&json!({ "$float": "Infinity" })).unwrap()),
        f64::INFINITY
    );
    assert_eq!(
        float(&json_to_value(&json!({ "$float": "-Infinity" })).unwrap()),
        f64::NEG_INFINITY
    );
    // A whole number, as a number or as text, forced to a float.
    for forced in [
        json!({ "$float": 3 }),
        json!({ "$float": "3" }),
        json!({ "$float": 3.0 }),
    ] {
        assert_eq!(
            json_to_value(&forced).unwrap(),
            Value::Float64(3.0),
            "{forced}"
        );
    }
    assert_eq!(
        json_to_value(&json!({ "$float": "-2.5e3" })).unwrap(),
        Value::Float64(-2500.0)
    );
}

#[test]
fn a_float_tag_that_is_no_float_or_beside_other_keys_is_a_map() {
    // As `{"$decimal": "1.2.3"}`: the map stays a map.
    for bad in [
        json!({ "$float": "x" }),
        json!({ "$float": true }),
        json!({ "$float": null }),
    ] {
        let value = json_to_value(&bad).unwrap();
        assert!(
            matches!(&value, Value::Map(m) if m.len() == 1),
            "{bad}: {value:?}"
        );
    }
    let value = json_to_value(&json!({ "$float": "NaN", "note": 1 })).unwrap();
    assert!(matches!(&value, Value::Map(m) if m.len() == 2), "{value:?}");
}

#[test]
fn special_floats_are_written_as_float_tags() {
    assert_eq!(
        value_to_json(&Value::Float64(f64::NAN)).unwrap(),
        json!({ "$float": "NaN" })
    );
    assert_eq!(
        value_to_json(&Value::Float64(f64::INFINITY)).unwrap(),
        json!({ "$float": "Infinity" })
    );
    assert_eq!(
        value_to_json(&Value::Float64(f64::NEG_INFINITY)).unwrap(),
        json!({ "$float": "-Infinity" })
    );
    // A finite float stays a number; a whole one keeps its ".0".
    assert_eq!(
        value_to_json(&Value::Float64(1.5)).unwrap().to_string(),
        "1.5"
    );
    assert_eq!(
        value_to_json(&Value::Float64(3.0)).unwrap().to_string(),
        "3.0"
    );
    assert_eq!(
        value_to_json(&Value::Float64(-0.0)).unwrap().to_string(),
        "-0.0"
    );
}

#[test]
fn special_floats_round_trip() {
    for f in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 3.0, -0.5, 1e300] {
        let back = json_to_value(&value_to_json(&Value::Float64(f)).unwrap()).unwrap();
        let back = float(&back);
        assert!(
            (f.is_nan() && back.is_nan()) || back.to_bits() == f.to_bits(),
            "{f} came back as {back}"
        );
    }
    // A string "NaN" round-trips as a string.
    let back = json_to_value(&value_to_json(&Value::from("NaN")).unwrap()).unwrap();
    assert_eq!(back, Value::from("NaN"));
}
