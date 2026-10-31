//! A JSON parameter object is a tagged value only when the tag is its only
//! key (0.6.1): `{"$date": "2026-10-18"}` is a date, but
//! `{"$date": "2026-10-18", "note": "x"}` is a map of both keys, as
//! `$decimal` and `$term` already were. The tagged object used to become the
//! value and lose its other keys, in every binding that reads JSON (Node.js,
//! WASM, C, Go, C# and Dart).
//!
//! ```bash
//! cargo test -p grafeo-bindings-common --test json_temporal_tags
//! ```

use grafeo_bindings_common::json::json_to_value;
use grafeo_common::types::{PropertyKey, Value};
use serde_json::json;

const TAGS: [(&str, &str); 4] = [
    ("$date", "2026-10-18"),
    ("$time", "14:30:00"),
    ("$duration", "P1D"),
    ("$zoned_datetime", "2026-10-18T14:30:00+02:00"),
];

#[test]
fn a_tag_alone_is_its_typed_value() {
    for (tag, text) in TAGS {
        let value = json_to_value(&json!({ tag: text })).unwrap();
        assert!(
            matches!(
                (tag, &value),
                ("$date", Value::Date(_))
                    | ("$time", Value::Time(_))
                    | ("$duration", Value::Duration(_))
                    | ("$zoned_datetime", Value::ZonedDatetime(_))
            ),
            "{tag}: {value:?}"
        );
    }
    let value = json_to_value(&json!({ "$timestamp_us": 1_700_000_000_000_000_i64 })).unwrap();
    assert!(matches!(value, Value::Timestamp(_)), "{value:?}");
}

#[test]
fn a_tag_beside_other_keys_is_a_map_of_all_its_keys() {
    for (tag, text) in TAGS {
        let value = json_to_value(&json!({ tag: text, "note": "x" })).unwrap();
        let Value::Map(map) = &value else {
            panic!("{tag} with another key became {value:?}");
        };
        assert_eq!(map.len(), 2, "{tag}: {map:?}");
        assert_eq!(map.get(&PropertyKey::new("note")), Some(&Value::from("x")));
        assert_eq!(
            map.get(&PropertyKey::new(tag)),
            Some(&Value::from(text)),
            "{tag}"
        );
    }
    let value =
        json_to_value(&json!({ "$timestamp_us": 1_700_000_000_000_000_i64, "note": "x" })).unwrap();
    let Value::Map(map) = &value else {
        panic!("$timestamp_us with another key became {value:?}");
    };
    assert_eq!(map.len(), 2);
}

#[test]
fn decimal_already_needs_to_be_the_only_key() {
    let alone = json_to_value(&json!({ "$decimal": "1.50" })).unwrap();
    assert!(matches!(alone, Value::Decimal(_)), "{alone:?}");
    let beside = json_to_value(&json!({ "$decimal": "1.50", "note": "x" })).unwrap();
    assert!(matches!(beside, Value::Map(_)), "{beside:?}");
}
