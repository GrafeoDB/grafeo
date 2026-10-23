//! One-shot helper to regenerate the golden snapshot fixture.
//!
//! Run with: cargo test --all-features -p grafeo-engine --test golden_format -- regenerate_snapshot_fixture --ignored
//! Then commit the updated fixture file. This test is ignored by default.
//!
//! **When to regenerate:** only after an intentional snapshot format change
//! (i.e. bumping `SNAPSHOT_VERSION`). If the golden test in `golden_format.rs`
//! fails without a version bump, fix the regression instead of regenerating.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Builds the canonical fixture graph used by both the generator and the
/// golden format tests. Keep this in sync with the assertions in
/// `golden_format.rs`.
///
/// # Panics
///
/// Never: an in-memory database has the built-in store.
pub fn build_fixture_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    // Written straight to the store, all at epoch 0, so the export matches the
    // checked-in bytes: the database's write API commits each call and moves
    // the epoch, which the exported property versions record.
    let store = db.store().unwrap();

    // 3 nodes with different labels and properties
    let alix = store.create_node(&["Person"]);
    store.set_node_property(alix, "name", Value::String("Alix".into()));
    store.set_node_property(alix, "age", Value::Int64(30));

    let gus = store.create_node(&["Person", "Employee"]);
    store.set_node_property(gus, "name", Value::String("Gus".into()));
    store.set_node_property(gus, "age", Value::Int64(25));

    let acme = store.create_node(&["Company"]);
    store.set_node_property(acme, "name", Value::String("Acme Corp".into()));

    // 2 edges with properties
    let knows = store.create_edge(alix, gus, "KNOWS");
    store.set_edge_property(knows, "since", Value::Int64(2020));

    let works = store.create_edge(gus, acme, "WORKS_AT");
    store.set_edge_property(works, "role", Value::String("Engineer".into()));

    db
}

/// A property of every kind of value a snapshot stores, in the order the
/// fixture writes them: one value of each kind, and the edge cases a kind's
/// encoding has (an empty string, a negative zero, a decimal's scale, each
/// kind of RDF term).
///
/// # Panics
///
/// Never: every constant parses.
#[must_use]
pub fn every_value_kind() -> Vec<(&'static str, Value)> {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use grafeo_common::types::rdf::Term;
    use grafeo_common::types::{
        Date, Decimal, Duration, PropertyKey, Time, Timestamp, ZonedDatetime,
    };

    let counter = |entries: &[(&str, u64)]| -> Arc<HashMap<String, u64>> {
        Arc::new(
            entries
                .iter()
                .map(|(replica, count)| ((*replica).to_string(), *count))
                .collect(),
        )
    };
    let term = |term: Term| Value::RdfTerm(Arc::new(term));
    vec![
        ("bool", Value::Bool(true)),
        ("int", Value::Int64(-19)),
        ("int_max", Value::Int64(i64::MAX)),
        ("float", Value::Float64(1.5)),
        ("float_negative_zero", Value::Float64(-0.0)),
        ("string", Value::from("Amsterdam")),
        ("string_empty", Value::from("")),
        ("bytes", Value::Bytes(Arc::from(vec![3u8, 19, 88]))),
        ("date", Value::Date(Date::from_days(-3))),
        ("time", Value::Time(Time::from_nanos(88).unwrap())),
        (
            "time_with_offset",
            Value::Time(
                Time::from_nanos(3_600_000_000_019)
                    .unwrap()
                    .with_offset(3600),
            ),
        ),
        (
            "timestamp",
            Value::Timestamp(Timestamp::from_micros(1_696_500_000_123_457)),
        ),
        (
            "zoned",
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1_696_500_000_123_457),
                7200,
            )),
        ),
        ("duration", Value::Duration(Duration::new(3, 19, 88))),
        (
            "list",
            Value::List(Arc::from(vec![
                Value::Int64(3),
                Value::List(Arc::from(vec![Value::from("Berlin")])),
                Value::Null,
            ])),
        ),
        (
            "map",
            Value::Map(Arc::new(BTreeMap::from([
                (PropertyKey::new("city"), Value::from("Prague")),
                (PropertyKey::new("stops"), Value::Int64(19)),
            ]))),
        ),
        (
            "vector",
            Value::Vector(Arc::from(vec![3.0f32, -19.5, 88.0])),
        ),
        (
            "gcounter",
            Value::GCounter(counter(&[("Alix", 3), ("Gus", 19)])),
        ),
        (
            "on_counter",
            Value::OnCounter {
                pos: counter(&[("Mia", 88)]),
                neg: counter(&[("Jules", 3)]),
            },
        ),
        (
            "decimal",
            Value::Decimal(Decimal::parse("-12345.678").unwrap()),
        ),
        (
            "decimal_scale",
            Value::Decimal(Decimal::parse("1.50").unwrap()),
        ),
        (
            "decimal_widest",
            Value::Decimal(Decimal::parse("99999999999999999999999999999999999999").unwrap()),
        ),
        ("iri", term(Term::iri("http://example.org/apple"))),
        ("blank_node", term(Term::blank("b0"))),
        ("tagged", term(Term::lang_literal("Pomme", "fr"))),
        (
            "typed",
            term(Term::typed_literal(
                "019",
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        ),
    ]
}

/// The every-kind fixture's graph: a node with [`every_value_kind`], an edge
/// with a decimal, written straight to the store at epoch 0 as
/// [`build_fixture_db`] does, and RDF triples in the default graph and a
/// named one.
///
/// # Panics
///
/// Never: an in-memory database has the built-in store.
#[must_use]
pub fn build_every_value_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let store = db.store().unwrap();
    let kinds = store.create_node(&["Kinds"]);
    for (key, value) in every_value_kind() {
        store.set_node_property(kinds, key, value);
    }
    let other = store.create_node(&["Kinds", "Other"]);
    let next = store.create_edge(kinds, other, "NEXT");
    store.set_edge_property(
        next,
        "weight",
        Value::Decimal(grafeo_common::types::Decimal::parse("0.10").unwrap()),
    );
    #[cfg(feature = "sparql")]
    db.execute_sparql(
        "INSERT DATA { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> ; \
         <http://example.org/label> \"Alix\"@nl ; \
         <http://example.org/age> \"030\"^^<http://www.w3.org/2001/XMLSchema#integer> . \
         GRAPH <http://example.org/trips> { <http://example.org/gus> <http://example.org/visited> \
         <http://example.org/prague> } }",
    )
    .unwrap();
    db
}

#[test]
#[ignore = "one-shot fixture generator, not a regular test"]
fn regenerate_every_value_snapshot_fixture() {
    let db = build_every_value_db();
    let bytes = db.export_snapshot().unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v5_values.bin"
    );
    std::fs::write(path, &bytes).unwrap();
    println!("Wrote {} bytes to {path}", bytes.len());
}

#[test]
#[ignore = "one-shot fixture generator, not a regular test"]
fn regenerate_snapshot_fixture() {
    let db = build_fixture_db();
    let bytes = db.export_snapshot().unwrap();

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot_v5.bin"
    );
    std::fs::write(path, &bytes).unwrap();
    println!("Wrote {} bytes to {path}", bytes.len());
    println!("First byte (snapshot version): {}", bytes[0]);
}
