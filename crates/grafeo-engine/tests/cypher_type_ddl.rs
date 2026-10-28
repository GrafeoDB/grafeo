//! Cypher inline type DDL reads the property types supported by GQL.

#![cfg(feature = "cypher")]

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod common;

use std::collections::HashMap;

use grafeo_common::types::{Decimal, Timestamp, Value, ZonedDatetime};
use grafeo_engine::GrafeoDB;

fn properties(db: &GrafeoDB, kind: &str, name: &str) -> String {
    let query = format!("SHOW {kind} TYPES");
    let result = db.execute(&query).unwrap();
    let matching: Vec<_> = result
        .rows()
        .iter()
        .filter(|row| row[0] == Value::from(name))
        .collect();
    assert_eq!(matching.len(), 1, "{query}: {name}: {:?}", result.rows());
    matching[0][1].as_str().unwrap().to_owned()
}

fn catalog(db: &GrafeoDB) -> Vec<String> {
    let mut rows = Vec::new();
    for query in ["SHOW NODE TYPES", "SHOW EDGE TYPES", "SHOW GRAPH TYPES"] {
        for row in db.execute(query).unwrap().rows() {
            rows.push(format!("{query}: {row:?}"));
        }
    }
    rows.sort();
    rows
}

#[test]
fn temporal_property_names_are_read_by_cypher_type_ddl() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "ALTER CURRENT GRAPH TYPE SET {
            (:Event {begins ZONED DATETIME, departs :: LOCAL DATETIME}),
            (:Event)-[:BOOKED {at :: zoned datetime, seen local datetime}]->(:Event)
        }",
    )
    .unwrap();
    assert_eq!(
        properties(&db, "NODE", "Event"),
        "begins ZONED DATETIME, departs LOCAL DATETIME"
    );
    assert_eq!(
        properties(&db, "EDGE", "BOOKED"),
        "at ZONED DATETIME, seen LOCAL DATETIME"
    );
}

#[test]
fn typed_lists_are_read_by_cypher_type_ddl() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "ALTER CURRENT GRAPH TYPE SET {
            (:Event {tags LIST<STRING>, scores :: list<list<int64>> NOT NULL}),
            (:Event)-[:BOOKED {stamps :: LIST<LOCAL DATETIME>,
                              fees LIST<DECIMAL(6,2)>}]->(:Event)
        }",
    )
    .unwrap();
    assert_eq!(
        properties(&db, "NODE", "Event"),
        "tags LIST<STRING>, scores LIST<LIST<INT64>> NOT NULL"
    );
    assert_eq!(
        properties(&db, "EDGE", "BOOKED"),
        "stamps LIST<LOCAL DATETIME>, fees LIST<DECIMAL(6,2)>"
    );
}

#[test]
fn aliases_quoted_names_nullability_and_decimal_precision_are_preserved() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "ALTER CURRENT GRAPH TYPE SET {
            (:Price {name :: `STRING` NOT NULL, at `ZONED DATETIME`,
                     local LOCAL_DATETIME, legacy TIMESTAMP, tags LIST, raw ARRAY,
                     amount DECIMAL(6,2), rate :: DEC(5,1), n INTEGER})
        }",
    )
    .unwrap();
    assert_eq!(
        properties(&db, "NODE", "Price"),
        "name STRING NOT NULL, at ZONED DATETIME, local LOCAL DATETIME, \
         legacy TIMESTAMP, tags LIST, raw LIST, amount DECIMAL(6,2), \
         rate DECIMAL(5,1), n INT64"
    );
    let params = HashMap::from([
        (
            "amount".to_owned(),
            Value::Decimal(Decimal::new(1234, 2).unwrap()),
        ),
        (
            "rate".to_owned(),
            Value::Decimal(Decimal::new(23, 1).unwrap()),
        ),
    ]);
    db.execute_cypher_with_params(
        "CREATE (:Price {name: 'ticket', amount: $amount, rate: $rate})",
        params,
    )
    .unwrap();
    assert!(
        db.execute_cypher("CREATE (:Price {amount: 1})").is_err(),
        "the existing NOT NULL rule still applies"
    );
    assert!(
        db.execute_cypher("CREATE (:Price {name: 'too wide', amount: 10000})")
            .is_err(),
        "DECIMAL(6, 2) still rejects an out-of-range integer"
    );
    assert_eq!(
        db.execute_cypher("MATCH (p:Price) RETURN p.name, p.amount, p.rate")
            .unwrap()
            .rows(),
        [vec![
            Value::from("ticket"),
            Value::Decimal(Decimal::new(1234, 2).unwrap()),
            Value::Decimal(Decimal::new(23, 1).unwrap()),
        ]]
    );
}

const EVENT_TYPES: &str = "ALTER CURRENT GRAPH TYPE SET {
    (:Event {begins ZONED DATETIME NOT NULL, departs :: LOCAL DATETIME,
             tags LIST<STRING>, stamps :: LIST<ZONED DATETIME>}),
    (:Event)-[:BOOKED {times LIST<LOCAL DATETIME> NOT NULL}]->(:Event)
}";

fn zoned() -> Value {
    Value::ZonedDatetime(ZonedDatetime::parse("2026-10-05T10:30:00+02:00").unwrap())
}

fn local() -> Value {
    Value::Timestamp(Timestamp::from_secs(1_791_000_000))
}

fn list(values: Vec<Value>) -> Value {
    Value::List(values.into())
}

fn valid_event_params() -> HashMap<String, Value> {
    HashMap::from([
        ("begins".to_owned(), zoned()),
        ("departs".to_owned(), local()),
        ("tags".to_owned(), list(vec![Value::from("fast")])),
        ("stamps".to_owned(), list(vec![zoned()])),
    ])
}

const CREATE_EVENT: &str = "CREATE (:Event {
    begins: $begins, departs: $departs, tags: $tags, stamps: $stamps})";

fn insert_valid_values(db: &GrafeoDB) {
    db.execute_cypher_with_params(CREATE_EVENT, valid_event_params())
        .unwrap();
    db.execute_cypher_with_params(
        "MATCH (e:Event) CREATE (e)-[:BOOKED {times: $times}]->(e)",
        HashMap::from([("times".to_owned(), list(vec![local()]))]),
    )
    .unwrap();
}

fn assert_values_and_rejected_writes(db: &GrafeoDB) {
    for (name, value) in [
        ("begins", Value::from("not a datetime")),
        ("begins", local()),
        ("begins", Value::Null),
        ("departs", zoned()),
        ("tags", list(vec![Value::Int64(3)])),
        ("stamps", list(vec![zoned(), local()])),
    ] {
        let mut params = valid_event_params();
        params.insert(name.to_owned(), value);
        let error = db
            .execute_cypher_with_params(CREATE_EVENT, params)
            .expect_err("a value of the wrong declared type must be refused")
            .to_string();
        assert!(error.contains(name), "{name}: {error}");
    }
    for value in [list(vec![zoned()]), Value::Null] {
        let error = db
            .execute_cypher_with_params(
                "MATCH (e:Event) CREATE (e)-[:BOOKED {times: $times}]->(e)",
                HashMap::from([("times".to_owned(), value)]),
            )
            .expect_err("an edge value of the wrong type must be refused")
            .to_string();
        assert!(error.contains("times"), "{error}");
    }
    assert_eq!(
        db.execute_cypher("MATCH (e:Event) RETURN count(e)")
            .unwrap()
            .rows(),
        [vec![Value::Int64(1)]],
        "failed node writes add no rows"
    );
    assert_eq!(
        db.execute_cypher(
            "MATCH (e:Event)-[b:BOOKED]->(e)
             RETURN e.begins, e.departs, e.tags, e.stamps, b.times"
        )
        .unwrap()
        .rows(),
        [vec![
            zoned(),
            local(),
            list(vec![Value::from("fast")]),
            list(vec![zoned()]),
            list(vec![local()]),
        ]],
        "the one valid node and edge retain their exact typed values"
    );
}

#[test]
fn node_and_edge_writes_validate_compound_types() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(EVENT_TYPES).unwrap();
    insert_valid_values(&db);
    assert_values_and_rejected_writes(&db);
}

#[test]
fn add_and_drop_change_membership_of_existing_typed_definitions() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(EVENT_TYPES).unwrap();
    let node = properties(&db, "NODE", "Event");
    let edge = properties(&db, "EDGE", "BOOKED");
    db.execute_cypher(
        "ALTER CURRENT GRAPH TYPE DROP {
            (:Event {begins :: ZONED DATETIME}),
            (:Event)-[:BOOKED {times LIST<LOCAL DATETIME>}]->(:Event)
        }",
    )
    .unwrap();
    assert_eq!(
        db.execute("SHOW GRAPH TYPE __current__").unwrap().rows(),
        [vec![
            Value::from("__current__"),
            Value::Bool(false),
            Value::from(""),
            Value::from(""),
        ]]
    );
    assert_eq!(properties(&db, "NODE", "Event"), node);
    assert_eq!(properties(&db, "EDGE", "BOOKED"), edge);
    db.execute_cypher(
        "ALTER CURRENT GRAPH TYPE ADD {
            (:Event {begins ZONED DATETIME}),
            (:Event)-[:BOOKED {times :: LIST<LOCAL DATETIME>}]->(:Event)
        }",
    )
    .unwrap();
    assert_eq!(
        db.execute("SHOW GRAPH TYPE __current__").unwrap().rows(),
        [vec![
            Value::from("__current__"),
            Value::Bool(false),
            Value::from("Event"),
            Value::from("BOOKED"),
        ]]
    );
    assert_eq!(properties(&db, "NODE", "Event"), node);
    assert_eq!(properties(&db, "EDGE", "BOOKED"), edge);
    insert_valid_values(&db);
    assert_values_and_rejected_writes(&db);
}

#[test]
fn malformed_and_overdeep_types_leave_the_catalog_unchanged() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:Kept {name STRING}) }")
        .unwrap();
    let before = catalog(&db);
    for type_name in [
        "ZONED",
        "LOCAL STRING",
        "LIST<>",
        "LIST<STRING",
        "LIST<STRING>>",
        "LIST<UNKNOWN_TYPE>",
        "LIST<DECIMAL(0,0)>",
        "`LIST<STRING>`",
    ] {
        let error = db
            .execute_cypher(&format!(
                "ALTER CURRENT GRAPH TYPE SET {{ (:Bad {{a {type_name}}}) }}"
            ))
            .expect_err("a malformed type must not change the catalog");
        assert_eq!(catalog(&db), before, "{type_name}: {error}");
    }
    let nested = |depth: usize| format!("{}INT64{}", "LIST<".repeat(depth), ">".repeat(depth));
    db.execute_cypher(&format!(
        "ALTER CURRENT GRAPH TYPE SET {{ (:Deep {{a {}}}) }}",
        nested(128)
    ))
    .unwrap();
    assert_eq!(
        properties(&db, "NODE", "Deep"),
        format!("a {}", nested(128))
    );
    let before = catalog(&db);
    for depth in [129, 100_000] {
        let error = db
            .execute_cypher(&format!(
                "ALTER CURRENT GRAPH TYPE SET {{ (:TooDeep {{a {}}}) }}",
                nested(depth)
            ))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("A property type nests at most 128 LIST<...> levels"),
            "depth {depth}: {error}"
        );
        assert_eq!(catalog(&db), before, "depth {depth}");
    }
}

#[test]
fn node_and_edge_types_enforce_the_total_list_level_budget() {
    let db = GrafeoDB::new_in_memory();
    let nested = format!("{}INT64{}", "LIST<".repeat(128), ">".repeat(128));
    let at_limit = (0..256)
        .map(|n| format!("p{n} {nested}"))
        .collect::<Vec<_>>()
        .join(", ");
    db.execute_cypher(&format!(
        "ALTER CURRENT GRAPH TYPE SET {{
            (:Wide {{{at_limit}}}), (:Wide)-[:WIDE {{{at_limit}}}]->(:Wide)
        }}"
    ))
    .unwrap();
    assert_eq!(properties(&db, "NODE", "Wide"), at_limit);
    assert_eq!(properties(&db, "EDGE", "WIDE"), at_limit);
    let before = catalog(&db);
    let past = format!("{at_limit}, extra LIST<INT64>");
    for element in [
        format!("(:Wider {{{past}}})"),
        format!("(:Wide)-[:WIDER {{{past}}}]->(:Wide)"),
    ] {
        let error = db
            .execute_cypher(&format!("ALTER CURRENT GRAPH TYPE SET {{ {element} }}"))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("nest at most 32768 LIST<...> levels in all"),
            "{error}"
        );
        assert_eq!(catalog(&db), before);
    }
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
fn assert_reopened_types(db: &GrafeoDB) {
    assert_eq!(
        properties(db, "NODE", "Event"),
        "begins ZONED DATETIME NOT NULL, departs LOCAL DATETIME, \
         tags LIST<STRING>, stamps LIST<ZONED DATETIME>"
    );
    assert_eq!(
        properties(db, "EDGE", "BOOKED"),
        "times LIST<LOCAL DATETIME> NOT NULL"
    );
    assert_values_and_rejected_writes(db);
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn compound_types_survive_checkpoint_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cypher-types.grafeo");
    let db = GrafeoDB::with_config(grafeo_engine::Config::persistent(&path)).unwrap();
    db.execute_cypher(EVENT_TYPES).unwrap();
    insert_valid_values(&db);
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(grafeo_engine::Config::persistent(&path)).unwrap();
    assert_reopened_types(&db);
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn compound_types_survive_wal_replay() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "compound_types_survive_wal_replay",
        |path| GrafeoDB::with_config(grafeo_engine::Config::persistent(path)).unwrap(),
        |db| {
            db.execute_cypher(EVENT_TYPES).unwrap();
            insert_valid_values(db);
        },
    );
    assert_reopened_types(&db);
}

/// GQL and Cypher read a property type with the same rules (one shared
/// grammar for the depth of `LIST<...>` and the `DATETIME` prefixes): the
/// same depth is the deepest accepted, a deeper one is refused with the same
/// message, and the same spellings of a two-word type are accepted, shown
/// alike and refused alike.
#[test]
fn gql_and_cypher_read_property_types_by_the_same_rules() {
    let nested = |depth: usize| format!("{}INT64{}", "LIST<".repeat(depth), ">".repeat(depth));
    let gql = |property_type: &str| {
        let db = GrafeoDB::new_in_memory();
        db.execute(&format!("CREATE NODE TYPE Typed (a {property_type})"))
            .map(|_| properties(&db, "NODE", "Typed"))
            .map_err(|error| error.to_string())
    };
    let cypher = |property_type: &str| {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher(&format!(
            "ALTER CURRENT GRAPH TYPE SET {{ (:Typed {{a {property_type}}}) }}"
        ))
        .map(|_| properties(&db, "NODE", "Typed"))
        .map_err(|error| error.to_string())
    };

    let deepest = nested(128);
    assert_eq!(gql(&deepest), Ok(format!("a {deepest}")));
    assert_eq!(cypher(&deepest), Ok(format!("a {deepest}")));
    let message = "A property type nests at most 128 LIST<...> levels";
    for depth in [129, 100_000] {
        for (language, result) in [
            ("GQL", gql(&nested(depth))),
            ("Cypher", cypher(&nested(depth))),
        ] {
            let error = result.expect_err(&format!("{language}, depth {depth}"));
            assert!(
                error.contains(message),
                "{language}, depth {depth}: {error}"
            );
        }
    }

    for (spelling, shown) in [
        ("ZONED DATETIME", "a ZONED DATETIME"),
        ("zoned datetime", "a ZONED DATETIME"),
        ("LOCAL DATETIME", "a LOCAL DATETIME"),
        ("Local Datetime", "a LOCAL DATETIME"),
    ] {
        assert_eq!(gql(spelling).as_deref(), Ok(shown), "GQL {spelling}");
        assert_eq!(cypher(spelling).as_deref(), Ok(shown), "Cypher {spelling}");
    }
    for spelling in [
        "ZONED",
        "LOCAL STRING",
        "GLOBAL DATETIME",
        "ZONED ZONED DATETIME",
    ] {
        assert!(gql(spelling).is_err(), "GQL accepts {spelling}");
        assert!(cypher(spelling).is_err(), "Cypher accepts {spelling}");
    }
}
