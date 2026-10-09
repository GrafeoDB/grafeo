//! `ZONED DATETIME`, `LOCAL DATETIME` and typed lists such as `LIST<STRING>`
//! are property types in type DDL (#569): node, edge and inline graph types
//! declare them, values are checked against them, and they survive a reopen
//! after WAL replay and after a checkpoint.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test type_ddl_types
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

mod common;

use grafeo_common::types::{Timestamp, Value, ZonedDatetime};
use grafeo_engine::{Config, GrafeoDB};

/// The `properties` column of the row of `type_name` in a SHOW statement.
fn property_list(db: &GrafeoDB, query: &str, type_name: &str) -> String {
    let result = db.execute(query).unwrap();
    let row = result
        .rows()
        .iter()
        .find(|row| row[0] == Value::from(type_name))
        .unwrap_or_else(|| panic!("{query} lists no {type_name}"));
    row[1].as_str().unwrap().to_string()
}

#[test]
fn zoned_and_local_datetimes_and_typed_lists_are_property_types() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE NODE TYPE Event (begins ZONED DATETIME, departs LOCAL DATETIME, \
         tags LIST<STRING>, scores LIST<LIST<INT64>>, stamps LIST<ZONED DATETIME>)",
    )
    .unwrap();
    db.execute("CREATE EDGE TYPE BOOKED (booked ZONED DATETIME)")
        .unwrap();
    db.execute("CREATE GRAPH TYPE trips (NODE TYPE Stop (arrives ZONED DATETIME))")
        .unwrap();
    db.execute("ALTER NODE TYPE Event ADD seen LOCAL DATETIME")
        .unwrap();
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Event"),
        "begins ZONED DATETIME, departs LOCAL DATETIME, tags LIST<STRING>, \
         scores LIST<LIST<INT64>>, stamps LIST<ZONED DATETIME>, seen LOCAL DATETIME"
    );
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Stop"),
        "arrives ZONED DATETIME"
    );
    assert_eq!(
        property_list(&db, "SHOW EDGE TYPES", "BOOKED"),
        "booked ZONED DATETIME"
    );
}

/// Pattern-form graph types (`{ ... }` property blocks) and lowercase
/// spellings read the same types.
#[test]
fn pattern_form_graph_types_and_lowercase_spellings_read_the_new_types() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE GRAPH TYPE routes ((:Station {opened zoned datetime, \
         lines list<local datetime>})-[:LEG {departs local datetime}]->(:Station))",
    )
    .unwrap();
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Station"),
        "opened ZONED DATETIME, lines LIST<LOCAL DATETIME>"
    );
    assert_eq!(
        property_list(&db, "SHOW EDGE TYPES", "LEG"),
        "departs LOCAL DATETIME"
    );
}

#[test]
fn values_are_checked_against_the_new_types() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE NODE TYPE Event (begins ZONED DATETIME, departs LOCAL DATETIME, \
         tags LIST<STRING>, stamps LIST<ZONED DATETIME>)",
    )
    .unwrap();
    let zoned = Value::ZonedDatetime(ZonedDatetime::parse("2026-10-05T10:30:00+02:00").unwrap());
    let local = Value::Timestamp(Timestamp::from_secs(1_791_000_000));
    let insert = |key: &str, value: Value| {
        db.execute_with_params(
            &format!("INSERT (:Event {{{key}: $v}})"),
            [("v".to_string(), value)].into_iter().collect(),
        )
    };

    insert("begins", zoned.clone()).unwrap();
    assert!(
        insert("begins", Value::from("Amsterdam")).is_err(),
        "a string is not a ZONED DATETIME"
    );
    assert!(
        insert("begins", local.clone()).is_err(),
        "a local datetime is not a ZONED DATETIME"
    );
    insert("departs", local.clone()).unwrap();
    assert!(
        insert("departs", zoned.clone()).is_err(),
        "a zoned datetime is not a LOCAL DATETIME"
    );
    insert("tags", Value::List(vec![Value::from("Berlin")].into())).unwrap();
    assert!(
        insert("tags", Value::List(vec![Value::Int64(3)].into())).is_err(),
        "an integer element is not a STRING"
    );
    insert("stamps", Value::List(vec![zoned.clone()].into())).unwrap();
    assert!(
        insert("stamps", Value::List(vec![zoned, local].into())).is_err(),
        "a local datetime element is not a ZONED DATETIME"
    );
    assert_eq!(
        db.execute("MATCH (e:Event) RETURN count(e)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(4),
        "only the four inserts of matching values went in"
    );
}

/// A property type nests at most 128 `LIST<...>` levels, as deep as a
/// property value may be. The parser refuses a deeper one, and a quoted type
/// name (one name, which no supported type has with `LIST<` in it), so no
/// statement builds a type deeper than the catalog records store.
#[test]
fn property_types_nest_at_most_128_lists() {
    let db = GrafeoDB::new_in_memory();
    let nested = |levels: usize| format!("{}INT64{}", "LIST<".repeat(levels), ">".repeat(levels));
    db.execute(&format!("CREATE NODE TYPE Deep (a {})", nested(128)))
        .unwrap();
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Deep"),
        format!("a {}", nested(128))
    );

    for statement in [
        format!("CREATE NODE TYPE Deeper (a {})", nested(129)),
        format!("CREATE NODE TYPE Deeper (a {})", nested(100_000)),
    ] {
        let error = db.execute(&statement).unwrap_err().to_string();
        assert!(
            error.contains("A property type nests at most 128 LIST<...> levels"),
            "{}: {error}",
            &statement[..statement.len().min(60)]
        );
    }
    // A quoted type name is one name, and no supported type is called
    // `LIST<...>`: the parser refuses it before the catalog reads it.
    for statement in [
        format!("CREATE NODE TYPE Deeper (a `{}`)", nested(129)),
        format!("CREATE EDGE TYPE DEEPER (a `{}`)", nested(129)),
        format!("ALTER NODE TYPE Deep ADD b `{}`", nested(129)),
        format!(
            "CREATE GRAPH TYPE deeper (NODE TYPE Deeper (a `{}`))",
            nested(129)
        ),
    ] {
        let error = db.execute(&statement).unwrap_err().to_string();
        assert!(
            error.contains("is not a property type"),
            "{}: {error}",
            &statement[..statement.len().min(60)]
        );
    }
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Deep"),
        format!("a {}", nested(128)),
        "the refused ALTER added nothing"
    );
    for (query, name) in [
        ("SHOW NODE TYPES", "Deeper"),
        ("SHOW EDGE TYPES", "DEEPER"),
        ("SHOW GRAPH TYPES", "deeper"),
    ] {
        let rows = db.execute(query).unwrap();
        assert!(
            rows.rows().iter().all(|row| row[0] != Value::from(name)),
            "{query}: the refused {name} was created"
        );
    }
}

/// The property types of one node or edge type nest at most 32,768
/// `LIST<...>` levels in all, as many as one catalog record holds: 256
/// properties of the deepest type. `CREATE`, `CREATE OR REPLACE`,
/// `ALTER ... ADD` and inline graph types refuse more and change nothing, so
/// no statement builds a type a checkpoint cannot write; a type at the cap
/// survives a checkpoint and a reopen.
#[test]
fn the_property_types_of_a_type_nest_at_most_32768_lists_in_all() {
    let nested = |levels: usize| format!("{}INT64{}", "LIST<".repeat(levels), ">".repeat(levels));
    let deepest: Vec<String> = (0..256).map(|n| format!("p{n} {}", nested(128))).collect();
    let deepest = deepest.join(", ");
    let past = format!("{deepest}, extra LIST<INT64>");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wide.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    db.execute(&format!("CREATE NODE TYPE Wide ({deepest})"))
        .unwrap();
    db.execute(&format!("CREATE EDGE TYPE WIDE ({deepest})"))
        .unwrap();
    for statement in [
        format!("CREATE NODE TYPE Wider ({past})"),
        format!("CREATE EDGE TYPE WIDER ({past})"),
        format!("CREATE OR REPLACE NODE TYPE Wide ({past})"),
        format!("CREATE OR REPLACE EDGE TYPE WIDE ({past})"),
        "ALTER NODE TYPE Wide ADD extra LIST<INT64>".to_string(),
        "ALTER EDGE TYPE WIDE ADD extra LIST<INT64>".to_string(),
        format!("CREATE GRAPH TYPE wider (NODE TYPE Wider ({past}))"),
        format!("CREATE GRAPH TYPE wider (EDGE TYPE WIDER ({past}))"),
    ] {
        let error = db.execute(&statement).unwrap_err().to_string();
        assert!(
            error.contains("nest at most 32768 LIST<...> levels in all"),
            "{}: {error}",
            &statement[..statement.len().min(60)]
        );
    }
    for (query, name) in [
        ("SHOW NODE TYPES", "Wider"),
        ("SHOW EDGE TYPES", "WIDER"),
        ("SHOW GRAPH TYPES", "wider"),
    ] {
        let rows = db.execute(query).unwrap();
        assert!(
            rows.rows().iter().all(|row| row[0] != Value::from(name)),
            "{query}: the refused {name} was created"
        );
    }
    // A property with no LIST level still fits.
    db.execute("ALTER NODE TYPE Wide ADD city STRING").unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Wide"),
        format!("{deepest}, city STRING"),
        "the refused statements left Wide as it was, and it reopens"
    );
    assert_eq!(
        property_list(&db, "SHOW EDGE TYPES", "WIDE"),
        deepest,
        "the refused statements left WIDE as it was, and it reopens"
    );
}

#[test]
fn the_new_types_survive_a_reopen_after_wal_replay() {
    let (_dir, db) = crate::common::replay::reopened_after_crash(
        "the_new_types_survive_a_reopen_after_wal_replay",
        |path| GrafeoDB::with_config(Config::persistent(path)).unwrap(),
        |db| {
            db.execute("CREATE NODE TYPE Event (begins ZONED DATETIME, tags LIST<STRING>)")
                .unwrap();
            db.execute("CREATE EDGE TYPE BOOKED (booked LIST<LOCAL DATETIME>)")
                .unwrap();
            db.execute("CREATE GRAPH TYPE trips (NODE TYPE Stop (arrives ZONED DATETIME))")
                .unwrap();
            db.execute("ALTER NODE TYPE Event ADD seen LOCAL DATETIME")
                .unwrap();
        },
    );
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Event"),
        "begins ZONED DATETIME, tags LIST<STRING>, seen LOCAL DATETIME"
    );
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Stop"),
        "arrives ZONED DATETIME"
    );
    assert_eq!(
        property_list(&db, "SHOW EDGE TYPES", "BOOKED"),
        "booked LIST<LOCAL DATETIME>"
    );
}

#[test]
fn the_new_types_survive_a_reopen_after_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    db.execute("CREATE NODE TYPE Event (begins ZONED DATETIME, departs LOCAL DATETIME)")
        .unwrap();
    db.execute("CREATE EDGE TYPE BOOKED (stamps LIST<ZONED DATETIME>)")
        .unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Event"),
        "begins ZONED DATETIME, departs LOCAL DATETIME"
    );
    assert_eq!(
        property_list(&db, "SHOW EDGE TYPES", "BOOKED"),
        "stamps LIST<ZONED DATETIME>"
    );
}
