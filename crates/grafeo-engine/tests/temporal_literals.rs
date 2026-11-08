//! A GQL datetime literal whose time carries a displacement is a zoned datetime that keeps it
//! (ISO/IEC 39075 21.2 Syntax Rule 65, General Rule 15), `Z` being the displacement +00:00; one
//! without a displacement stays a local datetime. It is the value the deprecated
//! `ZONED DATETIME '...'` gives for the same text, so the deprecation's named replacement works.
//! A stored zoned literal keeps its offset through a reopen, read in GQL and in Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main temporal_literals::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

// ISO: GV39, GV40

use grafeo_common::types::{Date, Value, ZonedDatetime};
use grafeo_engine::GrafeoDB;

/// The zoned value `text` is, as `ZonedDatetime::parse` reads it.
fn zoned(text: &str) -> Value {
    Value::ZonedDatetime(ZonedDatetime::parse(text).unwrap_or_else(|| panic!("{text} parses")))
}

/// The one row `query` returns.
fn row(db: &GrafeoDB, query: &str) -> Vec<Value> {
    let result = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    assert_eq!(result.rows().len(), 1, "{query}: one row");
    result.rows()[0].clone()
}

/// Each displacement is kept: the value is the zoned datetime of the text, at that offset, its
/// hour the local one written, and its text as written (`Z` for a zero displacement, `-00:00`
/// included).
#[test]
fn a_datetime_literal_with_a_displacement_keeps_it() {
    let db = GrafeoDB::new_in_memory();
    let cases = [
        (
            "2024-01-01T10:00:00+01:00",
            3_600,
            "2024-01-01T10:00:00+01:00",
        ),
        (
            "2024-01-01T10:00:00-05:30",
            -19_800,
            "2024-01-01T10:00:00-05:30",
        ),
        (
            "2024-01-01T10:00:00+14:00",
            50_400,
            "2024-01-01T10:00:00+14:00",
        ),
        ("2024-01-01T10:00:00Z", 0, "2024-01-01T10:00:00Z"),
        ("2024-01-01T10:00:00-00:00", 0, "2024-01-01T10:00:00Z"),
        (
            "2024-01-01T10:00:00.123456+01:00",
            3_600,
            "2024-01-01T10:00:00.123456+01:00",
        ),
    ];
    let mut wrong = Vec::new();
    for (text, offset, display) in cases {
        let got = row(
            &db,
            &format!(
                "RETURN DATETIME '{text}' AS v, hour(DATETIME '{text}') AS h, \
                 CAST(DATETIME '{text}' AS STRING) AS s, \
                 DATETIME '{text}' IS TYPED ZONED DATETIME AS z"
            ),
        );
        let want = [
            zoned(text),
            Value::Int64(10),
            Value::from(display),
            Value::Bool(true),
        ];
        if got != want {
            wrong.push(format!("DATETIME '{text}': {got:?}, want {want:?}"));
        }
        if let Value::ZonedDatetime(value) = &got[0]
            && value.offset_seconds() != offset
        {
            wrong.push(format!(
                "DATETIME '{text}': offset {}, want {offset}",
                value.offset_seconds()
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The control: without a displacement the literal stays a local datetime, at midnight for a
/// date alone.
#[test]
fn a_datetime_literal_without_a_displacement_stays_local() {
    let db = GrafeoDB::new_in_memory();
    for text in ["2024-01-01T10:00:00", "2024-01-01T10:00", "2024-01-01"] {
        let got = row(
            &db,
            &format!(
                "RETURN DATETIME '{text}' IS TYPED ZONED DATETIME AS z, \
                 DATETIME '{text}' = datetime('{text}') AS same"
            ),
        );
        assert_eq!(
            got,
            [Value::Bool(false), Value::Bool(true)],
            "DATETIME '{text}' is the local datetime() of the text"
        );
        assert!(
            matches!(
                row(&db, &format!("RETURN DATETIME '{text}' AS v"))[0],
                Value::Timestamp(_)
            ),
            "DATETIME '{text}' is a local datetime value"
        );
    }
}

/// The components of a zoned literal are the local ones written, not those of its instant at
/// UTC: half past midnight at +01:00 is hour 0 of the 15th, on the date 2024-03-15.
#[test]
fn a_zoned_literal_has_the_components_written() {
    let db = GrafeoDB::new_in_memory();
    let text = "2024-03-15T00:30:00+01:00";
    let got = row(
        &db,
        &format!(
            "RETURN hour(DATETIME '{text}') AS h, day(DATETIME '{text}') AS d, \
             CAST(DATETIME '{text}' AS DATE) AS date"
        ),
    );
    assert_eq!(
        got,
        [
            Value::Int64(0),
            Value::Int64(15),
            Value::Date(Date::parse("2024-03-15").unwrap()),
        ]
    );
}

/// `DATETIME '...'` with a displacement is the replacement the deprecation of
/// `ZONED DATETIME '...'` names: for the same text both give one value and one text, and only
/// the deprecated form warns (01X01).
#[test]
fn the_zoned_form_and_its_replacement_give_one_value() {
    let db = GrafeoDB::new_in_memory();
    for text in ["2024-01-01T10:00:00+01:00", "2024-01-01T10:00:00Z"] {
        let query = format!(
            "RETURN ZONED DATETIME '{text}' AS zoned, DATETIME '{text}' AS replacement, \
             CAST(ZONED DATETIME '{text}' AS STRING) = CAST(DATETIME '{text}' AS STRING) AS text"
        );
        let result = db
            .execute(&query)
            .unwrap_or_else(|error| panic!("{query}: {error}"));
        let got = &result.rows()[0];
        assert_eq!(got[0], got[1], "{text}: one value");
        assert_eq!(got[1], zoned(text), "{text}: the zoned value");
        assert_eq!(got[2], Value::Bool(true), "{text}: one text");
        let warnings: Vec<&str> = result.warnings().iter().map(|w| w.code.as_str()).collect();
        assert_eq!(
            warnings,
            ["01X01"],
            "{text}: the deprecated form alone warns"
        );

        let alone = db
            .execute(&format!("RETURN DATETIME '{text}' AS v"))
            .unwrap();
        assert!(
            alone.warnings().is_empty(),
            "{text}: the replacement raises nothing: {:?}",
            alone.warnings()
        );
    }
}

/// A zoned literal written to a property keeps its offset: read back in GQL and in Cypher, in
/// the open database and after a close and a reopen.
#[cfg(feature = "grafeo-file")]
#[test]
fn a_stored_zoned_literal_keeps_its_offset_through_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lisbon.grafeo");
    let plus_one = "2024-01-01T10:00:00+01:00";
    let utc = "2024-01-01T10:00:00Z";
    let want = vec![
        vec![Value::from("Alix"), zoned(plus_one)],
        vec![Value::from("Gus"), zoned(utc)],
    ];
    // Zoned values compare by their instant: the offsets are compared on their own.
    let offsets = |rows: &[Vec<Value>]| -> Vec<Option<i32>> {
        rows.iter()
            .map(|row| match &row[1] {
                Value::ZonedDatetime(at) => Some(at.offset_seconds()),
                _ => None,
            })
            .collect()
    };
    assert_eq!(offsets(&want), [Some(3_600), Some(0)]);
    let read = |db: &GrafeoDB, what: &str| {
        let gql = db
            .execute("MATCH (e:Event) RETURN e.who AS who, e.at AS at ORDER BY who")
            .unwrap();
        assert_eq!(gql.rows(), want, "{what}, GQL");
        assert_eq!(offsets(gql.rows()), offsets(&want), "{what}, GQL offsets");
        let cypher = db
            .execute_cypher("MATCH (e:Event) RETURN e.who AS who, e.at AS at ORDER BY who")
            .unwrap();
        assert_eq!(cypher.rows(), want, "{what}, Cypher");
        assert_eq!(
            offsets(cypher.rows()),
            offsets(&want),
            "{what}, Cypher offsets"
        );
    };

    let db = GrafeoDB::open(&path).unwrap();
    db.execute(&format!(
        "INSERT (:Event {{who: 'Alix', at: DATETIME '{plus_one}'}}), \
         (:Event {{who: 'Gus', at: DATETIME '{utc}'}})"
    ))
    .unwrap();
    read(&db, "open");
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    read(&db, "reopened");
    db.close().unwrap();
}

/// The catalog's error text writes a datetime as the query writes it: the `DATETIME '...'`
/// in a G2000 message reads back as the same value, offset kept (ISO/IEC 39075 21.2 SR 65).
#[test]
fn the_catalog_error_text_reads_back_as_the_same_value() {
    let db = GrafeoDB::new_in_memory();
    // A local value is refused by a ZONED DATETIME property and the other way round, so each
    // case writes into the type that refuses it.
    db.execute("CREATE NODE TYPE Local (at LOCAL DATETIME)")
        .unwrap();
    db.execute("CREATE NODE TYPE Zoned (at ZONED DATETIME)")
        .unwrap();
    let local = |text: &str| row(&db, &format!("RETURN DATETIME '{text}'"))[0].clone();
    let cases: [(&str, &str, Value); 5] = [
        ("2026-10-05T10:30:00", "Zoned", local("2026-10-05T10:30:00")),
        (
            "2026-10-05T10:30:00.25",
            "Zoned",
            local("2026-10-05T10:30:00.25"),
        ),
        (
            "2026-10-05T10:30:00+02:00",
            "Local",
            zoned("2026-10-05T10:30:00+02:00"),
        ),
        (
            "2026-10-05T10:30:00Z",
            "Local",
            zoned("2026-10-05T10:30:00Z"),
        ),
        (
            "2026-10-05T10:30:00-05:30",
            "Local",
            zoned("2026-10-05T10:30:00-05:30"),
        ),
    ];
    for (text, label, want) in cases {
        let error = db
            .execute(&format!("INSERT (:{label} {{at: DATETIME '{text}'}})"))
            .expect_err("the property type refuses the value");
        let message = error.to_string();
        let needle = "got DATETIME '";
        let start = message.find(needle).unwrap_or_else(|| {
            panic!("the message names the value as a query writes it: {message}")
        }) + needle.len();
        let end = message[start..]
            .find('\'')
            .map(|at| start + at)
            .expect("the written text closes");
        let written = &message[start..end];
        assert_eq!(
            row(&db, &format!("RETURN DATETIME '{written}'")),
            vec![want.clone()],
            "the message's text reads back as the value ({text})"
        );
    }
}
