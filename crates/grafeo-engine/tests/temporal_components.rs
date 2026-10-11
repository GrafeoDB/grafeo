//! Temporal values built from the Unix epoch and read by component (the
//! Cypher temporal functions and accessors of openCypher):
//!
//! - `datetime({epochMillis: n})` and `datetime({epochSeconds: n, nanosecond: m})`
//!   are the instant `n` milliseconds (seconds) after 1970-01-01T00:00:00Z.
//! - A component of a date, time, datetime or duration reads like a property:
//!   `d.year`, `d.month`, `d.day`, `t.hour`, `dt.epochMillis`, `dur.minutes`, ...
//!
//! Both were null, so a filter on a date component dropped every row. The
//! GQL component functions (`month(d)`, ...) read the same values.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test temporal_components
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::{Timestamp, Value};
use grafeo_engine::{GrafeoDB, Session};

/// 2020-05-25T00:00:00Z, a Monday, in milliseconds after the Unix epoch.
const MAY_25_2020_MILLIS: i64 = 1_590_364_800_000;

fn cypher(session: &Session, query: &str) -> Vec<Vec<Value>> {
    session
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("`{query}` failed: {error}"))
        .rows()
        .to_vec()
}

fn gql(session: &Session, query: &str) -> Vec<Vec<Value>> {
    session
        .execute(query)
        .unwrap_or_else(|error| panic!("`{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// Reads each `component` of the value `binding` (a Cypher expression), as
/// `WITH <binding> AS d RETURN d.<component>`, and checks it.
fn assert_components(session: &Session, binding: &str, expected: &[(&str, Value)]) {
    for (component, value) in expected {
        let query = format!("WITH {binding} AS d RETURN d.{component} AS c");
        assert_eq!(
            cypher(session, &query),
            [vec![value.clone()]],
            "{binding}.{component}"
        );
    }
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

#[test]
fn datetime_from_epoch_millis_is_that_instant() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (expression, expected) in [
        ("datetime({epochMillis: 0})", Timestamp::from_micros(0)),
        (
            "datetime({epochMillis: 1590364800000})",
            Timestamp::from_millis(MAY_25_2020_MILLIS),
        ),
        // Before the epoch: 1969-12-31T23:59:59.912Z.
        (
            "datetime({epochMillis: -88})",
            Timestamp::from_micros(-88_000),
        ),
        (
            "datetime({epochSeconds: 1590364800})",
            Timestamp::from_millis(MAY_25_2020_MILLIS),
        ),
        // Timestamps keep microseconds: 19,088 nanoseconds are 19 microseconds.
        (
            "datetime({epochSeconds: 3, nanosecond: 19088})",
            Timestamp::from_micros(3_000_019),
        ),
    ] {
        assert_eq!(
            cypher(&session, &format!("RETURN {expression} AS d")),
            [vec![Value::Timestamp(expected)]],
            "{expression}"
        );
    }
    assert_eq!(
        cypher(
            &session,
            "RETURN datetime({epochMillis: 1590364800000}) = datetime('2020-05-25T00:00:00Z') AS same"
        ),
        [vec![Value::Bool(true)]]
    );
}

#[test]
fn datetime_from_a_null_epoch_is_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(
        cypher(&session, "RETURN datetime({epochMillis: null}) AS d"),
        [vec![Value::Null]]
    );
}

#[test]
fn the_components_of_a_date() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_components(
        &session,
        "date('2020-05-25')",
        &[
            ("year", int(2020)),
            ("quarter", int(2)),
            ("month", int(5)),
            ("week", int(22)),
            ("weekYear", int(2020)),
            ("day", int(25)),
            ("ordinalDay", int(146)),
            ("dayOfWeek", int(1)),
            ("dayOfQuarter", int(55)),
        ],
    );
    // 2021-01-01 is a Friday in ISO week 53 of 2020.
    assert_components(
        &session,
        "date('2021-01-01')",
        &[
            ("week", int(53)),
            ("weekYear", int(2020)),
            ("dayOfWeek", int(5)),
            ("ordinalDay", int(1)),
        ],
    );
}

#[test]
fn the_components_of_a_datetime() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_components(
        &session,
        "datetime('2020-05-25T19:03:08.088Z')",
        &[
            ("year", int(2020)),
            ("month", int(5)),
            ("day", int(25)),
            ("dayOfWeek", int(1)),
            ("hour", int(19)),
            ("minute", int(3)),
            ("second", int(8)),
            ("millisecond", int(88)),
            ("microsecond", int(88_000)),
            ("nanosecond", int(88_000_000)),
            ("epochSeconds", int(1_590_433_388)),
            ("epochMillis", int(1_590_433_388_088)),
        ],
    );
}

#[test]
fn the_components_of_a_datetime_built_from_epoch_millis() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_components(
        &session,
        "datetime({epochMillis: 1590433388088})",
        &[
            ("year", int(2020)),
            ("month", int(5)),
            ("day", int(25)),
            ("hour", int(19)),
            ("epochMillis", int(1_590_433_388_088)),
        ],
    );
}

#[test]
fn the_components_of_a_zoned_datetime_are_local_to_its_offset() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_components(
        &session,
        "zoned_datetime('2020-05-25T01:03:08+02:00')",
        &[
            ("year", int(2020)),
            ("month", int(5)),
            ("day", int(25)),
            ("hour", int(1)),
            ("minute", int(3)),
            ("offsetSeconds", int(7200)),
            ("offsetMinutes", int(120)),
            ("offset", Value::from("+02:00")),
            // 2020-05-24T23:03:08Z
            ("epochSeconds", int(1_590_361_388)),
        ],
    );
}

#[test]
fn the_components_of_a_time() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_components(
        &session,
        "time('19:03:08.088')",
        &[
            ("hour", int(19)),
            ("minute", int(3)),
            ("second", int(8)),
            ("millisecond", int(88)),
            ("microsecond", int(88_000)),
            ("nanosecond", int(88_000_000)),
        ],
    );
    assert_components(
        &session,
        "time('19:03:08+02:00')",
        &[
            ("hour", int(19)),
            ("offsetSeconds", int(7200)),
            ("offset", Value::from("+02:00")),
        ],
    );
}

#[test]
fn the_components_of_a_duration() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // 15 months, 19 days and 3 h 8 min 19 s (11,299 seconds).
    assert_components(
        &session,
        "duration({years: 1, months: 3, days: 19, hours: 3, minutes: 8, seconds: 19})",
        &[
            ("years", int(1)),
            ("quarters", int(5)),
            ("months", int(15)),
            ("weeks", int(2)),
            ("days", int(19)),
            ("hours", int(3)),
            ("minutes", int(188)),
            ("seconds", int(11_299)),
            ("milliseconds", int(11_299_000)),
            ("quartersOfYear", int(1)),
            ("monthsOfQuarter", int(0)),
            ("monthsOfYear", int(3)),
            ("daysOfWeek", int(5)),
            ("minutesOfHour", int(8)),
            ("secondsOfMinute", int(19)),
            ("nanosecondsOfSecond", int(0)),
        ],
    );
}

#[test]
fn a_component_of_a_null_temporal_value_is_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_components(&session, "null", &[("month", Value::Null)]);
}

/// The shape of LDBC Interactive complex query 10: birthdays stored as epoch
/// milliseconds, filtered by the month and day of `datetime({epochMillis: ...})`.
#[test]
fn a_where_on_the_month_and_day_of_an_epoch_birthday_keeps_the_matching_rows() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // 1988-07-21, 1988-07-19 and 1988-08-03, each at 00:00:00Z.
    cypher(
        &session,
        "CREATE (:Person {name: 'Alix', birthday: 585446400000}), \
         (:Person {name: 'Gus', birthday: 585273600000}), \
         (:Person {name: 'Vincent', birthday: 586569600000})",
    );
    let rows = cypher(
        &session,
        "MATCH (p:Person) WITH p, datetime({epochMillis: p.birthday}) AS birthday \
         WHERE (birthday.month = 7 AND birthday.day >= 21) OR (birthday.month = 8 AND birthday.day < 22) \
         RETURN p.name AS name, birthday.month AS month, birthday.day AS day ORDER BY name",
    );
    assert_eq!(
        rows,
        [
            vec![Value::from("Alix"), int(7), int(21)],
            vec![Value::from("Vincent"), int(8), int(3)],
        ]
    );
}

#[test]
fn a_component_of_a_temporal_property_reads_through_the_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(
        &session,
        "CREATE (:Person {name: 'Alix', born: date('1988-07-21')})",
    );
    assert_eq!(
        cypher(
            &session,
            "MATCH (p:Person) RETURN p.born.year AS y, p.born.month AS m"
        ),
        [vec![int(1988), int(7)]]
    );
}

#[test]
fn a_component_written_into_a_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(&session, "CREATE (:Person {name: 'Alix'})");
    cypher(
        &session,
        "MATCH (p:Person) WITH p, date('2020-05-25') AS d SET p.month = d.month",
    );
    assert_eq!(
        cypher(&session, "MATCH (p:Person) RETURN p.month"),
        [vec![int(5)]]
    );
}

#[test]
fn gql_component_functions_read_the_same_values() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(
        gql(
            &session,
            "FOR d IN [datetime({epochMillis: 1590433388088})] \
             RETURN year(d), month(d), day(d), hour(d), minute(d), second(d)"
        ),
        [vec![int(2020), int(5), int(25), int(19), int(3), int(8)]]
    );
    assert_eq!(
        cypher(
            &session,
            "WITH datetime({epochMillis: 1590433388088}) AS d \
             RETURN d.year, d.month, d.day, d.hour, d.minute, d.second"
        ),
        [vec![int(2020), int(5), int(25), int(19), int(3), int(8)]]
    );
}
