//! A range read answers as the filter it stands for under the statement's temporal rule: a zoned
//! against a local datetime, and a time with an offset against one without, compare by the
//! language and the session time zone, also inside lists. Each temporal kind, alone and inside
//! lists (first, last, nested, beside another kind), as a stored value and as a bound, in GQL and in
//! Cypher, without a session time zone and with two, against the same query with the range read
//! turned off. `range_column_read.rs` covers the other kinds without a session time zone.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main range_temporal_rules::
//! ```

use std::collections::HashMap;

use grafeo_common::types::{Date, Duration, Time, Timestamp, Value, ZonedDatetime};
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::PlannerRules;
use grafeo_engine::session::Session;

/// The language a form is written in: they read a zoned against a local datetime differently.
#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

/// The session time zones: none, east of UTC, and west of it.
const TIME_ZONES: [Option<&str>; 3] = [None, Some("+02:00"), Some("-05:00")];

/// The temporal values: of each kind two that order, and the pairs the rules decide differently
/// (one instant at two offsets, a local datetime at the instant of a zoned one in UTC and at
/// +02:00, a time with and without an offset).
fn temporals() -> Vec<Value> {
    let instant = Timestamp::from_micros(1_696_500_000_000_000);
    let later = Timestamp::from_micros(1_696_500_000_000_001);
    let two_hours = Timestamp::from_micros(1_696_500_000_000_000 + 7_200_000_000);
    let noon = Time::from_hms(12, 0, 0).unwrap();
    let one = Time::from_hms(13, 0, 0).unwrap();
    vec![
        Value::Date(Date::from_days(19_635)),
        Value::Date(Date::from_days(19_636)),
        Value::Timestamp(instant),
        Value::Timestamp(later),
        Value::Timestamp(two_hours),
        Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, 0)),
        Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, 7_200)),
        Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(later, -18_000)),
        Value::Time(noon),
        Value::Time(one),
        Value::Time(noon.with_offset(0)),
        Value::Time(noon.with_offset(3_600)),
        Value::Duration(Duration::new(1, 0, 0)),
        Value::Duration(Duration::new(0, 30, 0)),
    ]
}

fn list(values: Vec<Value>) -> Value {
    Value::List(values.into())
}

/// `value` alone and inside lists: alone, first, last, nested, and beside a value of another kind.
fn shapes(value: &Value) -> Vec<Value> {
    vec![
        value.clone(),
        list(vec![value.clone()]),
        list(vec![value.clone(), Value::Int64(1)]),
        list(vec![Value::Int64(1), value.clone()]),
        list(vec![list(vec![value.clone()])]),
    ]
}

fn every_value() -> Vec<Value> {
    temporals().iter().flat_map(shapes).collect()
}

/// The forms whose bound a range read takes, one bound and two; `n.row` tells the rows apart.
const FORMS: &[&str] = &[
    "MATCH (n:L) WHERE n.k > $v RETURN n.row",
    "MATCH (n:L) WHERE n.k >= $v RETURN n.row",
    "MATCH (n:L) WHERE n.k < $v RETURN n.row",
    "MATCH (n:L) WHERE n.k <= $v RETURN n.row",
    "MATCH (n:L) WHERE $v < n.k RETURN n.row",
    "MATCH (n) WHERE n.k > $v RETURN n.row",
    "MATCH (n:L) WHERE n.k >= $v AND n.k <= $v RETURN n.row",
];

/// A session with `time_zone`, the range read on (`PlannerRules` default) or off (`plain`).
fn session(db: &GrafeoDB, time_zone: Option<&str>, range: bool) -> Session {
    let mut session = db.session();
    if !range {
        session.set_planner_rules(PlannerRules::plain());
    }
    if let Some(zone) = time_zone {
        session
            .execute(&format!("SESSION SET TIME ZONE '{zone}'"))
            .unwrap();
    }
    session
}

fn run(
    session: &Session,
    language: Language,
    query: &str,
    bound: &Value,
) -> Result<Vec<String>, String> {
    let params = HashMap::from([("v".to_string(), bound.clone())]);
    let result = match language {
        Language::Gql => session.execute_with_params(query, params),
        Language::Cypher => session.execute_cypher_with_params(query, params),
    };
    result
        .map(|result| {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            rows
        })
        .map_err(|error| error.to_string())
}

/// The temporal kind a bound holds: the value alone or inside lists (with integers beside it).
fn temporal_kind(value: &Value) -> &'static str {
    match value {
        Value::Date(_) => "date",
        Value::Timestamp(_) => "local datetime",
        Value::ZonedDatetime(_) => "zoned datetime",
        Value::Time(_) => "time",
        Value::Duration(_) => "duration",
        Value::List(items) => items
            .iter()
            .map(temporal_kind)
            .find(|kind| *kind != "other")
            .unwrap_or("other"),
        _ => "other",
    }
}

/// Whether the statement runs a range read: `PROFILE` lists the operators the physical planner
/// chose. `EXPLAIN`'s `[range: k]` hint reads the predicate's shape alone, so it shows for a bound
/// the physical planner leaves to the filter (a time, or a datetime outside the default rule).
fn reads_the_range(session: &Session, language: Language, query: &str, bound: &Value) -> bool {
    run(session, language, &format!("PROFILE {query}"), bound)
        .is_ok_and(|rows| rows.iter().any(|row| row.contains("RangeScan")))
}

#[test]
fn a_range_read_follows_the_temporal_rule_of_the_statement() {
    let db = GrafeoDB::new_in_memory();
    let values = every_value();
    for (row, value) in values.iter().enumerate() {
        db.create_node_with_props(
            &["L"],
            [
                ("row", Value::Int64(i64::try_from(row).unwrap())),
                ("k", value.clone()),
            ],
        )
        .unwrap();
    }
    let mut wrong = Vec::new();
    let mut ranged = 0;
    // The range reads per temporal kind of the bound, under the default rule (GQL in UTC) or not.
    let mut ranged_by: HashMap<(&str, bool), usize> = HashMap::new();
    for time_zone in TIME_ZONES {
        let read = session(&db, time_zone, true);
        let filter = session(&db, time_zone, false);
        for language in [Language::Gql, Language::Cypher] {
            for bound in &values {
                for form in FORMS {
                    if reads_the_range(&read, language, form, bound) {
                        ranged += 1;
                        let default_rule = matches!(language, Language::Gql) && time_zone.is_none();
                        *ranged_by
                            .entry((temporal_kind(bound), default_rule))
                            .or_default() += 1;
                    }
                    let found = run(&read, language, form, bound);
                    let wanted = run(&filter, language, form, bound);
                    if found != wanted {
                        wrong.push(format!(
                            "{time_zone:?} {language:?}: {form} with {bound:?}\n  read   {found:?}\n  filter {wanted:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        ranged > 0,
        "no form took the range read: the test compares the filter with itself"
    );
    // The rule, kind by kind: the store orders a time apart from the query, and a datetime only
    // as the default rule does; every other kind as every rule does.
    let count = |kind: &'static str, default_rule: bool| {
        ranged_by.get(&(kind, default_rule)).copied().unwrap_or(0)
    };
    for kind in ["date", "local datetime", "zoned datetime", "duration"] {
        assert!(
            count(kind, true) > 0,
            "a {kind} bound never took the range read in GQL in UTC: {ranged_by:?}"
        );
    }
    for kind in ["date", "duration"] {
        assert!(
            count(kind, false) > 0,
            "a {kind} bound took the range read only in GQL in UTC: {ranged_by:?}"
        );
    }
    for (kind, default_rule) in [
        ("time", true),
        ("time", false),
        ("local datetime", false),
        ("zoned datetime", false),
    ] {
        assert_eq!(
            count(kind, default_rule),
            0,
            "a {kind} bound took the range read (default rule {default_rule}): {ranged_by:?}"
        );
    }
    assert!(
        wrong.is_empty(),
        "{} wrong ({ranged} took the range read):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
