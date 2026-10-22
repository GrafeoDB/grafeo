//! Zoned datetimes, scaled durations and local datetimes that queries compute
//! and store come back the same after a close and a reopen: a zoned value
//! keeps its offset (Cypher's `datetime()` is zoned, and so is zoned datetime
//! arithmetic), a duration keeps every component a fraction carried into,
//! and a local datetime's text has no zone. Filters and indexes over the
//! reopened values find the same nodes.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test zoned_values_after_reopen
//! ```

#![cfg(all(
    feature = "gql",
    feature = "cypher",
    feature = "grafeo-file",
    feature = "wal"
))]

use std::path::Path;

use grafeo_engine::{Config, GrafeoDB};

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// The first row of `query` (GQL), each value as text.
fn row(db: &GrafeoDB, query: &str) -> Vec<String> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()[0]
        .iter()
        .map(|value| match value {
            grafeo_common::types::Value::String(s) => s.to_string(),
            other => other.to_string(),
        })
        .collect()
}

/// Writes one event with values computed in Cypher and in GQL.
fn write_event(db: &GrafeoDB) {
    db.execute_cypher(
        "CREATE (:Event {name: 'Amsterdam', at: datetime('2024-03-15T10:00:00+01:00'), \
         later: datetime('2024-03-15T10:00:00+01:00') + duration('PT1H'), \
         utc: datetime('2024-03-15T10:00:00'), \
         stay: duration('P1M') * 1.5, half: duration('P1D') / 2})",
    )
    .unwrap();
    db.execute(
        "MATCH (e:Event) SET e.month_later = zoned_datetime('2024-01-31T23:30:00+01:00') + DURATION 'P1M', \
         e.local = local_datetime('2024-03-15T10:00:00.25')",
    )
    .unwrap();
}

const READ: &str = "MATCH (e:Event) RETURN toString(e.at) AS at, toString(e.later) AS later, \
     toString(e.utc) AS utc, toString(e.stay) AS stay, toString(e.half) AS half, \
     toString(e.month_later) AS month_later, toString(e.local) AS local";

fn expected() -> Vec<String> {
    [
        "2024-03-15T10:00:00+01:00",
        "2024-03-15T11:00:00+01:00",
        "2024-03-15T10:00:00Z",
        "P1M15DT5H14M33S",
        "PT12H",
        "2024-02-29T23:30:00+01:00",
        "2024-03-15T10:00:00.25",
    ]
    .map(String::from)
    .to_vec()
}

#[test]
fn computed_temporal_values_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.grafeo");
    {
        let db = open(&path);
        write_event(&db);
        assert_eq!(row(&db, READ), expected(), "before the reopen");
        db.close().unwrap();
    }
    let db = open(&path);
    assert_eq!(row(&db, READ), expected(), "after the reopen");
    // The zoned value keeps its offset, as Cypher reads it
    let offset = db
        .execute_cypher("MATCH (e:Event) RETURN e.at.offset AS o, e.later.hour AS h")
        .unwrap();
    assert_eq!(
        offset.rows()[0],
        [
            grafeo_common::types::Value::from("+01:00"),
            grafeo_common::types::Value::Int64(11)
        ]
    );
}

/// A stored zoned datetime is found by its instant through a filter, with
/// and without a property index, before and after a reopen.
#[test]
fn a_reopened_zoned_datetime_is_found_by_its_instant() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.grafeo");
    let query = "MATCH (e:Event) WHERE e.at = zoned_datetime('2024-03-15T09:00:00Z') \
                 RETURN e.name AS name";
    let names = |db: &GrafeoDB| row(db, query);
    {
        let db = open(&path);
        write_event(&db);
        assert_eq!(names(&db), ["Amsterdam"]);
        db.close().unwrap();
    }
    let db = open(&path);
    assert_eq!(names(&db), ["Amsterdam"], "after the reopen");
    db.execute("CREATE INDEX event_at FOR (e:Event) ON (e.at)")
        .unwrap();
    assert_eq!(names(&db), ["Amsterdam"], "through the index");
}
