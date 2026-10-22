//! Zoned and local datetimes and times that Cypher stores (`datetime()` and
//! `time()` are zoned, `localdatetime()` and `localtime()` local) come back
//! the same after a close and a reopen, and compare as each language reads
//! them: GQL reads the local value at the session time zone (UTC without
//! one), Cypher keeps the two apart. A property index finds what the filter
//! finds.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test zoned_and_local_after_reopen
//! ```

#![cfg(all(
    feature = "gql",
    feature = "cypher",
    feature = "grafeo-file",
    feature = "wal"
))]

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, Session};

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// The values of one column of `query` in `language`, as text, sorted.
fn column(session: &Session, language: &str, query: &str) -> Vec<String> {
    let mut values: Vec<String> = session
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            other => other.to_string(),
        })
        .collect();
    values.sort();
    values
}

fn write(db: &GrafeoDB) {
    db.execute_cypher(
        "CREATE (:Event {name: 'zoned', at: datetime('2024-03-15T10:00:00+01:00'), \
         clock: time('10:00:00+01:00')}), \
         (:Event {name: 'local', at: localdatetime('2024-03-15T09:00:00'), \
         clock: localtime('09:00:00')})",
    )
    .unwrap();
}

fn check(db: &GrafeoDB, stage: &str) {
    let session = db.session();
    assert_eq!(
        column(
            &session,
            "cypher",
            "MATCH (e:Event) RETURN toString(e.at) + ' ' + toString(e.clock)"
        ),
        [
            "2024-03-15T09:00:00 09:00:00",
            "2024-03-15T10:00:00+01:00 10:00:00+01:00"
        ],
        "{stage}: the stored values"
    );
    // GQL without a session time zone: the local values read at UTC, so
    // both events are at 09:00Z
    let zoned_at = "zoned_datetime('2024-03-15T09:00:00Z')";
    let gql = format!("MATCH (e:Event) WHERE e.at = {zoned_at} RETURN e.name");
    assert_eq!(column(&session, "gql", &gql), ["local", "zoned"], "{stage}");
    assert_eq!(
        column(
            &session,
            "gql",
            "MATCH (e:Event) WHERE e.clock = zonedTime('09:00:00Z') RETURN e.name"
        ),
        ["local", "zoned"],
        "{stage}: times"
    );
    // Cypher keeps them apart
    assert_eq!(
        column(
            &session,
            "cypher",
            "MATCH (e:Event) WHERE e.at = datetime('2024-03-15T09:00:00Z') RETURN e.name"
        ),
        ["zoned"],
        "{stage}: Cypher"
    );
    // GQL at +01:00: the local event is at 08:00Z now
    session.set_time_zone("+01:00").unwrap();
    assert_eq!(
        column(&session, "gql", &gql),
        ["zoned"],
        "{stage}: at +01:00"
    );
    assert_eq!(
        column(
            &session,
            "gql",
            "MATCH (e:Event) WHERE e.at < zoned_datetime('2024-03-15T08:30:00Z') RETURN e.name"
        ),
        ["local"],
        "{stage}: a range at +01:00"
    );
}

#[test]
fn zoned_and_local_values_survive_a_reopen_and_compare_by_language() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.grafeo");
    {
        let db = open(&path);
        write(&db);
        check(&db, "before the reopen");
        db.close().unwrap();
    }
    let db = open(&path);
    check(&db, "after the reopen");
    db.execute("CREATE INDEX event_at FOR (e:Event) ON (e.at)")
        .unwrap();
    db.execute("CREATE INDEX event_clock FOR (e:Event) ON (e.clock)")
        .unwrap();
    check(&db, "through the indexes");
}
