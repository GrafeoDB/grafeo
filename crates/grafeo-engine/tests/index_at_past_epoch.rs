//! A read of a past epoch, outside a transaction, finds the nodes that had a
//! value at that epoch: a property index holds the values of now, so such a
//! read scans instead of looking a value up (#455). At the current epoch and
//! in a transaction the index is used.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test index_at_past_epoch
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "temporal"))]

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::{GrafeoDB, Session};

/// `X:Y` Alix and `X` Gus, both with `k` 1 at the returned epoch; Alix has `k`
/// 2 since. With `indexed`, a property index on `k`.
fn changed(indexed: bool) -> (GrafeoDB, EpochId) {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("k").unwrap();
    }
    db.execute("INSERT (:X:Y {name: 'Alix', k: 1}), (:X {name: 'Gus', k: 1})")
        .unwrap();
    let before = db.current_epoch();
    db.execute("MATCH (t {name: 'Alix'}) SET t.k = 2").unwrap();
    (db, before)
}

fn names(result: grafeo_engine::database::QueryResult) -> Vec<String> {
    let mut names: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    names.sort();
    names
}

fn plan(session: &Session, query: &str, epoch: Option<EpochId>) -> String {
    let explain = format!("EXPLAIN {query}");
    let result = match epoch {
        Some(epoch) => session.execute_at_epoch(&explain, epoch),
        None => session.execute(&explain),
    }
    .unwrap();
    result
        .rows()
        .iter()
        .map(|row| format!("{:?}", row[0]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Each query with the names it returns at the earlier epoch, and the hint
/// `EXPLAIN` shows for the index at the current epoch: a seek per row, the
/// lookup of a constant, a list of constants, a range.
fn cases() -> Vec<(&'static str, Vec<&'static str>, Option<&'static str>)> {
    vec![
        (
            "UNWIND [1] AS v MATCH (t:X {k: v}) RETURN t.name",
            vec!["Alix", "Gus"],
            Some("[index: k]"),
        ),
        (
            "UNWIND [1] AS v MATCH (t:X:Y {k: v}) RETURN t.name",
            vec!["Alix"],
            Some("[index: k]"),
        ),
        (
            "MATCH (t:X {k: 1}) RETURN t.name",
            vec!["Alix", "Gus"],
            Some("[index: k]"),
        ),
        (
            "MATCH (t:X:Y {k: 1}) RETURN t.name",
            vec!["Alix"],
            Some("[index: k]"),
        ),
        (
            "MATCH (t:X) WHERE t.k IN [1, 3] RETURN t.name",
            vec!["Alix", "Gus"],
            None,
        ),
        (
            "MATCH (t:X) WHERE t.k < 2 RETURN t.name",
            vec!["Alix", "Gus"],
            Some("[range: k]"),
        ),
        (
            "MATCH (t:X) WHERE t.k >= 1 AND t.k <= 1 RETURN t.name",
            vec!["Alix", "Gus"],
            None,
        ),
    ]
}

#[test]
fn a_read_of_a_past_epoch_finds_the_values_of_that_epoch() {
    for indexed in [true, false] {
        let (db, before) = changed(indexed);
        let session = db.session();
        for (query, expected, _) in cases() {
            assert_eq!(
                names(session.execute_at_epoch(query, before).unwrap()),
                expected,
                "`{query}` at the earlier epoch (index: {indexed})"
            );
        }
    }
}

/// `EXPLAIN` agrees: no index at a past epoch, the index at the current epoch
/// and in a transaction.
#[test]
fn explain_shows_the_index_only_where_it_is_used() {
    let (db, before) = changed(true);
    let mut session = db.session();
    for (query, _, hint) in cases() {
        let past = plan(&session, query, Some(before));
        assert!(
            !past.contains("[index") && !past.contains("[range"),
            "`{query}` at the earlier epoch:\n{past}"
        );
        if let Some(hint) = hint {
            let now = plan(&session, query, None);
            assert!(now.contains(hint), "`{query}` now:\n{now}");
            session.begin_transaction().unwrap();
            let in_transaction = plan(&session, query, None);
            session.rollback().unwrap();
            assert!(
                in_transaction.contains(hint),
                "`{query}` in a transaction:\n{in_transaction}"
            );
        }
    }
    // At the current epoch the rows are those of now.
    assert_eq!(
        names(session.execute("MATCH (t:X {k: 1}) RETURN t.name").unwrap()),
        ["Gus"]
    );
}
