//! An ended transaction's change set, emptied, is the next transaction's:
//! the next commit reports only its own writes, after a commit, a rollback
//! or a failed statement, in an autocommit write or an explicit transaction,
//! in another graph than the last one's, and after a set too large to keep.
//!
//! cargo test -p grafeo-engine --all-features --test reused_change_sets

#![cfg(all(feature = "cdc", feature = "gql"))]

use std::collections::BTreeMap;

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::cdc::ChangeEvent;
use grafeo_engine::{Config, GrafeoDB};

fn db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
}

/// The events of each epoch so far: how many, per epoch.
fn events_per_epoch(db: &GrafeoDB) -> BTreeMap<u64, Vec<ChangeEvent>> {
    let mut per_epoch: BTreeMap<u64, Vec<ChangeEvent>> = BTreeMap::new();
    for event in db
        .changes_between(EpochId::new(0), db.current_epoch())
        .unwrap()
    {
        per_epoch
            .entry(event.epoch.as_u64())
            .or_default()
            .push(event);
    }
    per_epoch
}

fn names(db: &GrafeoDB, label: &str) -> Vec<Value> {
    let mut names: Vec<Value> = db
        .execute(&format!("MATCH (n:{label}) RETURN n.name"))
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    names.sort_by_key(|name| name.to_string());
    names
}

/// Commits and rollbacks one after another, autocommit and in transactions:
/// every commit reports its own writes alone, and a rolled back
/// transaction's writes appear in no later commit.
#[test]
fn each_commit_reports_its_own_writes_alone() {
    let db = db();
    let mut session = db.session();
    let mut expected_writes = Vec::new();
    for round in 0..40_i64 {
        let before = db.current_epoch().as_u64();
        match round % 4 {
            // An autocommit write of one node.
            0 => {
                session
                    .execute(&format!("INSERT (:Kept {{name: 'Alix-{round}'}})"))
                    .unwrap();
                expected_writes.push(1);
            }
            // A transaction of three nodes.
            1 => {
                session.begin_transaction().unwrap();
                for part in 0..3 {
                    session
                        .execute(&format!("INSERT (:Kept {{name: 'Gus-{round}-{part}'}})"))
                        .unwrap();
                }
                session.commit().unwrap();
                expected_writes.push(3);
            }
            // A transaction of four nodes, rolled back: its set, emptied, is
            // the next transaction's.
            2 => {
                session.begin_transaction().unwrap();
                for part in 0..4 {
                    session
                        .execute(&format!(
                            "INSERT (:Gone {{name: 'Vincent-{round}-{part}'}})"
                        ))
                        .unwrap();
                }
                session.rollback().unwrap();
            }
            // A statement that fails rolls its own transaction back.
            _ => {
                session
                    .execute("INSERT (:Gone {name: 'Jules'}) RETURN 1 / 0")
                    .unwrap_err();
            }
        }
        let after = db.current_epoch().as_u64();
        if round % 4 < 2 {
            assert_eq!(after, before + 1, "round {round} committed once");
        }
    }
    let per_epoch = events_per_epoch(&db);
    let counts: Vec<usize> = per_epoch.values().map(Vec::len).collect();
    assert_eq!(counts, expected_writes, "the events of each commit");
    assert_eq!(
        names(&db, "Gone"),
        Vec::<Value>::new(),
        "no rolled back write is there"
    );
    assert_eq!(names(&db, "Kept").len(), 10 + 10 * 3);
}

/// A commit that wrote a named graph, then one that writes the default
/// graph: the second reports its write in its own graph alone, not in the
/// first one's (the set's graph table is emptied with it).
#[test]
fn a_reused_set_starts_without_the_graphs_of_the_last_one() {
    let db = db();
    let session = db.session();
    session.execute("CREATE GRAPH other").unwrap();
    session
        .execute("USE other INSERT (:Kept {name: 'Mia'})")
        .unwrap();
    session.execute("INSERT (:Kept {name: 'Butch'})").unwrap();
    let per_epoch = events_per_epoch(&db);
    let last: Vec<&ChangeEvent> = per_epoch.values().last().unwrap().iter().collect();
    assert_eq!(last.len(), 1, "{last:?}");
    assert_eq!(last[0].graph, None, "the default graph: {last:?}");
    let previous: Vec<&ChangeEvent> = per_epoch.values().rev().nth(1).unwrap().iter().collect();
    assert_eq!(previous.len(), 1, "{previous:?}");
    assert_eq!(previous[0].graph.as_deref(), Some("other"), "{previous:?}");
}

/// A large transaction, then small ones: each small commit reports its own
/// write alone, whether the large set was kept or dropped.
#[test]
fn after_a_large_transaction_the_next_commits_report_their_own_writes() {
    let db = db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("FOR i IN range(1, 6000) INSERT (:Bulk {name: 'Shosanna-' + toString(i)})")
        .unwrap();
    session.commit().unwrap();
    for round in 0..3 {
        session
            .execute(&format!("INSERT (:Kept {{name: 'Hans-{round}'}})"))
            .unwrap();
    }
    let per_epoch = events_per_epoch(&db);
    let counts: Vec<usize> = per_epoch.values().map(Vec::len).collect();
    assert_eq!(counts, vec![6000, 1, 1, 1]);
}
