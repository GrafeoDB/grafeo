//! One timestamp rule for change events (#679): a commit that reports events reserves its
//! timestamps once, `n` consecutive raw HLC values for its `n` writes, and write `i`, in applied
//! order, gets `first + i`. A write folded into an earlier create (a change to an entity the same
//! transaction made) uses its value up, so the next event's timestamp skips it. Timestamps of a
//! later commit are larger, also with eight sessions committing at once, and the direct API
//! follows the same rule.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_timestamps::
//! ```

#![cfg(all(feature = "cdc", feature = "gql"))]

use std::sync::Arc;

use grafeo_common::types::EpochId;
use grafeo_engine::cdc::{ChangeEvent, ChangeKind};
use grafeo_engine::{Config, GrafeoDB};

fn db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
}

fn events(db: &GrafeoDB) -> Vec<ChangeEvent> {
    db.changes_between(EpochId::new(0), db.current_epoch())
        .unwrap()
}

/// The events of the latest commit, by timestamp.
fn last_commit(db: &GrafeoDB) -> Vec<ChangeEvent> {
    let all = events(db);
    let epoch = all.iter().map(|e| e.epoch).max().unwrap();
    let mut mine: Vec<ChangeEvent> = all.into_iter().filter(|e| e.epoch == epoch).collect();
    mine.sort_by_key(|e| e.timestamp);
    mine
}

fn raw(event: &ChangeEvent) -> u64 {
    event.timestamp.as_u64()
}

/// Three writes in one transaction: three consecutive raw values, in the order applied.
#[test]
fn a_commits_writes_take_one_run_in_applied_order() {
    let db = db();
    db.execute("INSERT (:P {name: 'Alix'}), (:P {name: 'Gus'})")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:P {name: 'Alix'}) SET a.age = 30")
        .unwrap();
    session
        .execute("MATCH (g:P {name: 'Gus'}) SET g.age = 25")
        .unwrap();
    session
        .execute("MATCH (a:P {name: 'Alix'}), (g:P {name: 'Gus'}) INSERT (a)-[:K]->(g)")
        .unwrap();
    session.commit().unwrap();
    let commit = last_commit(&db);
    assert_eq!(commit.len(), 3, "{commit:?}");
    let first = raw(&commit[0]);
    assert_eq!(
        commit.iter().map(raw).collect::<Vec<_>>(),
        [first, first + 1, first + 2],
        "one run of consecutive values"
    );
    assert_eq!(commit[0].kind, ChangeKind::Update);
    assert_eq!(commit[1].kind, ChangeKind::Update);
    assert_eq!(commit[2].kind, ChangeKind::Create);
    assert!(
        commit[2].edge_type.is_some(),
        "the edge, written last, comes last"
    );
}

/// A change to a node the transaction made folds into its create, and its value is used up:
/// create, update (folded), create gives the creates `first` and `first + 2`.
#[test]
fn a_folded_write_uses_up_its_timestamp() {
    let db = db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Mia)").unwrap();
    session.execute("MATCH (m:Mia) SET m.x = 1").unwrap();
    session.execute("INSERT (:Hans)").unwrap();
    session.commit().unwrap();
    let commit = last_commit(&db);
    assert_eq!(
        commit.len(),
        2,
        "the update folds into the create: {commit:?}"
    );
    assert!(commit.iter().all(|e| e.kind == ChangeKind::Create));
    assert_eq!(raw(&commit[1]), raw(&commit[0]) + 2, "{commit:?}");
}

/// Every timestamp of a later commit is above every timestamp of an earlier one.
#[test]
fn a_later_commit_has_larger_timestamps() {
    let db = db();
    db.execute("INSERT (:P {name: 'Alix'}), (:P {name: 'Gus'}), (:P {name: 'Vincent'})")
        .unwrap();
    let earlier = last_commit(&db);
    db.execute("MATCH (p:P) SET p.seen = true").unwrap();
    let later = last_commit(&db);
    assert!(
        raw(earlier.last().unwrap()) < raw(&later[0]),
        "{earlier:?} then {later:?}"
    );
}

/// Eight sessions committing at once: within a commit one run, and across commits the
/// timestamps rise with the epochs.
#[test]
fn eight_sessions_at_once_keep_one_run_per_commit_rising_with_epochs() {
    let db = Arc::new(db());
    let handles: Vec<_> = (0..8)
        .map(|thread| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                for i in 0..50 {
                    db.execute(&format!(
                        "INSERT (:T {{t: {thread}, i: {i}}}), (:T {{t: {thread}}})"
                    ))
                    .unwrap();
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let mut by_epoch: std::collections::BTreeMap<EpochId, Vec<u64>> = Default::default();
    for event in events(&db) {
        by_epoch.entry(event.epoch).or_default().push(raw(&event));
    }
    assert_eq!(by_epoch.len(), 400, "one commit per statement");
    let mut previous_high = 0;
    for (epoch, mut stamps) in by_epoch {
        stamps.sort_unstable();
        assert_eq!(stamps.len(), 2, "epoch {epoch:?}");
        assert_eq!(stamps[1], stamps[0] + 1, "epoch {epoch:?}: one run");
        assert!(
            stamps[0] > previous_high,
            "epoch {epoch:?} rises above the one before"
        );
        previous_high = stamps[1];
    }
}

/// The direct API's commits follow the same rule.
#[test]
fn the_direct_api_follows_the_rule() {
    let db = db();
    let alix = db.create_node(&["P"]).unwrap();
    db.set_node_property(alix, "name", "Alix".into()).unwrap();
    let gus = db.create_node(&["P"]).unwrap();
    db.create_edge(alix, gus, "K").unwrap();
    let mut all = events(&db);
    all.sort_by_key(|e| e.timestamp);
    assert_eq!(all.len(), 4, "{all:?}");
    assert!(all.windows(2).all(|pair| raw(&pair[0]) < raw(&pair[1])));
    assert!(all.windows(2).all(|pair| pair[0].epoch < pair[1].epoch));
}
