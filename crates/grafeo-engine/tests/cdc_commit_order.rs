//! Change data capture follows the commits: each commit's events are
//! recorded when it is published, with its epoch, and their timestamps
//! follow the epochs (a later commit's events never carry an earlier
//! timestamp). A write with auto-commit off is a commit of its own and
//! reports its events at once; a statement that fails reports nothing.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test cdc_commit_order
//! ```

#![cfg(all(feature = "cdc", feature = "gql"))]

use std::sync::Arc;

use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::cdc::{ChangeEvent, ChangeKind, EntityId};
use grafeo_engine::{Config, GrafeoDB};

fn db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
}

/// Every event recorded so far, in epoch order.
fn all_events(db: &GrafeoDB) -> Vec<ChangeEvent> {
    db.changes_between(EpochId::new(0), db.current_epoch())
        .unwrap()
}

/// The id of the one node `query` returns as `id`.
fn node_id(db: &GrafeoDB, query: &str) -> NodeId {
    let result = db.execute(query).unwrap();
    match result.rows()[0][0] {
        Value::Int64(id) => NodeId::new(u64::try_from(id).unwrap()),
        ref other => panic!("expected a node id, got {other:?}"),
    }
}

/// Eight committers at once, each with transactions of a session and
/// direct calls: every commit's events carry its epoch, and ordered by
/// timestamp the events are in epoch order, so a consumer reading by
/// timestamp sees the commits in the order they were published.
#[test]
fn cdc_events_follow_commit_order() {
    const COMMITTERS: u64 = 8;
    const ROUNDS: i64 = 88;

    let db = Arc::new(db());
    let nodes: Vec<NodeId> = (0..COMMITTERS)
        .map(|committer| {
            db.create_node_with_props(
                &["Counter"],
                [("committer", Value::Int64(i64::try_from(committer).unwrap()))],
            )
            .unwrap()
        })
        .collect();
    let start = db.current_epoch();

    let threads: Vec<_> = nodes
        .iter()
        .enumerate()
        .map(|(committer, &node)| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                let mut session = db.session();
                for round in 0..ROUNDS {
                    if committer % 2 == 0 {
                        session.begin_transaction().unwrap();
                        session
                            .set_node_property(node, "round", Value::Int64(round))
                            .unwrap();
                        session.commit().unwrap();
                    } else {
                        db.set_node_property(node, "round", Value::Int64(round))
                            .unwrap();
                    }
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }

    let events: Vec<ChangeEvent> = all_events(&db)
        .into_iter()
        .filter(|event| event.epoch > start)
        .collect();
    assert_eq!(
        events.len(),
        usize::try_from(COMMITTERS).unwrap() * usize::try_from(ROUNDS).unwrap(),
        "one event per committed write"
    );
    assert!(
        events.iter().all(|event| event.kind == ChangeKind::Update),
        "every write is an update of a counter"
    );

    let mut by_timestamp = events.clone();
    by_timestamp.sort_by_key(|event| event.timestamp);
    let out_of_order = by_timestamp
        .windows(2)
        .find(|pair| pair[0].epoch > pair[1].epoch);
    assert!(
        out_of_order.is_none(),
        "an event of a later commit has an earlier timestamp: {out_of_order:?}"
    );

    // Each counter's history lists its rounds in order, one epoch each.
    for node in nodes {
        let rounds: Vec<Value> = db
            .history(node)
            .unwrap()
            .into_iter()
            .filter(|event| event.epoch > start)
            .map(|event| event.after.unwrap()["round"].clone())
            .collect();
        assert_eq!(rounds, (0..ROUNDS).map(Value::Int64).collect::<Vec<_>>());
    }
}

/// With auto-commit off and no transaction open, a write statement commits
/// on its own: its event is recorded at once, at the epoch that made the
/// write visible, not left for the session's next commit.
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_write_with_auto_commit_off_reports_its_event_at_its_own_epoch() {
    let db = db();
    let mut session = db.session();
    session.set_auto_commit(false);
    session
        .execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
        .unwrap();
    let visible_at = db.current_epoch();
    let alix = node_id(&db, "MATCH (p:Person) RETURN id(p)");

    let history = db.history(alix).unwrap();
    assert_eq!(history.len(), 1, "one event for the insert: {history:?}");
    assert_eq!(history[0].kind, ChangeKind::Create);
    assert_eq!(
        history[0].epoch, visible_at,
        "the event carries the statement's own commit epoch"
    );

    session
        .execute("MATCH (p:Person) SET p.city = 'Berlin'")
        .unwrap();
    let history = db.history(alix).unwrap();
    assert_eq!(history.len(), 2, "{history:?}");
    assert!(
        history[1].epoch > visible_at,
        "each statement commits apart"
    );
}

/// A write statement that fails with auto-commit off reports nothing, and
/// leaves nothing for a later commit to report.
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_failed_statement_with_auto_commit_off_reports_nothing() {
    let db = db();
    db.execute("CREATE NODE TYPE Doc (id INTEGER)").unwrap();
    let mut session = db.session();
    session.set_auto_commit(false);
    session
        .execute("UNWIND [3, 'x'] AS v INSERT (:Doc {id: v})")
        .unwrap_err();
    session.execute("INSERT (:Doc {id: 19})").unwrap();

    let creates: Vec<EntityId> = all_events(&db)
        .into_iter()
        .filter(|event| event.kind == ChangeKind::Create)
        .map(|event| event.entity_id)
        .collect();
    let doc = node_id(&db, "MATCH (d:Doc) RETURN id(d)");
    assert_eq!(
        creates,
        [EntityId::Node(doc)],
        "only the statement that succeeded is reported"
    );
}
