//! Small API results (0.6.1): `PreparedCommit::commit` returns the epoch its
//! own commit got, also while other transactions commit; a CDC event of a
//! triple in a named RDF graph names that graph in `graph`, as every event
//! names its graph; and `list_graphs` gives the same order on every run.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test small_api_results
//! ```

#![cfg(feature = "gql")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::config::{Config, HistoryRetention};

/// A database that keeps every state, so a read at an earlier epoch sees it,
/// with one commit before the test's own.
fn with_history() -> GrafeoDB {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    db.execute("INSERT (:Start)").unwrap();
    db
}

/// Whether the node `(:P {i: i})` exists at `epoch`.
fn visible(db: &GrafeoDB, i: i64, epoch: EpochId) -> bool {
    let rows = db
        .execute_at_epoch(
            &format!("MATCH (p:P {{i: {i}}}) RETURN count(p) AS n"),
            epoch,
        )
        .unwrap()
        .rows()
        .to_vec();
    rows[0][0] == Value::Int64(1)
}

#[test]
fn a_prepared_commit_returns_its_own_epoch() {
    let db = with_history();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:P {i: 0})").unwrap();
    let epoch = session.prepare_commit().unwrap().commit().unwrap();
    assert_eq!(epoch, db.current_epoch(), "nothing else committed");
    assert!(visible(&db, 0, epoch));
    assert!(!visible(&db, 0, EpochId::new(epoch.as_u64() - 1)));
}

#[test]
fn a_prepared_commit_returns_its_own_epoch_while_others_commit() {
    // Four writers commit as fast as they can while 300 prepared commits
    // run: each returned epoch is the one its node appeared at, so the node
    // is there at that epoch and not at the one before. Returning the
    // current epoch after the commit gave a later one whenever a writer
    // committed in between, and the node is then there at epoch - 1 too.
    let db = Arc::new(with_history());
    let stop = Arc::new(AtomicBool::new(false));
    let writers: Vec<_> = (0..4)
        .map(|_| {
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    db.execute("INSERT (:W)").unwrap();
                }
            })
        })
        .collect();
    let mut returned = Vec::new();
    for i in 0..300 {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute(&format!("INSERT (:P {{i: {i}}})")).unwrap();
        returned.push((i, session.prepare_commit().unwrap().commit().unwrap()));
    }
    stop.store(true, Ordering::Relaxed);
    for writer in writers {
        writer.join().unwrap();
    }
    let wrong: Vec<(i64, EpochId)> = returned
        .iter()
        .copied()
        .filter(|&(i, epoch)| {
            !visible(&db, i, epoch) || visible(&db, i, EpochId::new(epoch.as_u64() - 1))
        })
        .collect();
    assert_eq!(
        wrong.len(),
        0,
        "{} of 300 returned an epoch other than their own, the first: {:?}",
        wrong.len(),
        wrong.first()
    );
    assert!(
        returned.windows(2).all(|pair| pair[0].1 < pair[1].1),
        "the epochs of later commits are later"
    );
}

#[cfg(all(feature = "cdc", feature = "sparql", feature = "triple-store"))]
#[test]
fn a_cdc_event_of_a_triple_in_a_named_graph_names_its_graph() {
    use grafeo_engine::cdc::{ChangeKind, EntityId};

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute_sparql(
        "INSERT DATA { <http://ex.org/alix> <http://ex.org/name> \"Alix\" . \
         GRAPH <http://ex.org/g1> { <http://ex.org/gus> <http://ex.org/name> \"Gus\" } }",
    )
    .unwrap();
    db.execute_sparql(
        "DELETE DATA { GRAPH <http://ex.org/g1> { <http://ex.org/gus> <http://ex.org/name> \"Gus\" } }",
    )
    .unwrap();
    let events = db
        .changes_between(EpochId::new(0), db.current_epoch())
        .unwrap();
    let triples: Vec<_> = events
        .iter()
        .filter(|event| matches!(event.entity_id, EntityId::Triple(_)))
        .collect();
    assert_eq!(triples.len(), 3, "{triples:?}");
    for event in &triples {
        assert_eq!(event.graph, event.triple_graph, "{event:?}");
    }
    let default = triples
        .iter()
        .find(|event| event.triple_subject.as_deref() == Some("<http://ex.org/alix>"))
        .unwrap();
    assert_eq!(default.graph, None);
    let named: Vec<_> = triples
        .iter()
        .filter(|event| event.triple_subject.as_deref() == Some("<http://ex.org/gus>"))
        .collect();
    assert_eq!(named.len(), 2);
    assert_eq!(named[0].kind, ChangeKind::Create);
    assert_eq!(named[1].kind, ChangeKind::Delete);
    for event in named {
        assert_eq!(
            event.graph.as_deref(),
            Some("http://ex.org/g1"),
            "{event:?}"
        );
    }
    // The default graph's history holds its own triple, not the named one.
    let history = db.history(default.entity_id).unwrap();
    assert_eq!(history.len(), 1);
}

#[test]
fn list_graphs_gives_the_same_sorted_order_on_every_run() {
    let names = [
        "prague",
        "amsterdam",
        "paris",
        "berlin",
        "barcelona",
        "lisbon",
        "vienna",
    ];
    let mut sorted: Vec<String> = names.iter().map(ToString::to_string).collect();
    sorted.sort();
    for run in 0..20 {
        let db = GrafeoDB::new_in_memory();
        for name in names {
            db.execute(&format!("CREATE GRAPH {name}")).unwrap();
        }
        assert_eq!(db.list_graphs(), sorted, "run {run}");
    }
}
