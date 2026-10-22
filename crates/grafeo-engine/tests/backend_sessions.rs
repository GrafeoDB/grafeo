//! A session on a backend of one's own (`GrafeoDB::with_store`) is set up
//! as one on the built-in store is: it reports to change data capture and
//! to the metrics, takes the database's time zone, and starts at its home;
//! what only the built-in store has (named graphs, schemas with their
//! default graph, indexes) is an error, never a stand-in that is empty.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test backend_sessions
//! ```

#![cfg(feature = "gql")]

use std::sync::Arc;

use grafeo_common::types::{EpochId, Value};
use grafeo_core::graph::GraphBackend;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_engine::{Config, GrafeoDB};

fn on_backend(config: Config) -> GrafeoDB {
    GrafeoDB::with_store(
        Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
        config,
    )
    .unwrap()
}

/// The commits of a session on a backend of one's own reach change data
/// capture, with what each write replaced.
#[cfg(feature = "cdc")]
#[test]
fn change_data_capture_hears_a_backend_of_ones_own() {
    let db = on_backend(Config::in_memory().with_cdc());
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.execute("MATCH (p:Person) SET p.name = 'Gus'").unwrap();
    let events = db
        .changes_between(EpochId::INITIAL, EpochId::new(u64::MAX))
        .unwrap();
    let kinds: Vec<String> = events
        .iter()
        .map(|event| format!("{:?}", event.kind))
        .collect();
    assert_eq!(kinds, ["Create", "Update"], "{events:?}");
    let update = &events[1];
    assert_eq!(
        update
            .before
            .as_ref()
            .and_then(|before| before.get("name"))
            .cloned(),
        Some(Value::from("Alix")),
        "the update carries what it replaced: {update:?}"
    );
}

/// A session on a backend of one's own counts in the database's metrics.
#[cfg(feature = "metrics")]
#[test]
fn the_metrics_count_the_sessions_of_a_backend_of_ones_own() {
    let db = on_backend(Config::in_memory());
    let before = db.metrics().session_created;
    let session = db.session();
    assert_eq!(db.metrics().session_created, before + 1);
    drop(session);
}

/// A new session takes the time zone a one-shot statement set on the
/// database, on a backend of one's own as on the built-in store.
#[test]
fn a_new_session_takes_the_databases_time_zone() {
    let db = on_backend(Config::in_memory());
    db.execute("SESSION SET TIME ZONE '+01:00'").unwrap();
    assert_eq!(db.session().time_zone().as_deref(), Some("+01:00"));
}

/// A backend of one's own has the default graph and no other: selecting a
/// named graph or a schema fails, and a session starts at the default graph
/// as its home, where `SESSION RESET` takes it back.
#[test]
fn a_backend_of_ones_own_has_the_default_graph_and_no_other() {
    let db = on_backend(Config::in_memory());
    let error = db
        .set_current_graph(Some("trips"))
        .expect_err("no named graph")
        .to_string();
    assert!(error.contains("does not exist"), "{error}");
    db.set_current_graph(Some("default")).unwrap();
    db.set_current_graph(None).unwrap();

    let error = db
        .execute("CREATE GRAPH trips")
        .expect_err("a named graph needs the built-in store")
        .to_string();
    assert!(error.contains("built-in store"), "{error}");
    let error = db
        .execute("CREATE SCHEMA travel")
        .expect_err("a schema's default graph needs the built-in store")
        .to_string();
    assert!(error.contains("built-in store"), "{error}");
    assert!(db.set_current_schema(Some("travel")).is_err());

    let session = db.session();
    assert_eq!(session.current_graph(), None);
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session.execute("SESSION RESET").unwrap();
    assert_eq!(session.current_graph(), None, "home is the default graph");
    assert_eq!(
        session
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .rows(),
        [[Value::from("Mia")]]
    );
}

/// A schema statement on a backend of one's own applies its catalog
/// records, and the type it declares checks the writes.
#[test]
fn a_node_type_checks_the_writes_to_a_backend_of_ones_own() {
    let db = on_backend(Config::in_memory());
    db.execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    let error = db
        .execute("INSERT (:Person {city: 'Paris'})")
        .expect_err("the name is required")
        .to_string();
    assert!(error.contains("name"), "{error}");
    db.execute("INSERT (:Person {name: 'Butch'})").unwrap();
}
