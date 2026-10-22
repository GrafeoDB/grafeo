//! The direct write API (`create_node`, `set_node_property`, ...) writes
//! through the same path as a query: each call is one transaction on the
//! current graph, checked against the schema and constraints, versioned and
//! reported to CDC, and it applies completely or fails with an error.

use std::collections::HashMap;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// Sorted `n.id` of every node in the database's current graph.
fn ids(db: &GrafeoDB) -> Vec<String> {
    db.execute("MATCH (n) RETURN n.id ORDER BY n.id")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            other => panic!("unexpected id {other:?}"),
        })
        .collect()
}

#[test]
fn writes_go_to_the_selected_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH model").unwrap();
    db.set_current_graph(Some("model")).unwrap();

    let component = db
        .create_node_with_props(&["Component"], [("id", Value::from("c1"))])
        .unwrap();
    let data = db.create_node(&["Data"]).unwrap();
    db.set_node_property(data, "id", Value::from("d1")).unwrap();
    db.set_node_property(component, "name", Value::from("Billing"))
        .unwrap();
    assert!(db.add_node_label(component, "Service").unwrap());
    let access = db
        .create_edge_with_props(component, data, "ACCESS", [("id", Value::from("r1"))])
        .unwrap();
    db.set_edge_property(access, "weight", Value::from(2_i64))
        .unwrap();

    assert_eq!(ids(&db), ["c1", "d1"]);
    let result = db
        .execute("MATCH (a:Service)-[r:ACCESS]->(b) RETURN a.name, r.id, r.weight, b.id")
        .unwrap();
    assert_eq!(
        result.rows(),
        [vec![
            Value::from("Billing"),
            Value::from("r1"),
            Value::from(2_i64),
            Value::from("d1"),
        ]]
    );
    assert!(db.get_node(component).is_some());
    assert!(db.get_edge(access).is_some());

    // The default graph got none of it.
    db.set_current_graph(None).unwrap();
    assert!(ids(&db).is_empty(), "expected empty");
    assert!(db.get_node(component).is_none());
}

#[test]
fn property_index_calls_follow_the_selected_graph() {
    let db = GrafeoDB::new_in_memory();
    let in_default = db
        .create_node_with_props(&["Doc"], [("id", Value::from("x"))])
        .unwrap();
    db.execute("CREATE GRAPH model").unwrap();
    db.set_current_graph(Some("model")).unwrap();
    let in_model = db
        .create_node_with_props(&["Doc"], [("id", Value::from("x"))])
        .unwrap();
    db.create_property_index("id").unwrap();
    assert!(db.has_property_index("id"));
    assert_eq!(
        db.find_nodes_by_property("id", &Value::from("x")),
        [in_model]
    );

    db.set_current_graph(None).unwrap();
    assert!(!db.has_property_index("id"));
    assert_eq!(
        db.find_nodes_by_property("id", &Value::from("x")),
        [in_default]
    );
}

#[test]
fn writes_are_checked_against_constraints() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT person_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    db.execute("CREATE CONSTRAINT person_name FOR (n:Person) ON (n.name) NOT NULL")
        .unwrap();
    let alix = db
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::from("Alix")),
                ("email", Value::from("alix@example.org")),
            ],
        )
        .unwrap();
    let gus = db
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();

    let duplicate = db.create_node_with_props(
        &["Person"],
        [
            ("name", Value::from("Vincent")),
            ("email", Value::from("alix@example.org")),
        ],
    );
    assert!(duplicate.is_err(), "a duplicate UNIQUE value is rejected");
    assert!(
        db.set_node_property(gus, "email", Value::from("alix@example.org"))
            .is_err(),
        "setting a duplicate UNIQUE value is rejected"
    );
    assert!(
        db.remove_node_property(alix, "name").is_err(),
        "removing a NOT NULL property is rejected"
    );
    assert!(
        db.create_node(&["Person"]).is_err(),
        "a node without its NOT NULL property is rejected"
    );
    let guest = db
        .create_node_with_props(&["Guest"], [("email", Value::from("alix@example.org"))])
        .unwrap();
    assert!(
        db.add_node_label(guest, "Person").is_err(),
        "a label whose constraints the node breaks is rejected"
    );

    assert_eq!(
        db.execute("MATCH (n:Person) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(2)
    );
}

/// An edge type that declares its endpoints' labels holds for direct writes;
/// an undeclared edge type connects any nodes.
#[test]
fn edge_endpoint_types_are_checked() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE EDGE TYPE WORKS_AT CONNECTING (Person) TO (Company)")
        .unwrap();
    let alix = db.create_node(&["Person"]).unwrap();
    let company = db.create_node(&["Company"]).unwrap();
    let city = db.create_node(&["City"]).unwrap();

    assert!(db.create_edge(alix, company, "WORKS_AT").is_ok());
    assert!(
        db.create_edge(alix, city, "WORKS_AT").is_err(),
        "the target lacks the declared label"
    );
    assert!(
        db.create_edge(company, alix, "WORKS_AT").is_err(),
        "the source lacks the declared label"
    );
    assert!(db.create_edge(city, alix, "LIVES_IN").is_ok());
    assert_eq!(
        db.execute("MATCH ()-[r]->() RETURN count(r)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(2)
    );
}

/// With no schema at all a property value is still held to the size limit,
/// whichever way it is set, and a rejected write leaves the node as it was.
#[test]
fn the_size_limit_holds_without_a_schema() {
    let db = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_max_property_size(64))
        .unwrap();
    let alix = db
        .create_node_with_props(&["Person"], [("bio", Value::from("short"))])
        .unwrap();
    let long = "x".repeat(1000);

    let err = db
        .set_node_property(alix, "bio", Value::from(long.as_str()))
        .unwrap_err();
    assert!(err.to_string().contains("exceeds maximum size"), "{err}");
    for set in [
        format!("n.bio = '{long}'"),
        format!("n += {{bio: '{long}'}}"),
        format!("n = {{bio: '{long}'}}"),
    ] {
        let err = db
            .execute(&format!("MATCH (n:Person) SET {set}"))
            .unwrap_err();
        assert!(
            err.to_string().contains("exceeds maximum size"),
            "SET {}: {err}",
            &set[..10]
        );
    }
    assert_eq!(
        db.get_node(alix).unwrap().get_property("bio"),
        Some(&Value::from("short"))
    );
}

#[test]
fn a_failing_batch_creates_nothing() {
    // With a property index the UNIQUE check finds candidates through the
    // index, without one through the label: both see the batch's own nodes.
    for with_index in [false, true] {
        let db = GrafeoDB::new_in_memory();
        db.execute("CREATE CONSTRAINT doc_id FOR (n:Doc) ON (n.id) UNIQUE")
            .unwrap();
        if with_index {
            db.create_property_index("id").unwrap();
        }
        let row = |id: &str| HashMap::from([(PropertyKey::new("id"), Value::from(id))]);

        let err = db
            .batch_create_nodes_with_props("Doc", vec![row("a"), row("b"), row("a")])
            .unwrap_err();
        assert!(err.to_string().to_lowercase().contains("unique"), "{err}");
        assert_eq!(db.node_count(), 0, "with_index: {with_index}");
        assert!(
            db.find_nodes_by_property("id", &Value::from("a"))
                .is_empty(),
            "with_index: {with_index}"
        );

        let created = db
            .batch_create_nodes_with_props("Doc", vec![row("a"), row("b")])
            .unwrap();
        assert_eq!(created.len(), 2);
        assert!(
            db.create_node_with_props(&["Doc"], [("id", Value::from("b"))])
                .is_err(),
            "with_index: {with_index}"
        );
    }
}

#[test]
fn missing_entities_are_errors_or_false() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    db.create_edge(alix, gus, "KNOWS").unwrap();

    let missing = grafeo_common::types::NodeId::new(999);
    assert!(db.create_edge(alix, missing, "KNOWS").is_err());
    assert!(
        db.set_node_property(missing, "x", Value::from(1_i64))
            .is_err()
    );
    assert!(!db.delete_node(missing).unwrap());
    assert!(!db.add_node_label(missing, "Person").unwrap());
    // A node that still has edges is not deleted.
    assert!(db.delete_node(alix).is_err());
    assert!(db.get_node(alix).is_some());
}

#[test]
fn every_write_advances_the_epoch() {
    let db = GrafeoDB::new_in_memory();
    let start = db.current_epoch();
    let alix = db.create_node(&["Person"]).unwrap();
    let after_create = db.current_epoch();
    db.set_node_property(alix, "name", Value::from("Alix"))
        .unwrap();
    let after_set = db.current_epoch();
    assert!(after_create > start, "{after_create:?} after {start:?}");
    assert!(
        after_set > after_create,
        "{after_set:?} after {after_create:?}"
    );
}

/// A reader pinned at `current_epoch()` does not see nodes and edges written
/// or deleted after it: each direct write commits at a later epoch.
#[test]
fn a_pinned_epoch_does_not_see_later_writes() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    let knows = db.create_edge(alix, gus, "KNOWS").unwrap();
    let pinned = db.current_epoch();

    let vincent = db.create_node(&["Person"]).unwrap();
    let likes = db.create_edge(gus, vincent, "LIKES").unwrap();
    assert!(db.delete_edge(knows).unwrap());

    assert!(db.get_node_at_epoch(vincent, pinned).is_none());
    assert!(db.get_edge_at_epoch(likes, pinned).is_none());
    assert!(db.get_edge_at_epoch(knows, pinned).is_some());
    assert!(db.get_node_at_epoch(alix, pinned).is_some());
    assert!(db.get_edge(knows).is_none());
}

#[test]
fn session_writes_roll_back_with_the_transaction() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let alix = session
        .create_node_with_props(&["Person"], [("id", Value::from("alix"))])
        .unwrap();
    assert!(session.get_node(alix).is_some());
    session.rollback().unwrap();
    assert!(db.get_node(alix).is_none());
    assert_eq!(db.node_count(), 0);
}

#[cfg(feature = "cdc")]
#[test]
fn a_create_event_carries_the_labels_and_properties() {
    use grafeo_engine::cdc::ChangeKind;

    let db = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_cdc()).unwrap();
    let direct = db
        .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
        .unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let from_query = db.find_nodes_by_property("name", &Value::from("Gus"))[0];

    for node in [direct, from_query] {
        let history = db.history(node).unwrap();
        assert_eq!(history.len(), 1, "one create event: {history:?}");
        let create = &history[0];
        assert_eq!(create.kind, ChangeKind::Create);
        assert_eq!(create.labels.as_deref(), Some(&["Person".to_string()][..]));
        assert!(
            create
                .after
                .as_ref()
                .is_some_and(|after| after.contains_key("name"))
        );
    }
}

/// A direct write to a node that an open transaction has written is a
/// conflict, so the transaction's rollback cannot erase it; direct writes to
/// other nodes go through, and once the transaction ends so does the first.
#[test]
fn a_direct_write_conflicts_with_an_open_transaction() {
    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("city", Value::from("Amsterdam"))])
        .unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    let city = |id| db.get_node(id).unwrap().get_property("city").cloned();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(alix, "city", Value::from("Berlin"))
        .unwrap();
    assert!(
        db.set_node_property(alix, "city", Value::from("Paris"))
            .is_err(),
        "the transaction wrote Alix first"
    );
    db.set_node_property(gus, "city", Value::from("Prague"))
        .unwrap();
    session.rollback().unwrap();

    assert_eq!(city(alix), Some(Value::from("Amsterdam")));
    assert_eq!(city(gus), Some(Value::from("Prague")));
    db.set_node_property(alix, "city", Value::from("Paris"))
        .unwrap();
    assert_eq!(city(alix), Some(Value::from("Paris")));
}

/// Runs `attempt` until it succeeds, yielding between tries: a write
/// conflict that never clears fails the test instead of hanging it.
fn until_done(mut attempt: impl FnMut() -> bool) {
    for _ in 0..100_000 {
        if attempt() {
            return;
        }
        std::thread::yield_now();
    }
    panic!("a write conflict never cleared");
}

/// Direct writes from several threads next to transactions, all writing one
/// shared node too: nothing hangs, every committed write lands, nothing of a
/// rolled-back transaction remains, and a direct write that meets an open
/// transaction's write fails instead of being lost.
#[test]
fn direct_writes_and_transactions_run_side_by_side() {
    const THREADS: usize = 4;
    const ROUNDS: usize = 60;
    let db = GrafeoDB::new_in_memory();
    let hub = db.create_node(&["Hub"]).unwrap();

    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let db = &db;
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    until_done(|| {
                        let mut session = db.session();
                        session.begin_transaction().unwrap();
                        let kept = round % 3 != 0;
                        let label = if kept { "Kept" } else { "Dropped" };
                        session.create_node(&[label]).unwrap();
                        let wrote_hub = session
                            .set_node_property(hub, "by", Value::from(format!("tx {thread}")))
                            .is_ok();
                        if !kept {
                            session.rollback().unwrap();
                            return true;
                        }
                        if wrote_hub && session.commit().is_ok() {
                            return true;
                        }
                        let _ = session.rollback();
                        false
                    });
                }
            });
            scope.spawn(move || {
                for _ in 0..ROUNDS {
                    db.create_node(&["Mark"]).unwrap();
                    until_done(|| {
                        db.set_node_property(hub, "by", Value::from(format!("direct {thread}")))
                            .is_ok()
                    });
                }
            });
        }
    });

    let count = |label: &str| {
        db.execute(&format!("MATCH (n:{label}) RETURN count(n)"))
            .unwrap()
            .rows()[0][0]
            .clone()
    };
    let kept = (0..ROUNDS).filter(|round| round % 3 != 0).count() * THREADS;
    assert_eq!(count("Kept"), Value::Int64(i64::try_from(kept).unwrap()));
    assert_eq!(count("Dropped"), Value::Int64(0));
    assert_eq!(
        count("Mark"),
        Value::Int64(i64::try_from(THREADS * ROUNDS).unwrap())
    );
    assert!(db.get_node(hub).unwrap().get_property("by").is_some());
}

/// With versioned properties and labels, a reader pinned before a direct write
/// still sees the values and labels from before it.
#[cfg(feature = "temporal")]
#[test]
fn a_pinned_epoch_sees_the_properties_and_labels_it_had() {
    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(&["Person"], [("city", Value::from("Amsterdam"))])
        .unwrap();
    let pinned = db.current_epoch();
    db.set_node_property(alix, "city", Value::from("Berlin"))
        .unwrap();
    assert!(db.add_node_label(alix, "Employee").unwrap());

    assert_eq!(
        db.get_node_property_at_epoch(alix, "city", pinned),
        Some(Value::from("Amsterdam"))
    );
    assert!(
        !db.get_node_at_epoch(alix, pinned)
            .unwrap()
            .has_label("Employee")
    );
    assert_eq!(
        db.get_node_property_at_epoch(alix, "city", db.current_epoch()),
        Some(Value::from("Berlin"))
    );
    assert!(db.get_node(alix).unwrap().has_label("Employee"));
}

/// Each direct write is reported to CDC at the epoch it committed at.
#[cfg(feature = "cdc")]
#[test]
fn direct_writes_are_reported_at_their_own_epochs() {
    use grafeo_engine::cdc::ChangeKind;

    let db = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_cdc()).unwrap();
    let alix = db
        .create_node_with_props(&["Person"], [("city", Value::from("Amsterdam"))])
        .unwrap();
    let created = db.current_epoch();
    db.set_node_property(alix, "city", Value::from("Berlin"))
        .unwrap();
    let updated = db.current_epoch();

    let history: Vec<_> = db
        .history(alix)
        .unwrap()
        .into_iter()
        .map(|event| (event.kind, event.epoch))
        .collect();
    assert!(updated > created);
    assert_eq!(
        history,
        [(ChangeKind::Create, created), (ChangeKind::Update, updated)]
    );
}

/// A batch of edges is one transaction: each edge gets its own type and
/// properties, and an edge to a missing node fails the whole batch.
#[test]
fn a_batch_of_edges_is_all_or_nothing() {
    use grafeo_engine::database::BatchEdge;

    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    let vincent = db.create_node(&["Person"]).unwrap();
    let ids = db
        .batch_create_edges(vec![
            BatchEdge::new(alix, gus, "KNOWS").with_properties([("since", 2020_i64)]),
            BatchEdge::new(gus, vincent, "LIKES"),
        ])
        .unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(db.get_edge(ids[0]).unwrap().edge_type.as_str(), "KNOWS");
    assert_eq!(
        db.get_edge(ids[0]).unwrap().get_property("since"),
        Some(&Value::from(2020_i64))
    );
    assert_eq!(db.get_edge(ids[1]).unwrap().edge_type.as_str(), "LIKES");

    let missing = grafeo_common::types::NodeId::new(999);
    let err = db
        .batch_create_edges(vec![
            BatchEdge::new(alix, vincent, "KNOWS"),
            BatchEdge::new(alix, missing, "KNOWS"),
        ])
        .unwrap_err();
    assert!(
        matches!(err, grafeo_common::utils::error::Error::NodeNotFound(id) if id == missing),
        "{err}"
    );
    assert_eq!(
        db.execute("MATCH ()-[r]->() RETURN count(r)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(2),
        "the first edge of the failed batch is gone too"
    );
}

#[test]
fn a_batch_of_nodes_gets_every_label() {
    let db = GrafeoDB::new_in_memory();
    let row = |id: &str| HashMap::from([(PropertyKey::new("id"), Value::from(id))]);
    let ids = db
        .batch_create_nodes_with_labels(&["Graph", "File"], vec![row("f1"), row("f2")])
        .unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(
        db.execute("MATCH (n:Graph:File) RETURN count(n)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(2)
    );
}
