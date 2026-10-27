//! A direct call must undo completed writes if a later step fails or panics.

#![cfg(all(feature = "gql", feature = "testing-statement-injection"))]

use grafeo_common::testing::commit_hook::after_next_data_write;
use grafeo_common::testing::statement_failure::InjectedFailure;
use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_core::graph::Direction;
use grafeo_engine::GrafeoDB;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct Graph {
    center: NodeId,
    other: NodeId,
    edges: Vec<EdgeId>,
}

fn graph(db: &GrafeoDB) -> Graph {
    let center = db
        .create_node_with_props(&["Center"], [("name", Value::from("center"))])
        .unwrap();
    let other = db
        .create_node_with_props(&["Other"], [("name", Value::from("other"))])
        .unwrap();
    let edges = [
        (center, other),
        (center, other),
        (other, center),
        (center, center),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (src, dst))| {
        db.create_edge_with_props(
            src,
            dst,
            "LINK",
            [("weight", Value::Int64(i64::try_from(i).unwrap() + 1))],
        )
        .unwrap()
    })
    .collect();
    Graph {
        center,
        other,
        edges,
    }
}

fn unchanged(db: &GrafeoDB, graph: &Graph) {
    let mut actual = db
        .get_node_edges(graph.center, Direction::Both, &[])
        .unwrap();
    actual.sort_by_key(|(_, edge)| edge.as_u64());
    let mut expected = vec![
        (graph.other, graph.edges[0]),
        (graph.other, graph.edges[1]),
        (graph.other, graph.edges[2]),
        (graph.center, graph.edges[3]),
    ];
    expected.sort_by_key(|(_, edge)| edge.as_u64());
    assert_eq!(
        actual, expected,
        "all parallel edges and the self-loop remain"
    );
    assert!(
        db.get_node(graph.center)
            .unwrap()
            .unwrap()
            .has_label("Center")
    );
    assert_eq!(db.node_count().unwrap(), 2);
    assert_eq!(db.edge_count().unwrap(), 4);
    for (index, id) in graph.edges.iter().enumerate() {
        let edge = db.get_edge(*id).unwrap().expect("original edge remains");
        assert_eq!(
            edge.get_property("weight"),
            Some(&Value::Int64(i64::try_from(index).unwrap() + 1))
        );
    }
}

fn fail_after(writes: usize, panic: bool) {
    after_next_data_write(move || {
        if writes > 1 {
            fail_after(writes - 1, panic);
            Ok(())
        } else if panic {
            panic!("injected data-write panic");
        } else {
            Err(InjectedFailure)
        }
    });
}

#[test]
fn direct_detach_error_restores_every_completed_step() {
    for after in 1..=5 {
        let db = GrafeoDB::new_in_memory();
        let graph = graph(&db);
        let before = db.current_epoch();
        fail_after(after, false);
        let error = db.detach_delete_node(graph.center).unwrap_err();
        assert!(
            error.to_string().contains("injected data-write failure"),
            "{error}"
        );
        unchanged(&db, &graph);
        assert_eq!(
            db.current_epoch(),
            before,
            "a failed call publishes nothing"
        );
        assert!(db.detach_delete_node(graph.center).unwrap());
        assert_eq!(db.node_count().unwrap(), 1);
        assert_eq!(db.edge_count().unwrap(), 0);
    }
}

#[test]
fn direct_detach_panic_restores_every_completed_step() {
    for after in 1..=5 {
        let db = GrafeoDB::new_in_memory();
        let graph = graph(&db);
        let before = db.current_epoch();
        fail_after(after, true);
        let panic =
            catch_unwind(AssertUnwindSafe(|| db.detach_delete_node(graph.center))).unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"injected data-write panic")
        );
        unchanged(&db, &graph);
        assert_eq!(
            db.current_epoch(),
            before,
            "an unwound call publishes nothing"
        );
        assert!(db.detach_delete_node(graph.center).unwrap());
        assert_eq!(db.node_count().unwrap(), 1);
        assert_eq!(db.edge_count().unwrap(), 0);
    }
}

#[test]
fn failed_immediate_create_leaves_no_node_and_never_reuses_its_id() {
    let db = GrafeoDB::new_in_memory();
    let previous = db.create_node(&["Before"]).unwrap();
    let before = db.current_epoch();
    fail_after(1, false);
    assert!(
        db.create_node_with_props(&["Failed"], [("value", Value::Int64(7))])
            .is_err()
    );
    assert_eq!(db.node_count().unwrap(), 1);
    assert_eq!(db.current_epoch(), before);
    assert_eq!(
        db.execute("MATCH (n:Failed) RETURN n")
            .unwrap()
            .rows()
            .len(),
        0
    );
    let next = db.create_node(&["After"]).unwrap();
    assert!(next.as_u64() > previous.as_u64() + 1);
    assert_eq!(db.node_count().unwrap(), 2);
}

#[test]
fn failed_immediate_property_map_restores_values_and_index() {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    let db = GrafeoDB::new_in_memory();
    let id = db
        .create_node_with_props(
            &["Person"],
            [("age", Value::Int64(5)), ("name", Value::from("Alix"))],
        )
        .unwrap();
    db.create_property_index("age").unwrap();
    let before = db.current_epoch();
    for after in 1..=2 {
        fail_after(after, false);
        let map = Value::Map(Arc::new(BTreeMap::from([
            ("age".into(), Value::Int64(9)),
            ("name".into(), Value::from("Changed")),
        ])));
        assert!(db.set_node_property(id, "*", map).is_err());
        let node = db.get_node(id).unwrap().unwrap();
        assert_eq!(node.get_property("age"), Some(&Value::Int64(5)));
        assert_eq!(node.get_property("name"), Some(&Value::from("Alix")));
        assert_eq!(
            db.find_nodes_by_property("age", &Value::Int64(5)).unwrap(),
            [id]
        );
        let changed = db.find_nodes_by_property("age", &Value::Int64(9)).unwrap();
        assert!(changed.is_empty(), "{changed:?}");
        assert_eq!(db.current_epoch(), before);
    }
    db.set_node_property(id, "age", Value::Int64(6)).unwrap();
    assert_eq!(
        db.find_nodes_by_property("age", &Value::Int64(6)).unwrap(),
        [id]
    );
}

#[test]
fn failed_immediate_label_change_restores_label_counts() {
    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Before"]).unwrap();
    let before = db.current_epoch();
    fail_after(1, false);
    assert!(db.add_node_label(id, "Failed").is_err());
    let node = db.get_node(id).unwrap().unwrap();
    assert!(node.has_label("Before"));
    assert!(!node.has_label("Failed"));
    assert_eq!(db.current_epoch(), before);
    db.add_node_label(id, "After").unwrap();
    assert_eq!(
        db.execute("MATCH (n:Failed) RETURN count(n)")
            .unwrap()
            .rows(),
        &[vec![Value::Int64(0)]]
    );
    assert_eq!(
        db.execute("MATCH (n:After) RETURN count(n)")
            .unwrap()
            .rows(),
        &[vec![Value::Int64(1)]]
    );
}

#[cfg(feature = "wal")]
#[test]
fn failed_direct_detach_reopens_without_an_intervening_write() {
    use grafeo_engine::{Config, HistoryRetention};

    #[cfg(feature = "cdc")]
    use grafeo_common::types::EpochId;

    for retention in [HistoryRetention::Off, HistoryRetention::Forever] {
        for panic in [false, true] {
            for after in 1..=5 {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("graph.grafeo");
                let db = GrafeoDB::with_config(
                    Config::persistent(&path).with_history_retention(retention),
                )
                .unwrap();
                #[cfg(feature = "cdc")]
                db.set_cdc_enabled(true);
                let graph = graph(&db);
                let before = db.current_epoch();
                #[cfg(feature = "cdc")]
                let cdc_before = {
                    let events = db
                        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                        .unwrap();
                    assert_eq!(events.len(), 6, "both nodes and all four edges emit events");
                    serde_json::to_value(events).unwrap()
                };

                fail_after(after, panic);
                if panic {
                    let payload =
                        catch_unwind(AssertUnwindSafe(|| db.detach_delete_node(graph.center)))
                            .unwrap_err();
                    assert_eq!(
                        payload.downcast_ref::<&str>(),
                        Some(&"injected data-write panic"),
                        "the requested panic fires after step {after}"
                    );
                } else {
                    let error = db.detach_delete_node(graph.center).unwrap_err();
                    assert!(
                        error.to_string().contains("injected data-write failure"),
                        "the requested error fires after step {after}: {error}"
                    );
                }
                unchanged(&db, &graph);
                assert_eq!(
                    db.current_epoch(),
                    before,
                    "failed step {after}, panic={panic}, retention={retention:?}"
                );
                #[cfg(feature = "cdc")]
                assert_eq!(
                    serde_json::to_value(
                        db.changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                            .unwrap()
                    )
                    .unwrap(),
                    cdc_before,
                    "a failed call exposes no CDC events: step {after}, panic={panic}, retention={retention:?}"
                );

                // Close checkpoints immediately: no successful write can advance
                // the published epoch and conceal a failed call's store epoch.
                db.close().unwrap();
                drop(db);
                let reopened = GrafeoDB::open(&path).unwrap();
                unchanged(&reopened, &graph);
                assert_eq!(reopened.config().history_retention(), retention);
                assert_eq!(
                    reopened.current_epoch(),
                    before,
                    "reopen retains the published epoch: step {after}, panic={panic}, retention={retention:?}"
                );
                assert!(reopened.detach_delete_node(graph.center).unwrap());
                assert_eq!(reopened.node_count().unwrap(), 1);
                assert_eq!(reopened.edge_count().unwrap(), 0);
                reopened.close().unwrap();
            }
        }
    }
}

#[cfg(feature = "wal")]
#[test]
fn successful_direct_graph_reopens_at_its_published_epoch() {
    use grafeo_engine::{Config, HistoryRetention};

    for retention in [HistoryRetention::Off, HistoryRetention::Forever] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.grafeo");
        let db = GrafeoDB::with_config(Config::persistent(&path).with_history_retention(retention))
            .unwrap();
        let graph = graph(&db);
        let before = db.current_epoch();
        assert_eq!(before.as_u64(), 6, "the six direct calls committed");
        unchanged(&db, &graph);

        db.close().unwrap();
        drop(db);
        let reopened = GrafeoDB::open(&path).unwrap();
        unchanged(&reopened, &graph);
        assert_eq!(reopened.config().history_retention(), retention);
        assert_eq!(
            reopened.current_epoch(),
            before,
            "a successful graph reopens at its published epoch: retention={retention:?}"
        );
        reopened.close().unwrap();
    }
}
