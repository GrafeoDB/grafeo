//! Captures include successful writes through the documented raw store handle.

use grafeo_common::types::{EdgeId, EpochId, NodeId, ReadView, Value};
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::{Direction, GraphStore};
use grafeo_engine::GrafeoDB;

#[derive(Clone, Copy)]
struct Ids {
    first: NodeId,
    second: NodeId,
    edge: EdgeId,
}

#[derive(Clone, Copy)]
struct GraphData {
    label: &'static str,
    names: [&'static str; 2],
    ages: [i64; 2],
    edge_type: &'static str,
    weight: i64,
}

const ROOT: GraphData = GraphData {
    label: "Person",
    names: ["Alix", "Gus"],
    ages: [31, 41],
    edge_type: "KNOWS",
    weight: 3,
};
const NAMED: GraphData = GraphData {
    label: "City",
    names: ["Berlin", "Paris"],
    ages: [60, 61],
    edge_type: "ROUTE",
    weight: 7,
};

struct Fixture {
    root: Ids,
    named: Ids,
    epoch: EpochId,
}

fn raw_graph(store: &RowGroupStore, data: GraphData) -> Ids {
    let first = store.create_node(&[data.label]);
    store.set_node_property(first, "name", Value::from(data.names[0]));
    store.set_node_property(first, "age", Value::Int64(data.ages[0]));
    let second = store.create_node(&[data.label]);
    store.set_node_property(second, "name", Value::from(data.names[1]));
    store.set_node_property(second, "age", Value::Int64(data.ages[1]));
    let edge = store.create_edge(first, second, data.edge_type);
    store.set_edge_property(edge, "weight", Value::Int64(data.weight));
    Ids {
        first,
        second,
        edge,
    }
}

fn raw_fixture(db: &GrafeoDB) -> Fixture {
    assert!(db.create_graph("archive").unwrap());
    let root = db.store().unwrap();
    let named = root.graph("archive").unwrap();
    let root_ids = raw_graph(&root, ROOT);
    let named_ids = raw_graph(&named, NAMED);
    // The named graph must determine the capture epoch, rather than the root
    // store alone. These are successful raw writes, with a literal final age.
    for age in 52..=60 {
        named.set_node_property(named_ids.first, "age", Value::Int64(age));
    }
    let before_noop = named.current_epoch();
    assert_graph(&named, named_ids, NAMED, before_noop, (2, 1));
    assert!(
        named
            .remove_node_property(named_ids.first, "missing")
            .unwrap()
            .is_none()
    );
    let epoch = named.current_epoch();
    assert_eq!(
        epoch,
        before_noop.next(),
        "a successful raw no-op advances its epoch"
    );
    assert_graph(&named, named_ids, NAMED, epoch, (2, 1));
    assert!(epoch > root.current_epoch());
    Fixture {
        root: root_ids,
        named: named_ids,
        epoch,
    }
}

fn assert_graph(
    store: &RowGroupStore,
    ids: Ids,
    data: GraphData,
    epoch: EpochId,
    counts: (usize, usize),
) {
    let view = ReadView::committed(epoch);
    assert_eq!(
        (
            store.node_count(view).unwrap(),
            store.edge_count(view).unwrap()
        ),
        counts
    );
    for (id, name, age) in [
        (ids.first, data.names[0], data.ages[0]),
        (ids.second, data.names[1], data.ages[1]),
    ] {
        let node = store.node(id, view).unwrap().expect("captured raw node");
        assert_eq!(node.id, id);
        assert_eq!(node.labels.len(), 1);
        assert!(node.has_label(data.label));
        assert_eq!(node.properties.len(), 2);
        assert_eq!(node.get_property("name"), Some(&Value::from(name)));
        assert_eq!(node.get_property("age"), Some(&Value::Int64(age)));
    }
    let edge = store
        .edge(ids.edge, view)
        .unwrap()
        .expect("captured raw edge");
    assert_eq!(
        (edge.id, edge.src, edge.dst),
        (ids.edge, ids.first, ids.second)
    );
    assert_eq!(edge.edge_type.as_str(), data.edge_type);
    assert_eq!(edge.properties.len(), 1);
    assert_eq!(
        edge.get_property("weight"),
        Some(&Value::Int64(data.weight))
    );
    assert_eq!(
        store.edges(ids.first, Direction::Both, &[], view).unwrap(),
        [(ids.second, ids.edge)]
    );
}

fn assert_captured(db: &GrafeoDB, fixture: &Fixture) {
    let root = db.store().unwrap();
    assert_graph(&root, fixture.root, ROOT, db.current_epoch(), (2, 1));
    assert_graph(
        &root.graph("archive").unwrap(),
        fixture.named,
        NAMED,
        db.current_epoch(),
        (2, 1),
    );
    assert_eq!(
        db.current_epoch(),
        fixture.epoch,
        "the captured epoch includes both raw stores"
    );
}

fn later_engine_write(db: &GrafeoDB, fixture: &Fixture) -> (NodeId, EdgeId, EpochId) {
    let node = db
        .create_node_with_props(
            &["Person"],
            [("name", Value::from("Mia")), ("age", Value::Int64(71))],
        )
        .unwrap();
    let node_epoch = db.current_epoch();
    assert!(
        node_epoch > fixture.epoch,
        "a later WAL write must be beyond the captured checkpoint"
    );
    let edge = db
        .create_edge_with_props(
            fixture.root.second,
            node,
            "FOLLOWS",
            [("weight", Value::Int64(11))],
        )
        .unwrap();
    let epoch = db.current_epoch();
    assert!(epoch > node_epoch);
    (node, edge, epoch)
}

#[test]
fn export_import_keeps_raw_root_and_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let fixture = raw_fixture(&db);
    let imported = GrafeoDB::import_snapshot(&db.export_snapshot().unwrap()).unwrap();
    assert_captured(&imported, &fixture);
    assert_eq!(db.current_epoch(), fixture.epoch);
    later_engine_write(&db, &fixture);
    assert_captured(&imported, &fixture);
    later_engine_write(&imported, &fixture);
}

#[test]
fn to_memory_keeps_raw_root_and_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let fixture = raw_fixture(&db);
    let copied = db.to_memory().unwrap();
    assert_captured(&copied, &fixture);
    assert_eq!(db.current_epoch(), fixture.epoch);
    later_engine_write(&db, &fixture);
    // The copy remains independent of the later source write.
    assert_captured(&copied, &fixture);
    later_engine_write(&copied, &fixture);
}

#[cfg(feature = "wal")]
mod persistent {
    use super::*;
    use std::path::Path;

    fn assert_later(db: &GrafeoDB, fixture: &Fixture, later: (NodeId, EdgeId, EpochId)) {
        let (node, edge, epoch) = later;
        assert_eq!(db.current_epoch(), epoch);
        let root = db.store().unwrap();
        assert_graph(&root, fixture.root, ROOT, epoch, (3, 2));
        assert_graph(
            &root.graph("archive").unwrap(),
            fixture.named,
            NAMED,
            epoch,
            (2, 1),
        );
        let mia = db.get_node(node).unwrap().unwrap();
        assert_eq!(mia.id, node);
        assert!(mia.has_label("Person"));
        assert_eq!(mia.properties.len(), 2);
        assert_eq!(mia.get_property("name"), Some(&Value::from("Mia")));
        assert_eq!(mia.get_property("age"), Some(&Value::Int64(71)));
        let follows = db.get_edge(edge).unwrap().unwrap();
        assert_eq!(
            (follows.id, follows.src, follows.dst),
            (edge, fixture.root.second, node)
        );
        assert_eq!(follows.edge_type.as_str(), "FOLLOWS");
        assert_eq!(follows.properties.len(), 1);
        assert_eq!(follows.get_property("weight"), Some(&Value::Int64(11)));
    }

    fn reopen_twice(path: &Path, fixture: &Fixture, later: (NodeId, EdgeId, EpochId)) {
        for _ in 0..2 {
            let reopened = GrafeoDB::open(path).unwrap();
            assert_later(&reopened, fixture, later);
            reopened.close().unwrap();
        }
    }

    #[test]
    fn save_keeps_raw_data_and_later_engine_writes_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.grafeo");
        let saved_path = dir.path().join("saved.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        let fixture = raw_fixture(&db);
        db.save(&saved_path).unwrap();
        let saved = GrafeoDB::open(&saved_path).unwrap();
        assert_captured(&saved, &fixture);
        saved.close().unwrap();
        assert_eq!(db.current_epoch(), fixture.epoch);
        let later = later_engine_write(&db, &fixture);
        db.close().unwrap();
        reopen_twice(&path, &fixture, later);
        let saved = GrafeoDB::open(&saved_path).unwrap();
        assert_captured(&saved, &fixture);
        saved.close().unwrap();
    }

    #[test]
    fn checkpoint_adopts_raw_epoch_before_later_wal_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        let fixture = raw_fixture(&db);
        db.wal_checkpoint().unwrap();
        assert_captured(&db, &fixture);
        let later = later_engine_write(&db, &fixture);
        db.close().unwrap();
        reopen_twice(&path, &fixture, later);
    }

    #[test]
    fn close_without_an_earlier_capture_keeps_raw_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("close.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        let fixture = raw_fixture(&db);
        db.close().unwrap();
        for _ in 0..2 {
            let reopened = GrafeoDB::open(&path).unwrap();
            assert_captured(&reopened, &fixture);
            reopened.close().unwrap();
        }
    }
}
