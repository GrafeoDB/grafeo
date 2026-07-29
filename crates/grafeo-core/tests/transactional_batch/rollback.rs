//! Rollback / secondary-cleanup coverage for storage-level batch writes.
//!
//! Label and edge-type registries are append-only string intern tables: rolling
//! back a batch that introduced a new name does **not** free or reuse that
//! intern id. Row/index/adjacency/live-counter cleanup is in scope for rollback;
//! shrinking the intern tables is intentionally outside row rollback so stable
//! numeric ids never shift under concurrent readers.

use super::common::node;
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::types::{PropertyKey, TransactionId, Value};
use grafeo_core::graph::Direction;
use grafeo_core::graph::lpg::{BatchEdgeCreate, LpgStore};
use grafeo_core::index::ChunkedAdjacency;

#[test]
fn transactional_batch_rollback_discards_every_row() {
    let store = LpgStore::new().unwrap();
    let epoch = store.new_epoch();
    let tx = TransactionId::new(100);

    let input = vec![
        node(&["A"], vec![("k", Value::from(1i64))]),
        node(&["B"], vec![("k", Value::from(2i64))]),
        node(&["C"], vec![("k", Value::from(3i64))]),
    ];
    let ids = store.create_nodes_batch_versioned(&input, epoch, tx);
    assert_eq!(ids.len(), 3);

    store.discard_uncommitted_versions(tx);

    for id in &ids {
        assert!(
            store.get_node_versioned(*id, epoch, tx).is_none(),
            "rolled-back batch row must be gone"
        );
        assert!(store.get_node(*id).is_none());
    }
}

#[test]
fn transactional_batch_rollback_cleans_node_secondaries() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("name");
    let epoch = store.new_epoch();

    let baseline = store.create_nodes_batch_versioned(
        &[node(
            &["Person"],
            vec![("name", Value::from("keep")), ("age", Value::from(99i64))],
        )],
        epoch,
        TransactionId::SYSTEM,
    );
    let keep = baseline[0];

    let tx = TransactionId::new(200);
    let staged = store.create_nodes_batch_versioned(
        &[
            node(
                &["Person"],
                vec![("name", Value::from("gone")), ("age", Value::from(1i64))],
            ),
            node(
                &["Person", "Admin"],
                vec![("name", Value::from("gone2")), ("age", Value::from(2i64))],
            ),
        ],
        epoch,
        tx,
    );

    assert!(store.nodes_by_label("Person").contains(&staged[0]));
    assert!(store.nodes_by_label("Admin").contains(&staged[1]));
    assert!(
        store
            .find_nodes_by_property("name", &Value::from("gone"))
            .contains(&staged[0])
    );

    store.discard_uncommitted_versions(tx);

    for id in &staged {
        assert!(store.get_node(*id).is_none());
        assert!(store.get_node_versioned(*id, epoch, tx).is_none());
    }

    assert_eq!(store.nodes_by_label_count("Person"), 1);
    assert_eq!(store.nodes_by_label("Person"), vec![keep]);
    assert_eq!(store.nodes_by_label_count("Admin"), 0);
    assert!(
        store
            .find_nodes_by_property("name", &Value::from("gone"))
            .is_empty()
    );
    assert!(
        store
            .find_nodes_by_property("name", &Value::from("gone2"))
            .is_empty()
    );
    assert_eq!(
        store.find_nodes_by_property("name", &Value::from("keep")),
        vec![keep]
    );

    store.ensure_statistics_fresh();
    let stats = store.statistics();
    assert_eq!(stats.total_nodes, 1);
    assert_eq!(store.node_count(), 1);

    // Append-only intern: "Admin" may remain registered even with zero members.
    assert_eq!(store.nodes_by_label_count("Admin"), 0);

    let tx2 = TransactionId::new(201);
    let again = store.create_nodes_batch_versioned(
        &[node(&["Person"], vec![("name", Value::from("fresh"))])],
        store.new_epoch(),
        tx2,
    );
    assert_ne!(again[0], staged[0]);
    assert_ne!(again[0], staged[1]);
    assert!(store.get_node_versioned(staged[0], epoch, tx2).is_none());
    store.discard_uncommitted_versions(tx2);
}

#[test]
fn transactional_batch_rollback_cleans_edge_secondaries() {
    let store = LpgStore::new().unwrap();
    let epoch = store.new_epoch();
    let nodes = store.create_nodes_batch_versioned(
        &[
            node(&["N"], vec![]),
            node(&["N"], vec![]),
            node(&["N"], vec![]),
        ],
        epoch,
        TransactionId::SYSTEM,
    );

    let tx = TransactionId::new(300);
    let eids = store.create_edges_batch_versioned(
        &[
            BatchEdgeCreate {
                source: nodes[0],
                target: nodes[1],
                edge_type: "KNOWS",
                properties: vec![(PropertyKey::new("w"), Value::from(1i64))],
            },
            BatchEdgeCreate {
                source: nodes[1],
                target: nodes[2],
                edge_type: "LIKES",
                properties: vec![],
            },
        ],
        store.new_epoch(),
        tx,
    );

    let out: Vec<_> = store.edges_from(nodes[0], Direction::Outgoing).collect();
    assert!(out.iter().any(|(_, eid)| *eid == eids[0]));
    let back: Vec<_> = store.edges_from(nodes[1], Direction::Incoming).collect();
    assert!(back.iter().any(|(_, eid)| *eid == eids[0]));

    store.discard_uncommitted_versions(tx);

    for eid in &eids {
        assert!(store.get_edge(*eid).is_none());
    }
    assert!(
        store
            .edges_from(nodes[0], Direction::Outgoing)
            .next()
            .is_none(),
        "forward adjacency must drop rolled-back edges"
    );
    assert!(
        store
            .edges_from(nodes[1], Direction::Incoming)
            .next()
            .is_none(),
        "backward adjacency must drop rolled-back edges"
    );
    assert!(
        store
            .edges_from(nodes[1], Direction::Outgoing)
            .next()
            .is_none()
    );

    store.ensure_statistics_fresh();
    let stats = store.statistics();
    assert_eq!(stats.total_edges, 0);
    assert!(stats.get_edge_type("KNOWS").is_none());
    assert!(stats.get_edge_type("LIKES").is_none());
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn transactional_batch_cold_adjacency_rollback_is_physical() {
    // Discriminating adjacency proof: force cold compression, then physical
    // remove must leave total/deleted/node counts at zero (no soft residue).
    let adj = ChunkedAdjacency::with_chunk_capacity(8);
    let src = NodeId::new(7);
    let mut edges = Vec::new();
    for i in 0..48 {
        let eid = EdgeId::new(1_000 + i);
        adj.add_edge(src, NodeId::new(i + 1), eid);
        edges.push((src, eid));
    }
    adj.compact();
    adj.freeze_all();
    assert!(adj.memory_stats().cold_entries > 0);

    adj.batch_remove_edges(&edges);

    let stats = adj.memory_stats();
    assert_eq!(stats.cold_entries, 0);
    assert_eq!(stats.hot_entries, 0);
    assert_eq!(stats.node_count, 0);
    assert_eq!(adj.total_edge_count(), 0);
    assert_eq!(adj.active_edge_count(), 0);
}
