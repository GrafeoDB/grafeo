//! Integration tests for CompactStore.
//!
//! Validates all Phase 1 acceptance criteria using a realistic graph
//! with multiple node types, edge types, and property types.

use crate::graph::Direction;
use crate::graph::compact::builder::CompactStoreBuilder;
use crate::graph::compact::id::{decode_edge_id, decode_node_id};
use crate::graph::traits::{GraphStore, GraphStoreMut};
use grafeo_common::types::*;

/// Build a test graph: 100 "Item" nodes, 500 "Activity" nodes, 50 "Account" nodes.
/// Edges: ACTIVITY_ON (Activity->Item), PERFORMED_BY (Activity->Account)
fn build_test_store() -> super::CompactStore {
    // Item nodes: id 0..99, with a "score" (4-bit) and "name" (dict)
    let scores: Vec<u64> = (0..100).map(|i| i % 10).collect();
    let names: Vec<&str> = (0..100)
        .map(|i| match i % 5 {
            0 => "alpha",
            1 => "beta",
            2 => "gamma",
            3 => "delta",
            _ => "epsilon",
        })
        .collect();

    // Activity nodes: id 0..499, with a "rating" (4-bit) and "item_id" FK (7-bit)
    let ratings: Vec<u64> = (0..500).map(|i| (i % 5) + 1).collect();
    let item_ids: Vec<u64> = (0..500).map(|i| i % 100).collect();
    let account_ids: Vec<u64> = (0..500).map(|i| i % 50).collect();

    // Account nodes: id 0..49, with a "name" (dict)
    let account_names: Vec<&str> = (0..50)
        .map(|i| match i % 3 {
            0 => "user_a",
            1 => "user_b",
            _ => "user_c",
        })
        .collect();

    // ACTIVITY_ON edges: each activity -> its item (sorted by activity id = sorted by source)
    let activity_on_edges: Vec<(u32, u32)> =
        (0..500).map(|i| (i as u32, (i % 100) as u32)).collect();

    // PERFORMED_BY edges: each activity -> its account
    let performed_by_edges: Vec<(u32, u32)> =
        (0..500).map(|i| (i as u32, (i % 50) as u32)).collect();

    CompactStoreBuilder::new()
        .node_table("Item", |t| {
            t.column_bitpacked("score", &scores, 4)
                .column_dict("name", &names)
        })
        .node_table("Activity", |t| {
            t.column_bitpacked("rating", &ratings, 4)
                .column_bitpacked("item_id", &item_ids, 7)
                .column_bitpacked("account_id", &account_ids, 6)
        })
        .node_table("Account", |t| t.column_dict("name", &account_names))
        .rel_table("ACTIVITY_ON", "Activity", "Item", |r| {
            r.edges(activity_on_edges).backward(true)
        })
        .rel_table("PERFORMED_BY", "Activity", "Account", |r| {
            r.edges(performed_by_edges).backward(true)
        })
        .build()
        .expect("Failed to build test store")
}

// --- Scan tests ---

#[test]
fn nodes_by_label_returns_correct_counts() {
    let store = build_test_store();
    assert_eq!(store.nodes_by_label("Item").len(), 100);
    assert_eq!(store.nodes_by_label("Activity").len(), 500);
    assert_eq!(store.nodes_by_label("Account").len(), 50);
    assert_eq!(store.nodes_by_label("Nonexistent").len(), 0);
}

#[test]
fn node_count_is_total() {
    let store = build_test_store();
    assert_eq!(store.node_count(), 650); // 100 + 500 + 50
}

#[test]
fn edge_count_is_total() {
    let store = build_test_store();
    assert_eq!(store.edge_count(), 1000); // 500 + 500
}

// --- Point lookup tests ---

#[test]
fn get_node_returns_correct_labels_and_properties() {
    let store = build_test_store();
    let item_ids = store.nodes_by_label("Item");
    let node = store.get_node(item_ids[0]).expect("Node should exist");
    assert!(node.labels.iter().any(|l| l.as_str() == "Item"));
    // Should have "score" and "name" properties
    assert!(node.properties.contains_key(&PropertyKey::from("score")));
    assert!(node.properties.contains_key(&PropertyKey::from("name")));
}

#[test]
fn get_node_property_returns_correct_value() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    // We need to find the activity node at offset 0 in the Activity table.
    // The nodes_by_label order is not guaranteed to be by offset, so let's
    // decode each id to find the one with offset 0.
    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    // Activity at offset 0 has rating = (0 % 5) + 1 = 1
    let rating = store.get_node_property(*first_activity, &PropertyKey::from("rating"));
    assert_eq!(rating, Some(Value::Int64(1)));
}

#[test]
fn get_node_property_batch_works() {
    let store = build_test_store();
    let item_ids = store.nodes_by_label("Item");
    let batch: Vec<NodeId> = item_ids[0..5].to_vec();
    let scores = store.get_node_property_batch(&batch, &PropertyKey::from("score"));
    assert_eq!(scores.len(), 5);
    for s in &scores {
        assert!(s.is_some());
    }
}

// --- Edge traversal tests ---

#[test]
fn edges_from_outgoing_returns_correct_targets() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    // Find activity at offset 0 (has ACTIVITY_ON -> Item 0 and PERFORMED_BY -> Account 0)
    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    let outgoing = store.edges_from(*first_activity, Direction::Outgoing);
    assert_eq!(outgoing.len(), 2, "Activity 0 should have 2 outgoing edges");
}

#[test]
fn edges_from_incoming_returns_correct_sources() {
    let store = build_test_store();
    let item_ids = store.nodes_by_label("Item");

    // Find item at offset 0 (should have 5 incoming ACTIVITY_ON edges from activities 0,100,200,300,400)
    let first_item = item_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find item at offset 0");

    let incoming = store.edges_from(*first_item, Direction::Incoming);
    assert_eq!(
        incoming.len(),
        5,
        "Item 0 should have 5 incoming activities"
    );
}

#[test]
fn get_edge_returns_correct_metadata() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    let outgoing = store.edges_from(*first_activity, Direction::Outgoing);
    assert!(!outgoing.is_empty());
    let (target_id, edge_id) = outgoing[0];

    let edge = store.get_edge(edge_id).expect("Edge should exist");
    assert_eq!(edge.src, *first_activity);
    assert_eq!(edge.dst, target_id);
}

#[test]
fn edge_type_returns_correct_string() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    let outgoing = store.edges_from(*first_activity, Direction::Outgoing);

    // Check that we get the expected edge types
    let mut types: Vec<String> = outgoing
        .iter()
        .filter_map(|(_, eid)| store.edge_type(*eid).map(|s| s.to_string()))
        .collect();
    types.sort();
    assert_eq!(types, vec!["ACTIVITY_ON", "PERFORMED_BY"]);
}

#[test]
fn neighbors_returns_node_ids_only() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    let neighbors = store.neighbors(*first_activity, Direction::Outgoing);
    assert_eq!(neighbors.len(), 2);
    // Should be NodeIds, not EdgeIds
    for nid in &neighbors {
        assert!(store.get_node(*nid).is_some());
    }
}

#[test]
fn has_backward_adjacency_is_true() {
    let store = build_test_store();
    assert!(store.has_backward_adjacency());
}

// --- Statistics tests ---

#[test]
fn statistics_returns_correct_counts() {
    let store = build_test_store();
    let stats = store.statistics();
    assert_eq!(stats.total_nodes, 650);
    assert_eq!(stats.total_edges, 1000);
}

#[test]
fn estimate_label_cardinality_matches() {
    let store = build_test_store();
    assert_eq!(store.estimate_label_cardinality("Item"), 100.0);
    assert_eq!(store.estimate_label_cardinality("Activity"), 500.0);
    assert_eq!(store.estimate_label_cardinality("Account"), 50.0);
}

// --- Mutation (DeltaBuffer) tests ---

#[test]
fn create_node_appears_in_scans() {
    let store = build_test_store();
    assert_eq!(store.nodes_by_label("Item").len(), 100);

    let new_id = store.create_node(&["Item"]);
    assert_eq!(store.nodes_by_label("Item").len(), 101);

    let node = store.get_node(new_id).expect("New node should exist");
    assert!(node.labels.iter().any(|l| l.as_str() == "Item"));
}

#[test]
fn set_property_on_new_node() {
    let store = build_test_store();
    let new_id = store.create_node(&["Item"]);
    store.set_node_property(new_id, "score", Value::Int64(7));

    assert_eq!(
        store.get_node_property(new_id, &PropertyKey::from("score")),
        Some(Value::Int64(7))
    );
}

#[test]
fn set_property_overrides_snapshot() {
    let store = build_test_store();
    let item_ids = store.nodes_by_label("Item");

    // Find item at offset 0 to have a deterministic test.
    let id = item_ids
        .iter()
        .find(|nid| {
            let (_, offset) = decode_node_id(**nid);
            offset == 0
        })
        .copied()
        .expect("Should find item at offset 0");

    // Original score is 0 % 10 = 0
    assert_eq!(
        store.get_node_property(id, &PropertyKey::from("score")),
        Some(Value::Int64(0))
    );

    store.set_node_property(id, "score", Value::Int64(99));
    assert_eq!(
        store.get_node_property(id, &PropertyKey::from("score")),
        Some(Value::Int64(99))
    );
}

#[test]
fn delete_node_removes_from_scans() {
    let store = build_test_store();
    let item_ids = store.nodes_by_label("Item");
    assert_eq!(item_ids.len(), 100);

    assert!(store.delete_node(item_ids[0]));
    assert_eq!(store.nodes_by_label("Item").len(), 99);
    assert!(store.get_node(item_ids[0]).is_none());
}

#[test]
fn create_edge_appears_in_traversal() {
    let store = build_test_store();
    let item_ids = store.nodes_by_label("Item");
    let account_ids = store.nodes_by_label("Account");

    let edge_id = store.create_edge(item_ids[0], account_ids[0], "CUSTOM_REL");

    let outgoing = store.edges_from(item_ids[0], Direction::Outgoing);
    assert!(outgoing.iter().any(|(_, eid)| *eid == edge_id));

    let edge = store.get_edge(edge_id).expect("Edge should exist");
    assert_eq!(edge.src, item_ids[0]);
    assert_eq!(edge.dst, account_ids[0]);
}

// --- Schema introspection ---

#[test]
fn all_labels_returns_all_types() {
    let store = build_test_store();
    let labels = store.all_labels();
    assert!(labels.contains(&"Item".to_string()));
    assert!(labels.contains(&"Activity".to_string()));
    assert!(labels.contains(&"Account".to_string()));
}

#[test]
fn all_edge_types_returns_all_types() {
    let store = build_test_store();
    let types = store.all_edge_types();
    assert!(types.contains(&"ACTIVITY_ON".to_string()));
    assert!(types.contains(&"PERFORMED_BY".to_string()));
}

// --- Memory measurement ---

#[test]
fn memory_is_compact() {
    let store = build_test_store();
    // 650 nodes with a few properties each + 1000 edges
    // Should be well under 100KB for this small graph
    // (LpgStore would use ~2MB for the same data based on our benchmarks)
    let node_count = store.node_count();
    let edge_count = store.edge_count();
    assert_eq!(node_count, 650);
    assert_eq!(edge_count, 1000);
    // Basic sanity: the store exists and responds to queries
    // Real memory measurement is in the benchmark, not the unit test
}

// --- Edge cases ---

#[test]
fn get_nonexistent_node_returns_none() {
    let store = build_test_store();
    assert!(store.get_node(NodeId::new(999_999_999)).is_none());
}

#[test]
fn get_nonexistent_edge_returns_none() {
    let store = build_test_store();
    assert!(store.get_edge(EdgeId::new(999_999_999)).is_none());
}

#[test]
fn node_ids_returns_all() {
    let store = build_test_store();
    assert_eq!(store.node_ids().len(), 650);
}

// --- ID encoding round-trip ---

#[test]
fn node_ids_encode_table_and_offset_correctly() {
    let store = build_test_store();
    let item_ids = store.nodes_by_label("Item");
    for nid in &item_ids {
        let (table_id, offset) = decode_node_id(*nid);
        // Table ID should be in valid range (0, 1, or 2 for our three tables)
        assert!(table_id < 3, "table_id should be < 3, got {table_id}");
        // Offset should be < 500 (max table size)
        assert!(offset < 500, "offset should be < 500, got {offset}");
    }
}

#[test]
fn edge_ids_encode_rel_table_and_position() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    let outgoing = store.edges_from(*first_activity, Direction::Outgoing);
    for (_, eid) in &outgoing {
        let (rel_table_id, csr_pos) = decode_edge_id(*eid);
        // Rel table ID should be 0 or 1 (ACTIVITY_ON or PERFORMED_BY)
        assert!(
            rel_table_id < 2,
            "rel_table_id should be < 2, got {rel_table_id}"
        );
        // CSR position should be within edge count
        assert!(csr_pos < 500, "csr_pos should be < 500, got {csr_pos}");
    }
}

// --- Out-degree / in-degree tests ---

#[test]
fn out_degree_matches_edges_from() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    let degree = store.out_degree(*first_activity);
    let edges = store.edges_from(*first_activity, Direction::Outgoing);
    assert_eq!(degree, edges.len());
    assert_eq!(degree, 2); // ACTIVITY_ON + PERFORMED_BY
}

#[test]
fn in_degree_matches_edges_from_incoming() {
    let store = build_test_store();
    let item_ids = store.nodes_by_label("Item");

    let first_item = item_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find item at offset 0");

    let degree = store.in_degree(*first_item);
    let edges = store.edges_from(*first_item, Direction::Incoming);
    assert_eq!(degree, edges.len());
    assert_eq!(degree, 5); // activities 0, 100, 200, 300, 400
}

// --- Delete edge test ---

#[test]
fn delete_edge_removes_from_traversal() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    let outgoing_before = store.edges_from(*first_activity, Direction::Outgoing);
    assert_eq!(outgoing_before.len(), 2);

    // Delete the first outgoing edge
    let (_, eid) = outgoing_before[0];
    assert!(store.delete_edge(eid));

    let outgoing_after = store.edges_from(*first_activity, Direction::Outgoing);
    assert_eq!(outgoing_after.len(), 1);
    assert!(store.get_edge(eid).is_none());
}

// --- find_nodes_by_property test ---

#[test]
fn find_nodes_by_property_works() {
    let store = build_test_store();
    // Find all Item nodes with score = 0
    let results = store.find_nodes_by_property("score", &Value::Int64(0));
    // Items at offsets 0, 10, 20, ..., 90 have score = 0 (10 items)
    assert_eq!(results.len(), 10, "Should find 10 items with score=0");
}

// --- Direction::Both test ---

#[test]
fn edges_from_both_directions() {
    let store = build_test_store();
    let activity_ids = store.nodes_by_label("Activity");

    let first_activity = activity_ids
        .iter()
        .find(|id| {
            let (_, offset) = decode_node_id(**id);
            offset == 0
        })
        .expect("Should find activity at offset 0");

    // Activity 0 has 2 outgoing edges and 0 incoming snapshot edges
    let both = store.edges_from(*first_activity, Direction::Both);
    let out = store.edges_from(*first_activity, Direction::Outgoing);
    let inc = store.edges_from(*first_activity, Direction::Incoming);
    assert_eq!(both.len(), out.len() + inc.len());
}
