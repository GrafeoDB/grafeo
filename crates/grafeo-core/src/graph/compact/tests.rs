//! Comprehensive tests for the read-only CompactStore.
//!
//! Covers builder construction, GraphStore trait compliance, filtered search
//! with zone maps, statistics, introspection, and edge cases.

use arcstr::ArcStr;

use crate::graph::Direction;
use crate::graph::compact::builder::{CompactStoreBuilder, CompactStoreError};
use crate::graph::compact::id::{decode_edge_id, decode_node_id, encode_node_id};
use crate::graph::lpg::CompareOp;
use crate::graph::traits::{GraphStore, GraphStoreSearch};
use grafeo_common::types::*;

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

/// Builds a small graph with Person and City nodes, LIVES_IN and KNOWS edges.
///
/// Person nodes (5): Alix(25), Gus(30), Vincent(35), Jules(40), Mia(45)
/// City nodes   (3): Amsterdam, Berlin, Paris
/// LIVES_IN     (5): Person 0->City 0, 1->1, 2->2, 3->0, 4->1
/// KNOWS        (4): Person 0->1, 1->2, 2->3, 3->4
fn build_test_store() -> super::CompactStore {
    CompactStoreBuilder::new()
        .node_table("Person", |t| {
            t.column_bitpacked("age", &[25, 30, 35, 40, 45], 6)
                .column_dict("name", &["Alix", "Gus", "Vincent", "Jules", "Mia"])
        })
        .node_table("City", |t| {
            t.column_dict("name", &["Amsterdam", "Berlin", "Paris"])
        })
        .rel_table("LIVES_IN", "Person", "City", |r| {
            r.edges([(0, 0), (1, 1), (2, 2), (3, 0), (4, 1)])
                .backward(true)
        })
        .rel_table("KNOWS", "Person", "Person", |r| {
            r.edges([(0, 1), (1, 2), (2, 3), (3, 4)]).backward(true)
        })
        .build()
        .unwrap()
}

/// Returns the NodeId for Person at a given row offset within the Person table.
fn person_at(store: &super::CompactStore, offset: u64) -> NodeId {
    let ids = store.nodes_by_label("Person");
    ids.into_iter()
        .find(|id| {
            let (_, off) = decode_node_id(*id);
            off == offset
        })
        .expect("person at offset should exist")
}

/// Returns the NodeId for City at a given row offset within the City table.
fn city_at(store: &super::CompactStore, offset: u64) -> NodeId {
    let ids = store.nodes_by_label("City");
    ids.into_iter()
        .find(|id| {
            let (_, off) = decode_node_id(*id);
            off == offset
        })
        .expect("city at offset should exist")
}

// ===========================================================================
// A. Builder tests
// ===========================================================================

#[test]
fn test_builder_single_node_table() {
    let store = CompactStoreBuilder::new()
        .node_table("Person", |t| {
            t.column_bitpacked("age", &[25, 30], 6)
                .column_dict("name", &["Alix", "Gus"])
        })
        .build()
        .unwrap();

    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 0);
    assert_eq!(store.nodes_by_label("Person").len(), 2);
}

#[test]
fn test_builder_multiple_node_tables() {
    let store = CompactStoreBuilder::new()
        .node_table("Person", |t| t.column_dict("name", &["Alix", "Gus"]))
        .node_table("City", |t| t.column_dict("name", &["Amsterdam", "Berlin"]))
        .build()
        .unwrap();

    assert_eq!(store.node_count(), 4);
    assert_eq!(store.nodes_by_label("Person").len(), 2);
    assert_eq!(store.nodes_by_label("City").len(), 2);
}

#[test]
fn test_builder_with_edges_and_backward() {
    let store = build_test_store();
    assert_eq!(store.node_count(), 8);
    assert_eq!(store.edge_count(), 9);
    assert!(store.has_backward_adjacency());
}

#[test]
fn test_builder_with_edge_properties() {
    let store = CompactStoreBuilder::new()
        .node_table("Person", |t| {
            t.column_dict("name", &["Alix", "Gus", "Vincent"])
        })
        .rel_table("RATED", "Person", "Person", |r| {
            r.edges([(0, 1), (1, 2)])
                .column_bitpacked("score", &[5, 8], 4)
        })
        .build()
        .unwrap();

    let alix = person_at(&store, 0);
    let outgoing = store.edges_from(alix, Direction::Outgoing);
    assert_eq!(outgoing.len(), 1);

    let (_, edge_id) = outgoing[0];
    let score = store.get_edge_property(edge_id, &PropertyKey::from("score"));
    assert_eq!(score, Some(Value::Int64(5)));
}

#[test]
fn test_builder_label_not_found_error() {
    let result = CompactStoreBuilder::new()
        .node_table("Person", |t| t.column_dict("name", &["Alix"]))
        .rel_table("LIVES_IN", "Person", "City", |r| r.edges([(0, 0)]))
        .build();

    assert!(matches!(result, Err(CompactStoreError::LabelNotFound(ref s)) if s == "City"));
}

#[test]
fn test_builder_column_length_mismatch_error() {
    let result = CompactStoreBuilder::new()
        .node_table("Person", |t| {
            t.column_bitpacked("age", &[25, 30, 35], 6)
                .column_dict("name", &["Alix", "Gus"]) // 2 != 3
        })
        .build();

    assert!(matches!(
        result,
        Err(CompactStoreError::ColumnLengthMismatch {
            expected: 3,
            got: 2
        })
    ));
}

#[test]
fn test_builder_duplicate_label_error() {
    let result = CompactStoreBuilder::new()
        .node_table("Person", |t| t.column_dict("name", &["Alix"]))
        .node_table("Person", |t| t.column_dict("name", &["Gus"]))
        .build();

    assert!(matches!(
        result,
        Err(CompactStoreError::DuplicateLabel(ref s)) if s == "Person"
    ));
}

#[test]
fn test_builder_duplicate_edge_type_error() {
    let result = CompactStoreBuilder::new()
        .node_table("Person", |t| t.column_dict("name", &["Alix", "Gus"]))
        .node_table("City", |t| t.column_dict("name", &["Amsterdam"]))
        .rel_table("LIVES_IN", "Person", "City", |r| r.edges(vec![(0, 0)]))
        .rel_table("LIVES_IN", "Person", "City", |r| r.edges(vec![(1, 0)]))
        .build();

    assert!(matches!(
        result,
        Err(CompactStoreError::DuplicateEdgeType(ref s)) if s == "LIVES_IN (Person -> City)"
    ));
}

#[test]
fn test_builder_value_overflow_error() {
    // u64::MAX exceeds i64::MAX, should trigger ValueOverflow.
    let result = CompactStoreBuilder::new()
        .node_table("Bad", |t| t.column_bitpacked("big", &[u64::MAX], 64))
        .build();

    assert!(matches!(
        result,
        Err(CompactStoreError::ValueOverflow { .. })
    ));
}

#[test]
fn test_builder_bitmap_column() {
    let store = CompactStoreBuilder::new()
        .node_table("Person", |t| {
            t.column_dict("name", &["Alix", "Gus", "Vincent"])
                .column_bitmap("active", &[true, false, true])
        })
        .build()
        .unwrap();

    let alix = person_at(&store, 0);
    assert_eq!(
        store.get_node_property(alix, &PropertyKey::from("active")),
        Some(Value::Bool(true))
    );
    let gus = person_at(&store, 1);
    assert_eq!(
        store.get_node_property(gus, &PropertyKey::from("active")),
        Some(Value::Bool(false))
    );
}

#[test]
fn test_builder_int8_vector_column() {
    let store = CompactStoreBuilder::new()
        .node_table("Doc", |t| {
            t.column_dict("title", &["doc_a", "doc_b"])
                .column_int8_vector("embedding", vec![1, 2, 3, -1, -2, -3], 3)
        })
        .build()
        .unwrap();

    let ids = store.nodes_by_label("Doc");
    assert_eq!(ids.len(), 2);

    let first = ids
        .iter()
        .find(|id| decode_node_id(**id).1 == 0)
        .copied()
        .unwrap();
    let emb = store.get_node_property(first, &PropertyKey::from("embedding"));
    let expected: Vec<Value> = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
    assert_eq!(emb, Some(Value::List(std::sync::Arc::from(expected))));
}

#[test]
fn test_builder_empty_table() {
    let store = CompactStoreBuilder::new()
        .node_table("Empty", |t| t.column_dict("name", &[]))
        .build()
        .unwrap();

    assert_eq!(store.node_count(), 0);
    assert_eq!(store.nodes_by_label("Empty").len(), 0);
}

// ===========================================================================
// B. GraphStore trait compliance
// ===========================================================================

#[test]
fn test_get_node_returns_labels_and_properties() {
    let store = build_test_store();
    let alix = person_at(&store, 0);
    let node = store.get_node(alix).expect("node should exist");

    assert!(node.has_label("Person"));
    assert_eq!(
        node.properties.get(&PropertyKey::from("name")),
        Some(&Value::String(ArcStr::from("Alix")))
    );
    assert_eq!(
        node.properties.get(&PropertyKey::from("age")),
        Some(&Value::Int64(25))
    );
}

#[test]
fn test_get_node_nonexistent_returns_none() {
    let store = build_test_store();
    assert!(store.get_node(NodeId::new(999_999_999)).is_none());
}

#[test]
fn test_get_edge_returns_src_dst_type_properties() {
    let store = build_test_store();
    let alix = person_at(&store, 0);
    let outgoing = store.edges_from(alix, Direction::Outgoing);

    // Alix has 2 outgoing edges: LIVES_IN -> Amsterdam, KNOWS -> Gus
    assert_eq!(outgoing.len(), 2);

    for (target, eid) in &outgoing {
        let edge = store.get_edge(*eid).expect("edge should exist");
        assert_eq!(edge.src, alix);
        assert_eq!(edge.dst, *target);
        let etype = edge.edge_type.as_str();
        assert!(etype == "LIVES_IN" || etype == "KNOWS");
    }
}

#[test]
fn test_get_edge_nonexistent_returns_none() {
    let store = build_test_store();
    assert!(store.get_edge(EdgeId::new(999_999_999)).is_none());
}

#[test]
fn test_get_node_property_direct() {
    let store = build_test_store();
    let gus = person_at(&store, 1);
    assert_eq!(
        store.get_node_property(gus, &PropertyKey::from("age")),
        Some(Value::Int64(30))
    );
    assert_eq!(
        store.get_node_property(gus, &PropertyKey::from("name")),
        Some(Value::String(ArcStr::from("Gus")))
    );
}

#[test]
fn test_get_node_property_missing_key() {
    let store = build_test_store();
    let alix = person_at(&store, 0);
    assert_eq!(
        store.get_node_property(alix, &PropertyKey::from("nonexistent")),
        None
    );
}

#[test]
fn test_get_edge_property_direct() {
    let store = CompactStoreBuilder::new()
        .node_table("Person", |t| t.column_dict("name", &["Alix", "Gus"]))
        .rel_table("RATED", "Person", "Person", |r| {
            r.edges([(0, 1)]).column_bitpacked("weight", &[42], 8)
        })
        .build()
        .unwrap();

    let alix = person_at(&store, 0);
    let out = store.edges_from(alix, Direction::Outgoing);
    let (_, eid) = out[0];
    assert_eq!(
        store.get_edge_property(eid, &PropertyKey::from("weight")),
        Some(Value::Int64(42))
    );
    assert_eq!(
        store.get_edge_property(eid, &PropertyKey::from("missing")),
        None
    );
}

#[test]
fn test_get_node_property_batch() {
    let store = build_test_store();
    let person_ids = store.nodes_by_label("Person");
    let ages = store.get_node_property_batch(&person_ids, &PropertyKey::from("age"));
    assert_eq!(ages.len(), 5);
    for age in &ages {
        assert!(age.is_some());
    }
}

#[test]
fn test_nodes_by_label() {
    let store = build_test_store();
    let persons = store.nodes_by_label("Person");
    assert_eq!(persons.len(), 5);
    let cities = store.nodes_by_label("City");
    assert_eq!(cities.len(), 3);

    // Each node should actually exist and have the correct label.
    for nid in &persons {
        let node = store.get_node(*nid).unwrap();
        assert!(node.has_label("Person"));
    }
}

#[test]
fn test_nodes_by_label_nonexistent() {
    let store = build_test_store();
    assert!(store.nodes_by_label("Vehicle").is_empty(), "expected empty");
}

#[test]
fn test_node_ids_returns_all_sorted() {
    let store = build_test_store();
    let ids = store.node_ids();
    assert_eq!(ids.len(), 8); // 5 persons + 3 cities

    // Verify sorted.
    for window in ids.windows(2) {
        assert!(window[0].as_u64() <= window[1].as_u64());
    }
}

#[test]
fn test_node_count_and_edge_count() {
    let store = build_test_store();
    assert_eq!(store.node_count(), 8);
    assert_eq!(store.edge_count(), 9); // 5 LIVES_IN + 4 KNOWS
}

#[test]
fn test_neighbors_outgoing() {
    let store = build_test_store();
    let alix = person_at(&store, 0);
    let neighbors = store.neighbors(alix, Direction::Outgoing);
    // Alix -> LIVES_IN Amsterdam (city offset 0), KNOWS Gus (person offset 1)
    assert_eq!(neighbors.len(), 2);

    for nid in &neighbors {
        assert!(store.get_node(*nid).is_some());
    }
}

#[test]
fn test_neighbors_incoming() {
    let store = build_test_store();
    // Gus (person offset 1) has incoming KNOWS from Alix (offset 0)
    let gus = person_at(&store, 1);
    let incoming = store.neighbors(gus, Direction::Incoming);

    // At least one incoming neighbor (Alix via KNOWS).
    // May also have incoming from LIVES_IN backward if Gus is a target there (he's not).
    assert!(
        !incoming.is_empty(),
        "Gus should have at least 1 incoming neighbor"
    );
}

#[test]
fn test_neighbors_both() {
    let store = build_test_store();
    let gus = person_at(&store, 1);
    let both = store.neighbors(gus, Direction::Both);
    let out = store.neighbors(gus, Direction::Outgoing);
    let inc = store.neighbors(gus, Direction::Incoming);
    assert_eq!(both.len(), out.len() + inc.len());
}

#[test]
fn test_edges_from_outgoing() {
    let store = build_test_store();
    let vincent = person_at(&store, 2);
    let outgoing = store.edges_from(vincent, Direction::Outgoing);
    // Vincent -> LIVES_IN Paris (city 2) + KNOWS Jules (person 3)
    assert_eq!(outgoing.len(), 2);
}

#[test]
fn test_edges_from_incoming() {
    let store = build_test_store();
    // Amsterdam (city offset 0) should have 2 incoming LIVES_IN edges:
    // Person 0 (Alix) and Person 3 (Jules).
    let amsterdam = city_at(&store, 0);
    let incoming = store.edges_from(amsterdam, Direction::Incoming);
    assert_eq!(incoming.len(), 2);
}

#[test]
fn test_out_degree_and_in_degree() {
    let store = build_test_store();
    let alix = person_at(&store, 0);
    // Alix outgoing: LIVES_IN + KNOWS = 2
    assert_eq!(store.out_degree(alix), 2);

    // Berlin (city offset 1) has in-degree 2: Person 1 (Gus) + Person 4 (Mia)
    let berlin = city_at(&store, 1);
    assert_eq!(store.in_degree(berlin), 2);
}

#[test]
fn test_edge_type_returns_correct_string() {
    let store = build_test_store();
    let alix = person_at(&store, 0);
    let outgoing = store.edges_from(alix, Direction::Outgoing);

    let mut types: Vec<String> = outgoing
        .iter()
        .filter_map(|(_, eid)| store.edge_type(*eid).map(|s| s.to_string()))
        .collect();
    types.sort();
    assert_eq!(types, vec!["KNOWS", "LIVES_IN"]);
}

#[test]
fn test_has_backward_adjacency() {
    let store = build_test_store();
    assert!(store.has_backward_adjacency());

    // Store without backward CSR.
    let store_no_bwd = CompactStoreBuilder::new()
        .node_table("A", |t| t.column_bitpacked("x", &[1], 4))
        .node_table("B", |t| t.column_bitpacked("x", &[2], 4))
        .rel_table("R", "A", "B", |r| r.edges([(0, 0)]).backward(false))
        .build()
        .unwrap();
    assert!(!store_no_bwd.has_backward_adjacency());
}

// ===========================================================================
// C. Filtered search + zone maps
// ===========================================================================

#[test]
fn test_find_nodes_by_property_int() {
    let store = build_test_store();
    let results = store.find_nodes_by_property("age", &Value::Int64(35));
    assert_eq!(results.len(), 1);
    let node = store.get_node(results[0]).unwrap();
    assert_eq!(
        node.properties.get(&PropertyKey::from("name")),
        Some(&Value::String(ArcStr::from("Vincent")))
    );
}

#[test]
fn test_find_nodes_by_property_string() {
    let store = build_test_store();
    // "Amsterdam" appears in City table.
    let results = store.find_nodes_by_property("name", &Value::String(ArcStr::from("Amsterdam")));
    assert_eq!(results.len(), 1);
    let node = store.get_node(results[0]).unwrap();
    assert!(node.has_label("City"));
}

#[test]
fn test_find_nodes_by_properties_multi() {
    let store = build_test_store();
    let results = store.find_nodes_by_properties(&[
        ("name", Value::String(ArcStr::from("Alix"))),
        ("age", Value::Int64(25)),
    ]);
    assert_eq!(results.len(), 1);

    // No match: correct name but wrong age.
    let no_match = store.find_nodes_by_properties(&[
        ("name", Value::String(ArcStr::from("Alix"))),
        ("age", Value::Int64(99)),
    ]);
    assert!(no_match.is_empty(), "{no_match:?}");
}

#[test]
fn test_find_nodes_in_range_inclusive() {
    let store = build_test_store();
    // Ages 30..=40 inclusive: Gus(30), Vincent(35), Jules(40)
    let results = store.find_nodes_in_range(
        "age",
        Some(&Value::Int64(30)),
        Some(&Value::Int64(40)),
        true,
        true,
    );
    assert_eq!(results.len(), 3);
}

#[test]
fn test_find_nodes_in_range_exclusive() {
    let store = build_test_store();
    // Ages (30..40) exclusive on both ends: only Vincent(35)
    let results = store.find_nodes_in_range(
        "age",
        Some(&Value::Int64(30)),
        Some(&Value::Int64(40)),
        false,
        false,
    );
    assert_eq!(results.len(), 1);
}

#[test]
fn test_find_nodes_in_range_unbounded_min() {
    let store = build_test_store();
    // age <= 30: Alix(25), Gus(30)
    let results = store.find_nodes_in_range("age", None, Some(&Value::Int64(30)), true, true);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_node_property_might_match_zone_map_prune() {
    // Build a store where every table has the "age" column so zone maps are
    // authoritative (no conservative fallback for tables missing the column).
    let store = CompactStoreBuilder::new()
        .node_table("Young", |t| t.column_bitpacked("age", &[10, 20], 6))
        .node_table("Old", |t| t.column_bitpacked("age", &[60, 70], 7))
        .build()
        .unwrap();

    let key = PropertyKey::from("age");
    // 100 is above both tables' max (20 and 70): should be pruned.
    assert!(!store.node_property_might_match(&key, CompareOp::Eq, &Value::Int64(100)));
    // 5 is below both tables' min (10 and 60): should be pruned.
    assert!(!store.node_property_might_match(&key, CompareOp::Eq, &Value::Int64(5)));
}

#[test]
fn test_node_property_might_match_zone_map_pass() {
    let store = build_test_store();
    let key = PropertyKey::from("age");
    // 35 is in range [25, 45], so zone map should not prune.
    assert!(store.node_property_might_match(&key, CompareOp::Eq, &Value::Int64(35)));
    // Greater than 20: max(45) > 20, possible.
    assert!(store.node_property_might_match(&key, CompareOp::Gt, &Value::Int64(20)));
}

// ===========================================================================
// D. Statistics + introspection
// ===========================================================================

#[test]
fn test_statistics_node_and_edge_counts() {
    let store = build_test_store();
    let stats = store.statistics();
    assert_eq!(stats.total_nodes, 8);
    assert_eq!(stats.total_edges, 9);
}

#[test]
fn test_estimate_label_cardinality() {
    let store = build_test_store();
    assert_eq!(store.estimate_label_cardinality("Person"), 5.0);
    assert_eq!(store.estimate_label_cardinality("City"), 3.0);
    assert_eq!(store.estimate_label_cardinality("Unknown"), 0.0);
}

#[test]
fn test_estimate_avg_degree() {
    let store = build_test_store();
    // LIVES_IN: 5 edges, 5 source Person nodes -> avg out-degree = 1.0
    let avg_out = store.estimate_avg_degree("LIVES_IN", true);
    assert!((avg_out - 1.0).abs() < f64::EPSILON);

    // LIVES_IN: 5 edges, 3 target City nodes -> avg in-degree ~ 1.667
    let avg_in = store.estimate_avg_degree("LIVES_IN", false);
    assert!((avg_in - 5.0 / 3.0).abs() < 0.01);

    // Unknown edge type -> 0.0
    assert_eq!(store.estimate_avg_degree("NONEXISTENT", true), 0.0);
}

#[test]
fn test_all_labels() {
    let store = build_test_store();
    let mut labels = store.all_labels();
    labels.sort();
    assert_eq!(labels, vec!["City", "Person"]);
}

#[test]
fn test_all_edge_types() {
    let store = build_test_store();
    let mut types = store.all_edge_types();
    types.sort();
    assert_eq!(types, vec!["KNOWS", "LIVES_IN"]);
}

// ===========================================================================
// E. Edge cases
// ===========================================================================

#[test]
fn test_versioned_methods_delegate() {
    let store = build_test_store();
    let alix = person_at(&store, 0);

    // Versioned lookups should return the same result as non-versioned (read-only store).
    let node = store.get_node(alix).unwrap();
    let versioned = store
        .get_node_versioned(alix, EpochId(5), TransactionId(100))
        .unwrap();
    assert_eq!(node.id, versioned.id);
    assert_eq!(node.labels, versioned.labels);
    // Compare property values individually (PropertyMap order may differ).
    let age_key = PropertyKey::from("age");
    let name_key = PropertyKey::from("name");
    assert_eq!(
        node.properties.get(&age_key),
        versioned.properties.get(&age_key)
    );
    assert_eq!(
        node.properties.get(&name_key),
        versioned.properties.get(&name_key)
    );

    // Edge versioned lookup.
    let outgoing = store.edges_from(alix, Direction::Outgoing);
    let (_, eid) = outgoing[0];
    let edge = store.get_edge(eid).unwrap();
    let edge_v = store
        .get_edge_versioned(eid, EpochId(10), TransactionId(50))
        .unwrap();
    assert_eq!(edge.src, edge_v.src);
    assert_eq!(edge.dst, edge_v.dst);
    assert_eq!(edge.edge_type, edge_v.edge_type);
}

#[test]
fn test_current_epoch_returns_1() {
    let store = build_test_store();
    assert_eq!(store.current_epoch(), EpochId(1));
}

#[test]
fn test_memory_bytes_is_nonzero() {
    let store = build_test_store();
    assert!(
        store.memory_bytes() > 0,
        "non-empty store should report > 0 bytes"
    );
}

#[test]
fn test_node_ids_encode_correctly() {
    let store = build_test_store();
    let all_ids = store.node_ids();
    for nid in &all_ids {
        let (table_id, offset) = decode_node_id(*nid);
        // Round-trip: re-encoding should produce the same NodeId.
        let re_encoded = encode_node_id(table_id, offset);
        assert_eq!(*nid, re_encoded);
        // Table ID should be valid for our 2-table store.
        assert!(table_id < 2, "table_id should be 0 or 1, got {table_id}");
    }

    // Also verify edge ID round-trips.
    let alix = person_at(&store, 0);
    let edges = store.edges_from(alix, Direction::Outgoing);
    for (_, eid) in &edges {
        let (rel_id, pos) = decode_edge_id(*eid);
        assert!(rel_id < 2, "rel_table_id should be 0 or 1, got {rel_id}");
        // Position should be within edge count for that table.
        assert!(pos < 5, "csr_pos should be < 5, got {pos}");
    }
}

#[test]
fn test_empty_store() {
    let store = CompactStoreBuilder::new().build().unwrap();
    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);
    assert!(store.node_ids().is_empty(), "{:?}", store.node_ids());
    assert!(store.all_labels().is_empty(), "{:?}", store.all_labels());
    assert!(
        store.all_edge_types().is_empty(),
        "{:?}",
        store.all_edge_types()
    );
    assert!(store.get_node(NodeId::new(0)).is_none());
    assert!(store.get_edge(EdgeId::new(0)).is_none());
    assert_eq!(store.current_epoch(), EpochId(1));
    assert_eq!(store.memory_bytes(), 0);

    let stats = store.statistics();
    assert_eq!(stats.total_nodes, 0);
    assert_eq!(stats.total_edges, 0);
}

// ===========================================================================
// Additional coverage: all_property_keys, edge_property_might_match, etc.
// ===========================================================================

#[test]
fn test_all_property_keys() {
    let store = build_test_store();
    let mut keys = store.all_property_keys();
    keys.sort();
    // Person has "age" and "name", City has "name".
    assert_eq!(keys, vec!["age", "name"]);
}

#[test]
fn test_edge_property_might_match_conservative() {
    let store = build_test_store();
    // Edge properties have no zone maps, so this should always return true.
    assert!(store.edge_property_might_match(
        &PropertyKey::from("weight"),
        CompareOp::Eq,
        &Value::Int64(999)
    ));
}

#[test]
fn test_out_degree_matches_edges_from() {
    let store = build_test_store();
    for nid in store.node_ids() {
        let degree = store.out_degree(nid);
        let edges = store.edges_from(nid, Direction::Outgoing);
        assert_eq!(degree, edges.len(), "out_degree mismatch for node {nid:?}");
    }
}

#[test]
fn test_in_degree_matches_edges_from_incoming() {
    let store = build_test_store();
    for nid in store.node_ids() {
        let degree = store.in_degree(nid);
        let edges = store.edges_from(nid, Direction::Incoming);
        assert_eq!(degree, edges.len(), "in_degree mismatch for node {nid:?}");
    }
}

#[test]
fn test_edges_from_both_combines_directions() {
    let store = build_test_store();
    let vincent = person_at(&store, 2);
    let both = store.edges_from(vincent, Direction::Both);
    let out = store.edges_from(vincent, Direction::Outgoing);
    let inc = store.edges_from(vincent, Direction::Incoming);
    assert_eq!(both.len(), out.len() + inc.len());
}

#[test]
fn test_find_nodes_by_property_no_match() {
    let store = build_test_store();
    let results = store.find_nodes_by_property("age", &Value::Int64(999));
    assert!(results.is_empty(), "{results:?}");
}

#[test]
fn test_get_node_property_batch_mixed() {
    let store = build_test_store();
    let person_ids = store.nodes_by_label("Person");
    let city_ids = store.nodes_by_label("City");

    // Batch with mixed labels: "age" exists on Person but not on City.
    let mixed: Vec<NodeId> = vec![person_ids[0], city_ids[0]];
    let results = store.get_node_property_batch(&mixed, &PropertyKey::from("age"));
    assert!(results[0].is_some());
    assert!(results[1].is_none());
}

#[test]
fn test_edge_traversal_consistency() {
    let store = build_test_store();
    // For every outgoing edge, verify get_edge returns matching src/dst/type.
    for nid in store.node_ids() {
        for (target, eid) in store.edges_from(nid, Direction::Outgoing) {
            let edge = store
                .get_edge(eid)
                .expect("edge from edges_from should exist");
            assert_eq!(edge.src, nid, "edge src should match traversal origin");
            assert_eq!(edge.dst, target, "edge dst should match traversal target");
            let etype = store.edge_type(eid).expect("edge_type should exist");
            assert_eq!(edge.edge_type, etype);
        }
    }
}

#[test]
fn test_node_property_might_match_missing_column() {
    let store = build_test_store();
    // "color" does not exist on any table, so zone map is absent.
    // Should conservatively return true (no pruning possible).
    let key = PropertyKey::from("color");
    assert!(store.node_property_might_match(&key, CompareOp::Eq, &Value::Int64(0)));
}

#[test]
fn test_find_nodes_in_range_unbounded_max() {
    let store = build_test_store();
    // age >= 40: Jules(40), Mia(45)
    let results = store.find_nodes_in_range("age", Some(&Value::Int64(40)), None, true, true);
    assert_eq!(results.len(), 2);
}

// ===========================================================================
// F. Additional CompactStore coverage
// ===========================================================================

#[test]
fn test_get_nodes_properties_selective_batch() {
    let store = build_test_store();
    let ids = store.nodes_by_label("Person");
    let selective = store.get_nodes_properties_selective_batch(&ids, &[PropertyKey::from("age")]);
    assert_eq!(selective.len(), 5);
    for map in &selective {
        assert!(map.contains_key(&PropertyKey::from("age")));
        assert!(!map.contains_key(&PropertyKey::from("name")));
    }
}

#[test]
fn test_get_edges_properties_selective_batch() {
    let store = build_test_store();
    let person_ids = store.nodes_by_label("Person");
    let edges = store.edges_from(person_ids[0], Direction::Outgoing);
    let edge_ids: Vec<EdgeId> = edges.iter().map(|(_, eid)| *eid).collect();

    let selective =
        store.get_edges_properties_selective_batch(&edge_ids, &[PropertyKey::from("nonexistent")]);
    assert_eq!(selective.len(), edge_ids.len());
    for map in &selective {
        assert!(map.is_empty());
    }
}

#[test]
fn test_get_nodes_properties_batch() {
    let store = build_test_store();
    let ids = store.nodes_by_label("Person");
    let all_props = store.get_nodes_properties_batch(&ids);
    assert_eq!(all_props.len(), 5);
    for map in &all_props {
        assert!(map.contains_key(&PropertyKey::from("age")));
        assert!(map.contains_key(&PropertyKey::from("name")));
    }
}

#[test]
fn test_find_nodes_in_range_exclusive_bounds() {
    let store = build_test_store();
    // age in (25, 45) exclusive: Gus(30), Vincent(35), Jules(40)
    let results = store.find_nodes_in_range(
        "age",
        Some(&Value::Int64(25)),
        Some(&Value::Int64(45)),
        false,
        false,
    );
    assert_eq!(results.len(), 3);
}

#[test]
fn test_edge_property_access() {
    let store = CompactStoreBuilder::new()
        .node_table("Person", |t| t.column_dict("name", &["Alix", "Gus"]))
        .rel_table("RATED", "Person", "Person", |r| {
            r.edges([(0, 1)]).column_bitpacked("score", &[9], 4)
        })
        .build()
        .unwrap();

    let alix = person_at(&store, 0);
    let edges = store.edges_from(alix, Direction::Outgoing);
    let (_, eid) = edges[0];
    assert_eq!(
        store.get_edge_property(eid, &PropertyKey::from("score")),
        Some(Value::Int64(9))
    );
    assert_eq!(
        store.get_edge_property(eid, &PropertyKey::from("nonexistent")),
        None
    );
}

#[test]
fn test_memory_bytes_nonzero() {
    let store = build_test_store();
    assert!(store.memory_bytes() > 0);
}

#[test]
fn test_value_in_range_incomparable() {
    let store = build_test_store();
    // Search with string bounds on an integer column: no matches (incomparable types)
    let results = store.find_nodes_in_range(
        "age",
        Some(&Value::String(ArcStr::from("low"))),
        Some(&Value::String(ArcStr::from("high"))),
        true,
        true,
    );
    assert!(results.is_empty(), "{results:?}");
}

// ---------------------------------------------------------------------------
// GraphStore: invalid ID handling
// ---------------------------------------------------------------------------

#[test]
fn test_get_node_invalid_table_id() {
    let store = build_test_store();
    // Table ID 99 doesn't exist: should return None.
    let fake_id = super::id::encode_node_id(99, 0);
    assert!(store.get_node(fake_id).is_none());
}

#[test]
fn test_get_node_out_of_bounds_offset() {
    let store = build_test_store();
    // Table 0 (Person) has 5 nodes. Offset 999 is out of bounds.
    let fake_id = super::id::encode_node_id(0, 999);
    assert!(store.get_node(fake_id).is_none());
}

#[test]
fn test_get_edge_invalid_rel_table_id() {
    let store = build_test_store();
    let fake_id = super::id::encode_edge_id(99, 0);
    assert!(store.get_edge(fake_id).is_none());
}

#[test]
fn test_get_edge_out_of_bounds_position() {
    let store = build_test_store();
    // Rel table 0 exists but position 999 is beyond edge count.
    let fake_id = super::id::encode_edge_id(0, 999);
    assert!(store.get_edge(fake_id).is_none());
}

#[test]
fn test_get_node_property_invalid_id() {
    let store = build_test_store();
    let fake_id = super::id::encode_node_id(99, 0);
    assert!(
        store
            .get_node_property(fake_id, &PropertyKey::new("name"))
            .is_none()
    );
}

#[test]
fn test_get_edge_property_invalid_id() {
    let store = build_test_store();
    let fake_id = super::id::encode_edge_id(99, 0);
    assert!(
        store
            .get_edge_property(fake_id, &PropertyKey::new("score"))
            .is_none()
    );
}

// ---------------------------------------------------------------------------
// GraphStore: zone map pruning paths
// ---------------------------------------------------------------------------

#[test]
fn test_find_nodes_by_property_zone_map_prunes() {
    let store = build_test_store();
    // Search for age = 999, which is outside the zone map [25, 45].
    // The zone map should prune the Person table entirely.
    let results = store.find_nodes_by_property("age", &Value::Int64(999));
    assert!(results.is_empty(), "{results:?}");
}

#[test]
fn test_find_nodes_by_property_nonexistent_property() {
    let store = build_test_store();
    // Property "color" doesn't exist: no zone map, no column, no results.
    let results = store.find_nodes_by_property("color", &Value::Int64(1));
    assert!(results.is_empty(), "{results:?}");
}

#[test]
fn test_find_nodes_in_range_zone_map_prunes_min() {
    let store = build_test_store();
    // Range [100, 200]: min is above zone map max (45), should prune.
    let results = store.find_nodes_in_range(
        "age",
        Some(&Value::Int64(100)),
        Some(&Value::Int64(200)),
        true,
        true,
    );
    assert!(results.is_empty(), "{results:?}");
}

#[test]
fn test_find_nodes_in_range_zone_map_prunes_max() {
    let store = build_test_store();
    // Range [1, 10]: max is below zone map min (25), should prune.
    let results = store.find_nodes_in_range(
        "age",
        Some(&Value::Int64(1)),
        Some(&Value::Int64(10)),
        true,
        true,
    );
    assert!(results.is_empty(), "{results:?}");
}

// ---------------------------------------------------------------------------
// Phase 4b: find_nodes_in_range_iter (lazy block-skip)
// ---------------------------------------------------------------------------

#[test]
fn alix_find_nodes_in_range_iter_matches_eager() {
    let store = build_test_store();
    let min = Value::Int64(30);
    let max = Value::Int64(40);

    let mut from_iter: Vec<NodeId> = store
        .find_nodes_in_range_iter("age", Some(&min), Some(&max), true, true)
        .collect();
    let mut eager = store.find_nodes_in_range("age", Some(&min), Some(&max), true, true);

    from_iter.sort_unstable();
    eager.sort_unstable();
    assert_eq!(from_iter, eager);
}

#[test]
fn gus_find_nodes_in_range_iter_skips_disjoint_blocks() {
    // Build a single-label store with > DEFAULT_BLOCK_ROWS values spread
    // across 3 blocks. Per-block zone maps must let the iterator skip
    // blocks whose stats are disjoint from the query range.
    let mut values: Vec<u64> = Vec::with_capacity(3072);
    for v in 0..3072u64 {
        values.push(v);
    }
    let store = CompactStoreBuilder::new()
        .node_table("Big", |t| t.column_bitpacked("v", &values, 12))
        .build()
        .unwrap();

    // Range [1500, 1700] hits block 1 only (rows 1024..2048).
    let min = Value::Int64(1500);
    let max = Value::Int64(1700);
    let from_iter: Vec<NodeId> = store
        .find_nodes_in_range_iter("v", Some(&min), Some(&max), true, true)
        .collect();
    let eager = store.find_nodes_in_range("v", Some(&min), Some(&max), true, true);

    let mut from_iter_sorted = from_iter.clone();
    let mut eager_sorted = eager.clone();
    from_iter_sorted.sort_unstable();
    eager_sorted.sort_unstable();
    assert_eq!(from_iter_sorted, eager_sorted);
    // Sanity: 1500..=1700 = 201 nodes.
    assert_eq!(from_iter.len(), 201);
}

#[test]
fn vincent_find_nodes_in_range_iter_whole_table_pruned() {
    // Range entirely outside all zone maps: iterator yields nothing.
    let store = build_test_store();
    let min = Value::Int64(100);
    let max = Value::Int64(200);
    let result: Vec<NodeId> = store
        .find_nodes_in_range_iter("age", Some(&min), Some(&max), true, true)
        .collect();
    assert!(result.is_empty(), "{result:?}");
}

#[test]
fn jules_find_nodes_in_range_iter_open_bounds() {
    let store = build_test_store();
    // age <= 30: Alix(25), Gus(30)
    let max = Value::Int64(30);
    let result: Vec<NodeId> = store
        .find_nodes_in_range_iter("age", None, Some(&max), false, true)
        .collect();
    assert_eq!(result.len(), 2);
}

#[test]
fn mia_find_nodes_in_range_iter_missing_property_yields_nothing() {
    let store = build_test_store();
    let min = Value::Int64(0);
    let max = Value::Int64(1000);
    let result: Vec<NodeId> = store
        .find_nodes_in_range_iter("nonexistent", Some(&min), Some(&max), true, true)
        .collect();
    assert!(result.is_empty(), "{result:?}");
}

#[test]
fn test_node_property_might_match_no_zone_map() {
    let store = build_test_store();
    // Property "color" has no zone map: should conservatively return true.
    assert!(store.node_property_might_match(
        &PropertyKey::new("color"),
        CompareOp::Eq,
        &Value::Int64(1),
    ));
}

#[test]
fn test_node_property_might_match_zone_map_rejects() {
    // Use a single-label store so there's no table without a zone map
    // for the property (which would trigger the conservative true path).
    let store = CompactStoreBuilder::new()
        .node_table("Person", |t| t.column_bitpacked("age", &[25, 30, 35], 6))
        .build()
        .unwrap();

    // age zone map is [25, 35]. Asking "age == 999" should be rejected.
    assert!(!store.node_property_might_match(
        &PropertyKey::new("age"),
        CompareOp::Eq,
        &Value::Int64(999),
    ));
}

#[test]
fn test_node_property_might_match_multi_table_conservative() {
    let store = build_test_store();
    // City table has no zone map for "age", so even though Person's
    // zone map [25,45] would reject 999, the function conservatively
    // returns true because City *might* have matching values.
    assert!(store.node_property_might_match(
        &PropertyKey::new("age"),
        CompareOp::Eq,
        &Value::Int64(999),
    ));
}

// ---------------------------------------------------------------------------
// RelTable: source_node_id, dest_node_id, memory_bytes
// ---------------------------------------------------------------------------

#[test]
fn test_rel_table_source_node_id_out_of_bounds() {
    let store = build_test_store();
    let rt = store.rel_tables_for_type("LIVES_IN")[0];
    // CSR position far beyond edge count should return None.
    assert!(rt.source_node_id(9999).is_none());
}

#[test]
fn test_rel_table_dest_node_id_out_of_bounds() {
    let store = build_test_store();
    let rt = store.rel_tables_for_type("LIVES_IN")[0];
    assert!(rt.dest_node_id(9999).is_none());
}

#[test]
fn test_rel_table_memory_bytes_nonzero() {
    let store = build_test_store();
    let rt = store.rel_tables_for_type("LIVES_IN")[0];
    assert!(rt.memory_bytes() > 0);
    // With backward CSR, memory should be higher.
    let knows = store.rel_tables_for_type("KNOWS")[0];
    assert!(knows.memory_bytes() > 0);
}

#[test]
fn test_rel_table_no_backward_dest_node_id() {
    // Build a store without backward CSR.
    let store = CompactStoreBuilder::new()
        .node_table("A", |t| t.column_bitpacked("v", &[1, 2], 4))
        .rel_table("LINK", "A", "A", |r| r.edges([(0, 1)]).backward(false))
        .build()
        .unwrap();

    let rt = store.rel_tables_for_type("LINK")[0];
    // source_node_id should work (forward CSR).
    assert!(rt.source_node_id(0).is_some());
    // dest_node_id should also work (uses forward CSR only).
    assert!(rt.dest_node_id(0).is_some());
    // But in_degree should return None (no backward CSR).
    assert!(rt.in_degree(0).is_none());
}

// ---------------------------------------------------------------------------
// NodeTable: memory_bytes, property-not-found
// ---------------------------------------------------------------------------

#[test]
fn test_node_table_memory_bytes_nonzero() {
    let store = build_test_store();
    let nt = store.node_table("Person").unwrap();
    assert!(nt.memory_bytes() > 0);
}

#[test]
fn test_node_table_get_property_nonexistent_key() {
    let store = build_test_store();
    let nt = store.node_table("Person").unwrap();
    assert!(
        nt.get_property(0, &PropertyKey::new("nonexistent"))
            .is_none()
    );
}

#[test]
fn test_node_table_get_all_properties_out_of_bounds() {
    let store = build_test_store();
    let nt = store.node_table("Person").unwrap();
    // Person table has 5 rows. Offset 999 is out of bounds.
    let all = nt.get_all_properties(999);
    assert!(all.is_empty());
}

// ---------------------------------------------------------------------------
// CSR: edge_data_at, boundary conditions
// ---------------------------------------------------------------------------

#[test]
fn test_csr_edge_data_at_no_data() {
    use super::csr::CsrAdjacency;

    let csr = CsrAdjacency::from_sorted_edges(3, &[(0, 1), (1, 2)]);
    // No edge_data set: should return None.
    assert!(!csr.has_edge_data());
    assert!(csr.edge_data_at(0).is_none());
}

#[test]
fn test_csr_edge_data_at_valid() {
    use super::csr::CsrAdjacency;

    let mut csr = CsrAdjacency::from_sorted_edges(3, &[(0, 1), (1, 2)]);
    csr.set_edge_data(vec![10, 20]);
    assert!(csr.has_edge_data());
    assert_eq!(csr.edge_data_at(0), Some(10));
    assert_eq!(csr.edge_data_at(1), Some(20));
    assert_eq!(csr.edge_data_at(2), None); // out of bounds
}

#[test]
fn test_csr_neighbors_out_of_bounds() {
    use super::csr::CsrAdjacency;

    let csr = CsrAdjacency::from_sorted_edges(2, &[(0, 1)]);
    assert_eq!(csr.neighbors(0), &[1]);
    assert!(csr.neighbors(1).is_empty(), "expected empty");
    // Node 99 is out of bounds: should return empty.
    assert!(csr.neighbors(99).is_empty(), "expected empty");
}

#[test]
fn test_csr_source_for_position_zero_degree_nodes() {
    use super::csr::CsrAdjacency;

    // Node 0: no edges, Node 1: edge to 2, Node 2: no edges.
    let csr = CsrAdjacency::from_sorted_edges(3, &[(1, 2)]);
    assert_eq!(csr.degree(0), 0);
    assert_eq!(csr.degree(1), 1);
    assert_eq!(csr.degree(2), 0);

    // Position 0 should map to source node 1 (skipping zero-degree node 0).
    assert_eq!(csr.source_for_position(0), Some(1));
    assert_eq!(csr.source_for_position(1), None);
}

#[test]
fn test_csr_source_for_position_boundary() {
    use super::csr::CsrAdjacency;

    // Node 0: edges to [1, 2], Node 1: edge to [2].
    let csr = CsrAdjacency::from_sorted_edges(2, &[(0, 1), (0, 2), (1, 2)]);
    assert_eq!(csr.source_for_position(0), Some(0)); // first edge of node 0
    assert_eq!(csr.source_for_position(1), Some(0)); // second edge of node 0
    assert_eq!(csr.source_for_position(2), Some(1)); // boundary: first edge of node 1
    assert_eq!(csr.source_for_position(3), None); // out of bounds
}

// ---------------------------------------------------------------------------
// from_graph_store: additional coverage
// ---------------------------------------------------------------------------

#[test]
fn test_from_graph_store_edges_with_sparse_properties() {
    use crate::graph::compact::builder::from_graph_store;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);

    // Edge 1 has "weight", edge 2 has "label", edge 3 has both.
    // This exercises the null-padding logic for sparse edge properties.
    let e1 = store.create_edge(a, b, "LINK");
    store.set_edge_property(e1, "weight", Value::Int64(5));

    let e2 = store.create_edge(b, c, "LINK");
    store.set_edge_property(e2, "label", Value::from("fast"));

    let e3 = store.create_edge(a, c, "LINK");
    store.set_edge_property(e3, "weight", Value::Int64(10));
    store.set_edge_property(e3, "label", Value::from("slow"));

    let compact = from_graph_store(&store).unwrap();

    let ids = compact.nodes_by_label("Node");
    assert_eq!(ids.len(), 3);

    let mut total_edges = 0;
    for &id in &ids {
        total_edges += compact.edges_from(id, Direction::Outgoing).len();
    }
    assert_eq!(total_edges, 3);
}

#[test]
fn test_from_graph_store_multiple_edge_types() {
    use crate::graph::compact::builder::from_graph_store;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    store.set_node_property(a, "name", Value::from("Alix"));
    let b = store.create_node(&["Person"]);
    store.set_node_property(b, "name", Value::from("Gus"));
    let c = store.create_node(&["City"]);
    store.set_node_property(c, "name", Value::from("Amsterdam"));

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "LIVES_IN");
    store.create_edge(b, c, "LIVES_IN");

    let compact = from_graph_store(&store).unwrap();

    let mut edge_types = compact.all_edge_types();
    edge_types.sort();
    assert_eq!(edge_types.len(), 2);
    assert!(edge_types.contains(&"KNOWS".to_string()));
    assert!(edge_types.contains(&"LIVES_IN".to_string()));

    // Person a should have 2 outgoing edges (1 KNOWS + 1 LIVES_IN).
    let person_ids = compact.nodes_by_label("Person");
    let mut max_out = 0;
    for &pid in &person_ids {
        max_out = max_out.max(compact.edges_from(pid, Direction::Outgoing).len());
    }
    assert_eq!(max_out, 2);
}

#[test]
fn test_from_graph_store_nodes_without_edges() {
    use crate::graph::compact::builder::from_graph_store;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Orphan"]);
    store.set_node_property(a, "name", Value::from("solo"));

    let compact = from_graph_store(&store).unwrap();

    let ids = compact.nodes_by_label("Orphan");
    assert_eq!(ids.len(), 1);
    assert_eq!(compact.edge_count(), 0);
    assert!(
        compact.edges_from(ids[0], Direction::Outgoing).is_empty(),
        "expected empty"
    );
}

/// Regression test for GrafeoDB/grafeo#221: `compact()` fails with
/// "duplicate edge type" when the same edge type spans multiple label pairs.
#[test]
fn test_from_graph_store_multiple_label_pairs_same_edge_type() {
    use crate::graph::compact::builder::from_graph_store;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    store.set_node_property(a, "name", Value::from("a"));
    let b = store.create_node(&["B"]);
    store.set_node_property(b, "name", Value::from("b"));
    let c = store.create_node(&["C"]);
    store.set_node_property(c, "name", Value::from("c"));

    store.create_edge(a, b, "CALLS");
    store.create_edge(a, c, "USES_TYPE");

    // This used to fail with DuplicateEdgeType because the validation
    // checked only edge_type, not the full (edge_type, src, dst) triple.
    let compact = from_graph_store(&store).unwrap();

    // Verify both edge types survived.
    let a_ids = compact.nodes_by_label("A");
    assert_eq!(a_ids.len(), 1);

    let outgoing = compact.edges_from(a_ids[0], Direction::Outgoing);
    assert_eq!(outgoing.len(), 2);
}

/// Same edge type between different label pairs (e.g. code dependency graph).
#[test]
fn test_from_graph_store_same_edge_type_different_label_pairs() {
    use crate::graph::compact::builder::from_graph_store;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();

    let method = store.create_node(&["Method"]);
    store.set_node_property(method, "name", Value::from("main"));
    let class = store.create_node(&["Class"]);
    store.set_node_property(class, "name", Value::from("App"));
    let other_method = store.create_node(&["Method"]);
    store.set_node_property(other_method, "name", Value::from("helper"));

    // CALLS from Method->Method and Method->Class (same edge type, different dst labels)
    store.create_edge(method, other_method, "CALLS");
    store.create_edge(method, class, "CALLS");

    let compact = from_graph_store(&store).unwrap();

    let method_ids = compact.nodes_by_label("Method");
    assert_eq!(method_ids.len(), 2);

    // Find the "main" method node and verify it has 2 outgoing CALLS edges.
    let main_id = method_ids
        .iter()
        .find(|&&id| {
            compact.get_node(id).is_some_and(|n| {
                n.properties.get(&PropertyKey::from("name")) == Some(&Value::from("main"))
            })
        })
        .unwrap();

    let outgoing = compact.edges_from(*main_id, Direction::Outgoing);
    assert_eq!(outgoing.len(), 2);
    for (_target, eid) in &outgoing {
        let edge = compact.get_edge(*eid).unwrap();
        assert_eq!(edge.edge_type.as_str(), "CALLS");
    }
}

/// Builder allows same edge type with different label pairs.
#[test]
fn test_builder_same_edge_type_different_labels_ok() {
    let store = CompactStoreBuilder::new()
        .node_table("A", |t| t.column_dict("name", &["a"]))
        .node_table("B", |t| t.column_dict("name", &["b"]))
        .node_table("C", |t| t.column_dict("name", &["c"]))
        .rel_table("CALLS", "A", "B", |r| r.edges([(0, 0)]))
        .rel_table("CALLS", "A", "C", |r| r.edges([(0, 0)]))
        .build();

    assert!(store.is_ok());
    let store = store.unwrap();

    let a_ids = store.nodes_by_label("A");
    assert_eq!(a_ids.len(), 1);

    let outgoing = store.edges_from(a_ids[0], Direction::Outgoing);
    assert_eq!(outgoing.len(), 2);
}

/// Statistics for an edge type spanning multiple rel tables are aggregated.
#[test]
fn test_statistics_aggregate_multi_table_edge_type() {
    let store = CompactStoreBuilder::new()
        .node_table("A", |t| t.column_dict("name", &["a1", "a2"]))
        .node_table("B", |t| t.column_dict("name", &["b1"]))
        .node_table("C", |t| t.column_dict("name", &["c1", "c2", "c3"]))
        .rel_table("LINK", "A", "B", |r| r.edges([(0, 0), (1, 0)]))
        .rel_table("LINK", "A", "C", |r| r.edges([(0, 0), (0, 1), (1, 2)]))
        .build()
        .unwrap();

    let stats = store.statistics();
    let link_stats = stats
        .get_edge_type("LINK")
        .expect("LINK stats should exist");
    // 2 edges (A->B) + 3 edges (A->C) = 5 total
    assert_eq!(link_stats.edge_count, 5);
    assert_eq!(stats.total_edges, 5);
}

// ── ID-preserving CompactStore tests ───────────────────────────────

#[test]
fn test_preserving_ids_get_node() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let alix = store.create_node(&["Person"]);
    store.set_node_property(alix, "name", Value::from("Alix"));
    store.set_node_property(alix, "age", Value::Int64(30));

    let gus = store.create_node(&["Person"]);
    store.set_node_property(gus, "name", Value::from("Gus"));
    store.set_node_property(gus, "age", Value::Int64(25));

    let amsterdam = store.create_node(&["City"]);
    store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

    let compact = from_graph_store_preserving_ids(&store).unwrap();
    assert!(compact.preserves_ids());

    // Original NodeIds must work for lookup.
    let alix_node = compact
        .get_node(alix)
        .expect("Alix should be found by original ID");
    assert_eq!(
        alix_node.properties.get(&PropertyKey::new("name")),
        Some(&Value::String(ArcStr::from("Alix")))
    );

    let gus_node = compact
        .get_node(gus)
        .expect("Gus should be found by original ID");
    assert_eq!(
        gus_node.properties.get(&PropertyKey::new("name")),
        Some(&Value::String(ArcStr::from("Gus")))
    );

    let amsterdam_node = compact
        .get_node(amsterdam)
        .expect("Amsterdam should be found by original ID");
    assert_eq!(
        amsterdam_node.properties.get(&PropertyKey::new("name")),
        Some(&Value::String(ArcStr::from("Amsterdam")))
    );
}

#[test]
fn test_preserving_ids_node_ids_returns_originals() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Item"]);
    let b = store.create_node(&["Item"]);
    let c = store.create_node(&["Item"]);

    let compact = from_graph_store_preserving_ids(&store).unwrap();

    let mut ids = compact.node_ids();
    ids.sort_unstable();
    let mut expected = vec![a, b, c];
    expected.sort_unstable();
    assert_eq!(ids, expected, "node_ids() should return original IDs");
}

#[test]
fn test_preserving_ids_nodes_by_label() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let p1 = store.create_node(&["Person"]);
    let p2 = store.create_node(&["Person"]);
    let c1 = store.create_node(&["City"]);

    let compact = from_graph_store_preserving_ids(&store).unwrap();

    let mut person_ids = compact.nodes_by_label("Person");
    person_ids.sort_unstable();
    let mut expected = vec![p1, p2];
    expected.sort_unstable();
    assert_eq!(person_ids, expected);

    let city_ids = compact.nodes_by_label("City");
    assert_eq!(city_ids, vec![c1]);
}

#[test]
fn test_preserving_ids_property_access() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Item"]);
    store.set_node_property(a, "score", Value::Int64(42));

    let compact = from_graph_store_preserving_ids(&store).unwrap();

    assert_eq!(
        compact.get_node_property(a, &PropertyKey::new("score")),
        Some(Value::Int64(42)),
        "property lookup by original ID should work"
    );
}

#[test]
fn test_preserving_ids_traversal() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let alix = store.create_node(&["Person"]);
    store.set_node_property(alix, "name", Value::from("Alix"));
    let gus = store.create_node(&["Person"]);
    store.set_node_property(gus, "name", Value::from("Gus"));
    let amsterdam = store.create_node(&["City"]);
    store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

    let _e1 = store.create_edge(alix, amsterdam, "LIVES_IN");
    let _e2 = store.create_edge(gus, amsterdam, "LIVES_IN");

    let compact = from_graph_store_preserving_ids(&store).unwrap();

    // Outgoing neighbors should use original IDs.
    let alix_neighbors = compact.neighbors(alix, crate::graph::Direction::Outgoing);
    assert_eq!(alix_neighbors.len(), 1);
    assert_eq!(
        alix_neighbors[0], amsterdam,
        "neighbor should be the original Amsterdam ID"
    );

    // Incoming edges on Amsterdam should reference original Person IDs.
    let incoming = compact.edges_from(amsterdam, crate::graph::Direction::Incoming);
    assert_eq!(incoming.len(), 2);
    let mut sources: Vec<NodeId> = incoming.iter().map(|(target, _)| *target).collect();
    sources.sort_unstable();
    let mut expected = vec![alix, gus];
    expected.sort_unstable();
    assert_eq!(sources, expected);
}

#[test]
fn test_preserving_ids_edge_lookup() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let e = store.create_edge(a, b, "LINK");
    store.set_edge_property(e, "weight", Value::Int64(5));

    let compact = from_graph_store_preserving_ids(&store).unwrap();

    // Get the edge via traversal.
    let edges = compact.edges_from(a, crate::graph::Direction::Outgoing);
    assert_eq!(edges.len(), 1);
    let (target, eid) = edges[0];
    assert_eq!(target, b, "target should be original ID");

    // Look up the edge by its compact ID (returned from edges_from).
    let edge = compact.get_edge(eid).expect("edge should exist");
    assert_eq!(edge.src, a, "edge.src should be original ID");
    assert_eq!(edge.dst, b, "edge.dst should be original ID");
    assert_eq!(
        edge.properties.get(&PropertyKey::new("weight")),
        Some(&Value::Int64(5))
    );
}

#[test]
fn test_preserving_ids_find_nodes_by_property() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Item"]);
    store.set_node_property(a, "score", Value::Int64(10));
    let b = store.create_node(&["Item"]);
    store.set_node_property(b, "score", Value::Int64(20));
    let c = store.create_node(&["Item"]);
    store.set_node_property(c, "score", Value::Int64(10));

    let compact = from_graph_store_preserving_ids(&store).unwrap();

    let matches = compact.find_nodes_by_property("score", &Value::Int64(10));
    assert_eq!(matches.len(), 2);
    let mut match_set: Vec<NodeId> = matches;
    match_set.sort_unstable();
    let mut expected = vec![a, c];
    expected.sort_unstable();
    assert_eq!(match_set, expected, "find results should use original IDs");
}

#[test]
fn test_preserving_ids_degree() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    let c = store.create_node(&["Node"]);
    store.create_edge(a, b, "LINK");
    store.create_edge(a, c, "LINK");

    let compact = from_graph_store_preserving_ids(&store).unwrap();

    assert_eq!(compact.out_degree(a), 2);
    assert_eq!(compact.in_degree(b), 1);
    assert_eq!(compact.in_degree(c), 1);
}

#[test]
fn test_preserving_ids_edge_type() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Node"]);
    let b = store.create_node(&["Node"]);
    store.create_edge(a, b, "KNOWS");

    let compact = from_graph_store_preserving_ids(&store).unwrap();

    let edges = compact.edges_from(a, crate::graph::Direction::Outgoing);
    assert_eq!(edges.len(), 1);
    let eid = edges[0].1;
    assert_eq!(compact.edge_type(eid), Some(ArcStr::from("KNOWS")));
}

#[test]
fn test_preserving_ids_empty_store() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::lpg::LpgStore;

    let store = LpgStore::new().unwrap();
    let compact = from_graph_store_preserving_ids(&store).unwrap();
    assert!(compact.preserves_ids());
    assert_eq!(compact.node_count(), 0);
    assert_eq!(compact.edge_count(), 0);
}

// ── Task 7: failing tests for ColumnCodec slice accessors (Task 8 adds impl) ──

#[test]
fn test_raw_i64_column_built_from_vec_exposes_slice() {
    use crate::graph::compact::column::ColumnCodec;
    let values: Vec<i64> = (0..100).collect();
    let col = ColumnCodec::raw_i64(values.clone());
    let slice = col.as_raw_i64_slice();
    assert_eq!(slice, Some(values.as_slice()));
}

#[test]
fn test_float64_column_built_from_vec_exposes_slice() {
    use crate::graph::compact::column::ColumnCodec;
    let values: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
    let col = ColumnCodec::float64(values.clone());
    let slice = col.as_float64_slice();
    assert_eq!(slice, Some(values.as_slice()));
}

#[test]
fn test_raw_i64_inline_and_mapped_find_eq_match() {
    use crate::graph::compact::column::ColumnCodec;
    use grafeo_common::types::Value;

    let values: Vec<i64> = (0..1_000).map(|i| i % 13).collect();
    let inline = ColumnCodec::raw_i64(values.clone());

    // Serialize Inline to bytes and rebuild via the Mapped path.
    let mut buf = bytes::BytesMut::with_capacity(values.len() * 8);
    for &v in &values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mapped = ColumnCodec::raw_i64_from_bytes(buf.freeze());

    let target = Value::Int64(7);
    assert_eq!(inline.find_eq(&target), mapped.find_eq(&target));
    assert!(!inline.find_eq(&target).is_empty(), "expected non-empty");
}

// ===========================================================================
// Nodes with several labels: a table per label set, each label reads them
// ===========================================================================

/// Vincent is a Person and an Actor, Jules a Person and Mia an Actor.
/// Returns the store and the ids of Vincent, Jules and Mia.
fn people_and_actors() -> (crate::graph::lpg::LpgStore, NodeId, NodeId, NodeId) {
    let store = crate::graph::lpg::LpgStore::new().unwrap();
    let vincent = store.create_node(&["Person", "Actor"]);
    store.set_node_property(vincent, "name", Value::from("Vincent"));
    let jules = store.create_node(&["Person"]);
    store.set_node_property(jules, "name", Value::from("Jules"));
    let mia = store.create_node(&["Actor"]);
    store.set_node_property(mia, "name", Value::from("Mia"));
    (store, vincent, jules, mia)
}

/// The labels of node `id` in `store`, sorted.
fn sorted_labels_of(store: &dyn GraphStore, id: NodeId) -> Vec<String> {
    let mut labels: Vec<String> = store
        .get_node(id)
        .map(|node| node.labels.iter().map(ToString::to_string).collect())
        .unwrap_or_default();
    labels.sort();
    labels
}

fn sorted_ids(mut ids: Vec<NodeId>) -> Vec<NodeId> {
    ids.sort_unstable();
    ids
}

#[test]
fn a_compacted_node_keeps_each_of_its_labels() {
    use crate::graph::compact::from_graph_store_preserving_ids;

    let (store, vincent, jules, mia) = people_and_actors();
    let compact = from_graph_store_preserving_ids(&store).unwrap();

    assert_eq!(sorted_labels_of(&compact, vincent), ["Actor", "Person"]);
    assert_eq!(sorted_labels_of(&compact, jules), ["Person"]);
    assert_eq!(
        sorted_ids(compact.nodes_by_label("Person")),
        sorted_ids(vec![vincent, jules]),
        "a Person scan finds Vincent too"
    );
    assert_eq!(
        sorted_ids(compact.nodes_by_label("Actor")),
        sorted_ids(vec![vincent, mia]),
        "an Actor scan finds Vincent too"
    );
    assert!(
        compact.nodes_by_label("Actor|Person").is_empty(),
        "no label is named after a label set"
    );
    assert_eq!(compact.nodes_by_label_count("Person"), 2);
    assert_eq!(compact.nodes_by_label_count("Actor"), 2);
    assert!(compact.node_in_label(vincent, "Person"));
    assert!(compact.node_in_label(vincent, "Actor"));
    assert!(!compact.node_in_label(jules, "Actor"));
    assert!(
        (compact.estimate_label_cardinality("Actor") - 2.0).abs() < 1e-9,
        "the planner's estimate counts each node with the label: {}",
        compact.estimate_label_cardinality("Actor")
    );
    let statistics = compact.statistics();
    for label in ["Person", "Actor"] {
        assert_eq!(
            statistics.get_label(label).map(|s| s.node_count),
            Some(2),
            "{label} statistics"
        );
    }
    assert!(statistics.get_label("Actor|Person").is_none());
    let mut labels = compact.all_labels();
    labels.sort();
    assert_eq!(labels, ["Actor", "Person"]);
}

/// A label scan reads every table whose label set has the label, and
/// returns their rows in id order: Jules (id 0) is a row of the Person
/// table, which comes after the table of Vincent's set (Actor and Person)
/// because the tables follow the labels in name order.
#[test]
fn a_label_scan_spans_every_table_whose_set_has_the_label() {
    use crate::graph::compact::from_graph_store_preserving_ids;

    let store = crate::graph::lpg::LpgStore::new().unwrap();
    let jules = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person", "Actor"]);
    let mia = store.create_node(&["Actor"]);
    let compact = from_graph_store_preserving_ids(&store).unwrap();
    assert_eq!(compact.nodes_by_label("Person"), vec![jules, vincent]);
    assert_eq!(compact.nodes_by_label("Actor"), vec![vincent, mia]);

    // Without preserved ids, the ids encode the table and the row.
    let compact = super::from_graph_store(&store).unwrap();
    let people = compact.nodes_by_label("Person");
    assert_eq!(people.len(), 2, "{people:?}");
    assert_eq!(people, sorted_ids(people.clone()), "in id order");
}

/// The layered store reads the base's labels: its scans, counts and
/// statistics count Vincent under each of his labels, also after a delete.
#[test]
fn a_layered_store_reads_each_label_of_a_compacted_node() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::compact::layered::LayeredStore;
    use crate::graph::traits::GraphStoreMut;

    let (store, vincent, jules, mia) = people_and_actors();
    // Two more Persons, so that the base holds more of them than there are
    // tombstones and copies below.
    let alix = store.create_node(&["Person"]);
    let alix_in_berlin = store.create_node(&["Person"]);
    let base = from_graph_store_preserving_ids(&store).unwrap();
    let layered = LayeredStore::new(
        base,
        store.next_node_id() - 1,
        store.next_edge_id().saturating_sub(1),
    )
    .unwrap();
    let gus = layered.create_node(&["Person"]);

    assert_eq!(sorted_labels_of(&layered, vincent), ["Actor", "Person"]);
    assert_eq!(
        sorted_ids(layered.nodes_by_label("Person")),
        sorted_ids(vec![vincent, jules, alix, alix_in_berlin, gus])
    );
    assert_eq!(layered.nodes_by_label_count("Person"), 5);
    let statistics = layered.statistics();
    assert_eq!(
        statistics.get_label("Person").map(|s| s.node_count),
        Some(5)
    );
    assert_eq!(statistics.get_label("Actor").map(|s| s.node_count), Some(2));

    assert!(layered.delete_node(vincent));
    // Four base Persons, one tombstone and one id in the overlay: the count
    // checks the labels of the deleted id, Person being the second of his.
    assert_eq!(
        layered.nodes_by_label_count("Person"),
        4,
        "Vincent is no Person left"
    );
    assert_eq!(
        sorted_ids(layered.nodes_by_label("Person")),
        sorted_ids(vec![jules, alix, alix_in_berlin, gus])
    );
    assert_eq!(
        layered.nodes_by_label_count("Actor"),
        1,
        "Mia is the Actor left"
    );
    assert_eq!(layered.nodes_by_label("Actor"), vec![mia]);
}

/// `drop_base` leaves the overlay alone: the base's nodes, the tombstone of
/// a base delete and the ids of base nodes copied into the overlay go, and
/// the copies and the overlay's own nodes stay.
#[test]
fn dropping_the_base_leaves_only_the_overlay() {
    use crate::graph::compact::from_graph_store_preserving_ids;
    use crate::graph::compact::layered::LayeredStore;
    use crate::graph::traits::GraphStoreMut;

    let (store, vincent, jules, mia) = people_and_actors();
    let layered = LayeredStore::new(
        from_graph_store_preserving_ids(&store).unwrap(),
        store.next_node_id() - 1,
        store.next_edge_id().saturating_sub(1),
    )
    .unwrap();
    let gus = layered.create_node(&["Person"]);
    assert!(layered.delete_node(mia));
    layered.set_node_property(jules, "age", Value::Int64(19));
    assert_eq!(
        layered.overlay_mutation_count(),
        3,
        "Gus, the copy of Jules and the tombstone of Mia"
    );

    let empty = layered.drop_base();
    assert_eq!((empty.node_count(), empty.edge_count()), (0, 0));
    assert!(std::sync::Arc::ptr_eq(&empty, &layered.base_store_arc()));
    assert_eq!(
        layered.overlay_mutation_count(),
        0,
        "no tombstone and no copied id is left"
    );
    assert!(
        layered.snapshot_deleted_node_ids().is_empty(),
        "no delete of a base node is left to write"
    );
    assert!(layered.get_node(vincent).is_none(), "the base is gone");
    assert_eq!(
        sorted_ids(layered.nodes_by_label("Person")),
        sorted_ids(vec![jules, gus]),
        "the copy of Jules and Gus stay"
    );
    assert_eq!(layered.node_count(), 2);
}

/// The builder's single-label tables are named by their label, also when it
/// holds the separator or the escape character of a label set's key.
#[test]
fn a_builder_label_holding_the_separator_is_one_label() {
    let store = CompactStoreBuilder::new()
        .node_table("In|Out", |t| t.column_dict("name", &["Alix"]))
        .node_table("C:\\Data", |t| t.column_dict("name", &["Gus"]))
        .rel_table("LINKS", "In|Out", "C:\\Data", |r| {
            r.edges([(0, 0)]).backward(true)
        })
        .build()
        .unwrap();
    let alix = store.nodes_by_label("In|Out");
    assert_eq!(alix.len(), 1);
    assert_eq!(sorted_labels_of(&store, alix[0]), ["In|Out"]);
    assert!(store.nodes_by_label("In").is_empty(), "In|Out is one label");
    assert_eq!(store.node_table("In|Out").map(|t| t.len()), Some(1));
    let gus = store.nodes_by_label("C:\\Data");
    assert_eq!(sorted_labels_of(&store, gus[0]), ["C:\\Data"]);
    assert_eq!(store.neighbors(alix[0], Direction::Outgoing), gus);
}
