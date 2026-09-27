use super::*;
use crate::graph::Direction;
use crate::graph::lpg::property::CompareOp;
use grafeo_common::types::TransactionId;

#[test]
fn test_create_node() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person"]);
    assert!(id.is_valid());

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(!node.has_label("Animal"));
}

#[test]
fn test_create_node_with_props() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("age", Value::from(30i64))],
    );

    let node = store.get_node(id).unwrap();
    assert_eq!(
        node.get_property("name").and_then(|v| v.as_str()),
        Some("Alix")
    );
    assert_eq!(
        node.get_property("age").and_then(|v| v.as_int64()),
        Some(30)
    );
}

#[test]
fn test_delete_node() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person"]);
    assert_eq!(store.node_count(), 1);

    assert!(store.delete_node(id));
    assert_eq!(store.node_count(), 0);
    assert!(store.get_node(id).is_none());

    // Double delete should return false
    assert!(!store.delete_node(id));
}

#[test]
fn test_create_edge() {
    let store = LpgStore::new().unwrap();

    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);

    let edge_id = store.create_edge(alix, gus, "KNOWS");
    assert!(edge_id.is_valid());

    let edge = store.get_edge(edge_id).unwrap();
    assert_eq!(edge.src, alix);
    assert_eq!(edge.dst, gus);
    assert_eq!(edge.edge_type.as_str(), "KNOWS");
}

#[test]
fn test_neighbors() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");

    let outgoing: Vec<_> = store.neighbors(a, Direction::Outgoing).collect();
    assert_eq!(outgoing.len(), 2);
    assert!(outgoing.contains(&b));
    assert!(outgoing.contains(&c));

    let incoming: Vec<_> = store.neighbors(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 1);
    assert!(incoming.contains(&a));
}

#[test]
fn test_nodes_by_label() {
    let store = LpgStore::new().unwrap();

    let p1 = store.create_node(&["Person"]);
    let p2 = store.create_node(&["Person"]);
    let _a = store.create_node(&["Animal"]);

    let persons = store.nodes_by_label("Person");
    assert_eq!(persons.len(), 2);
    assert!(persons.contains(&p1));
    assert!(persons.contains(&p2));

    let animals = store.nodes_by_label("Animal");
    assert_eq!(animals.len(), 1);
}

#[test]
fn test_delete_edge() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    assert_eq!(store.edge_count(), 1);

    assert!(store.delete_edge(edge_id));
    assert_eq!(store.edge_count(), 0);
    assert!(store.get_edge(edge_id).is_none());
}

// === New tests for improved coverage ===

#[test]
fn test_lpg_store_config() {
    // Test with_config
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 100,
        initial_edge_capacity: 200,
    };
    let store = LpgStore::with_config(config).unwrap();

    // Store should work but without backward adjacency
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    store.create_edge(a, b, "KNOWS");

    // Outgoing should work
    let outgoing: Vec<_> = store.neighbors(a, Direction::Outgoing).collect();
    assert_eq!(outgoing.len(), 1);

    // Incoming should be empty (no backward adjacency)
    let incoming: Vec<_> = store.neighbors(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 0);
}

#[test]
fn test_epoch_management() {
    let store = LpgStore::new().unwrap();

    let epoch0 = store.current_epoch();
    assert_eq!(epoch0.as_u64(), 0);

    let epoch1 = store.new_epoch();
    assert_eq!(epoch1.as_u64(), 1);

    let current = store.current_epoch();
    assert_eq!(current.as_u64(), 1);
}

#[test]
fn test_node_properties() {
    let store = LpgStore::new().unwrap();
    let id = store.create_node(&["Person"]);

    // Set and get property
    store.set_node_property(id, "name", Value::from("Alix"));
    let name = store.get_node_property(id, &"name".into());
    assert!(matches!(name, Some(Value::String(s)) if s.as_str() == "Alix"));

    // Update property
    store.set_node_property(id, "name", Value::from("Gus"));
    let name = store.get_node_property(id, &"name".into());
    assert!(matches!(name, Some(Value::String(s)) if s.as_str() == "Gus"));

    // Remove property
    let old = store.remove_node_property(id, "name");
    assert!(matches!(old, Some(Value::String(s)) if s.as_str() == "Gus"));

    // Property should be gone
    let name = store.get_node_property(id, &"name".into());
    assert!(name.is_none());

    // Remove non-existent property
    let none = store.remove_node_property(id, "nonexistent");
    assert!(none.is_none());
}

#[test]
fn test_edge_properties() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    // Set and get property
    store.set_edge_property(edge_id, "since", Value::from(2020i64));
    let since = store.get_edge_property(edge_id, &"since".into());
    assert_eq!(since.and_then(|v| v.as_int64()), Some(2020));

    // Remove property
    let old = store.remove_edge_property(edge_id, "since");
    assert_eq!(old.and_then(|v| v.as_int64()), Some(2020));

    let since = store.get_edge_property(edge_id, &"since".into());
    assert!(since.is_none());
}

#[test]
fn test_add_remove_label() {
    let store = LpgStore::new().unwrap();
    let id = store.create_node(&["Person"]);

    // Add new label
    assert!(store.add_label(id, "Employee"));

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));

    // Adding same label again should fail
    assert!(!store.add_label(id, "Employee"));

    // Remove label
    assert!(store.remove_label(id, "Employee"));

    let node = store.get_node(id).unwrap();
    assert!(node.has_label("Person"));
    assert!(!node.has_label("Employee"));

    // Removing non-existent label should fail
    assert!(!store.remove_label(id, "Employee"));
    assert!(!store.remove_label(id, "NonExistent"));
}

#[test]
fn test_add_label_to_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.add_label(fake_id, "Label"));
}

#[test]
fn test_remove_label_from_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.remove_label(fake_id, "Label"));
}

#[test]
fn test_node_ids() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    let ids = store.node_ids();
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(&n1));
    assert!(ids.contains(&n2));
    assert!(ids.contains(&n3));

    // Delete one
    store.delete_node(n2);
    let ids = store.node_ids();
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&n2));
}

#[test]
fn test_delete_node_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(!store.delete_node(fake_id));
}

#[test]
fn test_delete_edge_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = EdgeId::new(999);
    assert!(!store.delete_edge(fake_id));
}

#[test]
fn test_delete_edge_double() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    assert!(store.delete_edge(edge_id));
    assert!(!store.delete_edge(edge_id)); // Double delete
}

#[test]
fn test_create_edge_with_props() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);

    let edge_id = store.create_edge_with_props(
        a,
        b,
        "KNOWS",
        [
            ("since", Value::from(2020i64)),
            ("weight", Value::from(1.0)),
        ],
    );

    let edge = store.get_edge(edge_id).unwrap();
    assert_eq!(
        edge.get_property("since").and_then(|v| v.as_int64()),
        Some(2020)
    );
    assert_eq!(
        edge.get_property("weight").and_then(|v| v.as_float64()),
        Some(1.0)
    );
}

#[test]
fn test_delete_node_edges() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS"); // a -> b
    store.create_edge(c, a, "KNOWS"); // c -> a

    assert_eq!(store.edge_count(), 2);

    // Delete all edges connected to a
    store.delete_node_edges(a);

    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_delete_node_edges_self_loop() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let _e = store.create_edge(a, a, "SELF"); // self-loop

    assert_eq!(store.edge_count(), 1);

    // Self-loop appears in both outgoing and incoming scans.
    // The fix deduplicates via HashSet, so only one delete happens.
    store.delete_node_edges(a);

    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_delete_node_edges_self_loop_plus_others() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, a, "SELF"); // self-loop on a
    store.create_edge(a, b, "KNOWS"); // outgoing from a
    store.create_edge(c, a, "KNOWS"); // incoming to a
    store.create_edge(b, c, "KNOWS"); // unrelated

    assert_eq!(store.edge_count(), 4);

    store.delete_node_edges(a);

    // Only the b->c edge should remain
    assert_eq!(store.edge_count(), 1);
}

#[test]
fn test_delete_node_edges_atomic_batch() {
    use std::sync::Arc;

    let store = Arc::new(LpgStore::new().unwrap());

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);
    let d = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(d, a, "KNOWS");

    assert_eq!(store.edge_count(), 3);

    // Spawn a reader thread that checks edge count.
    // With batch locking, the reader should never see 1 or 2
    // (partially deleted): it should see either 3 (before) or 0 (after).
    //
    // A barrier ensures both threads start at the same time, and an
    // AtomicBool keeps the reader spinning until deletion finishes,
    // so the two threads are guaranteed to overlap.
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};

    let barrier = Arc::new(Barrier::new(2));
    let done = Arc::new(AtomicBool::new(false));

    let reader = Arc::clone(&store);
    let reader_barrier = Arc::clone(&barrier);
    let reader_done = Arc::clone(&done);

    let handle = std::thread::spawn(move || {
        let mut saw_partial = false;
        reader_barrier.wait();
        while !reader_done.load(Ordering::Acquire) {
            let count = reader.edge_count();
            if count != 0 && count != 3 {
                saw_partial = true;
                break;
            }
        }
        saw_partial
    });

    barrier.wait();
    store.delete_node_edges(a);
    done.store(true, Ordering::Release);

    let saw_partial = handle.join().unwrap();
    assert!(
        !saw_partial,
        "concurrent reader observed partially deleted edges"
    );
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_neighbors_both_directions() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS"); // a -> b
    store.create_edge(c, a, "KNOWS"); // c -> a

    // Direction::Both for node a
    let neighbors: Vec<_> = store.neighbors(a, Direction::Both).collect();
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&b)); // outgoing
    assert!(neighbors.contains(&c)); // incoming
}

#[test]
fn test_edges_from() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    let e1 = store.create_edge(a, b, "KNOWS");
    let e2 = store.create_edge(a, c, "KNOWS");

    let edges: Vec<_> = store.edges_from(a, Direction::Outgoing).collect();
    assert_eq!(edges.len(), 2);
    assert!(edges.iter().any(|(_, e)| *e == e1));
    assert!(edges.iter().any(|(_, e)| *e == e2));

    // Incoming edges to b
    let incoming: Vec<_> = store.edges_from(b, Direction::Incoming).collect();
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0].1, e1);
}

#[test]
fn test_edges_to() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    let e1 = store.create_edge(a, b, "KNOWS");
    let e2 = store.create_edge(c, b, "KNOWS");

    // Edges pointing TO b
    let to_b = store.edges_to(b);
    assert_eq!(to_b.len(), 2);
    assert!(to_b.iter().any(|(src, e)| *src == a && *e == e1));
    assert!(to_b.iter().any(|(src, e)| *src == c && *e == e2));
}

#[test]
fn test_out_degree_in_degree() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(c, b, "KNOWS");

    assert_eq!(store.out_degree(a), 2);
    assert_eq!(store.out_degree(b), 0);
    assert_eq!(store.out_degree(c), 1);

    assert_eq!(store.in_degree(a), 0);
    assert_eq!(store.in_degree(b), 2);
    assert_eq!(store.in_degree(c), 1);
}

#[test]
fn test_edge_type() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let edge_id = store.create_edge(a, b, "KNOWS");

    let edge_type = store.edge_type(edge_id);
    assert_eq!(edge_type.as_deref(), Some("KNOWS"));

    // Non-existent edge
    let fake_id = EdgeId::new(999);
    assert!(store.edge_type(fake_id).is_none());
}

#[test]
fn test_count_methods() {
    let store = LpgStore::new().unwrap();

    assert_eq!(store.label_count(), 0);
    assert_eq!(store.edge_type_count(), 0);
    assert_eq!(store.property_key_count(), 0);

    let a = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let b = store.create_node(&["Company"]);
    store.create_edge_with_props(a, b, "WORKS_AT", [("since", Value::from(2020i64))]);

    assert_eq!(store.label_count(), 2); // Person, Company
    assert_eq!(store.edge_type_count(), 1); // WORKS_AT
    assert_eq!(store.property_key_count(), 2); // age, since
}

#[test]
fn test_all_nodes_and_edges() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    store.create_edge(a, b, "KNOWS");

    let nodes: Vec<_> = store.all_nodes().collect();
    assert_eq!(nodes.len(), 2);

    let edges: Vec<_> = store.all_edges().collect();
    assert_eq!(edges.len(), 1);
}

#[test]
fn test_all_labels_and_edge_types() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Company"]);
    let a = store.create_node(&["Animal"]);
    let b = store.create_node(&["Animal"]);
    store.create_edge(a, b, "EATS");

    let labels = store.all_labels();
    assert_eq!(labels.len(), 3);
    assert!(labels.contains(&"Person".to_string()));
    assert!(labels.contains(&"Company".to_string()));
    assert!(labels.contains(&"Animal".to_string()));

    let edge_types = store.all_edge_types();
    assert_eq!(edge_types.len(), 1);
    assert!(edge_types.contains(&"EATS".to_string()));
}

#[test]
fn test_all_property_keys() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node_with_props(&["Person"], [("name", Value::from("Alix"))]);
    let b = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    store.create_edge_with_props(a, b, "KNOWS", [("since", Value::from(2020i64))]);

    let keys = store.all_property_keys();
    assert!(keys.contains(&"name".to_string()));
    assert!(keys.contains(&"age".to_string()));
    assert!(keys.contains(&"since".to_string()));
}

#[test]
fn test_nodes_with_label() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Company"]);

    let persons: Vec<_> = store.nodes_with_label("Person").collect();
    assert_eq!(persons.len(), 2);

    let companies: Vec<_> = store.nodes_with_label("Company").collect();
    assert_eq!(companies.len(), 1);

    let none: Vec<_> = store.nodes_with_label("NonExistent").collect();
    assert_eq!(none.len(), 0);
}

#[test]
fn test_edges_with_type() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Company"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "WORKS_AT");

    let knows: Vec<_> = store.edges_with_type("KNOWS").collect();
    assert_eq!(knows.len(), 1);

    let works_at: Vec<_> = store.edges_with_type("WORKS_AT").collect();
    assert_eq!(works_at.len(), 1);

    let none: Vec<_> = store.edges_with_type("NonExistent").collect();
    assert_eq!(none.len(), 0);
}

#[test]
fn test_nodes_by_label_nonexistent() {
    let store = LpgStore::new().unwrap();
    store.create_node(&["Person"]);

    let empty = store.nodes_by_label("NonExistent");
    assert!(empty.is_empty());
}

#[test]
fn test_nodes_by_label_count_matches_vec_len() {
    // nodes_by_label_count is the O(1) fast path used by the planner to
    // bound unbounded VectorScan k; it must agree with the Vec-returning
    // variant for every label, including ones that don't exist.
    let store = LpgStore::new().unwrap();
    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Animal"]);

    for label in ["Person", "Animal", "NonExistent"] {
        assert_eq!(
            store.nodes_by_label_count(label),
            store.nodes_by_label(label).len(),
            "count mismatch for label {label:?}"
        );
    }
}

#[test]
fn test_statistics() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Company"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "WORKS_AT");

    store.compute_statistics();
    let stats = store.statistics();

    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.total_edges, 2);

    // Estimates
    let person_card = store.estimate_label_cardinality("Person");
    assert!(person_card > 0.0);

    let avg_degree = store.estimate_avg_degree("KNOWS", true);
    assert!(avg_degree >= 0.0);
}

#[test]
fn test_zone_maps() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(25i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(35i64))]);

    // Zone map should indicate possible matches (30 is within [25, 35] range)
    let might_match =
        store.node_property_might_match(&"age".into(), CompareOp::Eq, &Value::from(30i64));
    // Zone maps return true conservatively when value is within min/max range
    assert!(might_match);

    let zone = store.node_property_zone_map(&"age".into());
    assert!(zone.is_some());

    // Non-existent property
    let no_zone = store.node_property_zone_map(&"nonexistent".into());
    assert!(no_zone.is_none());

    // Edge zone maps
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    store.create_edge_with_props(a, b, "REL", [("weight", Value::from(1.0))]);

    let edge_zone = store.edge_property_zone_map(&"weight".into());
    assert!(edge_zone.is_some());
}

#[test]
fn test_rebuild_zone_maps() {
    let store = LpgStore::new().unwrap();
    store.create_node_with_props(&["Person"], [("age", Value::from(25i64))]);

    // Should not panic
    store.rebuild_zone_maps();
}

#[test]
fn test_create_node_with_id() {
    let store = LpgStore::new().unwrap();

    let specific_id = NodeId::new(100);
    store
        .create_node_with_id(specific_id, &["Person", "Employee"])
        .unwrap();

    let node = store.get_node(specific_id).unwrap();
    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));

    // Next auto-generated ID should be > 100
    let next = store.create_node(&["Other"]);
    assert!(next.as_u64() > 100);
}

#[test]
fn test_create_edge_with_id() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);

    let specific_id = EdgeId::new(500);
    store.create_edge_with_id(specific_id, a, b, "REL").unwrap();

    let edge = store.get_edge(specific_id).unwrap();
    assert_eq!(edge.src, a);
    assert_eq!(edge.dst, b);
    assert_eq!(edge.edge_type.as_str(), "REL");

    // Next auto-generated ID should be > 500
    let next = store.create_edge(a, b, "OTHER");
    assert!(next.as_u64() > 500);
}

#[test]
fn test_set_epoch() {
    let store = LpgStore::new().unwrap();

    assert_eq!(store.current_epoch().as_u64(), 0);

    store.set_epoch(EpochId::new(42));
    assert_eq!(store.current_epoch().as_u64(), 42);
}

#[test]
fn test_get_node_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = NodeId::new(999);
    assert!(store.get_node(fake_id).is_none());
}

#[test]
fn test_get_edge_nonexistent() {
    let store = LpgStore::new().unwrap();
    let fake_id = EdgeId::new(999);
    assert!(store.get_edge(fake_id).is_none());
}

#[test]
fn test_multiple_labels() {
    let store = LpgStore::new().unwrap();

    let id = store.create_node(&["Person", "Employee", "Manager"]);
    let node = store.get_node(id).unwrap();

    assert!(node.has_label("Person"));
    assert!(node.has_label("Employee"));
    assert!(node.has_label("Manager"));
    assert!(!node.has_label("Other"));
}

#[test]
fn test_new_store_is_empty() {
    let store = LpgStore::new().unwrap();
    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_edges_from_both_directions() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let c = store.create_node(&["C"]);

    let e1 = store.create_edge(a, b, "R1"); // a -> b
    let e2 = store.create_edge(c, a, "R2"); // c -> a

    // Both directions from a
    let edges: Vec<_> = store.edges_from(a, Direction::Both).collect();
    assert_eq!(edges.len(), 2);
    assert!(edges.iter().any(|(_, e)| *e == e1)); // outgoing
    assert!(edges.iter().any(|(_, e)| *e == e2)); // incoming
}

#[test]
fn test_no_backward_adj_in_degree() {
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 10,
        initial_edge_capacity: 10,
    };
    let store = LpgStore::with_config(config).unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    store.create_edge(a, b, "R");

    // in_degree should still work (falls back to scanning)
    let degree = store.in_degree(b);
    assert_eq!(degree, 1);
}

#[test]
fn test_no_backward_adj_edges_to() {
    let config = LpgStoreConfig {
        backward_edges: false,
        initial_node_capacity: 10,
        initial_edge_capacity: 10,
    };
    let store = LpgStore::with_config(config).unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let e = store.create_edge(a, b, "R");

    // edges_to should still work (falls back to scanning)
    let edges = store.edges_to(b);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].1, e);
}

#[test]
fn test_node_versioned_creation() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let id = store.create_node_versioned(&["Person"], epoch, transaction_id);
    assert!(store.get_node(id).is_some());
}

#[test]
fn test_edge_versioned_creation() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let edge_id = store.create_edge_versioned(a, b, "REL", epoch, transaction_id);
    assert!(store.get_edge(edge_id).is_some());
}

#[test]
fn test_node_with_props_versioned() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(1);

    let id = store.create_node_with_props_versioned(
        &["Person"],
        [("name", Value::from("Alix"))],
        epoch,
        transaction_id,
    );

    let node = store.get_node(id).unwrap();
    assert_eq!(
        node.get_property("name").and_then(|v| v.as_str()),
        Some("Alix")
    );
}

#[test]
fn test_discard_uncommitted_versions() {
    let store = LpgStore::new().unwrap();

    let epoch = store.new_epoch();
    let transaction_id = TransactionId::new(42);

    // Create node with specific tx (uses PENDING epoch, invisible to get_node)
    let node_id = store.create_node_versioned(&["Person"], epoch, transaction_id);
    // Verify the node exists via versioned lookup (own tx can see its PENDING writes)
    assert!(
        store
            .get_node_versioned(node_id, epoch, transaction_id)
            .is_some(),
        "Node should be visible to its own transaction"
    );

    // Discard uncommitted versions for this tx
    store.discard_uncommitted_versions(transaction_id);

    // Node should be gone (version chain was removed)
    assert!(
        store
            .get_node_versioned(node_id, epoch, transaction_id)
            .is_none(),
        "Node should be gone after discard"
    );
}

// === Property Index Tests ===

#[test]
fn test_property_index_create_and_lookup() {
    let store = LpgStore::new().unwrap();

    // Create nodes with properties
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person"]);

    store.set_node_property(alix, "city", Value::from("NYC"));
    store.set_node_property(gus, "city", Value::from("NYC"));
    store.set_node_property(vincent, "city", Value::from("LA"));

    // Before indexing, lookup still works (via scan)
    let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
    assert_eq!(nyc_people.len(), 2);

    // Create index
    store.create_property_index("city");
    assert!(store.has_property_index("city"));

    // Indexed lookup should return same results
    let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
    assert_eq!(nyc_people.len(), 2);
    assert!(nyc_people.contains(&alix));
    assert!(nyc_people.contains(&gus));

    let la_people = store.find_nodes_by_property("city", &Value::from("LA"));
    assert_eq!(la_people.len(), 1);
    assert!(la_people.contains(&vincent));
}

#[test]
fn test_property_index_maintained_on_update() {
    let store = LpgStore::new().unwrap();

    // Create index first
    store.create_property_index("status");

    let node = store.create_node(&["Task"]);
    store.set_node_property(node, "status", Value::from("pending"));

    // Should find by initial value
    let pending = store.find_nodes_by_property("status", &Value::from("pending"));
    assert_eq!(pending.len(), 1);
    assert!(pending.contains(&node));

    // Update the property
    store.set_node_property(node, "status", Value::from("done"));

    // Old value should not find it
    let pending = store.find_nodes_by_property("status", &Value::from("pending"));
    assert!(pending.is_empty());

    // New value should find it
    let done = store.find_nodes_by_property("status", &Value::from("done"));
    assert_eq!(done.len(), 1);
    assert!(done.contains(&node));
}

#[test]
fn test_property_index_maintained_on_remove() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("tag");

    let node = store.create_node(&["Item"]);
    store.set_node_property(node, "tag", Value::from("important"));

    // Should find it
    let found = store.find_nodes_by_property("tag", &Value::from("important"));
    assert_eq!(found.len(), 1);

    // Remove the property
    store.remove_node_property(node, "tag");

    // Should no longer find it
    let found = store.find_nodes_by_property("tag", &Value::from("important"));
    assert!(found.is_empty());
}

#[test]
fn test_property_index_drops_deleted_nodes() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("id");
    let first = store.create_node_with_props(&["Graph"], [("id", Value::from("a"))]);
    assert_eq!(
        store.find_nodes_by_property("id", &Value::from("a")),
        vec![first]
    );

    assert!(store.delete_node(first));
    assert!(
        store
            .find_nodes_by_property("id", &Value::from("a"))
            .is_empty(),
        "a deleted node must leave the index"
    );

    // Delete and re-create the same key many times: the index holds only the
    // live node instead of growing with every cycle.
    let mut live = store.create_node_with_props(&["Graph"], [("id", Value::from("a"))]);
    for _ in 0..100 {
        assert!(store.delete_node(live));
        live = store.create_node_with_props(&["Graph"], [("id", Value::from("a"))]);
    }
    assert_eq!(
        store.find_nodes_by_property("id", &Value::from("a")),
        vec![live]
    );
}

#[test]
fn test_property_index_restored_when_delete_rolls_back() {
    use crate::graph::GraphStoreMut;

    let store = LpgStore::new().unwrap();
    store.create_property_index("id");
    let node = store.create_node_with_props(&["Graph"], [("id", Value::from("a"))]);

    let tx = TransactionId::new(9);
    assert!(GraphStoreMut::delete_node_versioned(
        &store,
        node,
        store.current_epoch(),
        tx
    ));
    assert!(
        store
            .find_nodes_by_property("id", &Value::from("a"))
            .is_empty()
    );

    store.rollback_transaction_properties(tx);
    assert_eq!(
        store.find_nodes_by_property("id", &Value::from("a")),
        vec![node]
    );
}

#[test]
fn test_property_index_restored_when_set_rolls_back() {
    use crate::graph::GraphStoreMut;

    let store = LpgStore::new().unwrap();
    store.create_property_index("id");
    let node = store.create_node_with_props(&["Graph"], [("id", Value::from("a"))]);
    store.new_epoch();

    let tx = TransactionId::new(9);
    GraphStoreMut::set_node_property_versioned(&store, node, "id", Value::from("b"), tx);
    assert_eq!(
        store.find_nodes_by_property("id", &Value::from("b")),
        vec![node]
    );

    store.rollback_transaction_properties(tx);
    assert_eq!(
        store.find_nodes_by_property("id", &Value::from("a")),
        vec![node]
    );
    assert!(
        store
            .find_nodes_by_property("id", &Value::from("b"))
            .is_empty(),
        "the rolled-back value must leave the index"
    );
}

#[test]
fn test_property_index_drop() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("key");
    assert!(store.has_property_index("key"));

    assert!(store.drop_property_index("key"));
    assert!(!store.has_property_index("key"));

    // Dropping non-existent index returns false
    assert!(!store.drop_property_index("key"));
}

#[test]
fn test_property_index_multiple_values() {
    let store = LpgStore::new().unwrap();

    store.create_property_index("age");

    // Create multiple nodes with same and different ages
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);
    let n4 = store.create_node(&["Person"]);

    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "age", Value::from(25i64));
    store.set_node_property(n3, "age", Value::from(30i64));
    store.set_node_property(n4, "age", Value::from(25i64));

    let age_25 = store.find_nodes_by_property("age", &Value::from(25i64));
    assert_eq!(age_25.len(), 3);

    let age_30 = store.find_nodes_by_property("age", &Value::from(30i64));
    assert_eq!(age_30.len(), 1);

    let age_40 = store.find_nodes_by_property("age", &Value::from(40i64));
    assert!(age_40.is_empty());
}

#[test]
fn test_property_index_builds_from_existing_data() {
    let store = LpgStore::new().unwrap();

    // Create nodes first
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    store.set_node_property(n1, "email", Value::from("alix@example.com"));
    store.set_node_property(n2, "email", Value::from("gus@example.com"));

    // Create index after data exists
    store.create_property_index("email");

    // Index should include existing data
    let alix = store.find_nodes_by_property("email", &Value::from("alix@example.com"));
    assert_eq!(alix.len(), 1);
    assert!(alix.contains(&n1));

    let gus = store.find_nodes_by_property("email", &Value::from("gus@example.com"));
    assert_eq!(gus.len(), 1);
    assert!(gus.contains(&n2));
}

#[test]
fn test_get_node_property_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "age", Value::from(30i64));
    // n3 has no age property

    let age_key = PropertyKey::new("age");
    let values = store.get_node_property_batch(&[n1, n2, n3], &age_key);

    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Some(Value::from(25i64)));
    assert_eq!(values[1], Some(Value::from(30i64)));
    assert_eq!(values[2], None);
}

#[test]
fn test_get_node_property_batch_empty() {
    let store = LpgStore::new().unwrap();
    let key = PropertyKey::new("any");

    let values = store.get_node_property_batch(&[], &key);
    assert!(values.is_empty());
}

#[test]
fn test_get_nodes_properties_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    let n3 = store.create_node(&["Person"]);

    store.set_node_property(n1, "name", Value::from("Alix"));
    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n2, "name", Value::from("Gus"));
    // n3 has no properties

    let all_props = store.get_nodes_properties_batch(&[n1, n2, n3]);

    assert_eq!(all_props.len(), 3);
    assert_eq!(all_props[0].len(), 2); // name and age
    assert_eq!(all_props[1].len(), 1); // name only
    assert_eq!(all_props[2].len(), 0); // no properties

    assert_eq!(
        all_props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
    assert_eq!(
        all_props[1].get(&PropertyKey::new("name")),
        Some(&Value::from("Gus"))
    );
}

#[test]
fn test_get_nodes_properties_batch_empty() {
    let store = LpgStore::new().unwrap();

    let all_props = store.get_nodes_properties_batch(&[]);
    assert!(all_props.is_empty());
}

#[test]
fn test_get_nodes_properties_selective_batch() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);

    // Set multiple properties
    store.set_node_property(n1, "name", Value::from("Alix"));
    store.set_node_property(n1, "age", Value::from(25i64));
    store.set_node_property(n1, "email", Value::from("alix@example.com"));
    store.set_node_property(n2, "name", Value::from("Gus"));
    store.set_node_property(n2, "age", Value::from(30i64));
    store.set_node_property(n2, "city", Value::from("NYC"));

    // Request only name and age (not email or city)
    let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
    let props = store.get_nodes_properties_selective_batch(&[n1, n2], &keys);

    assert_eq!(props.len(), 2);

    // n1: should have name and age, but NOT email
    assert_eq!(props[0].len(), 2);
    assert_eq!(
        props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
    assert_eq!(
        props[0].get(&PropertyKey::new("age")),
        Some(&Value::from(25i64))
    );
    assert_eq!(props[0].get(&PropertyKey::new("email")), None);

    // n2: should have name and age, but NOT city
    assert_eq!(props[1].len(), 2);
    assert_eq!(
        props[1].get(&PropertyKey::new("name")),
        Some(&Value::from("Gus"))
    );
    assert_eq!(
        props[1].get(&PropertyKey::new("age")),
        Some(&Value::from(30i64))
    );
    assert_eq!(props[1].get(&PropertyKey::new("city")), None);
}

#[test]
fn test_get_nodes_properties_selective_batch_empty_keys() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", Value::from("Alix"));

    // Request no properties
    let props = store.get_nodes_properties_selective_batch(&[n1], &[]);

    assert_eq!(props.len(), 1);
    assert!(props[0].is_empty()); // Empty map when no keys requested
}

#[test]
fn test_get_nodes_properties_selective_batch_missing_keys() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", Value::from("Alix"));

    // Request a property that doesn't exist
    let keys = vec![PropertyKey::new("nonexistent"), PropertyKey::new("name")];
    let props = store.get_nodes_properties_selective_batch(&[n1], &keys);

    assert_eq!(props.len(), 1);
    assert_eq!(props[0].len(), 1); // Only name exists
    assert_eq!(
        props[0].get(&PropertyKey::new("name")),
        Some(&Value::from("Alix"))
    );
}

// === Range Query Tests ===

#[test]
fn test_find_nodes_in_range_inclusive() {
    let store = LpgStore::new().unwrap();

    let n1 = store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    let n2 = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let n3 = store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);
    let _n4 = store.create_node_with_props(&["Person"], [("age", Value::from(50i64))]);

    // age >= 20 AND age <= 40 (inclusive both sides)
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(20i64)),
        Some(&Value::from(40i64)),
        true,
        true,
    );
    assert_eq!(result.len(), 3);
    assert!(result.contains(&n1));
    assert!(result.contains(&n2));
    assert!(result.contains(&n3));
}

#[test]
fn test_find_nodes_in_range_exclusive() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    let n2 = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);

    // age > 20 AND age < 40 (exclusive both sides)
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(20i64)),
        Some(&Value::from(40i64)),
        false,
        false,
    );
    assert_eq!(result.len(), 1);
    assert!(result.contains(&n2));
}

#[test]
fn test_find_nodes_in_range_open_ended() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
    store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
    let n3 = store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);
    let n4 = store.create_node_with_props(&["Person"], [("age", Value::from(50i64))]);

    // age >= 35 (no upper bound)
    let result = store.find_nodes_in_range("age", Some(&Value::from(35i64)), None, true, true);
    assert_eq!(result.len(), 2);
    assert!(result.contains(&n3));
    assert!(result.contains(&n4));

    // age <= 25 (no lower bound)
    let result = store.find_nodes_in_range("age", None, Some(&Value::from(25i64)), true, true);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_find_nodes_in_range_empty_result() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);

    // Range that doesn't match anything
    let result = store.find_nodes_in_range(
        "age",
        Some(&Value::from(100i64)),
        Some(&Value::from(200i64)),
        true,
        true,
    );
    assert!(result.is_empty());
}

#[test]
fn test_find_nodes_in_range_nonexistent_property() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);

    let result = store.find_nodes_in_range(
        "weight",
        Some(&Value::from(50i64)),
        Some(&Value::from(100i64)),
        true,
        true,
    );
    assert!(result.is_empty());
}

// === Multi-Property Query Tests ===

#[test]
fn test_find_nodes_by_properties_multiple_conditions() {
    let store = LpgStore::new().unwrap();

    let alix = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("city", Value::from("NYC"))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Gus")), ("city", Value::from("NYC"))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("city", Value::from("LA"))],
    );

    // Match name="Alix" AND city="NYC"
    let result = store
        .find_nodes_by_properties(&[("name", Value::from("Alix")), ("city", Value::from("NYC"))]);
    assert_eq!(result.len(), 1);
    assert!(result.contains(&alix));
}

#[test]
fn test_find_nodes_by_properties_empty_conditions() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);

    // Empty conditions should return all nodes
    let result = store.find_nodes_by_properties(&[]);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_find_nodes_by_properties_no_match() {
    let store = LpgStore::new().unwrap();

    store.create_node_with_props(&["Person"], [("name", Value::from("Alix"))]);

    let result = store.find_nodes_by_properties(&[("name", Value::from("Nobody"))]);
    assert!(result.is_empty());
}

#[test]
fn test_find_nodes_by_properties_with_index() {
    let store = LpgStore::new().unwrap();

    // Create index on name
    store.create_property_index("name");

    let alix = store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Alix")), ("age", Value::from(30i64))],
    );
    store.create_node_with_props(
        &["Person"],
        [("name", Value::from("Gus")), ("age", Value::from(30i64))],
    );

    // Index should accelerate the lookup
    let result = store
        .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::from(30i64))]);
    assert_eq!(result.len(), 1);
    assert!(result.contains(&alix));
}

// === Cardinality Estimation Tests ===

#[test]
fn test_estimate_label_cardinality() {
    let store = LpgStore::new().unwrap();

    store.create_node(&["Person"]);
    store.create_node(&["Person"]);
    store.create_node(&["Animal"]);

    store.ensure_statistics_fresh();

    let person_est = store.estimate_label_cardinality("Person");
    let animal_est = store.estimate_label_cardinality("Animal");
    let unknown_est = store.estimate_label_cardinality("Unknown");

    assert!(
        person_est >= 1.0,
        "Person should have cardinality >= 1, got {person_est}"
    );
    assert!(
        animal_est >= 1.0,
        "Animal should have cardinality >= 1, got {animal_est}"
    );
    // Unknown label should return some default (not panic)
    assert!(unknown_est >= 0.0);
}

#[test]
fn test_estimate_avg_degree() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, b, "KNOWS");
    store.create_edge(a, c, "KNOWS");
    store.create_edge(b, c, "KNOWS");

    store.ensure_statistics_fresh();

    let outgoing = store.estimate_avg_degree("KNOWS", true);
    let incoming = store.estimate_avg_degree("KNOWS", false);

    assert!(
        outgoing > 0.0,
        "Outgoing degree should be > 0, got {outgoing}"
    );
    assert!(
        incoming > 0.0,
        "Incoming degree should be > 0, got {incoming}"
    );
}

// === Delete operations ===

#[test]
fn test_delete_node_does_not_cascade() {
    let store = LpgStore::new().unwrap();

    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let e = store.create_edge(a, b, "KNOWS");

    assert!(store.delete_node(a));
    assert!(store.get_node(a).is_none());

    // Edges are NOT automatically deleted (non-detach delete)
    assert!(
        store.get_edge(e).is_some(),
        "Edge should survive non-detach node delete"
    );
}

#[test]
fn test_delete_already_deleted_node() {
    let store = LpgStore::new().unwrap();
    let a = store.create_node(&["A"]);

    assert!(store.delete_node(a));
    // Second delete should return false (already deleted)
    assert!(!store.delete_node(a));
}

#[test]
fn test_delete_nonexistent_node() {
    let store = LpgStore::new().unwrap();
    assert!(!store.delete_node(NodeId::new(999)));
}

// === GraphStore / GraphStoreMut Trait Compliance ===

/// Verifies that LpgStore's trait implementations are object-safe and
/// produce identical results to the concrete methods.
mod graph_store_traits {
    use super::*;
    use crate::graph::Direction;
    use crate::graph::traits::{GraphStore, GraphStoreMut};

    #[test]
    fn trait_object_safety() {
        // Must compile: Arc<dyn GraphStoreMut> proves object safety
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let _read: &dyn GraphStore = &*store;
    }

    #[test]
    fn trait_round_trip() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        // Create nodes via trait
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person", "Developer"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::from(30i64));
        store.set_node_property(gus, "name", Value::from("Gus"));

        // Create edge via trait
        let edge = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(edge, "since", Value::from(2020i64));

        // Read back via GraphStore trait
        let read: &dyn GraphStore = store;

        // Point lookups
        let alice_node = read.get_node(alix).expect("alix should exist");
        assert!(alice_node.labels.contains(&arcstr::literal!("Person")));

        let edge_data = read.get_edge(edge).expect("edge should exist");
        assert_eq!(edge_data.src, alix);
        assert_eq!(edge_data.dst, gus);

        // Properties
        assert_eq!(
            read.get_node_property(alix, &PropertyKey::new("name")),
            Some(Value::from("Alix"))
        );
        assert_eq!(
            read.get_edge_property(edge, &PropertyKey::new("since")),
            Some(Value::from(2020i64))
        );

        // Traversal
        let neighbors = read.neighbors(alix, Direction::Outgoing);
        assert_eq!(neighbors, vec![gus]);

        let edges = read.edges_from(alix, Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (gus, edge));

        assert_eq!(read.out_degree(alix), 1);
        assert_eq!(read.in_degree(gus), 1);

        // Scans
        assert_eq!(read.node_count(), 2);
        assert_eq!(read.edge_count(), 1);
        assert_eq!(read.nodes_by_label("Person").len(), 2);
        assert_eq!(read.node_ids().len(), 2);

        // Edge type
        assert_eq!(read.edge_type(edge), Some(arcstr::literal!("KNOWS")));

        // Search
        let found = read.find_nodes_by_property("name", &Value::from("Alix"));
        assert_eq!(found, vec![alix]);
    }

    #[test]
    fn trait_mutation_operations() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        let node = store.create_node(&["A"]);

        // Label mutation
        assert!(store.add_label(node, "B"));
        assert!(store.remove_label(node, "B"));

        // Property mutation
        store.set_node_property(node, "key", Value::from("val"));
        let removed = store.remove_node_property(node, "key");
        assert_eq!(removed, Some(Value::from("val")));

        // Deletion
        assert!(store.delete_node(node));
        assert!(store.get_node(node).is_none());
    }

    #[test]
    fn trait_batch_edges() {
        let store = LpgStore::new().unwrap();
        let store: &dyn GraphStoreMut = &store;

        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);

        let ids = store.batch_create_edges(&[(a, b, "E"), (a, c, "E"), (b, c, "E")]);
        assert_eq!(ids.len(), 3);
        assert_eq!(store.edge_count(), 3);
    }
}

#[test]
fn test_clear() {
    let store = LpgStore::new().unwrap();
    let n1 = store.create_node(&["Person"]);
    let n2 = store.create_node(&["Person"]);
    store.set_node_property(n1, "name", "Alix".into());
    let _e = store.create_edge(n1, n2, "KNOWS");
    store.set_edge_property(_e, "since", 2024.into());

    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 1);

    store.clear();

    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);

    // Should be able to add new data after clear
    let n3 = store.create_node(&["Animal"]);
    assert_eq!(store.node_count(), 1);
    assert!(store.get_node(n3).is_some());
}
