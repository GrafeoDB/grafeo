//! Tests for grafeo-memory engine enhancements:
//! - batch_create_nodes_with_props
//! - filter optimization in compute_filter_allowlist
//! - temporal property versioning API
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test grafeo_memory_support
//! ```

#[cfg(feature = "temporal")]
use grafeo_common::types::NodeId;
use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

// =============================================================================
// batch_create_nodes_with_props
// =============================================================================

mod batch_create_with_props {
    use super::*;

    #[test]
    fn creates_nodes_with_mixed_properties() {
        let db = db();
        let mut props_list = Vec::new();

        let mut p1 = HashMap::new();
        p1.insert(PropertyKey::new("text"), Value::String("hello".into()));
        p1.insert(PropertyKey::new("user_id"), Value::String("u1".into()));
        p1.insert(PropertyKey::new("score"), Value::Float64(0.95));
        props_list.push(p1);

        let mut p2 = HashMap::new();
        p2.insert(PropertyKey::new("text"), Value::String("world".into()));
        p2.insert(PropertyKey::new("user_id"), Value::String("u1".into()));
        p2.insert(PropertyKey::new("score"), Value::Float64(0.80));
        props_list.push(p2);

        let ids = db
            .batch_create_nodes_with_props("Memory", props_list)
            .unwrap();
        assert_eq!(ids.len(), 2);

        // Verify properties
        let node = db.get_node(ids[0]).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("text")),
            Some(&Value::String("hello".into()))
        );
        assert_eq!(
            node.properties.get(&PropertyKey::new("user_id")),
            Some(&Value::String("u1".into()))
        );
    }

    #[test]
    fn creates_nodes_with_vector_properties() {
        let db = db();
        let mut props_list = Vec::new();

        let mut p1 = HashMap::new();
        p1.insert(PropertyKey::new("text"), Value::String("doc1".into()));
        p1.insert(
            PropertyKey::new("embedding"),
            Value::Vector(vec![1.0, 0.0, 0.0].into()),
        );
        props_list.push(p1);

        let mut p2 = HashMap::new();
        p2.insert(PropertyKey::new("text"), Value::String("doc2".into()));
        p2.insert(
            PropertyKey::new("embedding"),
            Value::Vector(vec![0.0, 1.0, 0.0].into()),
        );
        props_list.push(p2);

        let ids = db
            .batch_create_nodes_with_props("Document", props_list)
            .unwrap();
        assert_eq!(ids.len(), 2);

        // Verify vector property
        let node = db.get_node(ids[0]).unwrap();
        match node.properties.get(&PropertyKey::new("embedding")) {
            Some(Value::Vector(v)) => assert_eq!(v.len(), 3),
            other => panic!("Expected Vector, got {:?}", other),
        }
    }

    #[test]
    fn empty_list_returns_empty_ids() {
        let db = db();
        let ids = db
            .batch_create_nodes_with_props("Memory", Vec::new())
            .unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn nodes_with_different_property_sets() {
        let db = db();
        let mut props_list = Vec::new();

        // Node with 2 properties
        let mut p1 = HashMap::new();
        p1.insert(PropertyKey::new("text"), Value::String("short".into()));
        p1.insert(PropertyKey::new("type"), Value::String("note".into()));
        props_list.push(p1);

        // Node with 5 properties
        let mut p2 = HashMap::new();
        p2.insert(PropertyKey::new("text"), Value::String("detailed".into()));
        p2.insert(PropertyKey::new("type"), Value::String("memo".into()));
        p2.insert(PropertyKey::new("user_id"), Value::String("u1".into()));
        p2.insert(PropertyKey::new("priority"), Value::Int64(1));
        p2.insert(PropertyKey::new("archived"), Value::Bool(false));
        props_list.push(p2);

        let ids = db
            .batch_create_nodes_with_props("Item", props_list)
            .unwrap();
        assert_eq!(ids.len(), 2);

        let n1 = db.get_node(ids[0]).unwrap();
        let n2 = db.get_node(ids[1]).unwrap();
        assert_eq!(n1.properties.len(), 2);
        assert_eq!(n2.properties.len(), 5);
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn auto_inserts_into_vector_index() {
        let db = db();

        // Create a vector index first
        db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
            .unwrap();

        let mut props_list = Vec::new();
        let mut p1 = HashMap::new();
        p1.insert(
            PropertyKey::new("emb"),
            Value::Vector(vec![1.0, 0.0, 0.0].into()),
        );
        props_list.push(p1);

        let mut p2 = HashMap::new();
        p2.insert(
            PropertyKey::new("emb"),
            Value::Vector(vec![0.0, 1.0, 0.0].into()),
        );
        props_list.push(p2);

        let ids = db.batch_create_nodes_with_props("Doc", props_list).unwrap();
        assert_eq!(ids.len(), 2);

        // Search should find both
        let results = db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 10, None, None)
            .unwrap();
        assert_eq!(results.len(), 2);
    }
}

// =============================================================================
// Filter optimization (operator filters scan narrowed allowlist)
// =============================================================================

#[cfg(feature = "vector-index")]
mod filter_optimization {
    use super::*;

    fn make_memory(text: &str, user: &str, ts: i64, emb: Vec<f32>) -> HashMap<PropertyKey, Value> {
        let mut p = HashMap::new();
        p.insert(PropertyKey::new("text"), Value::String(text.into()));
        p.insert(PropertyKey::new("user_id"), Value::String(user.into()));
        p.insert(PropertyKey::new("created_at"), Value::Int64(ts));
        p.insert(PropertyKey::new("embedding"), Value::Vector(emb.into()));
        p
    }

    #[test]
    fn operator_filter_gte_works() {
        let db = db();

        // Create index first
        db.create_vector_index("Memory", "embedding", Some(3), None, None, None, None)
            .unwrap();

        // Create nodes via batch (auto-inserts into vector index)
        db.batch_create_nodes_with_props(
            "Memory",
            vec![
                make_memory("old", "u1", 1000, vec![1.0, 0.0, 0.0]),
                make_memory("new", "u1", 2000, vec![0.0, 1.0, 0.0]),
                make_memory("newest", "u1", 3000, vec![0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();

        // Filter: created_at >= 2000 should return 2 results
        let mut filters = HashMap::new();
        let mut gte_map = BTreeMap::new();
        gte_map.insert(PropertyKey::new("$gte"), Value::Int64(2000));
        filters.insert("created_at".to_string(), Value::Map(Arc::new(gte_map)));

        let results = db
            .vector_search(
                "Memory",
                "embedding",
                &[1.0, 0.0, 0.0],
                10,
                None,
                Some(&filters),
            )
            .unwrap();
        assert_eq!(
            results.len(),
            2,
            "Should find 2 memories with created_at >= 2000"
        );
    }

    #[test]
    fn combined_equality_and_operator_filter() {
        let db = db();

        db.create_vector_index("Memory", "embedding", Some(2), None, None, None, None)
            .unwrap();

        let mut p1 = HashMap::new();
        p1.insert(PropertyKey::new("text"), Value::String("u1 old".into()));
        p1.insert(PropertyKey::new("user_id"), Value::String("u1".into()));
        p1.insert(PropertyKey::new("created_at"), Value::Int64(1000));
        p1.insert(
            PropertyKey::new("embedding"),
            Value::Vector(vec![1.0, 0.0].into()),
        );

        let mut p2 = HashMap::new();
        p2.insert(PropertyKey::new("text"), Value::String("u1 new".into()));
        p2.insert(PropertyKey::new("user_id"), Value::String("u1".into()));
        p2.insert(PropertyKey::new("created_at"), Value::Int64(2000));
        p2.insert(
            PropertyKey::new("embedding"),
            Value::Vector(vec![0.0, 1.0].into()),
        );

        let mut p3 = HashMap::new();
        p3.insert(PropertyKey::new("text"), Value::String("u2 new".into()));
        p3.insert(PropertyKey::new("user_id"), Value::String("u2".into()));
        p3.insert(PropertyKey::new("created_at"), Value::Int64(2000));
        p3.insert(
            PropertyKey::new("embedding"),
            Value::Vector(vec![1.0, 1.0].into()),
        );

        db.batch_create_nodes_with_props("Memory", vec![p1, p2, p3])
            .unwrap();

        // Filter: user_id = u1 AND created_at >= 2000
        let mut filters = HashMap::new();
        filters.insert("user_id".to_string(), Value::String("u1".into()));
        let mut gte_map = BTreeMap::new();
        gte_map.insert(PropertyKey::new("$gte"), Value::Int64(2000));
        filters.insert("created_at".to_string(), Value::Map(Arc::new(gte_map)));

        let results = db
            .vector_search("Memory", "embedding", &[1.0, 0.0], 10, None, Some(&filters))
            .unwrap();
        assert_eq!(results.len(), 1, "Should find only u1's new memory");
    }

    #[test]
    fn operator_filter_lt_works() {
        let db = db();

        db.create_vector_index("Memory", "embedding", Some(2), None, None, None, None)
            .unwrap();

        db.batch_create_nodes_with_props(
            "Memory",
            vec![
                {
                    let mut p = HashMap::new();
                    p.insert(PropertyKey::new("score"), Value::Int64(10));
                    p.insert(
                        PropertyKey::new("embedding"),
                        Value::Vector(vec![1.0, 0.0].into()),
                    );
                    p
                },
                {
                    let mut p = HashMap::new();
                    p.insert(PropertyKey::new("score"), Value::Int64(50));
                    p.insert(
                        PropertyKey::new("embedding"),
                        Value::Vector(vec![0.0, 1.0].into()),
                    );
                    p
                },
                {
                    let mut p = HashMap::new();
                    p.insert(PropertyKey::new("score"), Value::Int64(90));
                    p.insert(
                        PropertyKey::new("embedding"),
                        Value::Vector(vec![1.0, 1.0].into()),
                    );
                    p
                },
            ],
        )
        .unwrap();

        let mut filters = HashMap::new();
        let mut lt_map = BTreeMap::new();
        lt_map.insert(PropertyKey::new("$lt"), Value::Int64(50));
        filters.insert("score".to_string(), Value::Map(Arc::new(lt_map)));

        let results = db
            .vector_search("Memory", "embedding", &[1.0, 0.0], 10, None, Some(&filters))
            .unwrap();
        assert_eq!(results.len(), 1, "Should find only score < 50");
    }
}

// =============================================================================
// Temporal property versioning API
// =============================================================================

#[cfg(feature = "temporal")]
mod temporal_versioning {
    use super::*;

    #[test]
    fn get_node_property_at_epoch() {
        let db = db();
        let s = db.session();

        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let e1 = db.current_epoch();

        s.execute("MATCH (p:Person {name: 'Alix'}) SET p.name = 'Alicia'")
            .unwrap();
        let e2 = db.current_epoch();

        // At epoch 1, name should be Alix
        let val = db.get_node_property_at_epoch(NodeId(0), "name", e1);
        assert_eq!(val, Some(Value::String("Alix".into())));

        // At current epoch, name should be Alicia
        let val = db.get_node_property_at_epoch(NodeId(0), "name", e2);
        assert_eq!(val, Some(Value::String("Alicia".into())));
    }

    #[test]
    fn get_node_property_history() {
        let db = db();
        let s = db.session();

        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("MATCH (p:Person {name: 'Alix'}) SET p.name = 'Alicia'")
            .unwrap();
        s.execute("MATCH (p:Person {name: 'Alicia'}) SET p.name = 'Ali'")
            .unwrap();

        let history = db.get_node_property_history(NodeId(0), "name");
        assert_eq!(
            history.len(),
            3,
            "Should have 3 versions: Alix, Alicia, Ali"
        );
        assert_eq!(history[0].1, Value::String("Alix".into()));
        assert_eq!(history[1].1, Value::String("Alicia".into()));
        assert_eq!(history[2].1, Value::String("Ali".into()));
    }

    #[test]
    fn get_all_node_property_history() {
        let db = db();
        let s = db.session();

        s.execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        s.execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
            .unwrap();

        let all_history = db.get_all_node_property_history(NodeId(0));
        // Should have entries for 'name' (1 version) and 'age' (2 versions)
        let name_hist: Vec<_> = all_history
            .iter()
            .filter(|(k, _)| k.as_ref() == "name")
            .collect();
        let age_hist: Vec<_> = all_history
            .iter()
            .filter(|(k, _)| k.as_ref() == "age")
            .collect();

        assert_eq!(name_hist.len(), 1);
        assert_eq!(name_hist[0].1.len(), 1, "name has 1 version");
        assert_eq!(age_hist.len(), 1);
        assert_eq!(age_hist[0].1.len(), 2, "age has 2 versions");
    }

    #[test]
    fn property_at_epoch_returns_none_for_missing() {
        let db = db();
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        let epoch = db.current_epoch();

        // Property that doesn't exist
        let val = db.get_node_property_at_epoch(NodeId(0), "nonexistent", epoch);
        assert_eq!(val, None);

        // Node that doesn't exist
        let val = db.get_node_property_at_epoch(NodeId(999), "name", epoch);
        assert_eq!(val, None);
    }

    #[test]
    fn property_history_empty_for_nonexistent() {
        let db = db();
        let history = db.get_node_property_history(NodeId(999), "name");
        assert!(history.is_empty());
    }

    #[test]
    fn epochs_are_ascending() {
        let db = db();
        let s = db.session();

        s.execute("INSERT (:Counter {val: 0})").unwrap();
        s.execute("MATCH (c:Counter) SET c.val = 1").unwrap();
        s.execute("MATCH (c:Counter) SET c.val = 2").unwrap();

        let history = db.get_node_property_history(NodeId(0), "val");
        let epochs: Vec<u64> = history.iter().map(|(e, _)| e.as_u64()).collect();
        assert_eq!(epochs, {
            let mut sorted = epochs.clone();
            sorted.sort_unstable();
            sorted
        });
    }
}
