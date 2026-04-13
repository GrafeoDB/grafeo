use super::*;
use crate::graph::compact::CompactStoreBuilder;
use crate::graph::compact::id::encode_node_id;
use crate::graph::Direction;
use grafeo_common::types::NodeId;

/// Shared fixture: 100 Students, 50 Courses, 150 edges
fn test_compact() -> crate::graph::compact::CompactStore {
    let grades: Vec<u64> = (0..100).collect();
    let names: Vec<&str> = (0..100).map(|i| if i % 2 == 0 { "Alice" } else { "Bob" }).collect();
    let active: Vec<bool> = (0..100).map(|i| i % 3 != 0).collect();

    let levels: Vec<u64> = (0..50).map(|i| (i % 5) as u64).collect();
    let depts: Vec<&str> = (0..50).map(|i| if i % 2 == 0 { "CS" } else { "Math" }).collect();

    let mut edges: Vec<(u32, u32)> = Vec::new();
    for i in 0..100u32 {
        for k in 0..3u32 {
            let course_offset = ((i + k) % 50) as u32;
            if !edges.contains(&(i, course_offset)) {
                edges.push((i, course_offset));
            }
        }
    }
    edges.truncate(150);

    CompactStoreBuilder::new()
        .node_table("Student", |b| {
            b.column_bitpacked("grade", &grades, 8)
            .column_dict("name", &names)
            .column_bitmap("active", &active)
        })
        .node_table("Course", |b| {
            b.column_bitpacked("level", &levels, 4)
            .column_dict("dept", &depts)
        })
        .rel_table("ENROLLED_IN", "Student", "Course", |b| {
                b.edges(edges).backward(true)
            })
        .build()
        .unwrap()
}

fn test_hybrid() -> HybridStore {
    HybridStore::open(test_compact()).unwrap()
}

fn student_id(i: usize) -> NodeId {
    encode_node_id(0, i as u64)
}

fn course_id(i: usize) -> NodeId {
    encode_node_id(1, i as u64)
}

#[test]
fn open_creates_hybrid_with_correct_offset() {
    let h = test_hybrid();
    let offset = h.id_offset();
    let expected = encode_node_id(1, 49).as_u64() + 1;
    assert_eq!(offset, expected);
}

#[test]
fn overlay_is_empty_after_open() {
    let h = test_hybrid();
    assert_eq!(h.overlay().node_count(), 0);
    assert_eq!(h.overlay().edge_count(), 0);
}

#[test]
fn is_compact_node_id_correct() {
    let h = test_hybrid();
    assert!(h.is_compact_node_id(student_id(0)));
    assert!(h.is_compact_node_id(student_id(99)));
    assert!(h.is_compact_node_id(course_id(0)));
    assert!(h.is_compact_node_id(course_id(49)));
    assert!(!h.is_compact_node_id(NodeId::new(h.id_offset())));
    assert!(!h.is_compact_node_id(NodeId::new(h.id_offset() + 100)));
}

// ---------------------------------------------------------------
// Task 3: GraphStore point reads
// ---------------------------------------------------------------

use crate::graph::GraphStore;
use crate::graph::GraphStoreMut;
use grafeo_common::types::{PropertyKey, TransactionId, Value};

#[test]
fn read_compact_node() {
    let h = test_hybrid();
    let node = h.get_node(student_id(0)).unwrap();
    assert!(node.labels.iter().any(|l| l.as_str() == "Student"));
    assert_eq!(
        node.properties.get(&PropertyKey::new("grade")),
        Some(&Value::Int64(0))
    );
}

#[test]
fn read_compact_node_property() {
    let h = test_hybrid();
    let val = h.get_node_property(student_id(5), &PropertyKey::new("grade"));
    assert_eq!(val, Some(Value::Int64(5)));
}

#[test]
fn read_nonexistent_node() {
    let h = test_hybrid();
    assert!(h.get_node(NodeId::new(999_999_999)).is_none());
}

#[test]
fn read_compact_node_with_property_override() {
    let h = test_hybrid();
    h.set_node_property(student_id(0), "grade", Value::Int64(999));
    let node = h.get_node(student_id(0)).unwrap();
    assert_eq!(
        node.properties.get(&PropertyKey::new("grade")),
        Some(&Value::Int64(999))
    );
    // Original compact property "name" should still be present
    assert!(node.properties.contains_key(&PropertyKey::new("name")));
}

#[test]
fn read_overlay_node() {
    let h = test_hybrid();
    let nid = h.overlay().create_node(&["Person"]);
    h.overlay()
        .set_node_property(nid, "age", Value::Int64(30));
    let node = h.get_node(nid).unwrap();
    assert!(node.labels.iter().any(|l| l.as_str() == "Person"));
    assert_eq!(
        node.properties.get(&PropertyKey::new("age")),
        Some(&Value::Int64(30))
    );
}

#[test]
fn node_count_is_compact_plus_overlay() {
    let h = test_hybrid();
    assert_eq!(h.node_count(), 150); // 100 students + 50 courses
    h.overlay().create_node(&["Person"]);
    assert_eq!(h.node_count(), 151);
}

#[test]
fn node_ids_merges_both_stores() {
    let h = test_hybrid();
    let nid = h.overlay().create_node(&["Extra"]);
    let ids = h.node_ids();
    assert!(ids.contains(&student_id(0)));
    assert!(ids.contains(&course_id(49)));
    assert!(ids.contains(&nid));
    assert_eq!(ids.len(), 151);
}

#[test]
fn nodes_by_label_merges() {
    let h = test_hybrid();
    h.overlay().create_node(&["Student"]);
    let students = h.nodes_by_label("Student");
    // 100 compact Students + 1 overlay Student
    assert_eq!(students.len(), 101);
}

#[test]
fn get_node_property_prefers_overlay() {
    let h = test_hybrid();
    // Set an overlay override (must go through HybridStore to mark dirty)
    h.set_node_property(student_id(3), "grade", Value::Int64(42));
    let val = h.get_node_property(student_id(3), &PropertyKey::new("grade"));
    assert_eq!(val, Some(Value::Int64(42)));
    // Property without override should still come from compact
    let name = h.get_node_property(student_id(3), &PropertyKey::new("name"));
    assert!(name.is_some());
}

// ---------------------------------------------------------------
// Task 4: GraphStoreMut — write delegation
// ---------------------------------------------------------------

#[test]
fn create_node_in_overlay() {
    let h = test_hybrid();
    let nid = h.create_node(&["Person"]);
    assert!(!h.is_compact_node_id(nid));
    assert!(h.get_node(nid).is_some());
}

#[test]
fn create_edge_between_compact_nodes() {
    let h = test_hybrid();
    let eid = h.create_edge(student_id(0), course_id(0), "LIKES");
    assert!(h.get_edge(eid).is_some());
}

#[test]
fn set_property_on_compact_node() {
    let h = test_hybrid();
    h.set_node_property(student_id(5), "grade", Value::Int64(999));
    assert_eq!(
        h.get_node_property(student_id(5), &PropertyKey::new("grade")),
        Some(Value::Int64(999))
    );
}

#[test]
fn delete_compact_node() {
    let h = test_hybrid();
    assert!(h.get_node(student_id(0)).is_some());
    let deleted = h.delete_node(student_id(0));
    assert!(deleted);
    assert!(h.get_node(student_id(0)).is_none());
}

#[test]
fn delete_compact_edge() {
    let h = test_hybrid();
    let edges = h.edges_from(student_id(0), crate::graph::Direction::Outgoing);
    assert!(!edges.is_empty());
    let (_, eid) = edges[0];
    assert!(h.delete_edge(eid));
    assert!(h.get_edge(eid).is_none());
}

// ---------------------------------------------------------------
// Task 5: Rollback/Commit hooks
// ---------------------------------------------------------------

#[test]
fn rollback_compact_node_deletion_restores_node() {
    let h = test_hybrid();
    let tx = TransactionId::new(100);
    let epoch = h.overlay().current_epoch();

    // Delete compact node within a transaction
    assert!(h.delete_node_versioned(student_id(0), epoch, tx));
    assert!(h.get_node(student_id(0)).is_none());

    // Rollback: overlay handles version chain, we handle deletion set
    h.on_rollback(tx);
    h.overlay().rollback_transaction_properties(tx);
    h.overlay().discard_uncommitted_versions(tx);

    // Node should be visible again
    assert!(h.get_node(student_id(0)).is_some());
}

#[test]
fn commit_compact_deletion_clears_pending() {
    let h = test_hybrid();
    let tx = TransactionId::new(100);
    let epoch = h.overlay().current_epoch();

    h.delete_node_versioned(student_id(0), epoch, tx);
    h.on_commit(tx);

    // Node stays deleted
    assert!(h.get_node(student_id(0)).is_none());
    // But pending_deletions for this tx should be cleared
    assert!(h.pending_deletions.read().get(&tx).is_none());
}

// ---------------------------------------------------------------
// Task 6: Compaction
// ---------------------------------------------------------------

#[test]
fn compact_merges_overlay_into_compact() {
    let h = test_hybrid();
    // Add overlay entity
    let nid = h.create_node(&["Person"]);
    h.set_node_property(nid, "age", Value::Int64(30));
    // Override compact entity property
    h.set_node_property(student_id(0), "grade", Value::Int64(999));

    let count_before = h.node_count();
    h.compact().unwrap();

    // Count should be preserved
    assert_eq!(h.node_count(), count_before);
    // Overlay should be empty
    assert_eq!(h.overlay().node_count(), 0);
    // Deleted set should be clear
    assert!(h.deleted_nodes.read().is_empty());
    // New entity should now be in compact
    assert!(!h.compact.read().nodes_by_label("Person").is_empty());
}

#[test]
fn compact_preserves_query_results() {
    let h = test_hybrid();
    let nid = h.create_node(&["Person"]);
    h.set_node_property(nid, "name", Value::String(arcstr::literal!("Alix")));
    h.create_edge(nid, student_id(0), "KNOWS");

    let nodes_before = h.node_count();
    let edges_before = h.edge_count();
    let labels_before = h.all_labels();

    h.compact().unwrap();

    assert_eq!(h.node_count(), nodes_before);
    assert_eq!(h.edge_count(), edges_before);
    assert_eq!(h.all_labels(), labels_before);
}

#[test]
fn double_compact() {
    let h = test_hybrid();
    h.create_node(&["A"]);
    h.compact().unwrap();
    h.create_node(&["B"]);
    h.compact().unwrap();
    // Both A and B should be in compact
    assert!(!h.compact.read().nodes_by_label("A").is_empty());
    assert!(!h.compact.read().nodes_by_label("B").is_empty());
    assert_eq!(h.overlay().node_count(), 0);
}

// ---------------------------------------------------------------
// Tier 2: Edge cases, deletions, labels, traversal, serialization
// ---------------------------------------------------------------

// 1. Edge property override on compact edge
#[test]
fn edge_property_override_compact_edge() {
    let h = test_hybrid();
    let edges = h.edges_from(student_id(0), Direction::Outgoing);
    let (_, eid) = edges[0];
    h.set_edge_property(eid, "weight", Value::Float64(0.5));
    assert_eq!(
        h.get_edge_property(eid, &PropertyKey::new("weight")),
        Some(Value::Float64(0.5))
    );
}

// 2. Overlay-to-overlay edges
#[test]
fn overlay_to_overlay_edge() {
    let h = test_hybrid();
    let a = h.create_node(&["X"]);
    let b = h.create_node(&["Y"]);
    let eid = h.create_edge(a, b, "LINKS");
    assert!(h.get_edge(eid).is_some());
    let neighbors = h.neighbors(a, Direction::Outgoing);
    assert!(neighbors.contains(&b));
}

// 3. Cross-store edges (overlay-to-compact)
#[test]
fn cross_store_edge() {
    let h = test_hybrid();
    let overlay_node = h.create_node(&["NewPerson"]);
    let eid = h.create_edge(overlay_node, student_id(0), "MENTORS");
    assert!(h.get_edge(eid).is_some());
    let neighbors = h.neighbors(overlay_node, Direction::Outgoing);
    assert!(neighbors.contains(&student_id(0)));
}

// 4. Delete then re-check counts
#[test]
fn delete_compact_node_decrements_count() {
    let h = test_hybrid();
    let before = h.node_count();
    h.delete_node(student_id(0));
    assert_eq!(h.node_count(), before - 1);
}

// 5. Double deletion returns false
#[test]
fn double_delete_returns_false() {
    let h = test_hybrid();
    assert!(h.delete_node(student_id(0)));
    assert!(!h.delete_node(student_id(0)));
}

// 6. Deleted node not in node_ids
#[test]
fn deleted_node_excluded_from_node_ids() {
    let h = test_hybrid();
    h.delete_node(student_id(0));
    let ids = h.node_ids();
    assert!(!ids.contains(&student_id(0)));
}

// 7. Deleted node not in nodes_by_label
#[test]
fn deleted_node_excluded_from_nodes_by_label() {
    let h = test_hybrid();
    h.delete_node(student_id(0));
    let students = h.nodes_by_label("Student");
    assert!(!students.contains(&student_id(0)));
}

// 8. Label mutation on compact node — verify via nodes_by_label index
// (compact_node_merged returns only compact labels; label mutations are
// reflected in the overlay's label index which nodes_by_label queries)
#[test]
fn add_label_to_compact_node() {
    let h = test_hybrid();
    h.add_label(student_id(0), "VIP");
    // New label should be queryable via label index
    let vips = h.nodes_by_label("VIP");
    assert!(vips.contains(&student_id(0)));
    // Original label should still be present in label index
    let students = h.nodes_by_label("Student");
    assert!(students.contains(&student_id(0)));
}

// 9. All labels includes both stores
#[test]
fn all_labels_includes_overlay() {
    let h = test_hybrid();
    h.create_node(&["NewLabel"]);
    let labels = h.all_labels();
    assert!(labels.contains(&"Student".to_string()));
    assert!(labels.contains(&"Course".to_string()));
    assert!(labels.contains(&"NewLabel".to_string()));
}

// 10. All edge types includes both stores
#[test]
fn all_edge_types_includes_overlay() {
    let h = test_hybrid();
    let a = h.create_node(&["X"]);
    let b = h.create_node(&["Y"]);
    h.create_edge(a, b, "NEW_TYPE");
    let types = h.all_edge_types();
    assert!(types.contains(&"ENROLLED_IN".to_string()));
    assert!(types.contains(&"NEW_TYPE".to_string()));
}

// 11. find_nodes_by_property on overlay
#[test]
fn find_nodes_by_property_overlay() {
    let h = test_hybrid();
    let nid = h.create_node(&["Person"]);
    h.set_node_property(nid, "unique_key", Value::Int64(42));
    let found = h.find_nodes_by_property("unique_key", &Value::Int64(42));
    assert!(found.contains(&nid));
}

// 12. Compact traversal both directions
#[test]
fn compact_traversal_both_directions() {
    let h = test_hybrid();
    let out = h.edges_from(student_id(0), Direction::Outgoing);
    assert!(!out.is_empty());
    // Incoming on a Course node (backward adjacency was built)
    let inc = h.edges_from(course_id(0), Direction::Incoming);
    assert!(!inc.is_empty());
}

// 13. Serialization roundtrip preserves hybrid behavior
#[test]
fn serialization_roundtrip_hybrid() {
    use crate::graph::compact::CompactStore;
    let compact = test_compact();
    let bytes = compact.to_bytes().unwrap();
    let restored = CompactStore::from_bytes(&bytes).unwrap();
    let h = HybridStore::open(restored).unwrap();
    // Should work the same as fresh hybrid
    assert!(h.get_node(student_id(0)).is_some());
    let nid = h.create_node(&["Test"]);
    assert!(h.get_node(nid).is_some());
}

// 14. Property override does NOT cause false positive in find_nodes_by_property
#[test]
fn find_nodes_by_property_no_false_positive_on_override() {
    let h = test_hybrid();
    // Student 5 originally has grade=5
    assert_eq!(
        h.get_node_property(student_id(5), &PropertyKey::new("grade")),
        Some(Value::Int64(5))
    );
    // Override grade to 999
    h.set_node_property(student_id(5), "grade", Value::Int64(999));
    // Searching for grade=5 should NOT return student 5 anymore
    let found = h.find_nodes_by_property("grade", &Value::Int64(5));
    assert!(!found.contains(&student_id(5)));
    // Point read correctly returns the overridden value
    assert_eq!(
        h.get_node_property(student_id(5), &PropertyKey::new("grade")),
        Some(Value::Int64(999))
    );
}

// 15. compact() works through &self (no &mut self required)
#[test]
fn remove_property_on_compact_node() {
    let h = test_hybrid();
    // Student 0 originally has grade=0
    assert!(h.get_node_property(student_id(0), &PropertyKey::new("grade")).is_some());
    h.remove_node_property(student_id(0), "grade");
    // Should be gone now
    assert_eq!(h.get_node_property(student_id(0), &PropertyKey::new("grade")), None);
    // Other properties should still be visible
    assert!(h.get_node_property(student_id(0), &PropertyKey::new("name")).is_some());
}

#[test]
fn remove_property_on_compact_node_full_node_read() {
    let h = test_hybrid();
    // Verify via full node read (compact_node_merged path)
    h.remove_node_property(student_id(1), "grade");
    let node = h.get_node(student_id(1)).unwrap();
    assert!(!node.properties.contains_key(&PropertyKey::new("grade")));
    assert!(node.properties.contains_key(&PropertyKey::new("name")));
}

#[test]
fn remove_property_on_compact_edge() {
    let h = test_hybrid();
    let edges = h.edges_from(student_id(0), Direction::Outgoing);
    let (_, eid) = edges[0];
    // First set a property so we can remove it
    h.set_edge_property(eid, "weight", Value::Float64(1.0));
    assert_eq!(
        h.get_edge_property(eid, &PropertyKey::new("weight")),
        Some(Value::Float64(1.0))
    );
    h.remove_edge_property(eid, "weight");
    assert_eq!(h.get_edge_property(eid, &PropertyKey::new("weight")), None);
}

// Tombstoned properties must not appear in any scan path
#[test]
fn tombstone_excluded_from_find_nodes_by_property() {
    let h = test_hybrid();
    // Student 0 has grade=0. Remove it.
    h.remove_node_property(student_id(0), "grade");
    let found = h.find_nodes_by_property("grade", &Value::Int64(0));
    assert!(!found.contains(&student_id(0)));
}

#[test]
fn tombstone_excluded_from_find_nodes_by_properties() {
    let h = test_hybrid();
    // Student 0 has grade=0 and name="Alice". Remove grade.
    h.remove_node_property(student_id(0), "grade");
    // Multi-condition: grade=0 AND name="Alice" should NOT find student 0
    let found = h.find_nodes_by_properties(&[
        ("grade", Value::Int64(0)),
        ("name", Value::String(arcstr::literal!("Alice"))),
    ]);
    assert!(!found.contains(&student_id(0)));
    // name="Alice" alone should still find student 0
    let found = h.find_nodes_by_property("name", &Value::String(arcstr::literal!("Alice")));
    assert!(found.contains(&student_id(0)));
}

#[test]
fn tombstone_excluded_from_find_nodes_in_range() {
    let h = test_hybrid();
    // Student 5 has grade=5. Remove it.
    h.remove_node_property(student_id(5), "grade");
    let found = h.find_nodes_in_range(
        "grade",
        Some(&Value::Int64(4)),
        Some(&Value::Int64(6)),
        true,
        true,
    );
    assert!(!found.contains(&student_id(5)));
}

#[test]
fn compact_works_through_shared_ref() {
    let h = std::sync::Arc::new(test_hybrid());
    h.create_node(&["Shared"]);
    h.compact().unwrap();
    assert_eq!(h.overlay().node_count(), 0);
    assert!(!h.compact.read().nodes_by_label("Shared").is_empty());
}

#[test]
fn incremental_compact_skips_unchanged_tables() {
    let h = test_hybrid();
    // Modify only one property on one student — table "Course" should be cloned
    h.set_node_property(student_id(0), "grade", Value::Int64(999));
    h.compact().unwrap();
    // Verify data is correct after incremental compaction
    assert_eq!(h.node_count(), 150); // 100 students + 50 courses
    // The modified property should be in the rebuilt compact
    assert_eq!(
        h.get_node_property(student_id(0), &PropertyKey::new("grade")),
        Some(Value::Int64(999))
    );
    // Unmodified course should be intact
    assert!(h.get_node(course_id(0)).is_some());
}
