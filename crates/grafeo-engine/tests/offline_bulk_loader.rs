use std::collections::HashMap;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

#[test]
fn bulk_loader_preserves_rows_and_node_ids_without_secondary_indexes() {
    let db = GrafeoDB::new_in_memory();
    let rows = (0..10_000)
        .map(|index| {
            HashMap::from([
                (PropertyKey::new("key"), Value::from(format!("row-{index}"))),
                (PropertyKey::new("ordinal"), Value::from(i64::from(index))),
            ])
        })
        .collect();

    let ids = db
        .bulk_load_nodes_with_props_unindexed("BulkRow", rows)
        .expect("fresh store accepts an offline bulk load");

    assert_eq!(ids.len(), 10_000);
    assert_eq!(db.graph_store().nodes_by_label("BulkRow").len(), 10_000);
    assert_eq!(
        db.graph_store()
            .get_node_property(ids[9_999], &PropertyKey::new("key")),
        Some(Value::from("row-9999"))
    );
}

#[test]
fn bulk_loader_rejects_a_store_after_vector_index_creation() {
    let db = GrafeoDB::new_in_memory();
    db.create_vector_index("BulkRow", "embedding", Some(2), None, None, None, None)
        .expect("empty vector index is valid with explicit dimensions");

    let result = db.bulk_load_nodes_with_props_unindexed(
        "BulkRow",
        vec![HashMap::from([(
            PropertyKey::new("key"),
            Value::from("row"),
        )])],
    );

    assert!(result.is_err());
}

#[test]
fn bulk_edge_loader_writes_adjacency_without_online_mutation_path() {
    let db = GrafeoDB::new_in_memory();
    let nodes = db
        .bulk_load_nodes_with_props_unindexed(
            "BulkRow",
            (0..10_000)
                .map(|index| HashMap::from([(PropertyKey::new("key"), Value::from(index as i64))]))
                .collect(),
        )
        .expect("fresh store accepts nodes");
    let edges: Vec<_> = nodes
        .windows(2)
        .map(|pair| (pair[0], pair[1], "NEXT"))
        .collect();

    let ids = db.bulk_load_edges_unindexed(&edges);

    assert_eq!(ids.len(), edges.len());
    assert_eq!(db.graph_store().edge_count(), edges.len());
}
