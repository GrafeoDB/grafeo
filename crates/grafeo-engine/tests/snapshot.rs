//! Integration tests for snapshot export/import.

use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};
use grafeo_engine::GrafeoDB;

/// Mirror of the private Snapshot struct for crafting test payloads.
#[derive(serde::Serialize, serde::Deserialize)]
struct TestSnapshot {
    version: u8,
    nodes: Vec<TestNode>,
    edges: Vec<TestEdge>,
    named_graphs: Vec<()>,
    rdf_triples: Vec<()>,
    rdf_named_graphs: Vec<()>,
    schema: TestSnapshotSchema,
    indexes: TestSnapshotIndexes,
    epoch: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct TestSnapshotSchema {
    node_types: Vec<()>,
    edge_types: Vec<()>,
    graph_types: Vec<()>,
    procedures: Vec<()>,
    schemas: Vec<()>,
    graph_type_bindings: Vec<()>,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct TestSnapshotIndexes {
    property_indexes: Vec<()>,
    vector_indexes: Vec<()>,
    text_indexes: Vec<()>,
}

impl TestSnapshot {
    fn new(version: u8, nodes: Vec<TestNode>, edges: Vec<TestEdge>) -> Self {
        Self {
            version,
            nodes,
            edges,
            named_graphs: vec![],
            rdf_triples: vec![],
            rdf_named_graphs: vec![],
            schema: TestSnapshotSchema::default(),
            indexes: TestSnapshotIndexes::default(),
            epoch: 0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestNode {
    id: NodeId,
    labels: Vec<String>,
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TestEdge {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    properties: Vec<(String, Vec<(EpochId, Value)>)>,
}

fn encode_snapshot(snap: &TestSnapshot) -> Vec<u8> {
    bincode::serde::encode_to_vec(snap, bincode::config::standard()).unwrap()
}

#[test]
fn export_import_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    assert_eq!(restored.node_count(), 0);
    assert_eq!(restored.edge_count(), 0);
}

#[test]
fn export_import_preserves_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
}

#[test]
fn export_import_preserves_edges() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]).unwrap();
    db.set_node_property(alix, "name", "Alix".into()).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    db.set_node_property(gus, "name", "Gus".into()).unwrap();
    db.create_edge(alix, gus, "KNOWS");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);
    assert_eq!(restored.edge_count(), 1);

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[test]
fn export_import_preserves_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Item {name: 'Widget', price: 9.99, active: true})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (i:Item) RETURN i.name, i.price, i.active")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[test]
fn import_rejects_invalid_data() {
    let result = GrafeoDB::import_snapshot(b"not a valid snapshot");
    assert!(result.is_err());
}

#[test]
fn snapshot_round_trip_schema() {
    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]).unwrap();
    db.set_node_property(alix, "name", "Alix".into()).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    db.set_node_property(gus, "name", "Gus".into()).unwrap();
    db.create_edge(alix, gus, "KNOWS");

    let schema_before = db.schema();
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    let schema_after = restored.schema();

    // Both schemas should report the same label/edge info
    let fmt_before = format!("{schema_before:?}");
    let fmt_after = format!("{schema_after:?}");
    assert_eq!(fmt_before, fmt_after);
}

// --- Edge property round-trip ---

#[test]
fn export_import_preserves_edge_properties() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]).unwrap();
    let b = db.create_node(&["Person"]).unwrap();
    let edge = db.create_edge(a, b, "KNOWS");
    db.set_edge_property(edge, "since", Value::Int64(2020));
    db.set_edge_property(edge, "strength", Value::Float64(0.95));

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session
        .execute("MATCH ()-[e:KNOWS]->() RETURN e.since, e.strength")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(2020));
    assert_eq!(result.rows()[0][1], Value::Float64(0.95));
}

// --- Multi-label nodes ---

#[test]
fn export_import_preserves_multiple_labels() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Person", "Employee"]).unwrap();
    db.create_node(&["Person", "Manager"]).unwrap();
    db.create_node(&["Animal"]).unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 3);

    let session = restored.session();
    let persons = session.execute("MATCH (p:Person) RETURN p").unwrap();
    assert_eq!(persons.rows().len(), 2);

    let employees = session.execute("MATCH (e:Employee) RETURN e").unwrap();
    assert_eq!(employees.rows().len(), 1);

    let managers = session.execute("MATCH (m:Manager) RETURN m").unwrap();
    assert_eq!(managers.rows().len(), 1);

    let animals = session.execute("MATCH (a:Animal) RETURN a").unwrap();
    assert_eq!(animals.rows().len(), 1);
}

// --- Temporal Value types ---

#[test]
fn export_import_preserves_temporal_values() {
    use grafeo_common::types::{Date, Duration, Time, Timestamp, ZonedDatetime};

    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Temporal"]).unwrap();

    let date = Date::from_ymd(2025, 6, 15).unwrap();
    let time = Time::from_hms(14, 30, 0).unwrap();
    let timestamp = Timestamp::from_secs(1_700_000_000);
    let duration = Duration::new(1, 15, 3_600_000_000_000); // 1 month, 15 days, 1 hour
    let zoned = ZonedDatetime::from_timestamp_offset(Timestamp::from_secs(1_700_000_000), 3600);

    db.set_node_property(id, "date_val", Value::Date(date))
        .unwrap();
    db.set_node_property(id, "time_val", Value::Time(time))
        .unwrap();
    db.set_node_property(id, "ts_val", Value::Timestamp(timestamp))
        .unwrap();
    db.set_node_property(id, "dur_val", Value::Duration(duration))
        .unwrap();
    db.set_node_property(id, "zdt_val", Value::ZonedDatetime(zoned))
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session
        .execute("MATCH (t:Temporal) RETURN t.date_val, t.time_val, t.ts_val, t.dur_val, t.zdt_val")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Date(date));
    assert_eq!(result.rows()[0][1], Value::Time(time));
    assert_eq!(result.rows()[0][2], Value::Timestamp(timestamp));
    assert_eq!(result.rows()[0][3], Value::Duration(duration));
    assert_eq!(result.rows()[0][4], Value::ZonedDatetime(zoned));
}

// --- All scalar Value types ---

#[test]
fn export_import_preserves_all_value_types() {
    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Test"]).unwrap();
    db.set_node_property(id, "str_val", Value::String("hello".into()))
        .unwrap();
    db.set_node_property(id, "int_val", Value::Int64(42))
        .unwrap();
    db.set_node_property(id, "float_val", Value::Float64(9.81))
        .unwrap();
    db.set_node_property(id, "bool_val", Value::Bool(true))
        .unwrap();
    db.set_node_property(id, "null_val", Value::Null).unwrap();
    db.set_node_property(
        id,
        "bytes_val",
        Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF].into()),
    )
    .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session
        .execute("MATCH (t:Test) RETURN t.str_val, t.int_val, t.float_val, t.bool_val, t.null_val, t.bytes_val")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("hello".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(42));
    assert_eq!(result.rows()[0][2], Value::Float64(9.81));
    assert_eq!(result.rows()[0][3], Value::Bool(true));
    assert_eq!(result.rows()[0][4], Value::Null);
    assert_eq!(
        result.rows()[0][5],
        Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF].into())
    );
}

// --- List and Map values ---

#[test]
fn export_import_preserves_collection_values() {
    let db = GrafeoDB::new_in_memory();
    let id = db.create_node(&["Test"]).unwrap();
    db.set_node_property(
        id,
        "tags",
        Value::List(vec![Value::String("a".into()), Value::String("b".into())].into()),
    )
    .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session = restored.session();
    let result = session.execute("MATCH (t:Test) RETURN t.tags").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::List(vec![Value::String("a".into()), Value::String("b".into()),].into())
    );
}

// --- Multiple edge types ---

#[test]
fn export_import_preserves_multiple_edge_types() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]).unwrap();
    let b = db.create_node(&["Person"]).unwrap();
    let c = db.create_node(&["Company"]).unwrap();

    db.create_edge(a, b, "KNOWS");
    db.create_edge(a, c, "WORKS_AT");
    db.create_edge(b, c, "WORKS_AT");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 3);
    assert_eq!(restored.edge_count(), 3);

    let session = restored.session();
    let knows = session.execute("MATCH ()-[e:KNOWS]->() RETURN e").unwrap();
    assert_eq!(knows.rows().len(), 1);

    let works = session
        .execute("MATCH ()-[e:WORKS_AT]->() RETURN e")
        .unwrap();
    assert_eq!(works.rows().len(), 2);
}

// --- Nodes with no properties ---

#[test]
fn export_import_preserves_empty_property_nodes() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Empty"]).unwrap();
    db.create_node(&["Empty"]).unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);

    let session = restored.session();
    let result = session.execute("MATCH (e:Empty) RETURN e").unwrap();
    assert_eq!(result.rows().len(), 2);
}

// --- Moderate dataset ---

#[test]
fn export_import_moderate_dataset() {
    let db = GrafeoDB::new_in_memory();

    // Create 100 nodes with properties
    let mut ids = Vec::new();
    for i in 0..100 {
        let id = db.create_node(&["Item"]).unwrap();
        db.set_node_property(id, "index", Value::Int64(i)).unwrap();
        db.set_node_property(id, "name", Value::String(format!("item_{i}").into()))
            .unwrap();
        ids.push(id);
    }

    // Create 50 edges
    for i in 0..50 {
        db.create_edge(ids[i], ids[i + 50], "LINKS_TO");
    }

    let bytes = db.export_snapshot().unwrap();
    assert!(!bytes.is_empty());

    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 100);
    assert_eq!(restored.edge_count(), 50);

    // Verify a sample property survived
    let session = restored.session();
    let result = session
        .execute("MATCH (i:Item) WHERE i.index = 42 RETURN i.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("item_42".into()));
}

// --- Import empty bytes ---

#[test]
fn import_rejects_empty_bytes() {
    let result = GrafeoDB::import_snapshot(&[]);
    assert!(result.is_err());
}

// --- Snapshot version mismatch ---

#[test]
fn import_rejects_unsupported_version() {
    // Export a valid snapshot, then tamper with the version byte to trigger the
    // "unsupported snapshot version" error path.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let mut bytes = db.export_snapshot().unwrap();
    // The first byte in bincode standard encoding for a struct starting with
    // a u8 field is the version byte itself.
    bytes[0] = 99; // Set to invalid version

    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for tampered snapshot"),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("snapshot")
                    || err_msg.contains("unsupported")
                    || err_msg.contains("import"),
                "Expected snapshot error, got: {err_msg}"
            );
        }
    }
}

// --- Double export produces identical bytes ---

#[test]
fn double_export_is_deterministic() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let bytes1 = db.export_snapshot().unwrap();
    let bytes2 = db.export_snapshot().unwrap();
    assert_eq!(bytes1, bytes2);
}

// --- Edge reference validation ---

#[test]
fn import_rejects_dangling_edge_source() {
    // Create a valid snapshot, then re-export after deleting the source node
    // to create a snapshot with a dangling edge reference.
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Person"]).unwrap();
    let b = db.create_node(&["Person"]).unwrap();
    db.create_edge(a, b, "KNOWS");

    // Export valid snapshot
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    assert_eq!(restored.edge_count(), 1);

    // Now create a corrupted snapshot by exporting nodes only from a db
    // that has 1 node, then manually crafting a snapshot with an edge
    // that references a non-existent source node.
    // The simplest way: delete a node, re-export (edge still exists in store)
    // Actually, let's tamper with binary data at a higher level:
    // Build a snapshot where edge references node ID 999 which doesn't exist.
    let db2 = GrafeoDB::new_in_memory();
    let n = db2.create_node(&["Person"]).unwrap();
    // Create an edge referencing a non-existent source node
    // We need to use the direct store API since create_edge validates at a higher level
    db2.store()
        .create_edge_with_id(
            grafeo_common::types::EdgeId::new(0),
            grafeo_common::types::NodeId::new(999), // doesn't exist
            n,
            "KNOWS",
        )
        .unwrap();

    let bytes = db2.export_snapshot().unwrap();
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for dangling source node"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("non-existent source node"),
                "Expected dangling source error, got: {err}"
            );
        }
    }
}

#[test]
fn import_rejects_dangling_edge_destination() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&["Person"]).unwrap();
    db.store()
        .create_edge_with_id(
            grafeo_common::types::EdgeId::new(0),
            n,
            grafeo_common::types::NodeId::new(999), // doesn't exist
            "KNOWS",
        )
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for dangling destination node"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("non-existent destination node"),
                "Expected dangling destination error, got: {err}"
            );
        }
    }
}

#[test]
fn import_rejects_duplicate_node_ids() {
    let snap = TestSnapshot::new(
        4,
        vec![
            TestNode {
                id: NodeId::new(0),
                labels: vec!["A".into()],
                properties: vec![],
            },
            TestNode {
                id: NodeId::new(0), // duplicate
                labels: vec!["B".into()],
                properties: vec![],
            },
        ],
        vec![],
    );
    let bytes = encode_snapshot(&snap);
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for duplicate node ID"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("duplicate node ID"),
                "Expected duplicate node error, got: {err}"
            );
        }
    }
}

#[test]
fn import_rejects_duplicate_edge_ids() {
    let snap = TestSnapshot::new(
        4,
        vec![
            TestNode {
                id: NodeId::new(0),
                labels: vec![],
                properties: vec![],
            },
            TestNode {
                id: NodeId::new(1),
                labels: vec![],
                properties: vec![],
            },
        ],
        vec![
            TestEdge {
                id: EdgeId::new(0),
                src: NodeId::new(0),
                dst: NodeId::new(1),
                edge_type: "REL".into(),
                properties: vec![],
            },
            TestEdge {
                id: EdgeId::new(0), // duplicate
                src: NodeId::new(0),
                dst: NodeId::new(1),
                edge_type: "REL".into(),
                properties: vec![],
            },
        ],
    );
    let bytes = encode_snapshot(&snap);
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Ok(_) => panic!("Expected error for duplicate edge ID"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("duplicate edge ID"),
                "Expected duplicate edge error, got: {err}"
            );
        }
    }
}

// =========================================================================
// Named Graph Snapshot Tests
// =========================================================================

#[test]
fn export_import_preserves_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("USE GRAPH analytics").unwrap();
    session
        .execute("INSERT (:KPI {name: 'pageviews', count: 42})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    // Default graph
    assert_eq!(restored.node_count(), 1);
    let session2 = restored.session();
    let result = session2.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

    // Named graph
    session2.execute("USE GRAPH analytics").unwrap();
    let result = session2
        .execute("MATCH (m:KPI) RETURN m.name, m.count")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("pageviews".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(42));
}

#[test]
fn export_import_preserves_multiple_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE GRAPH alpha").unwrap();
    session.execute("USE GRAPH alpha").unwrap();
    session.execute("INSERT (:Item {name: 'Widget'})").unwrap();

    session.execute("USE GRAPH default").unwrap();
    session.execute("CREATE GRAPH beta").unwrap();
    session.execute("USE GRAPH beta").unwrap();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    session.execute("INSERT (:City {name: 'Berlin'})").unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session2 = restored.session();

    session2.execute("USE GRAPH alpha").unwrap();
    let result = session2.execute("MATCH (i:Item) RETURN i.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Widget".into()));

    session2.execute("USE GRAPH beta").unwrap();
    let result = session2
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(result.rows()[1][0], Value::String("Berlin".into()));
}

#[test]
fn restore_snapshot_includes_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("CREATE GRAPH metrics").unwrap();
    session.execute("USE GRAPH metrics").unwrap();
    session.execute("INSERT (:KPI {name: 'clicks'})").unwrap();

    let snapshot = db.export_snapshot().unwrap();

    // Modify: add more data
    session.execute("USE GRAPH default").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // Restore
    db.restore_snapshot(&snapshot).unwrap();

    assert_eq!(db.node_count(), 1, "default graph restored to 1 node");

    let session2 = db.session();
    session2.execute("USE GRAPH metrics").unwrap();
    let result = session2.execute("MATCH (m:KPI) RETURN m.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("clicks".into()));
}

#[test]
fn import_v1_snapshot_is_rejected() {
    // V1 snapshots are no longer supported (current version is V4).
    let snap = TestSnapshot::new(
        1,
        vec![TestNode {
            id: NodeId::new(0),
            labels: vec!["Person".into()],
            properties: vec![(
                "name".into(),
                vec![(EpochId::new(0), Value::String("Alix".into()))],
            )],
        }],
        vec![],
    );
    let bytes = encode_snapshot(&snap);

    let result = GrafeoDB::import_snapshot(&bytes);
    assert!(result.is_err(), "V1 snapshots should be rejected");
    let err = result.err().unwrap().to_string();
    assert!(
        err.contains("unsupported snapshot version"),
        "Expected version error, got: {err}"
    );
}

#[test]
fn to_memory_copies_named_graphs() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("CREATE GRAPH backup").unwrap();
    session.execute("USE GRAPH backup").unwrap();
    session
        .execute("INSERT (:Archive {date: '2025-01-01'})")
        .unwrap();

    let copy = db.to_memory().unwrap();

    // Default graph copied
    assert_eq!(copy.node_count(), 1);

    // Named graph copied
    let session2 = copy.session();
    session2.execute("USE GRAPH backup").unwrap();
    let result = session2.execute("MATCH (a:Archive) RETURN a.date").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("2025-01-01".into()));

    // Independence: mutating original doesn't affect copy
    session
        .execute("INSERT (:Archive {date: '2025-02-01'})")
        .unwrap();
    let result2 = session2.execute("MATCH (a:Archive) RETURN a.date").unwrap();
    assert_eq!(result2.rows().len(), 1, "copy should still have 1 node");
}

// =========================================================================
// RDF Snapshot Tests
// =========================================================================

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod rdf_snapshots {
    use grafeo_common::types::Value;
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    #[test]
    fn export_import_preserves_rdf_triples() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();

        let bytes = db.export_snapshot().unwrap();
        let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

        let session2 = restored.session();
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name } ORDER BY ?name")
            .unwrap();
        assert_eq!(result.rows().len(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn export_import_preserves_rdf_named_graphs() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }
                }"#,
            )
            .unwrap();

        let bytes = db.export_snapshot().unwrap();
        let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

        let session2 = restored.session();

        // Default graph
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

        // Named graph
        let result = session2
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    }

    #[test]
    fn restore_snapshot_includes_rdf_data() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                }"#,
            )
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();

        // Add more data
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();

        // Restore: should go back to just Alix
        db.restore_snapshot(&snapshot).unwrap();

        let session2 = db.session();
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap();
        assert_eq!(result.rows().len(), 1, "restore should revert to snapshot");
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn to_memory_copies_rdf_data() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    GRAPH <http://ex.org/g1> {
                        <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    }
                }"#,
            )
            .unwrap();

        let copy = db.to_memory().unwrap();

        let session2 = copy.session();
        let result = session2
            .execute_sparql("SELECT ?name WHERE { ?s <http://ex.org/name> ?name }")
            .unwrap();
        assert_eq!(result.rows().len(), 1, "default RDF graph copied");
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

        let result = session2
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    GRAPH <http://ex.org/g1> { ?s <http://ex.org/name> ?name }
                }"#,
            )
            .unwrap();
        assert_eq!(result.rows().len(), 1, "named RDF graph copied");
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));

        // Independence: mutating original doesn't affect copy
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/mia> <http://ex.org/name> "Mia" .
                }"#,
            )
            .unwrap();
        let result = session2
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap();
        assert_eq!(result.rows().len(), 1, "copy should be independent");
    }

    #[test]
    fn export_import_preserves_typed_rdf_literals() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
                    <http://ex.org/alix> <http://ex.org/greeting> "Bonjour"@fr .
                }"#,
            )
            .unwrap();

        let bytes = db.export_snapshot().unwrap();
        let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

        let session2 = restored.session();
        let result = session2
            .execute_sparql("SELECT ?o WHERE { <http://ex.org/alix> ?p ?o } ORDER BY ?o")
            .unwrap();
        assert_eq!(
            result.rows().len(),
            2,
            "typed and lang literals should survive"
        );
    }
}

// ---------------------------------------------------------------------------
// Snapshot forward-compatibility (T3-04)
// ---------------------------------------------------------------------------

#[test]
fn import_unknown_snapshot_version_returns_clear_error() {
    // Craft a snapshot with version 99 (unknown future version)
    let snap = TestSnapshot::new(99, vec![], vec![]);
    let bytes = encode_snapshot(&snap);
    let result = GrafeoDB::import_snapshot(&bytes);
    match result {
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("version") || err.contains("unsupported") || err.contains("99"),
                "error should mention version issue, got: {err}"
            );
        }
        Ok(_) => panic!("importing an unknown snapshot version should error"),
    }
}

#[test]
fn import_truncated_snapshot_returns_error() {
    let db = GrafeoDB::new_in_memory();
    db.create_node(&["Test"]).unwrap();
    let bytes = db.export_snapshot().unwrap();

    // Truncate to half
    let truncated = &bytes[..bytes.len() / 2];
    let result = GrafeoDB::import_snapshot(truncated);
    assert!(
        result.is_err(),
        "importing a truncated snapshot should error"
    );
}

#[test]
fn import_empty_bytes_returns_error() {
    let result = GrafeoDB::import_snapshot(&[]);
    assert!(result.is_err(), "importing empty bytes should error");
}

// --- Multi-schema round-trip (ISO/IEC 39075 catalog hierarchy) ---

#[test]
fn snapshot_round_trip_multi_schema_isolation() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();

    s.execute("CREATE SCHEMA reporting").unwrap();
    s.execute("CREATE SCHEMA social").unwrap();

    s.execute("SESSION SET SCHEMA reporting").unwrap();
    s.execute("INSERT (:Report {title: 'Q1'})").unwrap();

    s.execute("SESSION SET SCHEMA social").unwrap();
    s.execute("INSERT (:Friend {name: 'Alix'})").unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    let r = restored.session();

    // Schemas survive
    let schemas = r.execute("SHOW SCHEMAS").unwrap();
    assert_eq!(schemas.row_count(), 2, "both schemas should be restored");

    // Cross-schema isolation is preserved
    r.execute("SESSION SET SCHEMA reporting").unwrap();
    let reports = r.execute("MATCH (n:Report) RETURN n").unwrap();
    assert_eq!(reports.row_count(), 1, "reporting should have 1 Report");
    let leaked = r.execute("MATCH (n:Friend) RETURN n").unwrap();
    assert_eq!(
        leaked.row_count(),
        0,
        "social data must not leak into reporting after restore"
    );

    r.execute("SESSION SET SCHEMA social").unwrap();
    let friends = r.execute("MATCH (n:Friend) RETURN n").unwrap();
    assert_eq!(friends.row_count(), 1, "social should have 1 Friend");
    let leaked = r.execute("MATCH (n:Report) RETURN n").unwrap();
    assert_eq!(
        leaked.row_count(),
        0,
        "reporting data must not leak into social after restore"
    );
}

#[test]
fn snapshot_round_trip_named_graph_within_schema() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();

    s.execute("CREATE SCHEMA reports").unwrap();
    s.execute("SESSION SET SCHEMA reports").unwrap();
    s.execute("CREATE GRAPH quarterly").unwrap();
    s.execute("SESSION SET GRAPH quarterly").unwrap();
    s.execute("INSERT (:Row {q: 1})").unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    let r = restored.session();

    r.execute("SESSION SET SCHEMA reports").unwrap();
    let graphs = r.execute("SHOW GRAPHS").unwrap();
    assert_eq!(
        graphs.row_count(),
        1,
        "schema-scoped named graph should survive round-trip"
    );

    r.execute("SESSION SET GRAPH quarterly").unwrap();
    let rows = r.execute("MATCH (n:Row) RETURN n.q").unwrap();
    assert_eq!(
        rows.row_count(),
        1,
        "data in schema-scoped named graph should survive round-trip"
    );
}
