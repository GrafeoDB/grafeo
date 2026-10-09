//! WASM binding integration tests.
//!
//! Run via: `wasm-pack test crates/bindings/wasm --node`

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

use grafeo_wasm::Database;

#[wasm_bindgen_test]
fn test_database_creation() {
    let db = Database::new().expect("should create in-memory database");
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
}

#[wasm_bindgen_test]
fn test_insert_and_query() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:Person {name: 'Alix', age: 30})")
        .expect("create node");
    assert_eq!(db.node_count(), 1);

    let result = db
        .execute("MATCH (n:Person) RETURN n.name, n.age")
        .expect("query");

    // Result is a JsValue (array of objects)
    assert!(!result.is_null());
    assert!(!result.is_undefined());
}

#[wasm_bindgen_test]
fn test_execute_raw_structure() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:N {x: 1}), (:N {x: 2})")
        .expect("create");

    let raw = db.execute_raw("MATCH (n:N) RETURN n.x").expect("raw query");

    // execute_raw returns { columns, rows, executionTimeMs }
    assert!(!raw.is_null());
    assert!(!raw.is_undefined());
}

#[wasm_bindgen_test]
fn test_execute_with_language_gql() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:Person {name: 'Alix'})")
        .expect("create");

    let result = db
        .execute_with_language("MATCH (n:Person) RETURN n.name", "gql")
        .expect("gql query");

    assert!(!result.is_null());
}

#[wasm_bindgen_test]
fn test_execute_with_unknown_language_error() {
    let db = Database::new().expect("create db");

    let result = db.execute_with_language("SELECT 1", "unknown_lang");
    assert!(result.is_err(), "unknown language should error");
}

#[wasm_bindgen_test]
fn test_snapshot_roundtrip() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:Person {name: 'Alix'})")
        .expect("create");
    assert_eq!(db.node_count(), 1);

    // Export snapshot
    let snapshot = db.export_snapshot().expect("export");
    assert!(!snapshot.is_empty());

    // Import into new database
    let restored = Database::import_snapshot(&snapshot).expect("import");
    assert_eq!(restored.node_count(), 1);
}

#[wasm_bindgen_test]
fn test_schema() {
    let db = Database::new().expect("create db");

    db.execute("CREATE (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .expect("create");

    let schema = db.schema().expect("schema");
    assert!(!schema.is_null());
    assert!(!schema.is_undefined());
}

#[wasm_bindgen_test]
fn test_version() {
    let v = Database::version();
    assert!(!v.is_empty());
}

// ---------------------------------------------------------------------------
// importLpg tests
// ---------------------------------------------------------------------------

fn lpg_result_counts(result: &wasm_bindgen::JsValue) -> (u32, u32) {
    let nodes = js_sys::Reflect::get(result, &wasm_bindgen::JsValue::from_str("nodes"))
        .unwrap()
        .as_f64()
        .unwrap() as u32;
    let edges = js_sys::Reflect::get(result, &wasm_bindgen::JsValue::from_str("edges"))
        .unwrap()
        .as_f64()
        .unwrap() as u32;
    (nodes, edges)
}

#[wasm_bindgen_test]
fn test_import_lpg_basic() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["Person"], "properties": { "name": "Alix", "age": 30 } },
            { "labels": ["Person"], "properties": { "name": "Gus", "age": 25 } }
        ],
        "edges": [
            { "source": 0, "target": 1, "type": "KNOWS", "properties": { "since": 2020 } }
        ]
    }))
    .unwrap();

    let result = db.import_lpg(data).expect("import");
    let (nodes, edges) = lpg_result_counts(&result);
    assert_eq!(nodes, 2);
    assert_eq!(edges, 1);
    assert_eq!(db.node_count(), 2);
    assert_eq!(db.edge_count(), 1);
}

#[wasm_bindgen_test]
fn test_import_lpg_nodes_only() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["Tag"], "properties": { "name": "rust" } },
            { "labels": ["Tag"], "properties": { "name": "wasm" } },
            { "labels": ["Tag"], "properties": { "name": "graph" } }
        ]
    }))
    .unwrap();

    let result = db.import_lpg(data).expect("import");
    let (nodes, edges) = lpg_result_counts(&result);
    assert_eq!(nodes, 3);
    assert_eq!(edges, 0);
    assert_eq!(db.node_count(), 3);
}

#[wasm_bindgen_test]
fn test_import_lpg_empty() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": []
    }))
    .unwrap();

    let result = db.import_lpg(data).expect("import");
    let (nodes, edges) = lpg_result_counts(&result);
    assert_eq!(nodes, 0);
    assert_eq!(edges, 0);
}

#[wasm_bindgen_test]
fn test_import_lpg_no_properties() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["A"] },
            { "labels": ["B"] }
        ],
        "edges": [
            { "source": 0, "target": 1, "type": "LINKED" }
        ]
    }))
    .unwrap();

    let result = db.import_lpg(data).expect("import");
    let (nodes, edges) = lpg_result_counts(&result);
    assert_eq!(nodes, 2);
    assert_eq!(edges, 1);
}

#[wasm_bindgen_test]
fn test_import_lpg_multiple_labels() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["Person", "Employee", "Developer"], "properties": { "name": "Alix" } }
        ]
    }))
    .unwrap();

    let result = db.import_lpg(data).expect("import");
    let (nodes, _) = lpg_result_counts(&result);
    assert_eq!(nodes, 1);
    assert_eq!(db.node_count(), 1);
}

#[wasm_bindgen_test]
fn test_import_lpg_self_loop() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [{ "labels": ["Node"] }],
        "edges": [{ "source": 0, "target": 0, "type": "SELF" }]
    }))
    .unwrap();

    let result = db.import_lpg(data).expect("import");
    let (nodes, edges) = lpg_result_counts(&result);
    assert_eq!(nodes, 1);
    assert_eq!(edges, 1);
}

#[wasm_bindgen_test]
fn test_import_lpg_multiple_edges_same_pair() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["Person"], "properties": { "name": "Alix" } },
            { "labels": ["Person"], "properties": { "name": "Gus" } }
        ],
        "edges": [
            { "source": 0, "target": 1, "type": "KNOWS" },
            { "source": 0, "target": 1, "type": "WORKS_WITH" },
            { "source": 1, "target": 0, "type": "KNOWS" }
        ]
    }))
    .unwrap();

    let result = db.import_lpg(data).expect("import");
    let (_, edges) = lpg_result_counts(&result);
    assert_eq!(edges, 3);
    assert_eq!(db.edge_count(), 3);
}

#[wasm_bindgen_test]
fn test_import_lpg_edge_source_out_of_bounds() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [{ "labels": ["A"] }],
        "edges": [{ "source": 5, "target": 0, "type": "BAD" }]
    }))
    .unwrap();

    let err = db.import_lpg(data).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("source index 5 out of bounds"),
        "unexpected error: {msg}"
    );
}

#[wasm_bindgen_test]
fn test_import_lpg_edge_target_out_of_bounds() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [{ "labels": ["A"] }],
        "edges": [{ "source": 0, "target": 99, "type": "BAD" }]
    }))
    .unwrap();

    let err = db.import_lpg(data).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("target index 99 out of bounds"),
        "unexpected error: {msg}"
    );
}

#[wasm_bindgen_test]
fn test_import_lpg_invalid_shape() {
    let db = Database::new().expect("create db");

    // Missing required 'nodes' field
    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "edges": []
    }))
    .unwrap();

    let err = db.import_lpg(data).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("Invalid LPG data"), "unexpected error: {msg}");
}

#[wasm_bindgen_test]
fn test_import_lpg_queryable_after_import() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["City"], "properties": { "name": "Amsterdam", "population": 905234 } },
            { "labels": ["City"], "properties": { "name": "Berlin", "population": 3748148 } }
        ],
        "edges": [
            { "source": 0, "target": 1, "type": "CONNECTED_TO", "properties": { "distance_km": 577 } }
        ]
    }))
    .unwrap();

    db.import_lpg(data).expect("import");

    // Verify nodes are queryable
    let result = db
        .execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .expect("query cities");
    assert!(!result.is_null());

    // Verify edges are queryable
    let result = db
        .execute("MATCH (:City)-[e:CONNECTED_TO]->(:City) RETURN e.distance_km")
        .expect("query edges");
    assert!(!result.is_null());
}

#[wasm_bindgen_test]
fn test_import_lpg_mixed_property_types() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [{
            "labels": ["Thing"],
            "properties": {
                "str_val": "hello",
                "int_val": 42,
                "float_val": 3.14,
                "bool_val": true,
                "null_val": null,
                "list_val": [1, 2, 3]
            }
        }]
    }))
    .unwrap();

    let result = db.import_lpg(data).expect("import");
    let (nodes, _) = lpg_result_counts(&result);
    assert_eq!(nodes, 1);
}

#[wasm_bindgen_test]
fn test_import_lpg_incremental() {
    let db = Database::new().expect("create db");

    // First batch
    let data1 = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["Person"], "properties": { "name": "Alix" } }
        ]
    }))
    .unwrap();
    db.import_lpg(data1).expect("import 1");
    assert_eq!(db.node_count(), 1);

    // Second batch
    let data2 = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["Person"], "properties": { "name": "Gus" } }
        ]
    }))
    .unwrap();
    db.import_lpg(data2).expect("import 2");
    assert_eq!(db.node_count(), 2);
}

#[wasm_bindgen_test]
fn test_import_lpg_snapshot_roundtrip() {
    let db = Database::new().expect("create db");

    let data = serde_wasm_bindgen::to_value(&serde_json::json!({
        "nodes": [
            { "labels": ["Person"], "properties": { "name": "Alix" } },
            { "labels": ["Person"], "properties": { "name": "Gus" } }
        ],
        "edges": [
            { "source": 0, "target": 1, "type": "KNOWS" }
        ]
    }))
    .unwrap();

    db.import_lpg(data).expect("import");

    // Export and re-import
    let snapshot = db.export_snapshot().expect("export");
    let restored = Database::import_snapshot(&snapshot).expect("restore");

    assert_eq!(restored.node_count(), 2);
    assert_eq!(restored.edge_count(), 1);
}

// ==========================================================================
// L25: transactions + close()
// ==========================================================================

#[wasm_bindgen_test]
fn test_begin_commit_transaction_persists_writes() {
    let db = Database::new().expect("create db");
    db.begin_transaction().expect("begin");
    assert!(db.is_transaction_active());

    db.execute("CREATE (:T {x: 1})").expect("insert in tx");
    assert_eq!(
        db.node_count(),
        0,
        "commit not yet called: count should be 0"
    );

    db.commit_transaction().expect("commit");
    assert!(!db.is_transaction_active());
    assert_eq!(db.node_count(), 1, "writes visible after commit");
}

#[wasm_bindgen_test]
fn test_rollback_transaction_discards_writes() {
    let db = Database::new().expect("create db");
    db.begin_transaction().expect("begin");
    db.execute("CREATE (:T {x: 1})").expect("insert in tx");

    db.rollback_transaction().expect("rollback");
    assert!(!db.is_transaction_active());
    assert_eq!(db.node_count(), 0, "rollback discards writes");
}

#[wasm_bindgen_test]
fn test_double_begin_errors() {
    let db = Database::new().expect("create db");
    db.begin_transaction().expect("first begin");
    assert!(db.begin_transaction().is_err(), "second begin should error");
    db.rollback_transaction().expect("cleanup rollback");
}

#[wasm_bindgen_test]
fn test_commit_without_active_errors() {
    let db = Database::new().expect("create db");
    assert!(db.commit_transaction().is_err());
    assert!(db.rollback_transaction().is_err());
}

#[wasm_bindgen_test]
fn test_close_blocks_subsequent_operations() {
    let db = Database::new().expect("create db");
    db.execute("CREATE (:T)").expect("insert");
    db.close();

    assert!(db.execute("CREATE (:T)").is_err(), "execute after close");
    assert!(db.export_snapshot().is_err(), "export after close");
    assert!(db.begin_transaction().is_err(), "begin after close");
}

/// The first cell of an `executeRaw*` result (`rows[0][0]`) as a number.
fn first_raw_number(raw: &wasm_bindgen::JsValue) -> f64 {
    let rows = js_sys::Reflect::get(raw, &"rows".into()).expect("rows");
    let row = js_sys::Array::from(&rows).get(0);
    js_sys::Array::from(&row)
        .get(0)
        .as_f64()
        .expect("a number in rows[0][0]")
}

/// `executeRawWithLanguage` refuses a closed database, like `executeRaw`.
#[wasm_bindgen_test]
fn test_execute_raw_with_language_after_close_errors() {
    let db = Database::new().expect("create db");
    db.close();
    assert!(
        db.execute_raw_with_language("MATCH (n) RETURN count(n) AS c", "gql")
            .is_err(),
        "executeRawWithLanguage after close"
    );
}

/// `executeRawWithLanguage` runs in the open transaction: it sees the
/// transaction's own writes before the commit.
#[wasm_bindgen_test]
fn test_execute_raw_with_language_runs_in_the_open_transaction() {
    let db = Database::new().expect("create db");
    db.begin_transaction().expect("begin");
    db.execute("CREATE (:T {x: 1})").expect("insert in tx");
    let raw = db
        .execute_raw_with_language("MATCH (n:T) RETURN count(n) AS c", "gql")
        .expect("raw query in tx");
    assert_eq!(first_raw_number(&raw), 1.0, "the transaction's own write");
    db.rollback_transaction().expect("rollback");
}

#[wasm_bindgen_test]
fn test_close_rolls_back_active_transaction() {
    let db = Database::new().expect("create db");
    db.begin_transaction().expect("begin");
    db.execute("CREATE (:T)").expect("insert in tx");
    db.close();
    // Second close is a no-op.
    db.close();
}

// ==========================================================================
// M7: signed-snapshot integrity
// ==========================================================================

#[wasm_bindgen_test]
fn test_signed_snapshot_roundtrip() {
    let db = Database::new().expect("create db");
    db.execute("CREATE (:Doc {title: 'A'})").expect("insert");
    db.execute("CREATE (:Doc {title: 'B'})").expect("insert");

    let key = b"0123456789abcdef0123456789abcdef";
    let signed = db.export_snapshot_signed(key).expect("export signed");

    // Signed format is distinguishable from raw-unsigned.
    assert_eq!(&signed[..4], b"GSN1");

    let restored = Database::import_snapshot_signed(&signed, key).expect("restore");
    assert_eq!(restored.node_count(), 2);
}

#[wasm_bindgen_test]
fn test_wasm_import_tampered_snapshot() {
    let db = Database::new().expect("create db");
    db.execute("CREATE (:T)").expect("insert");
    let key = b"secret-key";
    let mut signed = db.export_snapshot_signed(key).expect("export signed");

    // Flip a byte in the middle of the payload.
    let mid = signed.len() / 2;
    signed[mid] ^= 0x01;

    assert!(Database::import_snapshot_signed(&signed, key).is_err());
}

#[wasm_bindgen_test]
fn test_wasm_import_wrong_key_rejected() {
    let db = Database::new().expect("create db");
    db.execute("CREATE (:T)").expect("insert");
    let signed = db.export_snapshot_signed(b"key-a").expect("export signed");

    assert!(
        Database::import_snapshot_signed(&signed, b"key-b").is_err(),
        "wrong key must not verify"
    );
}

#[wasm_bindgen_test]
fn test_wasm_import_empty_snapshot() {
    assert!(Database::import_snapshot(b"").is_err());
    assert!(Database::import_snapshot_signed(b"", b"k").is_err());
}

#[wasm_bindgen_test]
fn test_signed_snapshot_rejected_by_unsigned_import() {
    let db = Database::new().expect("create db");
    db.execute("CREATE (:T)").expect("insert");
    let signed = db.export_snapshot_signed(b"k").expect("export signed");

    assert!(Database::import_snapshot(&signed).is_err());
}

#[wasm_bindgen_test]
fn test_export_signed_requires_key() {
    let db = Database::new().expect("create db");
    assert!(db.export_snapshot_signed(b"").is_err());
}

/// The message of an error a database method returned.
fn error_message(error: wasm_bindgen::JsError) -> String {
    js_sys::Error::from(wasm_bindgen::JsValue::from(error))
        .message()
        .into()
}

/// The statements of #573: a stack overflow used to trap the module, and
/// every later call (`new Database()` too) trapped after it. A long INSERT
/// and a long OR chain run; a statement nested beyond the limits fails with
/// an error that names the limit, and the module keeps working.
#[wasm_bindgen_test]
fn test_very_large_statements_run_or_fail_with_the_limit_and_the_module_keeps_working() {
    let db = Database::new().expect("create db");
    let insert = format!(
        "INSERT {}",
        (0..10_000)
            .map(|i| format!("(:Person {{v: {i}}})"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    db.execute(&insert)
        .expect("an INSERT of 10,000 patterns runs");
    assert_eq!(db.node_count(), 10_000);
    let or_chain = format!(
        "MATCH (n:Person) WHERE {} RETURN n.v",
        (0..10_000)
            .map(|i| format!("n.v = {}", i * 3))
            .collect::<Vec<_>>()
            .join(" OR ")
    );
    let rows = db.execute(&or_chain).expect("a 10,000-term OR runs");
    assert_eq!(
        js_sys::Array::from(&rows).length(),
        3_334,
        "v = 0, 3, ..., 9999"
    );

    let parentheses = format!("RETURN {}1{} AS v", "(".repeat(10_000), ")".repeat(10_000));
    let message = error_message(
        db.execute(&parentheses)
            .expect_err("beyond the nesting limit"),
    );
    assert!(message.contains("nesting depth of 64"), "{message}");

    db.execute("INSERT (:Person {name: 'Alix'})")
        .expect("the database still works");
    assert!(Database::new().is_ok(), "the module still works");
}

/// `key` of a JS object.
fn field(value: &wasm_bindgen::JsValue, key: &str) -> wasm_bindgen::JsValue {
    js_sys::Reflect::get(value, &key.into()).expect(key)
}

/// A node or edge in a list literal, a returned path and `startNode` give
/// node and edge objects (`_id`, `_labels` or `_type`, the properties), as
/// `RETURN n` does; they used to give bare IDs. A path keeps its object
/// shape, `{ nodes, edges, _type: "path" }`.
#[wasm_bindgen_test]
fn test_lists_maps_and_paths_hold_nodes_and_edges() {
    let db = Database::new().expect("create db");
    db.execute(
        "INSERT (:Person {name: 'Alix', age: 19})-[:KNOWS {w: 3}]->(:Person {name: 'Gus', age: 88})",
    )
    .expect("insert");
    let rows = js_sys::Array::from(
        &db.execute(
            "MATCH p = (:Person {name: 'Alix'})-[r:KNOWS]->(b) \
             RETURN p, [b, 3] AS l, {k: b} AS m, startNode(r) AS s",
        )
        .expect("query"),
    );
    let row = rows.get(0);
    let text = |value: wasm_bindgen::JsValue| value.as_string().expect("a string");

    let path = field(&row, "p");
    assert_eq!(text(field(&path, "_type")), "path");
    let nodes = js_sys::Array::from(&field(&path, "nodes"));
    let names: Vec<String> = nodes
        .iter()
        .map(|node| text(field(&node, "name")))
        .collect();
    assert_eq!(names, ["Alix", "Gus"]);
    let labels = js_sys::Array::from(&field(&nodes.get(0), "_labels"));
    assert_eq!(text(labels.get(0)), "Person");
    let knows = js_sys::Array::from(&field(&path, "edges")).get(0);
    assert_eq!(text(field(&knows, "_type")), "KNOWS");
    assert_eq!(field(&knows, "w").as_f64(), Some(3.0));

    let list = js_sys::Array::from(&field(&row, "l"));
    assert_eq!(text(field(&list.get(0), "name")), "Gus");
    assert_eq!(field(&list.get(0), "age").as_f64(), Some(88.0));
    assert_eq!(list.get(1).as_f64(), Some(3.0));
    assert_eq!(text(field(&field(&field(&row, "m"), "k"), "name")), "Gus");
    assert_eq!(text(field(&field(&row, "s"), "name")), "Alix");
}
