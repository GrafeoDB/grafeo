//! WASM binding integration tests.
//!
//! Run via: `wasm-pack test crates/bindings/wasm --node`

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

use grafeo_wasm::Database;

/// A JS value as JavaScript code writes it: JSON objects become plain objects
/// (`serde_wasm_bindgen::to_value` makes them `Map`s, which no caller passes).
fn to_js<T: serde::Serialize>(
    value: &T,
) -> Result<wasm_bindgen::JsValue, serde_wasm_bindgen::Error> {
    value.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
}

#[wasm_bindgen_test]
fn test_database_creation() {
    let db = Database::new().expect("should create in-memory database");
    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
}

/// A decimal parameter `{ "$decimal": "0.10" }` comes back from arithmetic as
/// a tagged object with its scale.
#[wasm_bindgen_test]
fn test_decimals_round_trip_as_tagged_objects() {
    let db = Database::new().expect("create db");
    let params = to_js(&serde_json::json!({
        "a": { "$decimal": "0.10" },
        "b": { "$decimal": "0.2" },
    }))
    .unwrap();
    let rows = db
        .execute_with_params("RETURN $a + $b AS total", params)
        .expect("query");
    let rows: serde_json::Value = serde_wasm_bindgen::from_value(rows).unwrap();
    assert_eq!(
        rows,
        serde_json::json!([{ "total": { "$decimal": "0.30" } }])
    );
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
    assert!(!snapshot.is_empty(), "a snapshot holds the node");

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
    assert!(!v.is_empty(), "the version is set");
}

// ---------------------------------------------------------------------------
// importLpg tests
// ---------------------------------------------------------------------------

fn lpg_result_counts(result: &wasm_bindgen::JsValue) -> (u32, u32) {
    let count = |key: &str| -> u32 {
        let value = js_sys::Reflect::get(result, &wasm_bindgen::JsValue::from_str(key)).unwrap();
        serde_wasm_bindgen::from_value(value).expect("a count is a whole number")
    };
    (count("nodes"), count("edges"))
}

#[wasm_bindgen_test]
fn test_import_lpg_basic() {
    let db = Database::new().expect("create db");

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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
    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
        "nodes": [{
            "labels": ["Thing"],
            "properties": {
                "str_val": "hello",
                "int_val": 42,
                "float_val": 2.5,
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
    let data1 = to_js(&serde_json::json!({
        "nodes": [
            { "labels": ["Person"], "properties": { "name": "Alix" } }
        ]
    }))
    .unwrap();
    db.import_lpg(data1).expect("import 1");
    assert_eq!(db.node_count(), 1);

    // Second batch
    let data2 = to_js(&serde_json::json!({
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

    let data = to_js(&serde_json::json!({
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
fn error_message(error: impl Into<wasm_bindgen::JsValue>) -> String {
    js_sys::Error::from(error.into()).message().into()
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

/// `hybridSearch` keeps only the nodes its `filters` option matches, and
/// still returns up to `k` of them (#397). Without options it searches as
/// before.
#[cfg(feature = "hybrid-search")]
#[wasm_bindgen_test]
fn hybrid_search_keeps_the_nodes_its_filters_match() {
    let db = Database::new().expect("create db");
    db.execute(
        "INSERT (:Doc {owner: 'Alix', city: 'Amsterdam', text: 'Alix rides along canals, canals and canals'}), \
         (:Doc {owner: 'Vincent', city: 'Berlin', text: 'Vincent paints canals'}), \
         (:Doc {owner: 'Jules', city: 'Berlin', text: 'Jules swims past the old canals of Berlin at dawn'}), \
         (:Doc {owner: 'Mia', city: 'Berlin', text: 'Mia dances in Berlin clubs'})",
    )
    .expect("insert");
    db.create_text_index("Doc", "text", None)
        .expect("text index");

    let ids = |results: &wasm_bindgen::JsValue| -> Vec<f64> {
        js_sys::Array::from(results)
            .iter()
            .map(|row| field(&row, "id").as_f64().expect("a numeric id"))
            .collect()
    };
    let id_of = |owner: &str| -> f64 {
        let rows = db
            .execute(&format!(
                "MATCH (d:Doc {{owner: '{owner}'}}) RETURN id(d) AS id"
            ))
            .expect("query");
        ids(&rows)[0]
    };

    let unfiltered = db
        .hybrid_search("Doc", "text", "emb", "canals", 1, None)
        .expect("hybrid search without options");
    assert_eq!(ids(&unfiltered), [id_of("Alix")], "Alix says canals most");

    let options = js_sys::JSON::parse(r#"{"filters": {"city": "Berlin"}}"#).expect("options");
    let berlin = db
        .hybrid_search("Doc", "text", "emb", "canals", 1, Some(options))
        .expect("hybrid search with filters");
    assert_eq!(ids(&berlin), [id_of("Vincent")], "the best match of Berlin");

    let invalid = js_sys::JSON::parse(r#"{"filters": 3}"#).expect("options");
    assert!(
        db.hybrid_search("Doc", "text", "emb", "canals", 1, Some(invalid))
            .is_err(),
        "filters that are not an object are refused"
    );
}

/// `createTextIndex` takes the BM25 parameters, the tokenizer and the stop
/// words, and `textSearch` property filters (#351, #397).
#[cfg(feature = "text-index")]
#[wasm_bindgen_test]
fn text_index_options_and_text_search_filters() {
    let db = Database::new().expect("create db");
    db.execute(
        "INSERT (:Note {owner: 'Alix', city: 'Berlin', body: '阿利克斯住在柏林'}), \
         (:Note {owner: 'Gus', city: 'Amsterdam', body: '古斯住在阿姆斯特丹'})",
    )
    .expect("insert");
    let ids = |results: &wasm_bindgen::JsValue| -> Vec<f64> {
        js_sys::Array::from(results)
            .iter()
            .map(|row| field(&row, "id").as_f64().expect("a numeric id"))
            .collect()
    };
    let id_of = |owner: &str| -> f64 {
        let rows = db
            .execute(&format!(
                "MATCH (n:Note {{owner: '{owner}'}}) RETURN id(n) AS id"
            ))
            .expect("query");
        ids(&rows)[0]
    };
    let options = js_sys::JSON::parse(r#"{"tokenizer": "cjk_bigram", "stopWords": ["阿姆"]}"#)
        .expect("options");
    db.create_text_index("Note", "body", Some(options))
        .expect("text index with options");

    let berlin = db
        .text_search("Note", "body", "柏林", 10, None)
        .expect("text search");
    assert_eq!(ids(&berlin), [id_of("Alix")], "a word inside a sentence");
    assert_eq!(
        ids(&db.text_search("Note", "body", "阿姆", 10, None).unwrap()),
        Vec::<f64>::new(),
        "a stop word"
    );
    let filters = js_sys::JSON::parse(r#"{"filters": {"city": "Amsterdam"}}"#).expect("filters");
    assert_eq!(
        ids(&db
            .text_search("Note", "body", "住在", 10, Some(filters))
            .unwrap()),
        [id_of("Gus")],
        "only the notes of Amsterdam"
    );

    for invalid in [r#"{"k1": -3}"#, r#"{"b": 19}"#, r#"{"tokenizer": "jieba"}"#] {
        let options = js_sys::JSON::parse(invalid).expect("options");
        assert!(
            db.create_text_index("Note", "body", Some(options)).is_err(),
            "{invalid} is refused"
        );
    }
}

// ---------------------------------------------------------------------------
// Batches, upserts, write counters and graph handles
// ---------------------------------------------------------------------------

#[cfg(any(feature = "direct-api", feature = "vector-index"))]
fn js(value: serde_json::Value) -> wasm_bindgen::JsValue {
    to_js(&value).unwrap()
}

fn json(value: &wasm_bindgen::JsValue) -> serde_json::Value {
    serde_wasm_bindgen::from_value(value.clone()).unwrap()
}

#[cfg(feature = "direct-api")]
fn error_text(result: Result<wasm_bindgen::JsValue, wasm_bindgen::JsError>) -> String {
    let error = wasm_bindgen::JsValue::from(result.expect_err("expected an error"));
    js_sys::Error::from(error).message().into()
}

#[cfg(feature = "direct-api")]
#[wasm_bindgen_test]
fn test_batch_create_nodes_and_edges() {
    let db = Database::new().unwrap();
    let ids = db
        .batch_create_nodes_with_props(
            js(serde_json::json!(["Person", "Employee"])),
            js(serde_json::json!([{ "name": "Alix" }, { "name": "Gus" }])),
        )
        .unwrap();
    let ids: Vec<u64> = serde_wasm_bindgen::from_value(ids).unwrap();
    assert_eq!(ids.len(), 2);
    let edge_ids = db
        .batch_create_edges(js(serde_json::json!([
            { "src": ids[0], "dst": ids[1], "type": "KNOWS", "properties": { "since": 2020 } },
            { "src": ids[1], "dst": ids[0], "type": "KNOWS" }
        ])))
        .unwrap();
    assert_eq!(json(&edge_ids).as_array().unwrap().len(), 2);
    let rows = db
        .execute(
            "MATCH (a:Employee {name: 'Alix'})-[k:KNOWS]->(b) RETURN b.name AS b, k.since AS since",
        )
        .unwrap();
    assert_eq!(
        json(&rows),
        serde_json::json!([{ "b": "Gus", "since": 2020 }])
    );

    let missing = db.batch_create_edges(js(serde_json::json!([
        { "src": ids[0], "dst": 88888, "type": "KNOWS" }
    ])));
    assert!(missing.is_err());
    assert_eq!(db.edge_count(), 2);
    let fractional = db.batch_create_edges(js(serde_json::json!([
        { "src": 0.5, "dst": 1, "type": "KNOWS" }
    ])));
    assert!(fractional.is_err());
}

#[cfg(feature = "direct-api")]
#[wasm_bindgen_test]
fn test_batches_refuse_an_open_transaction_and_upserts_join_it() {
    let db = Database::new().unwrap();
    db.begin_transaction().unwrap();
    let message = error_text(db.batch_create_nodes_with_props(
        js(serde_json::json!("Person")),
        js(serde_json::json!([{ "name": "Vincent" }])),
    ));
    assert!(message.contains("commit or roll back"), "{message}");

    let summary = db
        .upsert_nodes(
            js(serde_json::json!(["Person"])),
            js(serde_json::json!([{ "id": "v", "name": "Vincent" }])),
            wasm_bindgen::JsValue::UNDEFINED,
        )
        .unwrap();
    assert_eq!(json(&summary)["created"], 1);
    db.rollback_transaction().unwrap();
    assert_eq!(db.node_count(), 0);
}

#[cfg(feature = "direct-api")]
#[wasm_bindgen_test]
fn test_upserts_create_update_and_skip() {
    let db = Database::new().unwrap();
    let created = db
        .upsert_nodes(
            js(serde_json::json!(["File"])),
            js(serde_json::json!([{ "path": "a.rs", "size": 3 }, { "path": "b.rs" }, { "size": 1 }])),
            js(serde_json::json!({ "key": "path" })),
        )
        .unwrap();
    assert_eq!(
        json(&created),
        serde_json::json!({ "created": 2, "updated": 0, "skipped": 1, "skippedRows": [2] })
    );
    let updated = db
        .upsert_nodes(
            js(serde_json::json!("File")),
            js(serde_json::json!([{ "path": "a.rs", "size": 19 }])),
            js(serde_json::json!({ "key": "path" })),
        )
        .unwrap();
    assert_eq!(json(&updated)["updated"], 1);
    assert_eq!(db.node_count(), 2);

    let edges = db
        .upsert_edges(
            "IMPORTS",
            js(serde_json::json!([
                { "id": "a-b", "from": "a.rs", "to": "b.rs" },
                { "id": "a-x", "from": "a.rs", "to": "x.rs" }
            ])),
            js(serde_json::json!({
                "endpointKey": "path", "endpointLabels": ["File"], "srcField": "from", "dstField": "to"
            })),
        )
        .unwrap();
    assert_eq!(json(&edges)["created"], 1);
    assert_eq!(json(&edges)["skippedRows"], serde_json::json!([1]));
    assert_eq!(db.edge_count(), 1);
}

#[wasm_bindgen_test]
fn test_raw_results_count_writes() {
    let db = Database::new().unwrap();
    let write = db
        .execute_raw("INSERT (:Person {name: 'Mia'})-[:KNOWS]->(:Person {name: 'Jules'})")
        .unwrap();
    assert_eq!(
        json(&write)["counters"],
        serde_json::json!({
            "nodesCreated": 2, "nodesDeleted": 0, "edgesCreated": 1, "edgesDeleted": 0,
            "propertiesSet": 2, "labelsAdded": 2, "labelsRemoved": 0
        })
    );
    let read = db.execute_raw("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(json(&read)["counters"]["nodesCreated"], 0);
}

#[cfg(feature = "direct-api")]
#[wasm_bindgen_test]
fn test_graph_handle_works_in_its_graph() {
    let db = Database::new().unwrap();
    db.execute("CREATE GRAPH trips").unwrap();
    assert!(db.graph("nowhere").is_err());

    let trips = db.graph("trips").unwrap();
    assert_eq!(trips.name(), "trips");
    trips
        .execute("INSERT (:City {name: 'Prague'})-[:ROUTE {km: 350}]->(:City {name: 'Berlin'})")
        .unwrap();
    let rows = trips
        .execute_with_params(
            "MATCH (c:City {name: $name})-[r:ROUTE]->(d) RETURN d.name AS d, r.km AS km",
            js(serde_json::json!({ "name": "Prague" })),
        )
        .unwrap();
    assert_eq!(
        json(&rows),
        serde_json::json!([{ "d": "Berlin", "km": 350 }])
    );
    assert_eq!(db.node_count(), 0);

    let ids = trips
        .batch_create_nodes_with_props(
            js(serde_json::json!("City")),
            js(serde_json::json!([{ "name": "Paris" }, { "name": "Amsterdam" }])),
        )
        .unwrap();
    let ids: Vec<u64> = serde_wasm_bindgen::from_value(ids).unwrap();
    trips
        .batch_create_edges(js(
            serde_json::json!([{ "src": ids[0], "dst": ids[1], "type": "ROUTE" }]),
        ))
        .unwrap();
    let upsert = trips
        .upsert_nodes(
            js(serde_json::json!(["City"])),
            js(serde_json::json!([{ "name": "Prague", "country": "CZ" }])),
            js(serde_json::json!({ "key": "name" })),
        )
        .unwrap();
    assert_eq!(json(&upsert)["updated"], 1);
    let raw = trips
        .execute_raw("MATCH (c:City) RETURN count(c) AS n")
        .unwrap();
    assert_eq!(json(&raw)["rows"], serde_json::json!([[4]]));

    db.begin_transaction().unwrap();
    let message = error_text(trips.execute("MATCH (c:City) RETURN c"));
    assert!(message.contains("commit or roll back"), "{message}");
    db.rollback_transaction().unwrap();

    db.execute("DROP GRAPH trips").unwrap();
    assert!(trips.execute("RETURN 1").is_err());
    db.close();
    let message = error_text(trips.execute("RETURN 1"));
    assert!(message.contains("closed"), "{message}");
}

#[cfg(feature = "vector-index")]
#[wasm_bindgen_test]
fn test_batch_vector_search_answers_each_query() {
    let db = Database::new().unwrap();
    db.execute("INSERT (:Doc {topic: 'a', e: vector([1.0, 0.0])}), (:Doc {topic: 'b', e: vector([0.0, 1.0])})")
        .unwrap();
    db.create_vector_index(
        "Doc",
        "e",
        js(serde_json::json!({ "dimensions": 2, "metric": "cosine" })),
    )
    .unwrap();
    let topic = |hit: &serde_json::Value| {
        let rows = db
            .execute_with_params(
                "MATCH (d:Doc) WHERE id(d) = $id RETURN d.topic AS t",
                js(serde_json::json!({ "id": hit["id"] })),
            )
            .unwrap();
        json(&rows)[0]["t"].clone()
    };

    let results = db
        .batch_vector_search(
            "Doc",
            "e",
            js(serde_json::json!([[1.0, 0.0], [0.0, 1.0]])),
            1,
            wasm_bindgen::JsValue::UNDEFINED,
        )
        .unwrap();
    let results = json(&results);
    assert_eq!(results.as_array().unwrap().len(), 2);
    assert_eq!(topic(&results[0][0]), "a");
    assert_eq!(topic(&results[1][0]), "b");

    let filtered = db
        .batch_vector_search(
            "Doc",
            "e",
            js(serde_json::json!([[1.0, 0.0]])),
            2,
            js(serde_json::json!({ "filters": { "topic": "b" } })),
        )
        .unwrap();
    let filtered = json(&filtered);
    assert_eq!(filtered[0].as_array().unwrap().len(), 1);
    assert_eq!(topic(&filtered[0][0]), "b");
}

#[cfg(feature = "lpg-model")]
#[wasm_bindgen_test]
fn test_get_node_edges_by_direction_type_and_transaction() {
    let db = Database::new().unwrap();
    db.execute(
        "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(g:Person {name: 'Gus'}), \
         (g)-[:WORKS_WITH]->(a), (a)-[:NOTE]->(a)",
    )
    .unwrap();
    let id_of = |name: &str| {
        let rows = db
            .execute(&format!(
                "MATCH (p:Person {{name: '{name}'}}) RETURN id(p) AS id"
            ))
            .unwrap();
        json(&rows)[0]["id"].as_u64().unwrap()
    };
    let (alix, gus) = (id_of("Alix"), id_of("Gus"));
    let neighbors = |id: u64, options: serde_json::Value| -> Vec<u64> {
        let mut found: Vec<u64> = json(
            &db.get_node_edges(id as f64, to_js(&options).unwrap())
                .unwrap(),
        )
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| pair[0].as_u64().unwrap())
        .collect();
        found.sort_unstable();
        found
    };

    assert_eq!(neighbors(alix, serde_json::json!(null)), {
        let mut v = vec![gus, alix];
        v.sort_unstable();
        v
    });
    assert_eq!(neighbors(alix, serde_json::json!({ "direction": "in" })), {
        let mut v = vec![gus, alix];
        v.sort_unstable();
        v
    });
    // Both directions list the self-loop once.
    assert_eq!(
        neighbors(alix, serde_json::json!({ "direction": "both" })).len(),
        3
    );
    assert_eq!(
        neighbors(
            alix,
            serde_json::json!({ "direction": "both", "edgeTypes": ["KNOWS"] })
        ),
        vec![gus]
    );
    assert!(
        db.get_node_edges(
            alix as f64,
            to_js(&serde_json::json!({ "direction": "up" })).unwrap()
        )
        .is_err()
    );
    assert!(
        db.get_node_edges(0.5, wasm_bindgen::JsValue::UNDEFINED)
            .is_err()
    );

    db.begin_transaction().unwrap();
    db.execute("MATCH (g:Person {name: 'Gus'}) INSERT (g)-[:KNOWS]->(:Person {name: 'Mia'})")
        .unwrap();
    assert_eq!(
        neighbors(gus, serde_json::json!({ "edgeTypes": ["KNOWS"] })).len(),
        1
    );
    db.rollback_transaction().unwrap();
    assert_eq!(
        neighbors(gus, serde_json::json!({ "edgeTypes": ["KNOWS"] })),
        Vec::<u64>::new()
    );
}

/// A database error carries its GQLSTATUS as `gqlStatus` next to its message
/// (#639): `22012` for a division by zero, `22003` for a value out of range,
/// `42001` for a syntax error.
#[wasm_bindgen_test]
fn test_errors_carry_their_gql_status() {
    use wasm_bindgen::JsCast;

    let db = Database::new().expect("create db");
    for (query, status, message) in [
        ("RETURN 1 / 0 AS x", "22012", "division by zero: 1 / 0"),
        (
            "RETURN 9223372036854775807 + 1 AS x",
            "22003",
            "numeric value out of range: 9223372036854775807 + 1",
        ),
        (
            "UNWIND [9223372036854775807, 1] AS x RETURN sum(x) AS s",
            "22003",
            "numeric value out of range: sum",
        ),
        ("MATCH (n RETURN n", "42001", "GRAFEO-Q001"),
    ] {
        let error = db.execute(query).unwrap_err();
        let get = |name: &str| {
            js_sys::Reflect::get(&error, &wasm_bindgen::JsValue::from_str(name))
                .unwrap()
                .as_string()
        };
        assert_eq!(get("gqlStatus").as_deref(), Some(status), "{query}");
        let text = get("message").unwrap_or_default();
        assert!(text.contains(message), "{query}: {text}");
        assert!(error.is_instance_of::<js_sys::Error>(), "{query}");
    }
}

/// `executeRaw` of a `PROFILE` statement has `profile`, the operator tree
/// (#461); other statements have none.
#[wasm_bindgen_test]
fn test_execute_raw_profile_tree() {
    let db = Database::new().expect("create db");
    db.execute("INSERT (:Person {age: 30}), (:Person {age: 25}), (:Person {age: 35})")
        .expect("insert");

    let raw = db
        .execute_raw("PROFILE MATCH (p:Person) WHERE p.age > 28 RETURN p.age")
        .expect("profile");
    let profile = field(&raw, "profile");
    assert!(profile.is_object(), "PROFILE has a profile");
    assert!(field(&profile, "totalTimeMs").as_f64().is_some());
    let root = field(&profile, "root");
    assert_eq!(field(&root, "rows").as_f64(), Some(2.0));
    assert!(
        field(&root, "operator")
            .as_string()
            .is_some_and(|op| !op.is_empty())
    );
    for key in ["detail", "calls", "selfTimeMs", "timeMs"] {
        assert!(!field(&root, key).is_undefined(), "{key}");
    }
    assert!(js_sys::Array::is_array(&field(&root, "children")));

    let plain = db
        .execute_raw("MATCH (p:Person) RETURN p.age")
        .expect("query");
    assert!(field(&plain, "profile").is_undefined());
}

/// The engine has no clock on wasm32: the binding times every raw call (#461).
#[wasm_bindgen_test]
fn test_execute_raw_has_a_time() {
    let db = Database::new().expect("create db");
    db.execute("INSERT (:Person {age: 30})").expect("insert");
    let raw = db
        .execute_raw("MATCH (p:Person) RETURN p.age")
        .expect("gql");
    assert!(
        field(&raw, "executionTimeMs")
            .as_f64()
            .is_some_and(|ms| ms >= 0.0)
    );
    let raw = db
        .execute_raw_with_language("MATCH (p:Person) RETURN p.age", "gql")
        .expect("with language");
    assert!(
        field(&raw, "executionTimeMs")
            .as_f64()
            .is_some_and(|ms| ms >= 0.0)
    );
}
