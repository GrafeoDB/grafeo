//! The schema summary in realistic shapes: each label and edge type with the
//! properties its elements hold and the types of their values, and each edge
//! type with the label pairs it connects. Nodes with several labels count in
//! every pair, an end without a label is in none, a key holding values of
//! several types lists each type once, and `schema()` and the `db.*`
//! procedures agree on what is in use.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test schema_summary_scenarios
//! ```

#![cfg(feature = "gql")]

use grafeo_engine::GrafeoDB;
use grafeo_engine::admin::{EdgeTypeInfo, LabelInfo, LpgSchemaInfo, SchemaInfo};

fn lpg(db: &GrafeoDB) -> LpgSchemaInfo {
    match db.schema().unwrap() {
        SchemaInfo::Lpg(info) => info,
        other => panic!("expected an LPG schema, got {other:?}"),
    }
}

fn label<'a>(info: &'a LpgSchemaInfo, name: &str) -> &'a LabelInfo {
    info.labels
        .iter()
        .find(|label| label.name == name)
        .unwrap_or_else(|| panic!("no label {name} in {:?}", info.labels))
}

fn edge_type<'a>(info: &'a LpgSchemaInfo, name: &str) -> &'a EdgeTypeInfo {
    info.edge_types
        .iter()
        .find(|edge_type| edge_type.name == name)
        .unwrap_or_else(|| panic!("no edge type {name} in {:?}", info.edge_types))
}

/// The properties as `(name, types)`, in the order the summary gives them.
fn properties(properties: &[grafeo_engine::admin::PropertyInfo]) -> Vec<(String, Vec<String>)> {
    properties
        .iter()
        .map(|property| (property.name.clone(), property.types.clone()))
        .collect()
}

/// The endpoints as `(source, target, count)`, in the order the summary
/// gives them.
fn endpoints(edge_type: &EdgeTypeInfo) -> Vec<(String, String, usize)> {
    edge_type
        .endpoints
        .iter()
        .map(|endpoint| {
            (
                endpoint.source.clone(),
                endpoint.target.clone(),
                endpoint.count,
            )
        })
        .collect()
}

fn owned(pairs: &[(&str, &[&str])]) -> Vec<(String, Vec<String>)> {
    pairs
        .iter()
        .map(|(name, types)| {
            (
                (*name).to_string(),
                types.iter().map(|t| (*t).to_string()).collect(),
            )
        })
        .collect()
}

fn triples(items: &[(&str, &str, usize)]) -> Vec<(String, String, usize)> {
    items
        .iter()
        .map(|(source, target, count)| ((*source).to_string(), (*target).to_string(), *count))
        .collect()
}

fn column(db: &GrafeoDB, query: &str) -> Vec<String> {
    db.execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            grafeo_common::types::Value::String(text) => text.to_string(),
            other => panic!("`{query}`: {other:?}"),
        })
        .collect()
}

/// An edge between nodes with several labels counts once in every source
/// and target label pair, and the pairs come sorted.
#[test]
fn multi_label_endpoints_count_in_every_pair() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person:Employee {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (acme:Company:Org {name: 'Acme'}), \
         (alix)-[:WORKS_AT]->(acme), (gus)-[:WORKS_AT]->(acme)",
    )
    .unwrap();
    let info = lpg(&db);
    let works_at = edge_type(&info, "WORKS_AT");
    assert_eq!(works_at.count, 2);
    assert_eq!(
        endpoints(works_at),
        triples(&[
            ("Employee", "Company", 1),
            ("Employee", "Org", 1),
            ("Person", "Company", 2),
            ("Person", "Org", 2),
        ])
    );
    assert_eq!(label(&info, "Person").count, 2);
    assert_eq!(label(&info, "Employee").count, 1);
}

/// One key holding values of several types lists each type once, sorted;
/// a null is no value and adds no type.
#[test]
fn a_key_with_mixed_value_types_lists_each_type_once() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix', age: 30}), (:Person {name: 'Gus', age: '31'}), \
         (:Person {name: 'Vincent', age: 2.5e0}), (:Person {name: 'Jules', age: 40}), \
         (:Person {name: 'Mia', age: null})",
    )
    .unwrap();
    let info = lpg(&db);
    assert_eq!(
        properties(&label(&info, "Person").properties),
        owned(&[
            ("age", &["FLOAT64", "INT64", "STRING"]),
            ("name", &["STRING"]),
        ])
    );
}

/// A key some nodes of a label hold and others do not is listed for the
/// label; a key only another label's nodes hold is not.
#[test]
fn properties_are_per_label() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix', email: 'alix@example.org'}), (:Person {name: 'Gus'}), \
         (:City {name: 'Amsterdam', population: 921402})",
    )
    .unwrap();
    let info = lpg(&db);
    assert_eq!(
        properties(&label(&info, "Person").properties),
        owned(&[("email", &["STRING"]), ("name", &["STRING"])])
    );
    assert_eq!(
        properties(&label(&info, "City").properties),
        owned(&[("name", &["STRING"]), ("population", &["INT64"])])
    );
}

/// An edge whose ends have no label is counted for its type and is in no
/// pair; an edge from a labelled node to an unlabelled one neither.
#[test]
fn edges_between_unlabelled_nodes_have_no_endpoints() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT ({name: 'a'})-[:LINK {w: 1}]->({name: 'b'})")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})-[:LINK {w: 2.5e0}]->({name: 'c'})")
        .unwrap();
    let info = lpg(&db);
    let link = edge_type(&info, "LINK");
    assert_eq!(link.count, 2);
    assert_eq!(endpoints(link), triples(&[]));
    assert_eq!(
        properties(&link.properties),
        owned(&[("w", &["FLOAT64", "INT64"])])
    );
    // The unlabelled nodes are in no label; the labelled one is.
    assert_eq!(
        info.labels
            .iter()
            .map(|label| label.name.as_str())
            .collect::<Vec<_>>(),
        ["Person"]
    );
    // Their keys are property keys all the same.
    assert_eq!(info.property_keys, ["name", "w"]);
}

/// An edge type's endpoints change as edges and labels do: a deleted edge
/// leaves its pair, a removed label leaves every pair it was in.
#[test]
fn endpoints_follow_the_data() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person:Admin {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (paris:City {name: 'Paris'}), \
         (alix)-[:LIVES_IN]->(paris), (gus)-[:LIVES_IN]->(paris), (alix)-[:KNOWS]->(gus)",
    )
    .unwrap();
    db.execute("MATCH (n:Admin) REMOVE n:Admin").unwrap();
    db.execute("MATCH ()-[k:KNOWS]->() DELETE k").unwrap();
    let info = lpg(&db);
    assert_eq!(
        endpoints(edge_type(&info, "LIVES_IN")),
        triples(&[("Person", "City", 2)])
    );
    assert!(
        info.edge_types
            .iter()
            .all(|edge_type| edge_type.name != "KNOWS"),
        "{:?}",
        info.edge_types
    );
    assert!(
        info.labels.iter().all(|label| label.name != "Admin"),
        "{:?}",
        info.labels
    );
}

/// The summary lists labels, edge types and keys sorted by name, whatever
/// the insertion order.
#[test]
fn the_summary_is_sorted_by_name() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Zebra {z: 1, a: 2})-[:ZAPS]->(:Ant {m: 3}), (:Mole)-[:ATE]->(:Ant)")
        .unwrap();
    let info = lpg(&db);
    let labels: Vec<_> = info
        .labels
        .iter()
        .map(|label| label.name.as_str())
        .collect();
    let edge_types: Vec<_> = info
        .edge_types
        .iter()
        .map(|edge_type| edge_type.name.as_str())
        .collect();
    assert_eq!(labels, ["Ant", "Mole", "Zebra"]);
    assert_eq!(edge_types, ["ATE", "ZAPS"]);
    assert_eq!(info.property_keys, ["a", "m", "z"]);
    assert_eq!(
        properties(&label(&info, "Zebra").properties),
        owned(&[("a", &["INT64"]), ("z", &["INT64"])])
    );
}

/// `schema()` and `CALL db.labels()`, `db.relationshipTypes()` and
/// `db.propertyKeys()` list the same names.
#[test]
fn schema_and_the_db_procedures_agree() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person:Employee {name: 'Alix', since: date('2020-01-01')}), \
         (acme:Company {name: 'Acme', tags: ['a', 'b']}), ({loose: true}), \
         (alix)-[:WORKS_AT {role: 'dev'}]->(acme)",
    )
    .unwrap();
    db.execute("INSERT (:Temp {gone: 1})-[:TMP]->(:Temp)")
        .unwrap();
    db.execute("MATCH (t:Temp) DETACH DELETE t").unwrap();
    let info = lpg(&db);
    let labels: Vec<_> = info.labels.iter().map(|label| label.name.clone()).collect();
    let edge_types: Vec<_> = info
        .edge_types
        .iter()
        .map(|edge_type| edge_type.name.clone())
        .collect();
    assert_eq!(labels, ["Company", "Employee", "Person"]);
    assert_eq!(edge_types, ["WORKS_AT"]);
    assert_eq!(
        info.property_keys,
        ["loose", "name", "role", "since", "tags"]
    );
    assert_eq!(
        column(&db, "CALL db.labels() YIELD label RETURN label"),
        labels
    );
    assert_eq!(
        column(
            &db,
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        ),
        edge_types
    );
    assert_eq!(
        column(
            &db,
            "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"
        ),
        info.property_keys
    );
    assert_eq!(
        properties(&label(&info, "Person").properties),
        owned(&[("name", &["STRING"]), ("since", &["DATE"])])
    );
    assert_eq!(
        properties(&label(&info, "Company").properties),
        owned(&[("name", &["STRING"]), ("tags", &["LIST"])])
    );
}

/// The summary serializes with the new fields, and a summary without them
/// (as an older version wrote it) still reads, with empty lists.
#[test]
fn the_summary_serializes_with_its_new_fields() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})")
        .unwrap();
    let json = serde_json::to_value(db.schema().unwrap()).unwrap();
    assert_eq!(json["mode"], "lpg");
    let lpg = &json;
    assert_eq!(
        lpg["labels"][0]["properties"],
        serde_json::json!([{"name": "name", "types": ["STRING"]}])
    );
    assert_eq!(
        lpg["edge_types"][0]["endpoints"],
        serde_json::json!([{"source": "Person", "target": "Person", "count": 1}])
    );
    assert_eq!(
        lpg["edge_types"][0]["properties"],
        serde_json::json!([{"name": "since", "types": ["INT64"]}])
    );
    let old: LabelInfo =
        serde_json::from_value(serde_json::json!({"name": "Person", "count": 2})).unwrap();
    assert_eq!(old.properties.len(), 0);
    let old: EdgeTypeInfo =
        serde_json::from_value(serde_json::json!({"name": "KNOWS", "count": 1})).unwrap();
    assert_eq!((old.properties.len(), old.endpoints.len()), (0, 0));
}
