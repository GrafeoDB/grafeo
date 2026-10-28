//! A numeric list written to a vector-indexed property is an embedding
//! (#587): it is stored as a VECTOR and the index finds it. Before, a GQL or
//! Cypher list literal, a list parameter (what a JavaScript array or a Python
//! list with an integer becomes) or a list given to the direct API was stored
//! as a LIST and left out of the index, with no error, so `vector_search`
//! returned nothing for it.
//!
//! The rules, from every write path:
//! - a list of numbers (integers, floats or both) of the indexed dimension is
//!   stored as a VECTOR of the nearest 32-bit floats, and indexed;
//! - a list of numbers of another dimension fails with the error a VECTOR of
//!   that dimension gives, and nothing is written;
//! - a list with an element that is not a number fails, naming the index;
//! - a list on a property without a vector index stays a LIST.

#![cfg(all(feature = "vector-index", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// The error a write of a `found`-element vector to `:Doc(emb)` gives.
fn wrong_size(found: usize) -> String {
    format!("property 'emb' on :Doc has a vector index of 3 dimensions, got a vector of {found}")
}

/// A database with a 3-dimension cosine index on `:Doc(emb)` and one
/// document, `anchor`, written as a VECTOR.
fn indexed() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {name: 'anchor', emb: vector([0.0, 0.0, 1.0])})")
        .unwrap();
    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .unwrap();
    db
}

fn list(elements: &[Value]) -> Value {
    Value::List(elements.to_vec().into())
}

fn floats(elements: &[f64]) -> Value {
    list(
        &elements
            .iter()
            .map(|x| Value::Float64(*x))
            .collect::<Vec<_>>(),
    )
}

fn name_of(db: &GrafeoDB, id: NodeId) -> String {
    let node = db.get_node(id).unwrap().expect("a hit is a node");
    match node.get_property("name") {
        Some(Value::String(name)) => name.to_string(),
        other => panic!("node {id:?} has no name: {other:?}"),
    }
}

/// The value of `emb` on the `:Doc` named `name`, read through a query.
fn stored(db: &GrafeoDB, name: &str) -> Option<Value> {
    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::from(name));
    let result = db
        .execute_with_params("MATCH (d:Doc {name: $name}) RETURN d.emb AS emb", params)
        .unwrap();
    match result.rows() {
        [] => None,
        [row] => Some(row[0].clone()),
        rows => panic!("{name}: {} documents", rows.len()),
    }
}

/// Asserts that the `:Doc` named `name` holds `expected` as a VECTOR and that
/// a search for `expected` finds it at distance 0.
fn assert_indexed(db: &GrafeoDB, name: &str, expected: [f32; 3], what: &str) {
    match stored(db, name) {
        Some(Value::Vector(vector)) => {
            assert_eq!(&*vector, &expected, "{what}: the stored vector");
        }
        other => panic!("{what}: expected a VECTOR, stored {other:?}"),
    }
    let hits = db
        .vector_search("Doc", "emb", &expected, 100, None, None)
        .unwrap();
    let hit = hits
        .iter()
        .find(|(id, _)| name_of(db, *id) == name)
        .unwrap_or_else(|| panic!("{what}: the index does not find {name}: {hits:?}"));
    assert!(hit.1.abs() < 1e-6, "{what}: distance {}", hit.1);
}

/// Asserts that `result` failed with an error containing `message`, and that
/// the `:Doc` named `name` holds no `emb` (or does not exist).
fn assert_refused<T: std::fmt::Debug>(
    db: &GrafeoDB,
    result: grafeo_common::utils::error::Result<T>,
    message: &str,
    name: &str,
    what: &str,
) {
    let err = result.expect_err(&format!("{what}: expected an error ({message})"));
    assert!(err.to_string().contains(message), "{what}: {err}");
    assert!(
        matches!(stored(db, name), None | Some(Value::Null)),
        "{what}: the refused value was stored: {:?}",
        stored(db, name)
    );
}

/// Every GQL write of a list literal or a list parameter to the indexed
/// property: an INSERT, a SET, with floats, integers or both.
#[test]
fn gql_list_writes_are_indexed_vectors() {
    let db = indexed();

    db.execute("INSERT (:Doc {name: 'Alix', emb: [1.0, 0.0, 0.0]})")
        .unwrap();
    assert_indexed(&db, "Alix", [1.0, 0.0, 0.0], "INSERT a float list");

    db.execute("INSERT (:Doc {name: 'Gus'})").unwrap();
    db.execute("MATCH (d:Doc {name: 'Gus'}) SET d.emb = [0, 1, 0]")
        .unwrap();
    assert_indexed(&db, "Gus", [0.0, 1.0, 0.0], "SET an integer list");

    db.execute("INSERT (:Doc {name: 'Vincent', emb: [1, 0.5, 0]})")
        .unwrap();
    assert_indexed(&db, "Vincent", [1.0, 0.5, 0.0], "INSERT a mixed list");

    let mut params = HashMap::new();
    params.insert("vec".to_string(), floats(&[0.5, 1.0, 0.0]));
    db.execute("INSERT (:Doc {name: 'Jules'})").unwrap();
    db.execute_with_params("MATCH (d:Doc {name: 'Jules'}) SET d.emb = $vec", params)
        .unwrap();
    assert_indexed(&db, "Jules", [0.5, 1.0, 0.0], "SET a list parameter");

    // A JavaScript array of whole numbers arrives as integers.
    let mut params = HashMap::new();
    params.insert(
        "vec".to_string(),
        list(&[Value::Int64(0), Value::Float64(0.5), Value::Int64(1)]),
    );
    db.execute_with_params("INSERT (:Doc {name: 'Mia', emb: $vec})", params)
        .unwrap();
    assert_indexed(&db, "Mia", [0.0, 0.5, 1.0], "INSERT a list parameter");

    // A whole property map, as a parameter.
    let mut params = HashMap::new();
    let mut map = std::collections::BTreeMap::new();
    map.insert(PropertyKey::new("name"), Value::from("Butch"));
    map.insert(PropertyKey::new("emb"), floats(&[1.0, 1.0, 0.0]));
    params.insert("props".to_string(), Value::Map(map.into()));
    db.execute_with_params("INSERT (d:Doc) SET d = $props", params)
        .unwrap();
    assert_indexed(&db, "Butch", [1.0, 1.0, 0.0], "SET a map parameter");
}

/// 0.1 is exact in neither width: a list element becomes the nearest 32-bit
/// float, the one `vector()` makes.
#[test]
fn a_list_element_becomes_the_nearest_32_bit_float() {
    let db = indexed();
    db.execute("INSERT (:Doc {name: 'Alix', emb: [0.1, 0.2, 0.3]})")
        .unwrap();
    db.execute("INSERT (:Draft {name: 'Gus', emb: vector([0.1, 0.2, 0.3])})")
        .unwrap();
    let from_vector = db.execute("MATCH (d:Draft) RETURN d.emb").unwrap().rows()[0][0].clone();
    assert_eq!(stored(&db, "Alix"), Some(from_vector));
    assert_indexed(&db, "Alix", [0.1, 0.2, 0.3], "INSERT 0.1, 0.2, 0.3");
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_list_writes_are_indexed_vectors() {
    let db = indexed();

    db.execute_cypher("CREATE (:Doc {name: 'Alix', emb: [1.0, 0.0, 0.0]})")
        .unwrap();
    assert_indexed(&db, "Alix", [1.0, 0.0, 0.0], "CREATE");

    db.execute_cypher("CREATE (:Doc {name: 'Gus'})").unwrap();
    db.execute_cypher("MATCH (d:Doc {name: 'Gus'}) SET d.emb = [0, 1, 0]")
        .unwrap();
    assert_indexed(&db, "Gus", [0.0, 1.0, 0.0], "SET");

    db.execute_cypher("CREATE (:Doc {name: 'Vincent'})")
        .unwrap();
    db.execute_cypher("MATCH (d:Doc {name: 'Vincent'}) SET d += {emb: [1.0, 0.5, 0.0]}")
        .unwrap();
    assert_indexed(&db, "Vincent", [1.0, 0.5, 0.0], "SET +=");

    db.execute_cypher("MERGE (d:Doc {name: 'Jules'}) ON CREATE SET d.emb = [0.5, 1.0, 0.0]")
        .unwrap();
    assert_indexed(&db, "Jules", [0.5, 1.0, 0.0], "MERGE ON CREATE SET");

    let mut params = HashMap::new();
    params.insert("vec".to_string(), floats(&[0.0, 0.5, 1.0]));
    db.execute_cypher_with_params("CREATE (:Doc {name: 'Mia', emb: $vec})", params)
        .unwrap();
    assert_indexed(&db, "Mia", [0.0, 0.5, 1.0], "CREATE with a list parameter");
}

/// The direct API, on the database and on a session.
#[test]
fn direct_list_writes_are_indexed_vectors() {
    let db = indexed();
    let props = |name: &str, emb: Value| {
        let mut map = HashMap::new();
        map.insert(PropertyKey::new("name"), Value::from(name));
        map.insert(PropertyKey::new("emb"), emb);
        map
    };

    db.create_node_with_props(&["Doc"], props("Alix", floats(&[1.0, 0.0, 0.0])))
        .unwrap();
    assert_indexed(&db, "Alix", [1.0, 0.0, 0.0], "create_node_with_props");

    let gus = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(gus, "name", Value::from("Gus"))
        .unwrap();
    db.set_node_property(
        gus,
        "emb",
        list(&[Value::Int64(0), Value::Int64(1), Value::Int64(0)]),
    )
    .unwrap();
    assert_indexed(&db, "Gus", [0.0, 1.0, 0.0], "set_node_property");

    db.batch_create_nodes_with_props(
        "Doc",
        vec![
            props("Vincent", floats(&[1.0, 0.5, 0.0])),
            props("Jules", floats(&[0.5, 1.0, 0.0])),
        ],
    )
    .unwrap();
    assert_indexed(
        &db,
        "Vincent",
        [1.0, 0.5, 0.0],
        "batch_create_nodes_with_props",
    );
    assert_indexed(
        &db,
        "Jules",
        [0.5, 1.0, 0.0],
        "batch_create_nodes_with_props",
    );

    let session = db.session();
    session
        .create_node_with_props(&["Doc"], props("Mia", floats(&[0.0, 0.5, 1.0])))
        .unwrap();
    assert_indexed(
        &db,
        "Mia",
        [0.0, 0.5, 1.0],
        "session create_node_with_props",
    );
    let butch = session.create_node(&["Doc"]).unwrap();
    session
        .set_node_property(butch, "name", Value::from("Butch"))
        .unwrap();
    session
        .set_node_property(butch, "emb", floats(&[1.0, 1.0, 0.0]))
        .unwrap();
    assert_indexed(&db, "Butch", [1.0, 1.0, 0.0], "session set_node_property");
}

/// An imported JSON Lines file writes its arrays as lists.
#[cfg(feature = "jsonl-import")]
#[test]
fn imported_lists_are_indexed_vectors() {
    use grafeo_engine::import::{ImportFormat, import_nodes};

    let db = indexed();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("docs.jsonl");
    std::fs::write(
        &path,
        "{\"name\": \"Alix\", \"emb\": [1.0, 0.0, 0.0]}\n{\"name\": \"Gus\", \"emb\": [0, 1, 0]}\n",
    )
    .unwrap();
    assert_eq!(
        import_nodes(&db.session(), &path, "Doc", ImportFormat::Jsonl).unwrap(),
        2
    );
    assert_indexed(&db, "Alix", [1.0, 0.0, 0.0], "JSONL import of floats");
    assert_indexed(&db, "Gus", [0.0, 1.0, 0.0], "JSONL import of integers");
}

/// A numeric list of another dimension fails as a VECTOR of that dimension
/// does, from every path, and leaves nothing behind.
#[test]
fn a_list_of_another_dimension_fails_as_a_vector_does() {
    let db = indexed();
    for (literal, found) in [("[1.0, 0.0]", 2), ("[1, 0, 0, 0]", 4), ("[]", 0)] {
        let message = wrong_size(found);
        assert_refused(
            &db,
            db.execute(&format!("INSERT (:Doc {{name: 'Alix', emb: {literal}}})")),
            &message,
            "Alix",
            &format!("INSERT {literal}"),
        );
        db.execute("INSERT (:Doc {name: 'Gus'})").unwrap();
        assert_refused(
            &db,
            db.execute(&format!(
                "MATCH (d:Doc {{name: 'Gus'}}) SET d.emb = {literal}"
            )),
            &message,
            "Gus",
            &format!("SET {literal}"),
        );
        db.execute("MATCH (d:Doc {name: 'Gus'}) DELETE d").unwrap();
        #[cfg(feature = "cypher")]
        assert_refused(
            &db,
            db.execute_cypher(&format!(
                "CREATE (:Doc {{name: 'Vincent', emb: {literal}}})"
            )),
            &message,
            "Vincent",
            &format!("Cypher CREATE {literal}"),
        );
    }

    let mut params = HashMap::new();
    params.insert("vec".to_string(), floats(&[1.0, 0.0]));
    assert_refused(
        &db,
        db.execute_with_params("INSERT (:Doc {name: 'Jules', emb: $vec})", params),
        &wrong_size(2),
        "Jules",
        "INSERT a list parameter",
    );

    let mia = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(mia, "name", Value::from("Mia"))
        .unwrap();
    assert_refused(
        &db,
        db.set_node_property(mia, "emb", floats(&[1.0, 0.0])),
        &wrong_size(2),
        "Mia",
        "set_node_property",
    );
    let mut props = HashMap::new();
    props.insert(PropertyKey::new("name"), Value::from("Butch"));
    props.insert(PropertyKey::new("emb"), floats(&[1.0, 0.0, 0.0, 0.0]));
    assert_refused(
        &db,
        db.create_node_with_props(&["Doc"], props),
        &wrong_size(4),
        "Butch",
        "create_node_with_props",
    );

    // Only the anchor is indexed, and the database answers.
    let hits = db
        .vector_search("Doc", "emb", &[0.0, 0.0, 1.0], 100, None, None)
        .unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
}

/// A list on the indexed property with an element that is not a number
/// cannot be an embedding: it fails, naming the index, rather than being
/// stored where the index cannot see it.
#[test]
fn a_list_with_a_non_number_on_an_indexed_property_fails() {
    let db = indexed();
    for literal in [
        "[1.0, 'x', 0.0]",
        "[1.0, null, 0.0]",
        "[true, false, true]",
        "[[1.0], [0.0], [0.0]]",
    ] {
        assert_refused(
            &db,
            db.execute(&format!("INSERT (:Doc {{name: 'Alix', emb: {literal}}})")),
            "property 'emb' on :Doc has a vector index",
            "Alix",
            &format!("INSERT {literal}"),
        );
    }
    let gus = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(gus, "name", Value::from("Gus"))
        .unwrap();
    assert_refused(
        &db,
        db.set_node_property(
            gus,
            "emb",
            list(&[Value::Float64(1.0), Value::from("x"), Value::Float64(0.0)]),
        ),
        "property 'emb' on :Doc has a vector index",
        "Gus",
        "set_node_property",
    );
}

/// An element beyond the 32-bit range becomes infinity, which the index
/// cannot measure: the write fails as a VECTOR with infinity does.
#[test]
fn a_list_element_beyond_the_32_bit_range_fails() {
    let db = indexed();
    assert_refused(
        &db,
        db.execute("INSERT (:Doc {name: 'Alix', emb: [1.0e39, 0.0, 0.0]})"),
        "property 'emb' on :Doc has a vector index, which cannot measure inf (at position 0)",
        "Alix",
        "INSERT 1.0e39",
    );
}

/// Without a vector index the list is a value like any other: a LIST, also
/// on another label's `emb` and on another property of `:Doc`.
#[test]
fn a_list_on_an_unindexed_property_stays_a_list() {
    let db = indexed();
    db.execute("INSERT (:Doc {name: 'Alix', tags: [1.0, 0.0, 0.0]})")
        .unwrap();
    db.execute("INSERT (:Draft {name: 'Gus', emb: [1.0, 0.0, 0.0]})")
        .unwrap();
    db.execute("INSERT (:Draft {name: 'Vincent', emb: [1.0, 0.0]})")
        .unwrap();
    for (query, what) in [
        (
            "MATCH (d:Doc {name: 'Alix'}) RETURN d.tags",
            "another property",
        ),
        (
            "MATCH (d:Draft {name: 'Gus'}) RETURN d.emb",
            "another label",
        ),
        (
            "MATCH (d:Draft {name: 'Vincent'}) RETURN d.emb",
            "another label, another size",
        ),
    ] {
        let value = db.execute(query).unwrap().rows()[0][0].clone();
        assert!(matches!(value, Value::List(_)), "{what}: {value:?}");
    }
}

/// Setting the indexed property to NULL, or removing it, still takes the
/// node out of the index.
#[test]
fn a_list_vector_can_be_removed_again() {
    let db = indexed();
    db.execute("INSERT (:Doc {name: 'Alix', emb: [1.0, 0.0, 0.0]})")
        .unwrap();
    db.execute("INSERT (:Doc {name: 'Gus', emb: [0.0, 1.0, 0.0]})")
        .unwrap();
    db.execute("MATCH (d:Doc {name: 'Alix'}) SET d.emb = null")
        .unwrap();
    db.execute("MATCH (d:Doc {name: 'Gus'}) REMOVE d.emb")
        .unwrap();
    let hits = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 100, None, None)
        .unwrap();
    let names: Vec<String> = hits.iter().map(|(id, _)| name_of(&db, *id)).collect();
    assert_eq!(names, ["anchor"]);
}

/// A label added later makes the property an indexed one: a numeric list the
/// node already holds becomes a VECTOR the index finds, through `SET n:Doc`,
/// Cypher and the direct API, and on a node created with both labels.
#[test]
fn a_label_added_later_turns_the_list_into_an_indexed_vector() {
    let db = indexed();
    db.execute("INSERT (:Draft {name: 'Alix', emb: [1.0, 0.0, 0.0]})")
        .unwrap();
    db.execute("MATCH (d:Draft {name: 'Alix'}) SET d:Doc")
        .unwrap();
    assert_indexed(&db, "Alix", [1.0, 0.0, 0.0], "SET n:Doc");

    let gus = db
        .create_node_with_props(
            &["Draft"],
            [
                ("name", Value::from("Gus")),
                (
                    "emb",
                    list(&[Value::Int64(0), Value::Int64(1), Value::Int64(0)]),
                ),
            ],
        )
        .unwrap();
    assert!(db.add_node_label(gus, "Doc").unwrap());
    assert_indexed(&db, "Gus", [0.0, 1.0, 0.0], "add_node_label");

    #[cfg(feature = "cypher")]
    {
        db.execute_cypher("CREATE (:Draft {name: 'Vincent', emb: [1.0, 0.5, 0.0]})")
            .unwrap();
        db.execute_cypher("MATCH (d:Draft {name: 'Vincent'}) SET d:Doc")
            .unwrap();
        assert_indexed(&db, "Vincent", [1.0, 0.5, 0.0], "Cypher SET n:Doc");
    }

    db.execute("INSERT (:Draft:Doc {name: 'Jules', emb: [0.5, 1.0, 0.0]})")
        .unwrap();
    assert_indexed(&db, "Jules", [0.5, 1.0, 0.0], "INSERT with both labels");
}

/// A label that would put a list of another size under the index is
/// refused, as for a vector of that size: the label stays off and the list
/// stays as it was.
#[test]
fn a_label_that_would_index_a_list_of_another_size_is_refused() {
    let db = indexed();
    db.execute("INSERT (:Draft {name: 'Alix', emb: [1.0, 0.0]})")
        .unwrap();
    let alix = db
        .execute("MATCH (d:Draft {name: 'Alix'}) RETURN id(d)")
        .unwrap()
        .scalar::<i64>()
        .unwrap();
    let alix = NodeId::new(u64::try_from(alix).unwrap());
    let refused = |result: grafeo_common::utils::error::Result<()>, what: &str| {
        let err = result.expect_err(&format!("{what}: expected the size error"));
        assert!(err.to_string().contains(&wrong_size(2)), "{what}: {err}");
        let node = db.get_node(alix).unwrap().unwrap();
        assert!(!node.has_label("Doc"), "{what}: the label stays off");
        assert!(
            matches!(node.get_property("emb"), Some(Value::List(items)) if items.len() == 2),
            "{what}: the list stays: {:?}",
            node.get_property("emb")
        );
    };
    refused(
        db.execute("MATCH (d:Draft {name: 'Alix'}) SET d:Doc")
            .map(|_| ()),
        "SET n:Doc",
    );
    refused(db.add_node_label(alix, "Doc").map(|_| ()), "add_node_label");
    #[cfg(feature = "cypher")]
    refused(
        db.execute_cypher("MATCH (d:Draft {name: 'Alix'}) SET d:Doc")
            .map(|_| ()),
        "Cypher SET n:Doc",
    );
    let hits = db
        .vector_search("Doc", "emb", &[0.0, 0.0, 1.0], 100, None, None)
        .unwrap();
    assert_eq!(hits.len(), 1, "only the anchor: {hits:?}");
}
