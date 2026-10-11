//! MERGE of a relationship pattern in GQL and Cypher: an end node without a
//! variable is merged on its own when the pattern gives it a label or a
//! property, and a pattern of more than one relationship is refused instead
//! of merging its first relationship only.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test merge_paths
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix, Gus and Vincent, and Alix KNOWS Gus.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (vincent:Person {name: 'Vincent'}), (alix)-[:KNOWS]->(gus)",
    )
    .unwrap();
    db
}

/// The single value `query` (GQL) returns.
fn value(db: &GrafeoDB, query: &str) -> Value {
    let result = db.execute(query).unwrap();
    assert_eq!(result.rows().len(), 1, "one row from {query}");
    result.rows()[0][0].clone()
}

/// The number of relationships and nodes, for checking that a refused MERGE
/// wrote nothing.
fn size(db: &GrafeoDB) -> (Value, Value) {
    (
        value(db, "MATCH ()-[r]->() RETURN count(r)"),
        value(db, "MATCH (n) RETURN count(n)"),
    )
}

// ---------------------------------------------------------------------------
// An end node without a variable
// ---------------------------------------------------------------------------

/// GQL merges the labeled end node first, as Cypher does: the second MERGE
/// finds both the node and the relationship.
#[test]
fn gql_merges_a_relationship_to_an_anonymous_labeled_node() {
    let db = graph();
    let merge = "MATCH (v:Person {name: 'Vincent'}) MERGE (v)-[:KNOWS]->(:Person {name: 'Gus'})";
    db.execute(merge).unwrap();
    db.execute(merge).unwrap();
    assert_eq!(
        value(&db, "MATCH (p:Person {name: 'Gus'}) RETURN count(p)"),
        Value::Int64(1),
        "Gus is found, not created"
    );
    assert_eq!(
        value(
            &db,
            "MATCH (:Person {name: 'Vincent'})-[r:KNOWS]->(:Person {name: 'Gus'}) RETURN count(r)"
        ),
        Value::Int64(1),
        "the second MERGE finds the relationship the first created"
    );
}

/// The start node may be anonymous as well, and a node the MERGE creates
/// takes the relationship.
#[test]
fn gql_merges_a_relationship_from_an_anonymous_node_it_creates() {
    let db = graph();
    db.execute("MATCH (g:Person {name: 'Gus'}) MERGE (:City {name: 'Prague'})-[:HOME_OF]->(g)")
        .unwrap();
    assert_eq!(
        value(
            &db,
            "MATCH (c:City)-[:HOME_OF]->(p:Person) RETURN c.name || ' ' || p.name"
        ),
        Value::from("Prague Gus")
    );
}

/// An anonymous end node with neither a label nor a property could be any
/// node: GQL refuses it with the message Cypher gives.
#[test]
fn an_anonymous_node_without_a_label_or_property_is_refused() {
    let db = graph();
    let before = size(&db);
    let query = "MATCH (v:Person {name: 'Vincent'}) MERGE (v)-[:KNOWS]->()";
    let message = db.execute(query).unwrap_err().to_string();
    assert!(
        message.contains("anonymous node without a label or property"),
        "GQL: {message}"
    );
    #[cfg(feature = "cypher")]
    {
        let message = db.execute_cypher(query).unwrap_err().to_string();
        assert!(
            message.contains("anonymous node without a label or property"),
            "Cypher: {message}"
        );
    }
    assert_eq!(size(&db), before, "nothing written");
}

// ---------------------------------------------------------------------------
// A pattern of more than one relationship
// ---------------------------------------------------------------------------

/// `MERGE (a)-[:S]->(b)-[:T]->(:N {id: 3})` used to merge the S relationship
/// and drop the rest of the pattern without a word. It is refused, in both
/// languages, before it writes anything.
#[test]
fn a_merge_of_more_than_one_relationship_is_refused() {
    let db = graph();
    let before = size(&db);
    let query = "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 MERGE (a)-[:S]->(b)-[:T]->(:N {id: 3})";
    let message = db.execute(query).unwrap_err().to_string();
    assert!(
        message.contains("more than one relationship"),
        "GQL: {message}"
    );
    #[cfg(feature = "cypher")]
    {
        let message = db.execute_cypher(query).unwrap_err().to_string();
        assert!(
            message.contains("more than one relationship"),
            "Cypher: {message}"
        );
    }
    assert_eq!(size(&db), before, "nothing written");
}
