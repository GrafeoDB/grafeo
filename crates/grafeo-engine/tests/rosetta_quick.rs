//! Quick Rosetta tests for cross-language query comparison.

use grafeo_engine::GrafeoDB;

fn setup_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE (:Entity {name: 'Alice', mentions: 10})")
        .unwrap();
    db.execute("CREATE (:Entity {name: 'Bob', mentions: 5})")
        .unwrap();
    db.execute("CREATE (:Organization {name: 'Corp', mentions: 3})")
        .unwrap();
    db.execute("MATCH (a:Entity {name: 'Alice'}), (b:Entity {name: 'Bob'}) CREATE (a)-[:RELATED_TO {relationship: 'friend'}]->(b)").unwrap();
    db.execute("MATCH (a:Entity {name: 'Alice'}), (o:Organization {name: 'Corp'}) CREATE (a)-[:RELATED_TO {relationship: 'works_at'}]->(o)").unwrap();
    db.execute("MATCH (b:Entity {name: 'Bob'}), (o:Organization {name: 'Corp'}) CREATE (b)-[:RELATED_TO {relationship: 'works_at'}]->(o)").unwrap();
    db
}

#[test]
fn gql_count_star() {
    let db = setup_db();
    let r = db.execute("MATCH ()-[r:RELATED_TO]->() RETURN r.relationship AS rel_type, count(*) AS cnt ORDER BY cnt DESC LIMIT 20");
    assert!(r.is_ok(), "GQL count(*) failed: {:?}", r.err());
    assert!(!r.unwrap().rows().is_empty(), "r.unwrap().rows() is empty");
}

#[test]
fn gql_count_star_simple() {
    let db = setup_db();
    let r = db.execute("MATCH (n) RETURN count(*) AS total");
    assert!(r.is_ok(), "GQL count(*) simple failed: {:?}", r.err());
    let rows = r.unwrap();
    assert_eq!(rows.rows().len(), 1);
}

#[test]
fn gql_labels_index_access() {
    let db = setup_db();
    let r = db.execute("MATCH (n) RETURN labels(n)[0] AS type, n.name AS name, n.mentions AS mentions ORDER BY mentions DESC LIMIT 30");
    assert!(r.is_ok(), "GQL labels(n)[0] failed: {:?}", r.err());
}

#[test]
fn gql_undirected_edge() {
    let db = setup_db();
    let r = db.execute("MATCH (o:Organization)-[r]-(n) RETURN o, r, n LIMIT 100");
    assert!(r.is_ok(), "GQL undirected edge failed: {:?}", r.err());
}

#[test]
fn gql_connections() {
    let db = setup_db();
    let r = db.execute("MATCH (a {name: 'Alice'})-[r]->(b) RETURN a, r, b LIMIT 50");
    assert!(r.is_ok(), "GQL connections failed: {:?}", r.err());
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_count_star() {
    let db = setup_db();
    let r = db.execute_cypher("MATCH ()-[r:RELATED_TO]->() RETURN r.relationship AS rel_type, count(*) AS count ORDER BY count DESC LIMIT 20");
    assert!(r.is_ok(), "Cypher count(*) failed: {:?}", r.err());
}
