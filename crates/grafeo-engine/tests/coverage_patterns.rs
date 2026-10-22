//! Tests for pattern translator coverage gaps.
//!
//! Targets: pattern.rs (76.07%), expression.rs paths
//!
//! ```bash
//! cargo test -p grafeo-engine --test coverage_patterns
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Creates a chain graph: 5 nodes (A-E), 4 LINK edges, 1 SHORTCUT edge.
fn chain_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let na = session
        .create_node_with_props(&["Node"], [("name", Value::String("A".into()))])
        .unwrap();
    let nb = session
        .create_node_with_props(&["Node"], [("name", Value::String("B".into()))])
        .unwrap();
    let nc = session
        .create_node_with_props(&["Node"], [("name", Value::String("C".into()))])
        .unwrap();
    let and = session
        .create_node_with_props(&["Node"], [("name", Value::String("D".into()))])
        .unwrap();
    let ne = session
        .create_node_with_props(&["Node", "Special"], [("name", Value::String("E".into()))])
        .unwrap();

    session.create_edge(na, nb, "LINK").unwrap();
    session.create_edge(nb, nc, "LINK").unwrap();
    session.create_edge(nc, and, "LINK").unwrap();
    session.create_edge(and, ne, "LINK").unwrap();
    session.create_edge(na, nc, "SHORTCUT").unwrap();

    // Verify setup data
    assert_eq!(db.node_count(), 5, "chain_graph: expected 5 nodes");
    assert_eq!(
        db.edge_count(),
        5,
        "chain_graph: expected 5 edges (4 LINK + 1 SHORTCUT)"
    );

    db
}

// ---------------------------------------------------------------------------
// Variable-length edges: *min..max (exercises is_variable_length branch)
// ---------------------------------------------------------------------------

#[test]
fn test_variable_length_1_to_3() {
    let db = chain_graph();
    let s = db.session();
    let r = s
        .execute(
            "MATCH (a:Node {name: 'A'})-[:LINK *1..3]->(b:Node) \
             RETURN b.name AS name ORDER BY name",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 3, "1..3 hops from A via LINK: B, C, D");
    let names: Vec<&str> = r
        .rows()
        .iter()
        .filter_map(|row| match &row[0] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        vec!["B", "C", "D"],
        "ORDER BY name should sort alphabetically"
    );
}

#[test]
fn test_variable_length_exact_2() {
    let db = chain_graph();
    let s = db.session();
    let r = s
        .execute(
            "MATCH (a:Node {name: 'A'})-[:LINK *2..2]->(b:Node) \
             RETURN b.name AS name",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("C".into()));
}

#[test]
fn test_variable_length_unbounded() {
    let db = chain_graph();
    let s = db.session();
    let r = s
        .execute(
            "MATCH (a:Node {name: 'A'})-[:LINK*]->(b:Node) \
             RETURN b.name AS name ORDER BY name",
        )
        .unwrap();
    assert_eq!(
        r.rows().len(),
        4,
        "Unbounded hops from A via LINK: B, C, D, E"
    );
}

// ---------------------------------------------------------------------------
// Multi-label node matching
// ---------------------------------------------------------------------------

#[test]
fn test_multi_label_match_and_semantics() {
    let db = chain_graph();
    let s = db.session();
    // :Node:Special should match only nodes with BOTH labels (AND semantics per ISO GQL).
    // Only node E has both "Node" and "Special" labels.
    let r = s
        .execute("MATCH (n:Node:Special) RETURN n.name AS name")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("E".into()));
}

#[test]
fn test_multi_label_no_match_when_missing_label() {
    let db = chain_graph();
    let s = db.session();
    // No node has both "Special" and "Nonexistent", should return 0 rows.
    let r = s
        .execute("MATCH (n:Special:Nonexistent) RETURN n.name AS name")
        .unwrap();
    assert_eq!(r.rows().len(), 0);
}

// ---------------------------------------------------------------------------
// Label predicate in WHERE clause: WHERE n:Label
// ---------------------------------------------------------------------------

#[test]
fn test_where_label_predicate() {
    let db = chain_graph();
    let s = db.session();
    // WHERE n:Special filters to only nodes with the Special label
    let r = s
        .execute("MATCH (n) WHERE n:Special RETURN n.name AS name")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("E".into()));
}

#[test]
fn test_where_label_predicate_multi() {
    let db = chain_graph();
    let s = db.session();
    // WHERE n:Node:Special checks both labels (AND semantics)
    let r = s
        .execute("MATCH (n) WHERE n:Node:Special RETURN n.name AS name")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("E".into()));
}

#[test]
fn test_where_label_predicate_with_and() {
    let db = chain_graph();
    let s = db.session();
    // Combine label predicate with property filter
    let r = s
        .execute("MATCH (n) WHERE n:Node AND n.name = 'A' RETURN n.name AS name")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("A".into()));
}

// ---------------------------------------------------------------------------
// Undirected edge matching
// ---------------------------------------------------------------------------

#[test]
fn test_undirected_edge() {
    let db = chain_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (a:Node {name: 'B'})-[:LINK]-(b:Node) RETURN b.name AS name ORDER BY name")
        .unwrap();
    assert_eq!(r.rows().len(), 2, "Undirected LINK from B: A and C");
}

// ---------------------------------------------------------------------------
// MERGE with ON CREATE / ON MATCH
// ---------------------------------------------------------------------------

#[test]
fn test_merge_on_create_and_on_match() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "MERGE (p:Person {name: 'Django'}) \
             ON CREATE SET p.created = true \
             ON MATCH SET p.matched = true",
        )
        .unwrap();
    let r = session
        .execute("MATCH (p:Person {name: 'Django'}) RETURN p.created AS c, p.matched AS m")
        .unwrap();
    assert_eq!(r.rows()[0][0], Value::Bool(true));
    assert_eq!(r.rows()[0][1], Value::Null);

    session
        .execute(
            "MERGE (p:Person {name: 'Django'}) \
             ON CREATE SET p.created2 = true \
             ON MATCH SET p.matched = true",
        )
        .unwrap();
    let r = session
        .execute("MATCH (p:Person {name: 'Django'}) RETURN p.matched AS m")
        .unwrap();
    assert_eq!(r.rows()[0][0], Value::Bool(true));
}

// ---------------------------------------------------------------------------
// MERGE with edge pattern
// ---------------------------------------------------------------------------

#[test]
fn test_merge_edge_pattern() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .create_node_with_props(&["City"], [("name", Value::String("Amsterdam".into()))])
        .unwrap();
    session
        .create_node_with_props(&["City"], [("name", Value::String("Berlin".into()))])
        .unwrap();

    session
        .execute(
            "MATCH (a:City {name: 'Amsterdam'}), (b:City {name: 'Berlin'}) \
             MERGE (a)-[:ROUTE {distance: 650}]->(b)",
        )
        .unwrap();

    let r = session
        .execute("MATCH (:City)-[r:ROUTE]->(:City) RETURN r.distance AS d")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(650));
}

// ---------------------------------------------------------------------------
// Path length function
// ---------------------------------------------------------------------------

#[test]
fn test_path_length() {
    let db = chain_graph();
    let s = db.session();
    let r = s
        .execute(
            "MATCH p = (a:Node {name: 'A'})-[:LINK *1..4]->(b:Node {name: 'E'}) \
             RETURN length(p) AS len",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(4));
}

// ---------------------------------------------------------------------------
// WHERE on edge properties
// ---------------------------------------------------------------------------

#[test]
fn test_edge_property_filter() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let a = session
        .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
        .unwrap();
    let b = session
        .create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))])
        .unwrap();
    let e = session.create_edge(a, b, "RATED").unwrap();
    db.set_edge_property(e, "stars", Value::Int64(5)).unwrap();

    let r = session
        .execute("MATCH (a:Person)-[r:RATED]->(b:Person) WHERE r.stars >= 4 RETURN b.name AS name")
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Gus".into()));
}

// ---------------------------------------------------------------------------
// Multiple edge types in pattern (wildcard edge)
// ---------------------------------------------------------------------------

#[test]
fn test_wildcard_edge_type() {
    let db = chain_graph();
    let s = db.session();
    let r = s
        .execute("MATCH (a:Node {name: 'A'})-[]->(b:Node) RETURN b.name AS name ORDER BY name")
        .unwrap();
    assert_eq!(r.rows().len(), 2);
}

// ---------------------------------------------------------------------------
// Complex path with WHERE on intermediate nodes
// ---------------------------------------------------------------------------

#[test]
fn test_path_with_intermediate_filter() {
    let db = chain_graph();
    let s = db.session();
    let r = s
        .execute(
            "MATCH (a:Node {name: 'A'})-[:LINK]->(mid:Node)-[:LINK]->(c:Node) \
             WHERE mid.name = 'B' \
             RETURN c.name AS name",
        )
        .unwrap();
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("C".into()));
}

// ---------------------------------------------------------------------------
// DELETE pattern
// ---------------------------------------------------------------------------

#[test]
fn test_delete_node() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Temp"], [("x", Value::Int64(1))])
        .unwrap();
    session.execute("MATCH (t:Temp) DETACH DELETE t").unwrap();
    let r = session
        .execute("MATCH (t:Temp) RETURN count(t) AS cnt")
        .unwrap();
    assert_eq!(r.rows()[0][0], Value::Int64(0));
}

// ---------------------------------------------------------------------------
// SET with multiple properties
// ---------------------------------------------------------------------------

#[test]
fn test_set_multiple_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Item"], [("name", Value::String("widget".into()))])
        .unwrap();
    session
        .execute("MATCH (i:Item) SET i.price = 9.99, i.stock = 100")
        .unwrap();
    let r = session
        .execute("MATCH (i:Item) RETURN i.price AS p, i.stock AS s")
        .unwrap();
    assert_eq!(r.rows()[0][0], Value::Float64(9.99));
    assert_eq!(r.rows()[0][1], Value::Int64(100));
}
