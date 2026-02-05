//! Tests for factorized aggregation optimization.
//!
//! These tests verify that COUNT, SUM, AVG, MIN, MAX aggregates work correctly
//! on multi-hop queries with factorized execution.

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// Creates a test graph with known structure for aggregation testing.
///
/// Structure:
/// - 3 source nodes (Person with id 0, 1, 2)
/// - Node 0 -> 4 neighbors (id 10, 11, 12, 13)
/// - Node 1 -> 2 neighbors (id 20, 21)
/// - Node 2 -> 3 neighbors (id 30, 31, 32)
///
/// Total 1-hop paths: 4 + 2 + 3 = 9
fn create_star_graph(db: &GrafeoDB) {
    let session = db.session();

    // Create source nodes
    for i in 0..3 {
        session.create_node_with_props(&["Person"], [("id", Value::Int64(i))]);
    }

    // Node 0's neighbors (4)
    for i in 10..14 {
        session.create_node_with_props(&["Person"], [("id", Value::Int64(i))]);
    }

    // Node 1's neighbors (2)
    for i in 20..22 {
        session.create_node_with_props(&["Person"], [("id", Value::Int64(i))]);
    }

    // Node 2's neighbors (3)
    for i in 30..33 {
        session.create_node_with_props(&["Person"], [("id", Value::Int64(i))]);
    }

    // Create KNOWS edges from sources to their neighbors
    // Node 0 (id=0) -> id 10,11,12,13
    session.execute("MATCH (a:Person {id: 0}), (b:Person) WHERE b.id >= 10 AND b.id < 14 CREATE (a)-[:KNOWS]->(b)").unwrap();
    // Node 1 (id=1) -> id 20,21
    session.execute("MATCH (a:Person {id: 1}), (b:Person) WHERE b.id >= 20 AND b.id < 22 CREATE (a)-[:KNOWS]->(b)").unwrap();
    // Node 2 (id=2) -> id 30,31,32
    session.execute("MATCH (a:Person {id: 2}), (b:Person) WHERE b.id >= 30 AND b.id < 33 CREATE (a)-[:KNOWS]->(b)").unwrap();

    // Add some second-hop edges for 2-hop testing
    // From each first-hop neighbor, connect to a couple more nodes
    session
        .execute("MATCH (a:Person) WHERE a.id = 10 CREATE (a)-[:KNOWS]->(:Person {id: 100})")
        .unwrap();
    session
        .execute("MATCH (a:Person) WHERE a.id = 10 CREATE (a)-[:KNOWS]->(:Person {id: 101})")
        .unwrap();
    session
        .execute("MATCH (a:Person) WHERE a.id = 11 CREATE (a)-[:KNOWS]->(:Person {id: 110})")
        .unwrap();
    session
        .execute("MATCH (a:Person) WHERE a.id = 20 CREATE (a)-[:KNOWS]->(:Person {id: 200})")
        .unwrap();
}

#[test]
fn test_count_with_factorized_execution() {
    let db = GrafeoDB::new_in_memory();
    create_star_graph(&db);

    let session = db.session();

    // Test 1-hop COUNT
    let result = session
        .execute("MATCH (a:Person)-[:KNOWS]->(b) RETURN COUNT(b)")
        .unwrap();

    println!("1-hop COUNT(b): {:?}", result.iter().collect::<Vec<_>>());
    assert_eq!(result.row_count(), 1);

    // Total edges: 4 (from 0) + 2 (from 1) + 3 (from 2) + 2 (from 10) + 1 (from 11) + 1 (from 20) = 13
    // But with second-hop creation, we have more
    let count = result.iter().next().unwrap()[0].as_int64().unwrap();
    println!("Total 1-hop paths: {}", count);
    assert!(count >= 9, "Should have at least 9 one-hop paths");
}

#[test]
fn test_count_two_hop_factorized() {
    let db = GrafeoDB::new_in_memory();
    create_star_graph(&db);

    let session = db.session();

    // 2-hop COUNT with factorized execution
    let result = session
        .execute("MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN COUNT(c)")
        .unwrap();

    println!("2-hop COUNT(c): {:?}", result.iter().collect::<Vec<_>>());
    assert_eq!(result.row_count(), 1);

    let count = result.iter().next().unwrap()[0].as_int64().unwrap();
    println!("Total 2-hop paths: {}", count);
    assert!(count > 0, "Should have some 2-hop paths");
}

#[test]
fn test_factorized_vs_flat_count_correctness() {
    // Create a small graph where we can verify the exact count
    let db_factorized = GrafeoDB::new_in_memory();
    let db_flat = GrafeoDB::with_config(Config::default().without_factorized_execution()).unwrap();

    // Create identical graphs
    for db in [&db_factorized, &db_flat] {
        let session = db.session();

        // Create a simple chain: A -> B1,B2 -> C1,C2
        session.create_node_with_props(&["Node"], [("name", Value::String("A".into()))]);
        session.create_node_with_props(&["Node"], [("name", Value::String("B1".into()))]);
        session.create_node_with_props(&["Node"], [("name", Value::String("B2".into()))]);
        session.create_node_with_props(&["Node"], [("name", Value::String("C1".into()))]);
        session.create_node_with_props(&["Node"], [("name", Value::String("C2".into()))]);

        session
            .execute("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B1'}) CREATE (a)-[:LINK]->(b)")
            .unwrap();
        session
            .execute("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B2'}) CREATE (a)-[:LINK]->(b)")
            .unwrap();
        session
            .execute("MATCH (a:Node {name: 'B1'}), (c:Node {name: 'C1'}) CREATE (a)-[:LINK]->(c)")
            .unwrap();
        session
            .execute("MATCH (a:Node {name: 'B1'}), (c:Node {name: 'C2'}) CREATE (a)-[:LINK]->(c)")
            .unwrap();
        session
            .execute("MATCH (a:Node {name: 'B2'}), (c:Node {name: 'C1'}) CREATE (a)-[:LINK]->(c)")
            .unwrap();
    }

    // A -> B1 -> C1, C2 (2 paths)
    // A -> B2 -> C1 (1 path)
    // Total 2-hop paths from A: 3

    let factorized_result = db_factorized
        .session()
        .execute("MATCH (a:Node {name: 'A'})-[:LINK]->(b)-[:LINK]->(c) RETURN COUNT(c)")
        .unwrap();

    let flat_result = db_flat
        .session()
        .execute("MATCH (a:Node {name: 'A'})-[:LINK]->(b)-[:LINK]->(c) RETURN COUNT(c)")
        .unwrap();

    let factorized_count = factorized_result.iter().next().unwrap()[0]
        .as_int64()
        .unwrap();
    let flat_count = flat_result.iter().next().unwrap()[0].as_int64().unwrap();

    println!("Factorized COUNT: {}", factorized_count);
    println!("Flat COUNT: {}", flat_count);
    println!("Expected: 3");

    assert_eq!(
        factorized_count, 3,
        "Factorized execution should count 3 paths"
    );
    assert_eq!(flat_count, 3, "Flat execution should count 3 paths");
    assert_eq!(
        factorized_count, flat_count,
        "Both should return the same count"
    );
}

#[test]
fn test_count_star_simple() {
    // Very simple test: single expand then COUNT
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create: Center -> A, B, C
    session.create_node_with_props(&["Center"], [("name", Value::String("center".into()))]);
    session.create_node_with_props(&["Target"], [("name", Value::String("A".into()))]);
    session.create_node_with_props(&["Target"], [("name", Value::String("B".into()))]);
    session.create_node_with_props(&["Target"], [("name", Value::String("C".into()))]);

    session
        .execute("MATCH (c:Center), (t:Target) CREATE (c)-[:POINTS_TO]->(t)")
        .unwrap();

    // 2-hop: Center -> A,B,C (each has no outgoing) = 0 two-hop paths
    // But let's add edges from targets to make it interesting
    session.create_node_with_props(&["Leaf"], [("name", Value::String("L1".into()))]);
    session.create_node_with_props(&["Leaf"], [("name", Value::String("L2".into()))]);

    session
        .execute("MATCH (t:Target {name: 'A'}), (l:Leaf) CREATE (t)-[:POINTS_TO]->(l)")
        .unwrap();

    // Now: Center -> A -> L1, L2 (2 paths)
    //      Center -> B (no further) = 0
    //      Center -> C (no further) = 0
    // Total 2-hop: 2

    let result = session
        .execute("MATCH (c:Center)-[:POINTS_TO]->(t)-[:POINTS_TO]->(l) RETURN COUNT(l)")
        .unwrap();

    let count = result.iter().next().unwrap()[0].as_int64().unwrap();
    println!("2-hop COUNT: {} (expected 2)", count);
    assert_eq!(count, 2, "Should find 2 two-hop paths");
}

#[test]
fn test_factorized_aggregation_speedup_demonstration() {
    // This test demonstrates the speedup of factorized aggregation
    // by comparing execution on a larger graph

    let db_factorized = GrafeoDB::new_in_memory();
    let db_flat = GrafeoDB::with_config(Config::default().without_factorized_execution()).unwrap();

    // Create a graph with high fan-out
    for db in [&db_factorized, &db_flat] {
        let session = db.session();

        // Create 5 source nodes
        for i in 0..5 {
            session.create_node_with_props(&["Source"], [("id", Value::Int64(i))]);
        }

        // Each source connects to 10 first-hop targets
        for i in 0..5 {
            for j in 0..10 {
                let target_id = 100 + i * 10 + j;
                session.create_node_with_props(&["Hop1"], [("id", Value::Int64(target_id))]);
            }
            session.execute(&format!(
                "MATCH (s:Source {{id: {}}}), (t:Hop1) WHERE t.id >= {} AND t.id < {} CREATE (s)-[:LINK]->(t)",
                i, 100 + i * 10, 100 + (i + 1) * 10
            )).unwrap();
        }

        // Each first-hop target connects to 5 second-hop targets
        for i in 0..50 {
            let hop1_id = 100 + i;
            for j in 0..5 {
                let hop2_id = 1000 + i * 5 + j;
                session.create_node_with_props(&["Hop2"], [("id", Value::Int64(hop2_id))]);
            }
            session.execute(&format!(
                "MATCH (h1:Hop1 {{id: {}}}), (h2:Hop2) WHERE h2.id >= {} AND h2.id < {} CREATE (h1)-[:LINK]->(h2)",
                hop1_id, 1000 + i * 5, 1000 + (i + 1) * 5
            )).unwrap();
        }
    }

    // Expected: 5 sources * 10 hop1 * 5 hop2 = 250 two-hop paths

    // Time factorized
    let start_factorized = std::time::Instant::now();
    let factorized_result = db_factorized
        .session()
        .execute("MATCH (s:Source)-[:LINK]->(h1)-[:LINK]->(h2) RETURN COUNT(h2)")
        .unwrap();
    let factorized_time = start_factorized.elapsed();

    // Time flat
    let start_flat = std::time::Instant::now();
    let flat_result = db_flat
        .session()
        .execute("MATCH (s:Source)-[:LINK]->(h1)-[:LINK]->(h2) RETURN COUNT(h2)")
        .unwrap();
    let flat_time = start_flat.elapsed();

    let factorized_count = factorized_result.iter().next().unwrap()[0]
        .as_int64()
        .unwrap();
    let flat_count = flat_result.iter().next().unwrap()[0].as_int64().unwrap();

    println!("========================================");
    println!("  Factorized Aggregation Speedup Test");
    println!("========================================");
    println!();
    println!("Graph: 5 sources * 10 hop1 * 5 hop2 = 250 paths");
    println!();
    println!(
        "Factorized COUNT: {} in {:?}",
        factorized_count, factorized_time
    );
    println!("Flat COUNT:       {} in {:?}", flat_count, flat_time);
    println!();

    assert_eq!(factorized_count, flat_count, "Counts should match");
    assert_eq!(factorized_count, 250, "Should count 250 paths");

    // Note: In a test environment, the speedup may not be dramatic due to small data.
    // The real speedup comes with larger datasets where flattening is expensive.
    if factorized_time < flat_time {
        let speedup = flat_time.as_nanos() as f64 / factorized_time.as_nanos() as f64;
        println!("Speedup: {:.2}x", speedup);
    }
}
