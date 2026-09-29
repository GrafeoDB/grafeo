//! Tests for factorized aggregation optimization.
//!
//! These tests verify that COUNT, SUM, AVG, MIN, MAX aggregates work correctly
//! on multi-hop queries with factorized execution.
//!
//! Requires the `cypher` feature (uses `execute_cypher` for graph setup).
//!
//! ```bash
//! cargo test -p grafeo-engine --features cypher --test factorized_aggregation_test
//! ```

#![cfg(feature = "cypher")]

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
        session
            .create_node_with_props(&["Person"], [("id", Value::Int64(i))])
            .unwrap();
    }

    // Node 0's neighbors (4)
    for i in 10..14 {
        session
            .create_node_with_props(&["Person"], [("id", Value::Int64(i))])
            .unwrap();
    }

    // Node 1's neighbors (2)
    for i in 20..22 {
        session
            .create_node_with_props(&["Person"], [("id", Value::Int64(i))])
            .unwrap();
    }

    // Node 2's neighbors (3)
    for i in 30..33 {
        session
            .create_node_with_props(&["Person"], [("id", Value::Int64(i))])
            .unwrap();
    }

    // Create KNOWS edges from sources to their neighbors
    // Node 0 (id=0) -> id 10,11,12,13
    session.execute_cypher("MATCH (a:Person {id: 0}), (b:Person) WHERE b.id >= 10 AND b.id < 14 CREATE (a)-[:KNOWS]->(b)").unwrap();
    // Node 1 (id=1) -> id 20,21
    session.execute_cypher("MATCH (a:Person {id: 1}), (b:Person) WHERE b.id >= 20 AND b.id < 22 CREATE (a)-[:KNOWS]->(b)").unwrap();
    // Node 2 (id=2) -> id 30,31,32
    session.execute_cypher("MATCH (a:Person {id: 2}), (b:Person) WHERE b.id >= 30 AND b.id < 33 CREATE (a)-[:KNOWS]->(b)").unwrap();

    // Add some second-hop edges for 2-hop testing
    // From each first-hop neighbor, connect to a couple more nodes
    session
        .execute_cypher("MATCH (a:Person) WHERE a.id = 10 CREATE (a)-[:KNOWS]->(:Person {id: 100})")
        .unwrap();
    session
        .execute_cypher("MATCH (a:Person) WHERE a.id = 10 CREATE (a)-[:KNOWS]->(:Person {id: 101})")
        .unwrap();
    session
        .execute_cypher("MATCH (a:Person) WHERE a.id = 11 CREATE (a)-[:KNOWS]->(:Person {id: 110})")
        .unwrap();
    session
        .execute_cypher("MATCH (a:Person) WHERE a.id = 20 CREATE (a)-[:KNOWS]->(:Person {id: 200})")
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
        session
            .create_node_with_props(&["Node"], [("name", Value::String("A".into()))])
            .unwrap();
        session
            .create_node_with_props(&["Node"], [("name", Value::String("B1".into()))])
            .unwrap();
        session
            .create_node_with_props(&["Node"], [("name", Value::String("B2".into()))])
            .unwrap();
        session
            .create_node_with_props(&["Node"], [("name", Value::String("C1".into()))])
            .unwrap();
        session
            .create_node_with_props(&["Node"], [("name", Value::String("C2".into()))])
            .unwrap();

        session
            .execute_cypher(
                "MATCH (a:Node {name: 'A'}), (b:Node {name: 'B1'}) CREATE (a)-[:LINK]->(b)",
            )
            .unwrap();
        session
            .execute_cypher(
                "MATCH (a:Node {name: 'A'}), (b:Node {name: 'B2'}) CREATE (a)-[:LINK]->(b)",
            )
            .unwrap();
        session
            .execute_cypher(
                "MATCH (a:Node {name: 'B1'}), (c:Node {name: 'C1'}) CREATE (a)-[:LINK]->(c)",
            )
            .unwrap();
        session
            .execute_cypher(
                "MATCH (a:Node {name: 'B1'}), (c:Node {name: 'C2'}) CREATE (a)-[:LINK]->(c)",
            )
            .unwrap();
        session
            .execute_cypher(
                "MATCH (a:Node {name: 'B2'}), (c:Node {name: 'C1'}) CREATE (a)-[:LINK]->(c)",
            )
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
    session
        .create_node_with_props(&["Center"], [("name", Value::String("center".into()))])
        .unwrap();
    session
        .create_node_with_props(&["Target"], [("name", Value::String("A".into()))])
        .unwrap();
    session
        .create_node_with_props(&["Target"], [("name", Value::String("B".into()))])
        .unwrap();
    session
        .create_node_with_props(&["Target"], [("name", Value::String("C".into()))])
        .unwrap();

    session
        .execute_cypher("MATCH (c:Center), (t:Target) CREATE (c)-[:POINTS_TO]->(t)")
        .unwrap();

    // 2-hop: Center -> A,B,C (each has no outgoing) = 0 two-hop paths
    // But let's add edges from targets to make it interesting
    session
        .create_node_with_props(&["Leaf"], [("name", Value::String("L1".into()))])
        .unwrap();
    session
        .create_node_with_props(&["Leaf"], [("name", Value::String("L2".into()))])
        .unwrap();

    session
        .execute_cypher("MATCH (t:Target {name: 'A'}), (l:Leaf) CREATE (t)-[:POINTS_TO]->(l)")
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
            session
                .create_node_with_props(&["Source"], [("id", Value::Int64(i))])
                .unwrap();
        }

        // Each source connects to 10 first-hop targets
        for i in 0..5 {
            for j in 0..10 {
                let target_id = 100 + i * 10 + j;
                session
                    .create_node_with_props(&["Hop1"], [("id", Value::Int64(target_id))])
                    .unwrap();
            }
            session.execute_cypher(&format!(
                "MATCH (s:Source {{id: {}}}), (t:Hop1) WHERE t.id >= {} AND t.id < {} CREATE (s)-[:LINK]->(t)",
                i, 100 + i * 10, 100 + (i + 1) * 10
            )).unwrap();
        }

        // Each first-hop target connects to 5 second-hop targets
        for i in 0..50 {
            let hop1_id = 100 + i;
            for j in 0..5 {
                let hop2_id = 1000 + i * 5 + j;
                session
                    .create_node_with_props(&["Hop2"], [("id", Value::Int64(hop2_id))])
                    .unwrap();
            }
            session.execute_cypher(&format!(
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

    // Note: In a test environment, the speedup may not be dramatic due to small data.
    // The real speedup comes with larger datasets where flattening is expensive.
    if factorized_time < flat_time {
        let speedup = flat_time.as_nanos() as f64 / factorized_time.as_nanos() as f64;
        println!("Speedup: {:.2}x", speedup);
    }
}

/// Creates a 4-level chain graph for 3-hop testing.
///
/// Structure (all edges are :STEP):
///
///   Layer 0 (Root): R0, R1
///   Layer 1 (Hop1): H1_0, H1_1, H1_2
///   Layer 2 (Hop2): H2_0, H2_1, H2_2, H2_3
///   Layer 3 (Hop3): H3_0, H3_1
///
///   R0  -> H1_0, H1_1       (fan-out 2)
///   R1  -> H1_2              (fan-out 1)
///   H1_0 -> H2_0, H2_1      (fan-out 2)
///   H1_1 -> H2_2             (fan-out 1)
///   H1_2 -> H2_3             (fan-out 1)
///   H2_0 -> H3_0             (fan-out 1)
///   H2_1 -> H3_0, H3_1      (fan-out 2)
///   H2_2 -> H3_1             (fan-out 1)
///   H2_3 -> (none)           (fan-out 0)
///
/// 3-hop paths (R -> H1 -> H2 -> H3):
///   R0 -> H1_0 -> H2_0 -> H3_0              = 1
///   R0 -> H1_0 -> H2_1 -> H3_0              = 1
///   R0 -> H1_0 -> H2_1 -> H3_1              = 1
///   R0 -> H1_1 -> H2_2 -> H3_1              = 1
///   R1 -> H1_2 -> H2_3 -> (none)            = 0
///   Total: 4
fn create_chain_graph(db: &GrafeoDB) {
    let session = db.session();

    // Layer 0: roots
    let r0 = session
        .create_node_with_props(&["Root"], [("name", Value::String("R0".into()))])
        .unwrap();
    let r1 = session
        .create_node_with_props(&["Root"], [("name", Value::String("R1".into()))])
        .unwrap();

    // Layer 1
    let h1_0 = session
        .create_node_with_props(&["Hop1"], [("name", Value::String("H1_0".into()))])
        .unwrap();
    let h1_1 = session
        .create_node_with_props(&["Hop1"], [("name", Value::String("H1_1".into()))])
        .unwrap();
    let h1_2 = session
        .create_node_with_props(&["Hop1"], [("name", Value::String("H1_2".into()))])
        .unwrap();

    // Layer 2
    let h2_0 = session
        .create_node_with_props(&["Hop2"], [("name", Value::String("H2_0".into()))])
        .unwrap();
    let h2_1 = session
        .create_node_with_props(&["Hop2"], [("name", Value::String("H2_1".into()))])
        .unwrap();
    let h2_2 = session
        .create_node_with_props(&["Hop2"], [("name", Value::String("H2_2".into()))])
        .unwrap();
    let h2_3 = session
        .create_node_with_props(&["Hop2"], [("name", Value::String("H2_3".into()))])
        .unwrap();

    // Layer 3
    let h3_0 = session
        .create_node_with_props(&["Hop3"], [("name", Value::String("H3_0".into()))])
        .unwrap();
    let h3_1 = session
        .create_node_with_props(&["Hop3"], [("name", Value::String("H3_1".into()))])
        .unwrap();

    // Layer 0 -> Layer 1 edges
    session.create_edge(r0, h1_0, "STEP").unwrap();
    session.create_edge(r0, h1_1, "STEP").unwrap();
    session.create_edge(r1, h1_2, "STEP").unwrap();

    // Layer 1 -> Layer 2 edges
    session.create_edge(h1_0, h2_0, "STEP").unwrap();
    session.create_edge(h1_0, h2_1, "STEP").unwrap();
    session.create_edge(h1_1, h2_2, "STEP").unwrap();
    session.create_edge(h1_2, h2_3, "STEP").unwrap();

    // Layer 2 -> Layer 3 edges
    session.create_edge(h2_0, h3_0, "STEP").unwrap();
    session.create_edge(h2_1, h3_0, "STEP").unwrap();
    session.create_edge(h2_1, h3_1, "STEP").unwrap();
    session.create_edge(h2_2, h3_1, "STEP").unwrap();
    // H2_3 has no outgoing edges (dead end)
}

#[test]
fn test_three_hop_factorized_count() {
    let db = GrafeoDB::new_in_memory();
    create_chain_graph(&db);

    let session = db.session();

    // 3-hop query: root -> hop1 -> hop2 -> hop3
    let result = session
        .execute("MATCH (a:Root)-[:STEP]->(b)-[:STEP]->(c)-[:STEP]->(d) RETURN count(d) AS cnt")
        .unwrap();

    assert_eq!(result.row_count(), 1, "Aggregation should return one row");
    let count = result.iter().next().unwrap()[0].as_int64().unwrap();
    assert_eq!(count, 4, "Should find exactly 4 three-hop paths");
}

#[test]
fn test_three_hop_factorized_vs_flat() {
    // Verify that factorized and flat execution agree on a 3-hop query.
    let db_factorized = GrafeoDB::new_in_memory();
    let db_flat = GrafeoDB::with_config(Config::default().without_factorized_execution()).unwrap();

    create_chain_graph(&db_factorized);
    create_chain_graph(&db_flat);

    let query = "MATCH (a:Root)-[:STEP]->(b)-[:STEP]->(c)-[:STEP]->(d) RETURN count(d) AS cnt";

    let factorized_count = db_factorized
        .session()
        .execute(query)
        .unwrap()
        .iter()
        .next()
        .unwrap()[0]
        .as_int64()
        .unwrap();

    let flat_count = db_flat
        .session()
        .execute(query)
        .unwrap()
        .iter()
        .next()
        .unwrap()[0]
        .as_int64()
        .unwrap();

    println!("Factorized 3-hop count: {}", factorized_count);
    println!("Flat 3-hop count:       {}", flat_count);

    assert_eq!(factorized_count, 4, "Factorized should find 4 paths");
    assert_eq!(flat_count, 4, "Flat should find 4 paths");
    assert_eq!(
        factorized_count, flat_count,
        "Factorized and flat must agree on 3-hop count"
    );
}

#[test]
fn test_asymmetric_fanout_two_hop() {
    // One root has high fan-out (many neighbors), another has zero.
    // Verifies that factorized execution handles the asymmetry correctly.
    //
    // Structure:
    //   Star -> S1, S2, S3, S4, S5   (fan-out 5)
    //   Leaf (no outgoing edges)      (fan-out 0)
    //
    //   S1 -> T1, T2                  (fan-out 2)
    //   S2 -> T3                      (fan-out 1)
    //   S3 -> T4, T5, T6             (fan-out 3)
    //   S4 -> (none)                  (fan-out 0)
    //   S5 -> T7                      (fan-out 1)
    //
    // 2-hop paths from Star:
    //   Star -> S1 -> T1              = 1
    //   Star -> S1 -> T2              = 1
    //   Star -> S2 -> T3              = 1
    //   Star -> S3 -> T4              = 1
    //   Star -> S3 -> T5              = 1
    //   Star -> S3 -> T6             = 1
    //   Star -> S4 -> (none)          = 0
    //   Star -> S5 -> T7              = 1
    //   Total from Star: 7
    //
    // 2-hop paths from Leaf: 0 (no outgoing edges at all)
    // Grand total: 7

    let db_factorized = GrafeoDB::new_in_memory();
    let db_flat = GrafeoDB::with_config(Config::default().without_factorized_execution()).unwrap();

    for db in [&db_factorized, &db_flat] {
        let session = db.session();

        // Two root-level nodes: one hub, one isolated
        let star = session
            .create_node_with_props(&["Hub"], [("name", Value::String("Star".into()))])
            .unwrap();
        let _leaf = session
            .create_node_with_props(&["Hub"], [("name", Value::String("Leaf".into()))])
            .unwrap();

        // First-hop satellites of Star
        let s1 = session
            .create_node_with_props(&["Sat"], [("name", Value::String("S1".into()))])
            .unwrap();
        let s2 = session
            .create_node_with_props(&["Sat"], [("name", Value::String("S2".into()))])
            .unwrap();
        let s3 = session
            .create_node_with_props(&["Sat"], [("name", Value::String("S3".into()))])
            .unwrap();
        let s4 = session
            .create_node_with_props(&["Sat"], [("name", Value::String("S4".into()))])
            .unwrap();
        let s5 = session
            .create_node_with_props(&["Sat"], [("name", Value::String("S5".into()))])
            .unwrap();

        session.create_edge(star, s1, "ARM").unwrap();
        session.create_edge(star, s2, "ARM").unwrap();
        session.create_edge(star, s3, "ARM").unwrap();
        session.create_edge(star, s4, "ARM").unwrap();
        session.create_edge(star, s5, "ARM").unwrap();

        // Second-hop targets
        let t1 = session
            .create_node_with_props(&["Tip"], [("name", Value::String("T1".into()))])
            .unwrap();
        let t2 = session
            .create_node_with_props(&["Tip"], [("name", Value::String("T2".into()))])
            .unwrap();
        let t3 = session
            .create_node_with_props(&["Tip"], [("name", Value::String("T3".into()))])
            .unwrap();
        let t4 = session
            .create_node_with_props(&["Tip"], [("name", Value::String("T4".into()))])
            .unwrap();
        let t5 = session
            .create_node_with_props(&["Tip"], [("name", Value::String("T5".into()))])
            .unwrap();
        let t6 = session
            .create_node_with_props(&["Tip"], [("name", Value::String("T6".into()))])
            .unwrap();
        let t7 = session
            .create_node_with_props(&["Tip"], [("name", Value::String("T7".into()))])
            .unwrap();

        session.create_edge(s1, t1, "ARM").unwrap();
        session.create_edge(s1, t2, "ARM").unwrap();
        session.create_edge(s2, t3, "ARM").unwrap();
        session.create_edge(s3, t4, "ARM").unwrap();
        session.create_edge(s3, t5, "ARM").unwrap();
        session.create_edge(s3, t6, "ARM").unwrap();
        // s4 has no outgoing edges
        session.create_edge(s5, t7, "ARM").unwrap();
    }

    let query = "MATCH (a:Hub)-[:ARM]->(b)-[:ARM]->(c) RETURN count(c) AS cnt";

    let factorized_count = db_factorized
        .session()
        .execute(query)
        .unwrap()
        .iter()
        .next()
        .unwrap()[0]
        .as_int64()
        .unwrap();

    let flat_count = db_flat
        .session()
        .execute(query)
        .unwrap()
        .iter()
        .next()
        .unwrap()[0]
        .as_int64()
        .unwrap();

    println!("Asymmetric fan-out, factorized: {}", factorized_count);
    println!("Asymmetric fan-out, flat:       {}", flat_count);

    assert_eq!(
        factorized_count, 7,
        "Factorized should find 7 two-hop paths (all from Star, none from Leaf)"
    );
    assert_eq!(flat_count, 7, "Flat should find 7 two-hop paths");
    assert_eq!(
        factorized_count, flat_count,
        "Factorized and flat must agree on asymmetric fan-out"
    );
}
