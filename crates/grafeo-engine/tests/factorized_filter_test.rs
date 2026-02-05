//! Test to debug factorized execution with filters

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn test_filter_in_multihop_query() {
    // Enable factorized execution (default) to test the bug
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create 5 Person nodes with id property
    let mut nodes = Vec::new();
    for i in 0..5 {
        let id = session.create_node_with_props(&["Person"], [("id", Value::Int64(i))]);
        nodes.push(id);
        println!("Created node {} with id={}", id, i);
    }

    // Create edges: node 0 -> nodes 1,2,3 (3 neighbors)
    // node 1 -> nodes 2,3,4 (3 neighbors)
    // node 2 -> nodes 3,4 (2 neighbors)
    session.create_edge(nodes[0], nodes[1], "KNOWS");
    session.create_edge(nodes[0], nodes[2], "KNOWS");
    session.create_edge(nodes[0], nodes[3], "KNOWS");
    session.create_edge(nodes[1], nodes[2], "KNOWS");
    session.create_edge(nodes[1], nodes[3], "KNOWS");
    session.create_edge(nodes[1], nodes[4], "KNOWS");
    session.create_edge(nodes[2], nodes[3], "KNOWS");
    session.create_edge(nodes[2], nodes[4], "KNOWS");

    println!("\n=== Testing 1-hop from node 0 ===");
    let result = session
        .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id")
        .unwrap();
    println!("1-hop from node 0: {} rows", result.row_count());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: b.id = {:?}", i, row);
    }
    assert_eq!(result.row_count(), 3, "Node 0 should have 3 outgoing edges");

    println!("\n=== Testing 2-hop from node 0 ===");
    let result = session
        .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a.id, b.id, c.id")
        .unwrap();
    println!("2-hop from node 0: {} rows", result.row_count());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }

    // Node 0 -> [1,2,3]
    // Node 1 -> [2,3,4] = 3 paths
    // Node 2 -> [3,4] = 2 paths
    // Node 3 -> [] = 0 paths
    // Total: 3 + 2 + 0 = 5 2-hop paths from node 0
    assert_eq!(result.row_count(), 5, "Node 0 should have 5 two-hop paths");

    println!("\n=== Testing all 2-hop paths (no filter) ===");
    let result_all = session
        .execute("MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a.id, b.id, c.id")
        .unwrap();
    println!("All 2-hop paths: {} rows", result_all.row_count());
    // This should be more than 5 since it includes paths starting from all nodes
}
