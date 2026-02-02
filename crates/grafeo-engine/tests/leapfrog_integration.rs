//! Integration tests for Leapfrog TrieJoin operator.
//!
//! These tests verify that the LeapfrogJoinOperator works correctly
//! when used directly for multi-way joins and cyclic patterns.

use grafeo_common::types::LogicalType;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{LeapfrogJoinOperator, Operator};
use grafeo_core::execution::vector::ValueVector;
use grafeo_engine::GrafeoDB;

/// Creates a chunk with a single Int64 column.
fn create_int64_chunk(values: &[i64]) -> DataChunk {
    let mut col = ValueVector::with_type(LogicalType::Int64);
    for &v in values {
        col.push_int64(v);
    }
    DataChunk::new(vec![col])
}

/// Mock operator that returns a single chunk.
struct MockScanOperator {
    chunk: Option<DataChunk>,
    returned: bool,
}

impl MockScanOperator {
    fn new(chunk: DataChunk) -> Self {
        Self {
            chunk: Some(chunk),
            returned: false,
        }
    }
}

impl Operator for MockScanOperator {
    fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
        if self.returned {
            Ok(None)
        } else {
            self.returned = true;
            Ok(self.chunk.take())
        }
    }

    fn reset(&mut self) {
        self.returned = false;
    }

    fn name(&self) -> &'static str {
        "MockScan"
    }
}

#[test]
fn test_leapfrog_three_way_intersection() {
    // Three inputs with partial overlap:
    // Input 1: [1, 2, 3, 4, 5]
    // Input 2: [2, 3, 4, 5, 6]
    // Input 3: [3, 4, 5, 6, 7]
    // Intersection: [3, 4, 5]

    let chunk1 = create_int64_chunk(&[1, 2, 3, 4, 5]);
    let chunk2 = create_int64_chunk(&[2, 3, 4, 5, 6]);
    let chunk3 = create_int64_chunk(&[3, 4, 5, 6, 7]);

    let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
    let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));
    let op3: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk3));

    let mut leapfrog = LeapfrogJoinOperator::new(
        vec![op1, op2, op3],
        vec![vec![0], vec![0], vec![0]], // Join on first column of each
        vec![LogicalType::Int64, LogicalType::Int64, LogicalType::Int64],
        vec![(0, 0), (1, 0), (2, 0)], // Output all three columns
    );

    let mut results = Vec::new();
    while let Some(chunk) = leapfrog.next().unwrap() {
        for row in 0..chunk.row_count() {
            let v1 = chunk.column(0).unwrap().get_int64(row).unwrap();
            let v2 = chunk.column(1).unwrap().get_int64(row).unwrap();
            let v3 = chunk.column(2).unwrap().get_int64(row).unwrap();
            results.push((v1, v2, v3));
        }
    }

    // Should find 3 matches: (3,3,3), (4,4,4), (5,5,5)
    assert_eq!(results.len(), 3);
    assert!(results.contains(&(3, 3, 3)));
    assert!(results.contains(&(4, 4, 4)));
    assert!(results.contains(&(5, 5, 5)));
}

#[test]
fn test_leapfrog_with_duplicates() {
    // Input 1: [1, 1, 2, 2] - duplicates
    // Input 2: [1, 2, 2, 3] - duplicates
    // Expected: cross product of matching duplicates

    let chunk1 = create_int64_chunk(&[1, 1, 2, 2]);
    let chunk2 = create_int64_chunk(&[1, 2, 2, 3]);

    let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
    let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));

    let mut leapfrog = LeapfrogJoinOperator::new(
        vec![op1, op2],
        vec![vec![0], vec![0]],
        vec![LogicalType::Int64, LogicalType::Int64],
        vec![(0, 0), (1, 0)],
    );

    let mut count = 0;
    while let Some(chunk) = leapfrog.next().unwrap() {
        count += chunk.row_count();
    }

    // 1 appears 2x in input1, 1x in input2 = 2 matches
    // 2 appears 2x in input1, 2x in input2 = 4 matches
    // Total: 6 matches
    assert_eq!(count, 6);
}

#[test]
fn test_leapfrog_single_input() {
    // Edge case: single input (should just return all rows)
    let chunk = create_int64_chunk(&[1, 2, 3]);

    let op: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk));

    let mut leapfrog = LeapfrogJoinOperator::new(
        vec![op],
        vec![vec![0]],
        vec![LogicalType::Int64],
        vec![(0, 0)],
    );

    let mut count = 0;
    while let Some(chunk) = leapfrog.next().unwrap() {
        count += chunk.row_count();
    }

    assert_eq!(count, 3);
}

#[test]
fn test_leapfrog_empty_inputs() {
    // All inputs empty
    let chunk1 = create_int64_chunk(&[]);
    let chunk2 = create_int64_chunk(&[]);

    let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
    let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));

    let mut leapfrog = LeapfrogJoinOperator::new(
        vec![op1, op2],
        vec![vec![0], vec![0]],
        vec![LogicalType::Int64, LogicalType::Int64],
        vec![(0, 0), (1, 0)],
    );

    let result = leapfrog.next().unwrap();
    assert!(result.is_none());
}

#[test]
fn test_triangle_graph_manual() {
    // Manual test of triangle pattern.
    //
    // Triangle graph:
    //   1 -> 2
    //   2 -> 3
    //   3 -> 1
    //
    // This test creates a triangle and queries 2-hop paths to verify
    // the graph structure is set up correctly.

    // Create database and add triangle
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create nodes with properties
    session.execute("INSERT (:Node {id: 1})").unwrap();
    session.execute("INSERT (:Node {id: 2})").unwrap();
    session.execute("INSERT (:Node {id: 3})").unwrap();

    // Verify nodes exist
    let nodes = session.execute("MATCH (n:Node) RETURN COUNT(n)").unwrap();
    let node_count = nodes.iter().next().unwrap()[0].as_int64().unwrap();
    assert_eq!(node_count, 3, "Should have 3 nodes");

    // Create triangle edges using node IDs directly
    session
        .execute("MATCH (a:Node {id: 1}), (b:Node {id: 2}) CREATE (a)-[:EDGE]->(b)")
        .unwrap();
    session
        .execute("MATCH (a:Node {id: 2}), (b:Node {id: 3}) CREATE (a)-[:EDGE]->(b)")
        .unwrap();
    session
        .execute("MATCH (a:Node {id: 3}), (b:Node {id: 1}) CREATE (a)-[:EDGE]->(b)")
        .unwrap();

    // Verify edges exist - count 1-hop paths
    let edges = session
        .execute("MATCH (a:Node)-[:EDGE]->(b) RETURN COUNT(b)")
        .unwrap();
    let edge_count = edges.iter().next().unwrap()[0].as_int64().unwrap();
    assert_eq!(edge_count, 3, "Should have 3 edges");

    // Query all 2-hop paths
    let result = session
        .execute("MATCH (a:Node)-[:EDGE]->(b)-[:EDGE]->(c) RETURN COUNT(c)")
        .unwrap();

    // Should find 3 paths: 1->2->3, 2->3->1, 3->1->2
    let count = result.iter().next().unwrap()[0].as_int64().unwrap();
    assert_eq!(count, 3, "Should find 3 two-hop paths");
}

#[test]
fn test_leapfrog_large_intersection() {
    // Performance test with larger data
    let size = 1000;

    // Input 1: [0, 1, 2, ..., 999]
    // Input 2: [500, 501, ..., 1499]
    // Intersection: [500, 501, ..., 999] = 500 elements

    let chunk1 = create_int64_chunk(&(0..size).collect::<Vec<_>>());
    let chunk2 = create_int64_chunk(&((size / 2)..(size + size / 2)).collect::<Vec<_>>());

    let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
    let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));

    let mut leapfrog = LeapfrogJoinOperator::new(
        vec![op1, op2],
        vec![vec![0], vec![0]],
        vec![LogicalType::Int64, LogicalType::Int64],
        vec![(0, 0), (1, 0)],
    );

    let mut count = 0;
    while let Some(chunk) = leapfrog.next().unwrap() {
        count += chunk.row_count();
    }

    assert_eq!(count, 500);
}
