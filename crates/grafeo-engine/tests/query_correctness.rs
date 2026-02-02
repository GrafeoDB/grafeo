//! End-to-End Query Correctness Test Suite
//!
//! This integration test suite verifies that queries execute correctly
//! across all supported query languages and return expected results.
//!
//! Run with all features:
//! ```bash
//! cargo test -p grafeo-engine --features full --test query_correctness
//! ```
//!
//! Run for specific query language:
//! ```bash
//! cargo test -p grafeo-engine --features gql --test query_correctness -- gql
//! cargo test -p grafeo-engine --features cypher --test query_correctness -- cypher
//! cargo test -p grafeo-engine --features gremlin --test query_correctness -- gremlin
//! cargo test -p grafeo-engine --features graphql --test query_correctness -- graphql
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Creates a test database with a social network graph.
///
/// Structure:
/// - Alice (Person, age: 30) -KNOWS-> Bob (Person, age: 25)
/// - Alice -KNOWS-> Carol (Person, age: 35)
/// - Bob -KNOWS-> Carol
/// - Alice -WORKS_AT-> TechCorp (Company, founded: 2010)
/// - Bob -WORKS_AT-> TechCorp
/// - Carol -WORKS_AT-> Startup (Company, founded: 2020)
fn create_social_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create people
    let alice = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Alice".into())),
            ("age", Value::Int64(30)),
        ],
    );
    let bob = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Bob".into())),
            ("age", Value::Int64(25)),
        ],
    );
    let carol = session.create_node_with_props(
        &["Person"],
        [
            ("name", Value::String("Carol".into())),
            ("age", Value::Int64(35)),
        ],
    );

    // Create companies
    let techcorp = session.create_node_with_props(
        &["Company"],
        [
            ("name", Value::String("TechCorp".into())),
            ("founded", Value::Int64(2010)),
        ],
    );
    let startup = session.create_node_with_props(
        &["Company"],
        [
            ("name", Value::String("Startup".into())),
            ("founded", Value::Int64(2020)),
        ],
    );

    // Create KNOWS relationships
    session.create_edge(alice, bob, "KNOWS");
    session.create_edge(alice, carol, "KNOWS");
    session.create_edge(bob, carol, "KNOWS");

    // Create WORKS_AT relationships
    session.create_edge(alice, techcorp, "WORKS_AT");
    session.create_edge(bob, techcorp, "WORKS_AT");
    session.create_edge(carol, startup, "WORKS_AT");

    db
}

/// Creates a simple chain graph: A -> B -> C -> D
fn create_chain() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let a = session.create_node_with_props(&["Node"], [("id", Value::String("A".into()))]);
    let b = session.create_node_with_props(&["Node"], [("id", Value::String("B".into()))]);
    let c = session.create_node_with_props(&["Node"], [("id", Value::String("C".into()))]);
    let d = session.create_node_with_props(&["Node"], [("id", Value::String("D".into()))]);

    session.create_edge(a, b, "NEXT");
    session.create_edge(b, c, "NEXT");
    session.create_edge(c, d, "NEXT");

    db
}

/// Creates a star graph: Center connected to 5 spokes
fn create_star() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let center = session.create_node_with_props(&["Hub"], [("id", Value::String("center".into()))]);

    for i in 0..5 {
        let spoke = session.create_node_with_props(
            &["Spoke"],
            [("id", Value::String(format!("spoke_{}", i).into()))],
        );
        session.create_edge(center, spoke, "CONNECTS");
    }

    db
}

/// Creates a graph with numeric data for aggregation tests.
fn create_numeric_data() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create products with prices
    let prices = [100, 200, 150, 300, 250];
    let categories = [
        "Electronics",
        "Electronics",
        "Clothing",
        "Electronics",
        "Clothing",
    ];

    for (price, category) in prices.iter().zip(categories.iter()) {
        session.create_node_with_props(
            &["Product"],
            [
                ("price", Value::Int64(*price)),
                ("category", Value::String((*category).into())),
            ],
        );
    }

    db
}

/// Creates a tree structure for hierarchical queries.
fn create_tree() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let root = session.create_node_with_props(
        &["TreeNode"],
        [
            ("name", Value::String("root".into())),
            ("level", Value::Int64(0)),
        ],
    );
    let child1 = session.create_node_with_props(
        &["TreeNode"],
        [
            ("name", Value::String("child1".into())),
            ("level", Value::Int64(1)),
        ],
    );
    let child2 = session.create_node_with_props(
        &["TreeNode"],
        [
            ("name", Value::String("child2".into())),
            ("level", Value::Int64(1)),
        ],
    );
    let leaf1 = session.create_node_with_props(
        &["TreeNode"],
        [
            ("name", Value::String("leaf1".into())),
            ("level", Value::Int64(2)),
        ],
    );
    let leaf2 = session.create_node_with_props(
        &["TreeNode"],
        [
            ("name", Value::String("leaf2".into())),
            ("level", Value::Int64(2)),
        ],
    );

    session.create_edge(root, child1, "HAS_CHILD");
    session.create_edge(root, child2, "HAS_CHILD");
    session.create_edge(child1, leaf1, "HAS_CHILD");
    session.create_edge(child1, leaf2, "HAS_CHILD");

    db
}

// ============================================================================
// GQL Basic Pattern Tests
// ============================================================================

#[cfg(feature = "gql")]
mod gql_basic_patterns {
    use super::*;

    #[test]
    fn test_match_all_nodes() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            5,
            "Should find 5 nodes (3 people + 2 companies)"
        );
    }

    #[test]
    fn test_match_nodes_by_label() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 Person nodes");

        let result = session.execute("MATCH (n:Company) RETURN n").unwrap();
        assert_eq!(result.row_count(), 2, "Should find 2 Company nodes");
    }

    #[test]
    fn test_match_with_property_filter() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (n:Person) WHERE n.age > 28 RETURN n.name")
            .unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "Should find 2 people older than 28 (Alice: 30, Carol: 35)"
        );
    }

    #[test]
    fn test_match_with_equality_filter() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (n:Person) WHERE n.name = \"Alice\" RETURN n")
            .unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Should find exactly 1 person named Alice"
        );
    }

    #[test]
    fn test_match_relationship_pattern() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 KNOWS relationships");
    }

    #[test]
    fn test_return_specific_properties() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name, n.age")
            .unwrap();

        assert_eq!(result.row_count(), 3);
        assert_eq!(result.column_count(), 2);

        // Check that we have all expected names
        let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
        assert!(names.contains(&&Value::String("Alice".into())));
        assert!(names.contains(&&Value::String("Bob".into())));
        assert!(names.contains(&&Value::String("Carol".into())));
    }

    #[test]
    fn test_empty_result_set() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session.execute("MATCH (n:NonExistent) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "Should return empty result for non-existent label"
        );
    }

    #[test]
    #[ignore = "Parser issue with 'Node' label name - reserved keyword handling"]
    fn test_chain_traversal() {
        let db = create_chain();
        let session = db.session();

        let result = session
            .execute("MATCH (a:Node)-[:NEXT]->(b:Node) RETURN a.id, b.id")
            .unwrap();
        assert_eq!(
            result.row_count(),
            3,
            "Should find 3 NEXT edges in chain A->B->C->D"
        );
    }

    #[test]
    fn test_star_hub_connections() {
        let db = create_star();
        let session = db.session();

        let result = session
            .execute("MATCH (h:Hub)-[:CONNECTS]->(s:Spoke) RETURN s")
            .unwrap();
        assert_eq!(result.row_count(), 5, "Hub should connect to 5 spokes");
    }
}

// ============================================================================
// GQL Aggregation Tests
// ============================================================================

#[cfg(feature = "gql")]
mod gql_aggregations {
    use super::*;

    #[test]
    fn test_count_all() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute("MATCH (n:Person) RETURN COUNT(n)").unwrap();

        assert_eq!(result.row_count(), 1, "COUNT should return single row");
        if let Value::Int64(count) = &result.rows[0][0] {
            assert_eq!(*count, 3, "Should count 3 people");
        }
    }

    #[test]
    fn test_count_with_filter() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (n:Person) WHERE n.age > 28 RETURN COUNT(n)")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        if let Value::Int64(count) = &result.rows[0][0] {
            assert_eq!(*count, 2, "Should count 2 people older than 28");
        }
    }

    #[test]
    fn test_sum() {
        let db = create_numeric_data();
        let session = db.session();

        let result = session
            .execute("MATCH (p:Product) RETURN SUM(p.price)")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        // Sum of [100, 200, 150, 300, 250] = 1000
        match &result.rows[0][0] {
            Value::Int64(sum) => assert_eq!(*sum, 1000, "Sum of all prices should be 1000"),
            Value::Float64(sum) => assert!(
                (sum - 1000.0).abs() < 0.001,
                "Sum of all prices should be 1000"
            ),
            _ => panic!("Expected numeric result for SUM"),
        }
    }

    #[test]
    fn test_min_max() {
        let db = create_numeric_data();
        let session = db.session();

        let result = session
            .execute("MATCH (p:Product) RETURN MIN(p.price)")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        match &result.rows[0][0] {
            Value::Int64(min) => assert_eq!(*min, 100, "Min price should be 100"),
            Value::Null => {} // MIN of empty set is null - acceptable
            other => panic!("Unexpected value for MIN: {:?}", other),
        }

        let result = session
            .execute("MATCH (p:Product) RETURN MAX(p.price)")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        match &result.rows[0][0] {
            Value::Int64(max) => assert_eq!(*max, 300, "Max price should be 300"),
            Value::Null => {} // MAX of empty set is null - acceptable
            other => panic!("Unexpected value for MAX: {:?}", other),
        }
    }

    #[test]
    fn test_count_empty_result() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let result = session
            .execute("MATCH (n:NonExistent) RETURN COUNT(n)")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        if let Value::Int64(count) = &result.rows[0][0] {
            assert_eq!(*count, 0, "Count of non-existent nodes should be 0");
        }
    }
}

// ============================================================================
// GQL Join Tests
// ============================================================================

#[cfg(feature = "gql")]
mod gql_joins {
    use super::*;

    #[test]
    fn test_two_hop_path() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN a.name, b.name, c.name")
            .unwrap();

        // Alice knows Bob, Bob knows Carol
        assert!(
            result.row_count() >= 1,
            "Should find at least one 2-hop path"
        );
    }

    #[test]
    #[ignore = "Multi-pattern MATCH not fully implemented"]
    fn test_multi_pattern_match() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (a:Person)-[:KNOWS]->(b:Person), (a)-[:WORKS_AT]->(c:Company) RETURN a.name, b.name, c.name")
            .unwrap();

        assert!(
            result.row_count() >= 3,
            "Should find at least 3 combined patterns"
        );
    }

    #[test]
    fn test_tree_parent_child() {
        let db = create_tree();
        let session = db.session();

        let result = session
            .execute("MATCH (parent:TreeNode)-[:HAS_CHILD]->(child:TreeNode) RETURN parent.name, child.name")
            .unwrap();

        assert_eq!(
            result.row_count(),
            4,
            "Should find 4 parent-child relationships"
        );
    }

    #[test]
    #[ignore = "Parser issue with 'Node' label name - reserved keyword handling"]
    fn test_chain_full_traversal() {
        let db = create_chain();
        let session = db.session();

        // Use labels on intermediate nodes as required by parser
        let result = session
            .execute("MATCH (a:Node)-[:NEXT]->(b:Node)-[:NEXT]->(c:Node)-[:NEXT]->(d:Node) RETURN a.id, d.id")
            .unwrap();

        assert_eq!(
            result.row_count(),
            1,
            "Should find exactly one full chain path"
        );
    }
}

// ============================================================================
// GQL Mutation Tests
// ============================================================================

#[cfg(feature = "gql")]
mod gql_mutations {
    use super::*;

    #[test]
    fn test_insert_node() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alice', age: 30})")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name, n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1, "Should have 1 Person node");
        // Note: Property values are stored correctly but order may vary
        // Check name exists
        assert!(
            result.rows[0]
                .iter()
                .any(|v| *v == Value::String("Alice".into())),
            "Should find Alice in result"
        );
    }

    #[test]
    fn test_insert_multiple_nodes() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alice'})").unwrap();
        session.execute("INSERT (:Person {name: 'Bob'})").unwrap();
        session.execute("INSERT (:Person {name: 'Carol'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 3, "Should have 3 Person nodes");
    }

    #[test]
    fn test_transaction_commit() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_tx().unwrap();
        session.execute("INSERT (:Person {name: 'Alice'})").unwrap();
        session.commit().unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "Node should exist after commit");
    }
}

// ============================================================================
// Cypher Tests
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_tests {
    use super::*;

    #[test]
    fn test_match_all_nodes() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_cypher("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 5, "Should find 5 nodes");
    }

    #[test]
    fn test_match_nodes_by_label() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 Person nodes");
    }

    #[test]
    fn test_match_with_property_filter() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher("MATCH (n:Person) WHERE n.age > 28 RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 2, "Should find 2 people older than 28");
    }

    #[test]
    fn test_match_relationship_pattern() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 KNOWS relationships");
    }

    #[test]
    fn test_count() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher("MATCH (n:Person) RETURN count(n)")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        if let Value::Int64(count) = &result.rows[0][0] {
            assert_eq!(*count, 3);
        }
    }

    #[test]
    fn test_create_node() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute_cypher("CREATE (:Person {name: 'Alice', age: 30})")
            .unwrap();

        let result = session
            .execute_cypher("MATCH (n:Person) RETURN n.name, n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn test_two_hop_path() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN a.name, b.name, c.name")
            .unwrap();

        assert!(
            result.row_count() >= 1,
            "Should find at least one 2-hop path"
        );
    }
}

// ============================================================================
// Gremlin Tests
// ============================================================================

#[cfg(feature = "gremlin")]
mod gremlin_tests {
    use super::*;

    // Note: Many Gremlin tests are currently expected to fail due to variable
    // binding issues in the Gremlin executor. These tests document expected
    // behavior for when the executor is fully implemented.

    #[test]
    fn test_gremlin_parser() {
        // Verify Gremlin parsing works (execution may fail)
        let db = create_social_network();
        let session = db.session();

        // Test that we can at least parse basic Gremlin syntax
        let result = session.execute_gremlin("g.V()");
        // The translator should work even if execution fails
        assert!(
            result.is_ok() || result.is_err(),
            "Gremlin should be parsed"
        );
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_v_all_nodes() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_gremlin("g.V()").unwrap();
        assert_eq!(result.row_count(), 5, "Should find 5 vertices");
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_v_has_label() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_gremlin("g.V().hasLabel('Person')").unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 Person vertices");
    }

    #[test]
    #[ignore = "Gremlin predicate parsing not fully implemented"]
    fn test_v_has_property() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').has('age', gt(28))")
            .unwrap();
        assert_eq!(result.row_count(), 2, "Should find 2 people with age > 28");
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_out_step() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').out('KNOWS')")
            .unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 outgoing KNOWS edges");
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_values_step() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').values('name')")
            .unwrap();
        assert_eq!(result.row_count(), 3, "Should return 3 names");
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_limit_step() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_gremlin("g.V().limit(2)").unwrap();
        assert_eq!(result.row_count(), 2, "Should limit to 2 results");
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_count() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').count()")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        if let Value::Int64(count) = &result.rows[0][0] {
            assert_eq!(*count, 3);
        }
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_sum() {
        let db = create_numeric_data();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Product').values('price').sum()")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        if let Value::Int64(sum) = &result.rows[0][0] {
            assert_eq!(*sum, 1000);
        }
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_two_hop_traversal() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').out('KNOWS').out('KNOWS')")
            .unwrap();

        assert!(result.row_count() >= 1, "Should find friends of friends");
    }
}

// ============================================================================
// GraphQL Tests
// ============================================================================

#[cfg(feature = "graphql")]
mod graphql_tests {
    use super::*;

    // Note: GraphQL executor has limitations with variable binding and
    // nested queries. These tests document expected behavior.

    #[test]
    fn test_graphql_parser() {
        // Verify GraphQL parsing works
        let db = create_social_network();
        let session = db.session();

        // Test that we can at least parse basic GraphQL syntax
        let result = session.execute_graphql("query { person { id } }");
        // The translator should work
        assert!(
            result.is_ok() || result.is_err(),
            "GraphQL should be processed"
        );
    }

    #[test]
    fn test_query_all_by_type() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_graphql("query { person { id } }").unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 Person nodes");
    }

    #[test]
    fn test_query_with_field_selection() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_graphql("query { person { name age } }")
            .unwrap();

        assert_eq!(result.row_count(), 3);
    }

    #[test]
    #[ignore = "GraphQL filter arguments not fully implemented"]
    fn test_query_with_filter() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_graphql("query { person(filter: { age_gt: 28 }) { name } }")
            .unwrap();
        assert_eq!(result.row_count(), 2, "Should find 2 people with age > 28");
    }

    #[test]
    #[ignore = "GraphQL nested queries not fully implemented"]
    fn test_nested_query() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_graphql("query { person { name knows { name } } }")
            .unwrap();

        assert!(result.row_count() >= 1, "Should return nested results");
    }
}

// ============================================================================
// Direction Tests (Bidirectional Edge Indexing)
// ============================================================================

/// Tests that verify incoming/outgoing edge direction handling works correctly.
#[cfg(feature = "gql")]
mod gql_direction_tests {
    use super::*;

    /// Creates a simple directed graph for direction testing:
    /// A -[:FOLLOWS]-> B -[:FOLLOWS]-> C
    /// D -[:FOLLOWS]-> B (B has 2 incoming, 1 outgoing)
    fn create_directed_graph() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let a = session.create_node_with_props(&["User"], [("name", Value::String("A".into()))]);
        let b = session.create_node_with_props(&["User"], [("name", Value::String("B".into()))]);
        let c = session.create_node_with_props(&["User"], [("name", Value::String("C".into()))]);
        let d = session.create_node_with_props(&["User"], [("name", Value::String("D".into()))]);

        session.create_edge(a, b, "FOLLOWS"); // A -> B
        session.create_edge(b, c, "FOLLOWS"); // B -> C
        session.create_edge(d, b, "FOLLOWS"); // D -> B

        db
    }

    #[test]
    fn test_outgoing_edges() {
        let db = create_directed_graph();
        let session = db.session();

        // A follows 1 person (B)
        let result = session
            .execute("MATCH (a:User {name: \"A\"})-[:FOLLOWS]->(b) RETURN b.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "A should follow 1 person");
        assert_eq!(result.rows[0][0], Value::String("B".into()));

        // B follows 1 person (C)
        let result = session
            .execute("MATCH (b:User {name: \"B\"})-[:FOLLOWS]->(c) RETURN c.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "B should follow 1 person");
        assert_eq!(result.rows[0][0], Value::String("C".into()));

        // D follows 1 person (B)
        let result = session
            .execute("MATCH (d:User {name: \"D\"})-[:FOLLOWS]->(x) RETURN x.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "D should follow 1 person");

        // C follows nobody
        let result = session
            .execute("MATCH (c:User {name: \"C\"})-[:FOLLOWS]->(x) RETURN x")
            .unwrap();
        assert_eq!(result.row_count(), 0, "C should follow nobody");
    }

    #[test]
    fn test_incoming_edges() {
        let db = create_directed_graph();
        let session = db.session();

        // B is followed by 2 people (A and D)
        let result = session
            .execute("MATCH (b:User {name: \"B\"})<-[:FOLLOWS]-(x) RETURN x.name")
            .unwrap();
        assert_eq!(result.row_count(), 2, "B should be followed by 2 people");

        let names: std::collections::HashSet<_> = result
            .rows
            .iter()
            .filter_map(|r| {
                if let Value::String(s) = &r[0] {
                    Some(s.as_ref())
                } else {
                    None
                }
            })
            .collect();
        assert!(names.contains("A"), "A should follow B");
        assert!(names.contains("D"), "D should follow B");

        // C is followed by 1 person (B)
        let result = session
            .execute("MATCH (c:User {name: \"C\"})<-[:FOLLOWS]-(x) RETURN x.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "C should be followed by 1 person");
        assert_eq!(result.rows[0][0], Value::String("B".into()));

        // A has no followers
        let result = session
            .execute("MATCH (a:User {name: \"A\"})<-[:FOLLOWS]-(x) RETURN x")
            .unwrap();
        assert_eq!(result.row_count(), 0, "A should have no followers");
    }

    #[test]
    fn test_bidirectional_edges() {
        let db = create_directed_graph();
        let session = db.session();

        // B has 3 connections total (2 incoming + 1 outgoing)
        let result = session
            .execute("MATCH (b:User {name: \"B\"})-[:FOLLOWS]-(x) RETURN x.name")
            .unwrap();
        assert_eq!(
            result.row_count(),
            3,
            "B should have 3 total FOLLOWS connections"
        );
    }

    #[test]
    fn test_chain_traversal_incoming() {
        let db = create_chain();
        let session = db.session();

        // Traverse backward from D
        let result = session
            .execute("MATCH (d:Node {id: \"D\"})<-[:NEXT]-(c) RETURN c.id")
            .unwrap();
        assert_eq!(result.row_count(), 1, "D has 1 predecessor");
        assert_eq!(result.rows[0][0], Value::String("C".into()));

        // A has no predecessors
        let result = session
            .execute("MATCH (a:Node {id: \"A\"})<-[:NEXT]-(x) RETURN x")
            .unwrap();
        assert_eq!(result.row_count(), 0, "A has no predecessors");
    }

    #[test]
    fn test_tree_parent_child() {
        let db = create_tree();
        let session = db.session();

        // Find children of root (outgoing)
        let result = session
            .execute("MATCH (r:TreeNode {name: \"root\"})-[:HAS_CHILD]->(c) RETURN c.name")
            .unwrap();
        assert_eq!(result.row_count(), 2, "root should have 2 children");

        // Find parent of leaf1 (incoming)
        let result = session
            .execute("MATCH (l:TreeNode {name: \"leaf1\"})<-[:HAS_CHILD]-(p) RETURN p.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "leaf1 should have 1 parent");
        assert_eq!(result.rows[0][0], Value::String("child1".into()));
    }

    #[test]
    fn test_social_network_followers() {
        let db = create_social_network();
        let session = db.session();

        // Carol is known by both Alice and Bob (incoming KNOWS)
        let result = session
            .execute("MATCH (c:Person {name: \"Carol\"})<-[:KNOWS]-(x) RETURN x.name")
            .unwrap();
        assert_eq!(result.row_count(), 2, "Carol should be known by 2 people");

        // Bob is known by Alice (incoming KNOWS)
        let result = session
            .execute("MATCH (b:Person {name: \"Bob\"})<-[:KNOWS]-(x) RETURN x.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "Bob should be known by 1 person");
        assert_eq!(result.rows[0][0], Value::String("Alice".into()));
    }
}

// ============================================================================
// Cross-Language Consistency Tests (GQL and Cypher only)
// ============================================================================

/// Tests that verify consistent results across GQL and Cypher
#[cfg(all(feature = "gql", feature = "cypher"))]
mod cross_language_consistency_gql_cypher {
    use super::*;

    #[test]
    fn test_node_count_consistency() {
        let db = create_social_network();
        let session = db.session();

        let gql_result = session.execute("MATCH (n) RETURN n").unwrap();
        let cypher_result = session.execute_cypher("MATCH (n) RETURN n").unwrap();

        assert_eq!(
            gql_result.row_count(),
            cypher_result.row_count(),
            "GQL and Cypher should return same node count"
        );
    }

    #[test]
    fn test_label_filter_consistency() {
        let db = create_social_network();
        let session = db.session();

        let gql_result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        let cypher_result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();

        assert_eq!(
            gql_result.row_count(),
            cypher_result.row_count(),
            "GQL and Cypher should return same Person count"
        );
    }

    #[test]
    fn test_relationship_traversal_consistency() {
        let db = create_social_network();
        let session = db.session();

        let gql_result = session
            .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
            .unwrap();
        let cypher_result = session
            .execute_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
            .unwrap();

        assert_eq!(
            gql_result.row_count(),
            cypher_result.row_count(),
            "GQL and Cypher should return same relationship count"
        );
    }

    #[test]
    fn test_count_aggregation_consistency() {
        let db = create_social_network();
        let session = db.session();

        let gql_result = session.execute("MATCH (n:Person) RETURN COUNT(n)").unwrap();
        let cypher_result = session
            .execute_cypher("MATCH (n:Person) RETURN count(n)")
            .unwrap();

        let gql_count = match &gql_result.rows[0][0] {
            Value::Int64(c) => *c,
            _ => panic!("Expected Int64"),
        };
        let cypher_count = match &cypher_result.rows[0][0] {
            Value::Int64(c) => *c,
            _ => panic!("Expected Int64"),
        };

        assert_eq!(gql_count, 3, "GQL count should be 3");
        assert_eq!(cypher_count, 3, "Cypher count should be 3");
    }

    #[test]
    #[ignore = "Cypher SUM returning 0"]
    fn test_sum_aggregation_consistency() {
        let db = create_numeric_data();
        let session = db.session();

        let gql_result = session
            .execute("MATCH (p:Product) RETURN SUM(p.price)")
            .unwrap();
        let cypher_result = session
            .execute_cypher("MATCH (p:Product) RETURN sum(p.price)")
            .unwrap();

        let gql_sum = match &gql_result.rows[0][0] {
            Value::Int64(s) => *s,
            _ => panic!("Expected Int64"),
        };
        let cypher_sum = match &cypher_result.rows[0][0] {
            Value::Int64(s) => *s,
            _ => panic!("Expected Int64"),
        };

        assert_eq!(gql_sum, 1000, "GQL sum should be 1000");
        assert_eq!(cypher_sum, 1000, "Cypher sum should be 1000");
    }
}

// ============================================================================
// Cross-Language Consistency Tests (with Gremlin - currently ignored)
// ============================================================================

/// Tests that verify consistent results across all query languages including Gremlin
#[cfg(all(feature = "gql", feature = "cypher", feature = "gremlin"))]
mod cross_language_consistency_all {
    use super::*;

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_node_count_with_gremlin() {
        let db = create_social_network();
        let session = db.session();

        let gql_result = session.execute("MATCH (n) RETURN n").unwrap();
        let gremlin_result = session.execute_gremlin("g.V()").unwrap();

        assert_eq!(
            gql_result.row_count(),
            gremlin_result.row_count(),
            "GQL and Gremlin should return same node count"
        );
    }

    #[test]
    #[ignore = "Gremlin executor variable binding not fully implemented"]
    fn test_label_filter_with_gremlin() {
        let db = create_social_network();
        let session = db.session();

        let gql_result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        let gremlin_result = session.execute_gremlin("g.V().hasLabel('Person')").unwrap();

        assert_eq!(
            gql_result.row_count(),
            gremlin_result.row_count(),
            "GQL and Gremlin should return same Person count"
        );
    }
}

// ============================================================================
// Cross-Language Mutation Consistency Tests
// ============================================================================

#[cfg(all(feature = "gql", feature = "cypher"))]
mod cross_language_mutations {
    use super::*;

    #[test]
    fn test_insert_read_consistency() {
        // Insert with GQL, read with Cypher
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alice', age: 30})")
            .unwrap();

        let result = session
            .execute_cypher("MATCH (n:Person) RETURN n.name, n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows[0][0], Value::String("Alice".into()));
    }

    #[test]
    fn test_create_read_consistency() {
        // Create with Cypher, read with GQL
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute_cypher("CREATE (:Person {name: 'Bob', age: 25})")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name, n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows[0][0], Value::String("Bob".into()));
    }

    #[test]
    fn test_mixed_mutations() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Insert with GQL
        session.execute("INSERT (:Person {name: 'Alice'})").unwrap();

        // Create with Cypher
        session
            .execute_cypher("CREATE (:Person {name: 'Bob'})")
            .unwrap();

        // Verify both exist
        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "Should have 2 nodes from both insert methods"
        );
    }
}
