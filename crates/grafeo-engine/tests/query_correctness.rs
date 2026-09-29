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

/// Creates 3 Person + 2 Company nodes, 3 KNOWS + 3 WORKS_AT edges.
fn create_social_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create people
    let alix = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Alix".into())),
                ("age", Value::Int64(30)),
            ],
        )
        .unwrap();
    let gus = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Gus".into())),
                ("age", Value::Int64(25)),
            ],
        )
        .unwrap();
    let harm = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Harm".into())),
                ("age", Value::Int64(35)),
            ],
        )
        .unwrap();

    // Create companies
    let techcorp = session
        .create_node_with_props(
            &["Company"],
            [
                ("name", Value::String("TechCorp".into())),
                ("founded", Value::Int64(2010)),
            ],
        )
        .unwrap();
    let startup = session
        .create_node_with_props(
            &["Company"],
            [
                ("name", Value::String("Startup".into())),
                ("founded", Value::Int64(2020)),
            ],
        )
        .unwrap();

    // Create KNOWS relationships
    session.create_edge(alix, gus, "KNOWS").unwrap();
    session.create_edge(alix, harm, "KNOWS").unwrap();
    session.create_edge(gus, harm, "KNOWS").unwrap();

    // Create WORKS_AT relationships
    session.create_edge(alix, techcorp, "WORKS_AT").unwrap();
    session.create_edge(gus, techcorp, "WORKS_AT").unwrap();
    session.create_edge(harm, startup, "WORKS_AT").unwrap();

    db
}

/// Creates a simple chain graph: A -> B -> C -> D
fn create_chain() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let a = session
        .create_node_with_props(&["Node"], [("id", Value::String("A".into()))])
        .unwrap();
    let b = session
        .create_node_with_props(&["Node"], [("id", Value::String("B".into()))])
        .unwrap();
    let c = session
        .create_node_with_props(&["Node"], [("id", Value::String("C".into()))])
        .unwrap();
    let d = session
        .create_node_with_props(&["Node"], [("id", Value::String("D".into()))])
        .unwrap();

    session.create_edge(a, b, "NEXT").unwrap();
    session.create_edge(b, c, "NEXT").unwrap();
    session.create_edge(c, d, "NEXT").unwrap();

    db
}

/// Creates a star graph: Center connected to 5 spokes
fn create_star() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let center = session
        .create_node_with_props(&["Hub"], [("id", Value::String("center".into()))])
        .unwrap();

    for i in 0..5 {
        let spoke = session
            .create_node_with_props(
                &["Spoke"],
                [("id", Value::String(format!("spoke_{}", i).into()))],
            )
            .unwrap();
        session.create_edge(center, spoke, "CONNECTS").unwrap();
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
        session
            .create_node_with_props(
                &["Product"],
                [
                    ("price", Value::Int64(*price)),
                    ("category", Value::String((*category).into())),
                ],
            )
            .unwrap();
    }

    db
}

/// Creates a tree structure for hierarchical queries.
fn create_tree() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let root = session
        .create_node_with_props(
            &["TreeNode"],
            [
                ("name", Value::String("root".into())),
                ("level", Value::Int64(0)),
            ],
        )
        .unwrap();
    let child1 = session
        .create_node_with_props(
            &["TreeNode"],
            [
                ("name", Value::String("child1".into())),
                ("level", Value::Int64(1)),
            ],
        )
        .unwrap();
    let child2 = session
        .create_node_with_props(
            &["TreeNode"],
            [
                ("name", Value::String("child2".into())),
                ("level", Value::Int64(1)),
            ],
        )
        .unwrap();
    let leaf1 = session
        .create_node_with_props(
            &["TreeNode"],
            [
                ("name", Value::String("leaf1".into())),
                ("level", Value::Int64(2)),
            ],
        )
        .unwrap();
    let leaf2 = session
        .create_node_with_props(
            &["TreeNode"],
            [
                ("name", Value::String("leaf2".into())),
                ("level", Value::Int64(2)),
            ],
        )
        .unwrap();

    session.create_edge(root, child1, "HAS_CHILD").unwrap();
    session.create_edge(root, child2, "HAS_CHILD").unwrap();
    session.create_edge(child1, leaf1, "HAS_CHILD").unwrap();
    session.create_edge(child1, leaf2, "HAS_CHILD").unwrap();

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
            "Should find 2 people older than 28 (Alix: 30, Harm: 35)"
        );
    }

    #[test]
    fn test_match_with_equality_filter() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (n:Person) WHERE n.name = \"Alix\" RETURN n")
            .unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Should find exactly 1 person named Alix"
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
        let names: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
        assert!(names.contains(&&Value::String("Alix".into())));
        assert!(names.contains(&&Value::String("Gus".into())));
        assert!(names.contains(&&Value::String("Harm".into())));
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
        if let Value::Int64(count) = &result.rows()[0][0] {
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
        if let Value::Int64(count) = &result.rows()[0][0] {
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
        match &result.rows()[0][0] {
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
        match &result.rows()[0][0] {
            Value::Int64(min) => assert_eq!(*min, 100, "Min price should be 100"),
            Value::Null => {} // MIN of empty set is null - acceptable
            other => panic!("Unexpected value for MIN: {:?}", other),
        }

        let result = session
            .execute("MATCH (p:Product) RETURN MAX(p.price)")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        match &result.rows()[0][0] {
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
        if let Value::Int64(count) = &result.rows()[0][0] {
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

        // Only 2-hop KNOWS path: Alix->Gus->Harm
        assert_eq!(
            result.row_count(),
            1,
            "Should find exactly one 2-hop path: Alix->Gus->Harm"
        );
    }

    #[test]
    fn test_multi_pattern_match() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("MATCH (a:Person)-[:KNOWS]->(b:Person), (a)-[:WORKS_AT]->(c:Company) RETURN a.name, b.name, c.name")
            .unwrap();

        // (Alix,Gus,TechCorp), (Alix,Harm,TechCorp), (Gus,Harm,TechCorp)
        assert_eq!(
            result.row_count(),
            3,
            "Should find exactly 3 combined patterns"
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
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name, n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1, "Should have 1 Person node");
        // Note: Property values are stored correctly but order may vary
        // Check name exists
        assert!(
            result.rows()[0]
                .iter()
                .any(|v| *v == Value::String("Alix".into())),
            "Should find Alix in result"
        );
    }

    #[test]
    fn test_insert_multiple_nodes() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session.execute("INSERT (:Person {name: 'Harm'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 3, "Should have 3 Person nodes");
    }

    #[test]
    fn test_transaction_commit() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
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
        if let Value::Int64(count) = &result.rows()[0][0] {
            assert_eq!(*count, 3);
        }
    }

    #[test]
    fn test_create_node() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute_cypher("CREATE (:Person {name: 'Alix', age: 30})")
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

        // Only 2-hop KNOWS path: Alix->Gus->Harm
        assert_eq!(
            result.row_count(),
            1,
            "Should find exactly one 2-hop path: Alix->Gus->Harm"
        );
    }

    // === EXISTS Subquery Tests ===

    // Cypher EXISTS (GQL variant: test_exists_subquery_in_where in expression_and_projection.rs)
    #[test]
    fn test_cypher_exists_subquery_basic() {
        // Alix-[:KNOWS]->Gus, Alix-[:KNOWS]->Harm, Gus-[:KNOWS]->Harm
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n.name ORDER BY n.name",
            )
            .unwrap();

        // Alix knows Gus and Harm, Gus knows Harm
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn test_cypher_not_exists_subquery() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE NOT EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n.name",
            )
            .unwrap();

        // Harm has no outgoing KNOWS edges
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Harm".into()));
    }

    #[test]
    fn test_cypher_exists_with_edge_type_filter() {
        let db = create_social_network();
        let session = db.session();

        // No MANAGES edges exist in the social network
        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE EXISTS { MATCH (n)-[:MANAGES]->() } RETURN n.name",
            )
            .unwrap();

        assert_eq!(result.row_count(), 0);
    }

    #[test]
    fn test_cypher_exists_combined_with_predicate() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE EXISTS { MATCH (n)-[:KNOWS]->() } AND n.name = 'Alix' RETURN n.name",
            )
            .unwrap();

        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn test_cypher_exists_with_works_at() {
        let db = create_social_network();
        let session = db.session();

        // All three people have WORKS_AT edges
        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE EXISTS { MATCH (n)-[:WORKS_AT]->() } RETURN n.name ORDER BY n.name",
            )
            .unwrap();

        assert_eq!(result.row_count(), 3);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
        assert_eq!(result.rows()[2][0], Value::String("Harm".into()));
    }

    // Issue #173: EXISTS { MATCH ()-[:TYPE]->(m) } with correlated var on target side
    #[test]
    fn test_cypher_exists_target_side_correlation() {
        // Alix->Gus (KNOWS), Alix->Harm (KNOWS), Gus->Harm (KNOWS)
        // Gus and Harm have incoming KNOWS edges; Alix does not.
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE EXISTS { MATCH ()-[:KNOWS]->(n) } RETURN n.name ORDER BY n.name",
            )
            .unwrap();

        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
        assert_eq!(result.rows()[1][0], Value::String("Harm".into()));
    }

    #[test]
    fn test_cypher_not_exists_target_side_correlation() {
        // Alix has no incoming KNOWS edges.
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE NOT EXISTS { MATCH ()-[:KNOWS]->(n) } RETURN n.name",
            )
            .unwrap();

        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn test_anon_nodes_with_edge_variable() {
        let db = create_social_network();
        let session = db.session();

        // Issue 4: anonymous nodes with edge variable should work
        // Test type(r) function
        let result =
            session.execute_cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN type(r) AS t");
        assert!(
            result.is_ok(),
            "type(r) with anon nodes failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().row_count(), 3);

        // Test RETURN r directly (resolving edge as entity)
        let result = session.execute_cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN r");
        assert!(
            result.is_ok(),
            "RETURN r with anon nodes failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().row_count(), 3);

        // Test edge property access
        let result = session.execute_cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN r.since");
        assert!(
            result.is_ok(),
            "r.since with anon nodes failed: {:?}",
            result.err()
        );

        // Exact pattern from the bug report (no edge type filter)
        let result = session.execute_cypher("MATCH (:Person)-[r]->(:Person) RETURN r");
        assert!(
            result.is_ok(),
            "RETURN r (no type) with anon nodes failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().row_count(), 3);

        // WHERE clause referencing edge variable
        let result = session
            .execute_cypher("MATCH (:Person)-[r:KNOWS]->(:Person) WHERE r.since = 2020 RETURN r");
        // Note: since property only exists on some edges
        assert!(
            result.is_ok(),
            "WHERE r.since with anon nodes failed: {:?}",
            result.err()
        );
    }

    // Regression: GrafeoDB/discussions#155
    // Incoming edges with target node property filter must not return all edges.
    #[test]
    fn test_incoming_edge_target_node_property_filter() {
        let db = create_social_network();
        let session = db.session();

        // Harm is the target of 2 KNOWS edges (Alix->Harm, Gus->Harm).
        // The anonymous source node `()` should not broaden the result set.
        let result = session
            .execute_cypher(
                "MATCH ()-[r:KNOWS]->(o:Person {name: \"Harm\"}) RETURN count(r) AS cnt",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(2),
            "Only edges targeting Harm should be counted, not all KNOWS edges"
        );

        // Verify the symmetric outgoing case returns the same count.
        let result = session
            .execute_cypher(
                "MATCH (o:Person {name: \"Harm\"})<-[r:KNOWS]-() RETURN count(r) AS cnt",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(2),
            "Reversed arrow should match the same 2 edges"
        );

        // Without type filter: total incoming to Harm is still 2 KNOWS + 0 WORKS_AT = 2.
        let result = session
            .execute_cypher("MATCH ()-[r]->(o:Person {name: \"Harm\"}) RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(2),
            "Untyped edge should still filter by target node property"
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

        // Verify Gremlin parses and executes successfully
        let result = session.execute_gremlin("g.V()");
        assert!(
            result.is_ok(),
            "Gremlin g.V() should parse and execute: {result:?}"
        );
    }

    #[test]
    fn test_v_all_nodes() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_gremlin("g.V()").unwrap();
        assert_eq!(result.row_count(), 5, "Should find 5 vertices");
    }

    #[test]
    fn test_v_has_label() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_gremlin("g.V().hasLabel('Person')").unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 Person vertices");
    }

    #[test]
    fn test_v_has_property() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').has('age', gt(28))")
            .unwrap();
        assert_eq!(result.row_count(), 2, "Should find 2 people with age > 28");
    }

    #[test]
    fn test_out_step() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').out('KNOWS')")
            .unwrap();
        assert_eq!(result.row_count(), 3, "Should find 3 outgoing KNOWS edges");
    }

    #[test]
    fn test_values_step() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').values('name')")
            .unwrap();
        assert_eq!(result.row_count(), 3, "Should return 3 names");
    }

    #[test]
    fn test_limit_step() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_gremlin("g.V().limit(2)").unwrap();
        assert_eq!(result.row_count(), 2, "Should limit to 2 results");
    }

    #[test]
    fn test_count() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').count()")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        if let Value::Int64(count) = &result.rows()[0][0] {
            assert_eq!(*count, 3);
        }
    }

    #[test]
    fn test_sum() {
        let db = create_numeric_data();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Product').values('price').sum()")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        if let Value::Int64(sum) = &result.rows()[0][0] {
            assert_eq!(*sum, 1000);
        }
    }

    #[test]
    fn test_two_hop_traversal() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_gremlin("g.V().hasLabel('Person').out('KNOWS').out('KNOWS')")
            .unwrap();

        // Only path: Person.out(KNOWS).out(KNOWS) = Alix->Gus->Harm
        assert_eq!(
            result.row_count(),
            1,
            "Should find exactly one friend-of-friend: Harm via Gus"
        );
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

        // Verify GraphQL parses and executes successfully
        let result = session.execute_graphql("query { person { id } }");
        assert!(
            result.is_ok(),
            "GraphQL query should parse and execute: {result:?}"
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
    fn test_query_with_filter() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_graphql("query { person(filter: { age_gt: 28 }) { name } }")
            .unwrap();
        assert_eq!(result.row_count(), 2, "Should find 2 people with age > 28");
    }

    #[test]
    fn test_nested_query() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_graphql("query { person { name knows { name } } }")
            .unwrap();

        // 3 Person nodes, each with nested knows traversal
        assert_eq!(
            result.row_count(),
            3,
            "Should return one row per Person node"
        );
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

        let a = session
            .create_node_with_props(&["User"], [("name", Value::String("A".into()))])
            .unwrap();
        let b = session
            .create_node_with_props(&["User"], [("name", Value::String("B".into()))])
            .unwrap();
        let c = session
            .create_node_with_props(&["User"], [("name", Value::String("C".into()))])
            .unwrap();
        let d = session
            .create_node_with_props(&["User"], [("name", Value::String("D".into()))])
            .unwrap();

        session.create_edge(a, b, "FOLLOWS").unwrap(); // A -> B
        session.create_edge(b, c, "FOLLOWS").unwrap(); // B -> C
        session.create_edge(d, b, "FOLLOWS").unwrap(); // D -> B

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
        assert_eq!(result.rows()[0][0], Value::String("B".into()));

        // B follows 1 person (C)
        let result = session
            .execute("MATCH (b:User {name: \"B\"})-[:FOLLOWS]->(c) RETURN c.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "B should follow 1 person");
        assert_eq!(result.rows()[0][0], Value::String("C".into()));

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
            .rows()
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
        assert_eq!(result.rows()[0][0], Value::String("B".into()));

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
        assert_eq!(result.rows()[0][0], Value::String("C".into()));

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
        assert_eq!(result.rows()[0][0], Value::String("child1".into()));
    }

    #[test]
    fn test_social_network_followers() {
        let db = create_social_network();
        let session = db.session();

        // Harm is known by both Alix and Gus (incoming KNOWS)
        let result = session
            .execute("MATCH (c:Person {name: \"Harm\"})<-[:KNOWS]-(x) RETURN x.name")
            .unwrap();
        assert_eq!(result.row_count(), 2, "Harm should be known by 2 people");

        // Gus is known by Alix (incoming KNOWS)
        let result = session
            .execute("MATCH (b:Person {name: \"Gus\"})<-[:KNOWS]-(x) RETURN x.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "Gus should be known by 1 person");
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
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

        let gql_count = match &gql_result.rows()[0][0] {
            Value::Int64(c) => *c,
            _ => panic!("Expected Int64"),
        };
        let cypher_count = match &cypher_result.rows()[0][0] {
            Value::Int64(c) => *c,
            _ => panic!("Expected Int64"),
        };

        assert_eq!(gql_count, 3, "GQL count should be 3");
        assert_eq!(cypher_count, 3, "Cypher count should be 3");
    }

    #[test]
    fn test_sum_aggregation_consistency() {
        let db = create_numeric_data();
        let session = db.session();

        let gql_result = session
            .execute("MATCH (p:Product) RETURN SUM(p.price)")
            .unwrap();
        let cypher_result = session
            .execute_cypher("MATCH (p:Product) RETURN sum(p.price)")
            .unwrap();

        let gql_sum = match &gql_result.rows()[0][0] {
            Value::Int64(s) => *s,
            _ => panic!("Expected Int64"),
        };
        let cypher_sum = match &cypher_result.rows()[0][0] {
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
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        let result = session
            .execute_cypher("MATCH (n:Person) RETURN n.name, n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn test_create_read_consistency() {
        // Create with Cypher, read with GQL
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute_cypher("CREATE (:Person {name: 'Gus', age: 25})")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name, n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    }

    #[test]
    fn test_mixed_mutations() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Insert with GQL
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Create with Cypher
        session
            .execute_cypher("CREATE (:Person {name: 'Gus'})")
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

mod gql_in_operator {
    use super::*;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Harm', age: 35})")
            .unwrap();
        db
    }

    #[test]
    fn test_in_string_list() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.name IN ['Alix', 'Gus'] RETURN n.name ORDER BY n.name",
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn test_in_integer_list() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.age IN [25, 35] RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
        assert_eq!(result.rows()[1][0], Value::String("Harm".into()));
    }

    #[test]
    fn test_in_single_element() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.name IN ['Harm'] RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Harm".into()));
    }

    #[test]
    fn test_in_no_match() {
        let db = setup_db();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.name IN ['Dave', 'Eve'] RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 0);
    }

    #[test]
    fn test_string_with_escaped_quote() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(r"INSERT (:Person {name: 'O\'Brien'})")
            .unwrap();
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("O'Brien".into()));
    }
}

// ============================================================================
// Parameterized Query Tests
// ============================================================================

#[cfg(feature = "gql")]
mod parameterized_queries {
    use super::*;
    use std::collections::HashMap;

    fn setup_db() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Harm', age: 35})")
            .unwrap();
        db
    }

    #[test]
    fn test_execute_with_params_string() {
        let db = setup_db();
        let session = db.session();

        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::String("Alix".into()));

        let result = session
            .execute_with_params(
                "MATCH (n:Person) WHERE n.name = $name RETURN n.name",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn test_execute_with_params_integer() {
        let db = setup_db();
        let session = db.session();

        let mut params = HashMap::new();
        params.insert("min_age".to_string(), Value::Int64(28));

        let result = session
            .execute_with_params(
                "MATCH (n:Person) WHERE n.age > $min_age RETURN n.name ORDER BY n.name",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[1][0], Value::String("Harm".into()));
    }

    #[test]
    fn test_execute_with_multiple_params() {
        let db = setup_db();
        let session = db.session();

        let mut params = HashMap::new();
        params.insert("min_age".to_string(), Value::Int64(24));
        params.insert("max_age".to_string(), Value::Int64(31));

        let result = session
            .execute_with_params(
                "MATCH (n:Person) WHERE n.age >= $min_age AND n.age <= $max_age RETURN n.name ORDER BY n.name",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn test_db_level_execute_with_params() {
        let db = setup_db();

        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::String("Gus".into()));

        let result = db
            .execute_with_params("MATCH (n:Person) WHERE n.name = $name RETURN n.age", params)
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(25));
    }
}

// ============================================================================
// Language-specific execute methods on GrafeoDB
// ============================================================================

#[cfg(feature = "gremlin")]
mod gremlin_db_execute {
    use super::*;

    #[test]
    fn test_db_execute_gremlin() {
        let db = create_social_network();
        let result = db.execute_gremlin("g.V().hasLabel('Person')").unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_db_execute_gremlin_with_params() {
        let db = create_social_network();
        let params = std::collections::HashMap::new();
        let result = db
            .execute_gremlin_with_params("g.V().hasLabel('Person').count()", params)
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }
}

#[cfg(feature = "graphql")]
mod graphql_db_execute {
    use super::*;

    #[test]
    fn test_db_execute_graphql() {
        let db = create_social_network();
        let result = db.execute_graphql("query { person { name } }").unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_db_execute_graphql_with_params() {
        let db = create_social_network();
        let params = std::collections::HashMap::new();
        let result = db
            .execute_graphql_with_params("query { person { name } }", params)
            .unwrap();
        assert_eq!(result.row_count(), 3);
    }
}

#[cfg(feature = "cypher")]
mod cypher_db_execute {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_db_execute_cypher() {
        let db = create_social_network();
        let result = db.execute_cypher("MATCH (p:Person) RETURN p").unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_db_execute_cypher_with_params() {
        let db = create_social_network();
        let mut params = HashMap::new();
        params.insert("name".to_string(), Value::String("Alix".into()));

        let result = db
            .execute_cypher_with_params(
                "MATCH (p:Person) WHERE p.name = $name RETURN p.name",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn test_cypher_exists_as_alias() {
        let db = create_social_network();
        let result = db
            .execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' RETURN count(n) as exists")
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn test_cypher_multiple_match_clauses() {
        let db = create_social_network();
        let result = db
            .execute_cypher(
                "MATCH (a:Person) WHERE a.name = 'Alix' \
                 MATCH (b:Person) WHERE b.name = 'Gus' \
                 RETURN a.name, b.name",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[0][1], Value::String("Gus".into()));
    }

    #[test]
    fn test_cypher_multiple_match_with_create() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("id", Value::String("src".into())),
                    ("name", Value::String("Src".into())),
                ],
            )
            .unwrap();
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("id", Value::String("dst".into())),
                    ("name", Value::String("Dst".into())),
                ],
            )
            .unwrap();

        let mut params = HashMap::new();
        params.insert("src_id".to_string(), Value::String("src".into()));
        params.insert("dst_id".to_string(), Value::String("dst".into()));

        let result = db
            .execute_cypher_with_params(
                "MATCH (src:Person) WHERE src.id = $src_id \
                 MATCH (dst:Person) WHERE dst.id = $dst_id \
                 CREATE (src)-[r:KNOWS]->(dst) \
                 RETURN src.name, dst.name",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn test_cypher_merge_relationship() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("id", Value::String("a".into())),
                    ("name", Value::String("Alix".into())),
                ],
            )
            .unwrap();
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("id", Value::String("b".into())),
                    ("name", Value::String("Gus".into())),
                ],
            )
            .unwrap();

        // MERGE should create the relationship since it doesn't exist
        let result = db
            .execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 MERGE (a)-[r:KNOWS]->(b) \
                 RETURN r",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);

        // Running MERGE again should return the same relationship (idempotent)
        let result2 = db
            .execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 MERGE (a)-[r:KNOWS]->(b) \
                 RETURN r",
            )
            .unwrap();
        assert_eq!(result2.row_count(), 1);
    }

    #[test]
    fn test_cypher_merge_relationship_with_properties() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
            .unwrap();
        session
            .create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))])
            .unwrap();

        // MERGE with match properties
        let result = db
            .execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 MERGE (a)-[r:KNOWS {id: 'edge1'}]->(b) \
                 RETURN r",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);

        // Second MERGE with same id should not create duplicate
        let result2 = db
            .execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 MERGE (a)-[r:KNOWS {id: 'edge1'}]->(b) \
                 RETURN r",
            )
            .unwrap();
        assert_eq!(result2.row_count(), 1);
    }

    #[test]
    fn test_cypher_merge_relationship_then_set() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
            .unwrap();
        session
            .create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))])
            .unwrap();

        // MERGE relationship then SET a property on it
        let result = db
            .execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 MERGE (a)-[r:KNOWS]->(b) \
                 SET r.weight = 5 \
                 RETURN r",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn test_cypher_multi_match_with_merge_and_set() {
        // This is the Deriva-style query pattern
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .create_node_with_props(&["Node"], [("id", Value::String("src".into()))])
            .unwrap();
        session
            .create_node_with_props(&["Node"], [("id", Value::String("dst".into()))])
            .unwrap();

        let mut params = HashMap::new();
        params.insert("src_id".to_string(), Value::String("src".into()));
        params.insert("dst_id".to_string(), Value::String("dst".into()));
        params.insert("edge_id".to_string(), Value::String("e1".into()));
        params.insert("props".to_string(), Value::String("{}".into()));

        let result = db
            .execute_cypher_with_params(
                "MATCH (src:Node) WHERE src.id = $src_id \
                 MATCH (dst:Node) WHERE dst.id = $dst_id \
                 MERGE (src)-[r:INHERITS {id: $edge_id}]->(dst) \
                 SET r.properties_json = $props \
                 RETURN r",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    // === Cypher SET tests ===

    #[test]
    fn test_cypher_set_node_property() {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (n:Person {name: 'Alix'})")
            .unwrap();

        db.execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' SET n.age = 30")
            .unwrap();

        let result = db
            .execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        let age = &result.rows()[0][0];
        assert_eq!(age, &Value::Int64(30));
    }

    #[test]
    fn test_cypher_set_labels() {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (n:Person {name: 'Alix'})")
            .unwrap();

        db.execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' SET n:Employee")
            .unwrap();

        // Should now match as Employee
        let result = db
            .execute_cypher("MATCH (n:Employee) RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        let name = &result.rows()[0][0];
        assert_eq!(name, &Value::String("Alix".into()));
    }

    // === Cypher DELETE tests ===

    #[test]
    fn test_cypher_delete_node() {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (n:Temp {name: 'ToDelete'})")
            .unwrap();

        let before = db.execute_cypher("MATCH (n:Temp) RETURN n").unwrap();
        assert_eq!(before.row_count(), 1);

        db.execute_cypher("MATCH (n:Temp) WHERE n.name = 'ToDelete' DETACH DELETE n")
            .unwrap();

        let after = db.execute_cypher("MATCH (n:Temp) RETURN n").unwrap();
        assert_eq!(after.row_count(), 0);
    }

    #[test]
    fn test_cypher_detach_delete_with_edges() {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})")
            .unwrap();
        db.execute_cypher(
            "MATCH (a:Person), (b:Person) WHERE a.name = 'Alix' AND b.name = 'Gus' \
             CREATE (a)-[:KNOWS]->(b)",
        )
        .unwrap();

        // DETACH DELETE removes the node and its connected edges
        db.execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' DETACH DELETE n")
            .unwrap();

        let result = db.execute_cypher("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
        let name = &result.rows()[0][0];
        assert_eq!(name, &Value::String("Gus".into()));
    }

    #[test]
    fn test_cypher_delete_edge_variable() {
        // Regression: MATCH (a)-[e:KNOWS]->(b) DELETE e must emit DeleteEdgeOp,
        // not DeleteNodeOp, so the edge (not the node) is removed.
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})")
            .unwrap();
        db.execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             CREATE (a)-[:KNOWS]->(b)",
        )
        .unwrap();

        // Verify edge exists
        let before = db
            .execute_cypher("MATCH (a)-[e:KNOWS]->(b) RETURN e")
            .unwrap();
        assert_eq!(before.row_count(), 1, "Edge should exist before delete");

        // Delete the edge by variable
        db.execute_cypher("MATCH (a)-[e:KNOWS]->(b) DELETE e")
            .unwrap();

        // Edge should be gone; both nodes should remain
        let edges_after = db
            .execute_cypher("MATCH (a)-[e:KNOWS]->(b) RETURN e")
            .unwrap();
        assert_eq!(edges_after.row_count(), 0, "Edge should be deleted");

        let nodes_after = db.execute_cypher("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(nodes_after.row_count(), 2, "Both nodes should remain");
    }

    // === Cypher REMOVE tests ===

    #[test]
    fn test_cypher_remove_property() {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (n:Person {name: 'Alix', age: 30})")
            .unwrap();

        db.execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' REMOVE n.age")
            .unwrap();

        let result = db
            .execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        let age = &result.rows()[0][0];
        assert_eq!(age, &Value::Null);
    }

    #[test]
    fn test_cypher_remove_label() {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (n:Person:Employee {name: 'Alix'})")
            .unwrap();

        db.execute_cypher("MATCH (n:Person) WHERE n.name = 'Alix' REMOVE n:Employee")
            .unwrap();

        // Should no longer match as Employee
        let result = db.execute_cypher("MATCH (n:Employee) RETURN n").unwrap();
        assert_eq!(result.row_count(), 0);

        // Should still match as Person
        let result = db.execute_cypher("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
    }

    // === Binary expression in RETURN ===

    #[test]
    fn test_cypher_return_count_gt_zero() {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (n:Person {name: 'Alix'})")
            .unwrap();
        db.execute_cypher("CREATE (n:Person {name: 'Gus'})")
            .unwrap();

        let result = db
            .execute_cypher("MATCH (n:Person) RETURN count(n) > 0 AS exists")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        let exists = &result.rows()[0][0];
        assert_eq!(exists, &Value::Bool(true));
    }

    #[test]
    fn test_cypher_return_count_gt_zero_empty() {
        let db = GrafeoDB::new_in_memory();

        let result = db
            .execute_cypher("MATCH (n:Ghost) RETURN count(n) > 0 AS exists")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        let exists = &result.rows()[0][0];
        assert_eq!(exists, &Value::Bool(false));
    }
}

#[cfg(feature = "sql-pgq")]
mod sql_pgq_db_execute {
    use super::*;

    #[test]
    fn test_db_execute_sql() {
        let db = create_social_network();
        let result = db
            .execute_sql(
                "SELECT p.name FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))",
            )
            .unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_db_execute_sql_with_params() {
        let db = create_social_network();
        let params = std::collections::HashMap::new();
        let result = db
            .execute_sql_with_params(
                "SELECT p.name FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 3);
    }
}

// ============================================================================
// PROFILE Statement Tests
// ============================================================================

#[cfg(feature = "gql")]
mod profile_tests {
    use grafeo_common::types::Value;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn gql_profile_returns_single_profile_column() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();

        let result = session
            .execute("PROFILE MATCH (n:Person) RETURN n.name")
            .unwrap();

        // PROFILE returns a single "profile" column with one row
        assert_eq!(result.columns, vec!["profile"]);
        assert_eq!(result.row_count(), 1);

        // The value is a string containing operator names and metrics
        let profile_text = match &result.rows()[0][0] {
            Value::String(s) => s.to_string(),
            other => panic!("Expected String, got {other:?}"),
        };

        // Verify the profile output contains expected operator names.
        // The optimizer may replace Return with Project, so accept either.
        assert!(
            profile_text.contains("Return") || profile_text.contains("Project"),
            "Profile should contain Return or Project operator, got: {profile_text}"
        );
        assert!(
            profile_text.contains("Scan"),
            "Profile should contain a scan operator, got: {profile_text}"
        );
        assert!(
            profile_text.contains("rows="),
            "Profile should contain row counts, got: {profile_text}"
        );
        assert!(
            profile_text.contains("time="),
            "Profile should contain timing, got: {profile_text}"
        );
        assert!(
            profile_text.contains("Total time:"),
            "Profile should contain total time, got: {profile_text}"
        );
    }

    #[test]
    fn gql_profile_row_counts_are_accurate() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for i in 0..5 {
            session
                .execute(&format!("INSERT (:Person {{name: 'P{i}', age: {i}0}})"))
                .unwrap();
        }

        let result = session
            .execute("PROFILE MATCH (n:Person) RETURN n.name")
            .unwrap();

        let profile_text = match &result.rows()[0][0] {
            Value::String(s) => s.to_string(),
            other => panic!("Expected String, got {other:?}"),
        };

        // The Return operator should show rows=5
        assert!(
            profile_text.contains("rows=5"),
            "Profile should show 5 rows for 5 Person nodes, got: {profile_text}"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn cypher_profile_works() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session
            .execute_cypher("PROFILE MATCH (n:Person) RETURN n.name")
            .unwrap();

        assert_eq!(result.columns, vec!["profile"]);
        assert_eq!(result.row_count(), 1);

        let profile_text = match &result.rows()[0][0] {
            Value::String(s) => s.to_string(),
            other => panic!("Expected String, got {other:?}"),
        };
        assert!(
            profile_text.contains("rows="),
            "Cypher PROFILE should work, got: {profile_text}"
        );
    }
}

// ============================================================================
// SQL/PGQ Cross-Language Correctness
// ============================================================================

#[cfg(feature = "sql-pgq")]
mod gql_conformance_edge_cases {
    use super::*;

    /// FILTER WHERE is an ISO GQL synonym for WHERE.
    #[test]
    fn filter_where_clause() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute(
                "MATCH (a:Person)-[:KNOWS]->(b:Person) \
                 FILTER WHERE a.age > 28 \
                 RETURN a.name AS name ORDER BY name",
            )
            .unwrap();

        // Alix (30) satisfies the filter
        assert!(result.row_count() > 0);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    /// Post-edge quantifiers: ->{min,max}
    #[test]
    fn post_edge_quantifier_bounded() {
        let db = create_chain();
        let session = db.session();

        // A ->{1,2} should match 1-hop and 2-hop paths from each node
        let result = session
            .execute(
                "MATCH (a:Node)-[:NEXT]->{1,2}(b:Node) \
                 RETURN a.id AS src, b.id AS dst ORDER BY src, dst",
            )
            .unwrap();

        // 1-hop: A->B, B->C, C->D (3)
        // 2-hop: A->C, B->D (2)
        // Total: 5
        assert_eq!(result.row_count(), 5);
    }

    /// Post-edge quantifier: ->+ (one or more)
    #[test]
    fn post_edge_quantifier_plus() {
        let db = create_chain();
        let session = db.session();

        let result = session
            .execute(
                "MATCH (a:Node {id: 'A'})-[:NEXT]->{1,}(b:Node) \
                 RETURN b.id AS dst ORDER BY dst",
            )
            .unwrap();

        // From A: B (1-hop), C (2-hop), D (3-hop) = 3 results
        assert_eq!(result.row_count(), 3);
    }

    /// SELECT ... FROM GRAPH_TABLE ... MATCH is a GQL ISO construct.
    #[test]
    fn select_from_match_statement() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute("SELECT a.name AS name FROM MATCH (a:Person) ORDER BY name")
            .unwrap();

        assert_eq!(result.row_count(), 3);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    /// nullIf function works in GQL RETURN.
    #[test]
    fn nullif_in_return() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute(
                "MATCH (a:Person) \
                 RETURN a.name AS name, nullIf(a.name, 'Gus') AS filtered \
                 ORDER BY name",
            )
            .unwrap();

        // Alix: filtered = 'Alix'
        assert_eq!(result.rows()[0][1], Value::String("Alix".into()));
        // Gus: filtered = NULL
        assert!(result.rows()[1][1].is_null());
        // Harm: filtered = 'Harm'
        assert_eq!(result.rows()[2][1], Value::String("Harm".into()));
    }
}

mod sql_pgq_correctness {
    use super::*;

    /// Verify SQL/PGQ and GQL return the same results for equivalent queries.
    #[test]
    fn sql_pgq_matches_gql_for_node_query() {
        let db = create_social_network();
        let session = db.session();

        let gql = session
            .execute("MATCH (a:Person) RETURN a.name AS name ORDER BY name")
            .unwrap();
        let sql = session
            .execute_sql(
                "SELECT * FROM GRAPH_TABLE (
                    MATCH (a:Person)
                    COLUMNS (a.name AS name)
                ) ORDER BY name",
            )
            .unwrap();

        assert_eq!(gql.row_count(), sql.row_count());
        for (g, s) in gql.rows().iter().zip(sql.rows().iter()) {
            assert_eq!(g[0], s[0], "GQL and SQL/PGQ should return same names");
        }
    }

    /// Verify SQL/PGQ WHERE inside GRAPH_TABLE matches GQL WHERE.
    #[test]
    fn sql_pgq_inner_where_matches_gql_where() {
        let db = create_social_network();
        let session = db.session();

        let gql = session
            .execute(
                "MATCH (a:Person)-[:KNOWS]->(b:Person) \
                 WHERE a.age > 28 \
                 RETURN a.name AS source, b.name AS target \
                 ORDER BY source, target",
            )
            .unwrap();
        let sql = session
            .execute_sql(
                "SELECT * FROM GRAPH_TABLE (
                    MATCH (a:Person)-[:KNOWS]->(b:Person)
                    WHERE a.age > 28
                    COLUMNS (a.name AS source, b.name AS target)
                ) ORDER BY source, target",
            )
            .unwrap();

        assert_eq!(
            gql.row_count(),
            sql.row_count(),
            "GQL: {}, SQL/PGQ: {}",
            gql.row_count(),
            sql.row_count()
        );
        for (g, s) in gql.rows().iter().zip(sql.rows().iter()) {
            assert_eq!(g, s);
        }
    }

    /// SQL/PGQ GROUP BY produces correct aggregation.
    #[test]
    fn sql_pgq_group_by_aggregate_correctness() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_sql(
                "SELECT source, COUNT(*) AS cnt FROM GRAPH_TABLE (
                    MATCH (a:Person)-[:KNOWS]->(b:Person)
                    COLUMNS (a.name AS source)
                ) GROUP BY source ORDER BY source",
            )
            .unwrap();

        assert_eq!(result.row_count(), 2);
        // Alix knows 2, Gus knows 1
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(2));
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
        assert_eq!(result.rows()[1][1], Value::Int64(1));
    }

    /// SQL/PGQ SELECT DISTINCT deduplicates correctly.
    #[test]
    fn sql_pgq_select_distinct_correctness() {
        let db = create_social_network();
        let session = db.session();

        let all = session
            .execute_sql(
                "SELECT source FROM GRAPH_TABLE (
                    MATCH (a:Person)-[:KNOWS]->(b:Person)
                    COLUMNS (a.name AS source)
                )",
            )
            .unwrap();
        let distinct = session
            .execute_sql(
                "SELECT DISTINCT source FROM GRAPH_TABLE (
                    MATCH (a:Person)-[:KNOWS]->(b:Person)
                    COLUMNS (a.name AS source)
                )",
            )
            .unwrap();

        assert_eq!(all.row_count(), 3, "3 KNOWS edges total");
        assert_eq!(distinct.row_count(), 2, "2 distinct source names");
    }

    /// nullIf works in SQL/PGQ COLUMNS expressions.
    #[test]
    fn sql_pgq_nullif_in_columns() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_sql(
                "SELECT * FROM GRAPH_TABLE (
                    MATCH (a:Person)
                    COLUMNS (a.name AS name, nullIf(a.name, 'Alix') AS filtered)
                ) ORDER BY name",
            )
            .unwrap();

        assert_eq!(result.row_count(), 3);
        // Alix: nullIf('Alix', 'Alix') = NULL
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert!(result.rows()[0][1].is_null());
        // Gus: nullIf('Gus', 'Alix') = 'Gus'
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
        assert_eq!(result.rows()[1][1], Value::String("Gus".into()));
    }
}
