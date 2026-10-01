//! Seam tests for DML cross-feature interactions (ISO/IEC 39075 Section 13).
//!
//! Tests the boundaries between INSERT/SET/DELETE/MERGE and graph context,
//! transactions, constraints, and edge cases.
//!
//! ```bash
//! cargo test -p grafeo-engine --test seam_dml_interactions
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

// ============================================================================
// 1. DELETE edge cases
// ============================================================================

mod delete_edge_cases {
    use super::*;

    #[test]
    fn delete_node_with_edges_without_detach_errors() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        // DELETE without DETACH should fail when node has edges
        let result = session.execute("MATCH (n:Person {name: 'Alix'}) DELETE n");
        assert!(
            result.is_err(),
            "DELETE without DETACH on node with edges should error"
        );
    }

    #[test]
    fn detach_delete_removes_node_and_edges() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        session
            .execute("MATCH (n:Person {name: 'Alix'}) DETACH DELETE n")
            .unwrap();

        // Alix should be gone
        let result = session
            .execute("MATCH (n:Person {name: 'Alix'}) RETURN n")
            .unwrap();
        assert_eq!(result.row_count(), 0, "Alix should be deleted");

        // Gus should still exist
        let result = session
            .execute("MATCH (n:Person {name: 'Gus'}) RETURN n")
            .unwrap();
        assert_eq!(result.row_count(), 1, "Gus should still exist");

        // The KNOWS edge should be gone
        let result = session.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
        assert_eq!(result.row_count(), 0, "KNOWS edge should be deleted");
    }

    #[test]
    fn delete_then_match_in_same_transaction() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session
            .execute("MATCH (n:Person {name: 'Alix'}) DETACH DELETE n")
            .unwrap();

        // Within same transaction, deleted node should not be visible
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1, "Only Gus should remain");
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));

        session.execute("COMMIT").unwrap();
    }

    #[test]
    fn delete_all_nodes() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        session.execute("MATCH (n) DETACH DELETE n").unwrap();

        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 0, "All nodes should be deleted");
    }

    #[test]
    fn delete_preserves_other_graph() {
        let db = db();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("CREATE GRAPH other").unwrap();
        session.execute("USE GRAPH other").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Delete everything in 'other'
        session.execute("MATCH (n) DETACH DELETE n").unwrap();

        // Default graph should be untouched
        session.execute("USE GRAPH default").unwrap();
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Default graph should still have Alix"
        );
    }
}

// ============================================================================
// 2. SET/REMOVE interactions
// ============================================================================

mod set_remove {
    use super::*;

    #[test]
    fn set_property_then_query_by_it() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session
            .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 30")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person {age: 30}) RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn set_label_then_query_by_it() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session
            .execute("MATCH (n:Person {name: 'Alix'}) SET n:Engineer")
            .unwrap();

        let result = session.execute("MATCH (n:Engineer) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn remove_label_then_query_misses() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person:Engineer {name: 'Alix'})")
            .unwrap();

        session
            .execute("MATCH (n:Person {name: 'Alix'}) REMOVE n:Engineer")
            .unwrap();

        let result = session.execute("MATCH (n:Engineer) RETURN n.name").unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "Removed label should not match anymore"
        );

        // Person label should still work
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1, "Person label should still match");
    }

    #[test]
    fn remove_property() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        session
            .execute("MATCH (n:Person {name: 'Alix'}) REMOVE n.age")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "Removed property should be NULL"
        );
    }

    #[test]
    fn set_property_to_null_removes_it() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        session
            .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = NULL")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "Property set to NULL should read as NULL"
        );
    }

    #[test]
    fn set_in_transaction_visible_before_commit() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session
            .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 30")
            .unwrap();

        // Should be visible within the same transaction
        let result = session
            .execute("MATCH (n:Person {age: 30}) RETURN n.name")
            .unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "SET should be visible within transaction"
        );
        session.execute("COMMIT").unwrap();
    }

    #[test]
    fn set_rolled_back_not_visible() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session
            .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 30")
            .unwrap();
        session.execute("ROLLBACK").unwrap();

        let result = session
            .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "SET should be rolled back"
        );
    }
}

// ============================================================================
// 3. MERGE edge cases
// ============================================================================

mod merge_cases {
    use super::*;

    #[test]
    fn merge_creates_when_not_exists() {
        let db = db();
        let session = db.session();
        session.execute("MERGE (:Person {name: 'Alix'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn merge_matches_when_exists() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // MERGE should match existing, not create duplicate
        session.execute("MERGE (:Person {name: 'Alix'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "MERGE should not create duplicate");
    }

    #[test]
    fn merge_on_create_set() {
        let db = db();
        let session = db.session();
        session
            .execute("MERGE (n:Person {name: 'Alix'}) ON CREATE SET n.age = 30")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(30));
    }

    #[test]
    fn merge_on_match_set() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 25})")
            .unwrap();

        session
            .execute("MERGE (n:Person {name: 'Alix'}) ON MATCH SET n.age = 30")
            .unwrap();

        let result = session
            .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(30),
            "ON MATCH SET should update existing"
        );
    }
}

// ============================================================================
// 4. Cross-graph DML
// ============================================================================

mod cross_graph_dml {
    use super::*;

    #[test]
    fn insert_respects_graph_context() {
        let db = db();
        let session = db.session();

        session.execute("CREATE GRAPH alpha").unwrap();
        session.execute("CREATE GRAPH beta").unwrap();

        session.execute("USE GRAPH alpha").unwrap();
        session.execute("INSERT (:Item {name: 'widget'})").unwrap();

        session.execute("USE GRAPH beta").unwrap();
        session.execute("INSERT (:Item {name: 'gadget'})").unwrap();

        // Each graph should have its own data
        session.execute("USE GRAPH alpha").unwrap();
        let result = session.execute("MATCH (n:Item) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("widget".into()));

        session.execute("USE GRAPH beta").unwrap();
        let result = session.execute("MATCH (n:Item) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("gadget".into()));
    }

    #[test]
    fn delete_in_one_graph_does_not_affect_other() {
        let db = db();
        let session = db.session();

        session.execute("CREATE GRAPH alpha").unwrap();
        session.execute("CREATE GRAPH beta").unwrap();

        session.execute("USE GRAPH alpha").unwrap();
        session.execute("INSERT (:Item {name: 'widget'})").unwrap();

        session.execute("USE GRAPH beta").unwrap();
        session.execute("INSERT (:Item {name: 'gadget'})").unwrap();

        // Delete from beta
        session.execute("MATCH (n) DETACH DELETE n").unwrap();

        // Alpha should be unaffected
        session.execute("USE GRAPH alpha").unwrap();
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "alpha should still have its node");

        // Beta should be empty
        session.execute("USE GRAPH beta").unwrap();
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 0, "beta should be empty");
    }

    #[test]
    fn set_in_one_graph_does_not_affect_other() {
        let db = db();
        let session = db.session();

        session.execute("CREATE GRAPH alpha").unwrap();
        session.execute("CREATE GRAPH beta").unwrap();

        session.execute("USE GRAPH alpha").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("USE GRAPH beta").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Update in beta only
        session
            .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 30")
            .unwrap();

        // Alpha's Alix should not have age
        session.execute("USE GRAPH alpha").unwrap();
        let result = session
            .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
            .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Null,
            "Alpha's Alix should not have age set"
        );

        // Beta's Alix should have age
        session.execute("USE GRAPH beta").unwrap();
        let result = session
            .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(30));
    }

    #[test]
    fn merge_respects_graph_context() {
        let db = db();
        let session = db.session();

        session.execute("CREATE GRAPH alpha").unwrap();
        session.execute("USE GRAPH alpha").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // MERGE in default graph should create (not match alpha's data)
        session.execute("USE GRAPH default").unwrap();
        session.execute("MERGE (:Person {name: 'Alix'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Default graph should have Alix from MERGE"
        );

        session.execute("USE GRAPH alpha").unwrap();
        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "Alpha should still have its Alix");
    }
}

// ============================================================================
// 5. INSERT edge cases
// ============================================================================

mod insert_edge_cases {
    use super::*;

    #[test]
    fn insert_multiple_nodes_single_statement() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
            .unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn insert_node_with_edge_in_single_statement() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 2, "Should have 2 nodes");

        let result = session.execute("MATCH ()-[r:KNOWS]->() RETURN r").unwrap();
        assert_eq!(result.row_count(), 1, "Should have 1 edge");
    }

    #[test]
    fn insert_with_multiple_labels() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Person:Engineer {name: 'Alix'})")
            .unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1);

        let result = session.execute("MATCH (n:Engineer) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn insert_empty_node() {
        let db = db();
        let session = db.session();
        session.execute("INSERT ()").unwrap();

        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "Empty node should be created");
    }

    #[test]
    fn insert_preserves_types() {
        let db = db();
        let session = db.session();
        session
            .execute(
                "INSERT (:Data {str_val: 'hello', int_val: 42, float_val: 3.125, bool_val: true})",
            )
            .unwrap();

        let result = session
            .execute("MATCH (n:Data) RETURN n.str_val, n.int_val, n.float_val, n.bool_val")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String("hello".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(42));
        assert_eq!(result.rows()[0][2], Value::Float64(3.125));
        assert_eq!(result.rows()[0][3], Value::Bool(true));
    }
}

// ============================================================================
// 6. Negative numeric literals in property maps (issue #160)
// ============================================================================

mod negative_literal_properties {
    use super::*;

    #[test]
    fn insert_negative_float() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Point {lat: -6.248})").unwrap();

        let result = session.execute("MATCH (n:Point) RETURN n.lat").unwrap();
        assert_eq!(result.rows()[0][0], Value::Float64(-6.248));
    }

    #[test]
    fn insert_negative_integer() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Floor {level: -3})").unwrap();

        let result = session.execute("MATCH (n:Floor) RETURN n.level").unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(-3));
    }

    #[test]
    fn insert_mixed_positive_and_negative() {
        let db = db();
        let session = db.session();
        session
            .execute("INSERT (:Point {lat: -6.248, lon: 106.845})")
            .unwrap();

        let result = session
            .execute("MATCH (n:Point) RETURN n.lat, n.lon")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Float64(-6.248));
        assert_eq!(result.rows()[0][1], Value::Float64(106.845));
    }
}

// ============================================================================
// Reads after writes in one statement (#479)
// ============================================================================

mod reads_after_writes {
    use super::*;

    fn ints(result: &grafeo_engine::database::QueryResult) -> Vec<Vec<i64>> {
        result
            .rows()
            .iter()
            .map(|row| row.iter().map(|v| v.as_int64().unwrap()).collect())
            .collect()
    }

    /// Every row after the `WITH` sees the nodes that all rows inserted.
    #[test]
    fn every_row_sees_every_insert() {
        let db = db();
        let result = db
            .execute(
                "UNWIND [1, 2] AS i INSERT (:N {i: i}) WITH i \
                 MATCH (n:N) RETURN i, n.i ORDER BY i, n.i",
            )
            .unwrap();
        assert_eq!(
            ints(&result),
            vec![vec![1, 1], vec![1, 2], vec![2, 1], vec![2, 2]]
        );
    }

    /// The insert runs even when the `MATCH` after it finds nothing.
    #[test]
    fn an_insert_runs_when_the_match_after_it_finds_nothing() {
        let db = db();
        let result = db
            .execute("INSERT (:N {id: 'c'}) WITH 1 AS x MATCH (m:Missing) RETURN m")
            .unwrap();
        assert_eq!(result.row_count(), 0);

        let result = db.execute("MATCH (n:N) RETURN n.id").unwrap();
        assert_eq!(result.rows(), &[vec![Value::from("c")]]);
    }

    /// Both patterns of a comma-separated `MATCH` see the insert.
    #[test]
    fn every_pattern_of_a_match_sees_the_insert() {
        let db = db();
        let result = db
            .execute(
                "INSERT (:N {id: 'c'})-[:K]->(:M {id: 'd'}) WITH 1 AS x \
                 MATCH (a:N), (b:M) RETURN a.id, b.id",
            )
            .unwrap();
        assert_eq!(result.rows(), &[vec![Value::from("c"), Value::from("d")]]);
    }

    /// The `MATCH` sees the inserts of all input chunks, not only the first.
    #[test]
    fn a_match_sees_the_inserts_of_every_chunk() {
        let db = db();
        let result = db
            .execute(
                "UNWIND range(1, 5000) AS i INSERT (:N {i: i}) WITH i WHERE i = 1 \
                 MATCH (n:N) RETURN count(n)",
            )
            .unwrap();
        assert_eq!(ints(&result), vec![vec![5000]]);
    }
}
