//! Seam tests for transaction semantics (ISO/IEC 39075 Section 8).
//!
//! Tests the boundaries between transactions, session state, DDL, and
//! multi-graph operations. Covers: READ ONLY enforcement, session state
//! independence from transactions, DDL rollback, savepoint + graph
//! interactions, and basic isolation between sessions.
//!
//! ```bash
//! cargo test -p grafeo-engine --test seam_transaction
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

// ============================================================================
// 1. READ ONLY enforcement
// ============================================================================

mod read_only {
    use super::*;

    #[test]
    fn read_only_allows_match() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("START TRANSACTION READ ONLY").unwrap();
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
        session.execute("COMMIT").unwrap();
    }

    #[test]
    fn read_only_blocks_insert() {
        let db = db();
        let session = db.session();
        session.execute("START TRANSACTION READ ONLY").unwrap();
        let result = session.execute("INSERT (:Person {name: 'Alix'})");
        assert!(
            result.is_err(),
            "INSERT should fail in READ ONLY transaction"
        );
        session.execute("ROLLBACK").unwrap();
    }

    #[test]
    fn read_only_blocks_set() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("START TRANSACTION READ ONLY").unwrap();
        let result = session.execute("MATCH (n:Person) SET n.age = 30");
        assert!(result.is_err(), "SET should fail in READ ONLY transaction");
        session.execute("ROLLBACK").unwrap();
    }

    #[test]
    fn read_only_blocks_delete() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("START TRANSACTION READ ONLY").unwrap();
        let result = session.execute("MATCH (n:Person) DETACH DELETE n");
        assert!(
            result.is_err(),
            "DELETE should fail in READ ONLY transaction"
        );
        session.execute("ROLLBACK").unwrap();
    }

    #[test]
    fn read_only_blocks_merge() {
        let db = db();
        let session = db.session();
        session.execute("START TRANSACTION READ ONLY").unwrap();
        let result = session.execute("MERGE (:Person {name: 'Alix'})");
        assert!(
            result.is_err(),
            "MERGE should fail in READ ONLY transaction"
        );
        session.execute("ROLLBACK").unwrap();
    }

    #[test]
    fn read_only_blocks_ddl() {
        let db = db();
        let session = db.session();
        session.execute("START TRANSACTION READ ONLY").unwrap();
        let result = session.execute("CREATE GRAPH foo");
        assert!(
            result.is_err(),
            "CREATE GRAPH should fail in READ ONLY transaction"
        );
        session.execute("ROLLBACK").unwrap();
    }

    #[test]
    fn read_write_allows_insert() {
        // Baseline: READ WRITE (the default) should allow mutations
        let db = db();
        let session = db.session();
        session.execute("START TRANSACTION READ WRITE").unwrap();
        let result = session.execute("INSERT (:Person {name: 'Alix'})");
        assert!(
            result.is_ok(),
            "INSERT should succeed in READ WRITE transaction"
        );
        session.execute("COMMIT").unwrap();

        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn default_transaction_allows_mutations() {
        // START TRANSACTION without mode should default to READ WRITE
        let db = db();
        let session = db.session();
        session.execute("START TRANSACTION").unwrap();
        let result = session.execute("INSERT (:Person {name: 'Alix'})");
        assert!(result.is_ok(), "Default transaction should allow mutations");
        session.execute("COMMIT").unwrap();
    }

    #[test]
    fn read_only_error_does_not_abort_transaction() {
        // After a mutation attempt fails in READ ONLY, the transaction should still be active
        // for further reads
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("START TRANSACTION READ ONLY").unwrap();
        let _ = session.execute("INSERT (:Person {name: 'Gus'})"); // fails

        // Transaction should still be usable for reads
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Should still see Alix after failed INSERT"
        );
        session.execute("COMMIT").unwrap();
    }
}

// ============================================================================
// 2. Session state is not transactional (Section 4.7.3)
// ============================================================================

mod session_state_independence {
    use super::*;

    #[test]
    fn session_set_graph_survives_rollback() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        session.execute("ROLLBACK").unwrap();

        // Session state is not transactional: graph should still be set
        assert_eq!(
            session.current_graph(),
            Some("mydb".to_string()),
            "SESSION SET GRAPH should survive ROLLBACK"
        );
    }

    #[test]
    fn session_set_schema_survives_rollback() {
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();
        session.execute("ROLLBACK").unwrap();

        assert_eq!(
            session.current_schema(),
            Some("analytics".to_string()),
            "SESSION SET SCHEMA should survive ROLLBACK"
        );
    }

    #[test]
    fn session_set_time_zone_survives_rollback() {
        let db = db();
        let session = db.session();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET TIME ZONE 'UTC+5'").unwrap();
        session.execute("ROLLBACK").unwrap();

        assert_eq!(
            session.time_zone(),
            Some("UTC+5".to_string()),
            "SESSION SET TIME ZONE should survive ROLLBACK"
        );
    }

    #[test]
    fn session_reset_inside_transaction() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        session.execute("SESSION SET TIME ZONE 'UTC+5'").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION RESET").unwrap();

        assert_eq!(
            session.current_graph(),
            None,
            "RESET should clear graph in transaction"
        );
        assert_eq!(
            session.time_zone(),
            None,
            "RESET should clear timezone in transaction"
        );

        session.execute("COMMIT").unwrap();

        // State should remain reset after commit
        assert_eq!(session.current_graph(), None);
        assert_eq!(session.time_zone(), None);
    }

    #[test]
    fn session_graph_persists_across_commit() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("COMMIT").unwrap();

        assert_eq!(
            session.current_graph(),
            Some("mydb".to_string()),
            "Graph should persist after COMMIT"
        );
    }

    #[test]
    fn graph_switch_mid_transaction_routes_data() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH alpha").unwrap();
        session.execute("CREATE GRAPH beta").unwrap();

        session.execute("START TRANSACTION").unwrap();

        session.execute("USE GRAPH alpha").unwrap();
        session.execute("INSERT (:Item {name: 'widget'})").unwrap();

        session.execute("USE GRAPH beta").unwrap();
        session.execute("INSERT (:Item {name: 'gadget'})").unwrap();

        session.execute("COMMIT").unwrap();

        // Verify each graph has its own data
        session.execute("USE GRAPH alpha").unwrap();
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "alpha should have 1 node");

        session.execute("USE GRAPH beta").unwrap();
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "beta should have 1 node");
    }
}

// ============================================================================
// 3. DDL + transaction interactions
// ============================================================================

mod ddl_transactions {
    use super::*;

    #[test]
    fn create_graph_in_transaction_visible_after_commit() {
        let db = db();
        let session = db.session();

        session.execute("START TRANSACTION").unwrap();
        session.execute("CREATE GRAPH transient").unwrap();
        session.execute("COMMIT").unwrap();

        assert!(
            db.list_graphs().contains(&"transient".to_string()),
            "Graph should exist after COMMIT"
        );
    }

    #[test]
    fn create_graph_usable_within_transaction() {
        let db = db();
        let session = db.session();

        session.execute("START TRANSACTION").unwrap();
        session.execute("CREATE GRAPH workspace").unwrap();
        session.execute("USE GRAPH workspace").unwrap();
        session.execute("INSERT (:Item {name: 'widget'})").unwrap();
        session.execute("COMMIT").unwrap();

        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "Data should be visible after commit");
    }

    #[test]
    fn drop_graph_in_transaction_committed() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH temp").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("DROP GRAPH temp").unwrap();
        session.execute("COMMIT").unwrap();

        assert!(
            !db.list_graphs().contains(&"temp".to_string()),
            "Graph should not exist after DROP + COMMIT"
        );
    }

    #[test]
    fn create_schema_in_transaction() {
        let db = db();
        let session = db.session();

        session.execute("START TRANSACTION").unwrap();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("COMMIT").unwrap();

        // Verify schema exists by being able to set it
        let result = session.execute("SESSION SET SCHEMA analytics");
        assert!(result.is_ok(), "Schema should exist after CREATE + COMMIT");
    }

    #[test]
    fn multiple_ddl_in_single_transaction() {
        let db = db();
        let session = db.session();

        session.execute("START TRANSACTION").unwrap();
        session.execute("CREATE GRAPH alpha").unwrap();
        session.execute("CREATE GRAPH beta").unwrap();
        session.execute("CREATE SCHEMA reports").unwrap();
        session.execute("COMMIT").unwrap();

        let graphs = db.list_graphs();
        assert!(graphs.contains(&"alpha".to_string()));
        assert!(graphs.contains(&"beta".to_string()));

        let result = session.execute("SESSION SET SCHEMA reports");
        assert!(result.is_ok());
    }
}

// ============================================================================
// 4. Savepoint + graph interactions
// ============================================================================

mod savepoint_graphs {
    use super::*;

    #[test]
    fn savepoint_rollback_discards_post_savepoint_inserts() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH alpha").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("USE GRAPH alpha").unwrap();
        session.execute("INSERT (:Item {name: 'before'})").unwrap();

        session.execute("SAVEPOINT sp1").unwrap();

        session.execute("INSERT (:Item {name: 'after'})").unwrap();

        session.execute("ROLLBACK TO SAVEPOINT sp1").unwrap();
        session.execute("COMMIT").unwrap();

        let result = session
            .execute("MATCH (n:Item) RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Only pre-savepoint insert should survive"
        );
        assert_eq!(result.rows()[0][0], Value::String("before".into()));
    }

    #[test]
    fn savepoint_across_graph_switch() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH alpha").unwrap();
        session.execute("CREATE GRAPH beta").unwrap();

        session.execute("START TRANSACTION").unwrap();

        // Insert into alpha
        session.execute("USE GRAPH alpha").unwrap();
        session
            .execute("INSERT (:Item {name: 'alpha_item'})")
            .unwrap();

        // Savepoint
        session.execute("SAVEPOINT sp1").unwrap();

        // Switch to beta and insert
        session.execute("USE GRAPH beta").unwrap();
        session
            .execute("INSERT (:Item {name: 'beta_item'})")
            .unwrap();

        // Also insert more into alpha
        session.execute("USE GRAPH alpha").unwrap();
        session
            .execute("INSERT (:Item {name: 'alpha_extra'})")
            .unwrap();

        // Rollback to savepoint: should discard beta_item and alpha_extra
        session.execute("ROLLBACK TO SAVEPOINT sp1").unwrap();
        session.execute("COMMIT").unwrap();

        // Alpha should only have alpha_item
        session.execute("USE GRAPH alpha").unwrap();
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "alpha should have 1 node (pre-savepoint)"
        );

        // Beta should be empty
        session.execute("USE GRAPH beta").unwrap();
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "beta should be empty after savepoint rollback"
        );
    }

    #[test]
    fn nested_savepoints() {
        let db = db();
        let session = db.session();

        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        session.execute("SAVEPOINT sp1").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        session.execute("SAVEPOINT sp2").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        // Rollback to sp2: discard Vincent
        session.execute("ROLLBACK TO SAVEPOINT sp2").unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "Should have Alix and Gus after rollback to sp2"
        );

        // Rollback to sp1: discard Gus
        session.execute("ROLLBACK TO SAVEPOINT sp1").unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Should have only Alix after rollback to sp1"
        );

        session.execute("COMMIT").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn release_savepoint_keeps_data() {
        let db = db();
        let session = db.session();

        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("SAVEPOINT sp1").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session.execute("RELEASE SAVEPOINT sp1").unwrap();
        session.execute("COMMIT").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "Both inserts should survive after RELEASE + COMMIT"
        );
    }
}

// ============================================================================
// 5. Isolation between sessions
// ============================================================================

mod session_isolation {
    use super::*;

    #[test]
    fn uncommitted_data_invisible_to_other_session() {
        let db = db();
        let s1 = db.session();
        let s2 = db.session();

        s1.execute("START TRANSACTION").unwrap();
        s1.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // s2 should not see s1's uncommitted data
        let result = s2.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "Uncommitted data should be invisible to other sessions"
        );

        s1.execute("COMMIT").unwrap();

        // After commit, s2 should see the data
        let result = s2.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Committed data should be visible to other sessions"
        );
    }

    #[test]
    fn concurrent_sessions_different_graphs() {
        let db = db();
        db.execute("CREATE GRAPH alpha").unwrap();
        db.execute("CREATE GRAPH beta").unwrap();

        let s1 = db.session();
        let s2 = db.session();

        s1.execute("USE GRAPH alpha").unwrap();
        s2.execute("USE GRAPH beta").unwrap();

        s1.execute("INSERT (:Item {name: 'widget'})").unwrap();
        s2.execute("INSERT (:Item {name: 'gadget'})").unwrap();

        // Each session should only see its own graph's data
        let r1 = s1.execute("MATCH (n) RETURN n.name").unwrap();
        let r2 = s2.execute("MATCH (n) RETURN n.name").unwrap();

        assert_eq!(r1.row_count(), 1);
        assert_eq!(r2.row_count(), 1);
        assert_eq!(r1.rows()[0][0], Value::String("widget".into()));
        assert_eq!(r2.rows()[0][0], Value::String("gadget".into()));
    }

    #[test]
    fn session_graph_state_is_independent() {
        let db = db();
        db.execute("CREATE GRAPH shared").unwrap();

        let s1 = db.session();
        let s2 = db.session();

        s1.execute("USE GRAPH shared").unwrap();

        // s2's graph should be unaffected by s1's graph switch
        assert_eq!(s1.current_graph(), Some("shared".to_string()));
        assert_eq!(
            s2.current_graph(),
            None,
            "s2 should still be on default graph"
        );
    }

    #[test]
    fn rollback_does_not_affect_other_session() {
        let db = db();
        let s1 = db.session();
        let s2 = db.session();

        // s2 inserts some data (auto-committed)
        s2.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // s1 starts a transaction, inserts, then rolls back
        s1.execute("START TRANSACTION").unwrap();
        s1.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s1.execute("ROLLBACK").unwrap();

        // s2's data should still be intact
        let result = s2.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "s2's committed data should survive s1's rollback"
        );
        assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    }

    #[test]
    fn nested_transaction_via_savepoint() {
        // START TRANSACTION inside an existing transaction creates an auto-savepoint
        let db = db();
        let session = db.session();

        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Nested transaction (auto-savepoint)
        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Nested rollback (rolls back to auto-savepoint)
        session.execute("ROLLBACK").unwrap();

        // Outer transaction should still have Alix
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1, "Outer transaction should have Alix");
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

        session.execute("COMMIT").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "Only Alix should survive");
    }
}

// ============================================================================
// 6. Multi-schema transaction atomicity (ISO/IEC 39075 catalog hierarchy)
// ============================================================================

mod multi_schema_atomicity {
    use super::*;

    #[test]
    fn commit_is_atomic_across_schemas() {
        let db = db();
        let session = db.session();

        session.execute("CREATE SCHEMA alpha").unwrap();
        session.execute("CREATE SCHEMA beta").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET SCHEMA alpha").unwrap();
        session.execute("INSERT (:Item {owner: 'alpha'})").unwrap();
        // Visible within the same tx before the schema switch
        let mid = session.execute("MATCH (n:Item) RETURN n").unwrap();
        assert_eq!(
            mid.row_count(),
            1,
            "alpha write must be visible within its own tx"
        );
        session.execute("SESSION SET SCHEMA beta").unwrap();
        session.execute("INSERT (:Item {owner: 'beta'})").unwrap();
        session.execute("COMMIT").unwrap();

        session.execute("SESSION SET SCHEMA alpha").unwrap();
        let a = session.execute("MATCH (n:Item) RETURN n").unwrap();
        assert_eq!(a.row_count(), 1, "alpha's commit should be visible");

        session.execute("SESSION SET SCHEMA beta").unwrap();
        let b = session.execute("MATCH (n:Item) RETURN n").unwrap();
        assert_eq!(b.row_count(), 1, "beta's commit should be visible");
    }

    #[test]
    fn rollback_is_atomic_across_schemas() {
        let db = db();
        let session = db.session();

        session.execute("CREATE SCHEMA alpha").unwrap();
        session.execute("CREATE SCHEMA beta").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET SCHEMA alpha").unwrap();
        session.execute("INSERT (:Item {owner: 'alpha'})").unwrap();
        session.execute("SESSION SET SCHEMA beta").unwrap();
        session.execute("INSERT (:Item {owner: 'beta'})").unwrap();
        session.execute("ROLLBACK").unwrap();

        session.execute("SESSION SET SCHEMA alpha").unwrap();
        let a = session.execute("MATCH (n:Item) RETURN n").unwrap();
        assert_eq!(
            a.row_count(),
            0,
            "alpha's write must be undone by cross-schema rollback"
        );

        session.execute("SESSION SET SCHEMA beta").unwrap();
        let b = session.execute("MATCH (n:Item) RETURN n").unwrap();
        assert_eq!(
            b.row_count(),
            0,
            "beta's write must be undone by cross-schema rollback"
        );
    }

    // `partial_failure_rolls_back_all_schemas` was removed in 0.5.40:
    // forcing a failure mid-tx depended on a NOT NULL constraint firing
    // in a schema-scoped type after `SESSION SET SCHEMA`, which is
    // itself a bug.
    // The two tests below use a UNIQUE constraint declared in the default
    // schema, which fires deterministically at statement execution time
    // (`check_unique_node_property` in the catalog validator) regardless of
    // the session's current schema, so they cover rollback atomicity without
    // depending on that bug.

    #[test]
    fn unique_violation_rolls_back_prior_writes_single_schema() {
        let db = db();
        let session = db.session();

        session
            .execute("CREATE CONSTRAINT uniq_item FOR (n:Item) ON (n.id) UNIQUE")
            .unwrap();
        session.execute("INSERT (:Item {id: 1})").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Item {id: 2})").unwrap();
        session.execute("INSERT (:Item {id: 3})").unwrap();
        let violation = session.execute("INSERT (:Item {id: 1})");
        assert!(
            violation.is_err(),
            "INSERT with duplicate UNIQUE value must fail at statement exec"
        );
        session.execute("ROLLBACK").unwrap();

        let remaining = session.execute("MATCH (n:Item) RETURN n.id").unwrap();
        assert_eq!(
            remaining.row_count(),
            1,
            "only the pre-transaction seed row should remain after ROLLBACK"
        );
    }

    #[test]
    fn cross_schema_rollback_after_unique_violation() {
        let db = db();
        let session = db.session();

        // UNIQUE constraint in the default schema: the violation trigger is
        // schema-independent and fires whatever the current session schema is.
        session
            .execute("CREATE CONSTRAINT uniq_marker FOR (n:Marker) ON (n.id) UNIQUE")
            .unwrap();
        session.execute("INSERT (:Marker {id: 1})").unwrap();

        session.execute("CREATE SCHEMA alpha").unwrap();
        session.execute("CREATE SCHEMA beta").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET SCHEMA alpha").unwrap();
        session.execute("INSERT (:Row {owner: 'alpha'})").unwrap();
        session.execute("SESSION SET SCHEMA beta").unwrap();
        session.execute("INSERT (:Row {owner: 'beta'})").unwrap();
        session.execute("SESSION RESET SCHEMA").unwrap();
        let violation = session.execute("INSERT (:Marker {id: 1})");
        assert!(
            violation.is_err(),
            "duplicate UNIQUE in default schema must fail"
        );
        session.execute("ROLLBACK").unwrap();

        // Default schema: only the seed remains.
        let default_markers = session.execute("MATCH (n:Marker) RETURN n.id").unwrap();
        assert_eq!(
            default_markers.row_count(),
            1,
            "default schema should still have exactly the seed Marker"
        );

        // Alpha's Row writes are gone.
        session.execute("SESSION SET SCHEMA alpha").unwrap();
        let alpha_rows = session.execute("MATCH (n:Row) RETURN n").unwrap();
        assert_eq!(
            alpha_rows.row_count(),
            0,
            "alpha's Row writes must be undone by cross-schema rollback"
        );

        // Beta's Row writes are gone.
        session.execute("SESSION SET SCHEMA beta").unwrap();
        let beta_rows = session.execute("MATCH (n:Row) RETURN n").unwrap();
        assert_eq!(
            beta_rows.row_count(),
            0,
            "beta's Row writes must be undone by cross-schema rollback"
        );
    }

    /// Companion to `commit_is_atomic_across_schemas`: the same mid-tx schema
    /// switch must finalize writes into *both* schemas' named stores, not just
    /// the post-switch one. Uses a third SESSION SET SCHEMA back to the first
    /// schema to double-check that alpha's writes still route to alpha's store.
    #[test]
    fn mid_tx_schema_switch_finalizes_all_touched_stores() {
        let db = db();
        let session = db.session();

        session.execute("CREATE SCHEMA alpha").unwrap();
        session.execute("CREATE SCHEMA beta").unwrap();
        session.execute("CREATE SCHEMA gamma").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET SCHEMA alpha").unwrap();
        session.execute("INSERT (:Tag {v: 1})").unwrap();
        session.execute("SESSION SET SCHEMA beta").unwrap();
        session.execute("INSERT (:Tag {v: 2})").unwrap();
        session.execute("SESSION SET SCHEMA gamma").unwrap();
        session.execute("INSERT (:Tag {v: 3})").unwrap();
        session.execute("COMMIT").unwrap();

        for schema in ["alpha", "beta", "gamma"] {
            session
                .execute(&format!("SESSION SET SCHEMA {schema}"))
                .unwrap();
            let r = session.execute("MATCH (n:Tag) RETURN n").unwrap();
            assert_eq!(
                r.row_count(),
                1,
                "schema '{schema}' should have its own Tag after commit"
            );
        }
    }
}
