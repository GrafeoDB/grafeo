//! Seam tests for GQL session state (ISO/IEC 39075 Sections 4.7.3, 7.1, 7.2).
//!
//! These tests target the *boundaries* between features: negative validation,
//! independence of session fields, selective resets, state persistence across
//! queries, and introspection functions reflecting current state.
//!
//! ```bash
//! cargo test -p grafeo-engine --test seam_session_state
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

// ============================================================================
// 1. Negative validation: SESSION SET targets must exist
// ============================================================================

mod negative_validation {
    use super::*;

    #[test]
    fn session_set_graph_nonexistent_errors() {
        let db = db();
        let session = db.session();
        let result = session.execute("SESSION SET GRAPH nosuchgraph");
        assert!(
            result.is_err(),
            "SESSION SET GRAPH to nonexistent graph should error"
        );
    }

    #[test]
    fn session_set_schema_nonexistent_errors() {
        let db = db();
        let session = db.session();
        let result = session.execute("SESSION SET SCHEMA nosuchschema");
        assert!(
            result.is_err(),
            "SESSION SET SCHEMA to nonexistent schema should error"
        );
    }

    #[test]
    fn session_set_graph_after_drop_errors() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH temp").unwrap();
        session.execute("DROP GRAPH temp").unwrap();
        let result = session.execute("SESSION SET GRAPH temp");
        assert!(
            result.is_err(),
            "SESSION SET GRAPH to dropped graph should error"
        );
    }

    #[test]
    fn session_set_schema_after_drop_errors() {
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA temp").unwrap();
        session.execute("DROP SCHEMA temp").unwrap();
        let result = session.execute("SESSION SET SCHEMA temp");
        assert!(
            result.is_err(),
            "SESSION SET SCHEMA to dropped schema should error"
        );
    }

    #[test]
    fn session_set_graph_nonexistent_does_not_change_state() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH real").unwrap();
        session.execute("SESSION SET GRAPH real").unwrap();
        assert_eq!(session.current_graph(), Some("real".to_string()));

        // Failed SET should not change the current graph
        let _ = session.execute("SESSION SET GRAPH nosuchgraph");
        assert_eq!(
            session.current_graph(),
            Some("real".to_string()),
            "Failed SESSION SET GRAPH should not change current graph"
        );
    }

    #[test]
    fn session_set_schema_nonexistent_does_not_change_state() {
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA real").unwrap();
        session.execute("SESSION SET SCHEMA real").unwrap();
        assert_eq!(session.current_schema(), Some("real".to_string()));

        // Failed SET should not change the current schema
        let _ = session.execute("SESSION SET SCHEMA nosuchschema");
        assert_eq!(
            session.current_schema(),
            Some("real".to_string()),
            "Failed SESSION SET SCHEMA should not change current schema"
        );
    }

    #[test]
    fn use_graph_nonexistent_errors() {
        let db = db();
        let session = db.session();
        let result = session.execute("USE GRAPH nonexistent");
        assert!(
            result.is_err(),
            "USE GRAPH to nonexistent graph should error"
        );
    }

    #[test]
    fn session_set_graph_default_always_succeeds() {
        // "default" is always valid, even without creating anything
        let db = db();
        let session = db.session();
        let result = session.execute("SESSION SET GRAPH default");
        assert!(
            result.is_ok(),
            "SESSION SET GRAPH default should always succeed"
        );
    }
}

// ============================================================================
// 2. Independence: schema and graph are independent session fields
//    (ISO/IEC 39075 Section 4.7.3)
// ============================================================================

mod independence {
    use super::*;

    #[test]
    fn set_schema_does_not_affect_graph() {
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();
        assert_eq!(session.current_schema(), Some("analytics".to_string()));
        assert_eq!(
            session.current_graph(),
            None,
            "Setting schema should not set graph"
        );
    }

    #[test]
    fn set_graph_does_not_affect_schema() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        assert_eq!(session.current_graph(), Some("mydb".to_string()));
        assert_eq!(
            session.current_schema(),
            None,
            "Setting graph should not set schema"
        );
    }

    #[test]
    fn set_both_independently() {
        // Graphs resolve relative to current schema (Section 17.2),
        // so create graph after setting schema context.
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        assert_eq!(session.current_schema(), Some("analytics".to_string()));
        assert_eq!(session.current_graph(), Some("mydb".to_string()));
    }

    #[test]
    fn reset_schema_keeps_graph() {
        // Set graph without schema, then set schema separately
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();

        session.execute("SESSION RESET SCHEMA").unwrap();

        assert_eq!(session.current_schema(), None, "Schema should be reset");
        assert_eq!(
            session.current_graph(),
            Some("mydb".to_string()),
            "Graph should be unchanged after resetting schema"
        );
    }

    #[test]
    fn reset_graph_keeps_schema() {
        // Set graph without schema, then set schema separately
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();

        session.execute("SESSION RESET GRAPH").unwrap();

        assert_eq!(session.current_graph(), None, "Graph should be reset");
        assert_eq!(
            session.current_schema(),
            Some("analytics".to_string()),
            "Schema should be unchanged after resetting graph"
        );
    }

    #[test]
    fn schema_scoped_graph_isolation() {
        // Graphs created within a schema should be isolated from graphs in other schemas
        let db = db();
        let session = db.session();

        // Create schema + graph, insert data
        session.execute("CREATE SCHEMA dept_a").unwrap();
        session.execute("SESSION SET SCHEMA dept_a").unwrap();
        session.execute("CREATE GRAPH records").unwrap();
        session.execute("SESSION SET GRAPH records").unwrap();
        session.execute("INSERT (:Report {title: 'Q1'})").unwrap();

        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "dept_a/records should have 1 node");

        // Switch to a different schema with a same-named graph
        session.execute("SESSION RESET").unwrap();
        session.execute("CREATE SCHEMA dept_b").unwrap();
        session.execute("SESSION SET SCHEMA dept_b").unwrap();
        session.execute("CREATE GRAPH records").unwrap();
        session.execute("SESSION SET GRAPH records").unwrap();

        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "dept_b/records should be empty (isolated from dept_a/records)"
        );
    }

    #[test]
    fn data_isolated_when_schema_set() {
        // CREATE SCHEMA creates an isolated partition with its own default graph.
        // Data inserted before the schema switch should be invisible from the new schema.
        let db = db();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();

        // The analytics schema has its own empty default graph
        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "Schema isolation: data in root schema should be invisible from analytics schema"
        );

        // Insert into analytics schema and verify it's visible there
        session
            .execute("INSERT (:Metric {name: 'pageviews'})")
            .unwrap();
        let result = session.execute("MATCH (m:Metric) RETURN m").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Data inserted in analytics should be visible"
        );

        // Switch back: root schema should still have the Person, not the Metric
        session.execute("SESSION RESET SCHEMA").unwrap();
        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Root schema should still have Person"
        );
        let result = session.execute("MATCH (m:Metric) RETURN m").unwrap();
        assert_eq!(
            result.row_count(),
            0,
            "Root schema should not see analytics Metric"
        );
    }
}

// ============================================================================
// 3. Selective reset: each SESSION RESET target clears only its field
//    (ISO/IEC 39075 Section 7.2 GR1-GR4)
// ============================================================================

mod selective_reset {
    use super::*;

    /// Helper: set all four session fields.
    /// Creates graph without schema first, then sets schema separately,
    /// since graphs resolve relative to the current schema.
    fn set_all_fields(session: &grafeo_engine::Session) {
        session.execute("CREATE GRAPH mygraph").unwrap();
        session.execute("SESSION SET GRAPH mygraph").unwrap();
        session.execute("CREATE SCHEMA myschema").unwrap();
        session.execute("SESSION SET SCHEMA myschema").unwrap();
        session.execute("SESSION SET TIME ZONE 'UTC+2'").unwrap();
        session
            .execute("SESSION SET PARAMETER viewing_epoch = 0")
            .unwrap();
    }

    #[test]
    fn reset_schema_only() {
        let db = db();
        let session = db.session();
        set_all_fields(&session);

        session.execute("SESSION RESET SCHEMA").unwrap();

        assert_eq!(session.current_schema(), None, "Schema should be cleared");
        assert!(session.current_graph().is_some(), "Graph should remain");
        assert!(session.time_zone().is_some(), "Time zone should remain");
    }

    #[test]
    fn reset_graph_only() {
        let db = db();
        let session = db.session();
        set_all_fields(&session);

        session.execute("SESSION RESET GRAPH").unwrap();

        assert_eq!(session.current_graph(), None, "Graph should be cleared");
        assert!(session.current_schema().is_some(), "Schema should remain");
        assert!(session.time_zone().is_some(), "Time zone should remain");
    }

    #[test]
    fn reset_time_zone_only() {
        let db = db();
        let session = db.session();
        set_all_fields(&session);

        session.execute("SESSION RESET TIME ZONE").unwrap();

        assert_eq!(session.time_zone(), None, "Time zone should be cleared");
        assert!(session.current_schema().is_some(), "Schema should remain");
        assert!(session.current_graph().is_some(), "Graph should remain");
    }

    #[test]
    fn reset_parameters_only() {
        let db = db();
        let session = db.session();
        set_all_fields(&session);

        // viewing_epoch is a dedicated field, not a session parameter,
        // so use the Rust API to verify param clearing
        session.set_parameter("custom_key", Value::Int64(42));
        assert!(session.get_parameter("custom_key").is_some());

        session.execute("SESSION RESET PARAMETERS").unwrap();

        assert_eq!(
            session.get_parameter("custom_key"),
            None,
            "Session parameters should be cleared"
        );
        assert!(session.current_schema().is_some(), "Schema should remain");
        assert!(session.current_graph().is_some(), "Graph should remain");
        assert!(session.time_zone().is_some(), "Time zone should remain");
    }

    #[test]
    fn reset_all_clears_everything() {
        let db = db();
        let session = db.session();
        set_all_fields(&session);

        session.execute("SESSION RESET").unwrap();

        assert_eq!(session.current_schema(), None, "Schema should be cleared");
        assert_eq!(session.current_graph(), None, "Graph should be cleared");
        assert_eq!(session.time_zone(), None, "Time zone should be cleared");
        assert_eq!(
            session.viewing_epoch(),
            None,
            "Parameters should be cleared"
        );
    }

    #[test]
    fn reset_all_characteristics_clears_everything() {
        let db = db();
        let session = db.session();
        set_all_fields(&session);

        session
            .execute("SESSION RESET ALL CHARACTERISTICS")
            .unwrap();

        assert_eq!(session.current_schema(), None, "Schema should be cleared");
        assert_eq!(session.current_graph(), None, "Graph should be cleared");
        assert_eq!(session.time_zone(), None, "Time zone should be cleared");
    }
}

// ============================================================================
// 4. State persistence: session state survives across queries
// ============================================================================

mod state_persistence {
    use super::*;

    #[test]
    fn schema_persists_across_queries() {
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();

        // Run several unrelated queries
        session.execute("RETURN 1").unwrap();
        session.execute("RETURN 2").unwrap();
        session.execute("RETURN 3").unwrap();

        assert_eq!(
            session.current_schema(),
            Some("analytics".to_string()),
            "Schema should persist across queries"
        );
    }

    #[test]
    fn graph_persists_across_queries() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("RETURN 1").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.row_count(), 2, "Both inserts should go to mydb");
        assert_eq!(
            session.current_graph(),
            Some("mydb".to_string()),
            "Graph should persist"
        );
    }

    #[test]
    fn time_zone_persists_across_queries() {
        let db = db();
        let session = db.session();
        session.execute("SESSION SET TIME ZONE 'UTC+5'").unwrap();

        session.execute("RETURN 1").unwrap();
        session.execute("RETURN 2").unwrap();

        assert_eq!(
            session.time_zone(),
            Some("UTC+5".to_string()),
            "Time zone should persist across queries"
        );
    }

    #[test]
    fn graph_context_persists_across_insert_and_match() {
        let db = db();
        let session = db.session();

        // Insert data into default
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Switch to named graph, insert, then match
        session.execute("CREATE GRAPH isolated").unwrap();
        session.execute("SESSION SET GRAPH isolated").unwrap();
        session.execute("INSERT (:Event {type: 'click'})").unwrap();

        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Should only see Event, not Person from default"
        );

        // Switch back and verify default still has its data
        session.execute("SESSION SET GRAPH default").unwrap();
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 1, "Default should still have Person");
    }

    #[test]
    fn graph_persists_across_transaction_commit() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("COMMIT").unwrap();

        // Graph context should still be mydb after commit
        assert_eq!(
            session.current_graph(),
            Some("mydb".to_string()),
            "Graph should persist after COMMIT"
        );
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(
            result.row_count(),
            1,
            "Data should be visible after commit in mydb"
        );
    }

    #[test]
    fn graph_persists_after_rollback() {
        // Session state is not transactional (Section 4.7.3): ROLLBACK does not revert
        // session graph/schema.
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();

        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("ROLLBACK").unwrap();

        // Graph context should still be mydb (session state is not transactional)
        assert_eq!(
            session.current_graph(),
            Some("mydb".to_string()),
            "SESSION SET GRAPH should survive ROLLBACK"
        );

        // But the data should be rolled back
        let result = session.execute("MATCH (n) RETURN n").unwrap();
        assert_eq!(result.row_count(), 0, "Data should be rolled back");
    }
}

// ============================================================================
// 5. Introspection: functions reflect current session state
// ============================================================================

mod introspection {
    use super::*;

    #[test]
    fn current_schema_default_when_unset() {
        let db = db();
        let session = db.session();
        let result = session.execute("RETURN CURRENT_SCHEMA AS s").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("default".into()));
    }

    #[test]
    fn current_graph_default_when_unset() {
        let db = db();
        let session = db.session();
        let result = session.execute("RETURN CURRENT_GRAPH AS g").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("default".into()));
    }

    #[test]
    fn current_schema_after_set() {
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();
        let result = session.execute("RETURN CURRENT_SCHEMA AS s").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("analytics".into()));
    }

    #[test]
    fn current_graph_after_set() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        let result = session.execute("RETURN CURRENT_GRAPH AS g").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("mydb".into()));
    }

    #[test]
    fn current_schema_default_when_reset() {
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();
        session.execute("SESSION RESET SCHEMA").unwrap();
        let result = session.execute("RETURN CURRENT_SCHEMA AS s").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("default".into()));
    }

    #[test]
    fn current_graph_default_when_reset() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        session.execute("SESSION RESET GRAPH").unwrap();
        let result = session.execute("RETURN CURRENT_GRAPH AS g").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("default".into()));
    }

    #[test]
    fn info_returns_map() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let result = session.execute("RETURN info() AS i").unwrap();
        assert_eq!(result.rows().len(), 1);
        match &result.rows()[0][0] {
            Value::Map(m) => {
                assert!(
                    m.contains_key("node_count"),
                    "info() should have node_count"
                );
                assert!(m.contains_key("version"), "info() should have version");
            }
            other => panic!("info() should return a Map, got: {other:?}"),
        }
    }

    #[test]
    fn info_reflects_active_graph() {
        let db = db();
        let session = db.session();

        // Insert into default
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Create named graph with different data
        session.execute("CREATE GRAPH other").unwrap();
        session.execute("SESSION SET GRAPH other").unwrap();
        session.execute("INSERT (:Event {type: 'click'})").unwrap();

        let result = session.execute("RETURN info() AS i").unwrap();
        match &result.rows()[0][0] {
            Value::Map(m) => {
                let node_count = m.get("node_count").expect("should have node_count");
                assert_eq!(
                    *node_count,
                    Value::Int64(1),
                    "info() should reflect the active graph (1 node in 'other')"
                );
            }
            other => panic!("info() should return a Map, got: {other:?}"),
        }
    }

    #[test]
    fn schema_returns_map() {
        let db = db();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let result = session.execute("RETURN schema() AS s").unwrap();
        assert_eq!(result.rows().len(), 1);
        match &result.rows()[0][0] {
            Value::Map(m) => {
                assert!(m.contains_key("labels"), "schema() should have labels");
            }
            other => panic!("schema() should return a Map, got: {other:?}"),
        }
    }

    #[test]
    fn schema_reflects_active_graph() {
        let db = db();
        let session = db.session();

        // Insert labeled node in default
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Create named graph with different label
        session.execute("CREATE GRAPH other").unwrap();
        session.execute("SESSION SET GRAPH other").unwrap();
        session.execute("INSERT (:Event {type: 'click'})").unwrap();

        let result = session.execute("RETURN schema() AS s").unwrap();
        match &result.rows()[0][0] {
            Value::Map(m) => {
                let labels = m.get("labels").expect("should have labels");
                match labels {
                    Value::List(list) => {
                        let label_strs: Vec<_> = list
                            .iter()
                            .filter_map(|v| match v {
                                Value::String(s) => Some(s.as_str()),
                                _ => None,
                            })
                            .collect();
                        assert!(
                            label_strs.contains(&"Event"),
                            "schema() should show Event label from active graph, got: {label_strs:?}"
                        );
                        assert!(
                            !label_strs.contains(&"Person"),
                            "schema() should not show Person label from default graph"
                        );
                    }
                    other => panic!("labels should be a List, got: {other:?}"),
                }
            }
            other => panic!("schema() should return a Map, got: {other:?}"),
        }
    }

    #[test]
    fn introspection_in_where_clause() {
        // CURRENT_SCHEMA and CURRENT_GRAPH should work in WHERE, not just RETURN
        let db = db();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();

        let result = session
            .execute("RETURN CASE WHEN CURRENT_SCHEMA = 'analytics' THEN true ELSE false END AS ok")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Bool(true));
    }

    #[test]
    fn current_schema_and_graph_in_same_query() {
        let db = db();
        let session = db.session();
        // Graphs resolve relative to the schema: create mydb inside analytics.
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();

        let result = session
            .execute("RETURN CURRENT_SCHEMA AS s, CURRENT_GRAPH AS g")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String("analytics".into()));
        assert_eq!(result.rows()[0][1], Value::String("mydb".into()));
    }

    /// Switching the schema away from the selected graph leaves a selection
    /// that names no graph: statements fail until another graph is selected,
    /// instead of running on the default graph.
    #[test]
    fn a_schema_switch_away_from_the_graph_fails_statements() {
        let db = db();
        let session = db.session();
        session.execute("CREATE GRAPH mydb").unwrap();
        session.execute("SESSION SET GRAPH mydb").unwrap();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();

        let error = session
            .execute("RETURN CURRENT_GRAPH AS g")
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not exist"), "{error}");
        let error = session.execute("INSERT (:Person)").unwrap_err().to_string();
        assert!(error.contains("does not exist"), "{error}");
        let error = session
            .execute("EXPLAIN MATCH (n) RETURN n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not exist"), "{error}");
        let Err(error) = session.execute_streaming("MATCH (n) RETURN n") else {
            panic!("a stream of a graph that does not exist");
        };
        assert!(error.to_string().contains("does not exist"), "{error}");

        session.execute("SESSION RESET SCHEMA").unwrap();
        let result = session.execute("RETURN CURRENT_GRAPH AS g").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("mydb".into()));
        // A fresh session reads the default graph on purpose: the failed
        // INSERT must not have run there instead.
        let nodes = db.session().execute("MATCH (n) RETURN count(n)").unwrap();
        assert_eq!(nodes.rows()[0][0], Value::Int64(0));
    }
}
