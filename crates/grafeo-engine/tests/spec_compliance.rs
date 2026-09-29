//! Spec compliance tests for GQL, Cypher, and SPARQL.
//!
//! Tests the features added in 0.5.13 for full spec coverage.
//! Covers: set operations, apply operator, predicates, session commands,
//! GQL statements, Cypher features, and SPARQL features.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test spec_compliance
//! ```

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

// ============================================================================
// Test Fixtures
// ============================================================================

fn social_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

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
    let dave = session
        .create_node_with_props(
            &["Person", "Engineer"],
            [
                ("name", Value::String("Dave".into())),
                ("age", Value::Int64(28)),
            ],
        )
        .unwrap();
    let techcorp = session
        .create_node_with_props(
            &["Company"],
            [
                ("name", Value::String("TechCorp".into())),
                ("founded", Value::Int64(2010)),
            ],
        )
        .unwrap();

    let e1 = session.create_edge(alix, gus, "KNOWS").unwrap();
    db.set_edge_property(e1, "since", Value::Int64(2020))
        .unwrap();
    let e2 = session.create_edge(alix, harm, "KNOWS").unwrap();
    db.set_edge_property(e2, "since", Value::Int64(2019))
        .unwrap();
    let e3 = session.create_edge(gus, harm, "KNOWS").unwrap();
    db.set_edge_property(e3, "since", Value::Int64(2021))
        .unwrap();
    session.create_edge(alix, techcorp, "WORKS_AT").unwrap();
    session.create_edge(gus, techcorp, "WORKS_AT").unwrap();
    session.create_edge(dave, techcorp, "WORKS_AT").unwrap();

    // Verify setup: 4 Person + 1 Company = 5 nodes, 3 KNOWS + 3 WORKS_AT = 6 edges
    assert_eq!(db.node_count(), 5, "social_network: expected 5 nodes");
    assert_eq!(db.edge_count(), 6, "social_network: expected 6 edges");

    db
}

fn extract_strings(db: &GrafeoDB, query: &str) -> Vec<String> {
    let session = db.session();
    let result = session.execute(query).unwrap();
    result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

// ============================================================================
// GQL Set Operations (covers set_ops.rs, gql_translator.rs, planner)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_set_ops {
    use super::*;

    #[test]
    fn union_all_returns_duplicates() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.age < 30 RETURN n.name \
                 UNION ALL \
                 MATCH (n:Person) WHERE n.age > 24 RETURN n.name",
            )
            .unwrap();
        // Left (age<30): Gus(25), Dave(28) = 2; Right (age>24): Gus, Dave, Alix, Harm = 4
        assert_eq!(result.row_count(), 6);
    }

    #[test]
    fn union_distinct_deduplicates() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.age < 30 RETURN n.name \
                 UNION \
                 MATCH (n:Person) WHERE n.age > 24 RETURN n.name",
            )
            .unwrap();
        // Should be at most 4 distinct names
        assert!(result.row_count() <= 4);
        // All names should be unique
        let names: Vec<_> = result
            .rows()
            .iter()
            .map(|r| format!("{:?}", r[0]))
            .collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "UNION should deduplicate");
    }

    #[test]
    fn except_removes_common_rows() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) RETURN n.name \
                 EXCEPT \
                 MATCH (n:Person) WHERE n.age > 30 RETURN n.name",
            )
            .unwrap();
        // Should exclude Harm (35)
        let names = result
            .rows()
            .iter()
            .filter_map(|r| match &r[0] {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!names.contains(&"Harm".to_string()));
    }

    #[test]
    fn intersect_keeps_common_rows() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.age >= 25 RETURN n.name \
                 INTERSECT \
                 MATCH (n:Person) WHERE n.age <= 30 RETURN n.name",
            )
            .unwrap();
        // Intersection of age>=25 and age<=30: Gus(25), Dave(28), Alix(30)
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn otherwise_returns_left_if_non_empty() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.name = 'Alix' RETURN n.name \
                 OTHERWISE \
                 MATCH (n:Person) RETURN n.name",
            )
            .unwrap();
        // Left side matches Alix, so right side is ignored
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn otherwise_falls_through_when_left_empty() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.name = 'Nobody' RETURN n.name \
                 OTHERWISE \
                 MATCH (n:Person) RETURN n.name",
            )
            .unwrap();
        // Left side empty, falls through to right
        assert_eq!(result.row_count(), 4);
    }
}

// ============================================================================
// GQL Predicates (covers filter.rs IS TYPED, IS DIRECTED, etc.)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_predicates {
    use super::*;

    #[test]
    fn property_exists_check() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE property_exists(n, 'age') RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 4, "All persons have age property");
    }

    /// GQL NULLIF: returns NULL when both arguments are equal.
    #[test]
    fn nullif_returns_null_when_equal() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN NULLIF(n.age, 30) AS val")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Null);
    }

    /// GQL NULLIF: returns the first argument when they differ.
    #[test]
    fn nullif_returns_value_when_different() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.name = 'Gus' RETURN NULLIF(n.age, 30) AS val")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(25));
    }

    #[test]
    fn element_id_function() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) WHERE n.name = 'Alix' RETURN element_id(n)")
            .unwrap();
        let id_str = match &result.rows()[0][0] {
            Value::String(s) => s.to_string(),
            other => panic!("Expected string element ID, got {other:?}"),
        };
        assert!(
            id_str.starts_with("n:"),
            "element_id should start with 'n:'"
        );
    }

    #[test]
    fn cast_to_string() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person) WHERE n.name = 'Alix' RETURN CAST(n.age AS STRING) AS age_str",
            )
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String("30".into()));
    }

    #[test]
    fn cast_to_integer() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Item {value: '42'})").unwrap();
        let result = session
            .execute("MATCH (n:Item) RETURN CAST(n.value AS INTEGER) AS v")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(42));
    }

    #[test]
    fn cast_to_float() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Item {value: '3.14'})").unwrap();
        let result = session
            .execute("MATCH (n:Item) RETURN CAST(n.value AS FLOAT) AS v")
            .unwrap();
        match &result.rows()[0][0] {
            Value::Float64(f) => assert!((f - std::f64::consts::PI).abs() < 0.01),
            other => panic!("Expected Float64, got {other:?}"),
        }
    }

    #[test]
    fn cast_to_boolean() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Item {value: 'true'})").unwrap();
        let result = session
            .execute("MATCH (n:Item) RETURN CAST(n.value AS BOOLEAN) AS v")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Bool(true));
    }

    #[test]
    fn xor_operator() {
        let db = social_network();
        // XOR: (age>30)=Harm XOR (name=Gus)=Gus, but not both => Gus, Harm
        let names = extract_strings(
            &db,
            "MATCH (n:Person) WHERE (n.age > 30) XOR (n.name = 'Gus') RETURN n.name ORDER BY n.name",
        );
        assert_eq!(
            names,
            vec!["Gus", "Harm"],
            "ORDER BY n.name should produce alphabetical order"
        );
    }
}

// ============================================================================
// GQL Statements (covers parser.rs + translator: FINISH, SELECT, LET, etc.)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_statements {
    use super::*;

    #[test]
    fn finish_returns_empty() {
        let db = social_network();
        let session = db.session();
        let result = session.execute("MATCH (n:Person) FINISH").unwrap();
        assert_eq!(result.row_count(), 0, "FINISH should return no rows");
    }

    #[test]
    fn select_as_return_synonym() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) SELECT n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn element_where_on_node() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person WHERE n.age > 28) RETURN n.name ORDER BY n.name")
            .unwrap();
        let names: Vec<_> = result
            .rows()
            .iter()
            .filter_map(|r| match &r[0] {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
            .collect();
        // ORDER BY n.name: Alix (age 30) before Harm (age 35), Gus (age 25) excluded
        assert_eq!(
            names,
            vec!["Alix", "Harm"],
            "ORDER BY n.name should produce alphabetical order"
        );
    }

    #[test]
    fn element_where_on_edge() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (a:Person)-[e:KNOWS WHERE e.since >= 2020]->(b:Person) \
                 RETURN a.name, b.name ORDER BY a.name",
            )
            .unwrap();
        // since>=2020: Alix->Gus (2020), Gus->Harm (2021)
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn iso_path_quantifier_exact() {
        let db = social_network();
        let session = db.session();
        // {1} means exactly 1 hop
        let result = session
            .execute("MATCH (a:Person)-[:KNOWS{1}]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // Exactly 3 direct KNOWS edges: Alix->Gus, Alix->Harm, Gus->Harm
        assert_eq!(result.row_count(), 3, "Should find direct connections");
    }

    #[test]
    fn iso_path_quantifier_range() {
        let db = social_network();
        let session = db.session();
        // {1,2} means 1 to 2 hops
        let result = session
            .execute(
                "MATCH (a:Person)-[:KNOWS{1,2}]->(b:Person) WHERE a.name = 'Alix' RETURN b.name",
            )
            .unwrap();
        // 1-hop: Gus, Harm; 2-hop: Gus->Harm = Harm again
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn property_map_not_confused_with_quantifier() {
        // Regression: {since: 2020} was misinterpreted as path quantifier
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (a:Person)-[e:KNOWS {since: 2020}]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // Only Alix->Gus has since=2020
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn match_mode_different_edges() {
        let db = social_network();
        let session = db.session();
        // DIFFERENT EDGES is like TRAIL - no repeated edges
        let result = session
            .execute(
                "MATCH DIFFERENT EDGES (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name",
            )
            .unwrap();
        // 1-hop: (Alix,Gus), (Alix,Harm), (Gus,Harm); 2-hop: (Alix,Harm via Gus)
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn match_mode_repeatable_elements() {
        let db = social_network();
        let session = db.session();
        // REPEATABLE ELEMENTS is like WALK - edges and nodes may repeat
        let result = session
            .execute("MATCH REPEATABLE ELEMENTS (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // No cycles in KNOWS, so same as TRAIL: 3 one-hop + 1 two-hop
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn label_expression_disjunction() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n IS Person | Company) RETURN n.name ORDER BY n.name")
            .unwrap();
        // Should find all 4 persons + TechCorp
        assert_eq!(result.row_count(), 5);
    }

    #[test]
    fn label_expression_conjunction() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n IS Person & Engineer) RETURN n.name")
            .unwrap();
        // Only Dave has both Person and Engineer
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn label_expression_negation() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n IS Person & !Engineer) RETURN n.name ORDER BY n.name")
            .unwrap();
        // Alix, Gus, Harm have Person but not Engineer
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn group_by_explicit() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH (n:Person)-[:WORKS_AT]->(c:Company) \
                 RETURN c.name, count(n) AS cnt GROUP BY c.name",
            )
            .unwrap();
        // Only one company (TechCorp) has WORKS_AT edges
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn gql_filter_clause() {
        let db = social_network();
        let session = db.session();
        // FILTER is a GQL synonym for WHERE
        let result = session
            .execute("MATCH (n:Person) FILTER n.age > 28 RETURN n.name ORDER BY n.name")
            .unwrap();
        // age > 28: Alix(30), Harm(35)
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn gql_offset_clause() {
        let db = social_network();
        let session = db.session();
        // OFFSET is a GQL synonym for SKIP
        let result = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name OFFSET 2")
            .unwrap();
        assert_eq!(result.row_count(), 2); // 4 total, skip 2
    }

    #[test]
    fn list_index_access() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Item {tags: ['a', 'b', 'c']})")
            .unwrap();
        let result = session
            .execute("MATCH (n:Item) RETURN n.tags[0] AS first")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String("a".into()));
    }

    #[test]
    fn gql_block_comment() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH /* all people */ (n:Person) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(4));
    }
}

// ============================================================================
// GQL Session and DDL Commands (covers session.rs, parser DDL branches)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_session_commands {
    use super::*;

    #[test]
    fn create_graph_and_drop_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // CREATE GRAPH
        let result = session.execute("CREATE GRAPH test_graph");
        assert!(result.is_ok());
        // Verify it exists
        let graphs = db.list_graphs();
        assert!(graphs.contains(&"test_graph".to_string()));
        // DROP GRAPH
        let result = session.execute("DROP GRAPH test_graph");
        assert!(result.is_ok());
        let graphs = db.list_graphs();
        assert!(!graphs.contains(&"test_graph".to_string()));
    }

    #[test]
    fn create_graph_if_not_exists() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH g1").unwrap();
        // Without IF NOT EXISTS, should fail
        let result = session.execute("CREATE GRAPH g1");
        assert!(result.is_err());
        // With IF NOT EXISTS, should succeed
        let result = session.execute("CREATE GRAPH IF NOT EXISTS g1");
        assert!(result.is_ok());
    }

    #[test]
    fn drop_graph_if_exists() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Without IF EXISTS, should fail on nonexistent
        let result = session.execute("DROP GRAPH nonexistent");
        assert!(result.is_err());
        // With IF EXISTS, should succeed
        let result = session.execute("DROP GRAPH IF EXISTS nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn create_property_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("CREATE PROPERTY GRAPH my_graph");
        assert!(result.is_ok());
        assert!(db.list_graphs().contains(&"my_graph".to_string()));
    }

    #[test]
    fn drop_property_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE PROPERTY GRAPH pg").unwrap();
        let result = session.execute("DROP PROPERTY GRAPH pg");
        assert!(result.is_ok());
    }

    #[test]
    fn use_graph_command() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH workspace").unwrap();
        let result = session.execute("USE GRAPH workspace");
        assert!(result.is_ok());
        assert_eq!(session.current_graph(), Some("workspace".to_string()));
    }

    #[test]
    fn use_graph_nonexistent_errors() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("USE GRAPH nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn session_set_time_zone() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("SESSION SET TIME ZONE 'UTC+5'");
        assert!(result.is_ok());
        assert_eq!(session.time_zone(), Some("UTC+5".to_string()));
    }

    #[test]
    fn session_set_graph() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH analytics").unwrap();
        let result = session.execute("SESSION SET GRAPH analytics");
        assert!(result.is_ok());
        assert_eq!(session.current_graph(), Some("analytics".to_string()));
    }

    #[test]
    fn session_set_schema() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // ISO/IEC 39075 Section 7.1 GR1: SESSION SET SCHEMA sets session schema independently
        session.execute("CREATE SCHEMA myschema").unwrap();
        let result = session.execute("SESSION SET SCHEMA myschema");
        assert!(result.is_ok());
        assert_eq!(session.current_schema(), Some("myschema".to_string()));
        // Graph should remain unaffected
        assert_eq!(session.current_graph(), None);
    }

    #[test]
    fn session_set_schema_nonexistent_errors() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // SESSION SET SCHEMA should error if schema does not exist
        let result = session.execute("SESSION SET SCHEMA nosuchschema");
        assert!(result.is_err());
    }

    #[test]
    fn session_reset() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH g1").unwrap();
        session.execute("USE GRAPH g1").unwrap();
        session.execute("SESSION SET TIME ZONE 'EST'").unwrap();
        session.execute("CREATE SCHEMA s1").unwrap();
        session.execute("SESSION SET SCHEMA s1").unwrap();
        // Reset clears everything (Section 7.2 GR1+GR2+GR3)
        let result = session.execute("SESSION RESET");
        assert!(result.is_ok());
        assert_eq!(session.current_graph(), None);
        assert_eq!(session.current_schema(), None);
        assert_eq!(session.time_zone(), None);
    }

    #[test]
    fn session_reset_schema_only() {
        // ISO/IEC 39075 Section 7.2 GR1: SESSION RESET SCHEMA resets schema only
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH g1").unwrap();
        session.execute("USE GRAPH g1").unwrap();
        session.execute("CREATE SCHEMA s1").unwrap();
        session.execute("SESSION SET SCHEMA s1").unwrap();
        let result = session.execute("SESSION RESET SCHEMA");
        assert!(result.is_ok());
        assert_eq!(session.current_schema(), None);
        // Graph should remain set
        assert_eq!(session.current_graph(), Some("g1".to_string()));
    }

    #[test]
    fn session_reset_graph_only() {
        // ISO/IEC 39075 Section 7.2 GR2: SESSION RESET GRAPH resets graph only
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH g1").unwrap();
        session.execute("USE GRAPH g1").unwrap();
        session.execute("CREATE SCHEMA s1").unwrap();
        session.execute("SESSION SET SCHEMA s1").unwrap();
        let result = session.execute("SESSION RESET GRAPH");
        assert!(result.is_ok());
        assert_eq!(session.current_graph(), None);
        // Schema should remain set
        assert_eq!(session.current_schema(), Some("s1".to_string()));
    }

    // ========================================================================
    // QoL introspection functions: CURRENT_SCHEMA, CURRENT_GRAPH, info(), schema()
    // ========================================================================

    #[test]
    fn return_current_schema_default_when_unset() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN CURRENT_SCHEMA AS s").unwrap();
        assert_eq!(result.columns, vec!["s"]);
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0].len(), 1);
        assert_eq!(result.rows()[0][0], Value::String("default".into()));
    }

    #[test]
    fn return_current_schema_after_set() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE SCHEMA analytics").unwrap();
        session.execute("SESSION SET SCHEMA analytics").unwrap();
        let result = session.execute("RETURN CURRENT_SCHEMA AS s").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("analytics".into()));
    }

    #[test]
    fn return_current_graph_default_when_unset() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("RETURN CURRENT_GRAPH AS g").unwrap();
        assert_eq!(result.columns, vec!["g"]);
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::String("default".into()));
    }

    #[test]
    fn return_current_graph_after_use() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("CREATE GRAPH social").unwrap();
        session.execute("USE GRAPH social").unwrap();
        let result = session.execute("RETURN CURRENT_GRAPH AS g").unwrap();
        assert_eq!(result.rows()[0][0], Value::String("social".into()));
    }

    #[test]
    fn return_info_function() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Insert some data first
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus'})-[:KNOWS]->(:Person {name: 'Vincent'})")
            .unwrap();
        let result = session.execute("RETURN info() AS i").unwrap();
        assert_eq!(result.columns, vec!["i"]);
        assert_eq!(result.rows().len(), 1);
        let info = result.rows()[0][0]
            .as_map()
            .expect("info() should return a map");
        assert_eq!(
            info.get(&PropertyKey::from("mode")),
            Some(&Value::String("lpg".into()))
        );
        // Should have 3 nodes (Alix, Gus, Vincent)
        assert_eq!(
            info.get(&PropertyKey::from("node_count")),
            Some(&Value::Int64(3))
        );
        // Should have 1 edge (KNOWS)
        assert_eq!(
            info.get(&PropertyKey::from("edge_count")),
            Some(&Value::Int64(1))
        );
        assert!(info.get(&PropertyKey::from("version")).is_some());
    }

    #[test]
    fn return_schema_function() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();
        let result = session.execute("RETURN schema() AS s").unwrap();
        assert_eq!(result.columns, vec!["s"]);
        assert_eq!(result.rows().len(), 1);
        let schema = result.rows()[0][0]
            .as_map()
            .expect("schema() should return a map");
        // Labels should include "Person"
        let labels = schema
            .get(&PropertyKey::from("labels"))
            .expect("should have labels");
        if let Value::List(l) = labels {
            assert!(l.contains(&Value::String("Person".into())));
        } else {
            panic!("labels should be a list");
        }
        // Edge types should include "KNOWS"
        let edge_types = schema
            .get(&PropertyKey::from("edge_types"))
            .expect("should have edge_types");
        if let Value::List(l) = edge_types {
            assert!(l.contains(&Value::String("KNOWS".into())));
        } else {
            panic!("edge_types should be a list");
        }
    }

    #[test]
    fn session_close() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("SESSION CLOSE");
        assert!(result.is_ok());
    }

    #[test]
    fn start_transaction_commit_rollback_via_gql() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // START TRANSACTION works
        session.execute("START TRANSACTION").unwrap();
        assert!(session.in_transaction());

        // COMMIT works
        session.execute("COMMIT").unwrap();
        assert!(!session.in_transaction());

        // ROLLBACK works
        session.execute("START TRANSACTION").unwrap();
        session.execute("ROLLBACK").unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn commit_without_transaction_errors() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("COMMIT");
        assert!(result.is_err());
    }

    #[test]
    fn rollback_without_transaction_errors() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("ROLLBACK");
        assert!(result.is_err());
    }

    #[test]
    fn session_set_parameter() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute("SESSION SET PARAMETER my_param = 42");
        assert!(result.is_ok());
    }
}

// ============================================================================
// GQL Path Features (covers path search prefixes, questioned edges)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_path_features {
    use super::*;

    #[test]
    fn any_shortest_path() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH ANY SHORTEST (a:Person)-[:KNOWS*]->(b:Person) \
                 WHERE a.name = 'Alix' AND b.name = 'Harm' \
                 RETURN a.name, b.name",
            )
            .unwrap();
        // Alix->Harm is 1 hop (direct), the single shortest path
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn all_shortest_path() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute(
                "MATCH ALL SHORTEST (a:Person)-[:KNOWS*]->(b:Person) \
                 WHERE a.name = 'Alix' AND b.name = 'Harm' \
                 RETURN a.name, b.name",
            )
            .unwrap();
        // Only one shortest path: Alix->Harm (1 hop, direct)
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn path_mode_walk() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH WALK (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // No cycles in KNOWS: 3 one-hop + 1 two-hop = 4
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn path_mode_trail() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH TRAIL (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name, b.name")
            .unwrap();
        // No cycles in KNOWS: 3 one-hop + 1 two-hop = 4
        assert_eq!(result.row_count(), 4);
    }
}

// ============================================================================
// GQL INSERT Patterns (covers path INSERT decomposition)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_insert_patterns {
    use super::*;

    #[test]
    fn insert_node_with_properties() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Eve', age: 22})")
            .unwrap();
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn insert_path_creates_nodes_and_edge() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'X'})-[:KNOWS]->(:Person {name: 'Y'})")
            .unwrap();
        let result = session
            .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn create_edge_with_properties_in_query() {
        // Regression: {since: 2020} was misinterpreted as quantifier
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'A'})").unwrap();
        session.execute("INSERT (:Person {name: 'B'})").unwrap();
        session
            .execute(
                "MATCH (a:Person), (b:Person) WHERE a.name = 'A' AND b.name = 'B' \
                 CREATE (a)-[:KNOWS {since: 2020}]->(b) RETURN a.name",
            )
            .unwrap();
        let result = session
            .execute("MATCH (a)-[e:KNOWS]->(b) RETURN e.since")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(2020));
    }
}

// ============================================================================
// Cypher Features (covers cypher parser + translator)
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_features {
    use super::*;

    #[test]
    fn count_star() {
        // ISO/IEC 39075 Section 20.9: COUNT(*) counts all rows
        let db = social_network();
        let session = db.session();
        let result = session
            .execute("MATCH (n:Person) RETURN COUNT(*) AS cnt")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(4)); // Alix, Gus, Dave, Harm
    }

    #[test]
    fn count_subquery() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person) \
                 RETURN p.name, COUNT { MATCH (p)-[:KNOWS]->() } AS friends \
                 ORDER BY p.name",
            )
            .unwrap();
        // One row per Person: Alix, Dave, Gus, Harm
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn map_projection() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p { .name, .age }")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        // Should return a map value
        match &result.rows()[0][0] {
            Value::Map(m) => {
                let keys: Vec<String> = m.keys().map(|k| k.as_str().to_string()).collect();
                assert!(keys.contains(&"name".to_string()), "Map should have 'name'");
                assert!(keys.contains(&"age".to_string()), "Map should have 'age'");
            }
            other => panic!("Expected Map, got {other:?}"),
        }
    }

    #[test]
    fn reduce_function() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Item {nums: [1, 2, 3, 4, 5]})")
            .unwrap();
        let result = session
            .execute_cypher(
                "MATCH (n:Item) RETURN reduce(total = 0, x IN n.nums | total + x) AS sum",
            )
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(15));
    }

    #[test]
    fn cypher_case_inside_aggregate() {
        let db = social_network();
        let session = db.session();
        // sum(CASE WHEN ... THEN 1 ELSE 0 END) is a common conditional-count pattern
        let result = session
            .execute_cypher(
                "MATCH (p:Person) \
                 RETURN sum(CASE WHEN p.age >= 30 THEN 1 ELSE 0 END) AS over_30, \
                        sum(CASE WHEN p.age < 30 THEN 1 ELSE 0 END) AS under_30",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        // Alix=30, Gus=25, Harm=35, Dave=28 => over_30=2 (Alix, Harm), under_30=2 (Gus, Dave)
        let over_30 = &result.rows()[0][0];
        let under_30 = &result.rows()[0][1];
        assert_eq!(*over_30, Value::Int64(2), "Expected 2 people aged >= 30");
        assert_eq!(*under_30, Value::Int64(2), "Expected 2 people aged < 30");
    }

    #[test]
    fn cypher_math_functions() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE n.name = 'Alix' \
                 RETURN sign(n.age) AS s, abs(-5) AS a",
            )
            .unwrap();
        // sign(30) = 1
        assert_eq!(result.rows()[0][0], Value::Int64(1));
    }

    #[test]
    fn cypher_string_functions() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (n:Person) WHERE n.name = 'Alix' \
                 RETURN left(n.name, 3) AS l, right(n.name, 3) AS r",
            )
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::String("Ali".into()));
        assert_eq!(result.rows()[0][1], Value::String("lix".into()));
    }

    #[test]
    fn cypher_trig_functions() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Val {x: 0.0})").unwrap();
        let result = session
            .execute_cypher("MATCH (n:Val) RETURN sin(n.x) AS s, cos(n.x) AS c")
            .unwrap();
        match &result.rows()[0][0] {
            Value::Float64(f) => assert!((f - 0.0).abs() < 0.001, "sin(0) should be 0"),
            other => panic!("Expected Float64 for sin, got {other:?}"),
        }
        match &result.rows()[0][1] {
            Value::Float64(f) => assert!((f - 1.0).abs() < 0.001, "cos(0) should be 1"),
            other => panic!("Expected Float64 for cos, got {other:?}"),
        }
    }

    #[test]
    fn cypher_pi_and_e() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:X {x: 1})").unwrap();
        let result = session
            .execute_cypher("MATCH (n:X) RETURN pi() AS p, e() AS e")
            .unwrap();
        match &result.rows()[0][0] {
            Value::Float64(f) => assert!((f - std::f64::consts::PI).abs() < 0.001),
            other => panic!("Expected Float64, got {other:?}"),
        }
    }

    #[test]
    fn cypher_power_operator() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:X {x: 3})").unwrap();
        let result = session
            .execute_cypher("MATCH (n:X) RETURN n.x ^ 2 AS sq")
            .unwrap();
        match &result.rows()[0][0] {
            Value::Float64(f) => assert!((f - 9.0).abs() < 0.001),
            Value::Int64(n) => assert_eq!(*n, 9),
            other => panic!("Expected numeric, got {other:?}"),
        }
    }

    #[test]
    fn cypher_create_index() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Set up data so there's a label in the catalog
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        let result =
            session.execute_cypher("CREATE INDEX idx_person_name FOR (n:Person) ON (n.name)");
        assert!(
            result.is_ok(),
            "CREATE INDEX should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_create_index_if_not_exists() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute_cypher("CREATE INDEX idx_test IF NOT EXISTS FOR (n:Person) ON (n.name)")
            .unwrap();
        // Running again should not error with IF NOT EXISTS
        let result = session
            .execute_cypher("CREATE INDEX idx_test IF NOT EXISTS FOR (n:Person) ON (n.name)");
        assert!(result.is_ok());
    }

    #[test]
    fn cypher_drop_index() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session
            .execute_cypher("CREATE INDEX idx_drop FOR (n:Person) ON (n.name)")
            .unwrap();
        // DROP by index name (catalog tracks indexes by their user-defined name)
        let result = session.execute_cypher("DROP INDEX idx_drop");
        assert!(
            result.is_ok(),
            "DROP INDEX should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_drop_index_if_exists() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        // Dropping non-existent index with IF EXISTS should not error
        let result = session.execute_cypher("DROP INDEX nonexistent IF EXISTS");
        assert!(result.is_ok());
    }

    #[test]
    fn cypher_create_constraint() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let result = session.execute_cypher(
            "CREATE CONSTRAINT unique_name FOR (n:Person) REQUIRE n.name IS UNIQUE",
        );
        assert!(
            result.is_ok(),
            "CREATE CONSTRAINT should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_show_indexes() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher("SHOW INDEXES");
        assert!(
            result.is_ok(),
            "SHOW INDEXES should succeed: {:?}",
            result.err()
        );
        let qr = result.unwrap();
        assert_eq!(qr.columns.len(), 4);
        assert_eq!(qr.columns[0], "name");
    }

    #[test]
    fn cypher_show_constraints() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher("SHOW CONSTRAINTS");
        assert!(
            result.is_ok(),
            "SHOW CONSTRAINTS should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_relationship_where_clause() {
        let db = social_network();
        let session = db.session();
        // Inline WHERE on relationship pattern (Neo4j 5.x syntax)
        let result = session.execute_cypher(
            "MATCH (p:Person)-[r:KNOWS WHERE r.since IS NOT NULL]->(f:Person) \
                 RETURN p.name, f.name ORDER BY p.name",
        );
        // Should parse and execute without errors regardless of data
        assert!(
            result.is_ok(),
            "Relationship WHERE should work: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_label_check_in_where() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher("MATCH (n) WHERE n:Engineer RETURN n.name")
            .unwrap();
        assert_eq!(result.row_count(), 1); // Only Dave
    }

    #[test]
    fn cypher_exists_subquery() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:WORKS_AT]->(:Company) } \
                 RETURN p.name ORDER BY p.name",
            )
            .unwrap();
        // Alix, Dave, Gus all WORKS_AT TechCorp
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn cypher_exists_bare_pattern() {
        let db = social_network();
        let session = db.session();
        // Bare pattern form: no explicit MATCH keyword inside EXISTS
        let result = session
            .execute_cypher(
                "MATCH (p:Person) WHERE EXISTS { (p)-[:WORKS_AT]->(:Company) } \
                 RETURN p.name ORDER BY p.name",
            )
            .unwrap();
        // Same as explicit MATCH version: Alix, Dave, Gus
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn cypher_exists_bare_pattern_with_where() {
        let db = social_network();
        let session = db.session();
        // Bare pattern with WHERE inside EXISTS
        let result = session.execute_cypher(
            "MATCH (p:Person) WHERE EXISTS { (p)-[r:KNOWS]->() WHERE r.since IS NOT NULL } \
                 RETURN p.name ORDER BY p.name",
        );
        // This may or may not return results depending on test data,
        // but it should parse and execute without errors
        assert!(
            result.is_ok(),
            "Bare pattern EXISTS with WHERE should parse: {:?}",
            result.err()
        );
    }

    #[test]
    fn cypher_foreach_set() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'A', verified: false})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'B', verified: false})")
            .unwrap();
        session
            .execute_cypher(
                "MATCH (n:Person) WITH collect(n) AS people \
                 FOREACH (p IN people | SET p.verified = true) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();
        let result = session
            .execute("MATCH (n:Person) WHERE n.verified = true RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(2));
    }

    #[test]
    fn cypher_call_inline_subquery() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person) \
                 CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS cnt } \
                 RETURN p.name, cnt ORDER BY p.name",
            )
            .unwrap();
        // One row per Person: Alix, Dave, Gus, Harm
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn cypher_pattern_comprehension() {
        let db = social_network();
        let session = db.session();
        let result = session
            .execute_cypher(
                "MATCH (p:Person) WHERE p.name = 'Alix' \
                 RETURN p.name, [(p)-[:KNOWS]->(f) | f.name] AS friends",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        match &result.rows()[0][1] {
            Value::List(items) => assert_eq!(items.len(), 2, "Alix knows exactly Gus and Harm"),
            other => panic!("Expected list of friends, got {other:?}"),
        }
    }

    #[test]
    fn cypher_skip_with_parameter() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("offset".to_string(), Value::Int64(2));
        let result = db
            .execute_cypher_with_params(
                "MATCH (p:Person) RETURN p.name ORDER BY p.name SKIP $offset",
                params,
            )
            .unwrap();
        // 4 people total, skip 2 = 2 results
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn cypher_limit_with_parameter() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("count".to_string(), Value::Int64(2));
        let result = db
            .execute_cypher_with_params(
                "MATCH (p:Person) RETURN p.name ORDER BY p.name LIMIT $count",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn cypher_skip_and_limit_with_parameters() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("offset".to_string(), Value::Int64(1));
        params.insert("page_size".to_string(), Value::Int64(2));
        let result = db
            .execute_cypher_with_params(
                "MATCH (p:Person) RETURN p.name ORDER BY p.name SKIP $offset LIMIT $page_size",
                params,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
        // Ordered by name: Alix(0), Dave(1), Gus(2), Harm(3). Skip 1 = Dave first.
        assert_eq!(result.rows()[0][0], Value::String("Dave".into()));
    }

    #[test]
    fn cypher_limit_parameter_zero() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("n".to_string(), Value::Int64(0));
        let result = db
            .execute_cypher_with_params("MATCH (p:Person) RETURN p.name LIMIT $n", params)
            .unwrap();
        assert_eq!(result.row_count(), 0);
    }

    #[test]
    fn cypher_skip_parameter_negative_is_error() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("n".to_string(), Value::Int64(-1));
        let result =
            db.execute_cypher_with_params("MATCH (p:Person) RETURN p.name SKIP $n", params);
        assert!(result.is_err());
    }

    #[test]
    fn cypher_limit_parameter_non_integer_is_error() {
        let db = social_network();
        let mut params = std::collections::HashMap::new();
        params.insert("n".to_string(), Value::String("ten".into()));
        let result =
            db.execute_cypher_with_params("MATCH (p:Person) RETURN p.name LIMIT $n", params);
        assert!(result.is_err());
    }

    #[test]
    fn cypher_load_csv_with_headers() {
        use std::io::Write;
        // Create a temp CSV file
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_load_csv_headers.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,age,city").unwrap();
            writeln!(f, "Alix,30,Amsterdam").unwrap();
            writeln!(f, "Gus,25,Berlin").unwrap();
            writeln!(f, "Mia,28,Paris").unwrap();
        }

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row RETURN row.name AS name, row.age AS age ORDER BY row.name",
            csv_path.display()
        );
        let result = session.execute_cypher(&query);
        assert!(
            result.is_ok(),
            "LOAD CSV WITH HEADERS failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.row_count(), 3, "Should have 3 rows");
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[0][1], Value::String("30".into())); // CSV values are strings
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
        assert_eq!(result.rows()[2][0], Value::String("Mia".into()));

        std::fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn cypher_load_csv_without_headers() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_load_csv_no_headers.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "Alix,30,Amsterdam").unwrap();
            writeln!(f, "Gus,25,Berlin").unwrap();
        }

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV FROM '{}' AS row RETURN row[0] AS name, row[1] AS age",
            csv_path.display()
        );
        let result = session.execute_cypher(&query);
        assert!(
            result.is_ok(),
            "LOAD CSV without headers failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[0][1], Value::String("30".into()));

        std::fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn cypher_load_csv_create_nodes() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_load_csv_create.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,city").unwrap();
            writeln!(f, "Vincent,Amsterdam").unwrap();
            writeln!(f, "Jules,Paris").unwrap();
        }

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row CREATE (p:Person {{name: row.name, city: row.city}})",
            csv_path.display()
        );
        let result = session.execute_cypher(&query);
        assert!(
            result.is_ok(),
            "LOAD CSV + CREATE failed: {:?}",
            result.err()
        );

        // Verify nodes were created
        let check = session
            .execute_cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(check.row_count(), 2);
        assert_eq!(check.rows()[0][0], Value::String("Jules".into()));
        assert_eq!(check.rows()[1][0], Value::String("Vincent".into()));

        std::fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn cypher_load_csv_with_fieldterminator() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_load_csv_tab.tsv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name\tage").unwrap();
            writeln!(f, "Alix\t30").unwrap();
            writeln!(f, "Gus\t25").unwrap();
        }

        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row FIELDTERMINATOR '\\t' RETURN row.name, row.age",
            csv_path.display()
        );
        let result = session.execute_cypher(&query);
        assert!(
            result.is_ok(),
            "LOAD CSV with FIELDTERMINATOR failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));

        std::fs::remove_file(&csv_path).ok();
    }

    #[test]
    fn cypher_load_csv_file_not_found() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute_cypher(
            "LOAD CSV WITH HEADERS FROM '/nonexistent/path/file.csv' AS row RETURN row.name",
        );
        assert!(result.is_err(), "Should fail for missing file");
    }

    #[test]
    fn cypher_load_csv_parse_only() {
        // Verify LOAD CSV parses without executing (EXPLAIN)
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute_cypher("EXPLAIN LOAD CSV WITH HEADERS FROM 'test.csv' AS row RETURN row.name");
        assert!(
            result.is_ok(),
            "EXPLAIN LOAD CSV should parse: {:?}",
            result.err()
        );
    }
}

// ============================================================================
// LOAD DATA Features (GQL LOAD DATA, JSONL, Parquet)
// ============================================================================

#[cfg(feature = "gql")]
mod load_data_features {
    use grafeo_engine::GrafeoDB;
    use std::io::Write;

    #[test]
    fn gql_load_data_csv_with_headers() {
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_gql_load_csv.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,age,city").unwrap();
            writeln!(f, "Alix,30,Amsterdam").unwrap();
            writeln!(f, "Gus,25,Berlin").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name, row.age AS age ORDER BY row.name",
            csv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA CSV failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows().len(), 2);
    }

    #[test]
    fn gql_load_data_csv_without_headers() {
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_gql_load_csv_no_headers.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "Alix,30,Amsterdam").unwrap();
            writeln!(f, "Gus,25,Berlin").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT CSV AS row RETURN row[0] AS name, row[1] AS age",
            csv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA CSV without headers failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows().len(), 2);
    }

    #[test]
    fn gql_load_csv_compat_syntax() {
        // GQL parser also accepts Cypher-compatible LOAD CSV syntax
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_gql_load_csv_compat.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,city").unwrap();
            writeln!(f, "Alix,Amsterdam").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row RETURN row.name AS name",
            csv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD CSV compat failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows().len(), 1);
    }

    #[test]
    fn gql_load_data_csv_create_nodes() {
        let dir = std::env::temp_dir();
        let csv_path = dir.join("grafeo_test_gql_load_csv_create.csv");
        {
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "name,city").unwrap();
            writeln!(f, "Alix,Amsterdam").unwrap();
            writeln!(f, "Gus,Berlin").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row INSERT (:Person {{name: row.name, city: row.city}})",
            csv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA + INSERT failed: {:?}",
            result.err()
        );

        // Verify nodes were created
        let verify = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(verify.rows().len(), 2);
    }

    #[test]
    fn gql_load_data_csv_with_fieldterminator() {
        let dir = std::env::temp_dir();
        let tsv_path = dir.join("grafeo_test_gql_load_csv_tab.tsv");
        {
            let mut f = std::fs::File::create(&tsv_path).unwrap();
            writeln!(f, "name\tage").unwrap();
            writeln!(f, "Alix\t30").unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row FIELDTERMINATOR '\\t' RETURN row.name AS name",
            tsv_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA with FIELDTERMINATOR failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows().len(), 1);
    }

    #[test]
    fn gql_load_data_file_not_found() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session
            .execute("LOAD DATA FROM '/nonexistent/path/file.csv' FORMAT CSV AS row RETURN row");
        assert!(result.is_err(), "Should fail for missing file");
    }

    #[test]
    fn gql_load_data_explain() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let result = session.execute(
            "EXPLAIN LOAD DATA FROM 'test.csv' FORMAT CSV WITH HEADERS AS row RETURN row.name",
        );
        assert!(
            result.is_ok(),
            "EXPLAIN LOAD DATA should parse: {:?}",
            result.err()
        );
    }

    #[cfg(feature = "jsonl-import")]
    #[test]
    fn gql_load_data_jsonl() {
        let dir = std::env::temp_dir();
        let jsonl_path = dir.join("grafeo_test_gql_load_jsonl.jsonl");
        {
            let mut f = std::fs::File::create(&jsonl_path).unwrap();
            writeln!(f, r#"{{"name": "Alix", "age": 30, "city": "Amsterdam"}}"#).unwrap();
            writeln!(f, r#"{{"name": "Gus", "age": 25, "city": "Berlin"}}"#).unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT JSONL AS row RETURN row.name AS name, row.age AS age ORDER BY row.name",
            jsonl_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA JSONL failed: {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.rows().len(), 2);
    }

    #[cfg(feature = "jsonl-import")]
    #[test]
    fn gql_load_data_jsonl_create_nodes() {
        let dir = std::env::temp_dir();
        let jsonl_path = dir.join("grafeo_test_gql_load_jsonl_create.jsonl");
        {
            let mut f = std::fs::File::create(&jsonl_path).unwrap();
            writeln!(f, r#"{{"name": "Vincent", "city": "Paris"}}"#).unwrap();
            writeln!(f, r#"{{"name": "Jules", "city": "Amsterdam"}}"#).unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT JSONL AS row INSERT (:Person {{name: row.name, city: row.city}})",
            jsonl_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA JSONL + INSERT failed: {:?}",
            result.err()
        );

        let verify = session
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(verify.rows().len(), 2);
    }

    #[cfg(feature = "jsonl-import")]
    #[test]
    fn gql_load_data_ndjson_alias() {
        // NDJSON is an alias for JSONL
        let dir = std::env::temp_dir();
        let jsonl_path = dir.join("grafeo_test_gql_load_ndjson.jsonl");
        {
            let mut f = std::fs::File::create(&jsonl_path).unwrap();
            writeln!(f, r#"{{"x": 1}}"#).unwrap();
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let query = format!(
            "LOAD DATA FROM '{}' FORMAT NDJSON AS row RETURN row.x AS x",
            jsonl_path.display()
        );
        let result = session.execute(&query);
        assert!(
            result.is_ok(),
            "GQL LOAD DATA NDJSON alias failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn gql_load_data_parquet_disabled_error() {
        // When parquet-import feature is disabled, should give a clear error
        if cfg!(not(feature = "parquet-import")) {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            let result =
                session.execute("LOAD DATA FROM 'test.parquet' FORMAT PARQUET AS row RETURN row");
            assert!(
                result.is_err(),
                "Should fail when parquet-import is disabled"
            );
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("Parquet") || err.contains("parquet"),
                "Error should mention Parquet: {err}"
            );
        }
    }
}

// ============================================================================
// SPARQL Features (covers sparql_translator.rs)
// ============================================================================

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod sparql_features {
    use grafeo_common::types::Value;
    use grafeo_engine::{Config, GrafeoDB, GraphModel};

    fn rdf_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
    }

    #[test]
    fn inverse_property_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/knows> <http://ex.org/harm> .
                }"#,
            )
            .unwrap();
        // ^knows means inverse: find who knows gus
        let result = session
            .execute_sparql(
                r#"SELECT ?who WHERE {
                    <http://ex.org/gus> ^<http://ex.org/knows> ?who
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Alix knows Gus (inverse)");
    }

    #[test]
    fn zero_or_one_property_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                }"#,
            )
            .unwrap();
        // knows? means zero or one hop
        let result = session
            .execute_sparql(
                r#"SELECT ?who WHERE {
                    <http://ex.org/alix> <http://ex.org/knows>? ?who
                }"#,
            )
            .unwrap();
        // Should find alix herself (0 hops) and gus (1 hop)
        assert!(
            result.row_count() >= 1,
            "ZeroOrOne should find at least one match"
        );
    }

    #[test]
    fn sparql_optional_pattern() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/age> "30" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/age> ?age }
                }"#,
            )
            .unwrap();
        // Both Alix and Gus, but only Alix has age
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn sparql_optional_null_values() {
        // Verify that unbound variables from OPTIONAL produce NULL in results.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/age> "30" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?age WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/age> ?age }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
        // Alix has age, Gus does not
        // Verify Gus row has NULL for age
        let gus_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Gus".into()))
            .expect("Gus should appear in results");
        assert_eq!(gus_row[1], Value::Null, "Gus has no age, should be NULL");
    }

    #[test]
    fn sparql_nested_optional() {
        // Nested OPTIONAL: OPTIONAL { ?x <p> ?y OPTIONAL { ?y <q> ?z } }
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/gus> <http://ex.org/city> "Amsterdam" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?friend ?city WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL {
                        ?s <http://ex.org/knows> ?f .
                        ?f <http://ex.org/name> ?friend .
                        OPTIONAL { ?f <http://ex.org/city> ?city }
                    }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix knows Gus (who has city Amsterdam): Alix, "Gus", "Amsterdam"
        // Gus knows nobody: Gus, NULL, NULL
        // Harm knows nobody: Harm, NULL, NULL
        assert_eq!(result.row_count(), 3);
        let alix_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(alix_row[1], Value::String("Gus".into()));
        assert_eq!(alix_row[2], Value::String("Amsterdam".into()));

        let harm_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(harm_row[1], Value::Null, "Harm knows nobody");
        assert_eq!(harm_row[2], Value::Null, "Nested optional also NULL");
    }

    #[test]
    fn sparql_optional_with_filter_inside() {
        // SPARQL semantics: FILTER inside OPTIONAL acts as a join condition,
        // not a post-filter. Persons without a matching score should get NULL,
        // not be eliminated.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/score> "80" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/gus> <http://ex.org/score> "40" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();

        let result = session
            .execute_sparql(
                r#"SELECT ?name ?score WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL {
                        ?s <http://ex.org/score> ?score .
                        FILTER(?score > "50")
                    }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // All 3 persons should appear:
        // Alix: score "80" > "50" -> bound
        // Gus: score "40" NOT > "50" -> NULL (filter eliminates inside optional)
        // Harm: no score -> NULL
        assert_eq!(
            result.row_count(),
            3,
            "All 3 persons preserved: FILTER inside OPTIONAL is a join condition"
        );
        let alix_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(
            alix_row[1],
            Value::String("80".into()),
            "Alix score passes filter"
        );

        let gus_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Gus".into()))
            .expect("Gus should appear");
        assert_eq!(
            gus_row[1],
            Value::Null,
            "Gus score fails filter, should be NULL"
        );

        let harm_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(
            harm_row[1],
            Value::Null,
            "Harm has no score, should be NULL"
        );
    }

    #[test]
    fn sparql_optional_shared_variables() {
        // Shared variable between required and optional patterns.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?friend WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/knows> ?f . ?f <http://ex.org/name> ?friend }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix knows Gus -> Alix, Gus
        // Gus knows nobody -> Gus, NULL
        assert_eq!(result.row_count(), 2);
        let alix_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(alix_row[1], Value::String("Gus".into()));

        let gus_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Gus".into()))
            .expect("Gus should appear");
        assert_eq!(
            gus_row[1],
            Value::Null,
            "Gus has no knows, friend should be NULL"
        );
    }

    #[test]
    fn sparql_union_pattern() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/gus> <http://ex.org/label> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?val WHERE {
                    { ?s <http://ex.org/name> ?val }
                    UNION
                    { ?s <http://ex.org/label> ?val }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn sparql_filter_exists() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://ex.org/name> ?name .
                    FILTER EXISTS { ?s <http://ex.org/knows> ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Only Alix has knows relation");
    }

    #[test]
    fn sparql_filter_not_exists() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name WHERE {
                    ?s <http://ex.org/name> ?name .
                    FILTER NOT EXISTS { ?s <http://ex.org/knows> ?o }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Only Harm has no knows relation");
    }

    #[test]
    fn sparql_sequence_property_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/likes> <http://ex.org/harm> .
                }"#,
            )
            .unwrap();
        // Sequence: knows / likes
        let result = session
            .execute_sparql(
                r#"SELECT ?who WHERE {
                    <http://ex.org/alix> <http://ex.org/knows>/<http://ex.org/likes> ?who
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn sparql_alternative_property_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/alix> <http://ex.org/likes> <http://ex.org/harm> .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?who WHERE {
                    <http://ex.org/alix> (<http://ex.org/knows>|<http://ex.org/likes>) ?who
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2, "Alternative path finds both");
    }

    #[test]
    fn sparql_one_or_more_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/next> <http://ex.org/b> .
                    <http://ex.org/b> <http://ex.org/next> <http://ex.org/c> .
                    <http://ex.org/c> <http://ex.org/next> <http://ex.org/d> .
                }"#,
            )
            .unwrap();
        // + means one or more hops
        let result = session
            .execute_sparql(
                r#"SELECT ?end WHERE {
                    <http://ex.org/a> <http://ex.org/next>+ ?end
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 3, "One-or-more should find b, c, d");
    }

    #[test]
    fn sparql_zero_or_more_path() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/next> <http://ex.org/b> .
                    <http://ex.org/b> <http://ex.org/next> <http://ex.org/c> .
                }"#,
            )
            .unwrap();
        // * means zero or more hops (includes self)
        let result = session
            .execute_sparql(
                r#"SELECT ?end WHERE {
                    <http://ex.org/a> <http://ex.org/next>* ?end
                }"#,
            )
            .unwrap();
        // Zero-or-more from fixed subject: 0-hop (a), 1-hop (b), 2-hop (c) = 3 results
        assert_eq!(
            result.row_count(),
            3,
            "Zero-or-more from a with chain a->b->c"
        );
    }

    #[test]
    fn sparql_rdf_collections() {
        let db = rdf_db();
        let session = db.session();
        // RDF collection: linked list with rdf:first/rdf:rest
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/list> <http://ex.org/items> <http://ex.org/a> .
                    <http://ex.org/a> <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> "one" .
                    <http://ex.org/a> <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://ex.org/b> .
                    <http://ex.org/b> <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> "two" .
                    <http://ex.org/b> <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
                }"#,
            )
            .unwrap();
        // rest*/first: follow rest links zero or more times, then get rdf:first
        let result = session
            .execute_sparql(
                r#"SELECT ?item WHERE {
                    <http://ex.org/list> <http://ex.org/items> ?head .
                    ?head <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>*/<http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?item
                }"#,
            )
            .unwrap();
        // Should find "one" (0 rest hops + first) and "two" (1 rest hop + first)
        assert!(
            result.row_count() >= 2,
            "RDF collection traversal should find at least 2 items, got {}",
            result.row_count()
        );
    }

    #[test]
    fn sparql_aggregation_count() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/type> "person" .
                    <http://ex.org/gus> <http://ex.org/type> "person" .
                    <http://ex.org/techcorp> <http://ex.org/type> "company" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?type (COUNT(?s) AS ?count) WHERE {
                    ?s <http://ex.org/type> ?type
                } GROUP BY ?type ORDER BY ?type"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn sparql_having_clause() {
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/a> <http://ex.org/type> "x" .
                    <http://ex.org/b> <http://ex.org/type> "x" .
                    <http://ex.org/c> <http://ex.org/type> "y" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?type (COUNT(?s) AS ?cnt) WHERE {
                    ?s <http://ex.org/type> ?type
                } GROUP BY ?type HAVING (COUNT(?s) > 1)"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1, "Only 'x' has count > 1");
    }

    // --- P0 OPTIONAL bug probes ---

    #[test]
    fn sparql_optional_multiple_independent() {
        // Two independent OPTIONALLs: one matches, one doesn't.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/age> "30" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/gus> <http://ex.org/email> "gus@example.org" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?age ?email WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/age> ?age }
                    OPTIONAL { ?s <http://ex.org/email> ?email }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix: age=30, email=NULL
        // Gus: age=NULL, email=gus@example.org
        // Harm: age=NULL, email=NULL
        assert_eq!(
            result.row_count(),
            3,
            "All 3 persons should appear with independent OPTIONALLs"
        );
        let alix_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(alix_row[1], Value::String("30".into()), "Alix has age");
        assert_eq!(alix_row[2], Value::Null, "Alix has no email");

        let gus_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Gus".into()))
            .expect("Gus should appear");
        assert_eq!(gus_row[1], Value::Null, "Gus has no age");
        assert_eq!(
            gus_row[2],
            Value::String("gus@example.org".into()),
            "Gus has email"
        );

        let harm_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(harm_row[1], Value::Null, "Harm has no age");
        assert_eq!(harm_row[2], Value::Null, "Harm has no email");
    }

    #[test]
    fn sparql_optional_count_with_null() {
        // COUNT(expr) should skip NULLs, COUNT(*) should count all rows.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/score> "80" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                    <http://ex.org/harm> <http://ex.org/score> "60" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT (COUNT(?name) AS ?total) (COUNT(?score) AS ?with_score) WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/score> ?score }
                }"#,
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        // 3 people total, 2 with scores
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(3),
            "COUNT(?name) should be 3"
        );
        assert_eq!(
            result.rows()[0][1],
            Value::Int64(2),
            "COUNT(?score) should skip NULLs: 2"
        );
    }

    #[test]
    fn sparql_optional_inner_join_inside() {
        // OPTIONAL block with a join inside (two triple patterns).
        // Only matches if BOTH patterns match for the same binding.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                    <http://ex.org/gus> <http://ex.org/city> "Amsterdam" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                    <http://ex.org/harm> <http://ex.org/knows> <http://ex.org/jules> .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?city WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL {
                        ?s <http://ex.org/knows> ?friend .
                        ?friend <http://ex.org/city> ?city
                    }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix: knows Gus who has city Amsterdam
        // Harm: knows Jules who has NO city -> NULL
        assert_eq!(
            result.row_count(),
            2,
            "Both people should appear even if inner join of optional fails"
        );
        let alix_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Alix".into()))
            .expect("Alix should appear");
        assert_eq!(alix_row[1], Value::String("Amsterdam".into()));

        let harm_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(
            harm_row[1],
            Value::Null,
            "Harm's friend has no city, optional fails -> NULL"
        );
    }

    #[test]
    fn sparql_optional_all_unbound() {
        // Every person has no optional match at all.
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?email WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL { ?s <http://ex.org/email> ?email }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Both rows should appear with NULL email
        assert_eq!(result.row_count(), 2, "Both persons should appear");
        assert_eq!(result.rows()[0][1], Value::Null, "Alix: no email");
        assert_eq!(result.rows()[1][1], Value::Null, "Gus: no email");
    }

    #[test]
    fn sparql_optional_with_union_inside() {
        // OPTIONAL { { ?s <p1> ?val } UNION { ?s <p2> ?val } }
        let db = rdf_db();
        let session = db.session();
        session
            .execute_sparql(
                r#"INSERT DATA {
                    <http://ex.org/alix> <http://ex.org/name> "Alix" .
                    <http://ex.org/alix> <http://ex.org/nick> "Lix" .
                    <http://ex.org/gus> <http://ex.org/name> "Gus" .
                    <http://ex.org/gus> <http://ex.org/label> "Gustav" .
                    <http://ex.org/harm> <http://ex.org/name> "Harm" .
                }"#,
            )
            .unwrap();
        let result = session
            .execute_sparql(
                r#"SELECT ?name ?alt WHERE {
                    ?s <http://ex.org/name> ?name .
                    OPTIONAL {
                        { ?s <http://ex.org/nick> ?alt }
                        UNION
                        { ?s <http://ex.org/label> ?alt }
                    }
                }
                ORDER BY ?name"#,
            )
            .unwrap();
        // Alix: alt=Lix (from nick)
        // Gus: alt=Gustav (from label)
        // Harm: alt=NULL
        assert_eq!(
            result.row_count(),
            3,
            "All 3 persons should appear, UNION inside OPTIONAL"
        );
        let harm_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should appear");
        assert_eq!(harm_row[1], Value::Null, "Harm has neither nick nor label");
    }
}

// ============================================================================
// GQL Parser Unit Tests (covers DDL/session parsing branches)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_parser_unit {
    use grafeo_adapters::query::gql;
    use grafeo_adapters::query::gql::ast::{SessionCommand, SessionResetTarget, Statement};

    #[test]
    fn parse_create_graph() {
        let stmt = gql::parse("CREATE GRAPH mydb").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::CreateGraph {
                name,
                if_not_exists,
                ..
            }) => {
                assert_eq!(name, "mydb");
                assert!(!if_not_exists);
            }
            other => panic!("Expected CreateGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_create_graph_if_not_exists() {
        let stmt = gql::parse("CREATE GRAPH IF NOT EXISTS mydb").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::CreateGraph {
                name,
                if_not_exists,
                ..
            }) => {
                assert_eq!(name, "mydb");
                assert!(if_not_exists);
            }
            other => panic!("Expected CreateGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_create_property_graph() {
        let stmt = gql::parse("CREATE PROPERTY GRAPH pg1").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::CreateGraph { name, .. }) => {
                assert_eq!(name, "pg1");
            }
            other => panic!("Expected CreateGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_drop_graph() {
        let stmt = gql::parse("DROP GRAPH mydb").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::DropGraph { name, if_exists }) => {
                assert_eq!(name, "mydb");
                assert!(!if_exists);
            }
            other => panic!("Expected DropGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_drop_graph_if_exists() {
        let stmt = gql::parse("DROP GRAPH IF EXISTS mydb").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::DropGraph { name, if_exists }) => {
                assert_eq!(name, "mydb");
                assert!(if_exists);
            }
            other => panic!("Expected DropGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_drop_property_graph() {
        let stmt = gql::parse("DROP PROPERTY GRAPH pg1").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::DropGraph { name, .. }) => {
                assert_eq!(name, "pg1");
            }
            other => panic!("Expected DropGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_use_graph() {
        let stmt = gql::parse("USE GRAPH workspace").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::UseGraph(name)) => {
                assert_eq!(name, "workspace");
            }
            other => panic!("Expected UseGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_set_graph() {
        let stmt = gql::parse("SESSION SET GRAPH analytics").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::SessionSetGraph(name)) => {
                assert_eq!(name, "analytics");
            }
            other => panic!("Expected SessionSetGraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_set_time_zone() {
        let stmt = gql::parse("SESSION SET TIME ZONE 'UTC+5'").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::SessionSetTimeZone(tz)) => {
                assert_eq!(tz, "UTC+5");
            }
            other => panic!("Expected SessionSetTimeZone, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_set_schema() {
        let stmt = gql::parse("SESSION SET SCHEMA myschema").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::SessionSetSchema(name)) => {
                assert_eq!(name, "myschema");
            }
            other => panic!("Expected SessionSetSchema, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_set_parameter() {
        let stmt = gql::parse("SESSION SET PARAMETER timeout = 30").unwrap();
        match stmt {
            Statement::SessionCommand(SessionCommand::SessionSetParameter(name, _)) => {
                assert_eq!(name, "timeout");
            }
            other => panic!("Expected SessionSetParameter, got {other:?}"),
        }
    }

    #[test]
    fn parse_session_reset() {
        let stmt = gql::parse("SESSION RESET").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::All))
        ));
    }

    #[test]
    fn parse_session_reset_all() {
        let stmt = gql::parse("SESSION RESET ALL").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::SessionReset(SessionResetTarget::All))
        ));
    }

    #[test]
    fn parse_session_close() {
        let stmt = gql::parse("SESSION CLOSE").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::SessionClose)
        ));
    }

    #[test]
    fn parse_start_transaction() {
        let stmt = gql::parse("START TRANSACTION").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::StartTransaction { .. })
        ));
    }

    #[test]
    fn parse_commit() {
        let stmt = gql::parse("COMMIT").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::Commit)
        ));
    }

    #[test]
    fn parse_rollback() {
        let stmt = gql::parse("ROLLBACK").unwrap();
        assert!(matches!(
            stmt,
            Statement::SessionCommand(SessionCommand::Rollback)
        ));
    }

    #[test]
    fn parse_finish_statement() {
        let stmt = gql::parse("MATCH (n) FINISH").unwrap();
        match stmt {
            Statement::Query(q) => {
                assert!(q.return_clause.is_finish);
            }
            other => panic!("Expected Query with FINISH, got {other:?}"),
        }
    }

    #[test]
    fn parse_select_statement() {
        let stmt = gql::parse("MATCH (n:Person) SELECT n.name").unwrap();
        match stmt {
            Statement::Query(q) => {
                assert!(!q.return_clause.items.is_empty());
            }
            other => panic!("Expected Query with SELECT, got {other:?}"),
        }
    }

    #[test]
    fn parse_drop_graph_error_on_bad_syntax() {
        let result = gql::parse("DROP NOTHING");
        assert!(result.is_err());
    }

    #[test]
    fn parse_session_error_on_bad_action() {
        let result = gql::parse("SESSION DESTROY");
        assert!(result.is_err());
    }

    #[test]
    fn parse_start_error_without_transaction() {
        let result = gql::parse("START SOMETHING");
        assert!(result.is_err());
    }

    #[test]
    fn parse_use_error_without_graph() {
        let result = gql::parse("USE SOMETHING");
        assert!(result.is_err());
    }
}

// ============================================================================
// GQL Translator Unit Tests (covers translate_full, session command routing)
// ============================================================================

#[cfg(feature = "gql")]
mod gql_translator_unit {
    use grafeo_engine::query::translators::gql;

    #[test]
    fn translate_returns_plan_for_query() {
        let result = gql::translate("MATCH (n) RETURN n");
        assert!(result.is_ok());
    }

    #[test]
    fn translate_returns_error_for_session_command() {
        let result = gql::translate("CREATE GRAPH test");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Session commands"));
    }

    #[test]
    fn translate_full_returns_session_command() {
        let result = gql::translate_full("CREATE GRAPH test").unwrap();
        assert!(matches!(
            result,
            gql::GqlTranslationResult::SessionCommand(_)
        ));
    }

    #[test]
    fn translate_full_returns_plan_for_query() {
        let result = gql::translate_full("MATCH (n) RETURN n").unwrap();
        assert!(matches!(result, gql::GqlTranslationResult::Plan(_)));
    }

    #[test]
    fn translate_except_produces_plan() {
        let result = gql::translate(
            "MATCH (n:Person) RETURN n.name EXCEPT MATCH (m:Person) WHERE m.age > 30 RETURN m.name",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn translate_intersect_produces_plan() {
        let result = gql::translate(
            "MATCH (n:Person) RETURN n.name INTERSECT MATCH (m:Person) RETURN m.name",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn translate_otherwise_produces_plan() {
        let result = gql::translate(
            "MATCH (n:Person) RETURN n.name OTHERWISE MATCH (m:Person) RETURN m.name",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn translate_finish_produces_plan() {
        let result = gql::translate("MATCH (n) FINISH");
        assert!(result.is_ok());
    }

    #[test]
    fn translate_element_where_produces_plan() {
        let result = gql::translate("MATCH (n:Person WHERE n.age > 25) RETURN n.name");
        assert!(result.is_ok());
    }

    #[test]
    fn translate_count_subquery() {
        let result = gql::translate(
            "MATCH (n:Person) RETURN n.name, COUNT { MATCH (n)-[:KNOWS]->() } AS cnt",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn translate_schema_errors() {
        let result = gql::translate("CREATE NODE TYPE Foo");
        assert!(result.is_err());
    }
}
