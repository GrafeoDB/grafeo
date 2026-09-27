//! SQL/PGQ (SQL:2023 GRAPH_TABLE) Integration Tests
//!
//! Verifies end-to-end query execution through the full pipeline:
//! Parse → Translate → Bind → Optimize → Plan → Execute
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --features sql-pgq --test sql_pgq
//! ```

#![cfg(feature = "sql-pgq")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Creates a social network graph for testing.
///
/// Structure:
/// - Alix (Person, age: 30) -KNOWS-> Gus (Person, age: 25)
/// - Alix -KNOWS-> Harm (Person, age: 35)
/// - Gus -KNOWS-> Harm
fn create_social_network() -> GrafeoDB {
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

    session.create_edge(alix, gus, "KNOWS");
    session.create_edge(alix, harm, "KNOWS");
    session.create_edge(gus, harm, "KNOWS");

    db
}

// ============================================================================
// Basic GRAPH_TABLE Queries
// ============================================================================

#[test]
fn test_basic_node_query() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            )",
        )
        .unwrap();

    assert_eq!(result.row_count(), 3, "Should find 3 Person nodes");
}

#[test]
fn test_relationship_query() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            )",
        )
        .unwrap();

    assert_eq!(result.row_count(), 3, "Should find 3 KNOWS edges");
}

#[test]
fn test_multiple_columns() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, a.age AS age, b.name AS friend)
            )",
        )
        .unwrap();

    assert_eq!(result.row_count(), 3);
    assert_eq!(result.columns.len(), 3);
}

// ============================================================================
// SQL WHERE Clause
// ============================================================================

#[test]
fn test_where_with_table_alias() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT g.name FROM GRAPH_TABLE (
                MATCH (a:Person)
                COLUMNS (a.name AS name, a.age AS age)
            ) AS g
            WHERE g.age > 28",
        )
        .unwrap();

    // Alix (30) and Harm (35) have age > 28
    assert_eq!(result.row_count(), 2, "Should find 2 people with age > 28");
}

#[test]
fn test_where_without_table_alias() {
    let db = create_social_network();
    let session = db.session();

    // SQL WHERE references output column aliases, not graph variables
    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.age AS age)
            ) AS g
            WHERE g.age > 28",
        )
        .unwrap();

    // Alix (30) and Harm (35) have age > 28
    assert_eq!(result.row_count(), 2, "Should find 2 people older than 28");
}

// ============================================================================
// ORDER BY, LIMIT, OFFSET
// ============================================================================

#[test]
fn test_order_by_and_limit() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.age AS age)
            ) AS g
            ORDER BY g.age DESC
            LIMIT 2",
        )
        .unwrap();

    assert_eq!(result.row_count(), 2, "LIMIT should restrict to 2 rows");
}

#[test]
fn test_offset() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.age AS age)
            ) AS g
            ORDER BY g.age ASC
            LIMIT 10
            OFFSET 1",
        )
        .unwrap();

    // 3 total, skip 1 → 2 remaining
    assert_eq!(result.row_count(), 2, "OFFSET 1 should skip first row");
}

// ============================================================================
// SELECT * vs explicit columns
// ============================================================================

#[test]
fn test_select_star() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.age AS age)
            )",
        )
        .unwrap();

    assert_eq!(result.row_count(), 3);
}

// ============================================================================
// Database-level execute_sql
// ============================================================================

#[test]
fn test_database_execute_sql() {
    let db = create_social_network();

    let result = db
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            )",
        )
        .unwrap();

    assert_eq!(result.row_count(), 3);
}

// ============================================================================
// Path Functions (Phase 2)
// ============================================================================

/// Creates a chain graph for path function testing.
///
/// Structure:
/// - A (Person) -KNOWS-> B (Person) -KNOWS-> C (Person) -KNOWS-> D (Person)
fn create_chain_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let a = session
        .create_node_with_props(&["Person"], [("name", Value::String("A".into()))])
        .unwrap();
    let b = session
        .create_node_with_props(&["Person"], [("name", Value::String("B".into()))])
        .unwrap();
    let c = session
        .create_node_with_props(&["Person"], [("name", Value::String("C".into()))])
        .unwrap();
    let d = session
        .create_node_with_props(&["Person"], [("name", Value::String("D".into()))])
        .unwrap();

    session.create_edge(a, b, "KNOWS");
    session.create_edge(b, c, "KNOWS");
    session.create_edge(c, d, "KNOWS");

    db
}

#[test]
fn test_variable_length_path() {
    let db = create_chain_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (src:Person)-[p:KNOWS*1..3]->(dst:Person)
                COLUMNS (src.name AS source, dst.name AS target)
            )",
        )
        .unwrap();

    // A->B (1 hop), A->C (2 hops), A->D (3 hops),
    // B->C (1 hop), B->D (2 hops),
    // C->D (1 hop)
    assert_eq!(result.row_count(), 6, "Should find 6 variable-length paths");
}

#[test]
fn test_length_path_function() {
    let db = create_chain_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (src:Person)-[p:KNOWS*1..3]->(dst:Person)
                COLUMNS (src.name AS source, LENGTH(p) AS distance, dst.name AS target)
            )",
        )
        .unwrap();

    assert_eq!(result.row_count(), 6);
    assert_eq!(result.columns.len(), 3);

    // Verify distance values are present and are integers 1-3
    let distance_col = result
        .columns
        .iter()
        .position(|c| c == "distance")
        .expect("distance column should exist");

    let mut distances: Vec<i64> = result
        .rows()
        .iter()
        .map(|row| match &row[distance_col] {
            Value::Int64(d) => *d,
            other => panic!("Expected Int64 distance, got: {other:?}"),
        })
        .collect();
    distances.sort_unstable();

    // Expected: 1, 1, 1, 2, 2, 3  (three 1-hop, two 2-hop, one 3-hop)
    assert_eq!(distances, vec![1, 1, 1, 2, 2, 3]);
}

#[test]
fn test_nodes_path_function() {
    let db = create_chain_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (src:Person)-[p:KNOWS*1..2]->(dst:Person)
                COLUMNS (src.name AS source, NODES(p) AS path_nodes, dst.name AS target)
            )",
        )
        .unwrap();

    // A->B (1 hop), A->C (2 hops), B->C (1 hop), B->D (2 hops), C->D (1 hop)
    assert_eq!(result.row_count(), 5);

    let nodes_col = result
        .columns
        .iter()
        .position(|c| c == "path_nodes")
        .expect("path_nodes column should exist");

    // Every path_nodes value should be a list
    for row in result.rows() {
        assert!(
            matches!(&row[nodes_col], Value::List(_)),
            "path_nodes should be a list, got: {:?}",
            row[nodes_col]
        );
    }
}

#[test]
fn test_edges_path_function() {
    let db = create_chain_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (src:Person)-[p:KNOWS*1..2]->(dst:Person)
                COLUMNS (src.name AS source, EDGES(p) AS path_edges, dst.name AS target)
            )",
        )
        .unwrap();

    assert_eq!(result.row_count(), 5);

    let edges_col = result
        .columns
        .iter()
        .position(|c| c == "path_edges")
        .expect("path_edges column should exist");

    // Every path_edges value should be a list
    for row in result.rows() {
        assert!(
            matches!(&row[edges_col], Value::List(_)),
            "path_edges should be a list, got: {:?}",
            row[edges_col]
        );
    }
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_syntax_error() {
    let db = create_social_network();
    let session = db.session();

    let result = session.execute_sql("SELECT FROM");
    assert!(result.is_err(), "Should fail on syntax error");
}

#[test]
fn test_missing_columns_clause() {
    let db = create_social_network();
    let session = db.session();

    let result = session.execute_sql(
        "SELECT * FROM GRAPH_TABLE (
            MATCH (n:Person)
        )",
    );
    assert!(result.is_err(), "Should fail without COLUMNS clause");
}

// ============================================================================
// CREATE PROPERTY GRAPH Tests
// ============================================================================

#[test]
fn test_create_property_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute_sql(
            "CREATE PROPERTY GRAPH SocialGraph
             NODE TABLES (
                 Person (id BIGINT PRIMARY KEY, name VARCHAR, age INT)
             )
             EDGE TABLES (
                 KNOWS (
                     id BIGINT PRIMARY KEY,
                     source BIGINT REFERENCES Person(id),
                     target BIGINT REFERENCES Person(id),
                     since INT
                 )
             )",
        )
        .expect("CREATE PROPERTY GRAPH should succeed");

    assert_eq!(result.columns, vec!["status"]);
    assert_eq!(result.rows().len(), 1);
    let status = &result.rows()[0][0];
    match status {
        Value::String(s) => {
            assert!(
                s.contains("SocialGraph"),
                "Status should name the graph: {s}"
            );
            assert!(
                s.contains("1 node tables"),
                "Status should mention 1 node table: {s}"
            );
            assert!(
                s.contains("1 edge tables"),
                "Status should mention 1 edge table: {s}"
            );
        }
        other => panic!("Expected string status, got: {other:?}"),
    }
}

#[test]
fn test_create_property_graph_multiple_tables() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute_sql(
            "CREATE PROPERTY GRAPH CompanyGraph
             NODE TABLES (
                 Person (id BIGINT PRIMARY KEY, name VARCHAR),
                 Company (id BIGINT PRIMARY KEY, name VARCHAR(255), founded INT)
             )
             EDGE TABLES (
                 WORKS_AT (
                     id BIGINT PRIMARY KEY,
                     employee BIGINT REFERENCES Person(id),
                     employer BIGINT REFERENCES Company(id)
                 )
             )",
        )
        .expect("CREATE PROPERTY GRAPH should succeed");

    assert_eq!(result.rows().len(), 1);
    let status = &result.rows()[0][0];
    match status {
        Value::String(s) => {
            assert!(s.contains("CompanyGraph"));
            assert!(s.contains("2 node tables"));
        }
        other => panic!("Expected string status, got: {other:?}"),
    }
}

#[test]
fn test_create_property_graph_invalid_reference() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute_sql(
        "CREATE PROPERTY GRAPH BadGraph
         NODE TABLES (
             Person (id BIGINT PRIMARY KEY)
         )
         EDGE TABLES (
             FOLLOWS (
                 id BIGINT PRIMARY KEY,
                 source BIGINT REFERENCES Person(id),
                 target BIGINT REFERENCES NonExistent(id)
             )
         )",
    );
    assert!(
        result.is_err(),
        "Should fail: edge references non-existent table"
    );
}

// ============================================================================
// LEFT OUTER JOIN (OPTIONAL MATCH)
// ============================================================================

/// Creates a network where some nodes have no outgoing KNOWS edges.
fn create_partial_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alix = session
        .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))])
        .unwrap();
    let vincent = session
        .create_node_with_props(&["Person"], [("name", Value::String("Vincent".into()))])
        .unwrap();

    // Alix knows Gus, but Vincent knows nobody
    session.create_edge(alix, gus, "KNOWS");

    let _ = vincent; // Vincent has no outgoing KNOWS edges
    db
}

#[test]
fn test_left_outer_join_basic() {
    let db = create_partial_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)
                LEFT OUTER JOIN MATCH (a)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            )",
        )
        .unwrap();

    // Alix has a friend (Gus), Gus has none, Vincent has none
    // LEFT JOIN preserves all left-side rows
    assert_eq!(
        result.row_count(),
        3,
        "All 3 persons should appear (LEFT JOIN preserves left rows)"
    );

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap_or("NULL"))
        .collect();
    assert!(names.contains(&"Alix"), "Alix should be in results");
    assert!(names.contains(&"Gus"), "Gus should be in results");
    assert!(names.contains(&"Vincent"), "Vincent should be in results");

    // Alix's friend column should be Gus, others should be NULL
    let alix_row = result
        .rows()
        .iter()
        .find(|r| r[0].as_str() == Some("Alix"))
        .unwrap();
    assert_eq!(
        alix_row[1].as_str(),
        Some("Gus"),
        "Alix's friend should be Gus"
    );
}

#[test]
fn test_left_join_shorthand() {
    let db = create_partial_network();
    let session = db.session();

    // LEFT JOIN without OUTER
    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)
                LEFT JOIN MATCH (a)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            )",
        )
        .unwrap();

    assert_eq!(
        result.row_count(),
        3,
        "LEFT JOIN (without OUTER) should work identically"
    );
}

#[test]
fn test_optional_match_syntax() {
    let db = create_partial_network();
    let session = db.session();

    // OPTIONAL MATCH syntax (GQL-compatible)
    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)
                OPTIONAL MATCH (a)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            )",
        )
        .unwrap();

    assert_eq!(
        result.row_count(),
        3,
        "OPTIONAL MATCH should work like LEFT JOIN"
    );
}

#[test]
fn test_left_join_null_values() {
    let db = create_partial_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)
                LEFT OUTER JOIN MATCH (a)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            )",
        )
        .unwrap();

    // Vincent has no friends: friend column should be NULL
    let vincent_row = result
        .rows()
        .iter()
        .find(|r| r[0].as_str() == Some("Vincent"))
        .unwrap();
    assert!(
        vincent_row[1].is_null(),
        "Vincent's friend should be NULL (no outgoing KNOWS edges)"
    );
}

// ============================================================================
// WHERE inside GRAPH_TABLE (SQL:2023)
// ============================================================================

#[test]
fn test_where_inside_graph_table() {
    let db = create_social_network();
    let session = db.session();

    // WHERE between MATCH and COLUMNS filters at graph level
    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                WHERE a.age > 28
                COLUMNS (a.name AS source, b.name AS target)
            )",
        )
        .unwrap();

    // Only Alix (age 30) and Harm (age 35) satisfy a.age > 28,
    // but Harm has no outgoing KNOWS edges. So only Alix's edges match.
    assert_eq!(result.row_count(), 2, "only Alix's KNOWS edges match");
    for row in result.rows() {
        assert_eq!(row[0], Value::String("Alix".into()));
    }
}

#[test]
fn test_where_inside_graph_table_combined_with_sql_where() {
    let db = create_social_network();
    let session = db.session();

    // Inner WHERE filters at graph level, outer SQL WHERE filters on output columns
    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                WHERE a.age > 28
                COLUMNS (a.name AS source, b.name AS target)
            ) g WHERE g.target = 'Harm'",
        )
        .unwrap();

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("Harm".into()));
}

// ============================================================================
// SELECT DISTINCT
// ============================================================================

#[test]
fn test_select_distinct() {
    let db = create_social_network();
    let session = db.session();

    // Without DISTINCT: each edge yields a row, source names repeat
    let all = session
        .execute_sql(
            "SELECT source FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS source)
            )",
        )
        .unwrap();

    // With DISTINCT: duplicates removed
    let distinct = session
        .execute_sql(
            "SELECT DISTINCT source FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS source)
            )",
        )
        .unwrap();

    assert!(
        distinct.row_count() < all.row_count(),
        "DISTINCT should produce fewer rows: got {} vs {}",
        distinct.row_count(),
        all.row_count()
    );
    // Alix has 2 edges, Gus has 1, so ALL=3, DISTINCT=2
    assert_eq!(distinct.row_count(), 2);
}

// ============================================================================
// GROUP BY / HAVING
// ============================================================================

#[test]
fn test_group_by() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT source, COUNT(*) AS friend_count FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS source)
            ) GROUP BY source",
        )
        .unwrap();

    assert_eq!(
        result.row_count(),
        2,
        "two people have outgoing KNOWS edges"
    );

    // Find Alix's row (2 friends) and Gus's row (1 friend)
    let alix_row = result
        .rows()
        .iter()
        .find(|r| r[0].as_str() == Some("Alix"))
        .expect("Alix should appear");
    assert_eq!(alix_row[1], Value::Int64(2));

    let gus_row = result
        .rows()
        .iter()
        .find(|r| r[0].as_str() == Some("Gus"))
        .expect("Gus should appear");
    assert_eq!(gus_row[1], Value::Int64(1));
}

#[test]
fn test_group_by_having() {
    let db = create_social_network();
    let session = db.session();

    // HAVING references the output alias (friend_count), not the raw aggregate
    let result = session
        .execute_sql(
            "SELECT source, COUNT(*) AS friend_count FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS source)
            ) GROUP BY source HAVING friend_count > 1",
        )
        .unwrap();

    // Only Alix has more than 1 friend
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(2));
}

// ============================================================================
// Graph name reference in GRAPH_TABLE
// ============================================================================

#[test]
fn test_graph_name_reference_parses() {
    let db = create_social_network();
    let session = db.session();

    // The parser should accept a graph name before MATCH.
    // Execution routes to the default graph (named graph routing is session-level).
    let result = session.execute_sql(
        "SELECT * FROM GRAPH_TABLE (social_graph, MATCH (a:Person) COLUMNS (a.name AS name))",
    );
    // Should parse and execute (graph name is parsed but not yet routed)
    assert!(
        result.is_ok(),
        "graph name reference should parse, got: {result:?}"
    );
}

// ============================================================================
// nullIf function
// ============================================================================

#[test]
fn test_nullif_function() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)
                COLUMNS (a.name AS name, nullIf(a.name, 'Alix') AS filtered)
            )",
        )
        .unwrap();

    // Alix's filtered value should be NULL, others unchanged
    let alix_row = result
        .rows()
        .iter()
        .find(|r| r[0].as_str() == Some("Alix"))
        .expect("Alix should appear");
    assert!(
        alix_row[1].is_null(),
        "nullIf('Alix', 'Alix') should be NULL"
    );

    let gus_row = result
        .rows()
        .iter()
        .find(|r| r[0].as_str() == Some("Gus"))
        .expect("Gus should appear");
    assert_eq!(
        gus_row[1],
        Value::String("Gus".into()),
        "nullIf('Gus', 'Alix') should be 'Gus'"
    );
}

// ============================================================================
// GROUP BY
// ============================================================================

#[test]
fn test_group_by_with_non_aggregate_column() {
    let db = create_social_network();
    let session = db.session();

    // Explicit GROUP BY with a non-aggregate SELECT column.
    // Before the fix, 'source' would be silently dropped from output.
    let result = session
        .execute_sql(
            "SELECT source, COUNT(*) AS friend_count \
             FROM GRAPH_TABLE ( \
                 MATCH (a:Person)-[:KNOWS]->(b:Person) \
                 COLUMNS (a.name AS source) \
             ) \
             GROUP BY source \
             ORDER BY source",
        )
        .unwrap();

    // Alix -> Gus, Alix -> Harm, Gus -> Harm
    assert_eq!(result.row_count(), 2);
    // Alix has 2 KNOWS edges, Gus has 1
    let alix_row = result.rows().iter().find(|r| r[0].as_str() == Some("Alix"));
    assert!(alix_row.is_some(), "Alix should appear in results");
    assert_eq!(alix_row.unwrap()[1], Value::Int64(2));

    let gus_row = result.rows().iter().find(|r| r[0].as_str() == Some("Gus"));
    assert!(gus_row.is_some(), "Gus should appear in results");
    assert_eq!(gus_row.unwrap()[1], Value::Int64(1));
}

// ============================================================================
// HAVING inline aggregate coverage
// ============================================================================

/// HAVING COUNT(*) > N: inline aggregate in HAVING clause.
#[test]
fn having_count_star_filters_groups() {
    let db = create_social_network();
    let result = db
        .execute_sql(
            "SELECT name FROM GRAPH_TABLE (
                MATCH (n:Person)-[:KNOWS]->(m:Person)
                COLUMNS (n.name AS name)
            ) GROUP BY name HAVING COUNT(*) > 1",
        )
        .unwrap();
    // Alix knows 2 people, Gus knows 1 => only Alix passes HAVING COUNT(*) > 1
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("Alix"));
}

/// HAVING with SUM aggregate.
#[test]
fn having_sum_filters_groups() {
    let db = create_social_network();
    let result = db
        .execute_sql(
            "SELECT name FROM GRAPH_TABLE (
                MATCH (n:Person)-[:KNOWS]->(m:Person)
                COLUMNS (n.name AS name, m.age AS friend_age)
            ) GROUP BY name HAVING SUM(friend_age) > 50",
        )
        .unwrap();
    // Alix's friends: Gus(25) + Harm(35) = 60 > 50
    // Gus's friend: Harm(35) = 35 < 50
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("Alix"));
}

/// HAVING COUNT(*) = 0 on empty groups: should return no rows.
#[test]
fn having_count_zero_returns_empty() {
    let db = create_social_network();
    let result = db
        .execute_sql(
            "SELECT name FROM GRAPH_TABLE (
                MATCH (n:Person)-[:KNOWS]->(m:Person)
                COLUMNS (n.name AS name)
            ) GROUP BY name HAVING COUNT(*) > 10",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 0);
}
