//! Error path integration tests.
//!
//! Tests that exercise error variants for NodeNotFound, EdgeNotFound,
//! invalid queries, double-commit, transaction state violations,
//! and other edge cases that should produce clear errors rather than panics.

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_engine::GrafeoDB;

// ============================================================================
// Entity Not Found
// ============================================================================

#[test]
fn test_get_nonexistent_node() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.get_node(NodeId::new(999)).is_none());
}

#[test]
fn test_get_nonexistent_edge() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.get_edge(EdgeId::new(999)).is_none());
}

#[test]
fn test_set_property_on_nonexistent_node() {
    let db = GrafeoDB::new_in_memory();
    // Setting a property on a missing node should not panic
    db.set_node_property(NodeId::new(999), "key", Value::Int64(1))
        .unwrap();
    // Node still doesn't exist
    assert!(db.get_node(NodeId::new(999)).is_none());
}

#[test]
fn test_delete_nonexistent_node() {
    let db = GrafeoDB::new_in_memory();
    assert!(!db.delete_node(NodeId::new(999)).unwrap());
}

#[test]
fn test_delete_nonexistent_edge() {
    let db = GrafeoDB::new_in_memory();
    assert!(!db.delete_edge(EdgeId::new(999)));
}

#[test]
fn test_get_labels_nonexistent_node() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.get_node_labels(NodeId::new(999)).is_none());
}

#[test]
fn test_add_label_nonexistent_node() {
    let db = GrafeoDB::new_in_memory();
    // Should not panic - returns false since node doesn't exist
    assert!(!db.add_node_label(NodeId::new(999), "Label"));
}

#[test]
fn test_remove_label_nonexistent_node() {
    let db = GrafeoDB::new_in_memory();
    assert!(!db.remove_node_label(NodeId::new(999), "Label"));
}

// ============================================================================
// Query Errors
// ============================================================================

#[test]
fn test_query_syntax_error() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("THIS IS NOT VALID GQL");
    assert!(result.is_err(), "Invalid query should return error");

    let err = result.unwrap_err();
    let err_str = err.to_string();
    // Should be a query error, not a panic
    assert!(!err_str.is_empty(), "Error message should not be empty");
}

#[test]
fn test_query_empty_string() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("");
    assert!(result.is_err(), "Empty query should return error");
}

#[test]
fn test_query_unclosed_parenthesis() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("MATCH (n:Person RETURN n");
    assert!(result.is_err(), "Unclosed parenthesis should fail");
}

#[test]
fn test_query_undefined_variable() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Reference a variable that was never matched
    let result = session.execute("MATCH (n:Person) RETURN x.name");
    assert!(result.is_err(), "Undefined variable should fail");
}

#[test]
fn test_query_return_without_match() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // RETURN without MATCH is valid for constants
    let result = session.execute("RETURN 1 + 2");
    // This may or may not work depending on implementation; just check no panic
    let _ = result;
}

// ============================================================================
// Transaction State Violations
// ============================================================================

#[test]
fn test_commit_without_begin() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    let result = session.commit();
    assert!(result.is_err(), "Commit without begin should fail");
}

#[test]
fn test_rollback_without_begin() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    let result = session.rollback();
    assert!(result.is_err(), "Rollback without begin should fail");
}

#[test]
fn test_double_begin_creates_nested_transaction() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    let result = session.begin_transaction();
    assert!(result.is_ok(), "Double begin creates nested transaction");
    // Clean up: commit inner then outer
    session.commit().unwrap();
    session.commit().unwrap();
}

#[test]
fn test_begin_after_commit_succeeds() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.commit().unwrap();

    // Should be able to start new transaction
    let result = session.begin_transaction();
    assert!(result.is_ok(), "Begin after commit should succeed");
    session.commit().unwrap();
}

#[test]
fn test_begin_after_rollback_succeeds() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.rollback().unwrap();

    // Should be able to start new transaction
    let result = session.begin_transaction();
    assert!(result.is_ok(), "Begin after rollback should succeed");
    session.commit().unwrap();
}

// ============================================================================
// Edge Cases - Empty Database
// ============================================================================

#[test]
fn test_query_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Empty database should return no rows"
    );
}

#[test]
fn test_count_on_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("MATCH (n) RETURN COUNT(n) AS cnt").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "COUNT on empty db should return one row"
    );
}

#[test]
fn test_validate_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let result = db.validate();
    assert!(result.is_valid(), "Empty database should be valid");
}

#[test]
fn test_schema_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let schema = db.schema();
    match schema {
        grafeo_engine::SchemaInfo::Lpg(lpg) => {
            assert!(lpg.labels.is_empty(), "No labels in empty db");
            assert!(lpg.edge_types.is_empty(), "No edge types in empty db");
        }
        _ => panic!("Expected LPG schema for in-memory db"),
    }
}

#[test]
fn test_info_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let info = db.info();
    assert_eq!(info.node_count, 0);
    assert_eq!(info.edge_count, 0);
    assert!(!info.is_persistent);
}

// ============================================================================
// Edge Cases - Large IDs and Boundary Values
// ============================================================================

#[test]
fn test_property_with_null_value() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&["Test"]).unwrap();
    db.set_node_property(n, "key", Value::Null).unwrap();

    // In temporal mode, Null acts as a tombstone and is filtered out.
    // In non-temporal mode, Null is stored and returned.
    let node = db.get_node(n).unwrap();
    #[cfg(feature = "temporal")]
    assert_eq!(
        node.get_property("key"),
        None,
        "Temporal: Null acts as tombstone"
    );
    #[cfg(not(feature = "temporal"))]
    assert_eq!(
        node.get_property("key"),
        Some(&Value::Null),
        "Should store Null value"
    );
}

#[test]
fn test_property_with_empty_string() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&["Test"]).unwrap();
    db.set_node_property(n, "key", Value::String("".into()))
        .unwrap();

    let node = db.get_node(n).unwrap();
    assert_eq!(
        node.get_property("key"),
        Some(&Value::String("".into())),
        "Should store empty string"
    );
}

#[test]
fn test_node_with_no_labels() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&[]).unwrap();

    let node = db.get_node(n);
    assert!(node.is_some(), "Node with no labels should exist");
    assert_eq!(db.node_count(), 1);
}

#[test]
fn test_node_with_many_labels() {
    let db = GrafeoDB::new_in_memory();
    let labels: Vec<&str> = (0..10)
        .map(|i| match i {
            0 => "A",
            1 => "B",
            2 => "C",
            3 => "D",
            4 => "E",
            5 => "F",
            6 => "G",
            7 => "H",
            8 => "I",
            _ => "J",
        })
        .collect();
    let n = db.create_node(&labels).unwrap();

    let stored_labels = db.get_node_labels(n).unwrap();
    assert_eq!(stored_labels.len(), 10);
}

#[test]
fn test_self_referencing_edge() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&["Node"]).unwrap();
    let e = db.create_edge(n, n, "SELF");

    let edge = db.get_edge(e).unwrap();
    assert_eq!(edge.src, n);
    assert_eq!(edge.dst, n);
}

// ============================================================================
// Translator Error Quality (errors should be QueryError, not Internal)
// ============================================================================

#[test]
fn test_gql_syntax_error_has_position_info() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Syntax errors from the parser should produce positioned error messages,
    // not bare "Internal error" strings.
    let result = session.execute("MATCH (n:Person RETURN n");
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        !err_str.contains("Internal error"),
        "Parser error should NOT be an internal error, got: {}",
        err_str
    );
    // Should contain source position info (the --> arrow)
    assert!(
        err_str.contains("-->"),
        "Parser error should include source position, got: {}",
        err_str
    );
}

#[test]
fn test_translator_errors_not_internal() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Various translator errors should NOT produce GRAFEO-X internal errors.
    // They should be query errors (GRAFEO-Q*).
    let test_queries = [
        "MATCH (n:Person) RETURN x.name", // undefined variable
    ];

    for query in &test_queries {
        let result = session.execute(query);
        if let Err(err) = result {
            let err_str = err.to_string();
            assert!(
                !err_str.contains("GRAFEO-X"),
                "Query '{}' should NOT produce an internal error, got: {}",
                query,
                err_str
            );
        }
    }
}

#[cfg(feature = "algos")]
#[test]
fn test_gql_unknown_procedure_error_code() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("CALL grafeo.nonexistent()");
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("Unknown procedure"),
        "Should say 'Unknown procedure', got: {}",
        err_str
    );
}

#[cfg(feature = "algos")]
#[test]
fn test_gql_yield_nonexistent_column_error() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("CALL grafeo.pagerank() YIELD nonexistent_column");
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("not found"),
        "Should mention column not found, got: {}",
        err_str
    );
}

#[test]
#[cfg(feature = "cypher")]
fn test_cypher_pattern_comprehension_works() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Pattern comprehension is now supported after planner refactor
    let result = session.execute_cypher("MATCH (n) RETURN [(n)-[:KNOWS]->(m) | m.name] AS friends");
    assert!(
        result.is_ok(),
        "Pattern comprehension should succeed, got: {:?}",
        result.err()
    );
}

#[test]
#[cfg(feature = "graphql")]
fn test_graphql_range_filter_end_to_end() {
    let db = GrafeoDB::new_in_memory();
    // Create test data
    for i in 0..5 {
        let n = db.create_node(&["Person"]).unwrap();
        db.set_node_property(n, "name", Value::String(format!("Person{}", i).into()))
            .unwrap();
        db.set_node_property(n, "age", Value::Int64(20 + i * 10))
            .unwrap(); // 20, 30, 40, 50, 60
    }

    let session = db.session();
    let result = session.execute_graphql(r#"{ person(age_gt: 25, age_lt: 55) { name age } }"#);
    assert!(
        result.is_ok(),
        "GraphQL range filter should work: {:?}",
        result.err()
    );
    let result = result.unwrap();
    // Should match ages 30, 40, 50 (not 20, not 60)
    assert_eq!(
        result.row_count(),
        3,
        "Should find 3 persons with 25 < age < 55, got {}",
        result.row_count()
    );
}

// ============================================================================
// Query Language Edge Cases
// ============================================================================

#[test]
fn test_insert_and_match_special_characters_in_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("INSERT (:Test {name: 'Hello World'})")
        .unwrap();

    let result = session.execute("MATCH (t:Test) RETURN t.name").unwrap();
    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_match_with_multiple_labels() {
    let db = GrafeoDB::new_in_memory();
    let n = db.create_node(&["Person", "Employee"]).unwrap();
    db.set_node_property(n, "name", Value::String("Alix".into()))
        .unwrap();

    let session = db.session();
    let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_in_operator_with_empty_list() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    db.create_node(&["Person"]).unwrap();

    let result = session
        .execute("MATCH (n:Person) WHERE n.name IN [] RETURN n")
        .unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "IN with empty list should match nothing"
    );
}

// ============================================================================
// Error Position Info (0.5.5)
// ============================================================================

#[test]
fn test_gql_error_shows_line_and_column() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Missing closing paren: genuine syntax error
    let result = session.execute("MATCH (n:Person RETURN n");
    assert!(result.is_err(), "Expected parse error for malformed GQL");
    let err_str = result.unwrap_err().to_string();
    // Should contain the caret display with --> position
    assert!(
        err_str.contains("-->"),
        "GQL error should show position, got: {err_str}"
    );
    // Should contain the source line
    assert!(
        err_str.contains("RETURN"),
        "GQL error should show source query, got: {err_str}"
    );
    // Should contain caret markers
    assert!(
        err_str.contains('^'),
        "GQL error should show caret markers, got: {err_str}"
    );
}

#[test]
fn test_gql_multiline_error_position() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // RETURN is a typo: genuine syntax error on line 3
    let query = "MATCH (n:Person)\nWHERE n.age > 30\nRETRUN n";
    let result = session.execute(query);
    assert!(result.is_err(), "Expected parse error for RETURN typo");
    let err_str = result.unwrap_err().to_string();
    // Error should reference line 3 (where RETURN is)
    assert!(
        err_str.contains("--> query:3:"),
        "Multiline error should show line 3, got: {err_str}"
    );
}

#[test]
#[cfg(feature = "cypher")]
fn test_cypher_error_shows_position() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Missing closing paren: genuine syntax error
    let result = session.execute_cypher("MATCH (n:Person RETURN n");
    assert!(result.is_err(), "Expected parse error for malformed Cypher");
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("-->"),
        "Cypher error should show position, got: {err_str}"
    );
    assert!(
        err_str.contains("RETURN"),
        "Cypher error should show source, got: {err_str}"
    );
}

#[test]
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn test_sparql_error_shows_position() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Missing closing brace: genuine syntax error
    let result = session.execute_sparql("SELECT ?s WHERE { ?s ?p ?o");
    assert!(result.is_err(), "Expected parse error for malformed SPARQL");
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("-->"),
        "SPARQL error should show position, got: {err_str}"
    );
}

#[test]
#[cfg(feature = "sql-pgq")]
fn test_sql_pgq_error_shows_position() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute_sql("SELECT * FROM GRAPH_TABLE(g MATCH (n) COLUMNS (n.name))");
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("-->"),
        "SQL/PGQ error should show position, got: {err_str}"
    );
}

// ============================================================================
// Malformed / adversarial parser input (T3-09)
// ============================================================================

#[test]
fn test_whitespace_only_query() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("   \t\n  ");
    assert!(result.is_err(), "whitespace-only query should error");
}

#[test]
fn test_null_byte_in_query() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("MATCH (n\0:Person) RETURN n");
    // Should error or handle gracefully, not panic
    let _ = result;
}

#[test]
fn test_deeply_nested_parentheses() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // 50 levels of nested parens in WHERE clause
    let mut query = String::from("MATCH (n:Person) WHERE ");
    for _ in 0..50 {
        query.push('(');
    }
    query.push_str("n.age > 0");
    for _ in 0..50 {
        query.push(')');
    }
    query.push_str(" RETURN n");
    // Should either parse successfully or error gracefully
    let _ = session.execute(&query);
}

#[test]
fn test_very_long_identifier() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let long_label = "A".repeat(10_000);
    let query = format!("MATCH (n:{long_label}) RETURN n");
    // Should not panic
    let _ = session.execute(&query);
}

#[test]
fn test_unclosed_string_literal() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("MATCH (n:Person) WHERE n.name = 'unclosed RETURN n");
    assert!(result.is_err(), "unclosed string literal should error");
}

#[test]
fn test_mismatched_brackets_in_edge() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("MATCH (a)-[:KNOWS->(b) RETURN a");
    assert!(result.is_err(), "mismatched brackets should error");
}

#[test]
fn test_binary_garbage_input() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let garbage = String::from_utf8_lossy(&[0xFF, 0xFE, 0x00, 0x01, 0x80, 0x90]).to_string();
    let result = session.execute(&garbage);
    assert!(result.is_err(), "binary garbage should error");
}

// ============================================================================
// Parser error message content verification (T3-02)
// ============================================================================

#[test]
fn test_gql_error_names_unexpected_token() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // Use a genuinely invalid query: RETURN without MATCH is not valid GQL
    let result = session.execute("MATCH (n:Person) RETURNING n");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Error should mention the unexpected token or provide context
    assert!(
        err.contains("RETURNING") || err.contains("unexpected") || err.contains("expected"),
        "error should name the unexpected token or suggest expected tokens, got: {err}"
    );
}

#[test]
fn test_gql_error_includes_line_and_column() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("MATCH (n:Person)\nWHERE n.age >\nRETURN n");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Should include position info with line:column format
    assert!(
        err.contains("-->"),
        "multiline error should include position marker, got: {err}"
    );
}

#[test]
fn test_gql_error_includes_caret_marker() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("MATCH (n:Person) WHERE n. RETURN n");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Error should include the caret (^) pointing to the issue
    assert!(
        err.contains('^') || err.contains("-->"),
        "error should include visual position markers, got: {err}"
    );
}

#[test]
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn test_sparql_error_includes_position() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // Use genuinely invalid SPARQL syntax
    let result = session.execute_sparql("GRAB ?s WHERE { ?s ?p ?o }");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("-->") || err.contains("SELECT") || err.contains("expected"),
        "SPARQL error should include position or name bad token, got: {err}"
    );
}

#[test]
#[cfg(feature = "cypher")]
fn test_cypher_error_names_bad_token() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute_cypher("MTCH (n) RETURN n");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("-->") || err.contains("MTCH") || err.contains("expected"),
        "Cypher error should include position or name bad token, got: {err}"
    );
}
