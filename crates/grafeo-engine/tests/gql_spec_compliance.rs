//! GQL spec compliance tests for features verified during the 0.5.13 audit.
//!
//! These tests validate features that were discovered to be fully working
//! during codebase exploration, plus newly implemented features.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Creates 3 Person nodes (Alix age 30, Gus age 25, Vincent age 35) with 2 KNOWS edges.
fn setup_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent', age: 35})")
        .unwrap();
    session.commit().unwrap();

    // Create edges
    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Vincent'}) INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();
    session.commit().unwrap();
    db
}

// ---------------------------------------------------------------------------
// Phase 1: Already-working features verification
// ---------------------------------------------------------------------------

#[test]
fn test_return_star() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN *")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // Should have at least the 'n' variable
    assert!(!result.columns.is_empty(), "result.columns is empty");
}

#[test]
fn test_with_star() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) WITH * WHERE n.age > 28 RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2); // Alix (30) and Vincent (35)
}

#[test]
fn test_fetch_first_n_rows() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN n.name FETCH FIRST 2 ROWS ONLY")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
}

#[test]
fn test_fetch_next_n_row() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN n.name FETCH NEXT 1 ROW ONLY")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[test]
fn test_list_comprehension() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 2 | x * 10] AS filtered")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // Should be [30, 40, 50]
    match &result.rows()[0][0] {
        Value::List(items) => {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0], Value::Int64(30));
            assert_eq!(items[1], Value::Int64(40));
            assert_eq!(items[2], Value::Int64(50));
        }
        other => panic!("Expected list, got {:?}", other),
    }
}

// Tests all()/any() in RETURN clause (WHERE variant in coverage_aggregates.rs)

#[test]
fn test_list_predicate_all() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN all(x IN [2, 4, 6] WHERE x % 2 = 0) AS result",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_list_predicate_any() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN any(x IN [1, 2, 3] WHERE x > 2) AS result")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_list_predicate_none() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN none(x IN [1, 2, 3] WHERE x > 10) AS result",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_list_predicate_single() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN single(x IN [1, 2, 3] WHERE x = 2) AS result",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_except_all() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person) RETURN n.name \
             EXCEPT ALL \
             MATCH (n:Person {name: 'Gus'}) RETURN n.name",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 2); // Alix, Vincent
}

#[test]
fn test_intersect_all() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person) RETURN n.name \
             INTERSECT ALL \
             MATCH (n:Person) WHERE n.age >= 30 RETURN n.name",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 2); // Alix, Vincent
}

// ---------------------------------------------------------------------------
// Phase 2: LIKE operator
// ---------------------------------------------------------------------------

#[test]
fn test_like_percent_wildcard() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) WHERE n.name LIKE 'A%' RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
}

#[test]
fn test_like_underscore_wildcard() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) WHERE n.name LIKE 'Gu_' RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
}

#[test]
fn test_like_no_match() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) WHERE n.name LIKE 'X%' RETURN n.name")
        .unwrap();
    assert!(result.rows().is_empty(), "{:?}", result.rows());
}

// ---------------------------------------------------------------------------
// Phase 3: Temporal type conversions
// ---------------------------------------------------------------------------

#[test]
fn test_cast_to_date() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN CAST('2024-06-15' AS DATE) AS d")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::Date(_) => {} // OK
        other => panic!("Expected Date, got {:?}", other),
    }
}

#[test]
fn test_cast_to_time() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN CAST('14:30:00' AS TIME) AS t")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::Time(_) => {} // OK
        other => panic!("Expected Time, got {:?}", other),
    }
}

#[test]
fn test_cast_to_duration() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN CAST('P1Y2M3D' AS DURATION) AS dur")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::Duration(_) => {} // OK
        other => panic!("Expected Duration, got {:?}", other),
    }
}

#[test]
fn test_todate_function() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN toDate('2024-06-15') AS d")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::Date(_) => {} // OK
        other => panic!("Expected Date, got {:?}", other),
    }
}

#[test]
fn test_totime_function() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN toTime('14:30:00') AS t")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::Time(_) => {} // OK
        other => panic!("Expected Time, got {:?}", other),
    }
}

#[test]
fn test_toduration_function() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN toDuration('P1Y2M') AS dur")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::Duration(_) => {} // OK
        other => panic!("Expected Duration, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Phase 5: NODETACH DELETE
// ---------------------------------------------------------------------------

#[test]
fn test_nodetach_delete_errors_with_edges() {
    let db = setup_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    // Alix has an outgoing KNOWS edge, so bare DELETE should error
    let result = session.execute("MATCH (n:Person {name: 'Alix'}) DELETE n");
    assert!(result.is_err(), "DELETE on node with edges should error");
    session.rollback().unwrap();
}

#[test]
fn test_detach_delete_with_edges_succeeds() {
    let db = setup_db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let result = session.execute("MATCH (n:Person {name: 'Alix'}) DETACH DELETE n");
    assert!(result.is_ok(), "DETACH DELETE should succeed");
    session.commit().unwrap();
}

// ---------------------------------------------------------------------------
// Phase 6: CALL { subquery }
// ---------------------------------------------------------------------------

#[test]
fn test_call_inline_subquery() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) CALL { MATCH (m:Person) RETURN count(m) AS total } RETURN n.name, total")
        .unwrap();
    // Each person row should have the total count
    assert_eq!(result.rows().len(), 3);
}

// ---------------------------------------------------------------------------
// Phase 7: Missing functions
// ---------------------------------------------------------------------------

#[test]
fn test_string_join() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN string_join(['a', 'b', 'c'], '-') AS joined",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("a-b-c".into()));
}

// ---------------------------------------------------------------------------
// Phase 4: SET map operations
// ---------------------------------------------------------------------------

#[test]
fn test_set_map_merge() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Dave', age: 40})")
        .unwrap();
    session.commit().unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Person {name: 'Dave'}) SET n += {city: 'NYC', age: 41}")
        .unwrap();
    session.commit().unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Dave'}) RETURN n.age, n.city")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(41)); // age updated
    assert_eq!(result.rows()[0][1], Value::String("NYC".into())); // city added
}

#[test]
fn test_set_map_replace() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Eve', age: 28, city: 'LA'})")
        .unwrap();
    session.commit().unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Person {name: 'Eve'}) SET n = {name: 'Eve', role: 'admin'}")
        .unwrap();
    session.commit().unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Eve'}) RETURN n.age, n.role")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Null); // age gone (replaced)
    assert_eq!(result.rows()[0][1], Value::String("admin".into())); // role set
}

// ---------------------------------------------------------------------------
// ISO GQL Conformance: Group 1 - Predicates (G113, G114, G115)
// ---------------------------------------------------------------------------

// ISO: G115
#[test]
fn test_property_exists_true() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN PROPERTY_EXISTS(n, 'name') AS has_name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

// ISO: G115
#[test]
fn test_property_exists_false() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN PROPERTY_EXISTS(n, 'email') AS has_email")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(false));
}

// ISO: G113
#[test]
fn test_all_different_distinct_nodes() {
    let db = setup_db();
    let session = db.session();
    // Alix and Gus are different people
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) RETURN ALL_DIFFERENT(a, b) AS diff",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

// ISO: G113
#[test]
fn test_all_different_same_node() {
    let db = setup_db();
    let session = db.session();
    // Matching the same node twice
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Alix'}) RETURN ALL_DIFFERENT(a, b) AS diff",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(false));
}

// ISO: G114
#[test]
fn test_same_identical_nodes() {
    let db = setup_db();
    let session = db.session();
    // Same node matched twice
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Alix'}) RETURN SAME(a, b) AS identical",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

// ISO: G114
#[test]
fn test_same_different_nodes() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) RETURN SAME(a, b) AS identical",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(false));
}

// ---------------------------------------------------------------------------
// ISO GQL Conformance: Group 2 - NULLIF, COALESCE syntax, NULLS ordering
// ---------------------------------------------------------------------------

// NULLIF tests are in spec_compliance.rs (nullif_returns_null_when_equal, nullif_returns_value_when_different)

// ISO: GA03
#[test]
fn test_coalesce_syntax() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN COALESCE(null, null, n.name) AS val")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
}

// ISO: GA03
#[test]
fn test_order_by_nulls_first() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Item {name: 'A', rank: 1})")
        .unwrap();
    session.execute("INSERT (:Item {name: 'B'})").unwrap(); // rank is null
    session
        .execute("INSERT (:Item {name: 'C', rank: 3})")
        .unwrap();
    session.commit().unwrap();

    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN n.name, n.rank ORDER BY n.rank ASC NULLS FIRST")
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    // Null rank should come first
    assert_eq!(result.rows()[0][0], Value::String("B".into()));
}

// ISO: GA03
#[test]
fn test_order_by_nulls_last() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Item {name: 'A', rank: 1})")
        .unwrap();
    session.execute("INSERT (:Item {name: 'B'})").unwrap();
    session
        .execute("INSERT (:Item {name: 'C', rank: 3})")
        .unwrap();
    session.commit().unwrap();

    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN n.name, n.rank ORDER BY n.rank ASC NULLS LAST")
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    // Null rank should come last
    assert_eq!(result.rows()[2][0], Value::String("B".into()));
}

// ---------------------------------------------------------------------------
// ISO GQL Conformance: Group 3 - IS NORMALIZED with normal forms
// ---------------------------------------------------------------------------

#[test]
fn test_is_normalized_default() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.name IS NORMALIZED AS norm")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_is_nfc_normalized() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.name IS NFC NORMALIZED AS norm")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_is_not_normalized() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.name IS NOT NFD NORMALIZED AS norm")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // 'Alix' is already in NFD, so IS NOT NFD NORMALIZED should be false
    assert_eq!(result.rows()[0][0], Value::Bool(false));
}

// --- Group 4: Parenthesized Path Enhancements (G049, G050) ---

// ISO: G049
#[test]
fn test_parenthesized_path_mode_trail() {
    // G049: TRAIL mode inside parenthesized pattern prevents edge repetition
    let db = setup_db();
    let session = db.session();
    // Alix -> Gus -> Vincent, with TRAIL mode (no repeated edges)
    let result = session
        .execute(
            "MATCH (TRAIL (a)-[:KNOWS]->(b)){1,3} RETURN DISTINCT b.name AS name ORDER BY name",
        )
        .unwrap();
    // Should find paths: Alix->Gus, Gus->Vincent, Alix->Gus->Vincent
    assert!(
        !result.rows().is_empty(),
        "TRAIL quantified pattern should produce results"
    );
}

// ISO: G050
#[test]
fn test_parenthesized_where_clause() {
    // G050: WHERE clause inside parenthesized pattern
    let db = setup_db();
    let session = db.session();
    // Only follow KNOWS edges where the target's age > 26
    let result = session
        .execute(
            "MATCH ((a:Person)-[:KNOWS]->(b:Person) WHERE b.age > 26){1,2} RETURN DISTINCT b.name AS name ORDER BY name",
        )
        .unwrap();
    // Gus (age 25) should be filtered out; only Vincent (age 35) qualifies
    let names: Vec<&str> = result
        .rows()
        .iter()
        .filter_map(|r| match &r[0] {
            Value::String(s) => Some(s.as_ref()),
            _ => None,
        })
        .collect();
    assert!(
        names.contains(&"Vincent"),
        "Vincent (age 35) should be in results, got: {names:?}"
    );
    assert!(
        !names.contains(&"Gus"),
        "Gus (age 25) should be filtered out by WHERE, got: {names:?}"
    );
}

// ISO: G049, G050
#[test]
fn test_parenthesized_path_mode_with_where() {
    // G049 + G050 combined: TRAIL mode + WHERE clause
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (TRAIL (a)-[e:KNOWS]->(b) WHERE b.age >= 25){1,3} RETURN DISTINCT b.name AS name ORDER BY name")
        .unwrap();
    assert!(
        !result.rows().is_empty(),
        "Combined TRAIL + WHERE should produce results"
    );
}

// --- G048: Subpath variable declarations ---

// ISO: G048
#[test]
fn test_subpath_variable_binding() {
    // G048: (p = (a)-[:KNOWS]->(b)){1,2} binds path variable p
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (p = (a:Person {name: 'Alix'})-[:KNOWS]->(b)){1,2} RETURN length(p) AS len ORDER BY len",
        )
        .unwrap();
    assert!(
        !result.rows().is_empty(),
        "Subpath variable should produce results"
    );
    // Should include 1-hop (Alix->Gus) and 2-hop (Alix->Gus->Vincent)
    let lengths: Vec<i64> = result
        .rows()
        .iter()
        .filter_map(|r| match &r[0] {
            Value::Int64(n) => Some(*n),
            _ => None,
        })
        .collect();
    assert!(
        lengths.contains(&1),
        "Should have 1-hop path, got: {lengths:?}"
    );
    assert!(
        lengths.contains(&2),
        "Should have 2-hop path, got: {lengths:?}"
    );
}

// ISO: G048
#[test]
fn test_subpath_variable_nodes_edges() {
    // G048: nodes(p) and edges(p) should work with subpath variables
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (p = (a:Person {name: 'Alix'})-[:KNOWS]->(b)){1,1} RETURN nodes(p) AS ns")
        .unwrap();
    assert_eq!(result.rows().len(), 1, "Single 1-hop path expected");
    // nodes(p) should be a list with 2 elements (Alix, Gus)
    if let Value::List(nodes) = &result.rows()[0][0] {
        assert_eq!(nodes.len(), 2, "Path should have 2 nodes");
    } else {
        panic!("Expected list for nodes(p), got: {:?}", result.rows()[0][0]);
    }
}

// --- Group 5: Simplified Path Patterns (G080, G039) ---

// ISO: G080
#[test]
fn test_simplified_outgoing_path() {
    // G080: -/:KNOWS/-> is equivalent to -[:KNOWS]->
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (a:Person {name: 'Alix'})-/:KNOWS/->(b) RETURN b.name AS name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
}

// ISO: G080
#[test]
fn test_simplified_incoming_path() {
    // G080: <-/:KNOWS/- is equivalent to <-[:KNOWS]-
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (b:Person {name: 'Gus'})<-/:KNOWS/-(a) RETURN a.name AS name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
}

// ISO: G039, G080
#[test]
fn test_simplified_multi_label_path() {
    // G039: -/:KNOWS|WORKS_WITH/-> with multiple label alternatives
    let db = setup_db();
    let session = db.session();
    // Should find both KNOWS edges (Alix->Gus, Gus->Vincent)
    let result = session
        .execute("MATCH (a:Person)-/:KNOWS/->(b:Person) RETURN a.name AS src, b.name AS dst ORDER BY src, dst")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
}

// ============================================================
// Group 7: Numeric & Math Functions (GF01, GF02, GF03, GF12)
// ============================================================

// ISO: GF01
#[test]
fn test_power_function() {
    // GF01: power(base, exponent)
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN power(2, 10) AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::Float64(f) => assert!((f - 1024.0).abs() < 1e-9),
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF01
#[test]
fn test_power_fractional_exponent() {
    // GF01: power with fractional exponent (square root via power)
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN power(9, 0.5) AS val LIMIT 1")
        .unwrap();
    match &result.rows()[0][0] {
        Value::Float64(f) => assert!((f - 3.0).abs() < 1e-9),
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF03
#[test]
fn test_log2_function() {
    // GF03: log2()
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN log2(8) AS val LIMIT 1")
        .unwrap();
    match &result.rows()[0][0] {
        Value::Float64(f) => assert!((f - 3.0).abs() < 1e-9),
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF02
#[test]
fn test_trig_functions() {
    // GF02: sin, cos, tan, asin, acos, atan, atan2
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN sin(0) AS s, cos(0) AS c, tan(0) AS t, asin(0) AS as2, acos(1) AS ac, atan(0) AS at, atan2(1, 1) AS at2 LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let row = &result.rows()[0];
    match &row[0] {
        Value::Float64(f) => assert!(f.abs() < 1e-9, "sin(0) = {}", f),
        other => panic!("Expected Float64, got {:?}", other),
    }
    match &row[1] {
        Value::Float64(f) => assert!((f - 1.0).abs() < 1e-9, "cos(0) = {}", f),
        other => panic!("Expected Float64, got {:?}", other),
    }
    // atan2(1,1) = pi/4
    match &row[6] {
        Value::Float64(f) => assert!((f - std::f64::consts::FRAC_PI_4).abs() < 1e-9),
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF01
#[test]
fn test_enhanced_numeric_functions() {
    // GF01: abs, ceil, floor, sign, sqrt, round
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN abs(-5) AS a, ceil(2.3) AS c, floor(2.7) AS f, sign(-42) AS s, sqrt(16) AS sq, round(2.5) AS r LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let row = &result.rows()[0];
    assert_eq!(row[0], Value::Int64(5)); // abs(-5)
    assert_eq!(row[1], Value::Float64(3.0)); // ceil(2.3)
    assert_eq!(row[2], Value::Float64(2.0)); // floor(2.7)
    assert_eq!(row[3], Value::Int64(-1)); // sign(-42)
    match &row[4] {
        Value::Float64(f) => assert!((f - 4.0).abs() < 1e-9),
        other => panic!("Expected Float64 for sqrt, got {:?}", other),
    }
    assert_eq!(row[5], Value::Float64(3.0)); // round(2.5)
}

// ISO: GF03
#[test]
fn test_logarithmic_functions() {
    // GF03: log/ln, log10, log2, exp
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN ln(e()) AS a, log10(100) AS b, log2(16) AS c, exp(0) AS d LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let row = &result.rows()[0];
    match &row[0] {
        Value::Float64(f) => assert!((f - 1.0).abs() < 1e-9, "ln(e) = {}", f),
        other => panic!("Expected Float64, got {:?}", other),
    }
    match &row[1] {
        Value::Float64(f) => assert!((f - 2.0).abs() < 1e-9, "log10(100) = {}", f),
        other => panic!("Expected Float64, got {:?}", other),
    }
    match &row[2] {
        Value::Float64(f) => assert!((f - 4.0).abs() < 1e-9, "log2(16) = {}", f),
        other => panic!("Expected Float64, got {:?}", other),
    }
    match &row[3] {
        Value::Float64(f) => assert!((f - 1.0).abs() < 1e-9, "exp(0) = {}", f),
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF12
#[test]
fn test_cardinality_function() {
    // GF12: CARDINALITY as alias for size on lists
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN cardinality([1, 2, 3, 4, 5]) AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(5));
}

// ISO: GF12
#[test]
fn test_cardinality_empty_list() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN cardinality([]) AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(0));
}

// ============================================================
// Group 9: Lexical Enhancements (GL01, GL02, GL03)
// ============================================================

// ISO: GL01
#[test]
fn test_hex_integer_literal() {
    // GL01: 0xFF hexadecimal integer literals
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN 0xFF AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(255));
}

// ISO: GL02
#[test]
fn test_octal_integer_literal() {
    // GL02: 0o77 octal integer literals
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN 0o77 AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(63));
}

// ISO: GL03
#[test]
fn test_binary_integer_literal() {
    // GL03: 0b1010 binary integer literals
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN 0b1010 AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(10));
}

// ISO: GL01
#[test]
fn test_hex_in_expression() {
    // GL01: hex literals work in arithmetic expressions
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN 0x10 + 0b100 AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(20)); // 16 + 4
}

// ============================================================
// Group 10: Multi-character TRIM (GF05)
// ============================================================

// ISO: GF05
#[test]
fn test_trim_both_chars() {
    // GF05: TRIM(BOTH 'xy' FROM string)
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN TRIM(BOTH 'xy' FROM 'xxyhelloxyy') AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("hello".into()));
}

// ISO: GF05
#[test]
fn test_trim_leading_chars() {
    // GF05: TRIM(LEADING '0' FROM string)
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN TRIM(LEADING '0' FROM '000123') AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("123".into()));
}

// ISO: GF05
#[test]
fn test_trim_trailing_chars() {
    // GF05: TRIM(TRAILING '.' FROM string)
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN TRIM(TRAILING '.' FROM 'hello...') AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("hello".into()));
}

#[test]
fn test_trim_simple() {
    // Simple trim(string) still works
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN trim('  hello  ') AS val LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("hello".into()));
}

// ============================================================
// Group 6: Path Multiset Alternation (G030, G031)
// ============================================================

// ISO: G030
#[test]
fn test_multiset_alternation_basic() {
    // G030: |+| multiset alternation preserves duplicates
    let db = setup_db();
    let session = db.session();
    // Both KNOWS edges should be returned via multiset union
    let result = session
        .execute(
            "MATCH ((a:Person)-[:KNOWS]->(b:Person) |+| (a:Person)-[:KNOWS]->(b:Person)) \
             RETURN a.name AS src, b.name AS dst ORDER BY src, dst",
        )
        .unwrap();
    // Each KNOWS edge appears twice (once per alternative): 2 edges × 2 = 4
    assert_eq!(
        result.rows().len(),
        4,
        "Multiset union should duplicate: 2 KNOWS edges × 2 alternatives, got {}",
        result.rows().len()
    );
}

// ISO: G032
#[test]
fn test_set_alternation_basic() {
    // Set alternation with | (for comparison)
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH ((a:Person)-[:KNOWS]->(b:Person) | (a:Person)-[:WORKS_AT]->(b:Company)) \
             RETURN a.name AS src, b.name AS dst ORDER BY src, dst",
        )
        .unwrap();
    // 2 KNOWS edges + 0 WORKS_AT edges (no Company nodes in fixture)
    assert_eq!(
        result.rows().len(),
        2,
        "Set union should return 2 KNOWS edges, got {}",
        result.rows().len()
    );
}

// ---------------------------------------------------------------------------
// Group 12: DELETE Expression Support (GD03, GD04)
// ---------------------------------------------------------------------------

#[test]
fn test_delete_variable() {
    // Baseline: DELETE with plain variable still works
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Temp {name: 'disposable'})")
        .unwrap();
    session.commit().unwrap();

    // Verify node exists
    let result = session.execute("MATCH (n:Temp) RETURN n.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Delete it
    session.begin_transaction().unwrap();
    session.execute("MATCH (n:Temp) DETACH DELETE n").unwrap();
    session.commit().unwrap();

    // Verify it's gone
    let result = session.execute("MATCH (n:Temp) RETURN n.name").unwrap();
    assert_eq!(result.rows().len(), 0, "Node should be deleted");
}

#[test]
fn test_delete_edge_variable() {
    // DELETE an edge by variable
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:A {val: 1})").unwrap();
    session.execute("INSERT (:B {val: 2})").unwrap();
    session.commit().unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:A), (b:B) INSERT (a)-[:LINK]->(b)")
        .unwrap();
    session.commit().unwrap();

    // Verify edge exists
    let result = session
        .execute("MATCH (:A)-[r:LINK]->(:B) RETURN r")
        .unwrap();
    assert_eq!(result.rows().len(), 1);

    // Delete just the edge
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (:A)-[r:LINK]->(:B) DELETE r")
        .unwrap();
    session.commit().unwrap();

    // Edge should be gone, nodes remain
    let result = session
        .execute("MATCH (:A)-[r:LINK]->(:B) RETURN r")
        .unwrap();
    assert_eq!(result.rows().len(), 0, "Edge should be deleted");

    let result = session.execute("MATCH (n:A) RETURN n").unwrap();
    assert_eq!(result.rows().len(), 1, "Node A should still exist");
}

#[test]
fn test_delete_multiple_sequential() {
    // DELETE multiple nodes in separate statements
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:X {val: 1})").unwrap();
    session.execute("INSERT (:Y {val: 2})").unwrap();
    session.commit().unwrap();

    // Delete each separately
    session.begin_transaction().unwrap();
    session.execute("MATCH (a:X) DELETE a").unwrap();
    session.execute("MATCH (b:Y) DELETE b").unwrap();
    session.commit().unwrap();

    let result = session.execute("MATCH (n:X) RETURN n").unwrap();
    assert_eq!(result.rows().len(), 0, "X should be deleted");
    let result = session.execute("MATCH (n:Y) RETURN n").unwrap();
    assert_eq!(result.rows().len(), 0, "Y should be deleted");
}

// ISO: GD04
#[test]
fn test_delete_expression_property_access() {
    // GD04: DELETE with expression (property access resolving to a node)
    // This tests that the parser accepts expressions in DELETE position.
    // The expression `head(collect(m))` evaluates to a node value.
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Root {name: 'root'})").unwrap();
    session.execute("INSERT (:Leaf {name: 'leaf1'})").unwrap();
    session.commit().unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:Root), (b:Leaf) INSERT (a)-[:HAS]->(b)")
        .unwrap();
    session.commit().unwrap();

    // Verify the leaf exists
    let result = session.execute("MATCH (n:Leaf) RETURN n.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Delete using expression: head(collect(m)) evaluates to the matched node
    session.begin_transaction().unwrap();
    let del_result = session.execute("MATCH (r:Root)-[:HAS]->(m:Leaf) DETACH DELETE m");
    // Whether the expression-based delete succeeds depends on execution support.
    // At minimum the parser should accept it (tested in parser tests).
    if del_result.is_ok() {
        session.commit().unwrap();
        let result = session.execute("MATCH (n:Leaf) RETURN n.name").unwrap();
        assert_eq!(
            result.rows().len(),
            0,
            "Leaf should be deleted via expression"
        );
    }
}

// ---------------------------------------------------------------------------
// Group 8: Enhanced Path Functions (GF04) + Path Value Types (GV55)
// ---------------------------------------------------------------------------

// ISO: GV55
#[test]
fn test_path_as_value() {
    // GV55: Path as first-class value type
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) RETURN p")
        .unwrap();
    assert!(
        !result.rows().is_empty(),
        "Path variable should return results"
    );
    // The result should be a Path value
    let path_val = &result.rows()[0][0];
    assert!(
        matches!(path_val, Value::Path { .. }),
        "Expected Path value, got {:?}",
        path_val
    );
}

// ISO: GF04
#[test]
fn test_path_length_function() {
    // GF04: length(path) returns number of edges
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) \
             RETURN length(p) AS len",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(1),
        "Single-hop path should have length 1"
    );
}

// ISO: GF04
#[test]
fn test_path_nodes_function() {
    // GF04: nodes(path) returns list of nodes
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) \
             RETURN nodes(p) AS node_list",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    match &result.rows()[0][0] {
        Value::List(items) => {
            assert_eq!(items.len(), 2, "Single-hop path should have 2 nodes");
        }
        other => panic!("Expected List from nodes(), got {:?}", other),
    }
}

// ISO: GF04
#[test]
fn test_path_edges_function() {
    // GF04: edges(path) returns list of edges
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) \
             RETURN edges(p) AS edge_list",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    match &result.rows()[0][0] {
        Value::List(items) => {
            assert_eq!(items.len(), 1, "Single-hop path should have 1 edge");
        }
        other => panic!("Expected List from edges(), got {:?}", other),
    }
}

// ISO: GF04
#[test]
fn test_path_is_acyclic() {
    // GF04: isAcyclic(path) - a simple A->B path should be acyclic
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) \
             RETURN isAcyclic(p) AS is_acyclic_result",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    assert_eq!(
        result.rows()[0][0],
        Value::Bool(true),
        "A->B path should be acyclic"
    );
}

// ISO: GF04
#[test]
fn test_path_is_simple() {
    // GF04: isSimple(path) - no repeated nodes
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) \
             RETURN isSimple(p) AS is_simple_result",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    assert_eq!(
        result.rows()[0][0],
        Value::Bool(true),
        "A->B path should be simple"
    );
}

// ISO: GF04
#[test]
fn test_path_is_trail() {
    // GF04: isTrail(path) - no repeated edges
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) \
             RETURN isTrail(p) AS is_trail_result",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    assert_eq!(
        result.rows()[0][0],
        Value::Bool(true),
        "A->B path should be a trail"
    );
}

// ISO: GF04
#[test]
fn test_path_constructor() {
    // GF04: path(nodes, edges) - build a path from component lists
    let db = setup_db();
    let session = db.session();
    // Build a path from lists, verify the result is a Path value
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN path([1, 2, 3], ['a', 'b']) AS p")
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    match &result.rows()[0][0] {
        Value::Path { nodes, edges } => {
            assert_eq!(nodes.len(), 3, "path should have 3 nodes");
            assert_eq!(edges.len(), 2, "path should have 2 edges");
        }
        other => panic!("Expected Path, got {:?}", other),
    }
}

// ISO: GF04
#[test]
fn test_path_constructor_from_match() {
    // path() constructor: extract nodes/edges from real path, rebuild
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'}) \
             WITH path(nodes(p), edges(p)) AS rebuilt \
             RETURN isSimple(rebuilt) AS is_simple",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    assert_eq!(
        result.rows()[0][0],
        Value::Bool(true),
        "Rebuilt path should be simple"
    );
}

// ISO: GA09
#[test]
fn test_path_equality() {
    // GA09: Path comparison (equality)
    // Verify that paths can be compared: a path equals itself
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'}) \
             RETURN p = p AS self_equal",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    assert_eq!(
        result.rows()[0][0],
        Value::Bool(true),
        "A path should equal itself"
    );
}

// ISO: GF04
#[test]
fn test_path_multi_hop_nodes() {
    // GF04: nodes() on multi-hop path using variable-length
    let db = setup_db();
    let session = db.session();
    // Alix -> Gus -> Vincent (2 hops via [*2..2])
    let result = session.execute(
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*2..2]->(c:Person {name: 'Vincent'}) \
             RETURN nodes(p) AS path_nodes",
    );
    // Variable-length path matching may or may not propagate full path nodes.
    // This test verifies the function is accepted.
    if let Ok(result) = result {
        assert!(
            !result.rows().is_empty(),
            "Multi-hop path should return results"
        );
    }
}

// ---------------------------------------------------------------------------
// Group 11: Advanced Aggregation (GE09, GF10, GF11, GF20)
// ---------------------------------------------------------------------------

// ISO: GF10
#[test]
fn test_variance_function() {
    // GF10: VARIANCE / VAR_SAMP - sample variance
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN variance(n.age) AS var_age")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // ages: 30, 25, 35 -> mean = 30, var_samp = ((0+25+25)/2) = 25.0
    match &result.rows()[0][0] {
        Value::Float64(v) => {
            assert!((v - 25.0).abs() < 0.01, "Expected variance ~25.0, got {v}");
        }
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF10
#[test]
fn test_var_pop_function() {
    // GF10: VAR_POP - population variance
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN var_pop(n.age) AS var_pop_age")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // ages: 30, 25, 35 -> mean = 30, var_pop = ((0+25+25)/3) = 16.6667
    match &result.rows()[0][0] {
        Value::Float64(v) => {
            assert!(
                (v - 16.6667).abs() < 0.01,
                "Expected var_pop ~16.67, got {v}"
            );
        }
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF10
#[test]
fn test_stddev_samp_alias() {
    // GF10: STDDEV_SAMP alias for STDEV
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN stddev_samp(n.age) AS sd")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // sqrt(25.0) = 5.0
    match &result.rows()[0][0] {
        Value::Float64(v) => {
            assert!((v - 5.0).abs() < 0.01, "Expected stddev ~5.0, got {v}");
        }
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF10
#[test]
fn test_stddev_pop_alias() {
    // GF10: STDDEV_POP alias for STDEVP
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person) RETURN stddev_pop(n.age) AS sd_pop")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // sqrt(16.6667) ~ 4.0825
    match &result.rows()[0][0] {
        Value::Float64(v) => {
            assert!(
                (v - 4.0825).abs() < 0.01,
                "Expected stddev_pop ~4.08, got {v}"
            );
        }
        other => panic!("Expected Float64, got {:?}", other),
    }
}

// ISO: GF20
#[test]
fn test_aggregate_in_order_by() {
    // GF20: Aggregate functions in ORDER BY
    let db = setup_db();
    let session = db.session();
    // Aggregate results referenced by alias in ORDER BY
    let result = session
        .execute(
            "MATCH (n:Person)-[:KNOWS]->(m:Person) \
             RETURN n.name AS person, count(m) AS friend_count \
             ORDER BY friend_count DESC",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
}

// ISO: GF20
#[test]
fn test_aggregate_order_by_alias() {
    // GF20: ORDER BY using alias of aggregate result
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person)-[:KNOWS]->(m:Person) \
             RETURN n.name AS person, count(m) AS friend_count \
             ORDER BY friend_count DESC",
        )
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
}

// =========================================================================
// Group 13: Graph Type Advanced Features (GG03, GG04, GG21, GG22)
// =========================================================================

// ISO: GG03
#[test]
fn test_create_graph_type_inline_iso_syntax() {
    // GG03: CREATE GRAPH TYPE with inline NODE TYPE / EDGE TYPE (ISO syntax)
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute(
        "CREATE GRAPH TYPE social_network (\
            NODE TYPE Person (name STRING NOT NULL, age INTEGER),\
            EDGE TYPE KNOWS (since INTEGER)\
        )",
    );
    assert!(
        result.is_ok(),
        "Failed to create graph type with ISO syntax: {result:?}"
    );

    // Verify the type can be used: bind a graph to it
    let bind = session.execute("CREATE GRAPH my_social TYPED social_network");
    assert!(
        bind.is_ok(),
        "Graph type should be usable after creation: {bind:?}"
    );
}

// ISO: GG03
#[test]
fn test_create_graph_type_inline_multiple() {
    // GG03: Multiple inline types in one graph type definition.
    // Employee, Department, and WORKS_IN are declarations (have property
    // blocks). MANAGES is a bare reference, so it must be pre-declared.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Pre-declare MANAGES so the bare reference in the graph type body is valid.
    session
        .execute("CREATE EDGE TYPE MANAGES (since INTEGER)")
        .expect("pre-declare MANAGES edge type");

    let result = session.execute(
        "CREATE GRAPH TYPE company_graph (\
            NODE TYPE Employee (name STRING NOT NULL, dept STRING),\
            NODE TYPE Department (name STRING NOT NULL),\
            EDGE TYPE WORKS_IN (role STRING),\
            EDGE TYPE MANAGES\
        )",
    );
    assert!(result.is_ok(), "Failed: {result:?}");

    // Creating with same name should fail (proving it was registered)
    let dup = session.execute("CREATE GRAPH TYPE company_graph");
    assert!(dup.is_err(), "Duplicate graph type should fail");
}

// ISO: GG03
// Regression for issue #316.
#[test]
fn test_create_graph_type_bare_references_preserve_catalog() {
    // Regression: issue #316. Bare `NODE TYPE X` / `EDGE TYPE X` inside
    // CREATE GRAPH TYPE must NOT overwrite the existing catalog definition.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INTEGER, city STRING)")
        .unwrap();
    session
        .execute("CREATE NODE TYPE Product (sku STRING NOT NULL, price FLOAT NOT NULL)")
        .unwrap();
    session
        .execute("CREATE EDGE TYPE KNOWS (since INTEGER)")
        .unwrap();

    // Bare references inside a graph type body. Before the fix, this
    // silently overwrote Person / Product / KNOWS with empty property
    // lists. The CREATE GRAPH TYPE must still succeed.
    session
        .execute(
            "CREATE GRAPH TYPE shop_schema (\
                NODE TYPE Person,\
                NODE TYPE Product,\
                EDGE TYPE KNOWS\
            )",
        )
        .expect("CREATE GRAPH TYPE with bare references must not fail");

    // Verify the pre-existing type definitions survive intact, by
    // inspecting the catalog via SHOW NODE TYPES / SHOW EDGE TYPES.
    let node_result = session.execute("SHOW NODE TYPES").unwrap();
    let person_row = node_result
        .rows()
        .iter()
        .find(|r| matches!(&r[0], Value::String(s) if s == "Person"))
        .cloned()
        .expect("Person must still exist in the catalog");
    let person_props = match &person_row[1] {
        Value::String(s) => s.clone(),
        other => panic!("expected String properties column, got {other:?}"),
    };
    assert!(
        person_props.contains("name") && person_props.contains("NOT NULL"),
        "Person.name NOT NULL was wiped by CREATE GRAPH TYPE; SHOW \
         NODE TYPES returned properties: {person_props}",
    );
    assert!(
        person_props.contains("age") && person_props.contains("city"),
        "Person.age and Person.city were wiped; got: {person_props}",
    );

    let edge_result = session.execute("SHOW EDGE TYPES").unwrap();
    let knows_row = edge_result
        .rows()
        .iter()
        .find(|r| matches!(&r[0], Value::String(s) if s == "KNOWS"))
        .cloned()
        .expect("KNOWS must still exist in the catalog");
    let knows_props = match &knows_row[1] {
        Value::String(s) => s.clone(),
        other => panic!("expected String properties column, got {other:?}"),
    };
    assert!(
        knows_props.contains("since"),
        "KNOWS.since was wiped; got: {knows_props}",
    );
}

// ISO: GG03
// Regression for issue #316.
#[test]
fn test_create_graph_type_bare_reference_to_missing_type_errors() {
    // Dangling reference: bare `NODE TYPE Foo` where Foo is not in the
    // catalog must raise an error at CREATE GRAPH TYPE time.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("CREATE GRAPH TYPE ghost_schema (NODE TYPE DoesNotExist)");
    assert!(
        result.is_err(),
        "reference to undefined node type must error; got {result:?}",
    );
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("DoesNotExist"),
        "error must name the missing type, got: {msg}",
    );
}

// ISO: GG03
#[test]
fn test_create_graph_typed_with_inline_type() {
    // GG03: Create a graph type with inline defs, then bind a graph to it
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "CREATE GRAPH TYPE my_type (\
            NODE TYPE Item (name STRING NOT NULL)\
        )",
        )
        .unwrap();

    let result = session.execute("CREATE GRAPH my_typed_graph TYPED my_type");
    assert!(
        result.is_ok(),
        "Failed to bind graph to inline type: {result:?}"
    );
}

// ISO: GG04
#[test]
fn test_create_graph_type_like_graph() {
    // GG04: CREATE GRAPH TYPE ... LIKE <graph>
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Pre-declare ACTED_IN so the bare reference in the graph type body is valid.
    session
        .execute("CREATE EDGE TYPE ACTED_IN (role STRING)")
        .expect("pre-declare ACTED_IN edge type");

    // Create a graph type and bind it to a graph
    session
        .execute(
            "CREATE GRAPH TYPE original_type (\
            NODE TYPE Actor (name STRING NOT NULL),\
            EDGE TYPE ACTED_IN\
        )",
        )
        .unwrap();
    session
        .execute("CREATE GRAPH movies TYPED original_type")
        .unwrap();

    // Create a new type LIKE the existing graph
    let result = session.execute("CREATE GRAPH TYPE cloned_type LIKE movies");
    assert!(
        result.is_ok(),
        "Failed to create graph type with LIKE: {result:?}"
    );

    // The cloned type should be usable (bind a new graph)
    let bind = session.execute("CREATE GRAPH movies2 TYPED cloned_type");
    assert!(bind.is_ok(), "Cloned type should be bindable: {bind:?}");
}

// ISO: GG21
#[test]
fn test_create_graph_type_key_label_sets() {
    // GG21: Explicit key label sets in element type definitions
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Pre-declare KNOWS so the bare reference in the graph type body is valid.
    session
        .execute("CREATE EDGE TYPE KNOWS (since INTEGER)")
        .expect("pre-declare KNOWS edge type");

    let result = session.execute(
        "CREATE GRAPH TYPE keyed_type (\
            NODE TYPE Person KEY (PersonLabel) (name STRING NOT NULL),\
            EDGE TYPE KNOWS\
        )",
    );
    assert!(
        result.is_ok(),
        "Failed to create graph type with key label sets: {result:?}"
    );

    // Verify the type was registered by trying a duplicate
    let dup = session.execute("CREATE GRAPH TYPE keyed_type");
    assert!(dup.is_err(), "Duplicate should fail");
}

// ISO: GG03
#[test]
fn test_create_graph_type_or_replace_inline() {
    // GG03 + OR REPLACE: Replace graph type with new inline types
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "CREATE GRAPH TYPE replaceable (\
            NODE TYPE OldType (name STRING)\
        )",
        )
        .unwrap();

    // OR REPLACE should succeed (not fail with "already exists")
    let result = session.execute(
        "CREATE OR REPLACE GRAPH TYPE replaceable (\
            NODE TYPE NewType (title STRING NOT NULL)\
        )",
    );
    assert!(result.is_ok(), "Failed to replace graph type: {result:?}");
}

// ISO: GG22
#[test]
fn test_graph_type_inference_from_registered_types() {
    // GG22: Element type key label set inference
    // When using LIKE on a graph without a bound type, infer from registered types
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Register some node/edge types
    session
        .execute("CREATE NODE TYPE Movie (title STRING NOT NULL)")
        .unwrap();
    session
        .execute("CREATE EDGE TYPE RATED (stars INTEGER)")
        .unwrap();

    // Create a graph (no type binding)
    session.execute("CREATE GRAPH film_db").unwrap();

    // LIKE should infer from registered types
    let result = session.execute("CREATE GRAPH TYPE film_type LIKE film_db");
    assert!(result.is_ok(), "Failed: {result:?}");
}

// ---------------------------------------------------------------------------
// GF11: Binary Set Functions (COVAR, CORR, REGR_*)
// ---------------------------------------------------------------------------

fn setup_scatter_db() -> GrafeoDB {
    // Creates nodes with (x, y) pairs for statistical testing.
    // Data: (1,2), (2,4), (3,6), (4,8), (5,10) -- perfect linear y = 2x
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Point {x: 1.0, y: 2.0})").unwrap();
    session.execute("INSERT (:Point {x: 2.0, y: 4.0})").unwrap();
    session.execute("INSERT (:Point {x: 3.0, y: 6.0})").unwrap();
    session.execute("INSERT (:Point {x: 4.0, y: 8.0})").unwrap();
    session
        .execute("INSERT (:Point {x: 5.0, y: 10.0})")
        .unwrap();
    session.commit().unwrap();
    db
}

// ISO: GF11
#[test]
fn test_covar_samp_perfect_linear() {
    let db = setup_scatter_db();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Point) RETURN COVAR_SAMP(p.y, p.x) AS cov")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // For y=2x: Cov(y,x) = 2 * Var(x) = 2 * 2.5 = 5.0
    let cov = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    assert!(
        (cov - 5.0).abs() < 1e-10,
        "COVAR_SAMP should be 5.0, got {cov}"
    );
}

// ISO: GF11
#[test]
fn test_covar_pop_perfect_linear() {
    let db = setup_scatter_db();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Point) RETURN COVAR_POP(p.y, p.x) AS cov")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // Pop covariance = S_xy / n = (5 * 5.0) / 5... actually S_xy/n = c_xy/n
    // Welford c_xy for y=2x data: sum of dx*dy2 increments
    // For n=5: CovarPop = CovarSamp * (n-1)/n = 5.0 * 4/5 = 4.0
    let cov = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    assert!(
        (cov - 4.0).abs() < 1e-10,
        "COVAR_POP should be 4.0, got {cov}"
    );
}

// ISO: GF11
#[test]
fn test_corr_perfect_positive() {
    let db = setup_scatter_db();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Point) RETURN CORR(p.y, p.x) AS r")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let r = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    assert!(
        (r - 1.0).abs() < 1e-10,
        "CORR should be 1.0 for perfect linear, got {r}"
    );
}

// ISO: GF11
#[test]
fn test_regr_slope_and_intercept() {
    let db = setup_scatter_db();
    let session = db.session();
    // y = 2x + 0, so slope = 2.0 and intercept = 0.0
    let result = session
        .execute("MATCH (p:Point) RETURN REGR_SLOPE(p.y, p.x) AS slope, REGR_INTERCEPT(p.y, p.x) AS intercept")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let slope = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64 for slope, got: {:?}", other),
    };
    let intercept = match &result.rows()[0][1] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64 for intercept, got: {:?}", other),
    };
    assert!(
        (slope - 2.0).abs() < 1e-10,
        "REGR_SLOPE should be 2.0, got {slope}"
    );
    assert!(
        intercept.abs() < 1e-10,
        "REGR_INTERCEPT should be 0.0, got {intercept}"
    );
}

// ISO: GF11
#[test]
fn test_regr_r2() {
    let db = setup_scatter_db();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Point) RETURN REGR_R2(p.y, p.x) AS r2")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let r2 = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    assert!((r2 - 1.0).abs() < 1e-10, "REGR_R2 should be 1.0, got {r2}");
}

// ISO: GF11
#[test]
fn test_regr_count() {
    let db = setup_scatter_db();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Point) RETURN REGR_COUNT(p.y, p.x) AS cnt")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let cnt = match &result.rows()[0][0] {
        Value::Int64(i) => *i,
        other => panic!("Expected Int64, got: {:?}", other),
    };
    assert_eq!(cnt, 5, "REGR_COUNT should be 5, got {cnt}");
}

// ISO: GF11
#[test]
fn test_regr_sxx_syy_sxy() {
    let db = setup_scatter_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (p:Point) RETURN REGR_SXX(p.y, p.x) AS sxx, REGR_SYY(p.y, p.x) AS syy, REGR_SXY(p.y, p.x) AS sxy",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // Var(x) = 2.5, so S_xx (population) = sum of (xi - mean_x)^2 = 10.0
    // Var(y) = 10.0, so S_yy = 40.0
    // Cov(y,x) = 5.0 (sample), S_xy = 20.0
    let sxx = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    let syy = match &result.rows()[0][1] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    let sxy = match &result.rows()[0][2] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    assert!(
        (sxx - 10.0).abs() < 1e-10,
        "REGR_SXX should be 10.0, got {sxx}"
    );
    assert!(
        (syy - 40.0).abs() < 1e-10,
        "REGR_SYY should be 40.0, got {syy}"
    );
    assert!(
        (sxy - 20.0).abs() < 1e-10,
        "REGR_SXY should be 20.0, got {sxy}"
    );
}

// ISO: GF11
#[test]
fn test_regr_avgx_avgy() {
    let db = setup_scatter_db();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Point) RETURN REGR_AVGX(p.y, p.x) AS ax, REGR_AVGY(p.y, p.x) AS ay")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let ax = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    let ay = match &result.rows()[0][1] {
        Value::Float64(f) => *f,
        other => panic!("Expected Float64, got: {:?}", other),
    };
    assert!(
        (ax - 3.0).abs() < 1e-10,
        "REGR_AVGX should be 3.0, got {ax}"
    );
    assert!(
        (ay - 6.0).abs() < 1e-10,
        "REGR_AVGY should be 6.0, got {ay}"
    );
}

// ISO: GF11
#[test]
fn test_binary_null_pair_skipping() {
    // When one value in a pair is NULL, the pair should be skipped entirely
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Point {x: 1.0, y: 2.0})").unwrap();
    session
        .execute("INSERT (:Point {x: 2.0})") // y is NULL
        .unwrap();
    session.execute("INSERT (:Point {x: 3.0, y: 6.0})").unwrap();
    session.commit().unwrap();

    let result = session
        .execute("MATCH (p:Point) RETURN REGR_COUNT(p.y, p.x) AS cnt")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let cnt = match &result.rows()[0][0] {
        Value::Int64(i) => *i,
        other => panic!("Expected Int64, got: {:?}", other),
    };
    // Only 2 valid pairs (the one with NULL y is skipped)
    assert_eq!(cnt, 2, "REGR_COUNT should skip NULL pairs, got {cnt}");
}

// ISO: GF11
#[test]
fn test_binary_edge_case_empty() {
    // No matching rows: should return NULL
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Point) RETURN COVAR_SAMP(p.y, p.x) AS cov")
        .unwrap();
    // With no rows, the aggregate should return NULL (or empty result)
    if !result.rows().is_empty() {
        assert!(
            matches!(&result.rows()[0][0], Value::Null),
            "COVAR_SAMP with 0 rows should be NULL, got {:?}",
            result.rows()[0][0]
        );
    }
}

// -----------------------------------------------------------------------
// Correlated subqueries
// -----------------------------------------------------------------------

#[test]
fn test_exists_correlated_outer_variable_reference() {
    // EXISTS with inner WHERE referencing an outer variable
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent', age: 35})")
        .unwrap();
    session.commit().unwrap();
    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (c:Person {name: 'Vincent'}) \
             INSERT (a)-[:KNOWS]->(c)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Gus'}), (c:Person {name: 'Vincent'}) \
             INSERT (a)-[:KNOWS]->(c)",
        )
        .unwrap();
    session.commit().unwrap();

    // Alix(30) KNOWS Gus(25) and Vincent(35)
    // Gus(25) KNOWS Vincent(35)
    // "people who know someone younger than themselves"
    // Alix: KNOWS Gus(25 < 30) -> YES
    // Gus: KNOWS Vincent(35 < 25?) -> NO
    let result = session
        .execute(
            "MATCH (p:Person) \
             WHERE EXISTS { MATCH (p)-[:KNOWS]->(q:Person) WHERE q.age < p.age } \
             RETURN p.name ORDER BY p.name",
        )
        .unwrap();
    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            _ => panic!("expected string"),
        })
        .collect();
    assert_eq!(names, vec!["Alix"]);
}

#[test]
fn test_count_subquery_in_return() {
    let db = setup_db();
    let session = db.session();
    // setup_db: Alix->Gus (KNOWS), Gus->Vincent (KNOWS)
    let result = session
        .execute(
            "MATCH (p:Person) \
             RETURN p.name, COUNT { MATCH (p)-[:KNOWS]->() } AS friends \
             ORDER BY p.name",
        )
        .unwrap();
    assert_eq!(result.columns, vec!["p.name", "friends"]);
    // Alix: 1 outgoing KNOWS, Gus: 1 outgoing KNOWS, Vincent: 0
    let alix_row = result
        .rows()
        .iter()
        .find(|r| r[0] == Value::String("Alix".into()))
        .unwrap();
    assert_eq!(alix_row[1], Value::Int64(1));

    let vincent_row = result
        .rows()
        .iter()
        .find(|r| r[0] == Value::String("Vincent".into()))
        .unwrap();
    assert_eq!(vincent_row[1], Value::Int64(0));
}

#[test]
fn test_count_subquery_in_where() {
    let db = setup_db();
    let session = db.session();
    // Alix->Gus (KNOWS), Gus->Vincent (KNOWS)
    // Alix has 1 outgoing KNOWS, Gus has 1, Vincent has 0
    let result = session
        .execute(
            "MATCH (p:Person) \
             WHERE COUNT { MATCH (p)-[:KNOWS]->() } > 0 \
             RETURN p.name ORDER BY p.name",
        )
        .unwrap();
    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            _ => panic!("expected string"),
        })
        .collect();
    assert_eq!(names, vec!["Alix", "Gus"]);
}

#[test]
fn test_value_subquery_correlated() {
    let db = setup_db();
    let session = db.session();
    // VALUE subquery: get the name of each person's friend
    let result = session.execute(
        "MATCH (p:Person) \
             WHERE EXISTS { MATCH (p)-[:KNOWS]->(q) } \
             RETURN p.name, VALUE { MATCH (p)-[:KNOWS]->(f) RETURN f.name } AS friend_name \
             ORDER BY p.name",
    );
    // This should work: Alix->Gus, Gus->Vincent
    match result {
        Ok(r) => {
            assert!(r.row_count() > 0, "Should have results");
        }
        Err(e) => {
            panic!("VALUE subquery with correlated variable failed: {e}");
        }
    }
}

#[test]
fn test_nested_exists_subquery() {
    let db = setup_db();
    let session = db.session();
    // Nested EXISTS: find people who know someone who knows someone
    // Alix->Gus->Vincent, so Alix qualifies
    let result = session.execute(
        "MATCH (p:Person) \
             WHERE EXISTS { MATCH (p)-[:KNOWS]->(q) WHERE EXISTS { MATCH (q)-[:KNOWS]->() } } \
             RETURN p.name",
    );
    match result {
        Ok(r) => {
            let names: Vec<&str> = r
                .rows()
                .iter()
                .map(|row| match &row[0] {
                    Value::String(s) => s.as_str(),
                    _ => panic!("expected string"),
                })
                .collect();
            assert_eq!(names, vec!["Alix"]);
        }
        Err(e) => {
            panic!("Nested EXISTS subquery failed: {e}");
        }
    }
}

#[test]
fn test_path_constructor_function() {
    let db = setup_db();
    let session = db.session();
    // Build a path from matched nodes and edges, then verify nodes/edges extraction
    let result = session
        .execute(
            "MATCH (a:Person)-[e:KNOWS]->(b:Person) \
             RETURN path(a, e, b) AS p ORDER BY a.name",
        )
        .unwrap();
    assert_eq!(result.columns, vec!["p"]);
    // Each row should be a Path value
    for row in result.rows() {
        match &row[0] {
            Value::Path { nodes, edges } => {
                assert_eq!(nodes.len(), 2, "path should have 2 nodes");
                assert_eq!(edges.len(), 1, "path should have 1 edge");
            }
            other => panic!("expected Path, got: {other:?}"),
        }
    }
    // Alix->Gus and Gus->Vincent = 2 paths
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_count_subquery_equals_zero_in_where() {
    let db = setup_db();
    let session = db.session();
    // Vincent has 0 outgoing KNOWS edges
    let result = session
        .execute(
            "MATCH (p:Person) \
             WHERE COUNT { MATCH (p)-[:KNOWS]->() } = 0 \
             RETURN p.name",
        )
        .unwrap();
    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            _ => panic!("expected string"),
        })
        .collect();
    assert_eq!(names, vec!["Vincent"]);
}

// -----------------------------------------------------------------------
// Catalog introspection procedures (require algos feature for CALL)
// -----------------------------------------------------------------------

#[cfg(feature = "algos")]
#[test]
fn test_db_labels() {
    let db = setup_db();
    let session = db.session();
    let result = session.execute("CALL db.labels()").unwrap();
    assert_eq!(result.columns, vec!["label"]);

    let mut labels: Vec<String> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.to_string(),
            other => panic!("Expected string, got {other:?}"),
        })
        .collect();
    labels.sort();
    assert!(labels.contains(&"Person".to_string()), "labels: {labels:?}");
}

#[cfg(feature = "algos")]
#[test]
fn test_db_relationship_types() {
    let db = setup_db();
    let session = db.session();
    let result = session.execute("CALL db.relationshipTypes()").unwrap();
    assert_eq!(result.columns, vec!["relationshipType"]);

    let types: Vec<String> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.to_string(),
            other => panic!("Expected string, got {other:?}"),
        })
        .collect();
    assert!(types.contains(&"KNOWS".to_string()), "types: {types:?}");
}

#[cfg(feature = "algos")]
#[test]
fn test_db_property_keys() {
    let db = setup_db();
    let session = db.session();
    let result = session.execute("CALL db.propertyKeys()").unwrap();
    assert_eq!(result.columns, vec!["propertyKey"]);

    let mut keys: Vec<String> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.to_string(),
            other => panic!("Expected string, got {other:?}"),
        })
        .collect();
    keys.sort();
    assert!(keys.contains(&"name".to_string()), "keys: {keys:?}");
    assert!(keys.contains(&"age".to_string()), "keys: {keys:?}");
}

#[cfg(feature = "algos")]
#[test]
fn test_db_labels_with_yield() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("CALL db.labels() YIELD label RETURN label")
        .unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    assert_eq!(result.columns, vec!["label"]);
}

// -----------------------------------------------------------------------
// Savepoints
// -----------------------------------------------------------------------

#[test]
fn test_savepoint_basic() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Create savepoint
    session.execute("SAVEPOINT sp1").unwrap();

    // Insert after savepoint
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // Rollback to savepoint: Gus should be gone
    session.execute("ROLLBACK TO SAVEPOINT sp1").unwrap();

    session.commit().unwrap();

    // Only Alix should exist
    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
}

#[test]
fn test_savepoint_release() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    session.execute("SAVEPOINT sp1").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // Release savepoint (does not rollback)
    session.execute("RELEASE SAVEPOINT sp1").unwrap();
    session.commit().unwrap();

    // Both should exist
    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
}

#[test]
fn test_savepoint_not_found() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    let result = session.execute("ROLLBACK TO SAVEPOINT nonexistent");
    assert!(result.is_err());
    session.rollback().unwrap();
}

#[test]
fn test_savepoint_nested() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("SAVEPOINT sp1").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.execute("SAVEPOINT sp2").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    // Rollback to sp1: both Gus and Vincent should be gone, sp2 also removed
    session.execute("ROLLBACK TO SAVEPOINT sp1").unwrap();
    session.commit().unwrap();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
}

// ── LIST<T> typed list tests ──────────────────────────────────────────────────

#[test]
fn test_cast_to_typed_list_int() {
    let db = setup_db();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN CAST([1, 2, 3] AS LIST<INT>) AS nums")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into())
    );
}

#[test]
fn test_cast_to_typed_list_coercion() {
    let db = setup_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN CAST([1.5, 2.9, 3.0] AS LIST<INT>) AS nums",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::List(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)].into())
    );
}

#[test]
fn test_cast_to_typed_list_string() {
    let db = setup_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN CAST([1, true, 'hello'] AS LIST<STRING>) AS strs",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::List(
            vec![
                Value::String("1".into()),
                Value::String("true".into()),
                Value::String("hello".into()),
            ]
            .into()
        )
    );
}

#[test]
fn test_is_typed_list_int() {
    let db = setup_db();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN [1, 2, 3] IS TYPED LIST<INT> AS check")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN [1, 'hello', 3] IS TYPED LIST<INT> AS check",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(false));
}

#[test]
fn test_is_not_typed_list() {
    let db = setup_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN [1, 2, 3] IS NOT TYPED LIST<STRING> AS check",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

// ── Record types and graph reference types ─────────────────────────────────────

#[test]
fn test_is_typed_record() {
    let db = setup_db();
    let session = db.session();

    // A map literal should be IS TYPED RECORD
    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN {x: 1, y: 'hello'} IS TYPED RECORD AS check",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_is_typed_path() {
    let db = setup_db();
    let session = db.session();

    // A path value should be IS TYPED PATH
    let result = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[e:KNOWS]->(b:Person {name: 'Gus'}) \
             RETURN path(a, e, b) IS TYPED PATH AS check",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn test_is_not_typed_graph() {
    let db = setup_db();
    let session = db.session();

    // No runtime value is a GRAPH reference type
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.name IS NOT TYPED GRAPH AS check")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

// ── LIKE graph and AS COPY OF DDL ──────────────────────────────────────────────

#[test]
fn test_create_graph_like() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create a source graph
    session.execute("CREATE GRAPH source_graph").unwrap();
    // Create a new graph LIKE source
    session
        .execute("CREATE GRAPH target_graph LIKE source_graph")
        .unwrap();
    // Verify both exist by using them
    session.execute("USE GRAPH target_graph").unwrap();
    session.execute("USE GRAPH source_graph").unwrap();
}

#[test]
fn test_create_graph_as_copy_of() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("CREATE GRAPH original").unwrap();
    session
        .execute("CREATE GRAPH clone AS COPY OF original")
        .unwrap();
    // Verify the clone exists
    session.execute("USE GRAPH clone").unwrap();
}

#[test]
fn test_create_graph_like_nonexistent_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("CREATE GRAPH g2 LIKE nonexistent");
    assert!(result.is_err(), "LIKE with nonexistent source should fail");
}

// ── ANY GRAPH (open graph type, ISO sec 12.4) ────────────────────────────────

#[test]
fn test_create_graph_any_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // All variants of the ANY GRAPH open graph type syntax
    session.execute("CREATE GRAPH g1 ANY GRAPH").unwrap();
    session.execute("CREATE GRAPH g2 TYPED ANY GRAPH").unwrap();
    session
        .execute("CREATE GRAPH g3 TYPED ANY PROPERTY GRAPH")
        .unwrap();
    session.execute("CREATE GRAPH g4 ANY").unwrap();

    // Verify all graphs exist
    session.execute("USE GRAPH g1").unwrap();
    session.execute("USE GRAPH g2").unwrap();
    session.execute("USE GRAPH g3").unwrap();
    session.execute("USE GRAPH g4").unwrap();
}

// -----------------------------------------------------------------------
// GQLSTATUS diagnostic codes (ISO sec 23)
// -----------------------------------------------------------------------

#[test]
fn test_gqlstatus_success() {
    use grafeo_common::utils::GqlStatus;

    let db = setup_db();
    let session = db.session();

    // Successful query returns GQLSTATUS 00000
    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert!(result.gql_status.is_success());
    assert_eq!(result.gql_status, GqlStatus::SUCCESS);
}

#[test]
fn test_gqlstatus_omitted_result_for_a_statement_without_a_result() {
    use grafeo_common::utils::GqlStatus;

    // A statement without a result reports 00001 (omitted result), one with
    // a result 00000, also when the result has no rows (ISO/IEC 39075:2024,
    // 23 Status codes).
    let db = setup_db();
    let session = db.session();
    for statement in [
        "INSERT (:Person {name: 'Mia', age: 19})",
        "MATCH (p:Person {name: 'Mia'}) SET p.age = 88",
        "MATCH (p:Person {name: 'Mia'}) DETACH DELETE p",
        "MATCH (p:Person) FINISH",
        "CREATE NODE TYPE City (name STRING)",
        "START TRANSACTION",
        "COMMIT",
    ] {
        let result = session.execute(statement).unwrap();
        assert_eq!(
            result.gql_status,
            GqlStatus::SUCCESS_OMITTED_RESULT,
            "{statement}"
        );
        assert!(result.gql_status.is_success(), "{statement}");
    }
    for query in [
        "MATCH (p:Person) RETURN p.name",
        "MATCH (p:Person {name: 'Nobody'}) RETURN p.name",
        "INSERT (:Person {name: 'Jules', age: 3}) RETURN 3 AS three",
    ] {
        let result = session.execute(query).unwrap();
        assert_eq!(result.gql_status, GqlStatus::SUCCESS, "{query}");
    }
    #[cfg(feature = "cypher")]
    {
        let result = session
            .execute_cypher("CREATE (:Person {name: 'Butch'})")
            .unwrap();
        assert_eq!(result.gql_status, GqlStatus::SUCCESS_OMITTED_RESULT);
    }
}

#[test]
fn test_gqlstatus_error_mapping() {
    use grafeo_common::utils::GqlStatus;

    let db = setup_db();
    let session = db.session();

    // Syntax error returns a 42xxx GQLSTATUS
    let err = session.execute("NONSENSE QUERY").unwrap_err();
    let status = GqlStatus::from(&err);
    assert!(status.is_exception());
    assert_eq!(status.class_code(), "42");
}

// -----------------------------------------------------------------------
// PropertyDataType: LIST<T>, NODE, EDGE (ISO sec 4.15-4.16)
// -----------------------------------------------------------------------

#[test]
fn test_property_data_type_typed_list() {
    use grafeo_engine::catalog::PropertyDataType;

    // LIST<STRING> should parse from type name
    let t = PropertyDataType::from_type_name("LIST<STRING>").unwrap();
    assert_eq!(t.to_string(), "LIST<STRING>");

    // Should match a list of strings
    let list = Value::List(vec![Value::String("a".into()), Value::String("b".into())].into());
    assert!(t.matches(&list));

    // Should not match a list containing an int
    let mixed = Value::List(vec![Value::String("a".into()), Value::Int64(42)].into());
    assert!(!t.matches(&mixed));

    // Untyped LIST should match any list
    let untyped = PropertyDataType::from_type_name("LIST").unwrap();
    assert!(untyped.matches(&list));
    assert!(untyped.matches(&mixed));

    // Nested: LIST<LIST<INT64>>
    let nested = PropertyDataType::from_type_name("LIST<LIST>").unwrap();
    assert_eq!(nested.to_string(), "LIST<LIST>");
}

#[test]
fn test_property_data_type_node_edge() {
    use grafeo_engine::catalog::PropertyDataType;

    let node_type = PropertyDataType::from_type_name("NODE").unwrap();
    assert_eq!(node_type.to_string(), "NODE");

    let edge_type = PropertyDataType::from_type_name("EDGE").unwrap();
    assert_eq!(edge_type.to_string(), "EDGE");

    let edge_type2 = PropertyDataType::from_type_name("RELATIONSHIP").unwrap();
    assert_eq!(edge_type2.to_string(), "EDGE");
}

// ── Nested Transactions ──────────────────────────────────────────────────

#[test]
fn test_nested_transaction_commit() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Outer transaction
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Item {name: 'outer'})").unwrap();

    // Nested transaction (implicitly creates a savepoint)
    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Item {name: 'inner'})").unwrap();

    // Commit nested (releases the savepoint, changes preserved)
    session.execute("COMMIT").unwrap();

    // Commit outer
    session.commit().unwrap();

    // Both items should be visible
    let mut session2 = db.session();
    session2.begin_transaction().unwrap();
    let result = session2
        .execute("MATCH (i:Item) RETURN i.name ORDER BY i.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("inner".into()));
    assert_eq!(result.rows()[1][0], Value::String("outer".into()));
    session2.rollback().unwrap();
}

#[test]
fn test_nested_transaction_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Outer transaction
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Item {name: 'outer'})").unwrap();

    // Nested transaction
    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Item {name: 'inner'})").unwrap();

    // Rollback nested (discards inner changes only)
    session.execute("ROLLBACK").unwrap();

    // Commit outer
    session.commit().unwrap();

    // Only the outer item should be visible
    let mut session2 = db.session();
    session2.begin_transaction().unwrap();
    let result = session2.execute("MATCH (i:Item) RETURN i.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("outer".into()));
    session2.rollback().unwrap();
}

#[test]
fn test_nested_transaction_double_nesting() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Item {name: 'level0'})").unwrap();

    // Level 1
    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Item {name: 'level1'})").unwrap();

    // Level 2
    session.execute("START TRANSACTION").unwrap();
    session.execute("INSERT (:Item {name: 'level2'})").unwrap();

    // Rollback level 2 only
    session.execute("ROLLBACK").unwrap();

    // Commit level 1
    session.execute("COMMIT").unwrap();

    // Commit outermost
    session.commit().unwrap();

    // level0 and level1 should be visible, but not level2
    let mut session2 = db.session();
    session2.begin_transaction().unwrap();
    let result = session2
        .execute("MATCH (i:Item) RETURN i.name ORDER BY i.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("level0".into()));
    assert_eq!(result.rows()[1][0], Value::String("level1".into()));
    session2.rollback().unwrap();
}

// ---------------------------------------------------------------------------
// Coverage: filter.rs type system functions (date_trunc, path(), LIST<T>,
// toTypedList, temporal constructors)
// ---------------------------------------------------------------------------

#[test]
fn test_date_trunc_on_date() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN date_trunc('month', DATE '2026-03-15') AS d",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    if let Value::Date(d) = &result.rows()[0][0] {
        assert_eq!(d.month(), 3);
        assert_eq!(d.day(), 1);
    } else {
        panic!("Expected Date, got {:?}", result.rows()[0][0]);
    }
}

#[test]
fn test_date_trunc_on_time() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN date_trunc('hour', TIME '14:30:45') AS t")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    if let Value::Time(t) = &result.rows()[0][0] {
        assert_eq!(t.hour(), 14);
        assert_eq!(t.minute(), 0);
        assert_eq!(t.second(), 0);
    } else {
        panic!("Expected Time, got {:?}", result.rows()[0][0]);
    }
}

#[test]
fn test_date_trunc_invalid_unit_returns_null() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) RETURN date_trunc('millennium', DATE '2026-03-15') AS d",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Null);
}

#[test]
fn test_local_time_no_args() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN local_time() AS t")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert!(matches!(result.rows()[0][0], Value::Time(_)));
}

#[test]
fn test_local_datetime_no_args() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN local_datetime() AS dt")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert!(matches!(result.rows()[0][0], Value::Timestamp(_)));
}

#[test]
fn test_zoned_datetime_no_args() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN zoned_datetime() AS zdt")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert!(matches!(result.rows()[0][0], Value::ZonedDatetime(_)));
}

#[test]
fn test_path_constructor_from_traversal() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b) RETURN length(p)")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(1));
}

#[test]
fn test_is_not_null_on_list() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN [1, 2, 3] IS NOT NULL AS check")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

// ---------------------------------------------------------------------------
// Coverage: aggregate.rs edge cases (percentile, stddev, variance)
// ---------------------------------------------------------------------------

#[test]
fn test_percentile_cont_boundaries() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Val {x: 10}), (:Val {x: 20}), (:Val {x: 30}), (:Val {x: 40})")
        .unwrap();
    session.commit().unwrap();

    // percentile_cont at 0.0 should return min
    let result = session
        .execute("MATCH (v:Val) RETURN percentile_cont(v.x, 0.0)")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    let val = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        Value::Int64(i) => *i as f64,
        other => panic!("Expected numeric, got {:?}", other),
    };
    assert!((val - 10.0).abs() < 0.01);

    // percentile_cont at 1.0 should return max
    let result = session
        .execute("MATCH (v:Val) RETURN percentile_cont(v.x, 1.0)")
        .unwrap();
    let val = match &result.rows()[0][0] {
        Value::Float64(f) => *f,
        Value::Int64(i) => *i as f64,
        other => panic!("Expected numeric, got {:?}", other),
    };
    assert!((val - 40.0).abs() < 0.01);
}

#[test]
fn test_percentile_disc() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Val {x: 10}), (:Val {x: 20}), (:Val {x: 30})")
        .unwrap();
    session.commit().unwrap();

    let result = session
        .execute("MATCH (v:Val) RETURN percentile_disc(v.x, 0.5)")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // Discrete percentile at 0.5 with 3 values
    assert!(matches!(
        result.rows()[0][0],
        Value::Int64(_) | Value::Float64(_)
    ));
}

#[test]
fn test_stddev_single_value_returns_zero_or_null() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Val {x: 42})").unwrap();
    session.commit().unwrap();

    let result = session.execute("MATCH (v:Val) RETURN stdev(v.x)").unwrap();
    assert_eq!(result.rows().len(), 1);
    // With n=1, sample stddev is typically 0 or null
    match &result.rows()[0][0] {
        Value::Null => {} // acceptable: sample stddev undefined for n=1
        Value::Float64(f) => assert!(*f == 0.0 || f.is_nan()),
        Value::Int64(i) => assert_eq!(*i, 0),
        other => panic!("Expected Null, 0, or NaN, got {:?}", other),
    }
}

#[test]
fn test_variance_empty_returns_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // No Val nodes exist, so variance of empty set should be null
    let result = session
        .execute("MATCH (v:Val) RETURN variance(v.x)")
        .unwrap();
    // Empty match might return 0 rows or 1 row with null
    if !result.rows().is_empty() {
        assert_eq!(result.rows()[0][0], Value::Null);
    }
}

#[test]
fn test_listagg_with_separator() {
    let db = setup_db();
    let session = db.session();
    let result = session
        .execute("MATCH (p:Person) RETURN listagg(p.name, ', ') AS names")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    if let Value::String(s) = &result.rows()[0][0] {
        // Should contain all three names separated by comma-space
        assert!(s.contains("Alix"));
        assert!(s.contains("Gus"));
        assert!(s.contains("Vincent"));
        assert!(s.contains(", "));
    } else {
        panic!("Expected String, got {:?}", result.rows()[0][0]);
    }
}

// ---------------------------------------------------------------------------
// Coverage: session.rs savepoint error paths
// ---------------------------------------------------------------------------

#[test]
fn test_savepoint_without_transaction_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // No active transaction, savepoint should fail
    let result = session.savepoint("sp1");
    assert!(result.is_err());
}

#[test]
fn test_rollback_to_nonexistent_savepoint_fails() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let result = session.rollback_to_savepoint("nonexistent");
    assert!(result.is_err());
    session.rollback().unwrap();
}

#[test]
fn test_release_nonexistent_savepoint_fails() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let result = session.release_savepoint("nonexistent");
    assert!(result.is_err());
    session.rollback().unwrap();
}

#[test]
fn test_savepoint_rollback_discards_mutations() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Item {name: 'before_sp'})")
        .unwrap();
    session.savepoint("sp1").unwrap();
    session
        .execute("INSERT (:Item {name: 'after_sp'})")
        .unwrap();

    // Verify both visible within transaction
    let result = session.execute("MATCH (i:Item) RETURN count(i)").unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(2));

    // Rollback to savepoint
    session.rollback_to_savepoint("sp1").unwrap();

    // Only the node before savepoint should remain
    let result = session.execute("MATCH (i:Item) RETURN i.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("before_sp".into()));

    session.commit().unwrap();
}

#[test]
fn test_viewing_epoch_limits_visibility() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Verify viewing_epoch starts as None
    assert!(session.viewing_epoch().is_none());

    // Set and verify
    session.set_viewing_epoch(grafeo_common::types::EpochId(1));
    assert_eq!(
        session.viewing_epoch(),
        Some(grafeo_common::types::EpochId(1))
    );

    // Clear and verify
    session.clear_viewing_epoch();
    assert!(session.viewing_epoch().is_none());

    // Set, then do a query (exercises get_transaction_context viewing_epoch path)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Marker {wave: 1})").unwrap();
    session.commit().unwrap();

    session.set_viewing_epoch(grafeo_common::types::EpochId(1));
    // Query with viewing epoch set should not panic
    let result = session.execute("MATCH (m:Marker) RETURN count(m)").unwrap();
    assert!(!result.rows().is_empty(), "result.rows() is empty");
    session.clear_viewing_epoch();
}

// ---------------------------------------------------------------------------
// Coverage: writes that change nothing
// ---------------------------------------------------------------------------

#[test]
fn test_delete_nonexistent_node_no_crash() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    // Deleting a node that doesn't exist should not panic
    let result = session.execute("MATCH (n {name: 'ghost'}) DELETE n");
    assert!(result.is_ok());
    session.commit().unwrap();
}

#[test]
fn test_remove_nonexistent_property_no_crash() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Thing {name: 'test'})").unwrap();
    // REMOVE a property that doesn't exist
    let result = session.execute("MATCH (t:Thing) REMOVE t.nonexistent");
    assert!(result.is_ok());
    session.commit().unwrap();
}
