//! Targeted tests for the query processor: multi-statement execution,
//! parameter substitution edge cases, language dispatch, and error handling.
//!
//! ```bash
//! cargo test -p grafeo-engine --test main query_processor_coverage:: --all-features
//! ```

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ── Helpers ──────────────────────────────────────────────────────────

/// Creates a DB with 3 Person nodes: Alix (30), Gus (25), Vincent (40).
fn seed_people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    s.execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();
    s.execute("INSERT (:Person {name: 'Vincent', age: 40})")
        .unwrap();
    db
}

// =========================================================================
// 1. Multi-statement / composite query execution
// =========================================================================

/// UNION ALL preserves duplicates when two queries overlap.
#[test]
fn union_all_preserves_duplicates() {
    let db = seed_people();
    let s = db.session();
    // Both sides match Alix (age 30)
    let r = s
        .execute(
            "MATCH (n:Person) WHERE n.age >= 30 RETURN n.name \
             UNION ALL \
             MATCH (n:Person) WHERE n.age <= 30 RETURN n.name",
        )
        .unwrap();
    // Left: Alix(30), Vincent(40) = 2; Right: Alix(30), Gus(25) = 2; total = 4
    assert_eq!(r.row_count(), 4, "UNION ALL should not deduplicate");
}

/// UNION (distinct) deduplicates rows.
#[test]
fn union_distinct_deduplicates() {
    let db = seed_people();
    let s = db.session();
    let r = s
        .execute(
            "MATCH (n:Person) WHERE n.age >= 30 RETURN n.name \
             UNION \
             MATCH (n:Person) WHERE n.age <= 30 RETURN n.name",
        )
        .unwrap();
    // Should be exactly 3 unique names: Alix, Gus, Vincent
    assert_eq!(r.row_count(), 3, "UNION should deduplicate");
}

/// NEXT composition: left is evaluated, then right runs in the context of left's rows.
/// Implemented as Apply (correlated subquery), so right executes once per left row.
#[test]
fn next_composition_two_matches() {
    let db = seed_people();
    let s = db.session();
    // Left: 1 row (Alix), Right: runs once producing 1 row (Gus)
    let r = s
        .execute(
            "MATCH (n:Person) WHERE n.age = 30 RETURN n.name \
             NEXT \
             MATCH (m:Person) WHERE m.age = 25 RETURN m.name",
        )
        .unwrap();
    // NEXT (Apply with no shared vars) returns 1 row (1 left row x 1 right row)
    assert_eq!(r.row_count(), 1);
}

/// Chaining three statements with NEXT: each has its own RETURN.
/// Verifies that triple-NEXT chains parse and execute without error.
#[test]
fn next_triple_chain() {
    let db = seed_people();
    let s = db.session();
    let r = s
        .execute(
            "MATCH (a:Person) WHERE a.age = 25 RETURN a.name \
             NEXT \
             MATCH (b:Person) WHERE b.age = 30 RETURN b.name \
             NEXT \
             MATCH (c:Person) WHERE c.age = 40 RETURN c.name",
        )
        .unwrap();
    // Triple NEXT chain: each side produces 1 row, nested Apply yields 1 row
    assert_eq!(r.row_count(), 1);
}

/// NEXT with a left side producing multiple rows multiplies the output.
/// Left returns 2 rows, right returns 1 row per left row: 2 total.
#[test]
fn next_fan_out() {
    let db = seed_people();
    let s = db.session();
    // Left: Alix(30) and Vincent(40) match age >= 30 = 2 rows
    // Right: Gus(25) matches age = 25 = 1 row per left row
    let r = s
        .execute(
            "MATCH (n:Person) WHERE n.age >= 30 RETURN n.name \
             NEXT \
             MATCH (m:Person) WHERE m.age = 25 RETURN m.name",
        )
        .unwrap();
    // Apply: 2 left rows x 1 right row each = 2 result rows
    assert_eq!(r.row_count(), 2);
}

// =========================================================================
// 2. Parameter substitution edge cases
// =========================================================================

/// Missing parameter in WHERE clause produces a clear error.
#[test]
fn param_missing_returns_error() {
    let db = seed_people();
    let s = db.session();
    let params = HashMap::new(); // empty, but query references $min_age
    let r = s.execute_with_params("MATCH (n:Person) WHERE n.age > $min_age RETURN n", params);
    assert!(r.is_err());
    let err_msg = r.unwrap_err().to_string();
    assert!(
        err_msg.contains("Missing parameter") || err_msg.contains("min_age"),
        "Error should mention missing parameter, got: {err_msg}"
    );
}

/// Parameter with string type in WHERE equality filter.
#[test]
fn param_string_filter() {
    let db = seed_people();
    let s = db.session();
    let mut params = HashMap::new();
    params.insert("target".to_string(), Value::String("Gus".into()));
    let r = s
        .execute_with_params(
            "MATCH (n:Person) WHERE n.name = $target RETURN n.age",
            params,
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(25));
}

/// Multiple parameters in a single query.
#[test]
fn param_multiple_in_and_condition() {
    let db = seed_people();
    let s = db.session();
    let mut params = HashMap::new();
    params.insert("min_age".to_string(), Value::Int64(26));
    params.insert("max_age".to_string(), Value::Int64(35));
    let r = s
        .execute_with_params(
            "MATCH (n:Person) WHERE n.age >= $min_age AND n.age <= $max_age RETURN n.name",
            params,
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

/// Parameter in an INSERT property value.
#[test]
fn param_in_insert_property() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let mut params = HashMap::new();
    params.insert("city".to_string(), Value::String("Amsterdam".into()));
    s.execute_with_params("INSERT (:City {name: $city})", params)
        .unwrap();
    let r = s.execute("MATCH (c:City) RETURN c.name").unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Amsterdam".into()));
}

/// Boolean parameter substitution.
#[test]
fn param_boolean_value() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("INSERT (:Flag {name: 'Alix', active: true})")
        .unwrap();
    s.execute("INSERT (:Flag {name: 'Gus', active: false})")
        .unwrap();

    let mut params = HashMap::new();
    params.insert("wanted".to_string(), Value::Bool(true));
    let r = s
        .execute_with_params(
            "MATCH (f:Flag) WHERE f.active = $wanted RETURN f.name",
            params,
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

/// Float parameter substitution.
#[test]
fn param_float_comparison() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    s.execute("INSERT (:Metric {name: 'alpha', score: 3.14})")
        .unwrap();
    s.execute("INSERT (:Metric {name: 'beta', score: 2.71})")
        .unwrap();

    let mut params = HashMap::new();
    params.insert("threshold".to_string(), Value::Float64(3.0));
    let r = s
        .execute_with_params(
            "MATCH (m:Metric) WHERE m.score > $threshold RETURN m.name",
            params,
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("alpha".into()));
}

/// No parameters referenced in query, but params map provided: should succeed.
#[test]
fn param_extra_unused_params_ok() {
    let db = seed_people();
    let s = db.session();
    let mut params = HashMap::new();
    params.insert("unused".to_string(), Value::Int64(999));
    let r = s
        .execute_with_params("MATCH (n:Person) RETURN n.name ORDER BY n.name", params)
        .unwrap();
    assert_eq!(r.row_count(), 3);
}

// =========================================================================
// 3. Language dispatch
// =========================================================================

/// execute_language with "gql" dispatches correctly.
#[test]
fn language_dispatch_gql() {
    let db = seed_people();
    let s = db.session();
    let r = s
        .execute_language(
            "MATCH (n:Person) RETURN n.name ORDER BY n.name",
            "gql",
            None,
        )
        .unwrap();
    assert_eq!(r.row_count(), 3);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

/// execute_language with unknown language name returns an error.
#[test]
fn language_dispatch_unknown_returns_error() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s.execute_language("MATCH (n) RETURN n", "klingon", None);
    assert!(r.is_err());
    let err_msg = r.unwrap_err().to_string();
    assert!(
        err_msg.contains("Unknown query language") || err_msg.contains("klingon"),
        "Error should mention unknown language, got: {err_msg}"
    );
}

/// execute_language "gql" with params routes to the param-aware path.
#[test]
fn language_dispatch_gql_with_params() {
    let db = seed_people();
    let s = db.session();
    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::String("Vincent".into()));
    let r = s
        .execute_language(
            "MATCH (n:Person) WHERE n.name = $name RETURN n.age",
            "gql",
            Some(params),
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(40));
}

/// execute_language "cypher" works when the feature is enabled.
#[cfg(feature = "cypher")]
#[test]
fn language_dispatch_cypher() {
    let db = seed_people();
    let s = db.session();
    let r = s
        .execute_language(
            "MATCH (n:Person) WHERE n.age > 25 RETURN n.name ORDER BY n.name",
            "cypher",
            None,
        )
        .unwrap();
    assert_eq!(r.row_count(), 2);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(r.rows()[1][0], Value::String("Vincent".into()));
}

/// execute_language "cypher" with params.
#[cfg(feature = "cypher")]
#[test]
fn language_dispatch_cypher_with_params() {
    let db = seed_people();
    let s = db.session();
    let mut params = HashMap::new();
    params.insert("min".to_string(), Value::Int64(30));
    let r = s
        .execute_language(
            "MATCH (n:Person) WHERE n.age >= $min RETURN n.name ORDER BY n.name",
            "cypher",
            Some(params),
        )
        .unwrap();
    assert_eq!(r.row_count(), 2);
}

// =========================================================================
// 4. Error handling
// =========================================================================

/// Empty query string returns an error, not a panic.
#[test]
fn error_empty_query() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s.execute("");
    assert!(r.is_err(), "Empty query should produce an error");
}

/// Syntactically invalid query returns parse error.
#[test]
fn error_syntax_garbage() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s.execute("ZORK BLARG FLIM");
    assert!(r.is_err(), "Garbage query should produce a parse error");
}

/// Unclosed parenthesis in MATCH pattern.
#[test]
fn error_unclosed_paren() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s.execute("MATCH (n:Person RETURN n");
    assert!(r.is_err());
}

/// Whitespace-only query is treated as empty.
#[test]
fn error_whitespace_only_query() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s.execute("   \t\n  ");
    assert!(r.is_err(), "Whitespace-only query should produce an error");
}

/// EXPLAIN returns a single row with the plan column.
#[test]
fn explain_returns_plan_tree() {
    let db = seed_people();
    let s = db.session();
    let r = s
        .execute("EXPLAIN MATCH (n:Person) WHERE n.age > 25 RETURN n.name")
        .unwrap();
    assert_eq!(r.columns, vec!["plan"]);
    assert_eq!(r.row_count(), 1);
    // The plan text should mention key operators
    if let Value::String(plan_text) = &r.rows()[0][0] {
        assert!(
            !plan_text.is_empty(),
            "EXPLAIN plan text should not be empty"
        );
    } else {
        panic!("EXPLAIN should return a String value");
    }
}

// =========================================================================
// 5. Transaction atomicity
// =========================================================================

/// Rollback undoes mutations: insert inside a transaction, rollback, verify empty.
#[test]
fn transaction_rollback_undoes_inserts() {
    let db = GrafeoDB::new_in_memory();
    let mut s = db.session();

    s.begin_transaction().unwrap();
    s.execute("INSERT (:Ghost {name: 'Mia'})").unwrap();
    // Within the transaction we should see the node
    let r = s.execute("MATCH (g:Ghost) RETURN g.name").unwrap();
    assert_eq!(
        r.row_count(),
        1,
        "Should see node within active transaction"
    );

    s.rollback().unwrap();

    // After rollback, node should be gone
    let r2 = s.execute("MATCH (g:Ghost) RETURN g.name").unwrap();
    assert_eq!(r2.row_count(), 0, "Rollback should undo the insert");
}

/// Commit persists mutations across sessions.
#[test]
fn transaction_commit_persists() {
    let db = GrafeoDB::new_in_memory();
    let mut s = db.session();

    s.begin_transaction().unwrap();
    s.execute("INSERT (:Keeper {name: 'Butch'})").unwrap();
    s.commit().unwrap();

    // New session should see the committed data
    let s2 = db.session();
    let r = s2.execute("MATCH (k:Keeper) RETURN k.name").unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Butch".into()));
}

/// Multiple inserts in one transaction: all or nothing.
#[test]
fn transaction_multiple_inserts_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut s = db.session();

    s.begin_transaction().unwrap();
    s.execute("INSERT (:Item {name: 'Berlin'})").unwrap();
    s.execute("INSERT (:Item {name: 'Paris'})").unwrap();
    s.execute("INSERT (:Item {name: 'Prague'})").unwrap();
    s.rollback().unwrap();

    let r = s.execute("MATCH (i:Item) RETURN i.name").unwrap();
    assert_eq!(r.row_count(), 0, "All 3 inserts should be rolled back");
}

// =========================================================================
// 6. Cross-session isolation and state
// =========================================================================

/// Two sessions operate independently: mutations in one don't leak to the other
/// before commit.
#[test]
fn session_isolation() {
    let db = GrafeoDB::new_in_memory();
    let mut s1 = db.session();
    let s2 = db.session();

    s1.begin_transaction().unwrap();
    s1.execute("INSERT (:Secret {name: 'Django'})").unwrap();

    // s2 should not see the uncommitted node
    let r = s2.execute("MATCH (s:Secret) RETURN s").unwrap();
    assert_eq!(
        r.row_count(),
        0,
        "Uncommitted data should not be visible to other sessions"
    );

    s1.commit().unwrap();
}

/// Successive execute() calls on the same session share state.
#[test]
fn session_state_persists_across_queries() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();

    s.execute("INSERT (:Counter {val: 1})").unwrap();
    s.execute("INSERT (:Counter {val: 2})").unwrap();

    let r = s
        .execute("MATCH (c:Counter) RETURN c.val ORDER BY c.val")
        .unwrap();
    assert_eq!(r.row_count(), 2);
    assert_eq!(r.rows()[0][0], Value::Int64(1));
    assert_eq!(r.rows()[1][0], Value::Int64(2));
}

// =========================================================================
// 7. Result structure edge cases
// =========================================================================

/// Query on empty store returns 0 rows but correct column metadata.
#[test]
fn empty_result_preserves_columns() {
    let db = GrafeoDB::new_in_memory();
    let s = db.session();
    let r = s
        .execute("MATCH (n:Nothing) RETURN n.x AS x, n.y AS y")
        .unwrap();
    assert_eq!(r.row_count(), 0);
    assert_eq!(r.columns, vec!["x", "y"]);
}

/// execution_time_ms is populated on normal (non-WASM) targets.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn result_has_execution_time() {
    let db = seed_people();
    let s = db.session();
    let r = s.execute("MATCH (n:Person) RETURN n.name").unwrap();
    assert!(
        r.execution_time_ms.is_some(),
        "execution_time_ms should be populated"
    );
}
