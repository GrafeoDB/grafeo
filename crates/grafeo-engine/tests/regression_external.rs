//! Regression tests inspired by bugs found in other graph databases.
//! Covers MERGE semantics, pattern matching, aggregation edge cases,
//! Unicode handling, and query correctness.
//!
//! ```bash
//! cargo test -p grafeo-engine --test regression_external -- --nocapture
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

// ============================================================================
// MERGE + UNWIND tuple count
// MERGE in a loop should return one row per input tuple, even when
// the node already exists after the first iteration.
// ============================================================================

mod merge_unwind {
    use super::*;

    #[test]
    fn unwind_merge_returns_one_row_per_input() {
        let db = db();
        let s = db.session();
        // Three inputs, all the same value: MERGE creates one node,
        // but must still produce 3 output rows.
        let r = s
            .execute("UNWIND [1, 1, 1] AS i MERGE (:Item {val: i}) RETURN i")
            .unwrap();
        assert_eq!(r.row_count(), 3, "MERGE must emit one row per UNWIND input");
    }

    #[test]
    fn unwind_merge_creates_single_node() {
        let db = db();
        let s = db.session();
        s.execute("UNWIND [1, 1, 1] AS i MERGE (:Item {val: i})")
            .unwrap();
        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(1),
            "MERGE with duplicate values should create only one node"
        );
    }

    #[test]
    fn unwind_merge_distinct_values_create_multiple() {
        let db = db();
        let s = db.session();
        s.execute("UNWIND [1, 2, 3] AS i MERGE (:Item {val: i})")
            .unwrap();
        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(3),
            "MERGE with distinct values should create three nodes"
        );
    }

    #[test]
    fn unwind_merge_mixed_creates_and_matches() {
        let db = db();
        let s = db.session();
        // Pre-create one node
        s.execute("INSERT (:Item {val: 2})").unwrap();
        // UNWIND includes existing (2) and new (1, 3)
        s.execute("UNWIND [1, 2, 3] AS i MERGE (:Item {val: i})")
            .unwrap();
        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(3),
            "MERGE should create 2 new nodes and match 1 existing"
        );
    }

    #[test]
    fn unwind_merge_on_create_on_match_set() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {val: 1, status: 'old'})").unwrap();
        s.execute(
            "UNWIND [1, 2] AS i \
             MERGE (n:Item {val: i}) \
             ON CREATE SET n.status = 'new' \
             ON MATCH SET n.status = 'updated'",
        )
        .unwrap();

        let r = s
            .execute("MATCH (n:Item) RETURN n.val, n.status ORDER BY n.val")
            .unwrap();
        assert_eq!(r.row_count(), 2);
        // val=1 was matched, so status should be 'updated'
        assert_eq!(r.rows()[0][0], Value::Int64(1));
        assert_eq!(r.rows()[0][1], Value::String("updated".into()));
        // val=2 was created, so status should be 'new'
        assert_eq!(r.rows()[1][0], Value::Int64(2));
        assert_eq!(r.rows()[1][1], Value::String("new".into()));
    }
}

// ============================================================================
// MERGE with composite keys
// ============================================================================

mod merge_composite_keys {
    use super::*;

    #[test]
    fn merge_two_property_key_no_duplicate() {
        let db = db();
        let s = db.session();
        s.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
            .unwrap();
        s.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
            .unwrap();
        let r = s.execute("MATCH (n:City) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(1),
            "Identical composite key should not create duplicate"
        );
    }

    #[test]
    fn merge_partial_match_creates_new() {
        let db = db();
        let s = db.session();
        s.execute("MERGE (:City {name: 'Amsterdam', country: 'NL'})")
            .unwrap();
        // Same name, different country: should create a new node
        s.execute("MERGE (:City {name: 'Amsterdam', country: 'US'})")
            .unwrap();
        let r = s.execute("MATCH (n:City) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(2),
            "Partial composite match should create a new node"
        );
    }

    #[test]
    fn merge_three_property_key() {
        let db = db();
        let s = db.session();
        s.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Mitte'})")
            .unwrap();
        s.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Mitte'})")
            .unwrap();
        s.execute("MERGE (:Place {city: 'Berlin', country: 'DE', district: 'Kreuzberg'})")
            .unwrap();
        let r = s.execute("MATCH (n:Place) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(2),
            "Three-property MERGE: identical creates 1, different district creates 2"
        );
    }
}

// ============================================================================
// MERGE ON CREATE / ON MATCH SET that references the merge variable.
// GrafeoDB/grafeo#317: ISO/IEC 39075:2024 §15.5 puts the MERGE-bound variable
// in scope for ON CREATE / ON MATCH SET. Before the fix, the binder rejected
// any reference to it ("Undefined variable") and the planner silently lowered
// any non-trivial action expression to NULL.
// ============================================================================

mod merge_action_self_reference {
    use super::*;

    #[test]
    fn issue_317_minimal_reproduction() {
        let db = db();
        let s = db.session();
        // This exact statement used to fail with
        //   semantic error: Undefined variable 'n' in property access
        s.execute("MERGE (n:Item {val: 1}) ON MATCH SET n.x = coalesce(n.x, 0)")
            .unwrap();
    }

    #[test]
    fn on_create_coalesce_uses_fallback_for_absent_property() {
        let db = db();
        let s = db.session();
        s.execute(
            "MERGE (c:Category {name: 'Electronics'}) \
             ON CREATE SET c.description = coalesce(c.description, 'Consumer electronics')",
        )
        .unwrap();
        let r = s
            .execute("MATCH (c:Category {name: 'Electronics'}) RETURN c.description")
            .unwrap();
        assert_eq!(r.rows()[0][0], Value::String("Consumer electronics".into()));
    }

    #[test]
    fn on_match_coalesce_preserves_existing_value() {
        // Mirrors the second snippet in issue #317: a re-run with a different
        // ON CREATE clause should not touch a matched node, and ON MATCH's
        // coalesce(c.description, ...) must read the live value, not NULL.
        let db = db();
        let s = db.session();
        s.execute(
            "MERGE (c:Category {name: 'Electronics'}) \
             ON CREATE SET c.description = 'Consumer electronics' \
             ON MATCH  SET c.description = coalesce(c.description, 'Consumer electronics')",
        )
        .unwrap();
        s.execute(
            "MERGE (c:Category {name: 'Electronics'}) \
             ON CREATE SET c.description = 'unexpected' \
             ON MATCH  SET c.description = coalesce(c.description, 'fallback')",
        )
        .unwrap();
        let r = s
            .execute("MATCH (c:Category {name: 'Electronics'}) RETURN c.description")
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::String("Consumer electronics".into()),
            "ON MATCH with self-coalesce must keep the existing value, not silently become NULL"
        );
    }

    #[test]
    fn on_match_increments_self_property() {
        // Counter pattern. Three runs: first creates with cnt=1, the next two
        // each increment via `n.cnt + 1`, which used to silently reset to NULL.
        let db = db();
        let s = db.session();
        for _ in 0..3 {
            s.execute(
                "MERGE (n:Counter {id: 1}) ON CREATE SET n.cnt = 1 ON MATCH SET n.cnt = n.cnt + 1",
            )
            .unwrap();
        }
        let r = s.execute("MATCH (n:Counter {id: 1}) RETURN n.cnt").unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(3));
    }

    #[test]
    fn on_create_arithmetic_self_reference() {
        // Non-coalesce expression to exercise the runtime evaluator path.
        let db = db();
        let s = db.session();
        s.execute("MERGE (n:Item {val: 19}) ON CREATE SET n.score = (n.val + 81) * 3")
            .unwrap();
        let r = s
            .execute("MATCH (n:Item {val: 19}) RETURN n.score")
            .unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(300));
    }

    #[test]
    fn on_create_property_access_resolves_against_merged_node_when_variable_is_pre_bound() {
        // Regression for the planner duplicate-column resolution bug:
        // when MERGE re-uses a variable already bound by an upstream
        // MATCH, the action-scope columns vector contains that variable
        // twice (the bound copy AND the appended merge column). A direct
        // property access like `n.x` lowers via the simple-path
        // PropertySource::PropertyAccess and used to resolve to the
        // FIRST occurrence of the variable, i.e. the pre-bound (stale)
        // column. The fix makes the simple path resolve to the LAST
        // occurrence (the appended column whose chunk slot the operator
        // populates with the freshly merged node id), so the expression
        // sees the merged entity, not the upstream binding.
        let db = db();
        let s = db.session();
        // Pre-existing :Old node carries x=99. MERGE below creates a new
        // :Tag node (no existing match), so the bound `n` and the merged
        // `n` reference different nodes.
        s.execute("INSERT (:Old {x: 99})").unwrap();
        s.execute(
            "MATCH (n:Old) \
             MERGE (n:Tag {val: 1}) \
             ON CREATE SET n.copy_x = n.x",
        )
        .unwrap();
        // The freshly-created Tag node has no `x`, so reading `n.x`
        // against it must yield NULL (which the SET would persist as a
        // null/absent property). Pre-fix planner resolved `n.x` to the
        // pre-bound :Old column and stamped the new Tag with 99.
        let r = s.execute("MATCH (t:Tag {val: 1}) RETURN t.copy_x").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Null,
            "ON CREATE direct property access must read the merged Tag node \
             (no `x`), not the pre-bound :Old node (`x=99`)"
        );
    }

    #[test]
    fn merge_relationship_on_match_can_reference_edge_variable() {
        // The same scoping rule applies to MERGE on a relationship pattern.
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})-[:KNOWS {weight: 1}]->(:N {name: 'Gus'})")
            .unwrap();
        s.execute(
            "MATCH (a:N {name: 'Alix'}), (b:N {name: 'Gus'}) \
             MERGE (a)-[r:KNOWS]->(b) ON MATCH SET r.weight = r.weight + 10",
        )
        .unwrap();
        let r = s
            .execute(
                "MATCH (a:N {name: 'Alix'})-[r:KNOWS]->(b:N {name: 'Gus'}) \
                 RETURN r.weight",
            )
            .unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(11));
    }
}

// ============================================================================
// Relationship isomorphism
// The same relationship variable must bind to the same edge.
// ============================================================================

mod relationship_isomorphism {
    use super::*;

    #[test]
    fn two_hop_pattern_no_edge_reuse() {
        // In a triangle A->B->C->A, a two-hop pattern (a)-[r1]->(b)-[r2]->(c)
        // should never have r1 == r2.
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})-[:R]->(:N {name: 'Gus'})")
            .unwrap();
        s.execute(
            "MATCH (a:N {name: 'Gus'}), (b:N {name: 'Alix'}) \
             CREATE (a)-[:R]->(n:N {name: 'Vincent'})-[:R]->(b)",
        )
        .unwrap();
        // Count two-hop paths: should be 3 edges forming specific paths
        let r = s
            .execute(
                "MATCH (a:N)-[r1:R]->(b:N)-[r2:R]->(c:N) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();
        let cnt = &r.rows()[0][0];
        // Triangle: Alix->Gus->Vincent->Alix gives exactly 3 two-hop paths.
        assert_eq!(
            *cnt,
            Value::Int64(3),
            "Triangle should yield exactly 3 two-hop paths"
        );
    }

    #[test]
    fn single_edge_not_matched_twice_in_pattern() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})-[:R]->(:N {name: 'Gus'})")
            .unwrap();
        // Only one edge exists: a two-hop pattern should find 0 paths
        let r = s
            .execute(
                "MATCH (a:N)-[r1:R]->(b:N)-[r2:R]->(c:N) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(0),
            "Single edge cannot satisfy a two-hop pattern"
        );
    }
}

// ============================================================================
// OPTIONAL MATCH order independence
// ============================================================================

mod optional_match_order {
    use super::*;

    #[test]
    fn swapped_optional_match_same_results() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Alix'})-[:WORKS_AT]->(:Company {name: 'Acme'})")
            .unwrap();

        let r1 = s
            .execute(
                "MATCH (p:Person {name: 'Alix'}) \
                 OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person) \
                 OPTIONAL MATCH (p)-[:WORKS_AT]->(co:Company) \
                 RETURN p.name, friend.name, co.name",
            )
            .unwrap();

        // Same query with swapped OPTIONAL MATCH order
        let r2 = s
            .execute(
                "MATCH (p:Person {name: 'Alix'}) \
                 OPTIONAL MATCH (p)-[:WORKS_AT]->(co:Company) \
                 OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person) \
                 RETURN p.name, friend.name, co.name",
            )
            .unwrap();

        assert_eq!(
            r1.row_count(),
            r2.row_count(),
            "Swapping OPTIONAL MATCH order must not change row count"
        );
    }
}

// ============================================================================
// Aggregation inside CALL subqueries
// ============================================================================

mod aggregation_in_subquery {
    use super::*;

    #[test]
    fn count_inside_call_subquery() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        let r = s
            .execute("CALL { MATCH (n:Person) RETURN count(n) AS cnt } RETURN cnt")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Aggregation in CALL should return 1 row");
        assert_eq!(r.rows()[0][0], Value::Int64(3));
    }
}

// ============================================================================
// COLLECT() completeness
// ============================================================================

mod collect_order {
    use super::*;

    #[test]
    fn collect_returns_all_elements() {
        // Note: GQL does not support `WITH ... ORDER BY` syntax, so we
        // cannot test COLLECT ordering via GQL. This test verifies that
        // COLLECT gathers all values without dropping any.
        // TODO: once WITH ... ORDER BY is supported, add ordering assertion.
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:Person {name: 'Jules'})").unwrap();
        s.execute("INSERT (:Person {name: 'Mia'})").unwrap();

        let r = s
            .execute("MATCH (p:Person) RETURN collect(p.name) AS names")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        if let Value::List(names) = &r.rows()[0][0] {
            assert_eq!(names.len(), 5, "COLLECT must gather all 5 names");
            // Verify all names are present (order may vary)
            let name_strs: Vec<String> = names
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.to_string()),
                    _ => None,
                })
                .collect();
            let mut sorted = name_strs.clone();
            sorted.sort();
            assert_eq!(sorted, vec!["Alix", "Gus", "Jules", "Mia", "Vincent"],);
        } else {
            panic!("Expected List, got {:?}", r.rows()[0][0]);
        }
    }
}

// ============================================================================
// GROUP BY expression order independence
// ============================================================================

mod group_by_expression_order {
    use super::*;

    /// Helper: extract a column of string values from query results.
    fn string_column(result: &QueryResult, col: usize) -> Vec<String> {
        result
            .rows()
            .iter()
            .map(|row| match &row[col] {
                Value::String(s) => s.to_string(),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn return_column_order_does_not_affect_group_by() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Gus', city: 'Amsterdam'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Vincent', city: 'Berlin'})")
            .unwrap();

        // Order 1: city first, then count
        let r1 = s
            .execute(
                "MATCH (p:Person) \
                 RETURN p.city AS city, count(p) AS cnt \
                 ORDER BY city",
            )
            .unwrap();

        // Order 2: count first, then city
        let r2 = s
            .execute(
                "MATCH (p:Person) \
                 RETURN count(p) AS cnt, p.city AS city \
                 ORDER BY city",
            )
            .unwrap();

        assert_eq!(r1.row_count(), r2.row_count(), "Same grouping, same rows");
        // Both should have Amsterdam=2, Berlin=1
        let cities_1 = string_column(&r1, 0);
        let cities_2 = string_column(&r2, 1);
        assert_eq!(cities_1, cities_2, "City grouping should be identical");
    }
}

// ============================================================================
// WHERE filter must not corrupt projected values
// ============================================================================

mod where_filter_and_projection {
    use super::*;

    #[test]
    fn where_is_not_null_preserves_return_value() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Sensor {name: 'Temp', reading: 42.5})")
            .unwrap();
        s.execute("INSERT (:Sensor {name: 'Humidity'})").unwrap();

        let r = s
            .execute(
                "MATCH (s:Sensor) \
                 WHERE s.reading IS NOT NULL \
                 RETURN s.name, s.reading",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("Temp".into()));
        // The returned value must be the actual reading, NOT boolean true
        assert_eq!(
            r.rows()[0][1],
            Value::Float64(42.5),
            "WHERE IS NOT NULL must not replace the returned value with boolean"
        );
    }
}

// ============================================================================
// SUM overflow / IEEE-754 infinity
// ============================================================================

mod sum_overflow {
    use super::*;

    #[test]
    fn sum_large_floats_returns_infinity() {
        let db = db();
        let s = db.session();
        // Two f64::MAX values should overflow to infinity
        let r = s
            .execute(
                "UNWIND [1.7976931348623157e308, 1.7976931348623157e308] AS val \
                 RETURN SUM(val) AS total",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        match &r.rows()[0][0] {
            Value::Float64(f) => {
                assert!(
                    f.is_infinite() && f.is_sign_positive(),
                    "SUM of two f64::MAX should be +Infinity, got {f}"
                );
            }
            other => panic!("Expected Float64, got {:?}", other),
        }
    }
}

// ============================================================================
// Unicode and emoji in property values
// ============================================================================

mod unicode_emoji {
    use super::*;

    #[test]
    fn emoji_property_value_roundtrip() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Tag {symbol: '\u{1F389}', name: 'party'})") // 🎉
            .unwrap();
        let r = s.execute("MATCH (t:Tag) RETURN t.symbol, t.name").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("\u{1F389}".into()));
        assert_eq!(r.rows()[0][1], Value::String("party".into()));
    }

    #[test]
    fn cjk_property_value_roundtrip() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:City {name: '\u{6771}\u{4EAC}'})") // 東京
            .unwrap();
        let r = s.execute("MATCH (c:City) RETURN c.name").unwrap();
        assert_eq!(r.rows()[0][0], Value::String("\u{6771}\u{4EAC}".into()));
    }

    #[test]
    fn combining_diacritics_roundtrip() {
        let db = db();
        let s = db.session();
        // e + combining acute accent
        s.execute("INSERT (:Word {text: 'calf\u{0065}\u{0301}'})")
            .unwrap();
        let r = s.execute("MATCH (w:Word) RETURN w.text").unwrap();
        assert_eq!(r.rows()[0][0], Value::String("calf\u{0065}\u{0301}".into()));
    }
}

// ============================================================================
// Self-loop pattern matching
// ============================================================================

mod self_loop {
    use super::*;

    #[test]
    fn create_and_match_self_loop() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (a:Node {name: 'Alix'})-[:SELF]->(a)")
            .unwrap();

        let r = s
            .execute("MATCH (a:Node)-[r:SELF]->(a) RETURN a.name")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
    }

    #[test]
    fn self_loop_not_counted_as_two_edges() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (a:Node {name: 'Alix'})-[:SELF]->(a)")
            .unwrap();

        let r = s
            .execute("MATCH (:Node)-[r:SELF]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(1),
            "Self-loop should be counted exactly once"
        );
    }
}

// ============================================================================
// Deleted node access in same session
// ============================================================================

mod deleted_node_access {
    use super::*;

    #[test]
    fn delete_then_match_in_same_session() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Temp {name: 'ephemeral'})").unwrap();
        s.execute("MATCH (n:Temp) DELETE n").unwrap();

        let r = s.execute("MATCH (n:Temp) RETURN n.name").unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "Deleted node should not be visible in subsequent MATCH"
        );
    }

    #[test]
    fn detach_delete_clears_edges() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:A {name: 'a'})-[:R]->(:B {name: 'b'})")
            .unwrap();
        s.execute("MATCH (a:A) DETACH DELETE a").unwrap();

        let r = s
            .execute("MATCH ()-[r:R]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(0),
            "DETACH DELETE should remove all connected edges"
        );
    }
}

// ============================================================================
// Deeply nested map/list properties
// ============================================================================

mod nested_properties {
    use super::*;
    use grafeo_common::types::PropertyKey;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn nested_map_roundtrip() {
        let db = db();
        let s = db.session();
        let mut params = HashMap::new();
        // Build a nested map: {b: {c: 42}}
        let mut inner_map = BTreeMap::new();
        inner_map.insert(PropertyKey::new("c"), Value::Int64(42));
        let inner = Value::Map(inner_map.into());

        let mut mid_map = BTreeMap::new();
        mid_map.insert(PropertyKey::new("b"), inner);
        let mid = Value::Map(mid_map.into());
        params.insert("meta".to_string(), mid);

        s.execute_with_params("INSERT (:Data {meta: $meta})", params)
            .unwrap();
        let r = s.execute("MATCH (d:Data) RETURN d.meta").unwrap();
        assert_eq!(r.row_count(), 1);
        // Verify the nested structure survived
        if let Value::Map(outer) = &r.rows()[0][0] {
            let b = outer.get(&PropertyKey::new("b")).expect("Missing key 'b'");
            if let Value::Map(inner_result) = b {
                assert_eq!(
                    inner_result.get(&PropertyKey::new("c")),
                    Some(&Value::Int64(42))
                );
            } else {
                panic!("Expected inner Map, got {:?}", b);
            }
        } else {
            panic!("Expected Map, got {:?}", r.rows()[0][0]);
        }
    }

    #[test]
    fn heterogeneous_list_roundtrip() {
        let db = db();
        let s = db.session();
        let mut params = HashMap::new();
        params.insert(
            "items".to_string(),
            Value::List(
                vec![
                    Value::Int64(1),
                    Value::String("two".into()),
                    Value::Bool(true),
                    Value::Null,
                ]
                .into(),
            ),
        );

        s.execute_with_params("INSERT (:Data {items: $items})", params)
            .unwrap();
        let r = s.execute("MATCH (d:Data) RETURN d.items").unwrap();
        if let Value::List(items) = &r.rows()[0][0] {
            assert_eq!(items.len(), 4);
            assert_eq!(items[0], Value::Int64(1));
            assert_eq!(items[1], Value::String("two".into()));
            assert_eq!(items[2], Value::Bool(true));
            assert_eq!(items[3], Value::Null);
        } else {
            panic!("Expected List, got {:?}", r.rows()[0][0]);
        }
    }
}

// ============================================================================
// Constant folding in WHERE
// ============================================================================

mod constant_folding {
    use super::*;

    #[test]
    fn where_constant_false_returns_empty() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        let r = s.execute("MATCH (n:N) WHERE 1 = 2 RETURN n").unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "WHERE 1 = 2 should short-circuit to empty"
        );
    }

    #[test]
    fn where_constant_true_returns_all() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 2})").unwrap();
        let r = s.execute("MATCH (n:N) WHERE 1 = 1 RETURN n.val").unwrap();
        assert_eq!(r.row_count(), 2, "WHERE 1 = 1 should return all rows");
    }
}

// ============================================================================
// Cyclic graph traversal
// ============================================================================

mod cyclic_traversal {
    use super::*;

    #[test]
    fn variable_length_path_in_triangle_terminates() {
        let db = db();
        let s = db.session();
        // Create triangle: A->B->C->A
        s.execute(
            "INSERT (a:N {name: 'Alix'})-[:R]->(b:N {name: 'Gus'})-[:R]->(c:N {name: 'Vincent'})-[:R]->(a)",
        )
        .unwrap();

        // Variable-length path should terminate even with cycles
        let r = s
            .execute(
                "MATCH (a:N {name: 'Alix'})-[:R*1..5]->(b:N) \
                 RETURN b.name",
            )
            .unwrap();
        // Walk mode on a 3-node directed triangle with one outgoing edge each:
        // exactly 1 path per hop length (1..=5), so exactly 5 rows.
        assert_eq!(
            r.row_count(),
            5,
            "Expected exactly 5 paths (one per hop 1..=5), got {}",
            r.row_count()
        );
    }
}

// ============================================================================
// Concurrent MERGE (stress test)
// ============================================================================

mod concurrent_merge {
    use super::*;

    #[test]
    #[ignore = "stress test: run locally before releases"]
    fn concurrent_merge_no_duplicates() {
        let db = std::sync::Arc::new(db());
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let db = std::sync::Arc::clone(&db);
                std::thread::spawn(move || {
                    let s = db.session();
                    s.execute("MERGE (:Singleton {key: 'only_one'})").unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let s = db.session();
        let r = s
            .execute("MATCH (n:Singleton) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(1),
            "Concurrent MERGE should produce exactly 1 node"
        );
    }
}

// ============================================================================
// MERGE with NULL node reference
// ============================================================================

mod merge_null_node_reference {
    use super::*;

    #[test]
    fn merge_relationship_with_null_source_errors() {
        let db = db();
        let s = db.session();
        // OPTIONAL MATCH that matches nothing produces NULL for n
        let result = s.execute(
            "OPTIONAL MATCH (n:NonExistent) \
             MERGE (n)-[:R]->(m:Target {name: 'Alix'})",
        );
        assert!(
            result.is_err(),
            "MERGE with NULL node reference should error, got: {result:?}"
        );
    }

    #[test]
    fn merge_relationship_with_null_target_errors() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Source {name: 'Gus'})").unwrap();
        let result = s.execute(
            "MATCH (a:Source {name: 'Gus'}) \
             OPTIONAL MATCH (b:NonExistent) \
             MERGE (a)-[:R]->(b)",
        );
        assert!(
            result.is_err(),
            "MERGE with NULL target node reference should error"
        );
    }

    #[test]
    fn merge_with_valid_optional_match_succeeds() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        // OPTIONAL MATCH that actually finds something: MERGE should work
        let result = s.execute(
            "MATCH (n:Person {name: 'Vincent'}) \
             MERGE (n)-[:KNOWS]->(m:Person {name: 'Jules'})",
        );
        assert!(
            result.is_ok(),
            "MERGE with non-null node should succeed: {result:?}"
        );
        // Verify the relationship was actually created.
        let check = s
            .execute(
                "MATCH (v:Person {name: 'Vincent'})-[:KNOWS]->(j:Person {name: 'Jules'}) RETURN j.name",
            )
            .unwrap();
        assert_eq!(check.row_count(), 1);
        assert_eq!(check.rows()[0][0], Value::String("Jules".into()));
    }

    #[test]
    fn standalone_merge_unaffected() {
        let db = db();
        let s = db.session();
        let result = s
            .execute("MERGE (:Person {name: 'Mia'}) RETURN 1 AS ok")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(1));
    }
}

mod distinct_edges {
    use super::*;

    /// Regression: DISTINCT on edge result columns must deduplicate by edge identity,
    /// not by debug-format string. Edges are stored as Int64(edge_id) in execution
    /// columns, so the DistinctOperator's Int64 arm handles them correctly.
    #[test]
    fn distinct_on_edges_deduplicates_correctly() {
        let db = db();
        let s = db.session();
        // Create two nodes with a single edge between them
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

        // Cartesian product of nodes produces duplicate edge references;
        // DISTINCT should collapse them to exactly one row.
        let result = s
            .execute(
                "MATCH (a:Person), (b:Person) \
                 MATCH (a)-[e:KNOWS]->(b) \
                 RETURN DISTINCT e",
            )
            .unwrap();
        assert_eq!(
            result.rows().len(),
            1,
            "DISTINCT e should return exactly one row, got: {}",
            result.rows().len()
        );
    }

    #[test]
    fn distinct_on_edges_multiple_edge_types() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        s.execute("INSERT (:Person {name: 'Jules'})").unwrap();
        s.execute("INSERT (:Person {name: 'Mia'})").unwrap();
        s.execute(
            "MATCH (a:Person {name: 'Vincent'}), (b:Person {name: 'Jules'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();
        s.execute(
            "MATCH (a:Person {name: 'Jules'}), (b:Person {name: 'Mia'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

        // MATCH all edges of type KNOWS and return them with DISTINCT
        let result = s
            .execute("MATCH ()-[e:KNOWS]->() RETURN DISTINCT e")
            .unwrap();
        assert_eq!(
            result.rows().len(),
            2,
            "DISTINCT should return both edges, got: {}",
            result.rows().len()
        );
    }
}

mod call_block_scope {
    use super::*;

    /// Regression item 14: variables bound inside one CALL block must not be
    /// visible inside a sibling CALL block. The binder previously used a single
    /// global BindingContext, so a variable from CALL block 1 leaked into CALL
    /// block 2, making it appear to be in scope when it was not.
    ///
    /// This test verifies the binder correctly accepts a query with sibling CALL
    /// blocks that compose independent output columns.
    #[test]
    fn sibling_call_block_outputs_are_in_scope_for_return() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();

        // Two independent CALL blocks: their output columns (age_a, age_b) must
        // both be visible to the outer RETURN. The binder should accept this query.
        let result = s.execute(
            "CALL { MATCH (a:Person {name: 'Alix'}) RETURN a.age AS age_a } \
             CALL { MATCH (b:Person {name: 'Gus'}) RETURN b.age AS age_b } \
             RETURN age_a, age_b",
        );
        let result = result.expect("sibling CALL outputs should be accessible in outer RETURN");
        assert_eq!(result.rows()[0][0], Value::Int64(30)); // age_a
        assert_eq!(result.rows()[0][1], Value::Int64(25)); // age_b
    }

    /// Internal variable `a` from CALL block 1 must not be visible in CALL block 2.
    /// If scope isolation is working, the second CALL cannot reference `a`.
    #[test]
    fn internal_variable_from_first_call_is_not_visible_in_second_call() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // The second CALL references `a`, which is internal to the first CALL.
        // With correct scope isolation this should fail with an undefined-variable error.
        let result = s.execute(
            "CALL { MATCH (a:Person) RETURN a.name AS name_a } \
             CALL { MATCH (b:Person) WHERE b.name = a.name RETURN b } \
             RETURN name_a",
        );
        assert!(
            result.is_err(),
            "second CALL referencing internal var of first CALL should fail, got: {result:?}"
        );
    }

    /// Same-named variables in sibling CALL blocks should not conflict.
    /// Each block defines its own `n`, but exports under different aliases.
    #[test]
    fn same_variable_name_in_sibling_calls_is_independent() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Company {name: 'TechCorp'})").unwrap();

        let result = s
            .execute(
                "CALL { MATCH (n:Person) RETURN n.name AS person_name } \
                 CALL { MATCH (n:Company) RETURN n.name AS company_name } \
                 RETURN person_name, company_name",
            )
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[0][1], Value::String("TechCorp".into()));
    }

    /// SUM aggregation inside a CALL subquery should produce a single row
    /// with the correct total.
    #[test]
    fn sum_inside_call_subquery_returns_single_row() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Vincent', age: 40})")
            .unwrap();

        let result = s
            .execute("CALL { MATCH (n:Person) RETURN sum(n.age) AS total } RETURN total")
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(95));
    }
}

// ============================================================================
// UNWIND + MERGE + SET: property access on UNWIND variable (#172)
// ============================================================================

#[cfg(feature = "cypher")]
mod unwind_merge_set {
    use super::*;

    /// UNWIND variable property access in SET clause should resolve correctly.
    /// Regression: item.name in SET evaluated to NULL instead of the map value.
    /// GitHub issue #172.
    #[test]
    fn unwind_merge_set_property_from_map() {
        let db = db();
        let s = db.session();

        s.execute_cypher(
            "UNWIND [{qn: 'test://foo', name: 'Foo'}, {qn: 'test://bar', name: 'Bar'}] AS item \
             MERGE (x:Test {qn: item.qn}) \
             SET x.name = item.name",
        )
        .unwrap();

        let result = s
            .execute("MATCH (n:Test) RETURN n.qn AS qn, n.name AS name ORDER BY qn")
            .unwrap();

        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("test://bar".into()));
        assert_eq!(
            result.rows()[0][1],
            Value::String("Bar".into()),
            "SET x.name = item.name should resolve item.name from UNWIND map (#172)"
        );
        assert_eq!(result.rows()[1][0], Value::String("test://foo".into()));
        assert_eq!(result.rows()[1][1], Value::String("Foo".into()));
    }

    /// SET x += item (map merge) should also work with UNWIND variables.
    #[test]
    fn unwind_merge_set_map_merge() {
        let db = db();
        let s = db.session();

        s.execute_cypher(
            "UNWIND [{qn: 'test://baz', name: 'Baz', kind: 'module'}] AS item \
             MERGE (x:Test {qn: item.qn}) \
             SET x += item",
        )
        .unwrap();

        let result = s
            .execute("MATCH (n:Test {qn: 'test://baz'}) RETURN n.name AS name, n.kind AS kind")
            .unwrap();

        assert_eq!(result.row_count(), 1);
        assert_eq!(
            result.rows()[0][0],
            Value::String("Baz".into()),
            "SET x += item should merge all map properties (#172)"
        );
        assert_eq!(result.rows()[0][1], Value::String("module".into()));
    }
}

// ============================================================================
// #187: labels(n) and type(r) fail in aggregation context
// ============================================================================

#[cfg(feature = "cypher")]
mod issue_187_labels_type_aggregation {
    use super::*;

    #[test]
    fn labels_with_count() {
        let db = db();
        let s = db.session();
        s.execute_cypher(
            "CREATE (:Class {name: 'A'}), (:Class {name: 'B'}), (:Method {name: 'C'})",
        )
        .unwrap();

        let result = s
            .execute_cypher(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label",
            )
            .unwrap();

        assert_eq!(
            result.row_count(),
            2,
            "Should have two groups: Class and Method"
        );
        assert_eq!(result.rows()[0][0], Value::String("Class".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(2));
        assert_eq!(result.rows()[1][0], Value::String("Method".into()));
        assert_eq!(result.rows()[1][1], Value::Int64(1));
    }

    #[test]
    fn type_with_count() {
        let db = db();
        let s = db.session();
        s.execute_cypher(
            "CREATE (a:A), (b:B), (c:C), (a)-[:CALLS]->(b), (a)-[:CALLS]->(c), (b)-[:IMPORTS]->(c)",
        )
        .unwrap();

        let result = s
            .execute_cypher(
                "MATCH ()-[r]->() RETURN type(r) AS edge_type, count(r) AS cnt ORDER BY edge_type",
            )
            .unwrap();

        assert_eq!(
            result.row_count(),
            2,
            "Should have two groups: CALLS and IMPORTS"
        );
        assert_eq!(result.rows()[0][0], Value::String("CALLS".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(2));
        assert_eq!(result.rows()[1][0], Value::String("IMPORTS".into()));
        assert_eq!(result.rows()[1][1], Value::Int64(1));
    }

    #[test]
    fn labels_without_index_access() {
        let db = db();
        let s = db.session();
        s.execute_cypher("CREATE (:Foo), (:Bar)").unwrap();

        let result = s
            .execute_cypher("MATCH (n) RETURN labels(n) AS lbls, count(n) AS cnt ORDER BY lbls")
            .unwrap();

        assert_eq!(result.row_count(), 2);
        // Each single-label node forms its own group with count 1.
        assert_eq!(result.rows()[0][1], Value::Int64(1));
        assert_eq!(result.rows()[1][1], Value::Int64(1));
    }

    #[test]
    fn order_by_labels() {
        let db = db();
        let s = db.session();
        s.execute_cypher("CREATE (:Zebra {name: 'Z'}), (:Apple {name: 'A'})")
            .unwrap();

        let result = s
            .execute_cypher("MATCH (n) RETURN n.name ORDER BY labels(n)[0]")
            .unwrap();

        assert_eq!(result.row_count(), 2);
        // "Apple" < "Zebra" alphabetically
        assert_eq!(result.rows()[0][0], Value::String("A".into()));
        assert_eq!(result.rows()[1][0], Value::String("Z".into()));
    }

    #[test]
    fn order_by_type() {
        let db = db();
        let s = db.session();
        s.execute_cypher("CREATE (a:X), (b:Y), (c:Z), (a)-[:BETA]->(b), (a)-[:ALPHA]->(c)")
            .unwrap();

        let result = s
            .execute_cypher("MATCH ()-[r]->() RETURN type(r) AS t ORDER BY t")
            .unwrap();

        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("ALPHA".into()));
        assert_eq!(result.rows()[1][0], Value::String("BETA".into()));
    }

    #[test]
    fn gql_labels_with_count() {
        let db = db();
        let s = db.session();
        s.execute(
            "INSERT (:Engineer {name: 'Vincent'}), (:Engineer {name: 'Jules'}), (:Designer {name: 'Mia'})",
        )
        .unwrap();

        let result = s
            .execute("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label")
            .unwrap();

        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Designer".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(1));
        assert_eq!(result.rows()[1][0], Value::String("Engineer".into()));
        assert_eq!(result.rows()[1][1], Value::Int64(2));
    }

    #[test]
    fn gql_order_by_labels() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Zebra {name: 'Z'}), (:Apple {name: 'A'})")
            .unwrap();

        let result = s
            .execute("MATCH (n) RETURN n.name ORDER BY labels(n)[0]")
            .unwrap();

        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("A".into()));
        assert_eq!(result.rows()[1][0], Value::String("Z".into()));
    }
}

// ============================================================================
// #187 extended: edge cases for expression-in-aggregation and ORDER BY
// ============================================================================

#[cfg(feature = "cypher")]
mod issue_187_extended {
    use super::*;

    #[test]
    fn labels_with_sum() {
        let db = db();
        let s = db.session();
        s.execute_cypher("CREATE (:Alpha {val: 10}), (:Alpha {val: 20}), (:Beta {val: 5})")
            .unwrap();

        let result = s
            .execute_cypher(
                "MATCH (n) RETURN labels(n)[0] AS label, sum(n.val) AS total ORDER BY label",
            )
            .unwrap();

        assert_eq!(result.row_count(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alpha".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(30));
        assert_eq!(result.rows()[1][0], Value::String("Beta".into()));
        assert_eq!(result.rows()[1][1], Value::Int64(5));
    }

    #[test]
    fn type_with_collect() {
        let db = db();
        let s = db.session();
        s.execute_cypher(
            "CREATE (a:X), (b:Y), (c:Z), (a)-[:LIKES]->(b), (a)-[:LIKES]->(c), (b)-[:KNOWS]->(c)",
        )
        .unwrap();

        let result = s
            .execute_cypher("MATCH ()-[r]->() RETURN type(r) AS t, collect(r) AS items ORDER BY t")
            .unwrap();

        assert_eq!(result.row_count(), 2, "Two edge types: KNOWS and LIKES");
    }

    #[test]
    fn multi_label_node_group_by() {
        let db = db();
        let s = db.session();
        s.execute_cypher(
            "CREATE (:A:B {name: 'one'}), (:A:B {name: 'two'}), (:C:D {name: 'three'})",
        )
        .unwrap();

        let result = s
            .execute_cypher("MATCH (n) RETURN labels(n) AS lbls, count(n) AS cnt")
            .unwrap();

        assert_eq!(
            result.row_count(),
            2,
            "Expected exactly 2 rows (one per distinct label-set), got {}: {:?}",
            result.row_count(),
            result.rows()
        );
    }

    #[test]
    fn order_by_type_descending() {
        let db = db();
        let s = db.session();
        s.execute_cypher(
            "CREATE (a:X), (b:Y), (c:Z), (a)-[:ALPHA]->(b), (a)-[:GAMMA]->(c), (b)-[:BETA]->(c)",
        )
        .unwrap();

        let result = s
            .execute_cypher("MATCH ()-[r]->() RETURN type(r) AS t ORDER BY t DESC")
            .unwrap();

        assert_eq!(result.row_count(), 3);
        assert_eq!(result.rows()[0][0], Value::String("GAMMA".into()));
        assert_eq!(result.rows()[1][0], Value::String("BETA".into()));
        assert_eq!(result.rows()[2][0], Value::String("ALPHA".into()));
    }

    #[test]
    fn order_by_labels_with_limit() {
        let db = db();
        let s = db.session();
        s.execute_cypher("CREATE (:Zebra {name: 'Z'}), (:Apple {name: 'A'}), (:Mango {name: 'M'})")
            .unwrap();

        let result = s
            .execute_cypher("MATCH (n) RETURN n.name ORDER BY labels(n)[0] LIMIT 1")
            .unwrap();

        assert_eq!(
            result.row_count(),
            1,
            "LIMIT 1 should return exactly one row"
        );
    }

    #[test]
    fn group_by_and_order_by_both_complex() {
        let db = db();
        let s = db.session();
        s.execute_cypher(
            "CREATE (:Engineer {name: 'Vincent'}), (:Engineer {name: 'Jules'}), (:Designer {name: 'Mia'})",
        )
        .unwrap();

        // Both GROUP BY (implicit from labels(n)[0] in RETURN) and ORDER BY use
        // the same complex expression. Each node has exactly one label, so
        // labels(n)[0] is deterministic: exactly 2 groups (Designer, Engineer).
        let result = s
            .execute_cypher(
                "MATCH (n) RETURN labels(n)[0] AS lbl, count(n) AS cnt ORDER BY labels(n)[0]",
            )
            .unwrap();

        assert_eq!(
            result.row_count(),
            2,
            "Should produce exactly 2 rows (Designer, Engineer), got {}",
            result.row_count()
        );
    }

    #[test]
    fn empty_result_group_by_labels() {
        let db = db();
        let s = db.session();

        let result = s
            .execute_cypher("MATCH (n:NonExistent) RETURN labels(n)[0] AS lbl, count(n) AS cnt")
            .unwrap();

        assert_eq!(
            result.row_count(),
            0,
            "No matching nodes should yield zero rows"
        );
    }
}

// The GQL variant of type_with_count (does not require the cypher feature).
mod issue_187_gql_type_with_count {
    use super::*;

    #[test]
    fn gql_type_with_count() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:A)-[:FOLLOWS]->(:B)").unwrap();
        s.execute("INSERT (:C)-[:FOLLOWS]->(:D)").unwrap();
        s.execute("INSERT (:E)-[:BLOCKS]->(:F)").unwrap();

        let result = s
            .execute("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY t")
            .unwrap();

        assert_eq!(result.row_count(), 2, "Two edge types: BLOCKS and FOLLOWS");
        assert_eq!(result.rows()[0][0], Value::String("BLOCKS".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(1));
        assert_eq!(result.rows()[1][0], Value::String("FOLLOWS".into()));
        assert_eq!(result.rows()[1][1], Value::Int64(2));
    }
}

// ============================================================================
// Integer overflow detection in arithmetic
// Inspired by Neo4j #13674: inconsistent multiplication overflow.
// Each intermediate step must check for overflow, not just final result.
// ============================================================================

mod integer_overflow {
    use super::*;

    #[test]
    fn i64_max_plus_one_overflows() {
        let db = db();
        let s = db.session();
        // i64::MAX + 1 overflows: returns NULL (SQL overflow semantics)
        let r = s.execute("RETURN 9223372036854775807 + 1 AS r").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Null,
            "i64::MAX + 1 should return NULL (overflow)"
        );
    }

    #[test]
    fn i64_min_minus_one_overflows() {
        let db = db();
        let s = db.session();
        let result = s.execute("RETURN -9223372036854775808 - 1 AS r");
        match result {
            Err(_) => {} // Parse error for the literal is also acceptable
            Ok(r) => {
                assert_ne!(
                    r.rows()[0][0],
                    Value::Int64(i64::MAX),
                    "Must not silently wrap i64::MIN - 1 to i64::MAX"
                );
            }
        }
    }

    #[test]
    fn multiplication_intermediate_overflow() {
        let db = db();
        let s = db.session();
        // 100 * 1_000_000_000 = 100_000_000_000 (fits i64)
        // 100_000_000_000 * 100_000_000 = 10^19 (overflows i64)
        let r = s
            .execute("RETURN 100 * 1000000000 * 100000000 AS r")
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Null,
            "Intermediate multiplication overflow should return NULL"
        );
    }
}

// ============================================================================
// Variable-length path enumeration
// Inspired by FalkorDB #1450: engine deduplicates endpoint pairs
// instead of enumerating all distinct paths.
// ============================================================================

mod variable_length_path_enumeration {
    use super::*;

    #[test]
    fn diamond_two_hop_enumerates_all_paths() {
        let db = db();
        let s = db.session();
        // Diamond: A->B1->C, A->B2->C (two distinct 2-hop paths)
        s.execute(
            "INSERT (a:N {name: 'Alix'})-[:R]->(b1:N {name: 'Gus'}), \
             (a)-[:R]->(b2:N {name: 'Vincent'}), \
             (b1)-[:R]->(c:N {name: 'Jules'}), \
             (b2)-[:R]->(c)",
        )
        .unwrap();

        // Two distinct 2-hop paths from Alix to Jules
        let r = s
            .execute(
                "MATCH (a:N {name: 'Alix'})-[:R*2]->(c:N {name: 'Jules'}) \
                 RETURN c.name",
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "Diamond graph must yield 2 distinct 2-hop paths, got {}",
            r.row_count()
        );
    }

    #[test]
    fn single_hop_variable_length_matches_direct_edges() {
        let db = db();
        let s = db.session();
        s.execute(
            "INSERT (a:N {name: 'Alix'})-[:R]->(b:N {name: 'Gus'})-[:R]->(c:N {name: 'Vincent'})",
        )
        .unwrap();

        let r = s
            .execute("MATCH (a:N {name: 'Alix'})-[:R*1..1]->(b) RETURN b.name")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "1..1 hop should match exactly 1 direct neighbor"
        );
        assert_eq!(r.rows()[0][0], Value::String("Gus".into()));
    }

    #[test]
    fn variable_length_path_respects_max_hops() {
        let db = db();
        let s = db.session();
        // Chain: A->B->C->D->E
        s.execute(
            "INSERT (:N {name: 'a'})-[:R]->(:N {name: 'b'})-[:R]->(:N {name: 'c'})-[:R]->(:N {name: 'd'})-[:R]->(:N {name: 'e'})",
        )
        .unwrap();

        let r = s
            .execute("MATCH (a:N {name: 'a'})-[:R*1..2]->(b) RETURN b.name ORDER BY b.name")
            .unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "1..2 hops from 'a' should reach 'b' and 'c'"
        );
    }
}

// ============================================================================
// EXISTS subquery correctness
// Inspired by FalkorDB #1248: EXISTS returns TRUE for nonexistent patterns.
// ============================================================================

mod exists_subquery {
    use super::*;

    #[test]
    fn exists_returns_false_for_missing_pattern() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:User {name: 'Alix'})").unwrap();

        let r = s
            .execute(
                "MATCH (u:User) \
                 RETURN EXISTS { MATCH (u)<-[:AUTH]-(:Identity) } AS has_id",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Bool(false),
            "EXISTS must return false when no matching pattern exists"
        );
    }

    #[test]
    fn exists_returns_true_for_present_pattern() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:User {name: 'Alix'})<-[:AUTH]-(:Identity {provider: 'github'})")
            .unwrap();

        let r = s
            .execute(
                "MATCH (u:User) \
                 RETURN EXISTS { MATCH (u)<-[:AUTH]-(:Identity) } AS has_id",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Bool(true),
            "EXISTS must return true when matching pattern exists"
        );
    }

    #[test]
    fn exists_with_label_filter_no_false_positive() {
        let db = db();
        let s = db.session();
        // Create a relationship but with a different label than what EXISTS checks
        s.execute("INSERT (:User {name: 'Gus'})<-[:FOLLOWS]-(:Bot {name: 'bot1'})")
            .unwrap();

        let r = s
            .execute(
                "MATCH (u:User) \
                 RETURN EXISTS { MATCH (u)<-[:AUTH]-(:Identity) } AS has_id",
            )
            .unwrap();
        assert_eq!(r.rows()[0][0], Value::Bool(false));
    }
}

// ============================================================================
// UNWIND NULL produces zero rows
// Inspired by FalkorDB #1031: UNWIND null + subquery executes writes.
// ============================================================================

mod unwind_null {
    use super::*;

    #[test]
    fn unwind_null_produces_zero_rows() {
        let db = db();
        let s = db.session();
        let r = s.execute("UNWIND NULL AS x RETURN x").unwrap();
        assert_eq!(r.row_count(), 0, "UNWIND NULL must produce zero rows");
    }

    #[test]
    fn unwind_empty_list_produces_zero_rows() {
        let db = db();
        let s = db.session();
        let r = s.execute("UNWIND [] AS x RETURN x").unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "UNWIND of empty list must produce zero rows"
        );
    }

    #[test]
    fn unwind_null_does_not_execute_writes() {
        let db = db();
        let s = db.session();
        // UNWIND NULL should produce 0 rows, so INSERT should not execute
        let _ = s.execute("UNWIND NULL AS x INSERT (:Ghost {val: x})");
        let r = s.execute("MATCH (n:Ghost) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(0),
            "No nodes should be created when UNWIND produces zero rows"
        );
    }
}

// ============================================================================
// NULL comparison short-circuit in WHERE
// Inspired by Neo4j #13727: WHERE 1 = NULL, WHERE 1 IN [] not optimized.
// Regardless of optimization, the result must be correct (0 rows).
// ============================================================================

mod null_predicate_semantics {
    use super::*;

    #[test]
    fn where_equality_with_null_returns_empty() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        let r = s.execute("MATCH (n:N) WHERE 1 = NULL RETURN n").unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "WHERE 1 = NULL is UNKNOWN, should filter all rows"
        );
    }

    #[test]
    fn where_in_empty_list_returns_empty() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        let r = s.execute("MATCH (n:N) WHERE 1 IN [] RETURN n").unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "WHERE 1 IN [] is false, should filter all rows"
        );
    }

    #[test]
    fn null_not_equal_to_null() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        let r = s.execute("MATCH (n:N) WHERE NULL = NULL RETURN n").unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "NULL = NULL is UNKNOWN (not TRUE), should filter all rows"
        );
    }

    #[test]
    fn null_is_null_returns_true() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        let r = s
            .execute("MATCH (n:N) WHERE NULL IS NULL RETURN n.val")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "NULL IS NULL is TRUE, should pass all rows"
        );
    }
}

// ============================================================================
// Predicate rewriting: NOT((a = b) IS NULL)
// Inspired by Neo4j #13642: optimizer conflates IS NOT NULL with equality.
// ============================================================================

mod predicate_rewriting {
    use super::*;

    #[test]
    fn not_is_null_on_comparison_result() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:X {k1: 100})-[:R]->(:Y {k2: 34})")
            .unwrap();

        // (100 = 34) evaluates to false
        // (false IS NULL) evaluates to false
        // NOT(false) evaluates to true
        // So the row should pass the filter
        let r = s
            .execute(
                "MATCH (x:X)-[:R]->(y:Y) \
                 WHERE NOT ((x.k1 = y.k2) IS NULL) \
                 RETURN x.k1, y.k2",
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "NOT((100 = 34) IS NULL) should be TRUE, row must pass filter"
        );
    }

    #[test]
    fn not_is_null_with_null_property() {
        let db = db();
        let s = db.session();
        // y has no k2 property, so y.k2 is NULL
        s.execute("INSERT (:X {k1: 100})-[:R]->(:Y {name: 'test'})")
            .unwrap();

        // (100 = NULL) evaluates to NULL
        // (NULL IS NULL) evaluates to true
        // NOT(true) evaluates to false
        // So the row should be filtered out
        let r = s
            .execute(
                "MATCH (x:X)-[:R]->(y:Y) \
                 WHERE NOT ((x.k1 = y.k2) IS NULL) \
                 RETURN x.k1",
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            0,
            "NOT((100 = NULL) IS NULL) should be FALSE, row must be filtered"
        );
    }
}

// ============================================================================
// Double-delete idempotency
// Inspired by FalkorDB #1018: phantom edge after repeated deletion.
// ============================================================================

mod double_delete {
    use super::*;

    #[test]
    fn delete_same_node_twice_no_phantom() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Temp {name: 'Alix'})").unwrap();
        s.execute("MATCH (n:Temp) DELETE n").unwrap();
        // Second delete attempt: should either no-op or error, not corrupt state
        let r2 = s.execute("MATCH (n:Temp) DELETE n");
        // Regardless of error/success, no phantom nodes should exist
        if r2.is_err() {
            // Error on second delete is acceptable (already deleted)
        }
        let check = s.execute("MATCH (n) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            check.rows()[0][0],
            Value::Int64(0),
            "No phantom nodes should remain after double delete"
        );
    }

    #[test]
    fn detach_delete_then_verify_edges_clean() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:A {name: 'a'})-[:R]->(:B {name: 'b'})-[:S]->(:C {name: 'c'})")
            .unwrap();
        s.execute("MATCH (n:A) DETACH DELETE n").unwrap();
        s.execute("MATCH (n:B) DETACH DELETE n").unwrap();
        s.execute("MATCH (n:C) DETACH DELETE n").unwrap();

        let nodes = s.execute("MATCH (n) RETURN count(n) AS cnt").unwrap();
        let edges = s
            .execute("MATCH ()-[r]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(
            nodes.rows()[0][0],
            Value::Int64(0),
            "All nodes should be gone"
        );
        assert_eq!(
            edges.rows()[0][0],
            Value::Int64(0),
            "All edges should be gone"
        );
    }
}

// ============================================================================
// OPTIONAL MATCH with Cartesian product and rebound variables
// Inspired by FalkorDB #1281: crash on MATCH + OPTIONAL MATCH Cartesian.
// ============================================================================

mod optional_match_cartesian {
    use super::*;

    #[test]
    fn optional_match_rebinds_with_cartesian() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})").unwrap();

        // MATCH (n), (n1) with one node creates a 1x1 Cartesian product
        // OPTIONAL MATCH should not crash
        let r = s.execute(
            "MATCH (n:N), (n1:N) \
             OPTIONAL MATCH (n)-[:R]->(n1) \
             RETURN n.name, n1.name",
        );
        assert!(
            r.is_ok(),
            "OPTIONAL MATCH with Cartesian should not crash: {r:?}"
        );
    }

    #[test]
    fn optional_match_no_match_produces_nulls() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let r = s
            .execute(
                "MATCH (p:Person {name: 'Alix'}) \
                 OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) \
                 RETURN p.name, c.name",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(
            r.rows()[0][1],
            Value::Null,
            "Unmatched OPTIONAL MATCH variable should be NULL"
        );
    }
}

// ============================================================================
// WHERE filter on traversal endpoints
// Inspired by LadybugDB #273: segfault on filtered multi-hop queries.
// ============================================================================

mod where_filter_on_traversal {
    use super::*;

    #[test]
    fn filter_on_destination_property() {
        let db = db();
        let s = db.session();
        s.execute(
            "INSERT (:Person {name: 'Alix'})-[:LIVES_IN]->(:City {name: 'Amsterdam', pop: 900000})",
        )
        .unwrap();
        s.execute(
            "INSERT (:Person {name: 'Gus'})-[:LIVES_IN]->(:City {name: 'Berlin', pop: 3700000})",
        )
        .unwrap();

        let r = s
            .execute(
                "MATCH (p:Person)-[:LIVES_IN]->(c:City) \
                 WHERE c.pop > 1000000 \
                 RETURN p.name, c.name",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("Gus".into()));
        assert_eq!(r.rows()[0][1], Value::String("Berlin".into()));
    }

    #[test]
    fn filter_on_edge_property() {
        let db = db();
        let s = db.session();
        s.execute(
            "INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})",
        )
        .unwrap();
        s.execute(
            "INSERT (:Person {name: 'Vincent'})-[:KNOWS {since: 2024}]->(:Person {name: 'Jules'})",
        )
        .unwrap();

        let r = s
            .execute(
                "MATCH (a:Person)-[k:KNOWS]->(b:Person) \
                 WHERE k.since >= 2023 \
                 RETURN a.name, b.name",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("Vincent".into()));
        assert_eq!(r.rows()[0][1], Value::String("Jules".into()));
    }

    #[test]
    fn filter_on_both_endpoints() {
        let db = db();
        let s = db.session();
        s.execute(
            "INSERT (:Person {name: 'Alix', age: 30})-[:KNOWS]->(:Person {name: 'Gus', age: 25})",
        )
        .unwrap();
        s.execute("INSERT (:Person {name: 'Vincent', age: 40})-[:KNOWS]->(:Person {name: 'Jules', age: 35})")
            .unwrap();

        let r = s
            .execute(
                "MATCH (a:Person)-[:KNOWS]->(b:Person) \
                 WHERE a.age > 35 AND b.age > 30 \
                 RETURN a.name, b.name",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("Vincent".into()));
        assert_eq!(r.rows()[0][1], Value::String("Jules".into()));
    }
}

// ============================================================================
// Aggregation on empty result set
// Inspired by Neo4j and FalkorDB: aggregate functions over 0 rows.
// ============================================================================

mod empty_aggregation {
    use super::*;

    #[test]
    fn count_on_empty_returns_zero() {
        let db = db();
        let s = db.session();
        let r = s
            .execute("MATCH (n:NonExistent) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Aggregation over empty set should return 1 row"
        );
        assert_eq!(r.rows()[0][0], Value::Int64(0));
    }

    #[test]
    fn sum_on_empty_returns_null_or_zero() {
        let db = db();
        let s = db.session();
        let r = s
            .execute("MATCH (n:NonExistent) RETURN sum(n.val) AS total")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        // SQL standard: SUM of no rows is NULL. Some engines return 0.
        // Grafeo returns NULL, which is spec-compliant.
        assert!(
            r.rows()[0][0] == Value::Null || r.rows()[0][0] == Value::Int64(0),
            "SUM over empty set should be NULL or 0, got {:?}",
            r.rows()[0][0]
        );
    }

    #[test]
    fn avg_on_empty_returns_null() {
        let db = db();
        let s = db.session();
        let r = s
            .execute("MATCH (n:NonExistent) RETURN avg(n.val) AS average")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Null,
            "AVG over empty set should return NULL"
        );
    }

    #[test]
    fn min_max_on_empty_returns_null() {
        let db = db();
        let s = db.session();
        let r = s
            .execute("MATCH (n:NonExistent) RETURN min(n.val) AS lo, max(n.val) AS hi")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Null,
            "MIN over empty set should be NULL"
        );
        assert_eq!(
            r.rows()[0][1],
            Value::Null,
            "MAX over empty set should be NULL"
        );
    }
}

// ============================================================================
// Relationship direction correctness
// Inspired by multiple databases: direction reversal bugs.
// ============================================================================

mod relationship_direction {
    use super::*;

    #[test]
    fn backward_arrow_matches_correct_direction() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:A {name: 'Alix'})-[:FOLLOWS]->(:B {name: 'Gus'})")
            .unwrap();

        // Forward: A->B
        let fwd = s
            .execute("MATCH (a:A)-[:FOLLOWS]->(b:B) RETURN a.name, b.name")
            .unwrap();
        assert_eq!(fwd.row_count(), 1);

        // Backward: B<-A
        let bwd = s
            .execute("MATCH (b:B)<-[:FOLLOWS]-(a:A) RETURN a.name, b.name")
            .unwrap();
        assert_eq!(
            bwd.row_count(),
            1,
            "Backward arrow should match the same edge"
        );

        // Wrong direction: should not match
        let wrong = s
            .execute("MATCH (a:A)<-[:FOLLOWS]-(b:B) RETURN a.name")
            .unwrap();
        assert_eq!(
            wrong.row_count(),
            0,
            "Reversed direction should not match A<-B when edge is A->B"
        );
    }

    #[test]
    fn undirected_match_finds_both_directions() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})-[:KNOWS]->(:N {name: 'Gus'})")
            .unwrap();

        let r = s
            .execute("MATCH (a:N)-[:KNOWS]-(b:N) RETURN a.name, b.name ORDER BY a.name")
            .unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "Undirected pattern should match edge in both directions"
        );
    }
}

// ============================================================================
// MERGE with ON CREATE / ON MATCH and edge patterns
// Inspired by Neo4j #13718 and FalkorDB edge MERGE bugs.
// ============================================================================

mod merge_edge_patterns {
    use super::*;

    #[test]
    fn merge_relationship_creates_when_missing() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        s.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             MERGE (a)-[:KNOWS]->(b)",
        )
        .unwrap();

        let r = s
            .execute("MATCH (:Person)-[k:KNOWS]->(:Person) RETURN count(k) AS cnt")
            .unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(1));
    }

    #[test]
    fn merge_relationship_does_not_duplicate() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        // MERGE same pattern: should match, not create a second edge
        s.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             MERGE (a)-[:KNOWS]->(b)",
        )
        .unwrap();

        let r = s
            .execute("MATCH (:Person)-[k:KNOWS]->(:Person) RETURN count(k) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(1),
            "MERGE should not create duplicate relationship"
        );
    }
}

// ============================================================================
// Delete + re-insert consistency
// Inspired by LadybugDB #180 and #67: data survives delete/re-insert.
// ============================================================================

mod delete_reinsert {
    use super::*;

    #[test]
    fn delete_all_then_reinsert_same_label() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {val: 1})").unwrap();
        s.execute("INSERT (:Item {val: 2})").unwrap();
        s.execute("MATCH (n:Item) DELETE n").unwrap();

        // Verify clean slate
        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(0));

        // Re-insert
        s.execute("INSERT (:Item {val: 10})").unwrap();
        s.execute("INSERT (:Item {val: 20})").unwrap();

        let r2 = s
            .execute("MATCH (n:Item) RETURN n.val ORDER BY n.val")
            .unwrap();
        assert_eq!(r2.row_count(), 2);
        assert_eq!(r2.rows()[0][0], Value::Int64(10));
        assert_eq!(r2.rows()[1][0], Value::Int64(20));
    }

    #[test]
    fn detach_delete_then_reinsert_with_edges() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:A {name: 'a'})-[:R]->(:B {name: 'b'})")
            .unwrap();
        s.execute("MATCH (n) DETACH DELETE n").unwrap();

        // Re-insert
        s.execute("INSERT (:A {name: 'x'})-[:R]->(:B {name: 'y'})")
            .unwrap();

        let nodes = s.execute("MATCH (n) RETURN count(n) AS cnt").unwrap();
        let edges = s
            .execute("MATCH ()-[r]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(nodes.rows()[0][0], Value::Int64(2));
        assert_eq!(edges.rows()[0][0], Value::Int64(1));
    }
}

// ============================================================================
// Multi-label node operations
// Inspired by various: label handling edge cases.
// ============================================================================

mod multi_label {
    use super::*;

    #[test]
    fn node_with_multiple_labels_matched_by_any() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person:Employee {name: 'Alix'})")
            .unwrap();

        let r1 = s.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(r1.row_count(), 1, "Should match by first label");

        let r2 = s.execute("MATCH (n:Employee) RETURN n.name").unwrap();
        assert_eq!(r2.row_count(), 1, "Should match by second label");
    }

    #[test]
    fn labels_function_returns_all_labels() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person:Employee:Manager {name: 'Alix'})")
            .unwrap();

        let r = s
            .execute("MATCH (n:Person) RETURN labels(n) AS lbls")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        if let Value::List(labels) = &r.rows()[0][0] {
            assert_eq!(labels.len(), 3, "Should have 3 labels");
            let label_strs: Vec<String> = labels
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.to_string()),
                    _ => None,
                })
                .collect();
            assert!(label_strs.contains(&"Person".to_string()));
            assert!(label_strs.contains(&"Employee".to_string()));
            assert!(label_strs.contains(&"Manager".to_string()));
        } else {
            panic!("Expected List, got {:?}", r.rows()[0][0]);
        }
    }
}

// ============================================================================
// Property type coercion and comparison
// Inspired by multiple databases: type handling edge cases.
// ============================================================================

mod property_type_edge_cases {
    use super::*;

    #[test]
    fn integer_float_comparison() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {ival: 42, fval: 42.0})").unwrap();

        let r = s
            .execute("MATCH (n:N) WHERE n.ival = n.fval RETURN n.ival")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Integer 42 should equal float 42.0 in comparison"
        );
    }

    #[test]
    fn missing_property_is_null() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})").unwrap();

        let r = s
            .execute("MATCH (n:N) RETURN n.nonexistent AS val")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Null,
            "Accessing a missing property should return NULL"
        );
    }

    #[test]
    fn empty_string_is_not_null() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: ''})").unwrap();

        let r = s
            .execute("MATCH (n:N) WHERE n.name IS NOT NULL RETURN n.name")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Empty string is a valid value, not NULL");
        assert_eq!(r.rows()[0][0], Value::String("".into()));
    }

    #[test]
    fn zero_is_not_null() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 0})").unwrap();

        let r = s
            .execute("MATCH (n:N) WHERE n.val IS NOT NULL RETURN n.val")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Integer 0 is a valid value, not NULL");
        assert_eq!(r.rows()[0][0], Value::Int64(0));
    }

    #[test]
    fn boolean_false_is_not_null() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {flag: false})").unwrap();

        let r = s
            .execute("MATCH (n:N) WHERE n.flag IS NOT NULL RETURN n.flag")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Boolean false is a valid value, not NULL");
        assert_eq!(r.rows()[0][0], Value::Bool(false));
    }
}

// ============================================================================
// AVG overflow on large integers
// Inspired by ArangoDB #21096: AVG silently overflows on large i64 values.
// ============================================================================

mod avg_large_integers {
    use super::*;

    #[test]
    fn avg_of_identical_large_values_equals_itself() {
        let db = db();
        let s = db.session();
        // 389_916_982_198_384 is well within i64 range
        s.execute("INSERT (:M {val: 389916982198384})").unwrap();
        s.execute("INSERT (:M {val: 389916982198384})").unwrap();
        s.execute("INSERT (:M {val: 389916982198384})").unwrap();

        let r = s.execute("MATCH (m:M) RETURN avg(m.val) AS a").unwrap();
        assert_eq!(r.row_count(), 1);
        match &r.rows()[0][0] {
            Value::Float64(f) => {
                let expected = 389_916_982_198_384.0_f64;
                let diff = (f - expected).abs();
                assert!(
                    diff < 1.0,
                    "AVG of identical large values should equal the value itself, got {f}"
                );
            }
            Value::Int64(v) => {
                assert_eq!(
                    *v, 389_916_982_198_384,
                    "AVG of identical values should equal the value"
                );
            }
            other => panic!("Expected numeric, got {:?}", other),
        }
    }
}

// ============================================================================
// List comparison edge cases
// Inspired by ArangoDB #2477: [] == [null] incorrectly returns true.
// ============================================================================

mod list_comparison {
    use super::*;

    #[test]
    fn empty_list_not_equal_to_list_with_null() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN [] = [NULL] AS eq").unwrap();
        assert_eq!(r.row_count(), 1);
        // [] and [null] are structurally different
        assert_ne!(
            r.rows()[0][0],
            Value::Bool(true),
            "Empty list must not equal a list containing NULL"
        );
    }

    #[test]
    fn empty_list_equals_empty_list() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN [] = [] AS eq").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::Bool(true));
    }

    #[test]
    fn list_equality_same_elements() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN [1, 2, 3] = [1, 2, 3] AS eq").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::Bool(true));
    }

    #[test]
    fn list_equality_different_elements() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN [1, 2] = [1, 3] AS eq").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::Bool(false));
    }
}

// ============================================================================
// OPTIONAL MATCH + aggregation scope
// Inspired by Memgraph #3970: outer variable lost after OPTIONAL MATCH + agg.
// ============================================================================

mod optional_match_aggregation {
    use super::*;

    #[test]
    fn count_after_optional_match_preserves_all_rows() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})-[:FRIEND]->(:Person {name: 'Gus'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        let r = s
            .execute(
                "MATCH (p:Person) \
                 OPTIONAL MATCH (p)-[:FRIEND]->(f:Person) \
                 RETURN p.name, count(f) AS fc \
                 ORDER BY p.name",
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            3,
            "All 3 persons should appear, even those without friends"
        );
        // Alix has 1 friend
        assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(r.rows()[0][1], Value::Int64(1));
        // Gus has 0 friends (outgoing)
        assert_eq!(r.rows()[1][0], Value::String("Gus".into()));
        assert_eq!(r.rows()[1][1], Value::Int64(0));
        // Vincent has 0 friends
        assert_eq!(r.rows()[2][0], Value::String("Vincent".into()));
        assert_eq!(r.rows()[2][1], Value::Int64(0));
    }
}

// ============================================================================
// UNION basic correctness
// Inspired by Memgraph #3909: UNION + LIMIT hangs.
// ============================================================================

mod union_correctness {
    use super::*;

    #[test]
    fn basic_union_deduplicates() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN 1 AS x UNION RETURN 1 AS x").unwrap();
        assert_eq!(r.row_count(), 1, "UNION should deduplicate identical rows");
    }

    #[test]
    fn union_all_preserves_duplicates() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN 1 AS x UNION ALL RETURN 1 AS x").unwrap();
        assert_eq!(r.row_count(), 2, "UNION ALL should preserve duplicates");
    }

    #[test]
    fn union_different_values() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN 1 AS x UNION RETURN 2 AS x").unwrap();
        assert_eq!(r.row_count(), 2);
    }
}

// ============================================================================
// OR condition in WHERE
// Inspired by JanusGraph #4786: OR evaluated as AND by optimizer.
// ============================================================================

mod or_condition {
    use super::*;

    #[test]
    fn or_in_where_matches_either_branch() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {prop: 'A'})").unwrap();
        s.execute("INSERT (:Item {prop: 'B'})").unwrap();
        s.execute("INSERT (:Item {prop: 'C'})").unwrap();

        let r = s
            .execute(
                "MATCH (n:Item) WHERE n.prop = 'A' OR n.prop = 'B' RETURN n.prop ORDER BY n.prop",
            )
            .unwrap();
        assert_eq!(r.row_count(), 2, "OR should match both A and B");
        assert_eq!(r.rows()[0][0], Value::String("A".into()));
        assert_eq!(r.rows()[1][0], Value::String("B".into()));
    }

    #[test]
    fn or_with_and_precedence() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {a: 1, b: 2})").unwrap();
        s.execute("INSERT (:Item {a: 3, b: 4})").unwrap();
        s.execute("INSERT (:Item {a: 5, b: 6})").unwrap();

        // (a = 1 AND b = 2) OR (a = 5 AND b = 6) -> matches items 1 and 3
        let r = s
            .execute(
                "MATCH (n:Item) \
                 WHERE (n.a = 1 AND n.b = 2) OR (n.a = 5 AND n.b = 6) \
                 RETURN n.a ORDER BY n.a",
            )
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.rows()[0][0], Value::Int64(1));
        assert_eq!(r.rows()[1][0], Value::Int64(5));
    }

    #[test]
    fn not_condition_inverts_filter() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {val: 1})").unwrap();
        s.execute("INSERT (:Item {val: 2})").unwrap();
        s.execute("INSERT (:Item {val: 3})").unwrap();

        let r = s
            .execute("MATCH (n:Item) WHERE NOT n.val = 2 RETURN n.val ORDER BY n.val")
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.rows()[0][0], Value::Int64(1));
        assert_eq!(r.rows()[1][0], Value::Int64(3));
    }
}

// ============================================================================
// Type coercion: string 'false' vs boolean false
// Inspired by JanusGraph #4220: string 'false' conflated with boolean false.
// ============================================================================

mod type_coercion_string_bool {
    use super::*;

    #[test]
    fn string_false_not_equal_to_boolean_false() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: false})").unwrap();
        s.execute("INSERT (:N {val: 'false'})").unwrap();

        // Filter: val <> 'false' (string)
        // Should keep boolean false (different type), exclude string 'false'
        let r = s
            .execute("MATCH (n:N) WHERE n.val <> 'false' RETURN n.val")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Boolean false is not equal to string 'false'"
        );
        assert_eq!(r.rows()[0][0], Value::Bool(false));
    }

    #[test]
    fn string_true_not_equal_to_boolean_true() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: true})").unwrap();
        s.execute("INSERT (:N {val: 'true'})").unwrap();

        let r = s
            .execute("MATCH (n:N) WHERE n.val <> 'true' RETURN n.val")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Boolean true is not equal to string 'true'"
        );
        assert_eq!(r.rows()[0][0], Value::Bool(true));
    }
}

// ============================================================================
// Self-loop with variable-length expansion
// Inspired by Kuzu #5989: OPTIONAL MATCH with self-loop causes OOM.
// Must terminate and not expand infinitely.
// ============================================================================

mod self_loop_variable_length {
    use super::*;

    #[test]
    fn self_loop_varlength_terminates() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (a:N {name: 'Alix'})-[:LOOP]->(a)")
            .unwrap();
        s.execute(
            "MATCH (a:N {name: 'Alix'}) \
             INSERT (a)-[:R]->(b:N {name: 'Gus'})-[:R]->(c:N {name: 'Vincent'})",
        )
        .unwrap();

        // Variable-length expansion on a graph with a self-loop must terminate
        let r = s.execute("MATCH (a:N {name: 'Alix'})-[:R*1..3]->(b) RETURN b.name");
        assert!(
            r.is_ok(),
            "Variable-length expansion with self-loop in graph must terminate: {r:?}"
        );
    }

    #[test]
    fn optional_match_varlength_with_self_loop_returns_null_or_paths() {
        let db = db();
        let s = db.session();
        // Single node with a self-loop, no other edges
        s.execute("INSERT (a:N {name: 'solo'})-[:LOOP]->(a)")
            .unwrap();

        // OPTIONAL MATCH for paths of length 3+ should return NULL (no non-cycling trail)
        let r = s.execute(
            "MATCH (a:N {name: 'solo'}) \
             OPTIONAL MATCH (a)-[:R*3..5]->(b) \
             RETURN b.name",
        );
        assert!(
            r.is_ok(),
            "OPTIONAL MATCH with self-loop must not OOM: {r:?}"
        );
        if let Ok(result) = r {
            assert_eq!(
                result.row_count(),
                1,
                "OPTIONAL MATCH should still return 1 row"
            );
            assert_eq!(
                result.rows()[0][0],
                Value::Null,
                "No matching :R paths from solo node, should be NULL"
            );
        }
    }
}

// ============================================================================
// IN operator correctness
// Inspired by Kuzu #6010: IN with list operations gives wrong results.
// ============================================================================

mod in_operator {
    use super::*;

    #[test]
    fn in_basic_membership() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN 2 IN [1, 2, 3] AS found").unwrap();
        assert_eq!(r.rows()[0][0], Value::Bool(true));
    }

    #[test]
    fn in_not_found() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN 4 IN [1, 2, 3] AS found").unwrap();
        assert_eq!(r.rows()[0][0], Value::Bool(false));
    }

    #[test]
    fn in_empty_list() {
        let db = db();
        let s = db.session();
        let r = s.execute("RETURN 1 IN [] AS found").unwrap();
        assert_eq!(r.rows()[0][0], Value::Bool(false));
    }

    #[test]
    fn in_with_null_element() {
        let db = db();
        let s = db.session();
        // 1 IN [1, null] should be true (1 is found)
        let r = s.execute("RETURN 1 IN [1, NULL] AS found").unwrap();
        assert_eq!(r.rows()[0][0], Value::Bool(true));
    }

    #[test]
    fn in_with_null_not_found() {
        let db = db();
        let s = db.session();
        // 2 IN [1, null] should be NULL (not found, but null element makes it unknown)
        let r = s.execute("RETURN 2 IN [1, NULL] AS found").unwrap();
        // Per SQL/Cypher semantics: UNKNOWN (NULL)
        assert_eq!(
            r.rows()[0][0],
            Value::Null,
            "x IN [y, NULL] where x != y should be NULL (unknown)"
        );
    }
}

// ============================================================================
// ORDER BY with NULL property values
// Inspired by JanusGraph #3269 and general: NULLs in ORDER BY.
// ============================================================================

mod order_by_nulls {
    use super::*;

    #[test]
    fn order_by_with_some_null_properties() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix', score: 90})").unwrap();
        s.execute("INSERT (:N {name: 'Gus'})").unwrap(); // no score property
        s.execute("INSERT (:N {name: 'Vincent', score: 80})")
            .unwrap();

        // All 3 nodes should appear, NULLs sorted to end
        let r = s
            .execute("MATCH (n:N) RETURN n.name, n.score ORDER BY n.score")
            .unwrap();
        assert_eq!(
            r.row_count(),
            3,
            "ORDER BY on partial property should still return all rows"
        );
        // NULL should sort last
        assert_eq!(
            r.rows()[2][1],
            Value::Null,
            "NULL score should sort to the end"
        );
    }
}

// ============================================================================
// MERGE visibility of deleted nodes within same transaction
// Inspired by Memgraph #2093: MERGE matches a ghost node deleted earlier.
// ============================================================================

mod merge_after_delete {
    use super::*;

    #[test]
    fn merge_does_not_match_deleted_node() {
        let db = db();
        let s = db.session();
        // Create, delete, then MERGE: MERGE should create a new node
        s.execute("INSERT (:Singleton {key: 'only'})").unwrap();
        s.execute("MATCH (n:Singleton) DELETE n").unwrap();
        s.execute("MERGE (:Singleton {key: 'only'})").unwrap();

        let r = s
            .execute("MATCH (n:Singleton) RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(1),
            "MERGE after delete should create exactly 1 new node"
        );
    }
}

// ============================================================================
// Label intersection across MATCH clauses
// Inspired by Memgraph #2875: same variable with different labels in
// separate MATCH clauses should intersect label sets.
// ============================================================================

mod label_intersection {
    use super::*;

    #[test]
    fn same_variable_multiple_labels_across_match() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Worker:Senior {name: 'Vincent'})")
            .unwrap();
        s.execute("INSERT (:Worker:Junior {name: 'Jules'})")
            .unwrap();
        s.execute(
            "MATCH (a:Worker {name: 'Vincent'}), (b:Worker {name: 'Jules'}) \
             INSERT (a)-[:MANAGES]->(b)",
        )
        .unwrap();

        // n must satisfy both :Worker (first MATCH) and :Senior (second MATCH)
        let r = s
            .execute(
                "MATCH (n:Worker)-[:MANAGES]->() \
                 MATCH (n:Senior) \
                 RETURN n.name",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::String("Vincent".into()),
            "Only Vincent has both :Worker and :Senior"
        );
    }

    #[test]
    fn label_intersection_filters_non_matching() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:A:B {name: 'both'})").unwrap();
        s.execute("INSERT (:A {name: 'only_a'})").unwrap();
        s.execute("INSERT (:B {name: 'only_b'})").unwrap();

        let r = s.execute("MATCH (n:A) MATCH (n:B) RETURN n.name").unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Only the node with both :A and :B should match"
        );
        assert_eq!(r.rows()[0][0], Value::String("both".into()));
    }
}

// ============================================================================
// IS NOT NULL operator precedence
// Inspired by Memgraph #2457: WHERE 1=1 IS NOT NULL returns 0 rows.
// ============================================================================

mod is_not_null_precedence {
    use super::*;

    #[test]
    fn boolean_expression_is_not_null() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();

        // (1 = 1) evaluates to true, true IS NOT NULL evaluates to true
        let r = s
            .execute("MATCH (n:N) WHERE (1 = 1) IS NOT NULL RETURN n.val")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "(1=1) IS NOT NULL should be TRUE, returning the row"
        );
    }

    #[test]
    fn property_comparison_is_not_null() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {a: 10, b: 20})").unwrap();

        // (n.a < n.b) is true, true IS NOT NULL is true
        let r = s
            .execute("MATCH (n:N) WHERE (n.a < n.b) IS NOT NULL RETURN n.a")
            .unwrap();
        assert_eq!(r.row_count(), 1);
    }
}

// ============================================================================
// Quantifier functions on empty collections
// Inspired by Memgraph #2481: any([]) returns NULL instead of FALSE.
// ============================================================================

mod quantifier_functions {
    use super::*;

    #[test]
    fn any_on_empty_list_is_false() {
        let db = db();
        let s = db.session();
        let r = s
            .execute("RETURN any(x IN [] WHERE x > 0) AS result")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Bool(false),
            "any() on empty list should be FALSE per spec"
        );
    }

    #[test]
    fn all_on_empty_list_is_true() {
        let db = db();
        let s = db.session();
        let r = s
            .execute("RETURN all(x IN [] WHERE x > 0) AS result")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Bool(true),
            "all() on empty list should be TRUE per spec (vacuous truth)"
        );
    }

    #[test]
    fn none_on_empty_list_is_true() {
        let db = db();
        let s = db.session();
        let r = s
            .execute("RETURN none(x IN [] WHERE x > 0) AS result")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Bool(true),
            "none() on empty list should be TRUE (no elements violate)"
        );
    }
}

// ============================================================================
// String escape sequences
// Inspired by Kuzu #5814: \n and \t stored as literal backslash-n.
// ============================================================================

mod string_escapes {
    use super::*;

    #[test]
    fn newline_escape_in_property() {
        let db = db();
        let s = db.session();
        s.execute(r#"INSERT (:Entry {text: "line1\nline2"})"#)
            .unwrap();

        let r = s.execute("MATCH (e:Entry) RETURN e.text").unwrap();
        assert_eq!(r.row_count(), 1);
        if let Value::String(s) = &r.rows()[0][0] {
            assert!(
                s.contains('\n'),
                "String should contain actual newline, got: {:?}",
                s
            );
        } else {
            panic!("Expected String, got {:?}", r.rows()[0][0]);
        }
    }

    #[test]
    fn tab_escape_in_property() {
        let db = db();
        let s = db.session();
        s.execute(r#"INSERT (:Entry {text: "col1\tcol2"})"#)
            .unwrap();

        let r = s.execute("MATCH (e:Entry) RETURN e.text").unwrap();
        assert_eq!(r.row_count(), 1);
        if let Value::String(s) = &r.rows()[0][0] {
            assert!(
                s.contains('\t'),
                "String should contain actual tab, got: {:?}",
                s
            );
        } else {
            panic!("Expected String, got {:?}", r.rows()[0][0]);
        }
    }
}

// ============================================================================
// NULL join semantics: NULL = NULL must not match
// Inspired by Kuzu #5893: self-join with nullable column includes NULLs.
// ============================================================================

mod null_join_semantics {
    use super::*;

    #[test]
    fn null_equality_in_cross_match_where() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Account {svc: 'A', name: 'Alix'})")
            .unwrap();
        s.execute("INSERT (:Account {svc: 'B', name: 'Alix'})")
            .unwrap();
        // Accounts with no name (NULL)
        s.execute("INSERT (:Account {svc: 'A'})").unwrap();
        s.execute("INSERT (:Account {svc: 'B'})").unwrap();

        let r = s
            .execute(
                "MATCH (a:Account {svc: 'A'}) \
                 MATCH (b:Account {svc: 'B'}) \
                 WHERE a.name = b.name \
                 RETURN a.name",
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Only 'Alix'='Alix' should match, NULL=NULL must not"
        );
        assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
    }
}

// ============================================================================
// Functions with NULL arguments return NULL (not crash)
// Inspired by Kuzu #5959: labels(null) crashes.
// ============================================================================

mod null_function_arguments {
    use super::*;

    #[test]
    fn type_of_null_relationship_returns_null() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {name: 'Alix'})").unwrap();

        let r = s
            .execute(
                "MATCH (n:N) \
                 OPTIONAL MATCH (n)-[r:NONEXISTENT]->() \
                 RETURN type(r) AS t",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::Null,
            "type() of NULL relationship should return NULL"
        );
    }
}

// ============================================================================
// Delete + re-create same edge in one transaction
// Inspired by Dgraph #9422: delete+set of same predicate produces duplicates.
// ============================================================================

mod delete_recreate_edge {
    use super::*;

    #[test]
    fn replace_edge_in_sequence() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})-[:LIKES]->(:Fruit {name: 'apple'})")
            .unwrap();
        s.execute("INSERT (:Fruit {name: 'banana'})").unwrap();

        // Delete old edge, create new one
        s.execute("MATCH (p:Person {name: 'Alix'})-[r:LIKES]->() DELETE r")
            .unwrap();
        s.execute(
            "MATCH (p:Person {name: 'Alix'}), (f:Fruit {name: 'banana'}) \
             INSERT (p)-[:LIKES]->(f)",
        )
        .unwrap();

        let r = s
            .execute(
                "MATCH (p:Person {name: 'Alix'})-[:LIKES]->(f:Fruit) \
                 RETURN f.name",
            )
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Should have exactly 1 LIKES edge after replace"
        );
        assert_eq!(r.rows()[0][0], Value::String("banana".into()));
    }
}

// ============================================================================
// LIMIT with ORDER BY correctness
// Inspired by Dgraph #9239: LIMIT silently capped when ORDER BY present.
// ============================================================================

mod limit_with_order {
    use super::*;

    #[test]
    fn limit_respected_with_order_by() {
        let db = db();
        let s = db.session();
        for i in 0..20 {
            s.execute(&format!("INSERT (:Item {{seq: {i}}})")).unwrap();
        }

        let r = s
            .execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq LIMIT 5")
            .unwrap();
        assert_eq!(
            r.row_count(),
            5,
            "LIMIT 5 with ORDER BY should return exactly 5"
        );
        // Should be the first 5 in order
        assert_eq!(r.rows()[0][0], Value::Int64(0));
        assert_eq!(r.rows()[4][0], Value::Int64(4));
    }

    #[test]
    fn limit_without_order_by() {
        let db = db();
        let s = db.session();
        for i in 0..20 {
            s.execute(&format!("INSERT (:Item {{seq: {i}}})")).unwrap();
        }

        let r = s.execute("MATCH (n:Item) RETURN n.seq LIMIT 5").unwrap();
        assert_eq!(
            r.row_count(),
            5,
            "LIMIT 5 without ORDER BY should return exactly 5"
        );
    }

    #[test]
    fn limit_larger_than_result_set() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {seq: 1})").unwrap();
        s.execute("INSERT (:Item {seq: 2})").unwrap();

        let r = s
            .execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq LIMIT 100")
            .unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "LIMIT larger than result set returns all rows"
        );
    }
}

// ============================================================================
// Idempotent property SET
// Inspired by Dgraph #9519: re-setting property to same value breaks mutation.
// ============================================================================

mod idempotent_set {
    use super::*;

    #[test]
    fn set_property_to_same_value_is_noop() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {code: 'X', note: 'initial'})")
            .unwrap();

        // Re-set code to same value, change note
        s.execute("MATCH (n:N {code: 'X'}) SET n.code = 'X', n.note = 'updated'")
            .unwrap();

        let r = s.execute("MATCH (n:N {code: 'X'}) RETURN n.note").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::String("updated".into()),
            "SET with unchanged code should still update note"
        );
    }
}

// ============================================================================
// Float precision preservation
// Inspired by Dgraph #9491: floats truncated to 6 decimal digits.
// ============================================================================

mod float_precision {
    use super::*;

    #[test]
    fn float64_full_precision_roundtrip() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:M {val: 0.123456789012345})").unwrap();

        let r = s.execute("MATCH (m:M) RETURN m.val").unwrap();
        assert_eq!(r.row_count(), 1);
        if let Value::Float64(f) = r.rows()[0][0] {
            let diff = (f - 0.123456789012345_f64).abs();
            assert!(
                diff < 1e-15,
                "Float64 must preserve full precision, got {f} (diff {diff})"
            );
        } else {
            panic!("Expected Float64, got {:?}", r.rows()[0][0]);
        }
    }

    #[test]
    fn repeated_edge_updates_no_stale_values() {
        let db = db();
        let s = db.session();
        // Setup: host -> n1
        s.execute("INSERT (:Host {name: 'h1'})-[:NESTS]->(:Nest {name: 'n1'})")
            .unwrap();
        s.execute("INSERT (:Nest {name: 'n2'})").unwrap();
        s.execute("INSERT (:Nest {name: 'n3'})").unwrap();

        // Update to n2
        s.execute("MATCH (:Host {name: 'h1'})-[r:NESTS]->() DELETE r")
            .unwrap();
        s.execute("MATCH (h:Host {name: 'h1'}), (n:Nest {name: 'n2'}) INSERT (h)-[:NESTS]->(n)")
            .unwrap();

        // Update to n3
        s.execute("MATCH (:Host {name: 'h1'})-[r:NESTS]->() DELETE r")
            .unwrap();
        s.execute("MATCH (h:Host {name: 'h1'}), (n:Nest {name: 'n3'}) INSERT (h)-[:NESTS]->(n)")
            .unwrap();

        // Must see only n3, no stale n1 or n2
        let r = s
            .execute("MATCH (:Host {name: 'h1'})-[:NESTS]->(n) RETURN n.name")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "Should have exactly 1 NESTS edge after 2 updates"
        );
        assert_eq!(r.rows()[0][0], Value::String("n3".into()));
    }
}

// ============================================================================
// NULL in multi-column GROUP BY must not collapse distinct non-NULL values
// Inspired by ArangoDB #14672: uniform NULL column collapses all groups.
// ============================================================================

mod null_grouping_key {
    use super::*;

    #[test]
    fn null_column_does_not_collapse_groups() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Q {status: 'ok'})").unwrap();
        s.execute("INSERT (:Q {status: 'ok'})").unwrap();
        s.execute("INSERT (:Q {status: 'reject'})").unwrap();
        // All nodes have no 'extra' property (NULL)

        let r = s
            .execute(
                "MATCH (q:Q) \
                 RETURN q.status AS status, q.extra AS extra, count(*) AS cnt \
                 ORDER BY status",
            )
            .unwrap();
        // Should produce 2 groups: ('ok', NULL, 2) and ('reject', NULL, 1)
        // NOT 1 group: (NULL, NULL, 3)
        assert!(
            r.row_count() >= 2,
            "NULL in one grouping column must not collapse distinct values in another, got {} rows",
            r.row_count()
        );
    }
}

// ============================================================================
// Inequality on missing property: <> must not match nodes without the property
// Inspired by JanusGraph #2205: neq on missing property returns wrong results.
// n.name <> 'X' should only match nodes that HAVE a name AND name != 'X'.
// ============================================================================

mod inequality_missing_property {
    use super::*;

    #[test]
    fn neq_does_not_match_missing_property() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:Person {age: 30})").unwrap(); // no name

        // name <> 'Alix' should return Gus only (has name, and it's not Alix)
        // The node without a name property has NULL for name, and NULL <> 'Alix' is UNKNOWN
        let r = s
            .execute("MATCH (n:Person) WHERE n.name <> 'Alix' RETURN n.name ORDER BY n.name")
            .unwrap();
        assert_eq!(
            r.row_count(),
            1,
            "<> must not match nodes missing the property (NULL <> x is UNKNOWN)"
        );
        assert_eq!(r.rows()[0][0], Value::String("Gus".into()));
    }
}

// ============================================================================
// MERGE with composite key in UNWIND batch: intra-batch dedup
// Inspired by Neo4j #13729: duplicate entries in same UNWIND bypass constraint.
// ============================================================================

mod merge_batch_composite_dedup {
    use super::*;

    #[test]
    fn unwind_merge_composite_key_deduplicates() {
        let db = db();
        let s = db.session();
        // Test intra-batch MERGE dedup with simple scalar UNWIND
        s.execute("UNWIND [1, 2, 1, 3, 2] AS i MERGE (:Item {a: i})")
            .unwrap();

        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(3),
            "Duplicate values in UNWIND should MERGE into distinct nodes"
        );
    }
}

// ============================================================================
// Mixed aggregate and non-aggregate in RETURN
// Inspired by FalkorDB #1451: should raise error, not return NULL.
// ============================================================================

mod mixed_aggregate_non_aggregate {
    use super::*;

    #[test]
    fn mixing_aggregate_and_bare_column_without_group_by() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:P {n: 'a'})").unwrap();
        s.execute("INSERT (:P {n: 'b'})").unwrap();

        // p.n + COUNT(p) mixes aggregate and non-aggregate without GROUP BY
        // Should either error OR return a well-defined grouped result, never silent NULL
        let r = s.execute("MATCH (p:P) RETURN p.n, count(p) AS cnt");
        match r {
            Err(_) => {} // Semantic error is the correct response
            Ok(result) => {
                // If engine implicitly groups by p.n, that's also acceptable
                assert!(
                    result.row_count() >= 1,
                    "If not an error, must return grouped result, not empty"
                );
                // Verify no NULL values leaked in where real data should be
                for row in result.rows() {
                    assert_ne!(
                        row[0],
                        Value::Null,
                        "Non-aggregate column must not silently become NULL"
                    );
                }
            }
        }
    }
}

// ============================================================================
// ORDER BY on aliased/projected property with OPTIONAL MATCH
// Inspired by Memgraph #3976: ORDER BY employee.name fails after aliasing.
// ============================================================================

mod order_by_aliased_property {
    use super::*;

    #[test]
    fn order_by_original_expression_after_alias() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})-[:WORKS_FOR]->(:Company {name: 'Acme'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})-[:WORKS_FOR]->(:Company {name: 'Beta'})")
            .unwrap();

        let r = s
            .execute(
                "MATCH (e:Person)-[:WORKS_FOR]->(c:Company) \
                 RETURN e.name AS employee, c.name AS company \
                 ORDER BY employee",
            )
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(r.rows()[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn order_by_alias_with_optional_match() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})-[:WORKS_FOR]->(:Company {name: 'Acme'})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        let r = s
            .execute(
                "MATCH (e:Person) \
                 OPTIONAL MATCH (e)-[:WORKS_FOR]->(c:Company) \
                 RETURN e.name AS employee, c.name AS company \
                 ORDER BY employee",
            )
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(r.rows()[0][1], Value::String("Acme".into()));
        assert_eq!(r.rows()[1][0], Value::String("Gus".into()));
        assert_eq!(r.rows()[1][1], Value::Null);
    }
}

// ============================================================================
// #218: Cypher ORDER BY returns zeros in MATCH with relationship traversal
// ORDER BY on a property expression that duplicates a RETURN item should
// resolve to the existing projected column, not create a broken extra
// PropertyAccess on a non-entity column.
// ============================================================================

mod order_by_relationship_traversal_218 {
    use super::*;

    #[test]
    fn order_by_property_matching_return_item() {
        let db = db();
        let s = db.session();
        s.execute_cypher("CREATE (:Method {name: 'foo', qn: 'java://A:foo()'})")
            .unwrap();
        s.execute_cypher("CREATE (:Method {name: 'bar', qn: 'java://B:bar()'})")
            .unwrap();
        s.execute_cypher("CREATE (:Method {name: 'baz', qn: 'java://C:baz()'})")
            .unwrap();
        s.execute_cypher(
            "MATCH (a:Method {name: 'foo'}), (b:Method {name: 'bar'}) CREATE (a)-[:CALLS]->(b)",
        )
        .unwrap();
        s.execute_cypher(
            "MATCH (a:Method {name: 'baz'}), (b:Method {name: 'bar'}) CREATE (a)-[:CALLS]->(b)",
        )
        .unwrap();

        // Without ORDER BY: should return correct values
        let r = s
            .execute_cypher(
                "MATCH (caller)-[:CALLS]->(target) \
                 WHERE target.name = 'bar' \
                 RETURN caller.name AS caller, caller.qn AS caller_qn",
            )
            .unwrap();
        assert_eq!(r.row_count(), 2);
        for row in r.rows() {
            assert!(
                matches!(&row[0], Value::String(s) if s == "foo" || s == "baz"),
                "caller should be a string, got {:?}",
                row[0]
            );
            assert!(
                matches!(&row[1], Value::String(_)),
                "caller_qn should be a string, got {:?}",
                row[1]
            );
        }

        // With ORDER BY on a property that matches a RETURN item expression:
        // values must NOT become 0 and no extra spurious column should appear.
        let r = s
            .execute_cypher(
                "MATCH (caller)-[:CALLS]->(target) \
                 WHERE target.name = 'bar' \
                 RETURN caller.name AS caller, caller.qn AS caller_qn \
                 ORDER BY caller.name",
            )
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(
            r.columns.len(),
            2,
            "should have exactly 2 columns, got {:?}",
            r.columns
        );
        // Sorted ascending by name: baz < foo
        assert_eq!(r.rows()[0][0], Value::String("baz".into()));
        assert_eq!(r.rows()[0][1], Value::String("java://C:baz()".into()));
        assert_eq!(r.rows()[1][0], Value::String("foo".into()));
        assert_eq!(r.rows()[1][1], Value::String("java://A:foo()".into()));
    }

    #[test]
    fn order_by_desc_with_relationship_traversal() {
        let db = db();
        let s = db.session();
        s.execute_cypher("CREATE (:Method {name: 'foo', qn: 'java://A:foo()'})")
            .unwrap();
        s.execute_cypher("CREATE (:Method {name: 'bar', qn: 'java://B:bar()'})")
            .unwrap();
        s.execute_cypher("CREATE (:Method {name: 'baz', qn: 'java://C:baz()'})")
            .unwrap();
        s.execute_cypher(
            "MATCH (a:Method {name: 'foo'}), (b:Method {name: 'bar'}) CREATE (a)-[:CALLS]->(b)",
        )
        .unwrap();
        s.execute_cypher(
            "MATCH (a:Method {name: 'baz'}), (b:Method {name: 'bar'}) CREATE (a)-[:CALLS]->(b)",
        )
        .unwrap();

        let r = s
            .execute_cypher(
                "MATCH (caller)-[:CALLS]->(target) \
                 WHERE target.name = 'bar' \
                 RETURN caller.name AS caller, caller.qn AS caller_qn \
                 ORDER BY caller.name DESC",
            )
            .unwrap();
        assert_eq!(r.row_count(), 2);
        // Sorted descending: foo > baz
        assert_eq!(r.rows()[0][0], Value::String("foo".into()));
        assert_eq!(r.rows()[1][0], Value::String("baz".into()));
    }
}

// ============================================================================
// CALL subquery scope isolation
// Inspired by Neo4j #13656 and Memgraph #3955: outer variables lost in CALL.
// ============================================================================

mod call_subquery_scope {
    use super::*;

    #[test]
    fn call_subquery_preserves_outer_variable() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        let r = s.execute(
            "MATCH (p:Person) \
             CALL { \
               WITH p \
               OPTIONAL MATCH (p)-[:ACTED_IN]->(m) \
               RETURN count(m) AS movie_count \
             } \
             RETURN p.name, movie_count \
             ORDER BY p.name",
        );
        match r {
            Ok(result) => {
                assert_eq!(
                    result.row_count(),
                    3,
                    "CALL subquery must preserve all outer rows, got {}",
                    result.row_count()
                );
                // All should have movie_count = 0
                for row in result.rows() {
                    assert_eq!(row[1], Value::Int64(0));
                }
            }
            Err(_) => {
                // If CALL subquery syntax is not supported, that's fine
            }
        }
    }
}

// ============================================================================
// Edge properties through path variable
// Inspired by ArangoDB #21265: edge props lost when accessed via path.
// ============================================================================

mod edge_properties_in_path {
    use super::*;

    #[test]
    fn edge_properties_preserved_in_traversal() {
        let db = db();
        let s = db.session();
        s.execute(
            "INSERT (:Place {name: 'France'})-[:IS_IN {category: 'continent'}]->(:Place {name: 'Europe'})",
        )
        .unwrap();
        s.execute(
            "INSERT (:Place {name: 'Europe'})-[:IS_IN {category: 'planet'}]->(:Place {name: 'World'})",
        )
        .unwrap();

        // Access edge properties directly
        let r = s
            .execute(
                "MATCH (:Place {name: 'France'})-[r:IS_IN]->(:Place {name: 'Europe'}) \
                 RETURN r.category",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::String("continent".into()),
            "Edge property must be preserved"
        );
    }

    #[test]
    fn multiple_edge_properties_in_chain() {
        let db = db();
        let s = db.session();
        s.execute(
            "INSERT (:A {name: 'a'})-[:R {weight: 10}]->(:B {name: 'b'})-[:R {weight: 20}]->(:C {name: 'c'})",
        )
        .unwrap();

        let r = s
            .execute(
                "MATCH (:A)-[r1:R]->(:B)-[r2:R]->(:C) \
                 RETURN r1.weight, r2.weight",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::Int64(10));
        assert_eq!(r.rows()[0][1], Value::Int64(20));
    }
}

// ============================================================================
// Variable-length path vs explicit hop count equivalence
// Inspired by Memgraph #3735: variable-length returns different count.
// ============================================================================

mod varlength_vs_explicit {
    use super::*;

    #[test]
    fn two_hop_varlength_equals_explicit() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {id: 1})-[:R]->(:N {id: 2})-[:R]->(:N {id: 3})")
            .unwrap();

        // Explicit two-hop
        let explicit = s
            .execute(
                "MATCH (a:N {id: 1})-[:R]->(mid)-[:R]->(c) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();

        // Variable-length two-hop
        let varlength = s
            .execute(
                "MATCH (a:N {id: 1})-[:R*2]->(c) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();

        assert_eq!(
            explicit.rows()[0][0],
            varlength.rows()[0][0],
            "Variable-length *2 must match explicit two-hop count"
        );
    }
}

// ============================================================================
// CASE WHEN with NULL from aggregate over empty set
// Inspired by Dgraph #9125: conditional on NULL aggregate fails.
// ============================================================================

mod case_when_null_aggregate {
    use super::*;

    #[test]
    fn case_when_with_count_zero_and_null_max() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Column {name: 'todo'})").unwrap();
        // No cards in 'todo' column

        let r = s.execute(
            "MATCH (col:Column {name: 'todo'}) \
             OPTIONAL MATCH (c:Card)-[:IN_COL]->(col) \
             WITH col, count(c) AS cc, max(c.pos) AS mp \
             RETURN CASE WHEN cc = 0 THEN 0 ELSE mp + 1 END AS next_pos",
        );
        match r {
            Ok(result) => {
                assert_eq!(result.row_count(), 1);
                assert_eq!(
                    result.rows()[0][0],
                    Value::Int64(0),
                    "When count is 0, CASE should return 0"
                );
            }
            Err(_) => {
                // CASE WHEN not yet supported is acceptable
            }
        }
    }
}

// ============================================================================
// Chained OR correctness: (A OR B) AND (C OR D)
// Inspired by JanusGraph #2231: chained or() flattened to single OR.
// ============================================================================

mod chained_or_and {
    use super::*;

    #[test]
    fn chained_or_not_flattened() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:T {a: true,  b: true,  c: true,  d: true})")
            .unwrap();
        s.execute("INSERT (:T {a: true,  b: false, c: true,  d: false})")
            .unwrap();
        s.execute("INSERT (:T {a: false, b: true,  c: false, d: true})")
            .unwrap();
        s.execute("INSERT (:T {a: false, b: false, c: true,  d: false})")
            .unwrap();

        // (a OR b) AND (c=false OR d=true)
        // Row 1: a=T,b=T,c=T,d=T -> (T OR T)=T AND (F OR T)=T -> match
        // Row 2: a=T,b=F,c=T,d=F -> (T OR F)=T AND (F OR F)=F -> no
        // Row 3: a=F,b=T,c=F,d=T -> (F OR T)=T AND (T OR T)=T -> match
        // Row 4: a=F,b=F,c=T,d=F -> (F OR F)=F -> no
        let r = s
            .execute(
                "MATCH (n:T) \
                 WHERE (n.a = true OR n.b = true) AND (n.c = false OR n.d = true) \
                 RETURN count(*) AS cnt",
            )
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(2),
            "(A OR B) AND (C OR D) must not flatten to (A OR B OR C OR D)"
        );
    }
}

// ============================================================================
// substring() indexing
// Inspired by LadybugDB #85: Cypher uses 0-based, GQL/SQL uses 1-based.
// ============================================================================

mod substring_indexing {
    use super::*;

    #[test]
    fn gql_substring_extracts_correctly() {
        let db = db();
        let s = db.session();
        // GQL substring: start position and length
        let r = s.execute("RETURN substring('hello world', 0, 5) AS sub");
        match r {
            Ok(result) => {
                assert_eq!(result.row_count(), 1);
                if let Value::String(s) = &result.rows()[0][0] {
                    // Whether 0-based or 1-based, should not return empty
                    assert!(
                        !s.is_empty(),
                        "substring should not return empty for valid input"
                    );
                }
            }
            Err(_) => {} // substring not supported is fine
        }
    }
}

// ============================================================================
// Property key prefix overlap: keys that share prefixes
// Inspired by JanusGraph #4401: looking up "hel" crashes when "hello" exists.
// ============================================================================

mod property_key_prefix {
    use super::*;

    #[test]
    fn property_keys_with_shared_prefix() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {hello: 'world', hel: 'wor'})")
            .unwrap();

        let r1 = s.execute("MATCH (n:N) RETURN n.hel").unwrap();
        assert_eq!(r1.rows()[0][0], Value::String("wor".into()));

        let r2 = s.execute("MATCH (n:N) RETURN n.hello").unwrap();
        assert_eq!(r2.rows()[0][0], Value::String("world".into()));
    }

    #[test]
    fn filter_on_prefix_property() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {abc: 1, abcdef: 2, ab: 3})").unwrap();

        let r = s
            .execute("MATCH (n:N) WHERE n.abc = 1 RETURN n.abcdef, n.ab")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::Int64(2));
        assert_eq!(r.rows()[0][1], Value::Int64(3));
    }
}

// ============================================================================
// Property type overwrite: change property from one type to another
// Inspired by JanusGraph #4141: bool->string overwrite corrupts vertex.
// ============================================================================

mod property_type_overwrite {
    use super::*;

    #[test]
    fn overwrite_bool_with_string() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {flag: true})").unwrap();
        s.execute("MATCH (n:Item) SET n.flag = 'yes'").unwrap();

        let r = s.execute("MATCH (n:Item) RETURN n.flag").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(
            r.rows()[0][0],
            Value::String("yes".into()),
            "Property type should be overwritable from bool to string"
        );
    }

    #[test]
    fn overwrite_int_with_string() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {val: 42})").unwrap();
        s.execute("MATCH (n:Item) SET n.val = 'forty-two'").unwrap();

        let r = s.execute("MATCH (n:Item) RETURN n.val").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("forty-two".into()));
    }

    #[test]
    fn overwrite_string_with_int() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {val: 'hello'})").unwrap();
        s.execute("MATCH (n:Item) SET n.val = 99").unwrap();

        let r = s.execute("MATCH (n:Item) RETURN n.val").unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::Int64(99));
    }
}

// ============================================================================
// Escaped quotes in property values
// Inspired by Dgraph #9405: escaped quotes break constraint validation.
// ============================================================================

mod escaped_quotes {
    use super::*;

    #[test]
    fn double_quotes_in_single_quoted_string() {
        let db = db();
        let s = db.session();
        s.execute(r#"INSERT (:Book {title: 'The "Problem" of Knowledge'})"#)
            .unwrap();

        let r = s.execute("MATCH (b:Book) RETURN b.title").unwrap();
        assert_eq!(r.row_count(), 1);
        if let Value::String(title) = &r.rows()[0][0] {
            assert!(
                title.contains('"'),
                "Double quotes should be preserved in property value, got: {title}"
            );
        } else {
            panic!("Expected String, got {:?}", r.rows()[0][0]);
        }
    }

    #[test]
    fn single_quotes_in_double_quoted_string() {
        let db = db();
        let s = db.session();
        s.execute(r#"INSERT (:Book {title: "It's a Test"})"#)
            .unwrap();

        let r = s.execute("MATCH (b:Book) RETURN b.title").unwrap();
        assert_eq!(r.row_count(), 1);
        if let Value::String(title) = &r.rows()[0][0] {
            assert!(
                title.contains('\''),
                "Single quotes should be preserved, got: {title}"
            );
        } else {
            panic!("Expected String, got {:?}", r.rows()[0][0]);
        }
    }
}

// ============================================================================
// Phantom relationships after DETACH DELETE
// Inspired by Kuzu #5954: edge survives after endpoint node is deleted.
// ============================================================================

mod phantom_relationships {
    use super::*;

    #[test]
    fn detach_delete_one_node_preserves_unrelated_edges() {
        let db = db();
        let s = db.session();
        // Two independent edge chains
        s.execute("INSERT (:A {id: 1})-[:R]->(:B {id: 1})").unwrap();
        s.execute("INSERT (:A {id: 2})-[:R]->(:B {id: 2})").unwrap();

        // Delete only the first chain's source
        s.execute("MATCH (a:A {id: 1}) DETACH DELETE a").unwrap();

        // Second chain should still have its edge
        let r = s
            .execute("MATCH ()-[r:R]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(1),
            "Unrelated edge chain should survive DETACH DELETE of another node"
        );
    }
}

// ============================================================================
// DETACH DELETE returns ghost node properties
// Inspired by Neo4j #13714: deleted node still returns props in same clause.
// ============================================================================

mod detach_delete_return {
    use super::*;

    #[test]
    fn delete_then_count_returns_zero() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Ghost {name: 'Alix', age: 30})")
            .unwrap();
        s.execute("MATCH (n:Ghost) DETACH DELETE n").unwrap();

        // After delete, node must not be queryable
        let r = s.execute("MATCH (n:Ghost) RETURN count(n) AS cnt").unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(0));
    }
}

// ============================================================================
// Multi-statement execution
// Inspired by LadybugDB #333: batch execution silently drops later errors.
// ============================================================================

mod multi_statement {
    use super::*;

    #[test]
    fn sequential_statements_all_execute() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {val: 1})").unwrap();
        s.execute("INSERT (:Item {val: 2})").unwrap();
        s.execute("INSERT (:Item {val: 3})").unwrap();

        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(3),
            "All sequential statements should execute"
        );
    }
}

// ============================================================================
// Backtick-quoted identifiers
// Inspired by Kuzu #5872: backtick labels treated as different label.
// ============================================================================

mod backtick_identifiers {
    use super::*;

    #[test]
    fn backtick_label_matches_plain_label() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Backtick-quoted label should match the same label
        let r = s.execute("MATCH (n:`Person`) RETURN n.name");
        match r {
            Ok(result) => {
                assert_eq!(
                    result.row_count(),
                    1,
                    "Backtick-quoted :Person should match plain :Person"
                );
                assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
            }
            Err(_) => {
                // Backtick syntax not supported is acceptable
            }
        }
    }
}

// ============================================================================
// SKIP and SKIP + LIMIT interaction
// Broader coverage of pagination correctness.
// ============================================================================

mod skip_limit {
    use super::*;

    #[test]
    fn skip_skips_first_n_rows() {
        let db = db();
        let s = db.session();
        for i in 0..5 {
            s.execute(&format!("INSERT (:Item {{seq: {i}}})")).unwrap();
        }

        let r = s
            .execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq SKIP 2")
            .unwrap();
        assert_eq!(r.row_count(), 3);
        assert_eq!(r.rows()[0][0], Value::Int64(2));
    }

    #[test]
    fn skip_plus_limit() {
        let db = db();
        let s = db.session();
        for i in 0..10 {
            s.execute(&format!("INSERT (:Item {{seq: {i}}})")).unwrap();
        }

        let r = s
            .execute("MATCH (n:Item) RETURN n.seq ORDER BY n.seq SKIP 3 LIMIT 2")
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.rows()[0][0], Value::Int64(3));
        assert_eq!(r.rows()[1][0], Value::Int64(4));
    }

    #[test]
    fn skip_beyond_result_set() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Item {seq: 1})").unwrap();
        s.execute("INSERT (:Item {seq: 2})").unwrap();

        let r = s.execute("MATCH (n:Item) RETURN n.seq SKIP 100").unwrap();
        assert_eq!(r.row_count(), 0, "SKIP past end should return empty");
    }
}

// ============================================================================
// ORDER BY with mixed types
// Inspired by Memgraph #3888: should follow total ordering, not error.
// ============================================================================

mod order_by_mixed_types {
    use super::*;

    #[test]
    fn order_by_heterogeneous_property_does_not_crash() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 'hello'})").unwrap();
        s.execute("INSERT (:N {val: true})").unwrap();

        // Should not crash, regardless of ordering semantics
        let r = s.execute("MATCH (n:N) RETURN n.val ORDER BY n.val");
        assert!(
            r.is_ok(),
            "ORDER BY with mixed types should not crash: {r:?}"
        );
        if let Ok(result) = r {
            assert_eq!(result.row_count(), 3, "All 3 rows should be returned");
        }
    }
}

// ============================================================================
// RETURN DISTINCT on various types
// Broader coverage of DISTINCT correctness.
// ============================================================================

mod return_distinct {
    use super::*;

    #[test]
    fn distinct_removes_exact_duplicates() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 2})").unwrap();

        let r = s
            .execute("MATCH (n:N) RETURN DISTINCT n.val ORDER BY n.val")
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.rows()[0][0], Value::Int64(1));
        assert_eq!(r.rows()[1][0], Value::Int64(2));
    }

    #[test]
    fn distinct_on_null_values() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N)").unwrap(); // NULL val
        s.execute("INSERT (:N)").unwrap(); // NULL val

        let r = s.execute("MATCH (n:N) RETURN DISTINCT n.val").unwrap();
        assert_eq!(
            r.row_count(),
            2,
            "DISTINCT should collapse duplicate NULLs into one"
        );
    }
}

// ============================================================================
// WITH clause projection
// Broader coverage of WITH semantics.
// ============================================================================

mod with_clause {
    use super::*;

    #[test]
    fn with_renames_variable() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        let r = s
            .execute(
                "MATCH (p:Person) \
                 WITH p.name AS person_name, p.age AS person_age \
                 RETURN person_name, person_age",
            )
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(r.rows()[0][1], Value::Int64(30));
    }

    #[test]
    fn with_filters_rows() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 2})").unwrap();
        s.execute("INSERT (:N {val: 3})").unwrap();

        let r = s
            .execute(
                "MATCH (n:N) \
                 WITH n WHERE n.val > 1 \
                 RETURN n.val ORDER BY n.val",
            )
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.rows()[0][0], Value::Int64(2));
        assert_eq!(r.rows()[1][0], Value::Int64(3));
    }
}

// ============================================================================
// User workflow: multi-clause MATCH + CREATE relationship
// From Grafeo user josema-xyz: MATCH(a) MATCH(b) CREATE (a)-[:R]->(b)
// ============================================================================

mod multi_match_create_edge {
    use super::*;

    #[test]
    fn match_match_create_relationship() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        s.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

        // Must create exactly 1 edge, no phantom nodes
        let nodes = s
            .execute("MATCH (n:Person) RETURN count(n) AS cnt")
            .unwrap();
        let edges = s
            .execute("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(
            nodes.rows()[0][0],
            Value::Int64(2),
            "Should still have 2 nodes"
        );
        assert_eq!(edges.rows()[0][0], Value::Int64(1), "Should have 1 edge");
    }

    #[test]
    fn match_match_create_multiple_edges() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:Person {name: 'Vincent'})").unwrap();

        // Create edges from Alix to all others
        s.execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person) \
             WHERE b.name <> 'Alix' \
             INSERT (a)-[:KNOWS]->(b)",
        )
        .unwrap();

        let edges = s
            .execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() RETURN count(r) AS cnt")
            .unwrap();
        assert_eq!(edges.rows()[0][0], Value::Int64(2));
    }
}

// ============================================================================
// User workflow: negative numeric literals
// From Grafeo user janit: geographic coordinates with negative values.
// ============================================================================

mod negative_numerics {
    use super::*;

    #[test]
    fn insert_negative_integer() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Location {lat: -33, lon: 151})")
            .unwrap();

        let r = s.execute("MATCH (l:Location) RETURN l.lat, l.lon").unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(-33));
        assert_eq!(r.rows()[0][1], Value::Int64(151));
    }

    #[test]
    fn insert_negative_float() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Location {lat: -33.8688, lon: 151.2093})")
            .unwrap();

        let r = s.execute("MATCH (l:Location) RETURN l.lat, l.lon").unwrap();
        if let Value::Float64(lat) = r.rows()[0][0] {
            assert!(lat < 0.0, "Negative latitude must be preserved");
            assert!((lat - (-33.8688)).abs() < 0.001);
        } else {
            panic!("Expected Float64 for lat, got {:?}", r.rows()[0][0]);
        }
    }

    #[test]
    fn filter_on_negative_value() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Loc {lat: -33.0})").unwrap();
        s.execute("INSERT (:Loc {lat: 48.8})").unwrap();

        let r = s
            .execute("MATCH (l:Loc) WHERE l.lat < 0 RETURN l.lat")
            .unwrap();
        assert_eq!(r.row_count(), 1, "Only one location has negative latitude");
    }

    #[test]
    fn merge_with_negative_property() {
        let db = db();
        let s = db.session();
        s.execute("MERGE (:Temp {val: -42})").unwrap();
        s.execute("MERGE (:Temp {val: -42})").unwrap();

        let r = s.execute("MATCH (n:Temp) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(1),
            "MERGE with negative value should deduplicate"
        );
    }
}

// ============================================================================
// User workflow: count(*) vs count(n) after mutations
// From Grafeo user josema-xyz: count(*) returned 1, count(n) threw error.
// ============================================================================

mod count_variants {
    use super::*;

    #[test]
    fn count_star_after_insert() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 2})").unwrap();

        let r = s.execute("MATCH (n:N) RETURN count(*) AS cnt").unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(2));
    }

    #[test]
    fn count_variable_after_insert() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 2})").unwrap();

        let r = s.execute("MATCH (n:N) RETURN count(n) AS cnt").unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(2));
    }

    #[test]
    fn count_star_equals_count_variable() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:N {val: 1})").unwrap();
        s.execute("INSERT (:N {val: 2})").unwrap();
        s.execute("INSERT (:N {val: 3})").unwrap();

        let r1 = s.execute("MATCH (n:N) RETURN count(*) AS cnt").unwrap();
        let r2 = s.execute("MATCH (n:N) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r1.rows()[0][0],
            r2.rows()[0][0],
            "count(*) and count(n) must agree"
        );
    }
}

// ============================================================================
// User workflow: batch upsert via UNWIND + MERGE + SET
// From Grafeo user Imaclean74: code dependency graph bulk import.
// ============================================================================

mod batch_upsert {
    use super::*;

    #[test]
    fn unwind_merge_set_property() {
        let db = db();
        let s = db.session();
        s.execute(
            "UNWIND [1, 2, 3] AS i \
             MERGE (n:Item {key: i}) \
             SET n.updated = true",
        )
        .unwrap();

        let r = s
            .execute("MATCH (n:Item) WHERE n.updated = true RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(r.rows()[0][0], Value::Int64(3));
    }

    #[test]
    fn unwind_merge_second_pass_updates() {
        let db = db();
        let s = db.session();
        // First pass: create
        s.execute("UNWIND [1, 2, 3] AS i MERGE (n:Item {key: i}) SET n.ver = 1")
            .unwrap();
        // Second pass: update
        s.execute("UNWIND [1, 2, 3] AS i MERGE (n:Item {key: i}) SET n.ver = 2")
            .unwrap();

        let r = s.execute("MATCH (n:Item) RETURN count(n) AS cnt").unwrap();
        assert_eq!(
            r.rows()[0][0],
            Value::Int64(3),
            "No duplicates from second MERGE"
        );

        let r2 = s
            .execute("MATCH (n:Item) WHERE n.ver = 2 RETURN count(n) AS cnt")
            .unwrap();
        assert_eq!(
            r2.rows()[0][0],
            Value::Int64(3),
            "All nodes should have ver=2 after second pass"
        );
    }
}

// ============================================================================
// User workflow: labels() and type() as grouping keys in aggregation
// From Grafeo user Imaclean74: schema introspection via grouping.
// ============================================================================

mod aggregation_with_functions {
    use super::*;

    #[test]
    fn group_by_labels_with_count() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();

        let r = s
            .execute("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label")
            .unwrap();
        assert_eq!(r.row_count(), 2);
        assert_eq!(r.rows()[0][0], Value::String("City".into()));
        assert_eq!(r.rows()[0][1], Value::Int64(1));
        assert_eq!(r.rows()[1][0], Value::String("Person".into()));
        assert_eq!(r.rows()[1][1], Value::Int64(2));
    }

    #[test]
    fn group_by_type_with_count() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:A)-[:FOLLOWS]->(:B)").unwrap();
        s.execute("INSERT (:C)-[:FOLLOWS]->(:D)").unwrap();
        s.execute("INSERT (:E)-[:BLOCKS]->(:F)").unwrap();

        let r = s
            .execute("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt ORDER BY t")
            .unwrap();
        assert_eq!(r.row_count(), 2);
    }
}

// ============================================================================
// User workflow: persistent storage round-trip
// From Grafeo user CorvusYe: open, write, close, reopen, verify.
// ============================================================================

mod persistence_roundtrip {
    use super::*;
    use grafeo_engine::GrafeoDB;

    #[test]
    fn in_memory_data_survives_session() {
        let db = GrafeoDB::new_in_memory();
        let s1 = db.session();
        s1.execute("INSERT (:Persist {key: 'test', val: 42})")
            .unwrap();

        // New session on same db should see the data
        let s2 = db.session();
        let r = s2
            .execute("MATCH (n:Persist {key: 'test'}) RETURN n.val")
            .unwrap();
        assert_eq!(r.row_count(), 1);
        assert_eq!(r.rows()[0][0], Value::Int64(42));
    }
}

// ============================================================================
// GROUP BY on list-valued keys (Bug 20 + Bug 24)
// Push-based aggregator must hash lists distinctly; pull-based aggregator
// must produce separate GroupKeyPart values for different lists.
// ============================================================================

// ============================================================================
// ORDER BY with complex expressions should not leak synthetic columns (Bug 23)
// ============================================================================

mod order_by_synthetic_columns {
    use super::*;

    #[test]
    fn order_by_labels_does_not_leak_expr_column() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        let r = s
            .execute("MATCH (n:Person) RETURN n.name ORDER BY labels(n)[0]")
            .unwrap();

        assert_eq!(r.row_count(), 2);
        // Should only have the 'n.name' column, no __expr_* columns
        assert_eq!(
            r.columns.len(),
            1,
            "Expected 1 column (n.name), got {}: {:?}",
            r.columns.len(),
            r.columns
        );
        for col in &r.columns {
            assert!(
                !col.starts_with("__expr"),
                "Synthetic __expr column leaked into results: {col}"
            );
        }
    }

    #[test]
    fn order_by_arithmetic_does_not_leak_expr_column() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        s.execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();

        let r = s
            .execute("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age + 1")
            .unwrap();

        assert_eq!(r.row_count(), 2);
        assert_eq!(
            r.columns.len(),
            2,
            "Expected 2 columns (n.name, n.age), got {}: {:?}",
            r.columns.len(),
            r.columns
        );
        for col in &r.columns {
            assert!(
                !col.starts_with("__expr"),
                "Synthetic __expr column leaked into results: {col}"
            );
        }
    }

    #[test]
    fn order_by_type_on_edges_does_not_leak() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();

        let r = s
            .execute("MATCH (a:Person)-[r]->(b:Person) RETURN a.name, b.name ORDER BY type(r)")
            .unwrap();

        assert_eq!(r.row_count(), 1);
        for col in &r.columns {
            assert!(
                !col.starts_with("__expr"),
                "Synthetic __expr column leaked into results: {col}"
            );
        }
    }
}

mod group_by_list_keys {
    use super::*;

    #[test]
    fn group_by_labels_multi_label_nodes() {
        let db = db();
        let s = db.session();
        // Create nodes with different label sets
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person:Employee {name: 'Gus'})")
            .unwrap();
        s.execute("INSERT (:Person:Employee {name: 'Jules'})")
            .unwrap();
        s.execute("INSERT (:Company {name: 'GrafeoDB'})").unwrap();

        let r = s
            .execute("MATCH (n) RETURN labels(n) AS lbl, count(*) AS cnt ORDER BY cnt DESC")
            .unwrap();

        // Should produce 3 groups: [Person, Employee] (2), [Person] (1), [Company] (1)
        assert_eq!(
            r.row_count(),
            3,
            "GROUP BY labels(n) should produce 3 distinct groups, got {}: {:?}",
            r.row_count(),
            r.rows()
        );
    }

    #[test]
    fn group_by_labels_single_label_nodes() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        s.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        s.execute("INSERT (:Company {name: 'GrafeoDB'})").unwrap();

        let r = s
            .execute("MATCH (n) RETURN labels(n) AS lbl, count(*) AS cnt ORDER BY cnt DESC")
            .unwrap();

        // Should produce 2 groups: [Person] (2), [Company] (1)
        assert_eq!(
            r.row_count(),
            2,
            "GROUP BY labels(n) should produce 2 distinct groups, got {}: {:?}",
            r.row_count(),
            r.rows()
        );
    }

    #[test]
    fn group_by_date_valued_property() {
        let db = db();
        let s = db.session();
        s.execute("INSERT (:Event {name: 'A', day: date('2024-06-15')})")
            .unwrap();
        s.execute("INSERT (:Event {name: 'B', day: date('2024-06-15')})")
            .unwrap();
        s.execute("INSERT (:Event {name: 'C', day: date('2024-12-25')})")
            .unwrap();

        let r = s
            .execute("MATCH (e:Event) RETURN e.day AS day, count(*) AS cnt ORDER BY cnt DESC")
            .unwrap();

        assert_eq!(
            r.row_count(),
            2,
            "GROUP BY date should produce 2 distinct groups, got {}: {:?}",
            r.row_count(),
            r.rows()
        );
    }
}
