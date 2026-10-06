//! Integration tests for the push-based pipeline execution path.
//!
//! These tests verify that queries using filter, sort, aggregate, limit,
//! and distinct operators execute correctly through the push pipeline
//! (not the Volcano pull loop). Each test constructs a scenario that forces
//! the pipeline converter to decompose specific operators, exercising:
//!
//! - `into_any()` on each operator type
//! - `into_parts()` for decomposition
//! - `PredicateAdapter` bridging pull predicates to push
//! - `SortKey` conversion between pull/push types
//! - `Pipeline::execute()` main loop and `finalize_all()`
//! - `execute_pipeline()` in the Executor
//! - Early termination via Limit

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn setup_people_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (:Person {name: 'Alix', age: 30, city: 'Amsterdam'}),
                    (:Person {name: 'Gus', age: 25, city: 'Berlin'}),
                    (:Person {name: 'Vincent', age: 40, city: 'Amsterdam'}),
                    (:Person {name: 'Jules', age: 35, city: 'Paris'}),
                    (:Person {name: 'Mia', age: 28, city: 'Berlin'}),
                    (:Person {name: 'Butch', age: 45, city: 'Prague'}),
                    (:Person {name: 'Django', age: 32, city: 'Amsterdam'}),
                    (:Person {name: 'Shosanna', age: 27, city: 'Paris'})",
        )
        .unwrap();
    db
}

// ── Filter (exercises FilterPushOperator + PredicateAdapter) ─────────

#[test]
fn test_push_filter_equality() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) WHERE p.city = 'Amsterdam' RETURN p.name ORDER BY p.name")
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["Alix", "Django", "Vincent"]);
}

#[test]
fn test_push_filter_comparison() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) WHERE p.age > 35 RETURN p.name ORDER BY p.name")
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["Butch", "Vincent"]);
}

#[test]
fn test_push_filter_no_matches() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) WHERE p.age > 100 RETURN p.name")
        .unwrap();

    assert_eq!(result.rows().len(), 0);
}

// ── Sort (exercises SortPushOperator + convert_sort_key) ─────────────

#[test]
fn test_push_sort_ascending() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.age ASC")
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap())
        .collect();
    assert_eq!(names[0], "Gus"); // youngest (25)
    assert_eq!(names[names.len() - 1], "Butch"); // oldest (45)
    assert_eq!(names.len(), 8);
}

#[test]
fn test_push_sort_descending() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC")
        .unwrap();

    let first_age = &result.rows()[0][1];
    let last_age = &result.rows()[7][1];
    assert_eq!(*first_age, Value::Int64(45)); // Butch
    assert_eq!(*last_age, Value::Int64(25)); // Gus
}

#[test]
fn test_push_sort_multiple_keys() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.city, p.name ORDER BY p.city ASC, p.name ASC")
        .unwrap();

    // Amsterdam should come first, with names alphabetical within
    let first = &result.rows()[0];
    assert_eq!(first[0].as_str().unwrap(), "Amsterdam");
    assert_eq!(first[1].as_str().unwrap(), "Alix");
}

// ── Aggregate (exercises AggregatePushOperator) ──────────────────────

#[test]
fn test_push_aggregate_count() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN COUNT(*) AS total")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(8));
}

#[test]
fn test_push_aggregate_group_by() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.city, COUNT(*) AS cnt ORDER BY p.city ASC")
        .unwrap();

    // Amsterdam: 3, Berlin: 2, Paris: 2, Prague: 1
    assert_eq!(result.rows().len(), 4);

    let amsterdam = &result.rows()[0];
    assert_eq!(amsterdam[0].as_str().unwrap(), "Amsterdam");
    assert_eq!(amsterdam[1], Value::Int64(3));

    let prague = &result.rows()[3];
    assert_eq!(prague[0].as_str().unwrap(), "Prague");
    assert_eq!(prague[1], Value::Int64(1));
}

#[test]
fn test_push_aggregate_sum_avg() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN SUM(p.age) AS total_age, AVG(p.age) AS avg_age")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    // 30+25+40+35+28+45+32+27 = 262
    assert_eq!(result.rows()[0][0], Value::Int64(262));
}

#[test]
fn test_push_aggregate_min_max() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN MIN(p.age) AS youngest, MAX(p.age) AS oldest")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(25));
    assert_eq!(result.rows()[0][1], Value::Int64(45));
}

// ── Limit (exercises LimitPushOperator + early termination) ──────────

#[test]
fn test_push_limit() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name LIMIT 3")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Alix");
}

#[test]
fn test_push_limit_exceeds_data() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name LIMIT 100")
        .unwrap();

    // Only 8 people, limit 100 should return all 8
    assert_eq!(result.rows().len(), 8);
}

#[test]
fn test_push_limit_zero() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name LIMIT 0")
        .unwrap();

    assert_eq!(result.rows().len(), 0);
}

// ── Distinct (exercises DistinctPushOperator + hash_value_into) ──────

#[test]
fn test_push_distinct_strings() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN DISTINCT p.city ORDER BY p.city")
        .unwrap();

    let cities: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap())
        .collect();
    assert_eq!(cities, vec!["Amsterdam", "Berlin", "Paris", "Prague"]);
}

#[test]
fn test_push_distinct_integers() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:N {v: 1}), (:N {v: 2}), (:N {v: 1}), (:N {v: 3}), (:N {v: 2})")
        .unwrap();

    let result = session
        .execute("MATCH (n:N) RETURN DISTINCT n.v ORDER BY n.v")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0], Value::Int64(1));
    assert_eq!(result.rows()[1][0], Value::Int64(2));
    assert_eq!(result.rows()[2][0], Value::Int64(3));
}

// ── Combined operators (exercises multi-stage pipeline) ──────────────

#[test]
fn test_push_filter_sort_limit() {
    let db = setup_people_db();
    let session = db.session();

    // Filter + Sort + Limit: three push operators chained
    let result = session
        .execute("MATCH (p:Person) WHERE p.age >= 30 RETURN p.name ORDER BY p.age DESC LIMIT 3")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // Oldest 3 among age >= 30: Butch(45), Vincent(40), Jules(35)
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Butch");
    assert_eq!(result.rows()[1][0].as_str().unwrap(), "Vincent");
    assert_eq!(result.rows()[2][0].as_str().unwrap(), "Jules");
}

#[test]
fn test_push_aggregate_with_filter() {
    let db = setup_people_db();
    let session = db.session();

    // Filter + Aggregate: count people over 30 per city
    let result = session
        .execute(
            "MATCH (p:Person) WHERE p.age > 30 \
             RETURN p.city, COUNT(*) AS cnt \
             ORDER BY p.city ASC",
        )
        .unwrap();

    // Amsterdam: Vincent(40), Django(32) = 2
    // Paris: Jules(35) = 1
    // Prague: Butch(45) = 1
    assert!(result.rows().len() >= 3);
}

#[test]
fn test_push_distinct_with_sort() {
    let db = setup_people_db();
    let session = db.session();

    // Distinct + Sort: unique cities sorted
    let result = session
        .execute("MATCH (p:Person) RETURN DISTINCT p.city ORDER BY p.city DESC")
        .unwrap();

    let cities: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap())
        .collect();
    assert_eq!(cities, vec!["Prague", "Paris", "Berlin", "Amsterdam"]);
}

#[test]
fn test_push_filter_distinct_limit() {
    let db = setup_people_db();
    let session = db.session();

    // Filter + Distinct + Limit: unique cities of people over 25, limited to 2
    let result = session
        .execute(
            "MATCH (p:Person) WHERE p.age > 25 \
             RETURN DISTINCT p.city \
             ORDER BY p.city \
             LIMIT 2",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Amsterdam");
    assert_eq!(result.rows()[1][0].as_str().unwrap(), "Berlin");
}

// ── Pull fallback (verifies non-decomposable queries still work) ─────

#[test]
fn test_pull_fallback_scan_only() {
    let db = setup_people_db();
    let session = db.session();

    // Simple scan with no filter/sort/limit: should stay pull-based
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();

    assert_eq!(result.rows().len(), 8);
}

#[test]
fn test_pull_fallback_with_expand() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:A {name: 'Alix'})-[:KNOWS]->(:B {name: 'Gus'})")
        .unwrap();

    // Expand stays pull-based, but result should be correct
    let result = session
        .execute("MATCH (a:A)-[:KNOWS]->(b:B) RETURN a.name, b.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Alix");
    assert_eq!(result.rows()[0][1].as_str().unwrap(), "Gus");
}

// ── Edge cases ───────────────────────────────────────────────────────

#[test]
fn test_push_empty_result_through_pipeline() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // No data in the database, pipeline should handle empty gracefully
    let result = session
        .execute("MATCH (p:Person) WHERE p.age > 0 RETURN p.name ORDER BY p.name LIMIT 10")
        .unwrap();

    assert_eq!(result.rows().len(), 0);
}

#[test]
fn test_push_single_row_through_pipeline() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Solo {name: 'Alix', v: 1})")
        .unwrap();

    let result = session
        .execute("MATCH (s:Solo) WHERE s.v = 1 RETURN s.name ORDER BY s.name LIMIT 10")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Alix");
}

#[test]
fn test_push_aggregate_on_empty() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // COUNT on empty should return 0
    let result = session
        .execute("MATCH (p:Person) RETURN COUNT(*) AS cnt")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(0));
}

// ── Pipeline conversion tests (via query execution) ──────────────────

#[test]
fn test_pipeline_converts_filter_to_push() {
    // Verify the push path is actually taken by checking that queries
    // with WHERE clauses produce correct results (they go through
    // FilterPushOperator + PredicateAdapter)
    let db = setup_people_db();
    let session = db.session();

    // This query decomposes: Scan -> Filter(push) -> Sort(push) -> Limit(push)
    let result = session
        .execute(
            "MATCH (p:Person) WHERE p.city = 'Amsterdam' AND p.age > 29 \
             RETURN p.name ORDER BY p.name LIMIT 10",
        )
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap())
        .collect();
    // Alix(30), Django(32), Vincent(40) all in Amsterdam and age > 29
    assert_eq!(names, vec!["Alix", "Django", "Vincent"]);
}

#[test]
fn test_pipeline_aggregate_group_sort() {
    // Aggregate + Sort: both are pipeline breakers, exercises finalize chain
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (p:Person) \
             RETURN p.city, COUNT(*) AS cnt, AVG(p.age) AS avg_age \
             ORDER BY cnt DESC, p.city ASC",
        )
        .unwrap();

    // Amsterdam: 3 people, Berlin: 2, Paris: 2, Prague: 1
    assert_eq!(result.rows().len(), 4);
    // First row should be Amsterdam (count 3, highest)
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Amsterdam");
    assert_eq!(result.rows()[0][1], Value::Int64(3));
}

// ═══════════════════════════════════════════════════════════════════════
// Operator coverage: queries that exercise every operator type through
// the full engine, covering into_any(), into_parts(), and pipeline
// conversion for all current and future push operations.
// ═══════════════════════════════════════════════════════════════════════

// ── Mutation operators (INSERT/DELETE with RETURN) ────────────────────

#[test]
fn test_mutation_insert_return() {
    // INSERT ... RETURN exercises mutation operators through the pipeline
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("INSERT (p:Robot {name: 'Gort', year: 1951}) RETURN p")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    // INSERT RETURN p returns the full node; verify it exists
    assert!(!result.rows()[0][0].is_null());
}

#[test]
fn test_mutation_insert_multiple_return_count() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "INSERT (:Item {v: 1}), (:Item {v: 2}), (:Item {v: 3}), \
                    (:Item {v: 4}), (:Item {v: 5})",
        )
        .unwrap();

    let result = session
        .execute("MATCH (i:Item) WHERE i.v > 2 RETURN COUNT(*) AS cnt")
        .unwrap();

    assert_eq!(result.rows()[0][0], Value::Int64(3));
}

#[test]
fn test_mutation_delete_return() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("INSERT (:Temp {v: 1}), (:Temp {v: 2}), (:Keep {v: 3})")
        .unwrap();

    session
        .execute("MATCH (t:Temp) DELETE t RETURN COUNT(*) AS deleted")
        .unwrap();

    // Verify delete happened
    let remaining = session.execute("MATCH (n) RETURN COUNT(*) AS cnt").unwrap();
    assert_eq!(remaining.rows()[0][0], Value::Int64(1)); // Only :Keep remains
}

// ── Join operators (multi-pattern MATCH) ─────────────────────────────

#[test]
fn test_join_cartesian() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("INSERT (:Color {name: 'Red'}), (:Color {name: 'Blue'})")
        .unwrap();
    session
        .execute("INSERT (:Size {name: 'Small'}), (:Size {name: 'Large'})")
        .unwrap();

    // Cartesian join: 2 colors x 2 sizes = 4 combinations
    let result = session
        .execute(
            "MATCH (c:Color), (s:Size) RETURN c.name, s.name \
             ORDER BY c.name, s.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 4);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Blue");
    assert_eq!(result.rows()[0][1].as_str().unwrap(), "Large");
}

// ── Set operators (UNION) ────────────────────────────────────────────

#[test]
fn test_union_distinct() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (p:Person) WHERE p.city = 'Amsterdam' RETURN p.name \
             UNION \
             MATCH (p:Person) WHERE p.age > 40 RETURN p.name",
        )
        .unwrap();

    // Amsterdam: Alix, Django, Vincent. Age > 40: Butch, Vincent.
    // UNION deduplicates: Alix, Django, Vincent, Butch = 4
    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 4);
    assert!(names.contains(&"Alix"));
    assert!(names.contains(&"Butch"));
    assert!(names.contains(&"Vincent"));
}

#[test]
fn test_union_all() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (p:Person) WHERE p.city = 'Amsterdam' RETURN p.name \
             UNION ALL \
             MATCH (p:Person) WHERE p.age > 40 RETURN p.name",
        )
        .unwrap();

    // Amsterdam: 3, Age > 40: 1 (Butch), but Vincent is in both
    // UNION ALL keeps duplicates: 3 + 1 = 4
    assert!(result.rows().len() >= 4);
}

// ── Variable-length expand (path operators) ──────────────────────────

#[test]
fn test_variable_length_expand() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "INSERT (a:Node {name: 'A'})-[:NEXT]->(b:Node {name: 'B'})-[:NEXT]->(c:Node {name: 'C'})-[:NEXT]->(d:Node {name: 'D'})",
        )
        .unwrap();

    // 2..3 hops from A should reach C and D
    let result = session
        .execute(
            "MATCH (a:Node {name: 'A'})-[:NEXT*2..3]->(target) \
             RETURN target.name ORDER BY target.name",
        )
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| r[0].as_str().unwrap())
        .collect();
    assert!(names.contains(&"C")); // 2 hops
    assert!(names.contains(&"D")); // 3 hops
}

#[test]
fn test_variable_length_expand_with_filter() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "INSERT (:S {name: 'Start'})-[:HOP]->(:M {name: 'Mid', v: 10})-[:HOP]->(:E {name: 'End', v: 20})",
        )
        .unwrap();

    let result = session
        .execute(
            "MATCH (:S)-[:HOP*1..2]->(t) WHERE t.v > 15 \
             RETURN t.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "End");
}

// ── SingleRow operator (RETURN without MATCH) ────────────────────────

#[test]
fn test_single_row_return_literal() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("RETURN 42 AS answer").unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(42));
}

#[test]
fn test_single_row_return_expression() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("RETURN 2 + 3 AS sum, 'hello' AS greeting")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(5));
    assert_eq!(result.rows()[0][1].as_str().unwrap(), "hello");
}

// ── Project operator (RETURN with expressions) ───────────────────────

#[test]
fn test_project_computed_columns() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (p:Person) WHERE p.name = 'Alix' \
             RETURN p.name, p.age, p.age + 10 AS future_age",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Alix");
    assert_eq!(result.rows()[0][1], Value::Int64(30));
    assert_eq!(result.rows()[0][2], Value::Int64(40));
}

// ── Expand operator (traversals) ─────────────────────────────────────

#[test]
fn test_expand_with_filter_sort_limit() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "INSERT (a:Hub {name: 'Hub'}), \
                    (b:Spoke {name: 'B', v: 3}), \
                    (c:Spoke {name: 'C', v: 1}), \
                    (d:Spoke {name: 'D', v: 2}), \
                    (a)-[:LINK]->(b), (a)-[:LINK]->(c), (a)-[:LINK]->(d)",
        )
        .unwrap();

    // Expand + Filter + Sort + Limit: exercises full pipeline chain
    let result = session
        .execute(
            "MATCH (h:Hub)-[:LINK]->(s:Spoke) WHERE s.v > 1 \
             RETURN s.name ORDER BY s.v ASC LIMIT 2",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "D"); // v=2
    assert_eq!(result.rows()[1][0].as_str().unwrap(), "B"); // v=3
}

// ── Complex multi-operator pipelines ─────────────────────────────────

#[test]
fn test_full_pipeline_chain() {
    // This query exercises: Scan -> Expand -> Filter -> Aggregate -> Sort -> Limit
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "INSERT (:Dept {name: 'Eng'}), (:Dept {name: 'Sales'}), (:Dept {name: 'HR'}), \
                    (:Emp {name: 'Alix', salary: 100})-[:IN]->(:Dept {name: 'Eng'}), \
                    (:Emp {name: 'Gus', salary: 120})-[:IN]->(:Dept {name: 'Eng'}), \
                    (:Emp {name: 'Jules', salary: 90})-[:IN]->(:Dept {name: 'Sales'})",
        )
        .unwrap();

    let result = session
        .execute(
            "MATCH (e:Emp)-[:IN]->(d:Dept) \
             RETURN d.name, COUNT(*) AS headcount, SUM(e.salary) AS total \
             ORDER BY headcount DESC LIMIT 2",
        )
        .unwrap();

    assert!(!result.rows().is_empty(), "result.rows() is empty");
    // Eng has 2 employees, should be first
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Eng");
    assert_eq!(result.rows()[0][1], Value::Int64(2));
}

#[test]
fn test_distinct_after_expand() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "INSERT (a:A {name: 'Alix'})-[:R]->(b:B {name: 'Gus'}), \
                    (a)-[:R]->(c:B {name: 'Gus'})",
        )
        .unwrap();

    // Two edges from A, both targets named "Gus": DISTINCT should collapse
    let result = session
        .execute("MATCH (:A)-[:R]->(b) RETURN DISTINCT b.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Gus");
}

// ── SKIP operator ────────────────────────────────────────────────────

#[test]
fn test_skip_with_limit() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name SKIP 2 LIMIT 3")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // Sorted: Alix, Butch, Django, Gus, Jules, Mia, Shosanna, Vincent
    // Skip 2 (Alix, Butch), take 3: Django, Gus, Jules
    assert_eq!(result.rows()[0][0].as_str().unwrap(), "Django");
    assert_eq!(result.rows()[1][0].as_str().unwrap(), "Gus");
    assert_eq!(result.rows()[2][0].as_str().unwrap(), "Jules");
}

#[test]
fn test_skip_all() {
    let db = setup_people_db();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) RETURN p.name SKIP 100")
        .unwrap();

    assert_eq!(result.rows().len(), 0);
}
