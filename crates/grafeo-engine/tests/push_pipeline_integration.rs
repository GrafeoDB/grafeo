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

fn setup_mixed_identity_db() -> (GrafeoDB, Vec<Value>) {
    use std::collections::HashMap;
    use std::sync::Arc;

    let left = Arc::new(HashMap::from([("left".to_owned(), 1_u64)]));
    let right = Arc::new(HashMap::from([("right".to_owned(), 1_u64)]));
    let empty = Arc::new(HashMap::new());
    let values = [
        Value::Int64(1),
        Value::Float64(1.0),
        Value::List(vec![Value::Int64(1)].into()),
        Value::List(vec![Value::Float64(1.0)].into()),
        Value::GCounter(Arc::clone(&left)),
        Value::GCounter(right),
        Value::OnCounter {
            pos: Arc::clone(&left),
            neg: Arc::clone(&empty),
        },
        Value::OnCounter {
            pos: empty,
            neg: left,
        },
        Value::Vector(vec![1.0_f32, 2.0].into()),
        Value::Vector(vec![1.0_f32, 3.0].into()),
        Value::Null,
    ];
    let expected = [0, 2, 4, 5, 6, 7, 8, 9, 10]
        .into_iter()
        .map(|index| values[index].clone())
        .collect();
    let db = GrafeoDB::new_in_memory();
    // 2,816 rows cross the scan's chunk boundary and revisit every identity.
    for _ in 0..256 {
        for value in &values {
            db.create_node_with_props(&["IdentityValue"], [("v", value.clone())])
                .unwrap();
        }
    }
    (db, expected)
}

fn mixed_identity_class(value: &Value, expected: &[Value]) -> usize {
    // Literal fixture values define the oracle, independently of RowKey.
    let is_one = |value: &Value| {
        matches!(value, Value::Int64(1))
            || matches!(value, Value::Float64(number) if number.to_bits() == 1.0_f64.to_bits())
    };
    if is_one(value) {
        return 0;
    }
    if let Value::List(items) = value
        && items.len() == 1
        && is_one(&items[0])
    {
        return 1;
    }
    expected
        .iter()
        .position(|candidate| candidate == value)
        .unwrap_or_else(|| panic!("unexpected identity value: {value:?}"))
}

fn assert_mixed_identity_rows(
    result: &grafeo_engine::database::QueryResult,
    expected: &[Value],
    grouped: bool,
) {
    assert_eq!(result.rows().len(), 9);
    let mut seen = [false; 9];
    let mut total = 0;
    for row in result.rows() {
        assert_eq!(row.len(), if grouped { 2 } else { 1 });
        let class = mixed_identity_class(&row[0], expected);
        assert!(!seen[class], "duplicate identity class: {row:?}");
        seen[class] = true;
        if grouped {
            let count = row[1].as_int64().unwrap();
            assert_eq!(count, if class < 2 { 512 } else { 256 }, "{row:?}");
            total += count;
        }
    }
    assert!(seen.into_iter().all(|present| present));
    if grouped {
        assert_eq!(total, 2816);
    }
}

fn assert_profile_operator_rows(
    result: &grafeo_engine::database::QueryResult,
    operator: &str,
    expected_rows: usize,
) {
    let report = result.rows()[0][0].as_str().unwrap();
    let prefix = format!("{operator} ");
    let operators: Vec<_> = report
        .lines()
        .filter(|line| line.trim_start().starts_with(&prefix))
        .collect();
    assert!(
        !operators.is_empty(),
        "missing {operator} in PROFILE: {report}"
    );
    // PROFILE can give an aggregate and its projection the same operator name.
    for line in &operators {
        let rows = line
            .split_whitespace()
            .find_map(|field| field.strip_prefix("rows="))
            .unwrap_or_else(|| panic!("missing row count: {line}"))
            .parse::<usize>()
            .unwrap();
        assert_eq!(rows, expected_rows, "{report}");
    }
    let mut pending = vec![
        &result
            .profile()
            .expect("PROFILE has a structured tree")
            .root,
    ];
    let mut matching_nodes = 0;
    while let Some(node) = pending.pop() {
        if node.operator == operator {
            assert_eq!(node.rows, u64::try_from(expected_rows).unwrap(), "{node:?}");
            matching_nodes += 1;
        }
        pending.extend(node.children.iter());
    }
    assert_eq!(matching_nodes, operators.len(), "text and tree must agree");
}

#[test]
fn native_group_identity_preserves_mixed_values_across_chunks() {
    let (db, expected) = setup_mixed_identity_db();
    let session = db.session();
    let query = "MATCH (n:IdentityValue) RETURN n.v AS key, count(*) AS c";
    for _ in 0..2 {
        let result = session.execute(query).unwrap();
        assert_mixed_identity_rows(&result, &expected, true);
    }
    let profile = session.execute(&format!("PROFILE {query}")).unwrap();
    assert_profile_operator_rows(&profile, "HashAggregate", 9);
}

#[test]
fn native_row_distinct_preserves_mixed_values_across_chunks() {
    let (db, expected) = setup_mixed_identity_db();
    let session = db.session();
    let query = "MATCH (n:IdentityValue) RETURN DISTINCT n.v AS key";
    for _ in 0..2 {
        let result = session.execute(query).unwrap();
        assert_mixed_identity_rows(&result, &expected, false);
    }
    let profile = session.execute(&format!("PROFILE {query}")).unwrap();
    assert_profile_operator_rows(&profile, "Distinct", 9);
}

#[test]
fn native_count_distinct_preserves_mixed_value_identities() {
    let (db, _) = setup_mixed_identity_db();
    let session = db.session();
    let query = "MATCH (n:IdentityValue) \
                 RETURN count(DISTINCT n.v) AS distinct_values, count(*) AS total";
    for _ in 0..2 {
        let result = session.execute(query).unwrap();
        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0], vec![Value::Int64(8), Value::Int64(2816)]);
    }
    // PROFILE exposes the aggregate's output cardinality, not its count value.
    let profile = session.execute(&format!("PROFILE {query}")).unwrap();
    assert_profile_operator_rows(&profile, "SimpleAggregate", 1);
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

/// Later input chunks must update groups that pressure spilled to disk.
#[cfg(feature = "spill")]
#[test]
fn grouped_distinct_keeps_later_rows_after_pressure_spill() {
    use grafeo_common::memory::buffer::{MemoryRegion, PressureLevel};
    use grafeo_engine::Config;

    let spill_root = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_memory_limit(64 * 1024 * 1024)
            .with_spill_path(spill_root.path())
            // Under pressure every partition spills and reloads; on a busy
            // Windows runner that took 16 to 47 s, past the 30 s default.
            // The test checks the results, not the speed.
            .without_query_timeout(),
    )
    .unwrap();
    let manager = db.buffer_manager();
    let query = "UNWIND range(0, 8191) AS x \
                 RETURN x % 512 AS bucket, count(DISTINCT x) AS c";
    let session = db.session();

    let check = |result: &grafeo_engine::database::QueryResult| {
        let mut rows = result.rows().to_vec();
        rows.sort_by_key(|row| row[0].as_int64().unwrap());
        assert_eq!(rows.len(), 512);
        for (bucket, row) in rows.iter().enumerate() {
            assert_eq!(
                row,
                &vec![
                    Value::Int64(i64::try_from(bucket).unwrap()),
                    Value::Int64(16)
                ]
            );
        }
    };

    // Keep pressure High throughout execution. Each source chunk revisits all
    // 512 groups, above the spill guard, so later chunks reload spilled groups.
    let pressure = manager
        .try_allocate(
            manager.config().budget * 9 / 10,
            MemoryRegion::ExecutionBuffers,
        )
        .unwrap();
    assert_eq!(manager.pressure_level(), PressureLevel::High);
    check(&session.execute(query).unwrap());
    check(&session.execute(query).unwrap()); // Cached plan, still under pressure.
    assert_eq!(manager.pressure_level(), PressureLevel::High);
    drop(pressure);

    // Every spilled partition was reloaded and its file removed.
    assert!(
        std::fs::read_dir(spill_root.path())
            .unwrap()
            .next()
            .is_none()
    );
}

#[cfg(feature = "spill")]
fn grouped_result_fixture() -> (tempfile::TempDir, GrafeoDB) {
    let root = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory()
            .with_memory_limit(32 * 1024 * 1024)
            .with_spill_path(root.path())
            .without_query_timeout(),
    )
    .unwrap();
    (root, db)
}

#[cfg(feature = "spill")]
#[test]
fn grouped_consumer_scans_stay_bounded_across_spill_reload() {
    use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, StorageTier};
    use grafeo_common::utils::error::Error;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ScanCounter(AtomicUsize);
    impl MemoryConsumer for ScanCounter {
        fn name(&self) -> &str {
            "grouped-query-scan-counter"
        }
        fn memory_usage(&self) -> usize {
            self.0.fetch_add(1, Ordering::Relaxed);
            0
        }
        fn eviction_priority(&self) -> u8 {
            0
        }
        fn region(&self) -> MemoryRegion {
            MemoryRegion::ExecutionBuffers
        }
        fn evict(&self, _target_bytes: usize) -> usize {
            0
        }
        fn current_tier(&self) -> StorageTier {
            StorageTier::Uninitialized
        }
    }

    const GROUPS: usize = 1024;
    const BUDGET: usize = 32 << 20;
    const QUERY: &str = "MATCH (n:AggregateCost) RETURN n.bucket AS bucket, sum(n.value) AS total, count(*) AS count";
    let mut scans = Vec::new();
    for repetitions in [2_i64, 4] {
        let root = tempfile::tempdir().unwrap();
        let spill_path = root.path().join("spill");
        let db = GrafeoDB::with_config(
            grafeo_engine::Config::in_memory()
                .with_memory_limit(BUDGET)
                .with_spill_path(&spill_path)
                .without_query_timeout(),
        )
        .unwrap();
        for value in 1..=repetitions {
            for bucket in 0..GROUPS {
                db.create_node_with_props(
                    &["AggregateCost"],
                    [
                        ("bucket", Value::Int64(i64::try_from(bucket).unwrap())),
                        ("value", Value::Int64(value)),
                    ],
                )
                .unwrap();
            }
        }
        let manager = db.buffer_manager();
        let baseline = manager.allocated();
        let held = manager
            .try_allocate(BUDGET * 9 / 10 - baseline, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let held_total = manager.allocated();
        let session = db.session();
        // A blocked lazy directory proves this workload really spills.
        std::fs::write(&spill_path, b"blocked spill directory").unwrap();
        assert!(matches!(session.execute(QUERY), Err(Error::Io(_))));
        assert_eq!(manager.allocated(), held_total);
        std::fs::remove_file(&spill_path).unwrap();
        drop(session.execute(QUERY).unwrap());
        assert_eq!(manager.allocated(), held_total);

        let counter = Arc::new(ScanCounter(AtomicUsize::new(0)));
        manager.register_consumer(counter.clone());
        let before = db.query_cache().stats();
        let result = session.execute(QUERY).unwrap();
        scans.push(counter.0.load(Ordering::Relaxed));
        assert!(db.query_cache().stats().optimized_hits > before.optimized_hits);
        assert_eq!(result.row_count(), GROUPS);
        let mut seen = [false; GROUPS];
        for row in result.rows() {
            let [
                Value::Int64(bucket),
                Value::Int64(total),
                Value::Int64(count),
            ] = row.as_slice()
            else {
                panic!("unexpected grouped row: {row:?}");
            };
            let bucket = usize::try_from(*bucket).unwrap();
            assert!(bucket < GROUPS && !seen[bucket]);
            seen[bucket] = true;
            assert_eq!(
                (*total, *count),
                (repetitions * (repetitions + 1) / 2, repetitions)
            );
        }
        assert!(seen.into_iter().all(|present| present));
        assert!(manager.allocated() > held_total);
        drop(result);
        assert_eq!(manager.allocated(), held_total);
        assert!(std::fs::read_dir(&spill_path).unwrap().next().is_none());
        drop(held);
        assert_eq!(manager.allocated(), baseline);
    }
    // Initial state and input schema establish admission. Child owners must
    // grow that authority without repeating global scans during reload/output.
    assert!(
        scans.iter().all(|count| *count <= 2),
        "consumer scans for one/two input chunks: {scans:?}"
    );
    assert_eq!(scans[0], scans[1]);
}

#[cfg(feature = "spill")]
fn assert_grouped_result_rows(result: &grafeo_engine::database::QueryResult) {
    assert_eq!(result.row_count(), 2051);
    let mut seen = vec![false; 2051];
    for row in result.rows() {
        let key = usize::try_from(row[0].as_int64().unwrap()).unwrap();
        assert!(key < 2051);
        assert!(!seen[key]);
        seen[key] = true;
        assert_eq!(row[1], Value::Int64(2));
    }
    assert!(seen.into_iter().all(|present| present));
}

#[cfg(feature = "spill")]
#[test]
fn grouped_result_retains_cold_and_cached_rows_until_drop_or_transfer() {
    let (root, db) = grouped_result_fixture();
    let manager = db.buffer_manager();
    let session = db.session();
    let query = "UNWIND range(0, 4101) AS x RETURN x % 2051 AS bucket, count(*) AS c";
    let baseline = manager.allocated();
    let first = session.execute(query).unwrap();
    assert_grouped_result_rows(&first);
    assert!(
        manager.allocated() > baseline,
        "cold grouped result owns its output buffers"
    );
    drop(first);
    assert_eq!(manager.allocated(), baseline);
    let cache = db.query_cache().stats();
    let cached = session.execute(query).unwrap();
    assert!(db.query_cache().stats().optimized_hits > cache.optimized_hits);
    assert_grouped_result_rows(&cached);
    assert!(
        manager.allocated() > baseline,
        "cached plan preserves result ownership"
    );
    let rows = cached.into_rows();
    assert_eq!(rows.len(), 2051);
    assert_eq!(
        manager.allocated(),
        baseline,
        "caller-owned rows leave executor accounting"
    );
    assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
}

#[cfg(feature = "spill")]
#[test]
fn grouped_read_counts_survive_cold_and_cached_result_ownership() {
    use grafeo_common::memory::buffer::MemoryRegion;

    for pressured in [false, true] {
        let (root, db) = grouped_result_fixture();
        db.execute("UNWIND range(0, 3) AS i INSERT (:GroupedTelemetry {bucket: i % 2})")
            .unwrap();
        let manager = db.buffer_manager();
        let baseline = manager.allocated();
        let held = pressured.then(|| {
            manager
                .try_allocate(
                    manager.config().budget * 9 / 10 - baseline,
                    MemoryRegion::ExecutionBuffers,
                )
                .unwrap()
        });
        let held_total = manager.allocated();
        let session = db.session();
        let query = "MATCH (n:GroupedTelemetry) RETURN n.bucket AS bucket, count(*) AS c";
        for transfer in [false, true] {
            let before = db.query_cache().stats();
            let result = session.execute(query).unwrap();
            let after = db.query_cache().stats();
            if transfer {
                assert!(after.optimized_hits > before.optimized_hits);
            } else {
                assert_eq!(after.optimized_hits, before.optimized_hits);
                assert!(after.optimized_misses > before.optimized_misses);
            }
            assert_eq!(result.rows_scanned, Some(4));
            assert!(result.profile().is_none());
            assert_eq!(result.row_count(), 2);
            let mut seen = [false; 2];
            for row in result.rows() {
                let bucket = usize::try_from(row[0].as_int64().unwrap()).unwrap();
                assert!(!seen[bucket]);
                seen[bucket] = true;
                assert_eq!(row[1], Value::Int64(2));
            }
            assert_eq!(seen, [true; 2]);
            assert!(manager.allocated() > held_total);
            if transfer {
                let rows = result.into_rows();
                assert_eq!(manager.allocated(), held_total);
                assert_eq!(rows.len(), 2);
            } else {
                drop(result);
            }
            assert_eq!(manager.allocated(), held_total);
            assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
        }
        drop(held);
        assert_eq!(manager.allocated(), baseline);
    }
}

#[cfg(feature = "spill")]
#[test]
fn grouped_result_denial_cleans_cold_and_cached_queries() {
    use grafeo_common::memory::buffer::MemoryRegion;
    let (root, db) = grouped_result_fixture();
    let manager = db.buffer_manager();
    let session = db.session();
    let query = "UNWIND range(0, 4101) AS x RETURN x % 2051 AS bucket, count(*) AS c";
    let baseline = manager.allocated();
    // Config's default hard limit is 95% of the budget. Occupy that exact
    // limit so even the first grouped workspace cannot be admitted.
    let limit = manager.config().budget * 95 / 100;
    let held = manager
        .try_allocate(limit - baseline, MemoryRegion::ExecutionBuffers)
        .unwrap();
    for _ in 0..2 {
        let Err(error) = session.execute(query) else {
            panic!("no grouped workspace headroom remains");
        };
        assert!(error.to_string().contains("memory"), "{error}");
        assert_eq!(manager.allocated(), baseline + held.size());
        assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
    }
    drop(held);
    assert_eq!(manager.allocated(), baseline);
    let result = session.execute(query).unwrap();
    assert_grouped_result_rows(&result);
    drop(result);
    assert_eq!(manager.allocated(), baseline);
}

#[cfg(feature = "spill")]
#[test]
fn grouped_limit_retains_and_releases_results() {
    let (root, db) = grouped_result_fixture();
    let manager = db.buffer_manager();
    let baseline = manager.allocated();
    let session = db.session();
    for limit in [0, 1, 2050] {
        let query = format!(
            "UNWIND range(0, 4101) AS x RETURN x % 2051 AS bucket, count(*) AS c LIMIT {limit}"
        );
        for _ in 0..2 {
            let result = session.execute(&query).unwrap();
            assert_eq!(result.row_count(), limit);
            let mut seen = vec![false; 2051];
            for row in result.rows() {
                let key = usize::try_from(row[0].as_int64().unwrap()).unwrap();
                assert!(!seen[key]);
                seen[key] = true;
                assert_eq!(row[1], Value::Int64(2));
            }
            if limit > 0 {
                assert!(manager.allocated() > baseline);
            }
            drop(result);
            assert_eq!(manager.allocated(), baseline);
            assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
        }
    }
    let profile = session
        .execute(
            "PROFILE UNWIND range(0, 4101) AS x RETURN x % 2051 AS bucket, count(*) AS c LIMIT 1",
        )
        .unwrap();
    assert_eq!(profile.row_count(), 1);
    drop(profile);
    assert_eq!(manager.allocated(), baseline);
}

#[cfg(feature = "spill")]
#[test]
fn grouped_decimal_identities_survive_chunks_and_pressure_spill() {
    use grafeo_common::memory::buffer::{MemoryRegion, PressureLevel};
    use grafeo_common::types::Decimal;

    let root = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory()
            .with_memory_limit(64 * 1024 * 1024)
            .with_spill_path(root.path())
            .without_query_timeout(),
    )
    .unwrap();
    let precise = Decimal::parse("0.1234567890123456789012345678").unwrap();
    let precise_scaled = Decimal::parse("0.12345678901234567890123456780").unwrap();
    let values = [
        Value::Decimal(Decimal::new(5, 1).unwrap()),
        Value::Decimal(Decimal::new(50, 2).unwrap()),
        Value::Float64(0.5),
        Value::Int64(1),
        Value::Float64(1.0),
        Value::Decimal(Decimal::new(100, 2).unwrap()),
        Value::List(vec![Value::Decimal(Decimal::new(5, 1).unwrap())].into()),
        Value::List(vec![Value::Decimal(Decimal::new(50, 2).unwrap())].into()),
        Value::List(vec![Value::Float64(0.5)].into()),
        Value::Decimal(precise),
        Value::Decimal(precise_scaled),
        Value::Float64(precise.to_f64()),
    ];
    for value in &values {
        for bucket in 0..512_i64 {
            db.create_node_with_props(
                &["DecimalIdentity"],
                [("bucket", Value::Int64(bucket)), ("v", value.clone())],
            )
            .unwrap();
        }
    }
    let manager = db.buffer_manager();
    let baseline = manager.allocated();
    let session = db.session();
    let is_half = |value: &Value| match value {
        Value::Decimal(value) => *value == Decimal::new(5, 1).unwrap(),
        Value::Float64(value) => value.to_bits() == 0.5_f64.to_bits(),
        _ => false,
    };
    for pressure in [false, true] {
        let held = pressure.then(|| {
            manager
                .try_allocate(
                    (manager.config().budget * 9 / 10)
                        .checked_sub(baseline)
                        .unwrap(),
                    MemoryRegion::ExecutionBuffers,
                )
                .unwrap()
        });
        if pressure {
            assert_eq!(manager.pressure_level(), PressureLevel::High);
        }
        let held_total = manager.allocated();
        for _ in 0..2 {
            let result = session
                .execute("MATCH (n:DecimalIdentity) RETURN n.bucket AS bucket, count(DISTINCT n.v) AS distinct_values, count(*) AS total")
                .unwrap();
            assert_eq!(result.row_count(), 512);
            let mut seen = [false; 512];
            for row in result.rows() {
                let bucket = usize::try_from(row[0].as_int64().unwrap()).unwrap();
                assert!(!seen[bucket]);
                seen[bucket] = true;
                assert_eq!(&row[1..], &[Value::Int64(5), Value::Int64(12)]);
            }
            assert!(seen.into_iter().all(|present| present));
            drop(result);
            assert_eq!(manager.allocated(), held_total);

            let result = session
                .execute("MATCH (n:DecimalIdentity) RETURN n.bucket AS bucket, n.v AS value, count(*) AS c")
                .unwrap();
            assert_eq!(result.row_count(), 512 * 5);
            let mut seen = vec![[false; 5]; 512];
            for row in result.rows() {
                let bucket = usize::try_from(row[0].as_int64().unwrap()).unwrap();
                let value = &row[1];
                let class = if is_half(value) {
                    0
                } else if matches!(value, Value::Int64(1))
                    || matches!(value, Value::Float64(number) if number.to_bits() == 1.0_f64.to_bits())
                    || matches!(value, Value::Decimal(number) if *number == Decimal::from_i64(1))
                {
                    1
                } else if matches!(value, Value::List(items) if items.len() == 1 && is_half(&items[0]))
                {
                    2
                } else if matches!(value, Value::Decimal(number) if *number == precise) {
                    3
                } else if matches!(value, Value::Float64(number) if number.to_bits() == precise.to_f64().to_bits())
                {
                    4
                } else {
                    panic!("unexpected decimal identity: {value:?}");
                };
                assert!(!seen[bucket][class]);
                seen[bucket][class] = true;
                assert_eq!(row[2], Value::Int64([3, 3, 3, 2, 1][class]));
            }
            assert!(
                seen.iter()
                    .all(|classes| classes.iter().all(|&present| present))
            );
            drop(result);
            assert_eq!(manager.allocated(), held_total);
            assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
        }
        drop(held);
        assert_eq!(manager.allocated(), baseline);
    }
}

#[cfg(feature = "spill")]
#[test]
fn profile_grouped_denial_preserves_headroom() {
    use grafeo_common::memory::buffer::MemoryRegion;
    let (root, db) = grouped_result_fixture();
    let manager = db.buffer_manager();
    let baseline = manager.allocated();
    let held = manager
        .try_allocate(
            manager.hard_limit() - baseline,
            MemoryRegion::ExecutionBuffers,
        )
        .unwrap();
    let session = db.session();
    let query = "PROFILE UNWIND range(0, 4101) AS x RETURN x % 2051 AS bucket, count(*) AS c";
    for _ in 0..2 {
        let Err(error) = session.execute(query) else {
            panic!("PROFILE must not bypass grouped workspace admission");
        };
        assert!(error.to_string().contains("memory"), "{error}");
        assert_eq!(manager.allocated(), baseline + held.size());
        assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
    }
    drop(held);
    assert_eq!(manager.allocated(), baseline);
    for pressured in [false, true] {
        let held = pressured.then(|| {
            manager
                .try_allocate(
                    manager.config().budget * 9 / 10 - baseline,
                    MemoryRegion::ExecutionBuffers,
                )
                .unwrap()
        });
        let held_total = manager.allocated();
        let result = session.execute(query).unwrap();
        assert_profile_operator_rows(&result, "HashAggregate", 2051);
        drop(result);
        assert_eq!(manager.allocated(), held_total);
        assert!(std::fs::read_dir(root.path()).unwrap().next().is_none());
        drop(held);
        assert_eq!(manager.allocated(), baseline);
    }
}

#[cfg(feature = "spill")]
#[test]
fn grouped_collect_distinct_reloads_nested_values_in_cold_and_cached_plans() {
    use grafeo_common::memory::buffer::{MemoryRegion, PressureLevel};
    use grafeo_common::utils::error::Error;

    const BUDGET: usize = 64 * 1024 * 1024;
    const QUERY: &str = "UNWIND range(0, 4095) AS x \
        RETURN x % 512 AS bucket, \
        collect(DISTINCT [CASE WHEN x < 1024 THEN 0 WHEN x < 2048 THEN 1 ELSE 2 END, 'payload']) AS collected";
    let root = tempfile::tempdir().unwrap();
    let spill_path = root.path().join("spill");
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory()
            .with_memory_limit(BUDGET)
            .with_spill_path(&spill_path)
            .without_query_timeout(),
    )
    .unwrap();
    let manager = db.buffer_manager();
    let baseline = manager.allocated();
    let held = manager
        .try_allocate(BUDGET * 9 / 10 - baseline, MemoryRegion::ExecutionBuffers)
        .unwrap();
    let held_total = manager.allocated();
    assert_eq!(manager.pressure_level(), PressureLevel::High);
    let session = db.session();

    // An I/O failure proves that this query really reaches the spill path.
    std::fs::write(&spill_path, b"blocked spill directory").unwrap();
    assert!(matches!(session.execute(QUERY), Err(Error::Io(_))));
    assert_eq!(manager.allocated(), held_total);
    std::fs::remove_file(&spill_path).unwrap();
    db.query_cache().clear();

    for cached in [false, true] {
        let before = db.query_cache().stats();
        let result = session.execute(QUERY).unwrap();
        if cached {
            assert!(db.query_cache().stats().optimized_hits > before.optimized_hits);
        }
        let mut rows = result.rows().to_vec();
        rows.sort_by_key(|row| row[0].as_int64().unwrap());
        assert_eq!(rows.len(), 512);
        for (bucket, row) in rows.iter().enumerate() {
            assert_eq!(row[0], Value::Int64(i64::try_from(bucket).unwrap()));
            let Value::List(values) = &row[1] else {
                panic!("COLLECT returns a list");
            };
            assert_eq!(values.len(), 3);
            let mut identities = Vec::new();
            for value in values.iter() {
                let Value::List(pair) = value else {
                    panic!("COLLECT preserves each nested list");
                };
                assert_eq!(pair.len(), 2);
                assert_eq!(pair[1], Value::from("payload"));
                identities.push(pair[0].as_int64().unwrap());
            }
            identities.sort_unstable();
            assert_eq!(identities, [0, 1, 2]);
        }
        drop(rows);
        assert!(manager.allocated() > held_total);
        drop(result);
        assert_eq!(manager.allocated(), held_total);
        assert_eq!(manager.pressure_level(), PressureLevel::High);
        assert!(std::fs::read_dir(&spill_path).unwrap().next().is_none());
    }
    drop(held);
    assert_eq!(manager.allocated(), baseline);
}
