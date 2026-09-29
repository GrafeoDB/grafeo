//! Targeted tests for planner edge cases: expression conversion, filter
//! expressions, join planning, aggregate + GROUP BY, UNWIND combos, and
//! subquery scoping.
//!
//! Focuses on real-world query planning bugs (wrong results, crashes)
//! rather than line coverage.
//!
//! ```bash
//! cargo test -p grafeo-engine --test planner_coverage --all-features
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Fixtures
// ============================================================================

/// Creates a social graph: 4 Person nodes (Alix, Gus, Vincent, Jules) in
/// Amsterdam/Berlin/Paris, plus 2 Company nodes. Edges: KNOWS, WORKS_AT.
fn social_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alix = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Alix".into())),
                ("age", Value::Int64(30)),
                ("city", Value::String("Amsterdam".into())),
                ("score", Value::Float64(7.5)),
            ],
        )
        .unwrap();
    let gus = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Gus".into())),
                ("age", Value::Int64(25)),
                ("city", Value::String("Berlin".into())),
                ("score", Value::Float64(8.2)),
            ],
        )
        .unwrap();
    let vincent = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Vincent".into())),
                ("age", Value::Int64(40)),
                ("city", Value::String("Paris".into())),
                ("score", Value::Float64(6.0)),
            ],
        )
        .unwrap();
    let jules = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Jules".into())),
                ("age", Value::Int64(35)),
                ("city", Value::String("Amsterdam".into())),
                ("score", Value::Float64(9.1)),
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
    let startup = session
        .create_node_with_props(
            &["Company"],
            [
                ("name", Value::String("Startup".into())),
                ("founded", Value::Int64(2020)),
            ],
        )
        .unwrap();

    // KNOWS: Alix -> Gus, Alix -> Vincent, Gus -> Jules, Vincent -> Jules
    session.create_edge(alix, gus, "KNOWS").unwrap();
    session.create_edge(alix, vincent, "KNOWS").unwrap();
    session.create_edge(gus, jules, "KNOWS").unwrap();
    session.create_edge(vincent, jules, "KNOWS").unwrap();

    // WORKS_AT: Alix -> TechCorp, Gus -> TechCorp, Vincent -> Startup, Jules -> Startup
    session.create_edge(alix, techcorp, "WORKS_AT").unwrap();
    session.create_edge(gus, techcorp, "WORKS_AT").unwrap();
    session.create_edge(vincent, startup, "WORKS_AT").unwrap();
    session.create_edge(jules, startup, "WORKS_AT").unwrap();

    db
}

/// Creates a chain graph: Alix -> Gus -> Vincent -> Jules (via FOLLOWS edges)
/// with weight properties on edges.
fn chain_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alix = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Alix".into())),
                ("rank", Value::Int64(1)),
            ],
        )
        .unwrap();
    let gus = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Gus".into())),
                ("rank", Value::Int64(2)),
            ],
        )
        .unwrap();
    let vincent = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Vincent".into())),
                ("rank", Value::Int64(3)),
            ],
        )
        .unwrap();
    let jules = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Jules".into())),
                ("rank", Value::Int64(4)),
            ],
        )
        .unwrap();

    session
        .create_edge_with_props(alix, gus, "FOLLOWS", [("weight", Value::Float64(1.0))])
        .unwrap();
    session
        .create_edge_with_props(gus, vincent, "FOLLOWS", [("weight", Value::Float64(2.0))])
        .unwrap();
    session
        .create_edge_with_props(vincent, jules, "FOLLOWS", [("weight", Value::Float64(3.0))])
        .unwrap();

    db
}

/// Creates nodes with sparse/nullable properties for NULL testing.
fn sparse_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("alpha".into())),
                ("val", Value::Int64(10)),
                ("tag", Value::String("x".into())),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("beta".into())),
                // val is missing (NULL)
                ("tag", Value::String("x".into())),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("gamma".into())),
                ("val", Value::Int64(30)),
                // tag is missing (NULL)
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("delta".into())),
                ("val", Value::Int64(20)),
                ("tag", Value::String("y".into())),
            ],
        )
        .unwrap();

    db
}

// ============================================================================
// 1. Expression conversion edge cases: nested AND/OR, NOT, IS NULL, coercion
// ============================================================================

#[test]
fn nested_and_or_filter_produces_correct_results() {
    let db = social_graph();
    let session = db.session();

    // (city = 'Amsterdam' AND age > 25) OR city = 'Paris'
    // Should match: Alix (Amsterdam, 30), Jules (Amsterdam, 35), Vincent (Paris, 40)
    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE (n.city = 'Amsterdam' AND n.age > 25) OR n.city = 'Paris' \
             RETURN n.name ORDER BY n.name",
        )
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["Alix", "Jules", "Vincent"]);
}

#[test]
fn deeply_nested_boolean_logic_not_and_or() {
    let db = social_graph();
    let session = db.session();

    // NOT (city = 'Amsterdam' OR city = 'Paris') should leave only Berlin (Gus)
    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE NOT (n.city = 'Amsterdam' OR n.city = 'Paris') \
             RETURN n.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
}

#[test]
fn is_null_filter_finds_missing_properties() {
    let db = sparse_graph();
    let session = db.session();

    // beta has no val property, so val IS NULL should match
    let result = session
        .execute("MATCH (i:Item) WHERE i.val IS NULL RETURN i.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("beta".into()));
}

#[test]
fn is_not_null_filter_excludes_missing_properties() {
    let db = sparse_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (i:Item) WHERE i.val IS NOT NULL RETURN i.name ORDER BY i.name")
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["alpha", "delta", "gamma"]);
}

#[test]
fn int_vs_float_comparison_coercion() {
    let db = social_graph();
    let session = db.session();

    // age (Int64) compared to float literal 30.0
    // Should match Alix (age=30) since 30 == 30.0
    let result = session
        .execute("MATCH (n:Person) WHERE n.age = 30.0 RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
}

#[test]
fn int_vs_float_less_than_coercion() {
    let db = social_graph();
    let session = db.session();

    // age (Int64) < 30.5 should include Alix (30) and Gus (25)
    let result = session
        .execute("MATCH (n:Person) WHERE n.age < 30.5 RETURN n.name ORDER BY n.name")
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["Alix", "Gus"]);
}

#[test]
fn case_when_in_where_clause() {
    let db = social_graph();
    let session = db.session();

    // Use CASE in a WHERE predicate: filter only senior people (age > 30)
    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE CASE WHEN n.age > 30 THEN true ELSE false END = true \
             RETURN n.name ORDER BY n.name",
        )
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["Jules", "Vincent"]);
}

#[test]
fn case_with_multiple_when_branches() {
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             RETURN n.name, \
             CASE \
                 WHEN n.age < 30 THEN 'young' \
                 WHEN n.age < 35 THEN 'mid' \
                 ELSE 'senior' \
             END AS bracket \
             ORDER BY n.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 4);
    // Alix=30 -> mid, Gus=25 -> young, Jules=35 -> senior, Vincent=40 -> senior
    let brackets: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[1] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(brackets, vec!["mid", "young", "senior", "senior"]);
}

// ============================================================================
// 2. Join planning: multi-hop, shared variables, OPTIONAL MATCH
// ============================================================================

#[test]
fn two_hop_pattern_produces_correct_paths() {
    let db = social_graph();
    let session = db.session();

    // (a)-[:KNOWS]->(b)-[:KNOWS]->(c) starting from Alix
    // Alix->Gus->Jules, Alix->Vincent->Jules
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS]->(c) \
             RETURN b.name, c.name ORDER BY b.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    let paths: Vec<(String, String)> = result
        .rows()
        .iter()
        .map(|r| {
            let b = match &r[0] {
                Value::String(s) => s.to_string(),
                other => panic!("expected string, got {other:?}"),
            };
            let c = match &r[1] {
                Value::String(s) => s.to_string(),
                other => panic!("expected string, got {other:?}"),
            };
            (b, c)
        })
        .collect();
    assert_eq!(
        paths,
        vec![
            ("Gus".to_string(), "Jules".to_string()),
            ("Vincent".to_string(), "Jules".to_string()),
        ]
    );
}

#[test]
fn three_hop_chain_traversal() {
    let db = chain_graph();
    let session = db.session();

    // Full chain: Alix -> Gus -> Vincent -> Jules
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'})-[:FOLLOWS]->(b)-[:FOLLOWS]->(c)-[:FOLLOWS]->(d) \
             RETURN a.name, b.name, c.name, d.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("Gus".into()));
    assert_eq!(result.rows()[0][2], Value::String("Vincent".into()));
    assert_eq!(result.rows()[0][3], Value::String("Jules".into()));
}

#[test]
fn optional_match_produces_nulls_for_missing_edges() {
    let db = social_graph();
    let session = db.session();

    // Jules has no outgoing KNOWS edges, so OPTIONAL MATCH should produce NULL
    let result = session
        .execute(
            "MATCH (n:Person {name: 'Jules'}) \
             OPTIONAL MATCH (n)-[:KNOWS]->(m) \
             RETURN n.name, m.name AS friend",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Jules".into()));
    assert_eq!(result.rows()[0][1], Value::Null);
}

#[test]
fn optional_match_mixed_with_regular_match() {
    let db = social_graph();
    let session = db.session();

    // All persons: some have MANAGES edges (none do), so friend_count should be null for MANAGES
    let result = session
        .execute(
            "MATCH (n:Person) \
             OPTIONAL MATCH (n)-[:MANAGES]->(m) \
             RETURN n.name, m.name AS managed \
             ORDER BY n.name",
        )
        .unwrap();

    // All 4 persons appear, all with NULL managed
    assert_eq!(result.rows().len(), 4);
    for row in result.rows() {
        assert_eq!(row[1], Value::Null, "No one manages anyone");
    }
}

#[test]
fn shared_variable_across_match_clauses() {
    let db = social_graph();
    let session = db.session();

    // Two patterns sharing variable 'n': Person who KNOWS someone AND WORKS_AT somewhere
    let result = session
        .execute(
            "MATCH (n:Person)-[:KNOWS]->(friend) \
             MATCH (n)-[:WORKS_AT]->(company) \
             RETURN DISTINCT n.name ORDER BY n.name",
        )
        .unwrap();

    // Alix, Gus, Vincent all have both KNOWS and WORKS_AT edges
    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["Alix", "Gus", "Vincent"]);
}

// ============================================================================
// 3. Aggregate + GROUP BY planning
// ============================================================================

#[test]
fn group_by_on_property_expression() {
    let db = social_graph();
    let session = db.session();

    // GROUP BY on city (a property, not a variable)
    let result = session
        .execute(
            "MATCH (n:Person) \
             RETURN n.city AS city, COUNT(*) AS cnt \
             ORDER BY city",
        )
        .unwrap();

    // Amsterdam=2, Berlin=1, Paris=1
    assert_eq!(result.rows().len(), 3);

    let cities: Vec<(&str, i64)> = result
        .rows()
        .iter()
        .map(|r| {
            let city = match &r[0] {
                Value::String(s) => s.as_str(),
                other => panic!("expected string, got {other:?}"),
            };
            let cnt = match &r[1] {
                Value::Int64(n) => *n,
                other => panic!("expected int, got {other:?}"),
            };
            (city, cnt)
        })
        .collect();
    assert_eq!(cities, vec![("Amsterdam", 2), ("Berlin", 1), ("Paris", 1)]);
}

#[test]
fn having_with_complex_predicate() {
    let db = social_graph();
    let session = db.session();

    // HAVING COUNT(*) > 1: only Amsterdam has 2 people
    let result = session
        .execute(
            "MATCH (n:Person) \
             RETURN n.city AS city, COUNT(*) AS cnt \
             HAVING cnt > 1",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(2));
}

#[test]
fn aggregate_in_order_by() {
    let db = social_graph();
    let session = db.session();

    // ORDER BY count, then city for stable sort
    let result = session
        .execute(
            "MATCH (n:Person) \
             RETURN n.city AS city, COUNT(*) AS cnt \
             ORDER BY cnt DESC, city",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // Amsterdam=2 first, then Berlin=1, Paris=1 alphabetically
    assert_eq!(result.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(2));
}

#[test]
fn multiple_aggregates_in_single_query() {
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             RETURN COUNT(*) AS cnt, MIN(n.age) AS youngest, MAX(n.age) AS oldest, AVG(n.age) AS avg_age",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(4));
    assert_eq!(result.rows()[0][1], Value::Int64(25));
    assert_eq!(result.rows()[0][2], Value::Int64(40));
    // avg(25,30,35,40) = 32.5
    match &result.rows()[0][3] {
        Value::Float64(v) => assert!((v - 32.5).abs() < 0.01, "expected 32.5, got {v}"),
        other => panic!("expected float, got {other:?}"),
    }
}

#[test]
fn group_by_with_sum_and_collect() {
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             RETURN n.city AS city, SUM(n.age) AS total_age, COLLECT(n.name) AS names \
             ORDER BY city",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // Amsterdam: Alix(30) + Jules(35) = 65
    assert_eq!(result.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(65));
}

// ============================================================================
// 4. UNWIND + pattern combinations
// ============================================================================

#[test]
fn unwind_with_where_filter() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute(
            "UNWIND [1, 2, 3, 4, 5] AS x \
             WHERE x > 3 \
             RETURN x ORDER BY x",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::Int64(4));
    assert_eq!(result.rows()[1][0], Value::Int64(5));
}

#[test]
fn unwind_into_match_pattern() {
    let db = social_graph();
    let session = db.session();

    // UNWIND a list of names, then match persons by name
    let result = session
        .execute(
            "UNWIND ['Alix', 'Vincent'] AS target_name \
             MATCH (n:Person {name: target_name}) \
             RETURN n.name, n.age ORDER BY n.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(30));
    assert_eq!(result.rows()[1][0], Value::String("Vincent".into()));
    assert_eq!(result.rows()[1][1], Value::Int64(40));
}

#[test]
fn unwind_empty_list_produces_no_rows() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("UNWIND [] AS x RETURN x").unwrap();
    assert_eq!(result.rows().len(), 0);
}

#[test]
fn unwind_nested_list() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // UNWIND a list of lists, then UNWIND again
    let result = session
        .execute(
            "UNWIND [[1, 2], [3, 4]] AS sublist \
             UNWIND sublist AS x \
             RETURN x ORDER BY x",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 4);
    assert_eq!(result.rows()[0][0], Value::Int64(1));
    assert_eq!(result.rows()[1][0], Value::Int64(2));
    assert_eq!(result.rows()[2][0], Value::Int64(3));
    assert_eq!(result.rows()[3][0], Value::Int64(4));
}

// ============================================================================
// 5. Subquery scoping: CALL { ... }
// ============================================================================

#[test]
fn call_subquery_with_outer_variable() {
    let db = social_graph();
    let session = db.session();

    // Correlated subquery: count friends for each person
    let result = session
        .execute(
            "MATCH (n:Person) \
             CALL { WITH n MATCH (n)-[:KNOWS]->(m) RETURN COUNT(*) AS friend_count } \
             RETURN n.name, friend_count \
             ORDER BY n.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 4);
    // Alix: 2 friends, Gus: 1, Jules: 0, Vincent: 1
    let data: Vec<(&str, i64)> = result
        .rows()
        .iter()
        .map(|r| {
            let name = match &r[0] {
                Value::String(s) => s.as_str(),
                other => panic!("expected string, got {other:?}"),
            };
            let cnt = match &r[1] {
                Value::Int64(n) => *n,
                other => panic!("expected int, got {other:?}"),
            };
            (name, cnt)
        })
        .collect();
    assert_eq!(
        data,
        vec![("Alix", 2), ("Gus", 1), ("Jules", 0), ("Vincent", 1)]
    );
}

#[test]
fn uncorrelated_call_subquery() {
    let db = social_graph();
    let session = db.session();

    // Uncorrelated: count all persons, return alongside each person
    let result = session
        .execute(
            "MATCH (n:Person) \
             CALL { MATCH (m:Person) RETURN COUNT(*) AS total } \
             RETURN n.name, total \
             ORDER BY n.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 4);
    for row in result.rows() {
        assert_eq!(row[1], Value::Int64(4), "total should be 4 for all rows");
    }
}

#[test]
fn call_subquery_aggregation_per_outer_row() {
    let db = social_graph();
    let session = db.session();

    // For each person, collect the names of their friends
    let result = session
        .execute(
            "MATCH (n:Person) \
             CALL { WITH n MATCH (n)-[:KNOWS]->(m) RETURN COLLECT(m.name) AS friends } \
             RETURN n.name, friends \
             ORDER BY n.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 4);
    // Alix should have 2 friends
    match &result.rows()[0][1] {
        Value::List(items) => assert_eq!(items.len(), 2, "Alix should have 2 friends"),
        other => panic!("expected list, got {other:?}"),
    }
    // Jules should have 0 friends (empty list)
    match &result.rows()[2][1] {
        Value::List(items) => assert_eq!(items.len(), 0, "Jules should have 0 friends"),
        other => panic!("expected list, got {other:?}"),
    }
}

// ============================================================================
// 6. Complex filter combinations
// ============================================================================

#[test]
fn filter_with_arithmetic_expression() {
    let db = social_graph();
    let session = db.session();

    // Filter using an arithmetic expression: age * 2 > 70
    // Matches: Jules (35*2=70, not >70), Vincent (40*2=80, yes)
    let result = session
        .execute("MATCH (n:Person) WHERE n.age * 2 > 70 RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
}

#[test]
fn filter_with_string_operations() {
    let db = social_graph();
    let session = db.session();

    // STARTS WITH filter
    let result = session
        .execute("MATCH (n:Person) WHERE n.name STARTS WITH 'V' RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
}

#[test]
fn filter_with_in_list() {
    let db = social_graph();
    let session = db.session();

    // IN list filter
    let result = session
        .execute(
            "MATCH (n:Person) WHERE n.city IN ['Amsterdam', 'Paris'] \
             RETURN n.name ORDER BY n.name",
        )
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["Alix", "Jules", "Vincent"]);
}

#[test]
fn xor_boolean_filter() {
    let db = social_graph();
    let session = db.session();

    // XOR: exactly one of the conditions is true
    // city = 'Amsterdam' XOR age > 35: Alix (Ams, 30) = true XOR false = true,
    // Gus (Ber, 25) = false XOR false = false, Vincent (Par, 40) = false XOR true = true,
    // Jules (Ams, 35) = true XOR false = true
    let result = session
        .execute(
            "MATCH (n:Person) \
             WHERE n.city = 'Amsterdam' XOR n.age > 35 \
             RETURN n.name ORDER BY n.name",
        )
        .unwrap();

    let names: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec!["Alix", "Jules", "Vincent"]);
}

// ============================================================================
// 7. Edge cases in return projection and type handling
// ============================================================================

#[test]
fn return_expression_with_type_function() {
    let db = social_graph();
    let session = db.session();

    // type() on edges returns the relationship type
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'})-[r]->(b) \
             RETURN type(r) AS rel_type ORDER BY rel_type",
        )
        .unwrap();

    let types: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    // Alix has 2 KNOWS + 1 WORKS_AT edges
    assert_eq!(types, vec!["KNOWS", "KNOWS", "WORKS_AT"]);
}

#[test]
fn return_labels_function() {
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN labels(n) AS lbls")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::List(labels) => {
            assert!(
                labels.contains(&Value::String("Person".into())),
                "Should contain Person label"
            );
        }
        other => panic!("expected list, got {other:?}"),
    }
}

#[test]
fn distinct_on_computed_expression() {
    let db = social_graph();
    let session = db.session();

    // DISTINCT on city (computed from property access)
    let result = session
        .execute("MATCH (n:Person) RETURN DISTINCT n.city AS city ORDER BY city")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    let cities: Vec<&str> = result
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.as_str(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(cities, vec!["Amsterdam", "Berlin", "Paris"]);
}

#[test]
fn null_safe_aggregation_with_group_by() {
    let db = sparse_graph();
    let session = db.session();

    // GROUP BY tag (some items have NULL tag), aggregate val
    let result = session
        .execute(
            "MATCH (i:Item) \
             RETURN i.tag AS tag, SUM(i.val) AS total, COUNT(*) AS cnt \
             ORDER BY tag",
        )
        .unwrap();

    // Groups: NULL (gamma: val=30), 'x' (alpha: val=10, beta: val=NULL), 'y' (delta: val=20)
    // NULL group should appear (with tag=NULL), SUM should skip nulls
    assert!(
        result.rows().len() >= 2,
        "Should have at least x and y groups"
    );

    // Find the 'x' group
    let x_row = result
        .rows()
        .iter()
        .find(|r| r[0] == Value::String("x".into()));
    if let Some(row) = x_row {
        // SUM of alpha(10) + beta(NULL) = 10 (NULL skipped)
        assert_eq!(row[1], Value::Int64(10), "SUM should skip NULL values");
        assert_eq!(
            row[2],
            Value::Int64(2),
            "COUNT(*) includes NULL-valued rows"
        );
    }
}

#[test]
fn count_star_vs_count_property_with_nulls() {
    let db = sparse_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (i:Item) \
             RETURN COUNT(*) AS total, COUNT(i.val) AS non_null_vals",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(4),
        "COUNT(*) counts all rows"
    );
    assert_eq!(
        result.rows()[0][1],
        Value::Int64(3),
        "COUNT(i.val) skips NULL"
    );
}
