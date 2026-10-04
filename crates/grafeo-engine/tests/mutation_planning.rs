//! Integration tests for mutation-related planning paths.
//!
//! Targets low-coverage areas in `planner/mutation.rs`:
//! UNWIND, MERGE, CREATE edge/path, DELETE, SET property, label ops,
//! OPTIONAL MATCH (left join), CALL PROCEDURE.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test mutation_planning
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Fixtures
// ============================================================================

/// Creates 3 Person + 1 Company nodes, 3 KNOWS + 2 WORKS_AT edges.
fn create_social_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alix = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Alix".into())),
                ("age", Value::Int64(30)),
                ("city", Value::String("NYC".into())),
            ],
        )
        .unwrap();
    let gus = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Gus".into())),
                ("age", Value::Int64(25)),
                ("city", Value::String("NYC".into())),
            ],
        )
        .unwrap();
    let harm = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Harm".into())),
                ("age", Value::Int64(35)),
                ("city", Value::String("London".into())),
            ],
        )
        .unwrap();

    let techcorp = session
        .create_node_with_props(&["Company"], [("name", Value::String("TechCorp".into()))])
        .unwrap();

    session.create_edge(alix, gus, "KNOWS").unwrap();
    session.create_edge(alix, harm, "KNOWS").unwrap();
    session.create_edge(gus, harm, "KNOWS").unwrap();
    session.create_edge(alix, techcorp, "WORKS_AT").unwrap();
    session.create_edge(gus, techcorp, "WORKS_AT").unwrap();

    db
}

// ============================================================================
// UNWIND: covers plan_unwind() with Empty input, prior input, property refs
// ============================================================================

#[test]
fn test_unwind_literal_list() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

    assert_eq!(result.rows().len(), 3);
    let values: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(values.contains(&&Value::Int64(1)));
    assert!(values.contains(&&Value::Int64(2)));
    assert!(values.contains(&&Value::Int64(3)));
}

#[test]
fn test_unwind_after_match() {
    let db = create_social_network();
    let session = db.session();

    // GQL: MATCH and UNWIND are both pre-WHERE clauses, so UNWIND goes before WHERE
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) UNWIND [10, 20] AS x RETURN n.name, x")
        .unwrap();

    // Alix x 2 unwind elements = 2 rows
    assert_eq!(result.rows().len(), 2);
    for row in result.rows() {
        assert_eq!(row[0], Value::String("Alix".into()));
    }
}

#[test]
fn test_unwind_with_strings() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("UNWIND ['a', 'b', 'c'] AS letter RETURN letter")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    let values: Vec<&Value> = result.rows().iter().map(|r| &r[0]).collect();
    assert!(values.contains(&&Value::String("a".into())));
    assert!(values.contains(&&Value::String("b".into())));
    assert!(values.contains(&&Value::String("c".into())));
}

#[cfg(feature = "cypher")]
#[test]
fn test_unwind_create() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Cypher syntax for UNWIND + CREATE
    session
        .execute_cypher("UNWIND [1, 2, 3] AS x CREATE (:Number {val: x})")
        .unwrap();

    let result = session
        .execute("MATCH (n:Number) RETURN n.val ORDER BY n.val")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // Verify actual property values, not just row count
    assert_eq!(result.rows()[0][0], Value::Int64(1));
    assert_eq!(result.rows()[1][0], Value::Int64(2));
    assert_eq!(result.rows()[2][0], Value::Int64(3));
}

#[test]
fn test_unwind_create_map_property_access() {
    // Regression test: UNWIND with map list + property access in CREATE
    // Previously all properties resolved to NULL (bug in plan_create_node)
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute(
            "UNWIND [{id: 'u1', name: 'Gus'}, {id: 'u2', name: 'Harm'}] AS props \
             CREATE (:Test {id: props.id, name: props.name})",
        )
        .unwrap();

    let result = session
        .execute("MATCH (n:Test) RETURN n.id AS id, n.name AS name ORDER BY n.id")
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("u1".into()));
    assert_eq!(result.rows()[0][1], Value::String("Gus".into()));
    assert_eq!(result.rows()[1][0], Value::String("u2".into()));
    assert_eq!(result.rows()[1][1], Value::String("Harm".into()));
}

#[test]
fn test_unwind_param_create_map_property_access() {
    // Same as above but with parameter substitution ($nodes instead of literal list)
    use std::collections::BTreeMap;
    use std::collections::HashMap;
    use std::sync::Arc;

    use grafeo_common::types::PropertyKey;

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let nodes = Value::List(Arc::from(vec![
        Value::Map(Arc::new(BTreeMap::from([
            (PropertyKey::new("id"), Value::String("u1".into())),
            (PropertyKey::new("name"), Value::String("Gus".into())),
        ]))),
        Value::Map(Arc::new(BTreeMap::from([
            (PropertyKey::new("id"), Value::String("u2".into())),
            (PropertyKey::new("name"), Value::String("Harm".into())),
        ]))),
    ]));

    let params = HashMap::from([("nodes".to_string(), nodes)]);

    session
        .execute_with_params(
            "UNWIND $nodes AS props CREATE (:Test {id: props.id, name: props.name})",
            params,
        )
        .unwrap();

    let result = session
        .execute("MATCH (n:Test) RETURN n.id AS id, n.name AS name ORDER BY n.id")
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("u1".into()));
    assert_eq!(result.rows()[0][1], Value::String("Gus".into()));
    assert_eq!(result.rows()[1][0], Value::String("u2".into()));
    assert_eq!(result.rows()[1][1], Value::String("Harm".into()));
}

#[test]
fn test_for_with_ordinality() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("FOR x IN [10, 20, 30] WITH ORDINALITY i RETURN x, i")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // ORDINALITY is 1-based
    let mut pairs: Vec<(i64, i64)> = result
        .rows()
        .iter()
        .map(|r| {
            let x = match &r[0] {
                Value::Int64(v) => *v,
                _ => panic!("Expected Int64 for x"),
            };
            let i = match &r[1] {
                Value::Int64(v) => *v,
                _ => panic!("Expected Int64 for i"),
            };
            (x, i)
        })
        .collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(10, 1), (20, 2), (30, 3)]);
}

#[test]
fn test_for_with_offset() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("FOR x IN [10, 20, 30] WITH OFFSET idx RETURN x, idx")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // OFFSET is 0-based
    let mut pairs: Vec<(i64, i64)> = result
        .rows()
        .iter()
        .map(|r| {
            let x = match &r[0] {
                Value::Int64(v) => *v,
                _ => panic!("Expected Int64 for x"),
            };
            let idx = match &r[1] {
                Value::Int64(v) => *v,
                _ => panic!("Expected Int64 for idx"),
            };
            (x, idx)
        })
        .collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(10, 0), (20, 1), (30, 2)]);
}

#[test]
fn test_for_without_ordinality_or_offset() {
    // Ensure basic FOR without WITH still works
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("FOR x IN [1, 2, 3] RETURN x").unwrap();

    assert_eq!(result.rows().len(), 3);
}

// ============================================================================
// MERGE: covers plan_merge() Empty/non-Empty input, on_create, on_match
// ============================================================================

#[test]
fn test_merge_creates_when_not_exists() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("MERGE (n:Animal {species: 'Cat'}) RETURN n.species")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Cat".into()));
    assert_eq!(db.node_count(), 1);
}

#[test]
fn test_merge_matches_existing() {
    let db = create_social_network();
    let session = db.session();

    let before = db.node_count();

    let result = session
        .execute("MERGE (n:Person {name: 'Alix'}) RETURN n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    // No new node created
    assert_eq!(db.node_count(), before);
}

#[test]
fn test_merge_on_create_set() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute(
            "MERGE (n:Person {name: 'NewGuy'}) ON CREATE SET n.created = true RETURN n.name, n.created",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("NewGuy".into()));
    assert_eq!(result.rows()[0][1], Value::Bool(true));
}

#[test]
fn test_merge_on_match_set() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute(
            "MERGE (n:Person {name: 'Alix'}) ON MATCH SET n.found = true RETURN n.name, n.found",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::Bool(true));
}

// ============================================================================
// CREATE: covers plan_create_node, plan_create_edge, try_fold_expression
// ============================================================================

#[test]
fn test_create_node_with_list_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("CREATE (:Tag {names: ['rust', 'graph', 'db']})")
        .unwrap();

    let result = session.execute("MATCH (t:Tag) RETURN t.names").unwrap();

    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::List(items) => assert_eq!(items.len(), 3),
        other => panic!("expected list, got {:?}", other),
    }
}

#[cfg(feature = "cypher")]
#[test]
fn test_create_edge_named_variable() {
    let db = create_social_network();
    let session = db.session();

    // Cypher: MATCH + CREATE edge with named variable
    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             CREATE (a)-[r:LIKES]->(b) RETURN type(r)",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("LIKES".into()));
}

#[cfg(feature = "cypher")]
#[test]
fn test_create_edge_anonymous() {
    let db = create_social_network();
    let session = db.session();

    let edges_before = db.edge_count();

    session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             CREATE (a)-[:LIKES]->(b)",
        )
        .unwrap();

    assert_eq!(db.edge_count(), edges_before + 1);
}

#[test]
fn test_create_path_with_new_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create two nodes and an edge using the programmatic API
    // (exercises CreateNodeOp and CreateEdgeOp through a different code path)
    let paris = session
        .create_node_with_props(&["City"], [("name", Value::String("Paris".into()))])
        .unwrap();
    let france = session
        .create_node_with_props(&["Country"], [("name", Value::String("France".into()))])
        .unwrap();
    session.create_edge(paris, france, "IN").unwrap();

    assert_eq!(db.node_count(), 2);
    assert_eq!(db.edge_count(), 1);

    // Verify the path can be queried via GQL
    let result = session
        .execute("MATCH (c:City)-[:IN]->(co:Country) RETURN c.name, co.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Paris".into()));
    assert_eq!(result.rows()[0][1], Value::String("France".into()));
}

// ============================================================================
// DELETE: covers plan_delete_node, plan_delete_edge, detach path
// ============================================================================

#[test]
fn test_delete_node_by_match() {
    let db = create_social_network();
    let session = db.session();

    let before = db.node_count();

    // Delete Harm (who has KNOWS edges)
    session
        .execute("MATCH (n:Person) WHERE n.name = 'Harm' DETACH DELETE n")
        .unwrap();

    assert!(db.node_count() < before);
}

#[test]
fn test_delete_node_reduces_count() {
    let db = create_social_network();

    let before = db.node_count();

    // DETACH DELETE a single node removes it and its edges
    let session = db.session();
    session
        .execute("MATCH (n:Company {name: 'TechCorp'}) DETACH DELETE n")
        .unwrap();

    assert_eq!(db.node_count(), before - 1);
}

#[test]
fn test_detach_delete_all_nodes() {
    let db = create_social_network();
    let session = db.session();

    session.execute("MATCH (n) DETACH DELETE n").unwrap();

    assert_eq!(db.node_count(), 0);
    assert_eq!(db.edge_count(), 0);
}

// ============================================================================
// SET property: covers plan_set_property, expression_to_property_source
// ============================================================================

#[test]
fn test_set_property_literal() {
    let db = create_social_network();
    let session = db.session();

    // GQL: SET is processed before WHERE, so use pattern property match
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 31")
        .unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(31));
}

#[test]
fn test_set_property_string() {
    let db = create_social_network();
    let session = db.session();

    session
        .execute("MATCH (n:Person {name: 'Gus'}) SET n.city = 'Berlin'")
        .unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Gus'}) RETURN n.city")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Berlin".into()));
}

// ============================================================================
// Label operations: covers plan_add_label, plan_remove_label
// ============================================================================

#[test]
fn test_add_label_via_set() {
    let db = create_social_network();
    let session = db.session();

    // GQL: SET is processed before WHERE, so use pattern property match
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n:Employee")
        .unwrap();

    // Alix should now have both Person and Employee labels
    let result = session.execute("MATCH (n:Employee) RETURN n.name").unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
}

#[test]
fn test_remove_label() {
    let db = create_social_network();
    let session = db.session();

    // First add a label, then remove it (use pattern match, not WHERE)
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n:Temp")
        .unwrap();

    let result = session.execute("MATCH (n:Temp) RETURN n.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    session
        .execute("MATCH (n:Person {name: 'Alix'}) REMOVE n:Temp")
        .unwrap();

    let result = session.execute("MATCH (n:Temp) RETURN n.name").unwrap();
    assert!(result.rows().is_empty(), "{:?}", result.rows());
}

// ── Regression: SET/REMOVE label variable binding (#178, #182) ──────────────

#[test]
fn test_set_label_preserves_variable_binding() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let r = session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n:Employee RETURN n.name")
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

#[test]
fn test_remove_label_preserves_variable_binding() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person:Employee {name: 'Alix'})")
        .unwrap();
    let r = session
        .execute("MATCH (n:Person {name: 'Alix'}) REMOVE n:Employee RETURN n.name")
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

#[test]
fn test_count_star_after_set_label() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Node {id: '1'})").unwrap();
    session.execute("INSERT (:Node {id: '2'})").unwrap();
    session.execute("INSERT (:Node {id: '3'})").unwrap();
    let r = session
        .execute("MATCH (n:Node) SET n:Tagged RETURN count(*) AS cnt")
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::Int64(3));
}

#[test]
fn test_set_label_then_set_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n:Employee SET n.role = 'Engineer'")
        .unwrap();
    let r = session
        .execute("MATCH (n:Employee {name: 'Alix'}) RETURN n.role")
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Engineer".into()));
}

// ── Regression: phantom nodes from MATCH...CREATE edge (#181) ───────────────

#[test]
#[cfg(feature = "cypher")]
fn test_match_create_edge_no_phantom_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute_cypher("CREATE (:Person {name: 'Alix'})")
        .unwrap();
    session
        .execute_cypher("CREATE (:Person {name: 'Gus'})")
        .unwrap();

    let before = session
        .execute_cypher("MATCH (n) RETURN count(n) AS cnt")
        .unwrap();
    let count_before = match &before.rows()[0][0] {
        Value::Int64(n) => *n,
        other => panic!("expected Int64, got {other:?}"),
    };

    session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) CREATE (a)-[:KNOWS]->(b)",
        )
        .unwrap();

    let after = session
        .execute_cypher("MATCH (n) RETURN count(n) AS cnt")
        .unwrap();
    assert_eq!(
        after.rows()[0][0],
        Value::Int64(count_before),
        "phantom nodes created"
    );

    let edges = session
        .execute_cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS cnt")
        .unwrap();
    assert_eq!(edges.rows()[0][0], Value::Int64(1));
}

// ============================================================================
// OPTIONAL MATCH: covers plan_left_join
// ============================================================================

#[test]
fn test_optional_match_with_results() {
    let db = create_social_network();
    let session = db.session();

    // Alix works at TechCorp, use pattern property match (WHERE comes after OPTIONAL MATCH)
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'}) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             RETURN a.name, c.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("TechCorp".into()));
}

#[test]
fn test_optional_match_null_when_missing() {
    let db = create_social_network();
    let session = db.session();

    // Harm doesn't manage anyone, OPTIONAL MATCH should still produce a row
    // (LEFT JOIN preserves the left side even when right side has no matches)
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Harm'}) \
             OPTIONAL MATCH (a)-[:MANAGES]->(c:Company) \
             RETURN a, c",
        )
        .unwrap();

    // Should produce exactly 1 row: LEFT JOIN keeps Harm even without a MANAGES match
    assert_eq!(
        result.rows().len(),
        1,
        "OPTIONAL MATCH should return 1 row even when right side has no matches"
    );
}

// ============================================================================
// OPTIONAL MATCH WHERE pushdown: right-side predicates become join conditions
// ============================================================================

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_where_right_side_preserves_left() {
    // WHERE on right-side variable should not filter out left rows.
    // Harm has no WORKS_AT edge, so the OPTIONAL MATCH right side is empty.
    // The WHERE m.name = 'TechCorp' must NOT eliminate Harm's row.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             WHERE c.name = 'TechCorp' \
             RETURN a.name, c.name \
             ORDER BY a.name",
        )
        .unwrap();

    // All 3 persons should appear: Alix+TechCorp, Gus+TechCorp, Harm+null
    assert_eq!(
        result.rows().len(),
        3,
        "All persons should be preserved; WHERE on right side should not filter left rows"
    );

    // Alix -> TechCorp
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("TechCorp".into()));
    // Gus -> TechCorp
    assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[1][1], Value::String("TechCorp".into()));
    // Harm -> null (no WORKS_AT edge, preserved by left join)
    assert_eq!(result.rows()[2][0], Value::String("Harm".into()));
    assert_eq!(result.rows()[2][1], Value::Null);
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_where_left_side_filters_correctly() {
    // WHERE on left-side variable should filter left rows normally.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             WHERE a.city = 'NYC' \
             RETURN a.name, c.name \
             ORDER BY a.name",
        )
        .unwrap();

    // Only NYC persons: Alix+TechCorp, Gus+TechCorp (Harm is in London, filtered out)
    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_where_mixed_predicates() {
    // WHERE with both left-side and right-side predicates.
    // Left predicate (a.city = 'NYC') filters left rows.
    // Right predicate (c.name = 'TechCorp') becomes a join condition.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             WHERE a.city = 'NYC' AND c.name = 'TechCorp' \
             RETURN a.name, c.name \
             ORDER BY a.name",
        )
        .unwrap();

    // Only NYC persons: Alix+TechCorp, Gus+TechCorp
    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("TechCorp".into()));
    assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[1][1], Value::String("TechCorp".into()));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_where_right_property_no_match() {
    // WHERE right-side predicate that no right-side row satisfies.
    // All left rows should still appear with NULL right side.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             WHERE c.name = 'NonExistent' \
             RETURN a.name, c.name \
             ORDER BY a.name",
        )
        .unwrap();

    // All 3 persons should still appear, all with NULL company
    assert_eq!(
        result.rows().len(),
        3,
        "All persons preserved with NULL when no right match passes the filter"
    );
    for row in result.rows() {
        assert_eq!(row[1], Value::Null);
    }
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_where_cross_side_predicate() {
    // WHERE predicate referencing both left and right side variables.
    // m.age > n.age: find friends older than the person.
    // Since the right side also has `n` (from the pattern), this is classified
    // as a right-side predicate and pushed into the right filter.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
             WHERE b.age > a.age \
             RETURN a.name, b.name \
             ORDER BY a.name, b.name",
        )
        .unwrap();

    // Alix(30) knows Gus(25) and Harm(35). Only Harm is older -> Alix,Harm
    // Gus(25) knows Harm(35). Harm is older -> Gus,Harm
    // Harm(35) knows nobody (no outgoing KNOWS edges) -> Harm,null
    assert!(
        result.rows().len() >= 3,
        "Expected at least 3 rows: Alix+Harm, Gus+Harm, Harm+null, got {}",
        result.rows().len()
    );

    // Verify Harm appears with null (preserved by left join)
    let harm_rows: Vec<_> = result
        .rows()
        .iter()
        .filter(|r| r[0] == Value::String("Harm".into()))
        .collect();
    assert!(
        !harm_rows.is_empty(),
        "Harm should appear (no outgoing KNOWS edges, left join preserves)"
    );
    assert_eq!(
        harm_rows[0][1],
        Value::Null,
        "Harm has no outgoing KNOWS edges, right side should be NULL"
    );
}

#[test]
fn test_optional_match_gql_where_pushdown() {
    // Same test but using GQL syntax.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             WHERE c.name = 'TechCorp' \
             RETURN a.name, c.name \
             ORDER BY a.name",
        )
        .unwrap();

    assert_eq!(
        result.rows().len(),
        3,
        "GQL: all persons preserved with right-side WHERE pushdown"
    );
    assert_eq!(result.rows()[2][0], Value::String("Harm".into()));
    assert_eq!(result.rows()[2][1], Value::Null);
}

#[cfg(feature = "cypher")]
#[test]
fn test_chained_optional_matches() {
    // Two independent OPTIONAL MATCHHes.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Harm'}) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
             RETURN a.name, c.name, b.name",
        )
        .unwrap();

    // Harm has no WORKS_AT and no outgoing KNOWS edges
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Harm".into()));
    assert_eq!(result.rows()[0][1], Value::Null);
    assert_eq!(result.rows()[0][2], Value::Null);
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_is_null_check() {
    // Verify IS NULL works with OPTIONAL MATCH NULL values.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:MANAGES]->(c) \
             RETURN a.name, c IS NULL AS no_manages \
             ORDER BY a.name",
        )
        .unwrap();

    // Nobody has a MANAGES edge, so all rows should have c IS NULL = true
    assert_eq!(result.rows().len(), 3);
    for row in result.rows() {
        assert_eq!(row[1], Value::Bool(true));
    }
}

// ============================================================================
// OPTIONAL MATCH Phase 4: Chained OPTIONAL MATCH
// ============================================================================

#[cfg(feature = "cypher")]
#[test]
fn test_chained_independent_optional_matches_mixed() {
    // Two independent OPTIONAL MATCHHes where one matches and one does not.
    // Alix has WORKS_AT->TechCorp but no MANAGES edges.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alix'}) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             OPTIONAL MATCH (a)-[:MANAGES]->(m) \
             RETURN a.name, c.name, m",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("TechCorp".into()));
    assert_eq!(result.rows()[0][2], Value::Null);
}

#[cfg(feature = "cypher")]
#[test]
fn test_chained_dependent_optional_matches() {
    // Dependent optionals: second OPTIONAL depends on result of first.
    // OPTIONAL MATCH (a)-[:KNOWS]->(b), then OPTIONAL MATCH (b)-[:WORKS_AT]->(c).
    // Harm has no outgoing KNOWS, so b is NULL and c must also be NULL.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Harm'}) \
             OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
             OPTIONAL MATCH (b)-[:WORKS_AT]->(c:Company) \
             RETURN a.name, b.name, c.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Harm".into()));
    assert_eq!(result.rows()[0][1], Value::Null);
    assert_eq!(result.rows()[0][2], Value::Null);
}

#[cfg(feature = "cypher")]
#[test]
fn test_chained_dependent_optional_partial_match() {
    // Alix-KNOWS->Gus and Alix-KNOWS->Harm.
    // Gus has WORKS_AT->TechCorp, Harm does not.
    // So second OPTIONAL should match for Gus but NULL for Harm.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alix'}) \
             OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
             OPTIONAL MATCH (b)-[:WORKS_AT]->(c:Company) \
             RETURN a.name, b.name, c.name \
             ORDER BY b.name",
        )
        .unwrap();

    // Alix knows Gus (who works at TechCorp) and Harm (who doesn't)
    assert_eq!(result.rows().len(), 2);

    // Gus -> TechCorp
    assert_eq!(result.rows()[0][1], Value::String("Gus".into()));
    assert_eq!(result.rows()[0][2], Value::String("TechCorp".into()));

    // Harm -> null (no WORKS_AT edge)
    assert_eq!(result.rows()[1][1], Value::String("Harm".into()));
    assert_eq!(result.rows()[1][2], Value::Null);
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_with_node_label_filter() {
    // OPTIONAL MATCH with label on optional side node.
    // Create a Developer label for Gus only.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute_cypher(
            "CREATE (a:Person {name: 'Alix'}), \
             (b:Person:Developer {name: 'Gus'}), \
             (c:Person {name: 'Harm'}), \
             (a)-[:KNOWS]->(b), \
             (a)-[:KNOWS]->(c)",
        )
        .unwrap();

    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alix'}) \
             OPTIONAL MATCH (a)-[:KNOWS]->(m:Developer) \
             RETURN a.name, m.name",
        )
        .unwrap();

    // Only Gus is a Developer, so one row with Gus
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][1], Value::String("Gus".into()));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_with_vle() {
    // OPTIONAL MATCH with variable-length path.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute_cypher(
            "CREATE (a:Person {name: 'Alix'}), \
             (b:Person {name: 'Gus'}), \
             (c:Person {name: 'Harm'}), \
             (d:Person {name: 'Jules'}), \
             (a)-[:KNOWS]->(b), \
             (b)-[:KNOWS]->(c)",
        )
        .unwrap();

    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alix'}) \
             OPTIONAL MATCH (a)-[:KNOWS*1..2]->(m:Person) \
             RETURN a.name, m.name \
             ORDER BY m.name",
        )
        .unwrap();

    // Alix reaches Gus (1 hop) and Harm (2 hops). Jules is unreachable.
    assert!(
        result.rows().len() >= 2,
        "Should find at least Gus and Harm via VLE, got {}",
        result.rows().len()
    );

    let names: Vec<_> = result
        .rows()
        .iter()
        .filter_map(|r| match &r[1] {
            Value::String(s) => Some(s.as_str().to_owned()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"Gus".to_owned()));
    assert!(names.contains(&"Harm".to_owned()));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_with_vle_no_path() {
    // OPTIONAL MATCH with VLE where no path exists: should produce NULL.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute_cypher(
            "CREATE (a:Person {name: 'Alix'}), \
             (b:Person {name: 'Gus'})",
        )
        .unwrap();

    let result = session
        .execute_cypher(
            "MATCH (a:Person {name: 'Alix'}) \
             OPTIONAL MATCH (a)-[:KNOWS*1..3]->(m:Person) \
             RETURN a.name, m.name",
        )
        .unwrap();

    // No KNOWS edges, so m is NULL but Alix is preserved
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::Null);
}

#[test]
fn test_chained_independent_optional_gql() {
    // Same as Cypher chained test but in GQL syntax.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'}) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             OPTIONAL MATCH (a)-[:MANAGES]->(m) \
             RETURN a.name, c.name, m",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::String("TechCorp".into()));
    assert_eq!(result.rows()[0][2], Value::Null);
}

// ============================================================================
// OPTIONAL MATCH Phase 5: NULL Semantics
// ============================================================================

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_is_not_null() {
    // IS NOT NULL should correctly identify non-NULL optional values.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             RETURN a.name, c IS NOT NULL AS has_job \
             ORDER BY a.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // Alix has WORKS_AT -> true
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::Bool(true));
    // Gus has WORKS_AT -> true
    assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[1][1], Value::Bool(true));
    // Harm has no WORKS_AT -> false
    assert_eq!(result.rows()[2][0], Value::String("Harm".into()));
    assert_eq!(result.rows()[2][1], Value::Bool(false));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_coalesce() {
    // COALESCE should return the first non-NULL argument.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             RETURN a.name, COALESCE(c.name, 'unemployed') AS employer \
             ORDER BY a.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][1], Value::String("TechCorp".into()));
    assert_eq!(result.rows()[1][1], Value::String("TechCorp".into()));
    assert_eq!(result.rows()[2][1], Value::String("unemployed".into()));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_case_when() {
    // CASE WHEN with NULL from OPTIONAL MATCH.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             RETURN a.name, \
                    CASE WHEN c IS NOT NULL THEN c.name ELSE 'none' END AS employer \
             ORDER BY a.name",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][1], Value::String("TechCorp".into()));
    assert_eq!(result.rows()[1][1], Value::String("TechCorp".into()));
    assert_eq!(result.rows()[2][1], Value::String("none".into()));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_count_star_vs_count_expr() {
    // COUNT(*) should count all rows (including NULLs from OPTIONAL MATCH).
    // COUNT(c.name) should skip NULLs (property access on NULL node produces NULL).
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             RETURN COUNT(*) AS total, COUNT(c.name) AS with_job",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    // COUNT(*) = 3 (all persons)
    assert_eq!(result.rows()[0][0], Value::Int64(3));
    // COUNT(c.name) = 2 (only Alix and Gus have WORKS_AT with c.name = 'TechCorp')
    assert_eq!(result.rows()[0][1], Value::Int64(2));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_collect_skips_nulls() {
    // COLLECT should omit NULL values from the result list.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             RETURN COLLECT(c.name) AS employers",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::List(list) => {
            // Should only contain TechCorp entries (no NULLs)
            assert_eq!(
                list.len(),
                2,
                "COLLECT should have 2 items (no NULLs), got {:?}",
                list
            );
            for item in list.iter() {
                assert_ne!(*item, Value::Null, "COLLECT should not contain NULL values");
            }
        }
        other => panic!("Expected List from COLLECT, got {:?}", other),
    }
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_sum_skips_nulls() {
    // SUM should skip NULL values from OPTIONAL MATCH.
    let db = create_social_network();
    let session = db.session();

    // Add headcount property to TechCorp
    session
        .execute_cypher("MATCH (c:Company {name: 'TechCorp'}) SET c.headcount = 100")
        .unwrap();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:WORKS_AT]->(c:Company) \
             RETURN SUM(c.headcount) AS total_headcount",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    // Alix and Gus both see headcount=100, Harm sees NULL (skipped)
    // SUM = 100 + 100 = 200
    assert_eq!(result.rows()[0][0], Value::Int64(200));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_count_per_group() {
    // COUNT with GROUP BY from OPTIONAL MATCH.
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
             RETURN a.name AS person, COUNT(b.name) AS friend_count \
             ORDER BY person",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    // Alix knows Gus and Harm -> 2
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(2));
    // Gus knows Harm -> 1
    assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[1][1], Value::Int64(1));
    // Harm knows nobody -> 0
    assert_eq!(result.rows()[2][0], Value::String("Harm".into()));
    assert_eq!(result.rows()[2][1], Value::Int64(0));
}

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_boolean_null_comparison() {
    // Comparing NULL to a value should produce NULL (falsy), not true or false.
    // This means WHERE m.age > 30 should not match when m is NULL.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute_cypher(
            "CREATE (a:Person {name: 'Alix', active: true}), \
             (b:Person {name: 'Gus'}), \
             (a)-[:KNOWS]->(b)",
        )
        .unwrap();

    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:KNOWS]->(m:Person) \
             WHERE m.active = true \
             RETURN a.name, m.name \
             ORDER BY a.name",
        )
        .unwrap();

    // Alix has no outgoing KNOWS to someone with active=true (Gus has no active prop).
    // Gus has no outgoing KNOWS at all.
    // Both should be preserved by the left join.
    assert_eq!(
        result.rows().len(),
        2,
        "Both persons preserved by left join"
    );
    for row in result.rows() {
        assert_eq!(
            row[1],
            Value::Null,
            "No match for active=true, right side should be NULL"
        );
    }
}

// ============================================================================
// OPTIONAL MATCH WHERE cross-side predicates (Area 3a regression)
// ============================================================================

#[cfg(feature = "cypher")]
#[test]
fn test_optional_match_where_cross_predicate_preserves_null_rows() {
    // WHERE referencing both a left-side and right-side variable (cross-predicate).
    // The cross-side predicate must not eliminate NULL-padded rows (unmatched optional side).
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute_cypher(
            "CREATE (a:Person {name: 'Alix', age: 30}), \
             (b:Person {name: 'Gus', age: 25}), \
             (c:Person {name: 'Harm', age: 40}), \
             (a)-[:KNOWS]->(b)",
        )
        .unwrap();

    // Cross-predicate: b.age < a.age references both left (a) and right (b) sides.
    // Harm has no KNOWS edge, so the right side is NULL for Harm.
    // NULL-padded rows must pass through even though (NULL < 40) evaluates to NULL.
    let result = session
        .execute_cypher(
            "MATCH (a:Person) \
             OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
             WHERE b.age < a.age \
             RETURN a.name, b.name \
             ORDER BY a.name",
        )
        .unwrap();

    // All 3 persons must be preserved by the left join.
    assert_eq!(
        result.rows().len(),
        3,
        "Cross-predicate WHERE must not eliminate unmatched left rows, got {} rows",
        result.rows().len()
    );

    // Alix (age 30) knows Gus (age 25): cross-predicate 25 < 30 is TRUE, Gus should appear.
    let alix_row = result
        .rows()
        .iter()
        .find(|r| r[0] == Value::String("Alix".into()))
        .expect("Alix row missing");
    assert_eq!(
        alix_row[1],
        Value::String("Gus".into()),
        "Alix->Gus should match (25 < 30)"
    );

    // Gus knows nobody: right side NULL, row must be preserved.
    let gus_row = result
        .rows()
        .iter()
        .find(|r| r[0] == Value::String("Gus".into()))
        .expect("Gus row missing");
    assert_eq!(
        gus_row[1],
        Value::Null,
        "Gus has no KNOWS, right side must be NULL"
    );

    // Harm knows nobody: right side NULL, row must be preserved.
    let harm_row = result
        .rows()
        .iter()
        .find(|r| r[0] == Value::String("Harm".into()))
        .expect("Harm row missing");
    assert_eq!(
        harm_row[1],
        Value::Null,
        "Harm has no KNOWS, right side must be NULL"
    );
}

// ============================================================================
// CALL PROCEDURE: covers plan_call_procedure, plan_static_result
// ============================================================================

#[cfg(feature = "algos")]
#[test]
fn test_call_list_procedures() {
    let db = create_social_network();
    let session = db.session();

    let result = session.execute("CALL grafeo.procedures()").unwrap();

    // Should return a list of available procedures
    assert!(!result.rows().is_empty(), "result.rows() is empty");
}

#[cfg(feature = "algos")]
#[test]
fn test_call_degree_centrality() {
    let db = create_social_network();
    let session = db.session();

    let result = session.execute("CALL grafeo.degree_centrality()").unwrap();

    // Should return results for each node
    assert!(!result.rows().is_empty(), "result.rows() is empty");
}

#[cfg(feature = "algos")]
#[test]
fn test_call_procedure_with_yield() {
    let db = create_social_network();
    let session = db.session();

    let result = session
        .execute("CALL grafeo.pagerank() YIELD node_id, score RETURN node_id, score")
        .unwrap();

    assert!(!result.rows().is_empty(), "result.rows() is empty");
    // Each row should have node_id and score
    assert!(result.columns.len() >= 2);
}

// ============================================================================
// REMOVE property: covers gql.rs REMOVE clause (lines 516-533)
// ============================================================================

#[test]
fn test_gql_remove_property() {
    let db = create_social_network();
    let session = db.session();

    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n.temp = 'delete_me'")
        .unwrap();

    let before = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.temp")
        .unwrap();
    assert_eq!(before.rows()[0][0], Value::String("delete_me".into()));

    session
        .execute("MATCH (n:Person {name: 'Alix'}) REMOVE n.temp")
        .unwrap();

    let after = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.temp")
        .unwrap();
    assert_eq!(after.rows()[0][0], Value::Null);
}

// ============================================================================
// SET map assignment: covers gql.rs lines 368-377
// ============================================================================

#[test]
fn test_gql_set_map_merge() {
    let db = create_social_network();
    let session = db.session();

    session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) SET n += {email: 'alix@example.com', active: true}",
        )
        .unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.email, n.active, n.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(
        result.rows()[0][0],
        Value::String("alix@example.com".into())
    );
    assert_eq!(result.rows()[0][1], Value::Bool(true));
    assert_eq!(result.rows()[0][2], Value::String("Alix".into()));
}

// ============================================================================
// Multiple labels on SET: covers gql.rs label_operations loop (lines 378-384)
// ============================================================================

#[test]
fn test_gql_set_multiple_labels() {
    let db = create_social_network();
    let session = db.session();

    session
        .execute("MATCH (n:Person {name: 'Gus'}) SET n:Employee:Developer")
        .unwrap();

    let emp = session.execute("MATCH (n:Employee) RETURN n.name").unwrap();
    let dev = session
        .execute("MATCH (n:Developer) RETURN n.name")
        .unwrap();

    assert_eq!(emp.rows().len(), 1);
    assert_eq!(dev.rows().len(), 1);
    assert_eq!(emp.rows()[0][0], Value::String("Gus".into()));
}

// ============================================================================
// MERGE with chained input: covers merge.rs input branch (lines 173-206)
// ============================================================================

#[test]
fn test_gql_merge_with_match_input() {
    let db = create_social_network();
    let session = db.session();

    session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) \
             MERGE (n)-[:FOLLOWS]->(t:Trend {name: 'Rust'})",
        )
        .unwrap();

    let result = session.execute("MATCH (t:Trend) RETURN t.name").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Rust".into()));
}

// ============================================================================
// MERGE in ordered_clauses: covers gql.rs line 386-388
// ============================================================================

#[test]
fn test_gql_merge_in_ordered_clauses() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("MERGE (n:Config {key: 'theme'}) ON CREATE SET n.value = 'dark'")
        .unwrap();

    session
        .execute("MERGE (n:Config {key: 'theme'}) ON MATCH SET n.value = 'light'")
        .unwrap();

    let result = session
        .execute("MATCH (n:Config) RETURN n.key, n.value")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][1], Value::String("light".into()));
}

// ============================================================================
// MATCH + CREATE edge in ordered_clauses: covers gql.rs lines 347-349
// ============================================================================

#[test]
fn test_gql_match_create_edge_ordered() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute("INSERT (:City {name: 'Prague'})").unwrap();
    session
        .execute("INSERT (:Country {name: 'Czechia'})")
        .unwrap();

    session
        .execute(
            "MATCH (c:City {name: 'Prague'}), (co:Country {name: 'Czechia'}) \
             CREATE (c)-[:IN]->(co)",
        )
        .unwrap();

    let result = session
        .execute("MATCH (c:City)-[:IN]->(co:Country) RETURN c.name, co.name")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Prague".into()));
    assert_eq!(result.rows()[0][1], Value::String("Czechia".into()));
}

// ============================================================================
// MATCH + DETACH DELETE in ordered_clauses: covers gql.rs lines 350-356
// ============================================================================

#[test]
fn test_gql_match_detach_delete_ordered() {
    let db = create_social_network();
    let session = db.session();

    let before = db.node_count();

    session
        .execute("MATCH (n:Company {name: 'TechCorp'}) DETACH DELETE n")
        .unwrap();

    assert_eq!(db.node_count(), before - 1);
}

// ============================================================================
// FOR clause in ordered_clauses: covers gql.rs lines 337-346
// ============================================================================

#[test]
fn test_gql_for_in_ordered_clauses() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("FOR x IN [100, 200, 300] WITH ORDINALITY idx RETURN x, idx ORDER BY x")
        .unwrap();

    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0], Value::Int64(100));
    assert_eq!(result.rows()[0][1], Value::Int64(1));
    assert_eq!(result.rows()[2][0], Value::Int64(300));
    assert_eq!(result.rows()[2][1], Value::Int64(3));
}

// ============================================================================
// CREATE + DELETE ordered: covers gql.rs ordered Create/Delete paths
// ============================================================================

#[test]
fn test_gql_ordered_create_delete() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("INSERT (:Temp {name: 'ephemeral'})")
        .unwrap();
    assert_eq!(db.node_count(), 1);

    session.execute("MATCH (n:Temp) DETACH DELETE n").unwrap();
    assert_eq!(db.node_count(), 0);
}

// ============================================================================
// create_node_with_props convenience: covers traits.rs default implementations
// ============================================================================

#[test]
fn test_traits_create_with_props_convenience() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let node = session
        .create_node_with_props(
            &["Widget"],
            [
                ("color", Value::String("blue".into())),
                ("weight", Value::Int64(42)),
            ],
        )
        .unwrap();

    let result = session
        .execute("MATCH (w:Widget) RETURN w.color, w.weight")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("blue".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(42));

    let other = session
        .create_node_with_props(&["Box"], [("size", Value::Int64(10))])
        .unwrap();
    session.create_edge(node, other, "FITS_IN").unwrap();

    let edge_result = session
        .execute("MATCH (w:Widget)-[:FITS_IN]->(b:Box) RETURN w.color, b.size")
        .unwrap();

    assert_eq!(edge_result.rows().len(), 1);
}

// ============================================================================
// Cypher: MERGE relationship, DELETE without DETACH, UNWIND standalone,
// SET map replace/merge, FOREACH, multi-pattern MATCH
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_mutations {
    use super::*;

    #[cfg(feature = "cypher")]
    #[test]
    fn test_merge_relationship_creates() {
        let db = create_social_network();
        let session = db.session();
        let edges_before = db.edge_count();

        session
            .execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Harm'}) \
                 MERGE (a)-[:LIKES]->(b)",
            )
            .unwrap();

        assert_eq!(db.edge_count(), edges_before + 1);
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_merge_relationship_matches() {
        let db = create_social_network();
        let session = db.session();
        let edges_before = db.edge_count();

        session
            .execute_cypher(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 MERGE (a)-[:KNOWS]->(b)",
            )
            .unwrap();

        assert_eq!(db.edge_count(), edges_before);
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_merge_relationship_on_create() {
        let db = create_social_network();
        let session = db.session();

        session
            .execute_cypher(
                "MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Alix'}) \
                 MERGE (a)-[r:MENTORS]->(b) ON CREATE SET r.since = 2025",
            )
            .unwrap();

        let result = session
            .execute_cypher(
                "MATCH (a:Person {name: 'Gus'})-[r:MENTORS]->(b:Person {name: 'Alix'}) \
                 RETURN r.since",
            )
            .unwrap();

        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(2025));
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_delete_without_detach_connected_node() {
        let db = create_social_network();
        let session = db.session();

        let result = session.execute_cypher("MATCH (n:Person {name: 'Alix'}) DELETE n");
        assert!(
            result.is_err(),
            "DELETE without DETACH on connected node should fail"
        );
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_unwind_standalone_create() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute_cypher(
                "UNWIND [{name: 'Alix', age: 30}, {name: 'Gus', age: 25}] AS props \
                 CREATE (n:Person) SET n.name = props.name, n.age = props.age",
            )
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
            .unwrap();

        assert_eq!(result.rows().len(), 2);
        assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows()[1][0], Value::String("Gus".into()));
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_set_map_replace() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Item {name: 'Widget', price: 10, color: 'red'})")
            .unwrap();

        session
            .execute_cypher("MATCH (n:Item) SET n = {name: 'Gadget', price: 20}")
            .unwrap();

        let result = session
            .execute("MATCH (n:Item) RETURN n.name, n.price, n.color")
            .unwrap();

        assert_eq!(result.rows()[0][0], Value::String("Gadget".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(20));
        assert_eq!(result.rows()[0][2], Value::Null);
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_set_map_merge() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Item {name: 'Widget', price: 10})")
            .unwrap();

        session
            .execute_cypher("MATCH (n:Item) SET n += {color: 'blue', price: 15}")
            .unwrap();

        let result = session
            .execute("MATCH (n:Item) RETURN n.name, n.price, n.color")
            .unwrap();

        assert_eq!(result.rows()[0][0], Value::String("Widget".into()));
        assert_eq!(result.rows()[0][1], Value::Int64(15));
        assert_eq!(result.rows()[0][2], Value::String("blue".into()));
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_foreach_set_property() {
        let db = create_social_network();
        let session = db.session();

        session
            .execute_cypher(
                "MATCH (n:Person) \
                 FOREACH (val IN [1] | SET n.tagged = true)",
            )
            .unwrap();

        let result = session
            .execute("MATCH (n:Person) WHERE n.tagged = true RETURN n.name")
            .unwrap();

        assert_eq!(result.rows().len(), 3);
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_multi_pattern_shared_vars() {
        let db = create_social_network();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person)-[:KNOWS]->(m:Person), (n)-[:WORKS_AT]->(c:Company) \
                 RETURN DISTINCT n.name, c.name \
                 ORDER BY n.name",
            )
            .unwrap();

        assert!(!result.rows().is_empty(), "result.rows() is empty");
        for row in result.rows() {
            assert_eq!(row[1], Value::String("TechCorp".into()));
        }
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_multi_pattern_no_shared_vars() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:A {val: 1})").unwrap();
        session.execute("INSERT (:B {val: 2})").unwrap();

        let result = session
            .execute_cypher("MATCH (a:A), (b:B) RETURN a.val, b.val")
            .unwrap();

        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::Int64(1));
        assert_eq!(result.rows()[0][1], Value::Int64(2));
    }
}
