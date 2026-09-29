//! Integration tests for subquery paths: CALL { subquery }, EXISTS semi/anti
//! join, OPTIONAL MATCH NULL padding, and correlated subqueries.
//!
//! Covers uncovered paths in:
//! - gql.rs: CALL { subquery } (inline), lines 406-408, 993-1050
//! - cypher.rs: CALL { subquery } with WITH import, lines 265-285
//! - apply.rs: EXISTS semi-join, anti-join, optional NULL padding
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test subquery_integration
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Fixtures
// ============================================================================

/// Creates 3 Person + 1 Company nodes (Amsterdam/Berlin/Paris), 3 KNOWS + 2 WORKS_AT edges.
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
            ],
        )
        .unwrap();
    let harm = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Harm".into())),
                ("age", Value::Int64(35)),
                ("city", Value::String("Paris".into())),
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

    // Verify setup: 3 Person + 1 Company = 4 nodes, 3 KNOWS + 2 WORKS_AT = 5 edges
    assert_eq!(db.node_count(), 4, "social_graph: expected 4 nodes");
    assert_eq!(db.edge_count(), 5, "social_graph: expected 5 edges");

    db
}

// ============================================================================
// GQL: CALL { subquery } (inline)
// ============================================================================

#[test]
fn test_gql_inline_call_subquery() {
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Alix'}) \
             CALL { WITH n MATCH (n)-[:KNOWS]->(m) RETURN m.name AS friend } \
             RETURN n.name, friend \
             ORDER BY friend",
        )
        .unwrap();

    assert_eq!(result.rows().len(), 2);
    let friends: Vec<String> = result
        .rows()
        .iter()
        .map(|r| match &r[1] {
            Value::String(s) => s.to_string(),
            other => panic!("expected string, got {other:?}"),
        })
        .collect();
    assert_eq!(
        friends,
        vec!["Gus", "Harm"],
        "ORDER BY friend should sort alphabetically"
    );
}

#[test]
fn test_gql_inline_call_without_outer() {
    let db = social_graph();
    let session = db.session();

    let result = session
        .execute("CALL { MATCH (n:Person) RETURN count(n) AS cnt } RETURN cnt")
        .unwrap();

    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(3));
}

// ============================================================================
// Cypher: CALL { subquery } with WITH import
// ============================================================================

#[cfg(feature = "cypher")]
mod cypher_subqueries {
    use super::*;

    #[test]
    fn test_call_subquery_with_wildcard() {
        let db = social_graph();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person {name: 'Alix'}) \
                 CALL { WITH * MATCH (n)-[:KNOWS]->(m) RETURN m.name AS friend } \
                 RETURN n.name, friend \
                 ORDER BY friend",
            )
            .unwrap();

        // WITH * scopes n=Alix from the outer MATCH, giving 2 results (Gus, Harm).
        assert_eq!(
            result.rows().len(),
            2,
            "WITH * should scope outer variable, expected 2 rows"
        );
    }

    #[test]
    fn test_call_subquery_with_specific_var() {
        let db = social_graph();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 CALL { WITH n MATCH (n)-[:KNOWS]->(m) RETURN count(m) AS cnt } \
                 RETURN n.name, cnt \
                 ORDER BY n.name",
            )
            .unwrap();

        assert_eq!(result.rows().len(), 3);
    }

    // ============================================================================
    // EXISTS as semi-join and anti-join: covers apply.rs exists_mode
    // ============================================================================

    #[test]
    fn test_exists_semi_join() {
        let db = social_graph();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 WHERE EXISTS { MATCH (n)-[:KNOWS]->(m)-[:WORKS_AT]->(c) } \
                 RETURN n.name ORDER BY n.name",
            )
            .unwrap();

        // Alix->Gus->TechCorp path exists
        let mut names: Vec<String> = result
            .rows()
            .iter()
            .map(|r| match &r[0] {
                Value::String(s) => s.to_string(),
                other => panic!("expected string, got {other:?}"),
            })
            .collect();
        names.sort();
        assert!(names.contains(&"Alix".to_string()));
    }

    #[test]
    fn test_not_exists_anti_join() {
        let db = social_graph();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 WHERE NOT EXISTS { MATCH (n)-[:WORKS_AT]->() } \
                 RETURN n.name",
            )
            .unwrap();

        assert_eq!(result.rows().len(), 1);
        assert_eq!(result.rows()[0][0], Value::String("Harm".into()));
    }

    // ============================================================================
    // OPTIONAL MATCH NULL padding: covers apply.rs optional branch
    // ============================================================================

    #[test]
    fn test_optional_match_null_padding() {
        let db = social_graph();
        let session = db.session();

        let result = session
            .execute_cypher(
                "MATCH (n:Person) \
                 OPTIONAL MATCH (n)-[:WORKS_AT]->(c:Company) \
                 RETURN n.name, c.name \
                 ORDER BY n.name",
            )
            .unwrap();

        assert_eq!(result.rows().len(), 3);
        let harm_row = result
            .rows()
            .iter()
            .find(|r| r[0] == Value::String("Harm".into()))
            .expect("Harm should be in results");
        assert_eq!(harm_row[1], Value::Null);
    }
}
