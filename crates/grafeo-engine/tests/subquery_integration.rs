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

/// `EXISTS` and `COUNT` over a path: `top -> sub -> f` and `lone -> f`.
#[cfg(feature = "cypher")]
mod subqueries_over_paths {
    use super::*;

    fn tree() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:Directory {id: 'top'})-[:CONTAINS]->(:Directory {id: 'sub'})\
             -[:CONTAINS]->(:File {id: 'f'})",
        )
        .unwrap();
        db.execute("MATCH (f:File) INSERT (:Directory {id: 'lone'})-[:CONTAINS]->(f)")
            .unwrap();
        db
    }

    /// One edge from the outer node still decides a path with no condition on
    /// its end, also in `RETURN`.
    #[test]
    fn exists_over_a_path_with_no_end_condition_in_return() {
        let db = tree();
        let result = db
            .execute_cypher(
                "MATCH (n) RETURN n.id, EXISTS { MATCH (n)-[:CONTAINS*]->() } AS e ORDER BY n.id",
            )
            .unwrap();
        let rows: Vec<(Value, Value)> = result
            .rows()
            .iter()
            .map(|row| (row[0].clone(), row[1].clone()))
            .collect();
        assert_eq!(
            rows,
            [("f", false), ("lone", true), ("sub", true), ("top", true)]
                .map(|(id, e)| (Value::from(id), Value::Bool(e)))
        );
    }

    /// In `RETURN`, a subquery that one edge cannot decide fails rather than
    /// answering for the first hop only (`top` reaches a file in two hops, and
    /// counts two nodes below it). A correlated plan for these is #543.
    #[test]
    fn subqueries_one_edge_cannot_decide_fail_in_return() {
        let db = tree();
        for (query, reason) in [
            (
                "MATCH (n:Directory) RETURN n.id, EXISTS { MATCH (n)-[:CONTAINS*]->(:File) } AS e",
                "Unsupported EXISTS subquery pattern",
            ),
            (
                "MATCH (n:Directory) RETURN n.id, COUNT { MATCH (n)-[:CONTAINS*]->() } AS c",
                "Unsupported COUNT subquery pattern",
            ),
            (
                "MATCH (n:Directory) RETURN n.id, EXISTS { MATCH (n)-[:CONTAINS*2..]->() } AS e",
                "Unsupported EXISTS subquery pattern",
            ),
        ] {
            let error = db.execute_cypher(query).expect_err(query).to_string();
            assert!(error.contains(reason), "{query}: {error}");
        }
    }

    /// `COUNT` of single edges keeps the fast path: `f` has two incoming.
    #[test]
    fn count_of_single_edges_in_return() {
        let db = tree();
        let result = db
            .execute_cypher(
                "MATCH (n) RETURN n.id, COUNT { MATCH (n)<-[:CONTAINS]-() } AS c ORDER BY n.id",
            )
            .unwrap();
        let rows: Vec<(Value, Value)> = result
            .rows()
            .iter()
            .map(|row| (row[0].clone(), row[1].clone()))
            .collect();
        assert_eq!(
            rows,
            [("f", 2), ("lone", 0), ("sub", 1), ("top", 0)]
                .map(|(id, c)| (Value::from(id), Value::Int64(c)))
        );
    }
}

/// `EXISTS` and `COUNT` through the edge of the row: Alix knows Gus, and Mia
/// likes herself.
mod subqueries_through_a_bound_edge {
    use super::*;

    fn people() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'}), \
             (mia:Person {name: 'Mia'})-[:LIKES]->(mia)",
        )
        .unwrap();
        db
    }

    /// The pattern reads the row's edge from the ends it names: forward from
    /// the source, backward into the target, and both ways when undirected
    /// with free ends. Mia's self-loop is both her source and her target.
    #[test]
    fn the_edge_is_read_from_the_ends_the_pattern_names() {
        let db = people();
        let result = db
            .execute(
                "MATCH (a)-[r]->(b) RETURN a.name, \
                 COUNT { MATCH (a)-[r]->(x) } AS from_a, \
                 COUNT { MATCH (b)-[r]->(x) } AS from_b, \
                 COUNT { MATCH (b)<-[r]-(x) } AS into_b, \
                 COUNT { MATCH (x)-[r]-(y) } AS either_way, \
                 EXISTS { MATCH (x)-[r]->(:Person) } AS to_a_person, \
                 EXISTS { MATCH (x)-[r]->(:City) } AS to_a_city \
                 ORDER BY a.name",
            )
            .unwrap();
        let rows: Vec<Vec<Value>> = result.rows().to_vec();
        assert_eq!(
            rows,
            [("Alix", [1, 0, 1, 2]), ("Mia", [1, 1, 1, 2])]
                .map(|(name, counts)| {
                    let mut row = vec![Value::from(name)];
                    row.extend(counts.map(Value::Int64));
                    row.extend([Value::Bool(true), Value::Bool(false)]);
                    row
                })
                .to_vec()
        );
    }

    /// A compared `COUNT` in `WHERE` is planned as a join; it counts the
    /// undirected matches of the row's edge like the per-row check does.
    #[test]
    fn a_compared_count_counts_the_edge_both_ways() {
        let db = people();
        let result = db
            .execute(
                "MATCH (a)-[r]->(b) WHERE COUNT { MATCH (x)-[r]-(y) } = 2 \
                 RETURN a.name ORDER BY a.name",
            )
            .unwrap();
        let names: Vec<Value> = result.rows().iter().map(|row| row[0].clone()).collect();
        assert_eq!(names, [Value::from("Alix"), Value::from("Mia")]);
    }
}

/// `EXISTS` and `COUNT` whose pattern holds more than its edge: another node
/// pattern, or a path mode on a variable-length edge. Alix knows Gus and
/// lives in Amsterdam, and Mia likes herself; there is no Robot.
mod subqueries_beyond_the_edge {
    use super::*;

    fn people() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "INSERT (alix:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'}), \
             (alix)-[:LIVES_IN]->(:City {name: 'Amsterdam'}), \
             (mia:Person {name: 'Mia'})-[:LIKES]->(mia)",
        )
        .unwrap();
        db
    }

    /// In `RETURN`, such a subquery fails rather than answering from its edge
    /// alone: Alix knows someone but there is no Robot, and Mia's only path
    /// returns to her, which ACYCLIC excludes. A correlated plan for these is
    /// #543.
    #[test]
    fn subqueries_with_more_than_the_edge_fail_in_return() {
        let db = people();
        for query in [
            "MATCH (a:Person) RETURN a.name, EXISTS { MATCH (a)-[:KNOWS]->(b), (c:Robot) } AS e",
            "MATCH (a:Person) RETURN a.name, EXISTS { MATCH (c:Robot), (a)-[:KNOWS]->(b) } AS e",
            "MATCH (a:Person) RETURN a.name, COUNT { MATCH (a)-[:KNOWS]->(b), (c:Person) } AS n",
            "MATCH (a:Person) RETURN a.name, EXISTS { MATCH ACYCLIC (a)-[:LIKES*1..2]->(x) } AS e",
            "MATCH (a:Person) RETURN a.name, EXISTS { MATCH TRAIL (a)-[:KNOWS*1..2]-(x) } AS e",
        ] {
            let error = db.execute(query).expect_err(query).to_string();
            assert!(
                error.contains("Unsupported EXISTS subquery pattern"),
                "{query}: {error}"
            );
        }
    }

    /// A node pattern on the end of the edge is that end: its label joins the
    /// edge check, also in `RETURN`.
    #[cfg(feature = "cypher")]
    #[test]
    fn a_later_node_pattern_on_the_end_labels_it() {
        let db = people();
        let result = db
            .execute_cypher(
                "MATCH (a:Person) RETURN a.name, \
                 COUNT { MATCH (a)-[r]->(b) MATCH (b:Person) } AS people, \
                 EXISTS { MATCH (a)-[r]->(b) MATCH (b:Robot) } AS robots \
                 ORDER BY a.name",
            )
            .unwrap();
        let rows: Vec<Vec<Value>> = result.rows().to_vec();
        assert_eq!(
            rows,
            [("Alix", 1), ("Gus", 0), ("Mia", 1)]
                .map(|(name, people)| vec![
                    Value::from(name),
                    Value::Int64(people),
                    Value::Bool(false)
                ])
                .to_vec()
        );
    }
}
