//! GQL label expressions after `:` and `IS` (ISO/IEC 39075 16.8, 0.6.1): a
//! leading `!`, parentheses and the wildcard `%`, on the first node and on a
//! node an edge reaches. Each form is checked against the same condition as a
//! WHERE predicate (`n:Label`, `labels(n)`), on nodes with one label, two
//! labels, and none: `%` is "has at least one label", so it leaves out a node
//! without one (0.6.0 matched every node), and `!%` matches only such nodes.
//! SQL/PGQ, whose patterns the same parser reads, gives the same rows.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test gql_label_expressions
//! ```

#![cfg(all(feature = "gql", feature = "sql-pgq"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Nodes 1 to 6: `Message:Post`, `Message:Comment`, `Person`, no label,
/// `Post` alone and `Person:Moderator`. Person 3 KNOWS 1 and LIKES 4, person 6
/// KNOWS 2 and LIKES 5.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in [
        "INSERT (:Message:Post {id: 1}), (:Message:Comment {id: 2}), (:Person {id: 3}), \
         ({id: 4}), (:Post {id: 5}), (:Person:Moderator {id: 6})",
        "MATCH (a {id: 3}), (b {id: 1}) INSERT (a)-[:KNOWS]->(b)",
        "MATCH (a {id: 3}), (b {id: 4}) INSERT (a)-[:LIKES]->(b)",
        "MATCH (a {id: 6}), (b {id: 2}) INSERT (a)-[:KNOWS]->(b)",
        "MATCH (a {id: 6}), (b {id: 5}) INSERT (a)-[:LIKES]->(b)",
    ] {
        db.execute(statement).unwrap();
    }
    db
}

/// The ids `query` returns in its one column, sorted.
fn ids(db: &GrafeoDB, language: &str, query: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(id) => id,
            ref other => panic!("{query}: {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids
}

/// Each label expression, the WHERE condition on `n` it stands for, and
/// the nodes of the dataset it matches.
const FORMS: [(&str, &str, &[i64]); 9] = [
    ("!Comment", "NOT n:Comment", &[1, 3, 4, 5, 6]),
    (
        "(Post|Comment)&Message",
        "(n:Post OR n:Comment) AND n:Message",
        &[1, 2],
    ),
    ("%", "size(labels(n)) > 0", &[1, 2, 3, 5, 6]),
    ("!(Post|Comment)", "NOT (n:Post OR n:Comment)", &[3, 4, 6]),
    ("!%", "size(labels(n)) = 0", &[4]),
    (
        "%&!Message",
        "size(labels(n)) > 0 AND NOT n:Message",
        &[3, 5, 6],
    ),
    ("!(!Post)", "n:Post", &[1, 5]),
    ("Person&!Moderator", "n:Person AND NOT n:Moderator", &[3]),
    (
        "!Person|Moderator",
        "NOT n:Person OR n:Moderator",
        &[1, 2, 4, 5, 6],
    ),
];

#[test]
fn a_label_expression_on_the_first_node_matches_its_where_condition() {
    let db = database();
    for (form, condition, want) in FORMS {
        let colon = ids(&db, "gql", &format!("MATCH (n:{form}) RETURN n.id"));
        let is = ids(&db, "gql", &format!("MATCH (n IS {form}) RETURN n.id"));
        let oracle = ids(
            &db,
            "gql",
            &format!("MATCH (n) WHERE {condition} RETURN n.id"),
        );
        assert_eq!(colon, want, ":{form}");
        assert_eq!(is, want, "IS {form}");
        assert_eq!(oracle, want, "WHERE {condition}");
    }
}

#[test]
fn a_label_expression_on_a_node_an_edge_reaches_matches_its_where_condition() {
    let db = database();
    for (form, condition, _) in FORMS {
        let reached = ids(
            &db,
            "gql",
            &format!("MATCH (:Person)-[]->(n:{form}) RETURN n.id"),
        );
        let oracle = ids(
            &db,
            "gql",
            &format!("MATCH (:Person)-[]->(n) WHERE {condition} RETURN n.id"),
        );
        assert_eq!(reached, oracle, ":{form}");
    }
    // The persons reach 1, 2, 4 and 5: the wildcard leaves out 4 only.
    assert_eq!(
        ids(&db, "gql", "MATCH (:Person)-[]->(n:%) RETURN n.id"),
        [1, 2, 5]
    );
    assert_eq!(
        ids(&db, "gql", "MATCH (:Person)-[]->(n:!%) RETURN n.id"),
        [4]
    );
}

#[test]
fn the_wildcard_counts_the_nodes_with_a_label() {
    let db = database();
    let count = |query: &str| ids(&db, "gql", query)[0];
    assert_eq!(count("MATCH (n) RETURN count(n)"), 6);
    assert_eq!(count("MATCH (n IS %) RETURN count(n)"), 5);
    assert_eq!(count("MATCH (n:!%) RETURN count(n)"), 1);
    assert_eq!(
        count("MATCH (n:%) RETURN count(n)") + count("MATCH (n:!%) RETURN count(n)"),
        count("MATCH (n) RETURN count(n)"),
        "every node has a label or not"
    );
}

#[test]
fn sql_pgq_matches_the_same_nodes() {
    let db = database();
    for (form, _, want) in FORMS {
        for pattern in [format!("(n:{form})"), format!("(n IS {form})")] {
            let got = ids(
                &db,
                "sql-pgq",
                &format!("SELECT id FROM GRAPH_TABLE (MATCH {pattern} COLUMNS (n.id AS id))"),
            );
            assert_eq!(got, want, "SQL/PGQ {pattern}");
        }
    }
}
