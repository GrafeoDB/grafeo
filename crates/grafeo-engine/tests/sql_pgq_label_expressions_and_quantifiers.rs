//! SQL/PGQ label expressions, element WHERE and quantifiers after an edge
//! (0.6.1, ISO/IEC 9075-16). Each SQL/PGQ form is checked against the same
//! pattern in GQL (a label expression GQL does not parse yet, a leading `!`,
//! parentheses or `%`, as WHERE predicates), on the LDBC SNB mini
//! dataset of the spec tests:
//!
//! - `|`, `&`, `!`, `%` and parentheses in a label expression, on the first
//!   node and on a node an edge reaches;
//! - an element WHERE on a node and on an edge; a WHERE inside a quantified
//!   edge is refused (Unsupported);
//! - `*` (with its zero hops: the start node itself), `+`, `{2}`, `{1,3}`,
//!   `{,2}` and `{2,}` after an edge; `{3,1}` and a quantifier both inside
//!   the brackets and after the edge are refused;
//! - graph-bench's IS4 text, verbatim, on a graph named `snb`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sql_pgq_label_expressions_and_quantifiers
//! ```

#![cfg(all(feature = "gql", feature = "sql-pgq"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const DATASET: &str = include_str!("../../../tests/spec/datasets/ldbc_snb_mini.setup");

fn statements() -> impl Iterator<Item = &'static str> {
    DATASET
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
}

fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in statements() {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

/// The rows of a SQL/PGQ `MATCH {pattern} COLUMNS ({columns})` against GQL's
/// `MATCH {gql_pattern} RETURN {returns}`; they must agree and not be empty.
fn agree(db: &GrafeoDB, pattern: &str, columns: &str, gql_pattern: &str, returns: &str) -> usize {
    let sql = rows(
        db,
        "sql-pgq",
        &format!("SELECT * FROM GRAPH_TABLE (MATCH {pattern} COLUMNS ({columns}))"),
    );
    let gql = rows(db, "gql", &format!("MATCH {gql_pattern} RETURN {returns}"));
    assert_eq!(sql, gql, "SQL/PGQ `{pattern}` against GQL `{gql_pattern}`");
    assert_ne!(sql.len(), 0, "`{pattern}` matches something");
    sql.len()
}

fn refused(db: &GrafeoDB, pattern: &str, says: &str) {
    let query = format!("SELECT * FROM GRAPH_TABLE (MATCH {pattern} COLUMNS (1 AS one))");
    let error = db
        .execute_language(&query, "sql-pgq", None)
        .expect_err(&query)
        .to_string();
    assert!(error.contains(says), "`{pattern}`: {error}");
}

#[test]
fn label_expressions_on_the_first_node() {
    let db = database();
    let messages = agree(
        &db,
        "(m:Post|Comment)",
        "m.id AS id",
        "(m:Post|Comment)",
        "m.id",
    );
    let posts = agree(&db, "(m:Post)", "m.id AS id", "(m:Post)", "m.id");
    let comments = agree(
        &db,
        "(m:Message&!Post)",
        "m.id AS id",
        "(m:Message&!Post)",
        "m.id",
    );
    assert_eq!(messages, posts + comments, "Post|Comment is every message");
    agree(
        &db,
        "(m:Message&Comment)",
        "m.id AS id",
        "(m:Message&Comment)",
        "m.id",
    );
    agree(
        &db,
        "(m:(Post|Comment)&Message)",
        "m.id AS id",
        "(m) WHERE (m:Post OR m:Comment) AND m:Message",
        "m.id",
    );
    agree(
        &db,
        "(n:!Message)",
        "n.id AS id",
        "(n) WHERE NOT n:Message",
        "n.id",
    );
    let any = agree(
        &db,
        "(n:%)",
        "n.id AS id",
        "(n) WHERE size(labels(n)) > 0",
        "n.id",
    );
    let all = agree(&db, "(n)", "n.id AS id", "(n)", "n.id");
    assert_eq!(any, all, "every node of the dataset has a label");
}

#[test]
fn label_expressions_on_a_node_an_edge_reaches() {
    let db = database();
    agree(
        &db,
        "(p:Person)<-[c:HAS_CREATOR]-(m:Post|Comment)",
        "p.id AS person, m.id AS message",
        "(p:Person)<-[c:HAS_CREATOR]-(m:Post|Comment)",
        "p.id, m.id",
    );
    agree(
        &db,
        "(c:Comment)-[r:REPLY_OF]->(m:!Comment)",
        "c.id AS comment, m.id AS message",
        "(c:Comment)-[r:REPLY_OF]->(m) WHERE NOT m:Comment",
        "c.id, m.id",
    );
    agree(
        &db,
        "(f:Forum)-[x:CONTAINER_OF]->(m:Message&(Post|Comment))",
        "f.id AS forum, m.id AS message",
        "(f:Forum)-[x:CONTAINER_OF]->(m) WHERE m:Message AND (m:Post OR m:Comment)",
        "f.id, m.id",
    );
}

#[test]
fn element_where_on_a_node_and_an_edge() {
    let db = database();
    agree(
        &db,
        "(m:Post WHERE m.length > 100)",
        "m.id AS id",
        "(m:Post WHERE m.length > 100)",
        "m.id",
    );
    agree(
        &db,
        "(a:Person)-[k:KNOWS WHERE k.creationDate > 1267000000000]->(b:Person)",
        "a.id AS a, b.id AS b",
        "(a:Person)-[k:KNOWS WHERE k.creationDate > 1267000000000]->(b:Person)",
        "a.id, b.id",
    );
    agree(
        &db,
        "(a:Person WHERE a.gender = 'female')-[k:KNOWS]->(b:Person WHERE b.gender = 'male')",
        "a.id AS a, b.id AS b",
        "(a:Person WHERE a.gender = 'female')-[k:KNOWS]->(b:Person WHERE b.gender = 'male')",
        "a.id, b.id",
    );
    refused(
        &db,
        "(c:Comment)-[r:REPLY_OF WHERE r.x = 1]->*(m:Message)",
        "quantified edge pattern",
    );
}

#[test]
fn quantifiers_after_an_edge() {
    let db = database();
    for (sql, gql) in [
        ("*", "*"),
        ("+", "+"),
        ("{2}", "{2}"),
        ("{1,3}", "{1,3}"),
        ("{,2}", "{0,2}"),
        ("{2,}", "{2,}"),
    ] {
        agree(
            &db,
            &format!("(c:Comment)-[r:REPLY_OF]->{sql}(m:Message)"),
            "c.id AS comment, m.id AS message",
            &format!("(c:Comment)-[r:REPLY_OF]->{gql}(m:Message)"),
            "c.id, m.id",
        );
    }
    // `*` has zero hops: each comment reaches itself.
    let comments = rows(&db, "gql", "MATCH (c:Comment) RETURN c.id").len();
    let itself = rows(
        &db,
        "sql-pgq",
        "SELECT * FROM GRAPH_TABLE (MATCH (c:Comment)-[r:REPLY_OF]->*(m:Message) \
         WHERE c.id = m.id COLUMNS (c.id AS id))",
    );
    assert_eq!(itself.len(), comments, "zero hops: m = c for every comment");
    refused(
        &db,
        "(c:Comment)-[r:REPLY_OF]->{3,1}(m:Message)",
        "upper bound is below its lower bound",
    );
    refused(
        &db,
        "(c:Comment)-[r:REPLY_OF*1..2]->{2}(m:Message)",
        "one quantifier",
    );
}

/// graph-bench's IS4 (graph_bench/suites/snb_sql.py), with its parameter
/// replaced by a message id, on the dataset loaded into a graph named `snb`.
#[test]
fn graph_bench_is4_verbatim() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH snb").unwrap();
    let snb = db.graph("snb").unwrap();
    for statement in statements() {
        snb.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    let messages = snb
        .execute("MATCH (m:Post|Comment) RETURN m.id, m.creationDate, m.content, m.imageFile")
        .unwrap()
        .rows()
        .to_vec();
    assert_ne!(messages.len(), 0, "the dataset has messages");
    let is4 = "SELECT messageCreationDate,
       CASE WHEN content IS NULL OR content = '' THEN imageFile ELSE content END AS messageContent
FROM GRAPH_TABLE (snb
  MATCH (m:Post|Comment)
  WHERE m.id = $messageId
  COLUMNS (m.creationDate AS messageCreationDate, m.content AS content, m.imageFile AS imageFile)
)";
    for message in messages.iter().take(12) {
        let Value::Int64(id) = message[0] else {
            panic!("{:?}", message[0]);
        };
        let got = db
            .execute_language(&is4.replace("$messageId", &id.to_string()), "sql-pgq", None)
            .unwrap_or_else(|error| panic!("IS4 {id}: {error}"))
            .rows()
            .to_vec();
        let content = match &message[2] {
            Value::Null => message[3].clone(),
            Value::String(text) if text.is_empty() => message[3].clone(),
            other => other.clone(),
        };
        assert_eq!(got, [vec![message[1].clone(), content]], "message {id}");
    }
}
