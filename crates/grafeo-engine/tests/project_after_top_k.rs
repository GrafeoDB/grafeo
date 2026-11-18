//! An `ORDER BY ... LIMIT` computes only its sort keys before the top-k, and the items that cannot
//! raise after it (`PlannerRule::ProjectAfterTopK`): a property of a node or an edge the pattern
//! binds, a literal or a parameter is read for the rows the limit keeps, not for every row. Every
//! other item (a function call, a property of a value `WITH` binds, anything under `DISTINCT`)
//! stays before, so no error moves or disappears.
//!
//! The answer is the one the rule off gives, its column types and rows in order, an error the
//! same error, in GQL and Cypher. Each comparison runs on one session with the rule toggled.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main project_after_top_k::
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

/// Twelve `:C` comments (two share a creation time, one has no content), five `:P` people, and
/// replies from the first eight comments to people, with a weight.
fn comments() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "UNWIND range(0, 11) AS i INSERT (:C {id: 'comment_' + toString(i), \
         content: CASE WHEN i = 3 THEN null ELSE 'text ' + toString(i % 5) END, \
         created: CASE WHEN i = 7 THEN 1000 ELSE 1000 + i * 37 % 11 END, length: 20 + i % 4 * 5})",
    )
    .unwrap();
    db.execute(
        "UNWIND ['Vincent', 'Jules', 'Mia', 'Butch', 'Django'] AS name INSERT (:P {name: name})",
    )
    .unwrap();
    db.execute(
        "MATCH (m:C), (p:P) WHERE toInteger(substring(m.id, 8)) < 8 \
         AND p.name = ['Vincent', 'Jules', 'Mia', 'Butch', 'Django'][toInteger(substring(m.id, 8)) % 5] \
         INSERT (m)-[:REPLY {w: toInteger(substring(m.id, 8)) * 3}]->(p)",
    )
    .unwrap();
    db
}

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

fn answer_with(
    session: &mut Session,
    rules: PlannerRules,
    language: Language,
    query: &str,
    params: &HashMap<String, Value>,
) -> String {
    session.set_planner_rules(rules);
    let result = match language {
        Language::Gql => session.execute_with_params(query, params.clone()),
        Language::Cypher => session.execute_cypher_with_params(query, params.clone()),
    };
    match result {
        // The column types are part of the answer: a kept entity must stay an entity.
        Ok(result) => format!(
            "{:?} {:?} {:?}",
            result.columns,
            result.column_types,
            result.rows()
        ),
        Err(error) => format!("error: {error}"),
    }
}

/// The query with the rule and without it, on one session.
fn both(session: &mut Session, language: Language, query: &str) -> (String, String) {
    let params = HashMap::from([("p".to_string(), Value::from("Alix"))]);
    let all = PlannerRules::all();
    let after = answer_with(session, all, language, query, &params);
    let before = answer_with(
        session,
        all.with(PlannerRule::ProjectAfterTopK, false),
        language,
        query,
        &params,
    );
    session.set_planner_rules(all);
    (after, before)
}

/// Each shape the rule meets, in both languages (the same text reads in both).
const QUERIES: &[&str] = &[
    // graph-bench's order plus limit
    "MATCH (m:C) WHERE m.length >= 30 RETURN m.id AS id, m.content AS content, m.created AS created \
     ORDER BY created DESC, id ASC LIMIT 5",
    // ties on the key, a limit past the rows, a null content
    "MATCH (m:C) RETURN m.content AS content, m.created AS created ORDER BY created LIMIT 4",
    "MATCH (m:C) RETURN m.id AS id, m.content AS content ORDER BY id LIMIT 50",
    // a literal, a parameter and the node itself beside the deferred property
    "MATCH (m:C) RETURN m.id AS id, 1 AS one, $p AS p, m AS node, m.content AS content ORDER BY id DESC LIMIT 3",
    // a sort key the return leaves out
    "MATCH (m:C) RETURN m.content AS content ORDER BY m.created DESC, m.id LIMIT 3",
    // a function call stays before the top-k
    "MATCH (m:C) RETURN m.id AS id, toUpper(m.content) AS loud, m.length AS length ORDER BY id LIMIT 3",
    // DISTINCT keeps every item before
    "MATCH (m:C) RETURN DISTINCT m.content AS content, m.length AS length ORDER BY content LIMIT 3",
    // an OPTIONAL MATCH's nulls: their properties are null after the top-k too
    "MATCH (m:C) OPTIONAL MATCH (m)-[r:REPLY]->(p:P) RETURN m.id AS id, p.name AS name, r.w AS w \
     ORDER BY id LIMIT 10",
    // an edge's and a far end's properties
    "MATCH (m:C)-[r:REPLY]->(p:P) RETURN r.w AS w, p.name AS name, m.id AS id ORDER BY id DESC LIMIT 4",
    // a property of a value WITH binds stays before
    "MATCH (m:C) WITH m, {a: m.content} AS r RETURN m.id AS id, r.a AS a, m.length AS length \
     ORDER BY id LIMIT 3",
    // a key computed from a returned alias
    "MATCH (m:C) RETURN m.id AS id, m.content AS content ORDER BY m.length + 0 DESC, id LIMIT 3",
    // SKIP before the LIMIT
    "MATCH (m:C) RETURN m.id AS id, m.content AS content ORDER BY id SKIP 2 LIMIT 3",
    // kept items that hold entities, through the projection below: a list of a path's nodes and a
    // map holding a node, beside a deferred property
    concat!(
        "MATCH p = (m:C)-[r:REPLY]->(x:P) RETURN nodes(p) AS ns, m.content AS content, ",
        "m.id AS id ORDER BY id LIMIT 3"
    ),
    "MATCH (m:C) RETURN {n: m} AS wrapped, m.content AS content, m.id AS id ORDER BY id LIMIT 3",
    // an item that raises fails as it does without the rule
    "MATCH (m:C) RETURN m.id AS id, m.length / 0 AS broken, m.content AS content ORDER BY id LIMIT 3",
];

/// Every shape gives the answer the rule off gives, in GQL and Cypher.
#[test]
fn every_shape_answers_as_without_the_rule() {
    let db = comments();
    let mut session = db.session();
    let mut wrong = Vec::new();
    for language in [Language::Gql, Language::Cypher] {
        for query in QUERIES {
            let (after, before) = both(&mut session, language, query);
            if after != before {
                wrong.push(format!(
                    "{language:?}: {query}\n  after  {after}\n  before {before}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// An old snapshot reads its own view through the deferred items: content changed after it began
/// is read as it was, for the rows the limit keeps.
#[test]
fn an_old_snapshot_reads_the_deferred_items_at_its_view() {
    let db = comments();
    let query = "MATCH (m:C) RETURN m.id AS id, m.content AS content ORDER BY id LIMIT 3";
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let before = both(&mut reader, Language::Gql, query);
    db.execute("MATCH (m:C) SET m.content = 'changed'").unwrap();
    let under = both(&mut reader, Language::Gql, query);
    assert_eq!(under.0, under.1);
    assert_eq!(under, before, "the snapshot reads what it saw");
    reader.rollback().unwrap();
}

fn plan(db: &GrafeoDB, query: &str, rules: PlannerRules) -> String {
    let mut session = db.session();
    session.set_planner_rules(rules);
    let result = session.execute(&format!("EXPLAIN {query}")).unwrap();
    result.rows()[0][0].as_str().unwrap().to_string()
}

/// The deferred items move above the limit where the rule may move them, and nothing moves where
/// it may not: under `DISTINCT`, when no item can move, or with the rule off.
#[test]
fn the_plan_defers_only_what_it_may() {
    let db = comments();
    let all = PlannerRules::all();
    let moved = "MATCH (m:C) RETURN m.id AS id, m.content AS content ORDER BY id LIMIT 3";
    for query in [
        moved,
        // entities kept below, beside the deferred content
        concat!(
            "MATCH p = (m:C)-[r:REPLY]->(x:P) RETURN nodes(p) AS ns, m.content AS content, ",
            "m.id AS id ORDER BY id LIMIT 3"
        ),
        "MATCH (m:C) RETURN {n: m} AS wrapped, m.content AS content, m.id AS id ORDER BY id LIMIT 3",
    ] {
        let text = plan(&db, query, all);
        assert!(
            text.starts_with("Return") && text.contains("Limit"),
            "the content is read above the limit for {query}:\n{text}"
        );
    }
    for query in [
        "MATCH (m:C) RETURN DISTINCT m.content AS content, m.id AS id ORDER BY id LIMIT 3",
        "MATCH (m:C) RETURN m.id AS id, toUpper(m.content) AS loud ORDER BY id LIMIT 3",
        "MATCH (m:C) WITH m, {a: m.content} AS r RETURN m.id AS id, r.a AS a ORDER BY id LIMIT 3",
        "MATCH (m:C) RETURN m.id AS id ORDER BY id LIMIT 3",
        "MATCH (m:C) RETURN m.id AS id, m.content AS content ORDER BY id",
    ] {
        let text = plan(&db, query, all);
        assert!(
            !text.starts_with("Return"),
            "nothing moves for {query}:\n{text}"
        );
    }
    let off = plan(&db, moved, all.with(PlannerRule::ProjectAfterTopK, false));
    assert!(!off.starts_with("Return"), "{off}");
}
