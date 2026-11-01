//! A cheap conjunct before an EXISTS decides first: in
//! `n.active AND ((n:Environment AND EXISTS {...}) OR n.zz = 1)` the
//! subquery runs only for active Environment nodes, never for a row that
//! `n.active` or `n:Environment` already rejects. A subquery over every row
//! makes a selector on a large code graph cost one subquery per node.
//!
//! Checked by behavior: the subquery divides by `n.zero`, which is 0 on
//! every node the cheap conjuncts reject and 1 on the others, so running it
//! for a rejected row fails the query with division by zero. Every node has
//! an outgoing edge, so the subquery's WHERE is reached whenever it runs.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test exists_after_cheap_conjuncts
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// 40 inactive Graph nodes, 40 active ones that are no Environment, and 4
/// active Environments; only the Environments have `zero: 1`. Each node has
/// an outgoing `R` edge to a Target.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(concat!(
        "UNWIND range(1, 40) AS i ",
        "INSERT (:Graph {i: i, active: false, zero: 0})-[:R]->(:Target)"
    ))
    .unwrap();
    db.execute(concat!(
        "UNWIND range(41, 80) AS i ",
        "INSERT (:Graph:Method {i: i, active: true, zero: 0})-[:R]->(:Target)"
    ))
    .unwrap();
    db.execute(concat!(
        "UNWIND range(81, 84) AS i ",
        "INSERT (:Graph:Environment {i: i, active: true, zero: 1})-[:R]->(:Target)"
    ))
    .unwrap();
    db
}

/// The count of `query`, in GQL or Cypher.
fn count(db: &GrafeoDB, language: &str, query: &str) -> Value {
    let result = db
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
    result.rows()[0][0].clone()
}

/// Control: the subquery fails when it runs for a rejected row.
#[test]
fn the_subquery_fails_for_a_rejected_row() {
    let db = database();
    let error = db
        .execute("MATCH (n:Graph) WHERE EXISTS { MATCH (n)-[:R]->(m) WHERE 10 / n.zero > 0 } RETURN count(n) AS c")
        .expect_err("a subquery over every row divides by zero");
    assert!(error.to_string().contains("division by zero"), "{error}");
}

/// The selector of the issue: `n.active` first, then the label in the OR.
#[test]
fn the_subquery_runs_only_after_its_cheap_conjuncts() {
    let db = database();
    for language in ["gql", "cypher"] {
        let query = concat!(
            "MATCH (n:Graph) WHERE n.active ",
            "AND ((n:Environment AND EXISTS { MATCH (n)-[:R]->(m) WHERE 10 / n.zero > 0 }) ",
            "OR n.zz = 1) ",
            "RETURN count(n) AS c"
        );
        assert_eq!(count(&db, language, query), Value::Int64(4), "{language}");
    }
}

/// The cheap conjunct after the EXISTS in the text decides first all the
/// same: the order a user writes conjuncts in is no plan.
#[test]
fn the_cheap_conjunct_decides_first_in_any_order() {
    let db = database();
    for language in ["gql", "cypher"] {
        for query in [
            concat!(
                "MATCH (n:Graph) WHERE EXISTS { MATCH (n)-[:R]->(m) WHERE 10 / n.zero > 0 } ",
                "AND n.active AND n:Environment RETURN count(n) AS c"
            ),
            concat!(
                "MATCH (n:Graph) WHERE ((EXISTS { MATCH (n)-[:R]->(m) WHERE 10 / n.zero > 0 } ",
                "AND n:Environment) OR n.zz = 1) AND n.active RETURN count(n) AS c"
            ),
        ] {
            assert_eq!(
                count(&db, language, query),
                Value::Int64(4),
                "{language} `{query}`"
            );
        }
    }
}

/// A negated EXISTS waits for its cheap conjuncts too.
#[test]
fn a_negated_subquery_runs_only_after_its_cheap_conjuncts() {
    let db = database();
    for language in ["gql", "cypher"] {
        let query = concat!(
            "MATCH (n:Graph) WHERE n.active ",
            "AND ((n:Environment AND NOT EXISTS { MATCH (n)-[:R]->(m) WHERE 10 / n.zero > 50 }) ",
            "OR n.zz = 1) ",
            "RETURN count(n) AS c"
        );
        assert_eq!(count(&db, language, query), Value::Int64(4), "{language}");
    }
}
