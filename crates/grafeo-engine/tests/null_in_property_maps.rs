//! A property map in a pattern is a list of equalities: `{k: NULL}` is `k = NULL`, which is
//! unknown for every element, so it finds nothing (ISO/IEC 39075 16.3 SR 21 h iii; openCypher
//! and SQL/PGQ read a map the same way). It never means "has no `k`": an element without the key
//! is not found either. A null parameter in the map and `WHERE n.k = NULL` are the controls that
//! already find nothing.
//!
//! The class test runs every place a map can stand (a node, a node with a second key, a node
//! without a label, an edge, inside `EXISTS`, under `OPTIONAL MATCH`) in every language that
//! writes maps, with and without an index on the key, against the `WHERE ... = NULL` form of the
//! same pattern as the oracle.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main null_in_property_maps::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::optimizer::PlannerRules;
use grafeo_engine::session::Session;

/// Alix and Gus hold `k`; Hans and Butch do not, and the edge from Hans to Butch has no `k`.
const GRAPH: &[&str] = &[
    "INSERT (:L {name: 'Alix', k: 1})-[:R {k: 1}]->(:L {name: 'Gus', k: 2})",
    "INSERT (:L {name: 'Hans'})-[:R]->(:L {name: 'Butch'})",
];

/// A language that writes property maps in patterns.
#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
    Sql,
}

const LANGUAGES: [Language; 3] = [Language::Gql, Language::Cypher, Language::Sql];

/// Each place a map can stand, as `(with the map, the oracle with the equality in WHERE)`, in
/// `language`. The exhaustive match makes a new language list its forms.
fn forms(language: Language) -> Vec<(&'static str, &'static str)> {
    match language {
        Language::Gql => vec![
            (
                "MATCH (n:L {k: NULL}) RETURN n.name",
                "MATCH (n:L) WHERE n.k = NULL RETURN n.name",
            ),
            (
                "MATCH (n:L {k: NULL, name: 'Hans'}) RETURN n.name",
                "MATCH (n:L) WHERE n.k = NULL AND n.name = 'Hans' RETURN n.name",
            ),
            (
                "MATCH (n {k: NULL}) RETURN n.name",
                "MATCH (n) WHERE n.k = NULL RETURN n.name",
            ),
            (
                "MATCH (a)-[e:R {k: NULL}]->(b) RETURN b.name",
                "MATCH (a)-[e:R]->(b) WHERE e.k = NULL RETURN b.name",
            ),
            (
                "MATCH (n:L) WHERE EXISTS { MATCH (n)-[:R]->(:L {k: NULL}) } RETURN n.name",
                "MATCH (n:L) WHERE EXISTS { MATCH (n)-[:R]->(m:L) WHERE m.k = NULL } RETURN n.name",
            ),
            (
                "OPTIONAL MATCH (n:L {k: NULL}) RETURN n.name",
                "OPTIONAL MATCH (n:L) WHERE n.k = NULL RETURN n.name",
            ),
        ],
        Language::Cypher => vec![
            (
                "MATCH (n:L {k: null}) RETURN n.name",
                "MATCH (n:L) WHERE n.k = null RETURN n.name",
            ),
            (
                "MATCH (n:L {k: null, name: 'Hans'}) RETURN n.name",
                "MATCH (n:L) WHERE n.k = null AND n.name = 'Hans' RETURN n.name",
            ),
            (
                "MATCH ()-[e:R {k: null}]->(b) RETURN b.name",
                "MATCH ()-[e:R]->(b) WHERE e.k = null RETURN b.name",
            ),
            (
                "OPTIONAL MATCH (n:L {k: null}) RETURN n.name",
                "OPTIONAL MATCH (n:L) WHERE n.k = null RETURN n.name",
            ),
        ],
        Language::Sql => vec![
            (
                "SELECT name FROM GRAPH_TABLE (MATCH (n:L {k: NULL}) COLUMNS (n.name AS name))",
                "SELECT name FROM GRAPH_TABLE (MATCH (n:L) WHERE n.k = NULL COLUMNS (n.name AS name))",
            ),
            (
                "SELECT name FROM GRAPH_TABLE (MATCH (a)-[e:R {k: NULL}]->(b) COLUMNS (b.name AS name))",
                "SELECT name FROM GRAPH_TABLE (MATCH (a)-[e:R]->(b) WHERE e.k = NULL COLUMNS (b.name AS name))",
            ),
        ],
    }
}

fn database(index: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in GRAPH {
        db.execute(statement).unwrap();
    }
    if index {
        db.execute("CREATE INDEX by_k FOR (n:L) ON (n.k)").unwrap();
    }
    db
}

fn run(session: &Session, language: Language, query: &str) -> Result<Vec<String>, String> {
    let result: grafeo_common::utils::error::Result<QueryResult> = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
        Language::Sql => session.execute_sql(query),
    };
    result
        .map(|result| {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            rows
        })
        .map_err(|error| error.to_string())
}

/// The sessions every form runs under: all planner rules (a key column read or an index
/// lookup), and every rule off (a scan and a filter).
fn sessions(db: &GrafeoDB) -> [(&'static str, Session); 2] {
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    [("all rules", db.session()), ("plain", plain)]
}

/// The class: every form, every language, with and without an index, answers as its `WHERE`
/// form, which finds nothing (one null row under `OPTIONAL MATCH`).
#[test]
fn a_null_in_a_property_map_finds_nothing() {
    let mut wrong = Vec::new();
    for index in [false, true] {
        let db = database(index);
        for (rules, session) in sessions(&db) {
            for (language, (map, oracle)) in LANGUAGES.into_iter().flat_map(|language| {
                forms(language)
                    .into_iter()
                    .map(move |form| (language, form))
            }) {
                let want = run(&session, language, oracle);
                let optional = map.starts_with("OPTIONAL");
                let nothing = if optional {
                    vec![format!("{:?}", vec![Value::Null])]
                } else {
                    Vec::new()
                };
                if want.as_ref() != Ok(&nothing) {
                    wrong.push(format!(
                        "index {index}, {rules}, {language:?}, oracle {oracle}: {want:?}"
                    ));
                }
                let got = run(&session, language, map);
                if got != want {
                    wrong.push(format!(
                        "index {index}, {rules}, {language:?}, {map}: {got:?}, want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The controls: a null parameter in the map, and the equality in `WHERE`, find nothing today.
#[test]
fn a_null_parameter_in_a_map_finds_nothing() {
    let db = database(false);
    let params: HashMap<String, Value> = [("v".to_string(), Value::Null)].into();
    let result = db
        .execute_with_params("MATCH (n:L {k: $v}) RETURN n.name", params)
        .unwrap();
    assert!(result.rows().is_empty(), "{:?}", result.rows());
}

/// A map with a null and a value that holds still finds nothing: the conjunction is unknown.
#[test]
fn a_null_beside_a_matching_key_finds_nothing() {
    let db = database(false);
    let result = db
        .execute("MATCH (n:L {name: 'Alix', k: NULL}) RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 0, "{:?}", result.rows());
}
