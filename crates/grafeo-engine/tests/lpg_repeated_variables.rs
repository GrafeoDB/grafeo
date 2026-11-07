//! A variable named twice in one LPG pattern binds one element (0.6.2, the class test after the
//! SPARQL repeated-variable bug): `(a)-[:R]->(a)` matches only self-loops, in every language and
//! whatever the planner does with the pattern. GQL and SQL/PGQ match walks with repeatable
//! elements (ISO/IEC 39075 16.4, the implementation-defined default Grafeo takes), Cypher matches
//! with edge isomorphism (no relationship twice in one `MATCH`), so the counts of a closed
//! variable-length path differ by language. Every expected row is enumerated from the graph by
//! hand (the walks and trails are listed in the comments).
//!
//! Each case runs under every planner rule set (all rules, no rule, each rule off alone) and on
//! a database with and without a property index on `name`, so a rewrite that loses the shared
//! binding (a seek, an expand-into, a join, a factorized hop) fails here.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main lpg_repeated_variables::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

/// Alix:P:Q, Gus:P, Vincent:Q; R edges Alix->Alix (the self-loop), Alix->Gus, Gus->Alix,
/// Gus->Vincent; one T edge Vincent->Gus.
const SETUP: &[&str] = &[
    "INSERT (:P:Q {name: 'Alix'}), (:P {name: 'Gus'}), (:Q {name: 'Vincent'})",
    "MATCH (a {name: 'Alix'}) INSERT (a)-[:R]->(a)",
    "MATCH (a {name: 'Alix'}), (g {name: 'Gus'}) INSERT (a)-[:R]->(g), (g)-[:R]->(a)",
    "MATCH (g {name: 'Gus'}), (v {name: 'Vincent'}) INSERT (g)-[:R]->(v), (v)-[:T]->(g)",
];

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
    Sql,
}

struct Case {
    name: &'static str,
    language: Language,
    query: &'static str,
    /// A `$n` parameter, for the cases that run one text with two values.
    name_param: Option<&'static str>,
    rows: &'static [&'static str],
}

const fn case(
    name: &'static str,
    language: Language,
    query: &'static str,
    rows: &'static [&'static str],
) -> Case {
    Case {
        name,
        language,
        query,
        name_param: None,
        rows,
    }
}

const fn with_param(
    name: &'static str,
    language: Language,
    query: &'static str,
    value: &'static str,
    rows: &'static [&'static str],
) -> Case {
    Case {
        name,
        language,
        query,
        name_param: Some(value),
        rows,
    }
}

use Language::{Cypher, Gql, Sql};

const CASES: &[Case] = &[
    case(
        "rep_node",
        Gql,
        "MATCH (a)-[:R]->(a) RETURN a.name",
        &["Alix"],
    ),
    case(
        "rep_node",
        Cypher,
        "MATCH (a)-[:R]->(a) RETURN a.name",
        &["Alix"],
    ),
    case(
        "rep_node",
        Sql,
        "SELECT * FROM GRAPH_TABLE (MATCH (a)-[:R]->(a) COLUMNS (a.name AS n))",
        &["Alix"],
    ),
    case(
        "rep_node_count",
        Gql,
        "MATCH (a)-[:R]->(a) RETURN count(*)",
        &["1"],
    ),
    case(
        "rep_node_count",
        Cypher,
        "MATCH (a)-[:R]->(a) RETURN count(*)",
        &["1"],
    ),
    case(
        "rep_node_where",
        Gql,
        "MATCH (a)-[:R]->(b) WHERE a = b RETURN a.name",
        &["Alix"],
    ),
    case(
        "rep_node_reversed",
        Gql,
        "MATCH (a)<-[:R]-(a) RETURN a.name",
        &["Alix"],
    ),
    case(
        "rep_node_typed_t",
        Gql,
        "MATCH (a)-[:T]->(a) RETURN a.name",
        &[],
    ),
    // a->b->a: with repeatable edges the self-loop twice counts (Alix, Alix); Cypher's two edges
    // must differ.
    case(
        "two_hop",
        Gql,
        "MATCH (a)-[:R]->(b)-[:R]->(a) RETURN a.name, b.name",
        &["Alix|Alix", "Alix|Gus", "Gus|Alix"],
    ),
    case(
        "two_hop",
        Cypher,
        "MATCH (a)-[:R]->(b)-[:R]->(a) RETURN a.name, b.name",
        &["Alix|Gus", "Gus|Alix"],
    ),
    case(
        "two_hop",
        Sql,
        "SELECT * FROM GRAPH_TABLE (MATCH (a)-[:R]->(b)-[:R]->(a) COLUMNS (a.name AS a, b.name AS b))",
        &["Alix|Alix", "Alix|Gus", "Gus|Alix"],
    ),
    case(
        "two_hop_pinned",
        Gql,
        "MATCH (a {name: 'Gus'})-[:R]->(b)-[:R]->(a) RETURN b.name",
        &["Alix"],
    ),
    // An undirected self-loop matches once (openCypher TCK); the other three edges twice each.
    case(
        "undirected_loop",
        Gql,
        "MATCH (a)-[:R]-(a) RETURN a.name",
        &["Alix"],
    ),
    case(
        "undirected_loop",
        Cypher,
        "MATCH (a)-[:R]-(a) RETURN a.name",
        &["Alix"],
    ),
    case(
        "undirected_all",
        Gql,
        "MATCH (a)-[:R]-(b) RETURN count(*)",
        &["7"],
    ),
    case(
        "undirected_all",
        Cypher,
        "MATCH (a)-[:R]-(b) RETURN count(*)",
        &["7"],
    ),
    case(
        "two_labels",
        Gql,
        "MATCH (a:P)-[:R]->(a:Q) RETURN a.name",
        &["Alix"],
    ),
    case(
        "two_labels",
        Cypher,
        "MATCH (a:P)-[:R]->(a:Q) RETURN a.name",
        &["Alix"],
    ),
    case(
        "two_labels",
        Sql,
        "SELECT * FROM GRAPH_TABLE (MATCH (a:P)-[:R]->(a:Q) COLUMNS (a.name AS n))",
        &["Alix"],
    ),
    case(
        "two_labels_none",
        Gql,
        "MATCH (a:P)-[:R]->(a:Nobody) RETURN a.name",
        &[],
    ),
    case(
        "two_labels_none",
        Cypher,
        "MATCH (a:P)-[:R]->(a:Nobody) RETURN a.name",
        &[],
    ),
    case(
        "two_props",
        Gql,
        "MATCH (a {name: 'Gus'})-[:R]->(a) RETURN a.name",
        &[],
    ),
    case(
        "two_props",
        Cypher,
        "MATCH (a {name: 'Gus'})-[:R]->(a) RETURN a.name",
        &[],
    ),
    case(
        "two_props_end",
        Gql,
        "MATCH (a)-[:R]->(a {name: 'Alix'}) RETURN a.name",
        &["Alix"],
    ),
    // One edge variable twice: the same edge both times, so only the self-loop chains to itself.
    // Cypher's edge isomorphism forbids it.
    case(
        "rep_edge",
        Gql,
        "MATCH (a)-[e:R]->(b)-[e:R]->(c) RETURN a.name, b.name, c.name",
        &["Alix|Alix|Alix"],
    ),
    case(
        "rep_edge",
        Cypher,
        "MATCH (a)-[e:R]->(b)-[e:R]->(c) RETURN a.name, b.name, c.name",
        &[],
    ),
    case(
        "rep_edge",
        Sql,
        "SELECT * FROM GRAPH_TABLE (MATCH (a)-[e:R]->(b)-[e:R]->(c) COLUMNS (a.name AS a, b.name AS b, c.name AS c))",
        &["Alix|Alix|Alix"],
    ),
    case(
        "rep_edge_two_matches",
        Cypher,
        "MATCH (a)-[e:R]->(b) MATCH (c)-[e]->(d) RETURN a.name, d.name",
        &["Alix|Alix", "Alix|Gus", "Gus|Alix", "Gus|Vincent"],
    ),
    // Closed R walks of length 1 to 3: Alix 6 (A-A; A-A-A, A-G-A; A-A-A-A, A-A-G-A, A-G-A-A),
    // Gus 2 (G-A-G, G-A-A-G). The trails drop the walks using the self-loop twice: Alix 4.
    case(
        "varlen_closed",
        Gql,
        "MATCH (a)-[:R]->{1,3}(a) RETURN a.name, count(*)",
        &["Alix|6", "Gus|2"],
    ),
    case(
        "varlen_closed",
        Cypher,
        "MATCH (a)-[:R*1..3]->(a) RETURN a.name, count(*)",
        &["Alix|4", "Gus|2"],
    ),
    case(
        "varlen_closed",
        Sql,
        "SELECT n, count(*) AS c FROM GRAPH_TABLE (MATCH (a)-[:R]->{1,3}(a) COLUMNS (a.name AS n)) GROUP BY n",
        &["Alix|6", "Gus|2"],
    ),
    case(
        "varlen_trail",
        Gql,
        "MATCH TRAIL (a)-[:R]->{1,3}(a) RETURN a.name, count(*)",
        &["Alix|4", "Gus|2"],
    ),
    // Walks of two R edges: 6 in all; the closed ones A-A-A, A-G-A, G-A-G.
    case(
        "group_quantified",
        Gql,
        "MATCH ((x)-[:R]->(y)){2} RETURN count(*)",
        &["6"],
    ),
    case(
        "group_closed",
        Gql,
        "MATCH (a) ((x)-[:R]->(y)){2} (a) RETURN a.name, count(*)",
        &["Alix|2", "Gus|1"],
    ),
    case(
        "shortest_plus",
        Gql,
        "MATCH ANY SHORTEST (a)-[:R]->+(a) RETURN a.name",
        &["Alix", "Gus"],
    ),
    case(
        "shortest_star",
        Gql,
        "MATCH ANY SHORTEST (a)-[:R]->*(a) RETURN a.name",
        &["Alix", "Gus", "Vincent"],
    ),
    case(
        "exists",
        Gql,
        "MATCH (a) WHERE EXISTS { MATCH (a)-[:R]->(a) } RETURN a.name",
        &["Alix"],
    ),
    case(
        "exists",
        Cypher,
        "MATCH (a) WHERE EXISTS { (a)-[:R]->(a) } RETURN a.name",
        &["Alix"],
    ),
    case(
        "not_exists",
        Gql,
        "MATCH (a) WHERE NOT EXISTS { MATCH (a)-[:R]->(a) } RETURN a.name",
        &["Gus", "Vincent"],
    ),
    case(
        "count_subquery",
        Gql,
        "MATCH (a) RETURN a.name, COUNT { MATCH (a)-[:R]->(a) }",
        &["Alix|1", "Gus|0", "Vincent|0"],
    ),
    case(
        "optional",
        Gql,
        "MATCH (a) OPTIONAL MATCH (a)-[r:R]->(a) RETURN a.name, r IS NULL",
        &["Alix|false", "Gus|true", "Vincent|true"],
    ),
    case(
        "optional",
        Cypher,
        "MATCH (a) OPTIONAL MATCH (a)-[r:R]->(a) RETURN a.name, r IS NULL",
        &["Alix|false", "Gus|true", "Vincent|true"],
    ),
    case(
        "call_subquery",
        Gql,
        "CALL { MATCH (a)-[:R]->(a) RETURN a.name AS n } RETURN n",
        &["Alix"],
    ),
    case(
        "call_subquery",
        Cypher,
        "CALL { MATCH (a)-[:R]->(a) RETURN a.name AS n } RETURN n",
        &["Alix"],
    ),
    case(
        "bound_before",
        Gql,
        "MATCH (a {name: 'Alix'}) MATCH (a)-[:R]->(a) RETURN a.name",
        &["Alix"],
    ),
    case(
        "bound_before",
        Cypher,
        "MATCH (a {name: 'Gus'}) MATCH (a)-[:R]->(a) RETURN a.name",
        &[],
    ),
    // One text, first a value with no row, then one with: the second run reuses the cached plan.
    with_param(
        "param",
        Gql,
        "MATCH (a {name: $n})-[:R]->(a) RETURN a.name",
        "Gus",
        &[],
    ),
    with_param(
        "param",
        Gql,
        "MATCH (a {name: $n})-[:R]->(a) RETURN a.name",
        "Alix",
        &["Alix"],
    ),
    with_param(
        "param",
        Cypher,
        "MATCH (a {name: $n})-[:R]->(a) RETURN a.name",
        "Gus",
        &[],
    ),
    with_param(
        "param",
        Cypher,
        "MATCH (a {name: $n})-[:R]->(a) RETURN a.name",
        "Alix",
        &["Alix"],
    ),
];

fn cell(value: &Value) -> String {
    match value {
        Value::String(text) => text.as_str().to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}

fn rows_of(session: &Session, case: &Case) -> Result<Vec<String>, String> {
    let params = || {
        case.name_param
            .map(|value| HashMap::from([("n".to_string(), Value::from(value))]))
    };
    let result = match (case.language, params()) {
        (Gql, None) => session.execute(case.query),
        (Gql, Some(params)) => session.execute_with_params(case.query, params),
        (Cypher, None) => session.execute_cypher(case.query),
        (Cypher, Some(params)) => session.execute_cypher_with_params(case.query, params),
        (Sql, _) => session.execute_sql(case.query),
    }
    .map_err(|error| error.to_string())?;
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join("|"))
        .collect();
    rows.sort();
    Ok(rows)
}

fn database(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("name").unwrap();
    }
    let session = db.session();
    for statement in SETUP {
        session.execute(statement).unwrap();
    }
    db
}

/// All rules, no rule, and every rule off alone.
fn rule_sets() -> Vec<(String, PlannerRules)> {
    let mut sets = vec![
        ("all rules".to_string(), PlannerRules::all()),
        ("no rule".to_string(), PlannerRules::plain()),
    ];
    sets.extend(PlannerRule::ALL.iter().map(|rule| {
        (
            format!("all but {rule:?}"),
            PlannerRules::all().with(*rule, false),
        )
    }));
    sets
}

#[test]
fn a_repeated_variable_binds_one_element_under_every_plan() {
    let mut wrong = Vec::new();
    for indexed in [false, true] {
        let db = database(indexed);
        for (rules_name, rules) in rule_sets() {
            let mut session = db.session();
            session.set_planner_rules(rules);
            for case in CASES {
                let found = rows_of(&session, case);
                let right = found.as_ref().is_ok_and(|rows| {
                    rows.iter()
                        .map(String::as_str)
                        .eq(case.rows.iter().copied())
                });
                if !right {
                    wrong.push(format!(
                        "{} [{:?}{}] ({rules_name}{}): {found:?}, want {:?}",
                        case.name,
                        case.language,
                        case.name_param.map_or(String::new(), |v| format!(" n={v}")),
                        if indexed { ", name indexed" } else { "" },
                        case.rows,
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A self-loop merged twice is one edge, and the pattern then finds it.
#[test]
fn merging_a_self_loop_twice_creates_one_edge() {
    for (language, statement) in [
        (Cypher, "MATCH (v {name: 'Vincent'}) MERGE (v)-[:R]->(v)"),
        (Gql, "MATCH (v {name: 'Vincent'}) MERGE (v)-[:R]->(v)"),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Q {name: 'Vincent'})").unwrap();
        for _ in 0..2 {
            match language {
                Cypher => session.execute_cypher(statement),
                _ => session.execute(statement),
            }
            .unwrap_or_else(|error| panic!("{language:?}: {error}"));
        }
        let loops = rows_of(
            &session,
            &case("loops", Gql, "MATCH (v)-[:R]->(v) RETURN v.name", &[]),
        );
        let edges = rows_of(
            &session,
            &case("edges", Gql, "MATCH ()-[r]->() RETURN count(*)", &[]),
        );
        assert_eq!(
            (loops, edges),
            (Ok(vec!["Vincent".to_string()]), Ok(vec!["1".to_string()])),
            "{language:?}"
        );
    }
}
