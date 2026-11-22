//! The way each aggregate is built, as PROFILE names it: the counted edges of a chain
//! (`EdgeCount`), the factorized chain (`FactorizedAggregate`), a label's count
//! (`LabelCount`), a range's fold (`RangeFold`), or the aggregate over the rows
//! (`SimpleAggregate`, `HashAggregate`). The route is decided once, in this order.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main aggregate_routes::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

/// Alix, Gus, Vincent and Jules: a chain of `R` steps, values `x` 1 to 4, and Alix in a city.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:P {name: 'Alix', x: 1})-[:R]->(g:P {name: 'Gus', x: 2}), \
         (g)-[:R]->(v:P {name: 'Vincent', x: 3}), (v)-[:R]->(a), \
         (g)-[:R]->(:P {name: 'Jules', x: 4}), (a)-[:LIVES_IN]->(:City {name: 'Amsterdam'})",
    )
    .unwrap();
    db
}

/// The operator PROFILE shows at the top of `query`'s plan.
fn route(session: &Session, query: &str) -> String {
    let profile = session.execute(&format!("PROFILE {query}")).unwrap();
    match &profile.rows()[0][0] {
        Value::String(text) => text
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().next())
            .unwrap_or_default()
            .to_string(),
        other => panic!("PROFILE {query}: {other:?}"),
    }
}

/// Each route's statement takes it, and the boundary between two routes holds: a count of the
/// chain's own variable is counted from the edges, a count of a property is not.
#[test]
fn each_aggregate_takes_its_route() {
    let db = people();
    let session = db.session();
    for (query, expected) in [
        (
            "MATCH (:P)-[:R]->(:P)-[:R]->() RETURN count(*) AS n",
            "EdgeCount",
        ),
        (
            "MATCH (a:P)-[:R]->(b)-[:R]->(c) RETURN count(c) AS n",
            "EdgeCount",
        ),
        (
            "MATCH (a:P)-[:R]->(b) RETURN a.name AS name, count(*) AS n",
            "HashAggregate",
        ),
        (
            "MATCH (a:P)-[:R]->(b)-[:R]->(c) RETURN count(c.x) AS n",
            "SimpleAggregate",
        ),
        (
            "CALL grafeo.procedures() MATCH (p:P)-[:R]->(b)-[:R]->(c) RETURN count(*) AS n",
            "FactorizedAggregate",
        ),
        ("MATCH (n:P) RETURN count(n) AS n", "LabelCount"),
        (
            "MATCH (n:P) WHERE n.x > 1 RETURN count(*) AS c, avg(n.x) AS a",
            "RangeFold",
        ),
        (
            "MATCH (n:P) RETURN n.name AS name, count(*) AS c",
            "HashAggregate",
        ),
        (
            "MATCH (p:P) OPTIONAL MATCH (p)-[:LIVES_IN]->(t:City) \
             MATCH (p)-[:R]->(f)-[:R]->(g) RETURN count(t) AS n",
            "SimpleAggregate",
        ),
    ] {
        assert_eq!(route(&session, query), expected, "{query}");
    }
}

/// Without factorized execution no count is read from the edges or the factorized chain:
/// the label count and the range fold stay, as their own rules decide them.
#[test]
fn without_factorized_execution_the_counts_run_over_the_rows() {
    let db = people();
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::FactorizedExecution, false));
    for (query, expected) in [
        (
            "MATCH (:P)-[:R]->(:P)-[:R]->() RETURN count(*) AS n",
            "SimpleAggregate",
        ),
        (
            "CALL grafeo.procedures() MATCH (p:P)-[:R]->(b)-[:R]->(c) RETURN count(*) AS n",
            "SimpleAggregate",
        ),
        ("MATCH (n:P) RETURN count(n) AS n", "LabelCount"),
        (
            "MATCH (n:P) WHERE n.x > 1 RETURN count(*) AS c, avg(n.x) AS a",
            "RangeFold",
        ),
    ] {
        assert_eq!(route(&session, query), expected, "{query}");
    }
}
