//! An aggregate over a chain of two or more expands without GROUP BY runs
//! factorized (FactorizedAggregateOperator over the lazy chain): COUNT, SUM,
//! AVG, MIN and MAX of a variable. Factorizing must not change the answer, so
//! each query is checked against the same query on a database with factorized
//! execution off, on every variable of the chain: the start node, a middle
//! node, the last node and each edge, with and without DISTINCT.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test factorized_aggregates_match_the_regular_path
//! ```

#![cfg(feature = "gql")]

use grafeo_engine::{Config, GrafeoDB};

/// Alix reaches Gus and Vincent; Gus reaches Jules and Mia, Vincent reaches
/// Mia and Butch: two paths end at Mia, so DISTINCT counts differ from the
/// plain ones.
fn database(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute(
        "INSERT (alix:P {name: 'Alix'}), (gus:P {name: 'Gus'}), (vincent:P {name: 'Vincent'}), \
         (jules:P {name: 'Jules'}), (mia:P {name: 'Mia'}), (butch:P {name: 'Butch'}), \
         (alix)-[:R {w: 1}]->(gus), (alix)-[:R {w: 2}]->(vincent), \
         (gus)-[:R {w: 3}]->(jules), (gus)-[:R {w: 4}]->(mia), \
         (vincent)-[:R {w: 5}]->(mia), (vincent)-[:R {w: 6}]->(butch)",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, query: &str) -> String {
    match db.execute(query) {
        Ok(result) => format!("{:?}", result.rows()),
        Err(error) => format!("error: {error}"),
    }
}

const CHAIN: &str = "MATCH (a:P {name: 'Alix'})-[e1:R]->(b)-[e2:R]->(c)";

#[test]
fn each_aggregate_of_each_chain_variable_matches_the_regular_path() {
    let factorized = database(Config::in_memory());
    let regular = database(Config::in_memory().without_factorized_execution());
    let mut differ = Vec::new();
    for function in ["count", "min", "max", "sum", "avg"] {
        for variable in ["a", "b", "c", "e1", "e2"] {
            for distinct in ["", "DISTINCT "] {
                let query = format!("{CHAIN} RETURN {function}({distinct}{variable}) AS x");
                let (got, want) = (rows(&factorized, &query), rows(&regular, &query));
                if got != want {
                    differ.push(format!(
                        "{query}\n    factorized {got}\n    regular    {want}"
                    ));
                }
            }
        }
    }
    assert_eq!(
        differ.len(),
        0,
        "{} queries differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

#[test]
fn several_aggregates_in_one_return_match_the_regular_path() {
    let factorized = database(Config::in_memory());
    let regular = database(Config::in_memory().without_factorized_execution());
    for query in [
        format!("{CHAIN} RETURN count(*) AS n, count(b) AS nb, count(DISTINCT c) AS dc"),
        format!("{CHAIN} RETURN min(b) AS x, max(c) AS y"),
        format!("{CHAIN} RETURN count(*) AS n, sum(c) AS s"),
    ] {
        assert_eq!(rows(&factorized, &query), rows(&regular, &query), "{query}");
    }
}

#[test]
fn counts_over_the_chain_by_hand() {
    // Four paths; three distinct ends (Mia twice), two distinct middles.
    let db = database(Config::in_memory());
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(*) AS n")),
        "[[Int64(4)]]"
    );
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(c) AS n")),
        "[[Int64(4)]]"
    );
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(DISTINCT c) AS n")),
        "[[Int64(3)]]"
    );
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(DISTINCT b) AS n")),
        "[[Int64(2)]]"
    );
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(DISTINCT a) AS n")),
        "[[Int64(1)]]"
    );
}
