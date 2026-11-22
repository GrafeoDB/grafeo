//! ISO GQL has no power operator, so it says nothing of how `2 ^ 3 ^ 2`
//! groups; `POWER(b, x)` is the conformant form. GQL refuses an
//! unparenthesized `^` chain with a syntax error that names `POWER`; a single
//! `^` and a parenthesized chain work as before. Cypher reads the chain from
//! the left, as openCypher does.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main gql_power_chain::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn value(db: &GrafeoDB, language: &str, query: &str) -> Value {
    let result = match language {
        "gql" => db.execute(query),
        _ => db.execute_cypher(query),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
    result.rows()[0][0].clone()
}

/// `POWER` nests either way in GQL.
#[test]
fn power_nests_either_way() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        value(&db, "gql", "RETURN POWER(2, POWER(3, 2)) AS r"),
        Value::Float64(512.0)
    );
    assert_eq!(
        value(&db, "gql", "RETURN POWER(POWER(2, 3), 2) AS r"),
        Value::Float64(64.0)
    );
}

/// A single `^` and a parenthesized chain give what `POWER` gives.
#[test]
fn a_single_caret_and_a_parenthesized_chain_work() {
    let db = GrafeoDB::new_in_memory();
    for (caret, power) in [
        ("2 ^ 3", "POWER(2, 3)"),
        ("(2 ^ 3) ^ 2", "POWER(POWER(2, 3), 2)"),
        ("2 ^ (3 ^ 2)", "POWER(2, POWER(3, 2))"),
        ("-2 ^ 2", "POWER(-2, 2)"),
    ] {
        assert_eq!(
            value(&db, "gql", &format!("RETURN {caret} AS r")),
            value(&db, "gql", &format!("RETURN {power} AS r")),
            "{caret}"
        );
    }
}

/// An unparenthesized chain is a syntax error naming `POWER`.
#[test]
fn an_unparenthesized_chain_is_a_syntax_error() {
    let db = GrafeoDB::new_in_memory();
    for query in [
        "RETURN 2 ^ 3 ^ 2 AS r",
        "RETURN 1 + 2 ^ 3 ^ 2 AS r",
        "RETURN 2 ^ 3 ^ 2 ^ 1 AS r",
    ] {
        let error = db.execute(query).expect_err(query);
        let status = grafeo_common::utils::gqlstatus::GqlStatus::from(&error);
        assert!(
            status.as_str().starts_with("42"),
            "{query}: {status:?} {error}"
        );
        assert!(error.to_string().contains("POWER"), "{query}: {error}");
    }
}

/// Cypher reads the chain from the left (openCypher): `2 ^ 3 ^ 2` is 64.
#[test]
fn cypher_reads_the_chain_from_the_left() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        value(&db, "cypher", "RETURN 2 ^ 3 ^ 2 AS r"),
        Value::Float64(64.0)
    );
}
