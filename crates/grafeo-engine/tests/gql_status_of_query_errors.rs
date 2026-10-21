//! The GQLSTATUS of a query error says what kind of mistake it is, as
//! ISO/IEC 39075 defines the codes: a name that resolves to nothing is 42002
//! (invalid reference); any other rule the statement breaks beyond the
//! grammar, and a feature Grafeo lacks, is 42000, the code ISO gives a
//! violated syntax rule without a condition of its own; a negative limit is
//! 22G02, a value of the wrong type 22G03, DDL in a read-only transaction
//! 25G03. Every semantic error used to be 42002.

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::GqlStatus;
use grafeo_engine::GrafeoDB;

fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', age: 30})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db
}

/// The GQLSTATUS `query` fails with in `language`, with `params`.
fn status(
    db: &GrafeoDB,
    language: &str,
    query: &str,
    params: Option<HashMap<String, Value>>,
) -> String {
    let error = db
        .execute_language(query, language, params)
        .expect_err(&format!("{language} `{query}` should fail"));
    GqlStatus::from(&error).as_str().to_string()
}

fn assert_status(db: &GrafeoDB, language: &str, query: &str, expected: &str) {
    assert_eq!(
        status(db, language, query, None),
        expected,
        "{language} `{query}`"
    );
}

#[test]
fn a_name_that_resolves_to_nothing_is_an_invalid_reference() {
    let db = people();
    let mut cases = vec![
        ("gql", "MATCH (n:Person) RETURN m"),
        ("gql", "RETURN upperr('x') AS v"),
        (
            "gql",
            "MATCH (n:Person) RETURN count(*) AS c ORDER BY n.age",
        ),
        (
            "gql",
            "MATCH (n:Person) RETURN DISTINCT n.name AS name ORDER BY n.age",
        ),
        ("gql", "USE GRAPH nowhere"),
        ("gql", "DROP NODE TYPE Nowhere"),
    ];
    #[cfg(feature = "cypher")]
    cases.extend([
        ("cypher", "MATCH (n:Person) RETURN m"),
        ("cypher", "RETURN upperr('x') AS v"),
        ("cypher", "DELETE n"),
        (
            "cypher",
            "MATCH (n:Person) RETURN count(*) AS c ORDER BY n.age",
        ),
    ]);
    for (language, query) in cases {
        assert_status(&db, language, query, "42002");
    }
}

#[cfg(feature = "algos")]
#[test]
fn an_unknown_procedure_or_yield_column_is_an_invalid_reference() {
    let db = people();
    assert_status(
        &db,
        "gql",
        "CALL grafeo.nowhere() YIELD x RETURN x",
        "42002",
    );
    assert_status(
        &db,
        "gql",
        "CALL grafeo.procedures() YIELD nowhere RETURN nowhere",
        "42002",
    );
}

#[test]
fn a_missing_parameter_is_an_invalid_reference() {
    let db = people();
    assert_status(
        &db,
        "gql",
        "MATCH (n:Person) RETURN n.name AS name LIMIT $k",
        "42002",
    );
    let params = HashMap::from([("other".to_string(), Value::Int64(1))]);
    assert_eq!(
        status(
            &db,
            "gql",
            "MATCH (n:Person) RETURN n.name AS name SKIP $k",
            Some(params)
        ),
        "42002"
    );
}

#[test]
fn another_broken_rule_is_42000() {
    let db = people();
    let mut cases = vec![
        // An aggregate where none may stand.
        (
            "gql",
            "MATCH (p:Person) WHERE count(p) > 0 RETURN p.name AS name",
        ),
        // A wrong number of arguments.
        ("gql", "RETURN abs(1, 2) AS v"),
        // A pattern MERGE cannot take.
        ("gql", "MERGE (a:Person)-[:KNOWS*1..2]->(b:Person)"),
        // An element bound as a node and as an edge.
        ("gql", "MATCH (n)-[n]->() RETURN n"),
        // An ORDER BY of an aggregate the RETURN does not compute.
        (
            "gql",
            "MATCH (n:Person) RETURN n.name AS name, count(*) AS c ORDER BY min(n.age)",
        ),
    ];
    #[cfg(feature = "cypher")]
    cases.extend([
        (
            "cypher",
            "MATCH (p:Person) WHERE count(p) > 0 RETURN p.name AS name",
        ),
        ("cypher", "RETURN abs(1, 2) AS v"),
        ("cypher", "CREATE (a)-[]->(b)"),
        ("cypher", "MERGE (a:Person)-[:KNOWS*1..2]->(b:Person)"),
    ]);
    for (language, query) in cases {
        assert_status(&db, language, query, "42000");
    }
}

/// A feature Grafeo lacks: 42000 too, as GQL defines no class for it, under
/// the code of an unsupported statement.
#[cfg(feature = "cypher")]
#[test]
fn a_feature_grafeo_lacks_is_42000_and_unsupported() {
    use grafeo_common::utils::error::ErrorCode;

    let db = people();
    for (language, query) in [
        (
            "gql",
            "MATCH ANY SHORTEST (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a.name AS name",
        ),
        (
            "cypher",
            "MATCH (n:Person) RETURN n.name AS name SKIP 1 + 1",
        ),
    ] {
        let error = db.execute_language(query, language, None).expect_err(query);
        assert_eq!(GqlStatus::from(&error).as_str(), "42000", "{query}");
        assert_eq!(
            error.error_code(),
            ErrorCode::QueryUnsupported,
            "{query}: {error}"
        );
    }
}

#[test]
fn a_negative_or_non_integer_limit_parameter_is_a_data_exception() {
    let db = people();
    let query = "MATCH (n:Person) RETURN n.name AS name LIMIT $k";
    let negative = HashMap::from([("k".to_string(), Value::Int64(-1))]);
    assert_eq!(status(&db, "gql", query, Some(negative)), "22G02");
    let text = HashMap::from([("k".to_string(), Value::from("three"))]);
    assert_eq!(status(&db, "gql", query, Some(text)), "22G03");
}

#[cfg(feature = "vector-index")]
#[test]
fn a_procedure_argument_of_the_wrong_type_is_an_invalid_value_type() {
    let db = people();
    assert_status(
        &db,
        "gql",
        "CALL grafeo.search.vector('Person', 'embedding', 'Amsterdam', 2)",
        "22G03",
    );
}

/// DDL in a read-only transaction is refused as read-only, in GQL and in
/// Cypher (which used to report a semantic error).
#[cfg(feature = "lpg")]
#[test]
fn ddl_in_a_read_only_transaction_is_read_only() {
    let db = people();
    let session = db.session();
    session.execute("START TRANSACTION READ ONLY").unwrap();
    let error = session
        .execute("CREATE NODE TYPE City (name STRING)")
        .unwrap_err();
    assert_eq!(GqlStatus::from(&error).as_str(), "25G03", "{error}");
    #[cfg(feature = "cypher")]
    {
        let error = session
            .execute_cypher("CREATE INDEX person_name FOR (n:Person) ON (n.name)")
            .unwrap_err();
        assert_eq!(GqlStatus::from(&error).as_str(), "25G03", "{error}");
    }
    session.execute("ROLLBACK").unwrap();
}
