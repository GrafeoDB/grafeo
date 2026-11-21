//! The parse limits through the database (#492): `Config::with_max_query_length(bytes)` and
//! `Config::with_max_identifier_length(bytes)`, 16 MiB and 16 KiB by default. Every language
//! refuses a statement one byte over its lowered limit, and a name one byte over, with
//! `GRAFEO-Q002` (GQLSTATUS `42000`) naming the setting; under the limits the same statement
//! returns the same rows as with the defaults. Parameter values do not count; `EXPLAIN` and
//! `PROFILE` are refused as their statement is; a stored procedure's body is held to the limits
//! of the open that runs it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main query_limits::
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, ErrorCode, LimitBound};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::config::ConfigError;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB};

type Run = fn(&GrafeoDB, &str) -> grafeo_common::utils::error::Result<QueryResult>;

/// Each language, a statement in it that returns rows on [`seeded`], and a statement shape
/// holding one name (`{}`).
fn languages() -> Vec<(&'static str, Run, &'static str, &'static str)> {
    let mut all: Vec<(&'static str, Run, &'static str, &'static str)> = vec![(
        "gql",
        |db, q| db.execute(q),
        "MATCH (n:Person) RETURN n.name AS name ORDER BY name",
        "MATCH ({}:Person) RETURN {}.name AS name",
    )];
    #[cfg(feature = "cypher")]
    all.push((
        "cypher",
        |db, q| db.execute_cypher(q),
        "MATCH (n:Person) RETURN n.name AS name ORDER BY name",
        "MATCH ({}:Person) RETURN {}.name AS name",
    ));
    #[cfg(feature = "sql-pgq")]
    all.push((
        "sql-pgq",
        |db, q| db.execute_sql(q),
        "SELECT * FROM GRAPH_TABLE (MATCH (n:Person) COLUMNS (n.name AS name))",
        "SELECT * FROM GRAPH_TABLE (MATCH ({}:Person) COLUMNS ({}.name AS name))",
    ));
    #[cfg(feature = "gremlin")]
    all.push((
        "gremlin",
        |db, q| db.execute_gremlin(q),
        "g.V().hasLabel('Person').values('name')",
        "g.V().hasLabel('Person').as('x').{}()",
    ));
    #[cfg(feature = "graphql")]
    all.push((
        "graphql",
        |db, q| db.execute_graphql(q),
        "{ Person { name } }",
        "{ Person { {} } }",
    ));
    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    all.push((
        "sparql",
        |db, q| db.execute_sparql(q),
        "SELECT ?s WHERE { ?s ?p ?o }",
        "SELECT ?{} WHERE { ?{} ?p ?o }",
    ));
    all
}

/// Two people and one triple, written without a statement: the seeding is held to no limit.
fn seeded(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    for name in ["Alix", "Gus"] {
        db.create_node_with_props(&["Person"], [("name", Value::from(name))])
            .unwrap();
    }
    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    db.load_rdf_text(
        "<http://example.org/alix> <http://example.org/knows> <http://example.org/gus> .",
        grafeo_engine::RdfFormat::NTriples,
        None,
    )
    .unwrap();
    db
}

/// `error` is the refusal of `bound`: Q002, 42000, the limit and the setting named.
fn assert_refused(case: &str, error: &Error, bound: LimitBound) {
    let message = error.to_string();
    assert_eq!(
        error.error_code(),
        ErrorCode::QuerySemantic,
        "{case}: {message}"
    );
    assert_eq!(
        GqlStatus::from(error).as_str(),
        "42000",
        "{case}: {message}"
    );
    assert!(message.contains(bound.name()), "{case}: {message}");
    assert!(message.contains(bound.setting()), "{case}: {message}");
}

#[test]
fn the_defaults_are_16_mib_and_16_kib() {
    let config = Config::in_memory();
    assert_eq!(config.max_query_length(), 16 << 20);
    assert_eq!(config.max_identifier_length(), 16 << 10);
}

#[test]
fn zero_parse_limits_are_refused() {
    assert!(matches!(
        Config::in_memory().with_max_query_length(0).validate(),
        Err(ConfigError::ZeroMaxQueryLength)
    ));
    assert!(matches!(
        Config::in_memory().with_max_identifier_length(0).validate(),
        Err(ConfigError::ZeroMaxIdentifierLength)
    ));
}

/// At its exact length a statement runs and returns what it does under the defaults; one byte
/// more is refused, in every language.
#[test]
fn every_language_refuses_a_statement_one_byte_over_the_limit() {
    let defaults = seeded(Config::in_memory());
    for (language, run, statement, _) in languages() {
        let limit = statement.len() + 8;
        let db = seeded(Config::in_memory().with_max_query_length(limit));
        let at = format!("{statement}{}", " ".repeat(8));
        let rows = run(&db, &at).unwrap_or_else(|e| panic!("{language}: {e}"));
        assert_eq!(
            rows.rows(),
            run(&defaults, statement).unwrap().rows(),
            "{language}: the same rows"
        );
        let error = run(&db, &format!("{at} ")).expect_err(language);
        assert_refused(language, &error, LimitBound::QueryLength);
        assert!(
            error.to_string().contains(&(limit + 1).to_string()),
            "{language}: the length found: {error}"
        );
    }
}

#[test]
fn every_language_refuses_a_name_one_byte_over_the_limit() {
    const NAME: usize = 12;
    for (language, run, _, shape) in languages() {
        let db = seeded(Config::in_memory().with_max_identifier_length(NAME));
        let at = shape.replace("{}", &"a".repeat(NAME));
        if let Err(error) = run(&db, &at) {
            // A Gremlin step of that name does not exist: the error is that, not the length.
            assert!(
                !error
                    .to_string()
                    .contains(LimitBound::IdentifierLength.name()),
                "{language}: a name of the limit is not too long: {error}"
            );
        }
        let over = shape.replace("{}", &"a".repeat(NAME + 1));
        let error = run(&db, &over).expect_err(language);
        assert_refused(language, &error, LimitBound::IdentifierLength);
    }
}

/// A large value passed as a parameter does not count; the same value written into the text
/// does.
#[test]
fn parameters_do_not_count_toward_the_statement_length() {
    let db = seeded(Config::in_memory().with_max_query_length(256));
    let big = "x".repeat(10_000);
    let params = HashMap::from([("bio".to_string(), Value::from(big.as_str()))]);
    db.execute_with_params(
        "MATCH (n:Person {name: 'Alix'}) SET n.bio = $bio RETURN n.name",
        params.clone(),
    )
    .unwrap();
    // Run again: the plan cache answers, with the same limits.
    db.execute_with_params(
        "MATCH (n:Person {name: 'Alix'}) SET n.bio = $bio RETURN n.name",
        params,
    )
    .unwrap();
    let error = db
        .execute(&format!(
            "MATCH (n:Person {{name: 'Alix'}}) SET n.bio = '{big}' RETURN n.name"
        ))
        .expect_err("the value in the text counts");
    assert_refused("inline value", &error, LimitBound::QueryLength);
    #[cfg(feature = "cypher")]
    db.execute_cypher_with_params(
        "MATCH (n:Person {name: 'Gus'}) SET n.bio = $bio RETURN n.name",
        HashMap::from([("bio".to_string(), Value::from(big.as_str()))]),
    )
    .unwrap();
}

/// `EXPLAIN` and `PROFILE` of a statement over the limit are refused as it is.
#[test]
fn explain_and_profile_are_refused_with_their_statement() {
    let statement = "MATCH (n:Person) RETURN n.name AS name";
    let db = seeded(Config::in_memory().with_max_query_length(statement.len() + 4));
    let long = format!("{statement}{}", " ".repeat(20));
    for prefix in ["EXPLAIN ", "PROFILE "] {
        let error = db.execute(&format!("{prefix}{long}")).expect_err(prefix);
        assert_refused(prefix, &error, LimitBound::QueryLength);
    }
}

/// A stored procedure's body is parsed when it runs, under the limits of the open that runs it:
/// created under the defaults, called after a reopen with a lower limit, it is refused.
#[test]
fn a_stored_procedure_body_is_held_to_the_limits_of_its_call() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("procedures.grafeo");
    let body = "MATCH (p:Person) RETURN p.name AS name ORDER BY name";
    {
        let db = seeded(Config::persistent(&path));
        db.execute(&format!(
            "CREATE PROCEDURE people() RETURNS (name STRING) AS {{ {body} }}"
        ))
        .unwrap();
        assert_eq!(
            db.execute("CALL people() YIELD name RETURN name")
                .unwrap()
                .rows()
                .len(),
            2
        );
        db.close().unwrap();
    }
    let db = GrafeoDB::with_config(Config::persistent(&path).with_max_query_length(40)).unwrap();
    let call = "CALL people() YIELD name RETURN name";
    assert!(call.len() <= 40);
    let error = db
        .execute(call)
        .expect_err("the body is longer than 40 bytes");
    assert_refused("procedure body", &error, LimitBound::QueryLength);
}
