//! The parse limits (#492): every parser refuses a statement longer than `max_query_length`
//! bytes and a name longer than `max_identifier_length` bytes, as written between its
//! delimiters, with `GRAFEO-Q002`, the figures and the setting that moves the limit. A
//! statement or a name of exactly the limit parses. String literals and SPARQL IRIs are not
//! names: only the statement length bounds them. A too-long name is reported even when the
//! statement also has a syntax error, and the first one by position is the one named.
//!
//! ```bash
//! cargo test -p grafeo-adapters --all-features --test main parse_limits::
//! ```

use grafeo_adapters::query::limits::{
    DEFAULT_MAX_IDENTIFIER_LENGTH, DEFAULT_MAX_QUERY_LENGTH, ParseLimits,
};
use grafeo_common::utils::error::{Error, ErrorCode, LimitBound};

type Parse = fn(&str, &ParseLimits) -> Result<(), Error>;

/// Each language's `parse_with`, a statement that parses, and its comment start.
fn languages() -> Vec<(&'static str, Parse, &'static str, &'static str)> {
    vec![
        #[cfg(feature = "gql")]
        (
            "gql",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "MATCH (n:Person) RETURN n.name",
            "// ",
        ),
        #[cfg(feature = "cypher")]
        (
            "cypher",
            |q, l| grafeo_adapters::query::cypher::parse_with(q, l).map(drop),
            "MATCH (n:Person) RETURN n.name",
            "// ",
        ),
        #[cfg(feature = "sql-pgq")]
        (
            "sql-pgq",
            |q, l| grafeo_adapters::query::sql_pgq::parse_with(q, l).map(drop),
            "SELECT * FROM GRAPH_TABLE (MATCH (n:Person) COLUMNS (n.name AS name))",
            "-- ",
        ),
        #[cfg(feature = "sparql")]
        (
            "sparql",
            |q, l| grafeo_adapters::query::sparql::parse_with(q, l).map(drop),
            "SELECT ?s WHERE { ?s ?p ?o }",
            "# ",
        ),
        #[cfg(feature = "gremlin")]
        (
            "gremlin",
            |q, l| grafeo_adapters::query::gremlin::parse_with(q, l).map(drop),
            "g.V().hasLabel('Person').values('name')",
            "",
        ),
        #[cfg(feature = "graphql")]
        (
            "graphql",
            |q, l| grafeo_adapters::query::graphql::parse_with(q, l).map(drop),
            "{ Person { name } }",
            "# ",
        ),
    ]
}

fn limits(query: usize, identifier: usize) -> ParseLimits {
    ParseLimits {
        max_query_length: query,
        max_identifier_length: identifier,
    }
}

/// `result` is the refusal of `bound` (`GRAFEO-Q002`), naming `figures` and the setting.
fn assert_refused(case: &str, result: Result<(), Error>, bound: LimitBound, figures: &[usize]) {
    let error = result.expect_err(case);
    let message = error.to_string();
    assert_eq!(
        error.error_code(),
        ErrorCode::QuerySemantic,
        "{case}: {message}"
    );
    assert!(
        message.contains(bound.name()),
        "{case}: names the limit: {message}"
    );
    assert!(
        message.contains(bound.setting()),
        "{case}: names the setting: {message}"
    );
    for figure in figures {
        assert!(
            message.contains(&figure.to_string()),
            "{case}: names {figure}: {message}"
        );
    }
}

#[test]
fn the_defaults_are_16_mib_and_16_kib() {
    assert_eq!(DEFAULT_MAX_QUERY_LENGTH, 16 << 20);
    assert_eq!(DEFAULT_MAX_IDENTIFIER_LENGTH, 16 << 10);
    assert_eq!(
        ParseLimits::default(),
        limits(DEFAULT_MAX_QUERY_LENGTH, DEFAULT_MAX_IDENTIFIER_LENGTH)
    );
}

/// A statement of exactly the limit parses; one byte over is refused, whitespace and comments
/// counted.
#[test]
fn a_statement_one_byte_over_its_limit_is_refused_in_every_language() {
    for (language, parse, statement, comment) in languages() {
        let limit = statement.len() + 20;
        let padded = format!("{statement}{}", " ".repeat(limit - statement.len()));
        assert_eq!(padded.len(), limit);
        parse(&padded, &limits(limit, 1024)).unwrap_or_else(|e| panic!("{language}: {e}"));
        let over = format!("{padded} ");
        assert_refused(
            &format!("{language}: one byte over"),
            parse(&over, &limits(limit, 1024)),
            LimitBound::QueryLength,
            &[limit + 1, limit],
        );
        if !comment.is_empty() {
            // A leading comment counts.
            let commented = format!("{comment}{}\n{statement}", "c".repeat(30));
            assert_refused(
                &format!("{language}: a leading comment"),
                parse(&commented, &limits(statement.len() + 10, 1024)),
                LimitBound::QueryLength,
                &[],
            );
        }
        // The defaults take the statement.
        parse(statement, &ParseLimits::default()).unwrap_or_else(|e| panic!("{language}: {e}"));
    }
}

/// The length is in bytes: 'é' is two.
#[cfg(feature = "gql")]
#[test]
fn the_statement_length_counts_bytes_not_characters() {
    let statement = "RETURN 'éééé' AS s";
    let bytes = statement.len();
    let characters = statement.chars().count();
    assert_eq!(bytes, characters + 4);
    grafeo_adapters::query::gql::parse_with(statement, &limits(bytes, 1024)).unwrap();
    assert_refused(
        "multibyte at its character count",
        grafeo_adapters::query::gql::parse_with(statement, &limits(characters, 1024)).map(drop),
        LimitBound::QueryLength,
        &[bytes],
    );
}

/// The identifier limit of these tests: longer than every keyword the statements use.
const NAME: usize = 10;

/// Plain names, at the limit and one byte over, in each language: `{}` is the name.
fn plain_names() -> Vec<(&'static str, Parse, &'static str)> {
    vec![
        #[cfg(feature = "gql")]
        (
            "gql variable",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "MATCH ({}) RETURN {}",
        ),
        #[cfg(feature = "gql")]
        (
            "gql label",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "MATCH (n:{}) RETURN n",
        ),
        #[cfg(feature = "cypher")]
        (
            "cypher property key",
            |q, l| grafeo_adapters::query::cypher::parse_with(q, l).map(drop),
            "MATCH (n) RETURN n.{}",
        ),
        #[cfg(feature = "cypher")]
        (
            "cypher delimited, an escaped backtick counted as written",
            |q, l| grafeo_adapters::query::cypher::parse_with(q, l).map(drop),
            "MATCH (`{}`) RETURN 1",
        ),
        #[cfg(feature = "sql-pgq")]
        (
            "sql-pgq column alias",
            |q, l| grafeo_adapters::query::sql_pgq::parse_with(q, l).map(drop),
            "SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS (n.v AS {}))",
        ),
        #[cfg(feature = "sparql")]
        (
            "sparql variable, without its ?",
            |q, l| grafeo_adapters::query::sparql::parse_with(q, l).map(drop),
            "SELECT ?{} WHERE { ?{} ?p ?o }",
        ),
        #[cfg(feature = "graphql")]
        (
            "graphql field",
            |q, l| grafeo_adapters::query::graphql::parse_with(q, l).map(drop),
            "{ Person { {} } }",
        ),
    ]
}

#[test]
fn a_name_one_byte_over_its_limit_is_refused() {
    for (case, parse, shape) in plain_names() {
        // The backtick case holds an escaped backtick, two bytes as written.
        let at = if case.contains("backtick") {
            format!("ab``{}", "c".repeat(NAME - 4))
        } else {
            "a".repeat(NAME)
        };
        assert_eq!(at.len(), NAME);
        let statement = shape.replace("{}", &at);
        parse(&statement, &limits(1 << 20, NAME)).unwrap_or_else(|e| panic!("{case}: {e}"));
        let over = shape.replace("{}", &format!("{at}x"));
        assert_refused(
            case,
            parse(&over, &limits(1 << 20, NAME)),
            LimitBound::IdentifierLength,
            &[NAME + 1, NAME],
        );
    }
}

/// The other kinds of names, a short one and a far longer one.
#[test]
fn every_kind_of_name_is_bounded() {
    let cases: Vec<(&'static str, Parse, &'static str)> = vec![
        #[cfg(feature = "gql")]
        (
            "gql parameter",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "MATCH (n) WHERE n.v = ${} RETURN n",
        ),
        #[cfg(feature = "gql")]
        (
            "gql edge type",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "MATCH (a)-[:{}]->(b) RETURN a",
        ),
        #[cfg(feature = "cypher")]
        (
            "cypher parameter",
            |q, l| grafeo_adapters::query::cypher::parse_with(q, l).map(drop),
            "MATCH (n) WHERE n.v = ${} RETURN n",
        ),
        #[cfg(feature = "sql-pgq")]
        (
            "sql-pgq delimited identifier",
            |q, l| grafeo_adapters::query::sql_pgq::parse_with(q, l).map(drop),
            "SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS (n.v AS \"{}\"))",
        ),
        #[cfg(feature = "sql-pgq")]
        (
            "a name inside a GRAPH_TABLE's MATCH",
            |q, l| grafeo_adapters::query::sql_pgq::parse_with(q, l).map(drop),
            "SELECT * FROM GRAPH_TABLE (MATCH (n:{}) COLUMNS (n.v AS v))",
        ),
        #[cfg(feature = "sparql")]
        (
            "sparql prefixed name",
            |q, l| grafeo_adapters::query::sparql::parse_with(q, l).map(drop),
            "PREFIX ex: <http://example.org/> SELECT ?s WHERE { ?s ex:{} ?o }",
        ),
        #[cfg(feature = "sparql")]
        (
            "sparql blank node label",
            |q, l| grafeo_adapters::query::sparql::parse_with(q, l).map(drop),
            "SELECT ?p WHERE { _:{} ?p ?o }",
        ),
        #[cfg(feature = "gremlin")]
        (
            "gremlin step name",
            |q, l| grafeo_adapters::query::gremlin::parse_with(q, l).map(drop),
            "g.V().{}()",
        ),
    ];
    for (case, parse, shape) in cases {
        let short = shape.replace("{}", "abcd");
        let long = shape.replace("{}", &"a".repeat(4 * NAME));
        // A short one parses, unless it is a name the language does not know (a Gremlin step):
        // then it fails for that, never for its length.
        if let Err(error) = parse(&short, &limits(1 << 20, NAME)) {
            assert!(
                !error
                    .to_string()
                    .contains(LimitBound::IdentifierLength.name()),
                "{case}: a short name is not too long: {error}"
            );
        }
        // A SPARQL prefixed name is one name, its prefix and colon included (`ex:`).
        let written = if case == "sparql prefixed name" {
            4 * NAME + 3
        } else {
            4 * NAME
        };
        assert_refused(
            case,
            parse(&long, &limits(1 << 20, NAME)),
            LimitBound::IdentifierLength,
            &[written],
        );
    }
}

/// String literals and IRIs are not names: they parse under a small identifier limit.
#[test]
fn literals_and_iris_are_not_names() {
    let long = "x".repeat(200);
    #[cfg(feature = "gql")]
    grafeo_adapters::query::gql::parse_with(
        &format!("RETURN '{long}' AS s"),
        &limits(1 << 20, NAME),
    )
    .unwrap();
    #[cfg(feature = "cypher")]
    grafeo_adapters::query::cypher::parse_with(
        &format!("RETURN '{long}' AS s"),
        &limits(1 << 20, NAME),
    )
    .unwrap();
    #[cfg(feature = "sparql")]
    grafeo_adapters::query::sparql::parse_with(
        &format!("SELECT ?s WHERE {{ ?s <http://example.org/{long}> ?o }}"),
        &limits(1 << 20, NAME),
    )
    .unwrap();
    let _ = long;
}

/// A too-long name is the error even when the statement also has a syntax error, and the
/// first long name by position is the one named.
#[cfg(feature = "gql")]
#[test]
fn the_first_long_name_is_reported_over_a_syntax_error() {
    let gql =
        |q: &str| grafeo_adapters::query::gql::parse_with(q, &limits(1 << 20, NAME)).map(drop);
    assert_refused(
        "after an unclosed node pattern",
        gql(&format!("MATCH (n RETURN {}", "a".repeat(20))),
        LimitBound::IdentifierLength,
        &[20],
    );
    let both = format!(
        "MATCH ({})-[:{}]->(b) RETURN b",
        "a".repeat(16),
        "b".repeat(32)
    );
    let message = gql(&both).unwrap_err().to_string();
    assert!(message.contains("16"), "the first name: {message}");
    assert!(!message.contains("32"), "not the second: {message}");
}

/// GQL's double-quoted names (ISO's delimited identifiers, which stand where a string literal
/// cannot: a label, an edge type, a name a clause binds), in GQL and inside a SQL/PGQ
/// `GRAPH_TABLE`'s `MATCH`: `{}` is the name between its quotes.
fn double_quoted_names() -> Vec<(&'static str, Parse, &'static str)> {
    vec![
        #[cfg(feature = "gql")]
        (
            "gql double-quoted label",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "MATCH (n:\"{}\") RETURN n",
        ),
        #[cfg(feature = "gql")]
        (
            "gql double-quoted edge type",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "MATCH (a)-[:\"{}\"]->(b) RETURN a",
        ),
        #[cfg(feature = "gql")]
        (
            "gql double-quoted LET variable",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "LET \"{}\" = 1 RETURN 1",
        ),
        #[cfg(feature = "gql")]
        (
            "gql double-quoted alias",
            |q, l| grafeo_adapters::query::gql::parse_with(q, l).map(drop),
            "MATCH (n) RETURN n AS \"{}\"",
        ),
        #[cfg(feature = "sql-pgq")]
        (
            "a double-quoted label inside a GRAPH_TABLE's MATCH",
            |q, l| grafeo_adapters::query::sql_pgq::parse_with(q, l).map(drop),
            "SELECT * FROM GRAPH_TABLE (MATCH (n:\"{}\") COLUMNS (n.v AS v))",
        ),
    ]
}

/// A double-quoted name one byte over the limit is refused like any other name, its length
/// counted between its quotes.
#[test]
fn a_double_quoted_name_one_byte_over_its_limit_is_refused() {
    for (case, parse, shape) in double_quoted_names() {
        let over = shape.replace("{}", &"a".repeat(NAME + 1));
        assert_refused(
            case,
            parse(&over, &limits(1 << 20, NAME)),
            LimitBound::IdentifierLength,
            &[NAME + 1, NAME],
        );
    }
}

/// A double-quoted name of exactly the limit, between its quotes, parses.
#[test]
fn a_double_quoted_name_of_exactly_its_limit_parses() {
    for (case, parse, shape) in double_quoted_names() {
        let at = shape.replace("{}", &"a".repeat(NAME));
        parse(&at, &limits(1 << 20, NAME)).unwrap_or_else(|e| panic!("{case}: {e}"));
    }
}

/// A SPARQL prefixed name is one name, counted whole as written, its colon included: one of
/// 11 bytes (a prefix of 5, the colon, a local part of 5) is over a limit of 10, though neither
/// half is over 5.
#[cfg(feature = "sparql")]
#[test]
fn a_sparql_prefixed_name_over_its_limit_as_a_whole_is_refused() {
    let name = "abcde:fghij";
    assert_eq!(name.len(), NAME + 1);
    let query = format!("PREFIX abcde: <http://example.org/> SELECT ?s WHERE {{ ?s {name} ?o }}");
    assert_refused(
        "a prefixed name of two short halves",
        grafeo_adapters::query::sparql::parse_with(&query, &limits(1 << 20, NAME)).map(drop),
        LimitBound::IdentifierLength,
        &[NAME + 1, NAME],
    );
}

/// A SPARQL prefixed name of exactly the limit, its colon included, parses.
#[cfg(feature = "sparql")]
#[test]
fn a_sparql_prefixed_name_of_exactly_its_limit_parses() {
    let name = "abcd:fghij";
    assert_eq!(name.len(), NAME);
    let query = format!("PREFIX abcd: <http://example.org/> SELECT ?s WHERE {{ ?s {name} ?o }}");
    grafeo_adapters::query::sparql::parse_with(&query, &limits(1 << 20, NAME)).unwrap();
}

/// A long parameter name inside a `GRAPH_TABLE`'s `MATCH` is the error over a syntax error after
/// it, as in GQL.
#[cfg(feature = "sql-pgq")]
#[test]
fn a_long_parameter_in_a_graph_table_is_reported_over_a_syntax_error() {
    for (case, shape) in [
        (
            "a stray parenthesis after the GRAPH_TABLE",
            "SELECT * FROM GRAPH_TABLE (MATCH (n) WHERE n.v = ${} COLUMNS (n.v AS v)) )",
        ),
        (
            "a syntax error inside the MATCH",
            "SELECT * FROM GRAPH_TABLE (MATCH (n) WHERE n.v = ${} AND ) COLUMNS (n.v AS v))",
        ),
        (
            "a backtick-quoted parameter name",
            "SELECT * FROM GRAPH_TABLE (MATCH (n) WHERE n.v = $`{}` AND ) COLUMNS (n.v AS v))",
        ),
    ] {
        let statement = shape.replace("{}", &"a".repeat(4 * NAME));
        assert_refused(
            case,
            grafeo_adapters::query::sql_pgq::parse_with(&statement, &limits(1 << 20, NAME))
                .map(drop),
            LimitBound::IdentifierLength,
            &[4 * NAME],
        );
    }
}
