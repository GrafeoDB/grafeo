//! Cypher query language parser.
//!
//! Implements openCypher 9.0 specification.
//!
//! Cypher is a declarative graph query language that originated in Neo4j
//! and forms the basis for the ISO GQL standard.

pub mod ast;
mod lexer;
mod parser;

pub use ast::*;
pub use lexer::Lexer;
pub use parser::Parser;

use grafeo_common::utils::error::Result;

use crate::query::limits::{ParseLimits, with_long_names};

/// Parses a Cypher query string into an AST, under the default
/// [`ParseLimits`].
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid or too long.
pub fn parse(query: &str) -> Result<Statement> {
    parse_with(query, &ParseLimits::default())
}

/// Parses a Cypher query string into an AST, refusing a statement or a name
/// longer than `limits` allows.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid, or longer
/// than `limits` allows, or has a name longer than that (`GRAFEO-Q002`).
pub fn parse_with(query: &str, limits: &ParseLimits) -> Result<Statement> {
    limits.check_query(query)?;
    let max = limits.max_identifier_length;
    let mut parser = Parser::with_max_identifier_length(query, max);
    let result = parser.parse();
    with_long_names(
        result,
        parser.names(),
        || Lexer::scan_names(query, max),
        query,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_match() {
        let result = parse("MATCH (n) RETURN n");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_match_with_label() {
        let result = parse("MATCH (n:Person) RETURN n");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_match_with_relationship() {
        let result = parse("MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_create() {
        let result = parse("CREATE (n:Person {name: 'Alix'})");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_match_where() {
        let result = parse("MATCH (n:Person) WHERE n.age > 30 RETURN n.name");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_clause() {
        let result = parse("MATCH (n) WITH n.name AS name RETURN name");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_optional_match() {
        let result = parse("MATCH (a) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a, b");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_merge() {
        let result = parse("MERGE (n:Person {name: 'Alix'}) RETURN n");
        assert!(result.is_ok());
    }
}
