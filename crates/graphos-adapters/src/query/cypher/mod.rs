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

use graphos_common::utils::error::Result;

/// Parses a Cypher query string into an AST.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid.
pub fn parse(query: &str) -> Result<Statement> {
    let mut parser = Parser::new(query);
    parser.parse()
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
        let result = parse("CREATE (n:Person {name: 'Alice'})");
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
        let result = parse("MERGE (n:Person {name: 'Alice'}) RETURN n");
        assert!(result.is_ok());
    }
}
