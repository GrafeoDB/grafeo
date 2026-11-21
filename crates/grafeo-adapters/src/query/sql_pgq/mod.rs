//! SQL/PGQ query language parser.
//!
//! Implements SQL:2023 GRAPH_TABLE (ISO/IEC 9075-16) for SQL-native
//! graph pattern matching. The inner MATCH clause uses GQL pattern
//! syntax, so SQL developers can query graphs without learning GQL.
//!
//! # Example
//!
//! ```sql
//! SELECT *
//! FROM GRAPH_TABLE (
//!     MATCH (a:Person)-[e:KNOWS]->(b:Person)
//!     COLUMNS (a.name AS person, e.since AS year, b.name AS friend)
//! ) result
//! WHERE result.person = 'Alix'
//! ORDER BY result.year DESC
//! LIMIT 10;
//! ```

pub mod ast;
mod lexer;
mod parser;

pub use ast::*;
pub use lexer::Lexer;
pub use parser::Parser;

use grafeo_common::utils::error::Result;

use crate::query::limits::{ParseLimits, with_long_names};

/// Parses a SQL/PGQ query string into an AST, under the default
/// [`ParseLimits`].
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid or too long.
pub fn parse(query: &str) -> Result<Statement> {
    parse_with(query, &ParseLimits::default())
}

/// Parses a SQL/PGQ query string into an AST, refusing a statement or a name
/// longer than `limits` allows, in its `GRAPH_TABLE`s too.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid, or longer
/// than `limits` allows, or has a name longer than that (`GRAPH_TABLE`'s
/// included, `GRAFEO-Q002`).
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
    fn test_parse_basic_graph_table() {
        let result = parse(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            )",
        );
        assert!(result.is_ok(), "Parse failed: {result:?}");
    }

    #[test]
    fn test_parse_graph_table_with_relationship() {
        let result = parse(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            )",
        );
        assert!(result.is_ok(), "Parse failed: {result:?}");
    }

    #[test]
    fn test_parse_graph_table_with_where_and_order() {
        let result = parse(
            "SELECT g.person FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, e.since AS year)
            ) AS g
            WHERE g.year > 2020
            ORDER BY g.year DESC
            LIMIT 10",
        );
        assert!(result.is_ok(), "Parse failed: {result:?}");
    }

    #[test]
    fn test_parse_syntax_error() {
        let result = parse("SELECT FROM");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_create_property_graph() {
        let result = parse(
            "CREATE PROPERTY GRAPH TestGraph
             NODE TABLES (
                 Person (id BIGINT PRIMARY KEY, name VARCHAR)
             )",
        );
        assert!(result.is_ok(), "Parse failed: {result:?}");
        assert!(matches!(result.unwrap(), Statement::CreatePropertyGraph(_)));
    }
}
