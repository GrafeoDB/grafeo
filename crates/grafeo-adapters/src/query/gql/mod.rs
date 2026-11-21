//! GQL (Graph Query Language) parser.
//!
//! Implements ISO/IEC 39075:2024 GQL standard.

pub mod ast;
mod lexer;
mod parser;

pub use ast::*;
pub use lexer::Lexer;
pub use parser::Parser;

use grafeo_common::utils::error::Result;

use crate::query::limits::{ParseLimits, with_long_names};

/// Parses a GQL query string into an AST, under the default [`ParseLimits`].
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid or too long.
pub fn parse(query: &str) -> Result<Statement> {
    parse_with(query, &ParseLimits::default())
}

/// Parses a GQL query string into an AST, refusing a statement or a name
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
    fn parse_simple_match() {
        let result = parse("MATCH (n) RETURN n");
        assert!(result.is_ok());
    }
}
