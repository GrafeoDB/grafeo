//! GQL (Graph Query Language) parser.
//!
//! Implements ISO/IEC 39075:2024 GQL standard.

mod ast;
mod lexer;
mod parser;

pub use ast::*;
pub use lexer::Lexer;
pub use parser::Parser;

use graphos_common::utils::error::{QueryError, QueryErrorKind, Result, SourceSpan};

/// Parses a GQL query string into an AST.
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
        // This is a placeholder - actual parsing will be implemented
        // let result = parse("MATCH (n) RETURN n");
        // assert!(result.is_ok());
    }
}
