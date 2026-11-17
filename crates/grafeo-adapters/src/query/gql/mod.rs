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

/// Parses a GQL query string into an AST.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid.
pub fn parse(query: &str) -> Result<Statement> {
    grafeo_common::work::count(grafeo_common::work::Work::Parse, 1);
    let mut parser = Parser::new(query);
    parser.parse()
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
