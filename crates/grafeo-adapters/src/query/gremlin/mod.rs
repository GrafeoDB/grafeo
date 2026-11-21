//! Gremlin (Apache TinkerPop) Query Language Parser.
//!
//! Implements a parser for the Gremlin graph traversal language.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use grafeo_adapters::query::gremlin;
//!
//! let query = "g.V().hasLabel('Person').out('knows').values('name')";
//! let ast = gremlin::parse(query)?;
//! # Ok(())
//! # }
//! ```

pub mod ast;
mod lexer;
mod parser;

pub use ast::*;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;

use grafeo_common::utils::error::Result;

use crate::query::limits::ParseLimits;

/// Parses a Gremlin query string into an AST, under the default
/// [`ParseLimits`].
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid or too long.
pub fn parse(query: &str) -> Result<Statement> {
    parse_with(query, &ParseLimits::default())
}

/// Parses a Gremlin query string into an AST, refusing a statement or a name
/// longer than `limits` allows.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid, or longer
/// than `limits` allows, or has a name longer than that (`GRAFEO-Q002`).
pub fn parse_with(query: &str, limits: &ParseLimits) -> Result<Statement> {
    limits.check_query(query)?;
    let mut parser = Parser::new(query);
    // The parser reads every token first, so a long name is found before a
    // syntax error it may cause.
    if let Some(error) = parser.names(limits.max_identifier_length).error(query) {
        return Err(error);
    }
    parser.parse()
}
