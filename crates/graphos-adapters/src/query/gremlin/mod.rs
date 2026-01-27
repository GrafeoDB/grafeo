//! Gremlin (Apache TinkerPop) Query Language Parser.
//!
//! Implements a parser for the Gremlin graph traversal language.
//!
//! # Example
//!
//! ```ignore
//! use graphos_adapters::query::gremlin;
//!
//! let query = "g.V().hasLabel('Person').out('knows').values('name')";
//! let ast = gremlin::parse(query)?;
//! ```

pub mod ast;
mod lexer;
mod parser;

pub use ast::*;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;

use graphos_common::utils::error::Result;

/// Parses a Gremlin query string into an AST.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid.
pub fn parse(query: &str) -> Result<Statement> {
    let mut parser = Parser::new(query);
    parser.parse()
}
