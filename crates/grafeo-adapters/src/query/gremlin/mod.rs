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

/// Parses a Gremlin query string into an AST.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid.
pub fn parse(query: &str) -> Result<Statement> {
    grafeo_common::work::count(grafeo_common::work::Work::Parse, 1);
    let mut parser = Parser::new(query);
    parser.parse()
}
