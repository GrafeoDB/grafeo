//! GraphQL Query Language Parser.
//!
//! Implements a parser for the GraphQL query language specification.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use grafeo_adapters::query::graphql;
//!
//! let query = r#"
//!     query GetUser($id: ID!) {
//!         user(id: $id) {
//!             name
//!             email
//!             friends {
//!                 name
//!             }
//!         }
//!     }
//! "#;
//!
//! let doc = graphql::parse(query)?;
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

/// Parses a GraphQL query string into a Document AST.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid.
pub fn parse(query: &str) -> Result<Document> {
    grafeo_common::work::count(grafeo_common::work::Work::Parse, 1);
    let mut parser = Parser::new(query);
    parser.parse()
}
