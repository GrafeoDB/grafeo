//! GraphQL Query Language Parser.
//!
//! Implements a parser for the GraphQL query language specification.
//!
//! # Example
//!
//! ```ignore
//! use graphos_adapters::query::graphql;
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
//! ```

pub mod ast;
mod lexer;
mod parser;

pub use ast::*;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;

use graphos_common::utils::error::Result;

/// Parses a GraphQL query string into a Document AST.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid.
pub fn parse(query: &str) -> Result<Document> {
    let mut parser = Parser::new(query);
    parser.parse()
}
