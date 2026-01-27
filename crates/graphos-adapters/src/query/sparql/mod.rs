//! SPARQL (W3C SPARQL 1.1) Query Language Parser.
//!
//! Implements the SPARQL 1.1 Query Language specification for querying RDF data.
//!
//! # Example
//!
//! ```ignore
//! use graphos_adapters::query::sparql;
//!
//! let query = r#"
//!     PREFIX foaf: <http://xmlns.com/foaf/0.1/>
//!     SELECT ?name
//!     WHERE { ?x foaf:name ?name }
//! "#;
//!
//! let ast = sparql::parse(query)?;
//! ```

pub mod ast;
mod lexer;
mod parser;

pub use ast::*;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;

use graphos_common::utils::error::Result;

/// Parses a SPARQL query string into an AST.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid.
pub fn parse(query: &str) -> Result<Query> {
    let mut parser = Parser::new(query);
    parser.parse()
}
