//! SPARQL (W3C SPARQL 1.1) Query Language Parser.
//!
//! Implements the SPARQL 1.1 Query Language specification for querying RDF data.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use grafeo_adapters::query::sparql;
//!
//! let query = r#"
//!     PREFIX foaf: <http://xmlns.com/foaf/0.1/>
//!     SELECT ?name
//!     WHERE { ?x foaf:name ?name }
//! "#;
//!
//! let ast = sparql::parse(query)?;
//! # Ok(())
//! # }
//! ```

pub mod ast;
mod lexer;
mod parser;
pub use parser::PREDEFINED_PREFIXES;

pub use ast::*;
pub use lexer::{Lexer, Token, TokenKind};
pub use parser::Parser;

use grafeo_common::utils::error::Result;

use crate::query::limits::{ParseLimits, with_long_names};

/// Parses a SPARQL query string into an AST, under the default
/// [`ParseLimits`].
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid or too long.
pub fn parse(query: &str) -> Result<Query> {
    parse_with(query, &ParseLimits::default())
}

/// Parses a SPARQL query string into an AST, refusing a statement or a name
/// longer than `limits` allows.
///
/// # Errors
///
/// Returns a `QueryError` if the query is syntactically invalid, or longer
/// than `limits` allows, or has a name longer than that (`GRAFEO-Q002`).
pub fn parse_with(query: &str, limits: &ParseLimits) -> Result<Query> {
    grafeo_common::work::count(grafeo_common::work::Work::Parse, 1);
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
