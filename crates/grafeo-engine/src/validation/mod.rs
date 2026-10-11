//! SHACL validation engine.
//!
//! This module provides the engine-level SHACL validation integration,
//! including SPARQL constraint dispatch via `SessionSparqlExecutor`.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_core::graph::rdf::shacl::{ShaclError, SparqlExecutor, ValidationReport};
use grafeo_core::graph::rdf::{RdfStore, Term};

use crate::session::Session;

/// SPARQL executor backed by a `Session`.
///
/// Implements the `SparqlExecutor` trait from grafeo-core, allowing
/// SHACL-SPARQL constraints to be evaluated via the engine's SPARQL pipeline.
pub struct SessionSparqlExecutor<'a> {
    session: &'a Session,
    /// When set, wraps queries in `GRAPH <name> { ... }` to scope execution
    /// to a named data graph (used by `validate_shacl_graph`).
    graph_name: Option<String>,
}

impl<'a> SessionSparqlExecutor<'a> {
    /// Creates a new executor wrapping the given session (default graph scope).
    pub fn new(session: &'a Session) -> Self {
        Self {
            session,
            graph_name: None,
        }
    }

    /// Creates an executor scoped to a named graph.
    pub fn with_graph(session: &'a Session, graph_name: String) -> Self {
        Self {
            session,
            graph_name: Some(graph_name),
        }
    }
}

impl SparqlExecutor for SessionSparqlExecutor<'_> {
    fn execute(
        &self,
        query: &str,
        this_binding: &Term,
    ) -> Result<Vec<HashMap<String, Term>>, ShaclError> {
        // Substitute $this with the N-Triples representation of the focus node.
        // IRIs are validated to prevent SPARQL injection via crafted terms.
        let this_str = match this_binding {
            Term::Iri(iri) => {
                let iri_str = iri.as_str();
                if !is_safe_iri(iri_str) {
                    return Err(ShaclError::SparqlError(format!(
                        "IRI contains characters unsafe for SPARQL embedding: {iri_str}"
                    )));
                }
                format!("<{iri_str}>")
            }
            Term::BlankNode(bnode) => format!("_:{}", bnode.id()),
            Term::Literal(lit) => {
                let escaped = escape_ntriples(lit.value());
                if let Some(lang) = lit.language() {
                    format!("\"{escaped}\"@{}", lang)
                } else if lit.datatype() != "http://www.w3.org/2001/XMLSchema#string" {
                    let dt = lit.datatype();
                    if !is_safe_iri(dt) {
                        return Err(ShaclError::SparqlError(format!(
                            "Datatype IRI contains unsafe characters: {dt}"
                        )));
                    }
                    format!("\"{escaped}\"^^<{dt}>")
                } else {
                    format!("\"{escaped}\"")
                }
            }
            _ => return Ok(Vec::new()),
        };

        let mut substituted = query.replace("$this", &this_str);

        // Scope to named data graph via FROM clause when configured
        if let Some(ref graph) = self.graph_name {
            if !is_safe_iri(graph) {
                return Err(ShaclError::SparqlError(format!(
                    "Graph name contains characters unsafe for SPARQL embedding: {graph}"
                )));
            }
            // Case-insensitive WHERE detection
            let upper = substituted.to_uppercase();
            if let Some(pos) = upper.find("WHERE") {
                substituted.insert_str(pos, &format!("FROM <{graph}> "));
            }
        }

        let result = self
            .session
            .execute_sparql(&substituted)
            .map_err(|e| ShaclError::SparqlError(e.to_string()))?;

        // Convert QueryResult rows to Vec<HashMap<String, Term>>
        let columns = &result.columns;
        let mut rows = Vec::new();
        for row in result.rows() {
            let mut map = HashMap::new();
            for (i, col) in columns.iter().enumerate() {
                if let Some(value) = row.get(i)
                    && let Some(term) = value_to_term(value)
                {
                    map.insert(col.clone(), term);
                }
            }
            rows.push(map);
        }

        Ok(rows)
    }
}

/// Escapes a string for N-Triples literal representation.
/// Checks whether an IRI string is safe for embedding in a SPARQL query.
///
/// Rejects characters that could break out of `<...>` IRI delimiters or
/// inject additional SPARQL syntax.
fn is_safe_iri(iri: &str) -> bool {
    !iri.chars().any(|c| {
        matches!(c, '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`') || c.is_whitespace()
    })
}

fn escape_ntriples(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

/// Converts a `grafeo_common::types::Value` to an RDF `Term`.
fn value_to_term(value: &Value) -> Option<Term> {
    match value {
        Value::Null => None,
        Value::String(s) => {
            if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                Some(Term::iri(s.as_str()))
            } else {
                Some(Term::literal(s.as_str()))
            }
        }
        Value::Int64(n) => Some(Term::typed_literal(
            n.to_string(),
            "http://www.w3.org/2001/XMLSchema#integer",
        )),
        Value::Float64(f) => Some(Term::typed_literal(
            f.to_string(),
            "http://www.w3.org/2001/XMLSchema#double",
        )),
        Value::Bool(b) => Some(Term::typed_literal(
            if *b { "true" } else { "false" },
            "http://www.w3.org/2001/XMLSchema#boolean",
        )),
        _ => Some(Term::literal(value.to_string())),
    }
}

/// Validates the default graph against shapes in a named graph.
///
/// This is the engine-level entry point that wires up the SPARQL executor.
///
/// # Errors
///
/// Returns an error if shape parsing fails, the shapes graph doesn't exist,
/// or a SPARQL constraint fails.
pub fn validate_shacl(
    session: &Session,
    rdf_store: &Arc<RdfStore>,
    shapes_graph_name: &str,
) -> grafeo_common::utils::error::Result<ValidationReport> {
    let shapes_store = rdf_store.graph(shapes_graph_name).ok_or_else(|| {
        grafeo_common::utils::error::Error::InvalidValue(format!(
            "there is no named graph '{shapes_graph_name}' to read the shapes from"
        ))
    })?;

    let executor = SessionSparqlExecutor::new(session);
    grafeo_core::graph::rdf::shacl::validate(rdf_store, &shapes_store, Some(&executor))
        .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── escape_ntriples ─────────────────────────────────────────────

    #[test]
    fn test_escape_ntriples_individual_chars() {
        assert_eq!(escape_ntriples(r"\"), r"\\");
        assert_eq!(escape_ntriples("\""), "\\\"");
        assert_eq!(escape_ntriples("\n"), "\\n");
        assert_eq!(escape_ntriples("\r"), "\\r");
        assert_eq!(escape_ntriples("\t"), "\\t");
        assert_eq!(escape_ntriples("\\\"\n\r\t"), "\\\\\\\"\\n\\r\\t");
    }

    #[test]
    fn test_escape_ntriples_passthrough_and_mixed() {
        assert_eq!(escape_ntriples(""), "");
        assert_eq!(escape_ntriples("hello"), "hello");
        assert_eq!(
            escape_ntriples("Gus lives in Amsterdam"),
            "Gus lives in Amsterdam"
        );
        assert_eq!(
            escape_ntriples("Alix said \"hello\"\npath: C:\\data"),
            "Alix said \\\"hello\\\"\\npath: C:\\\\data"
        );
    }

    // ── value_to_term ───────────────────────────────────────────────

    #[test]
    fn test_value_to_term_null_and_primitives() {
        assert_eq!(value_to_term(&Value::Null), None);

        assert_eq!(
            value_to_term(&Value::Int64(42)),
            Some(Term::typed_literal(
                "42",
                "http://www.w3.org/2001/XMLSchema#integer"
            ))
        );
        assert_eq!(
            value_to_term(&Value::Float64(2.72)),
            Some(Term::typed_literal(
                "2.72",
                "http://www.w3.org/2001/XMLSchema#double"
            ))
        );
        assert_eq!(
            value_to_term(&Value::Bool(true)),
            Some(Term::typed_literal(
                "true",
                "http://www.w3.org/2001/XMLSchema#boolean"
            ))
        );
    }

    #[test]
    fn test_value_to_term_strings_and_uris() {
        // Plain string
        assert_eq!(
            value_to_term(&Value::String("hello".into())),
            Some(Term::literal("hello"))
        );
        // Non-URI string
        assert_eq!(
            value_to_term(&Value::String("Amsterdam".into())),
            Some(Term::literal("Amsterdam"))
        );
        // http, https, and urn URIs
        assert_eq!(
            value_to_term(&Value::String("http://example.org/Alix".into())),
            Some(Term::iri("http://example.org/Alix"))
        );
        assert_eq!(
            value_to_term(&Value::String("https://schema.org/Person".into())),
            Some(Term::iri("https://schema.org/Person"))
        );
        assert_eq!(
            value_to_term(&Value::String("urn:isbn:0451450523".into())),
            Some(Term::iri("urn:isbn:0451450523"))
        );
    }

    #[test]
    fn test_value_to_term_edge_cases() {
        // Negative int
        assert_eq!(
            value_to_term(&Value::Int64(-7)),
            Some(Term::typed_literal(
                "-7",
                "http://www.w3.org/2001/XMLSchema#integer"
            ))
        );
        // Zero
        assert_eq!(
            value_to_term(&Value::Int64(0)),
            Some(Term::typed_literal(
                "0",
                "http://www.w3.org/2001/XMLSchema#integer"
            ))
        );
        // Bool false
        assert_eq!(
            value_to_term(&Value::Bool(false)),
            Some(Term::typed_literal(
                "false",
                "http://www.w3.org/2001/XMLSchema#boolean"
            ))
        );
        // Float with no fractional part
        assert_eq!(
            value_to_term(&Value::Float64(1.0)),
            Some(Term::typed_literal(
                "1",
                "http://www.w3.org/2001/XMLSchema#double"
            ))
        );
        // Fallback: types not explicitly handled use Value::to_string()
        let result = value_to_term(&Value::Bytes(vec![0xCA, 0xFE].into()));
        assert!(result.is_some());
        assert!(result.unwrap().is_literal());
    }
}
