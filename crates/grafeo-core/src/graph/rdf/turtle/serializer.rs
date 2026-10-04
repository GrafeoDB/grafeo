//! Turtle serializer.
//!
//! Converts RDF triples to [W3C Turtle](https://www.w3.org/TR/turtle/) format.
//! Groups triples by subject, uses `;` for predicate lists and `,` for object lists,
//! and emits `@prefix` declarations for common namespaces.

use crate::graph::rdf::term::{Iri, Literal, Term};
use crate::graph::rdf::triple::Triple;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;

/// Well-known namespace prefixes automatically detected in output.
const WELL_KNOWN_PREFIXES: &[(&str, &str)] = &[
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("foaf", "http://xmlns.com/foaf/0.1/"),
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("dcterms", "http://purl.org/dc/terms/"),
    ("skos", "http://www.w3.org/2004/02/skos/core#"),
    ("schema", "http://schema.org/"),
];

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// Serializes RDF triples to Turtle format.
pub struct TurtleSerializer {
    /// Custom prefix mappings (prefix -> namespace IRI).
    prefixes: Vec<(String, String)>,
}

impl TurtleSerializer {
    /// Creates a new serializer with automatic prefix detection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prefixes: Vec::new(),
        }
    }

    /// Adds a custom prefix mapping.
    pub fn with_prefix(mut self, prefix: &str, namespace: &str) -> Self {
        self.prefixes
            .push((prefix.to_string(), namespace.to_string()));
        self
    }

    /// Serializes triples to Turtle format, writing to the given writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write<W: Write>(&self, writer: &mut W, triples: &[Arc<Triple>]) -> io::Result<()> {
        if triples.is_empty() {
            return Ok(());
        }

        // Collect used namespaces and build active prefix map.
        let active_prefixes = self.collect_prefixes(triples);

        // Emit @prefix declarations.
        for (prefix, namespace) in &active_prefixes {
            writeln!(writer, "@prefix {prefix}: <{namespace}> .")?;
        }
        if !active_prefixes.is_empty() {
            writeln!(writer)?;
        }

        // Group triples by subject, then by predicate within each subject.
        let grouped = group_by_subject(triples);

        let mut first_subject = true;
        for (subject, pred_groups) in &grouped {
            if !first_subject {
                writeln!(writer)?;
            }
            first_subject = false;

            // Subject
            write!(writer, "{}", format_term(subject, &active_prefixes))?;

            let mut first_pred = true;
            for (predicate, objects) in pred_groups {
                if first_pred {
                    write!(writer, " ")?;
                } else {
                    write!(writer, " ;\n    ")?;
                }
                first_pred = false;

                // Predicate (use `a` shorthand for rdf:type)
                if predicate
                    .as_iri()
                    .is_some_and(|iri| iri.as_str() == RDF_TYPE)
                {
                    write!(writer, "a")?;
                } else {
                    write!(writer, "{}", format_term(predicate, &active_prefixes))?;
                }

                // Objects (comma-separated for multiple)
                for (i, object) in objects.iter().enumerate() {
                    if i == 0 {
                        write!(writer, " ")?;
                    } else {
                        write!(writer, ", ")?;
                    }
                    write!(writer, "{}", format_term(object, &active_prefixes))?;
                }
            }

            writeln!(writer, " .")?;
        }

        Ok(())
    }

    /// Serializes triples to a Turtle string.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if serialization fails.
    ///
    /// # Panics
    ///
    /// Panics if the Turtle output is not valid UTF-8 (should not occur with well-formed data).
    pub fn to_string(&self, triples: &[Arc<Triple>]) -> io::Result<String> {
        let mut buf = Vec::new();
        self.write(&mut buf, triples)?;
        // SAFETY: Turtle output is always valid UTF-8.
        Ok(String::from_utf8(buf).expect("Turtle output should be valid UTF-8"))
    }

    /// Collects prefix mappings used by the given triples.
    fn collect_prefixes(&self, triples: &[Arc<Triple>]) -> Vec<(String, String)> {
        let mut used: BTreeMap<String, String> = BTreeMap::new();

        // Add custom prefixes first.
        for (prefix, namespace) in &self.prefixes {
            used.insert(prefix.clone(), namespace.clone());
        }

        // Scan triples for well-known namespaces.
        for triple in triples {
            for term in [triple.subject(), triple.predicate(), triple.object()] {
                if let Some(iri) = term.as_iri() {
                    for &(prefix, namespace) in WELL_KNOWN_PREFIXES {
                        if iri.as_str().starts_with(namespace) && !used.contains_key(prefix) {
                            used.insert(prefix.to_string(), namespace.to_string());
                        }
                    }
                }
            }
        }

        used.into_iter().collect()
    }
}

impl Default for TurtleSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Groups triples by subject, preserving insertion order for subjects.
/// Within each subject, groups by predicate preserving order.
fn group_by_subject(triples: &[Arc<Triple>]) -> Vec<(&Term, Vec<(&Term, Vec<&Term>)>)> {
    // Use a Vec of (subject, preds) to preserve ordering.
    let mut subjects: Vec<(&Term, Vec<(&Term, Vec<&Term>)>)> = Vec::new();

    for triple in triples {
        let subject = triple.subject();
        let predicate = triple.predicate();
        let object = triple.object();

        // Find or create subject group.
        let pred_groups = match subjects.iter().position(|(s, _)| *s == subject) {
            Some(idx) => &mut subjects[idx].1,
            None => {
                subjects.push((subject, Vec::new()));
                &mut subjects.last_mut().expect("just pushed").1
            }
        };

        // Find or create predicate group within subject.
        match pred_groups.iter().position(|(p, _)| *p == predicate) {
            Some(idx) => pred_groups[idx].1.push(object),
            None => pred_groups.push((predicate, vec![object])),
        }
    }

    subjects
}

/// Formats a term as a Turtle string, using prefix notation where possible.
fn format_term(term: &Term, prefixes: &[(String, String)]) -> String {
    match term {
        Term::Iri(iri) => format_iri(iri, prefixes),
        Term::BlankNode(bn) => format!("_:{}", bn.id()),
        Term::Literal(lit) => format_literal(lit, prefixes),
    }
}

/// Formats an IRI, using prefixed form if a matching namespace is found.
fn format_iri(iri: &Iri, prefixes: &[(String, String)]) -> String {
    let iri_str = iri.as_str();
    for (prefix, namespace) in prefixes {
        if let Some(local) = iri_str.strip_prefix(namespace.as_str()) {
            // Validate that the local name contains only ON_LOCAL characters.
            if !local.is_empty() && is_valid_local_name(local) {
                return format!("{prefix}:{local}");
            }
        }
    }
    format!("<{iri_str}>")
}

/// Formats a literal in Turtle syntax.
fn format_literal(lit: &Literal, prefixes: &[(String, String)]) -> String {
    let escaped = escape_turtle_string(lit.value());

    if let Some(lang) = lit.language() {
        format!("\"{escaped}\"@{lang}")
    } else if lit.datatype() == Literal::XSD_STRING {
        format!("\"{escaped}\"")
    } else if lit.datatype() == Literal::XSD_INTEGER {
        // Numeric shorthand: bare integer if the lexical form is a valid integer.
        if lit.value().parse::<i64>().is_ok() {
            return lit.value().to_string();
        }
        format!(
            "\"{escaped}\"^^{}",
            format_iri(&Iri::new(lit.datatype()), prefixes)
        )
    } else if lit.datatype() == Literal::XSD_DOUBLE {
        // Numeric shorthand: bare double if the lexical form contains `.` or `e`/`E`.
        let val = lit.value();
        if val.parse::<f64>().is_ok()
            && (val.contains('.') || val.contains('e') || val.contains('E'))
        {
            return val.to_string();
        }
        format!(
            "\"{escaped}\"^^{}",
            format_iri(&Iri::new(lit.datatype()), prefixes)
        )
    } else if lit.datatype() == Literal::XSD_BOOLEAN {
        let val = lit.value();
        if val == "true" || val == "false" {
            return val.to_string();
        }
        format!(
            "\"{escaped}\"^^{}",
            format_iri(&Iri::new(lit.datatype()), prefixes)
        )
    } else {
        format!(
            "\"{escaped}\"^^{}",
            format_iri(&Iri::new(lit.datatype()), prefixes)
        )
    }
}

/// Escapes a string for Turtle literal output.
fn escape_turtle_string(s: &str) -> String {
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

/// Checks if a string is a valid Turtle ON_LOCAL (prefixed name local part).
fn is_valid_local_name(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    // First character: letter, digit, underscore, or colon.
    if !first.is_alphanumeric() && first != '_' && first != ':' {
        return false;
    }

    // Remaining characters: letters, digits, underscore, hyphen, period, colon.
    for ch in chars {
        if !ch.is_alphanumeric() && ch != '_' && ch != '-' && ch != '.' && ch != ':' {
            return false;
        }
    }

    // Must not end with a period.
    !s.ends_with('.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::triple::Triple;

    fn sample_triples() -> Vec<Arc<Triple>> {
        vec![
            Arc::new(Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                Term::iri("http://xmlns.com/foaf/0.1/Person"),
            )),
            Arc::new(Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            )),
            Arc::new(Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/age"),
                Term::typed_literal("30", Literal::XSD_INTEGER),
            )),
            Arc::new(Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://example.org/gus"),
            )),
            Arc::new(Triple::new(
                Term::iri("http://example.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            )),
        ]
    }

    #[test]
    fn test_serialize_basic() {
        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&sample_triples()).unwrap();

        // Should have prefix declarations.
        assert!(output.contains("@prefix foaf: <http://xmlns.com/foaf/0.1/> ."));
        assert!(output.contains("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."));

        // Should use `a` shorthand for rdf:type.
        assert!(output.contains(" a "));

        // Should use prefixed names.
        assert!(output.contains("foaf:name"));
        assert!(output.contains("foaf:Person"));
        assert!(output.contains("foaf:knows"));

        // Should have numeric shorthand for integer.
        assert!(output.contains(" 30"));
    }

    #[test]
    fn test_serialize_predicate_grouping() {
        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&sample_triples()).unwrap();

        // Alix should have multiple predicates separated by `;`.
        assert!(output.contains(';'));
    }

    #[test]
    fn test_serialize_object_list() {
        let triples = vec![
            Arc::new(Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://example.org/gus"),
            )),
            Arc::new(Triple::new(
                Term::iri("http://example.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://example.org/jules"),
            )),
        ];

        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&triples).unwrap();

        // Multiple objects for same subject+predicate should use `,`.
        assert!(output.contains(", "));
    }

    #[test]
    fn test_serialize_language_tagged() {
        let triples = vec![Arc::new(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::lang_literal("Alix", "en"),
        ))];

        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&triples).unwrap();
        assert!(output.contains("\"Alix\"@en"));
    }

    #[test]
    fn test_serialize_blank_node() {
        let triples = vec![Arc::new(Triple::new(
            Term::blank("b0"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ))];

        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&triples).unwrap();
        assert!(output.contains("_:b0"));
    }

    #[test]
    fn test_serialize_escape() {
        let triples = vec![Arc::new(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("line1\nline2\ttab \"quoted\""),
        ))];

        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&triples).unwrap();
        assert!(output.contains("\\n"));
        assert!(output.contains("\\t"));
        assert!(output.contains("\\\""));
    }

    #[test]
    fn test_serialize_empty() {
        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&[]).unwrap();
        assert!(output.is_empty(), "{output:?}");
    }

    #[test]
    fn test_serialize_custom_prefix() {
        let triples = vec![Arc::new(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://example.org/name"),
            Term::literal("Alix"),
        ))];

        let serializer = TurtleSerializer::new().with_prefix("ex", "http://example.org/");
        let output = serializer.to_string(&triples).unwrap();
        assert!(output.contains("@prefix ex: <http://example.org/> ."));
        assert!(output.contains("ex:alix"));
        assert!(output.contains("ex:name"));
    }

    #[test]
    fn test_serialize_escape_cr_and_backslash() {
        let triples = vec![
            Arc::new(Triple::new(
                Term::iri("http://example.org/s"),
                Term::iri("http://example.org/p1"),
                Term::literal("line1\rline2"),
            )),
            Arc::new(Triple::new(
                Term::iri("http://example.org/s"),
                Term::iri("http://example.org/p2"),
                Term::literal("back\\slash"),
            )),
        ];

        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&triples).unwrap();

        // Carriage return should be escaped as \r
        assert!(output.contains("\\r"));
        // Backslash should be escaped as \\
        assert!(output.contains("\\\\"));
    }

    #[test]
    fn test_serialize_double_shorthand() {
        let triples = vec![
            Arc::new(Triple::new(
                Term::iri("http://example.org/s"),
                Term::iri("http://example.org/negpi"),
                Term::typed_literal("-3.14", Literal::XSD_DOUBLE),
            )),
            Arc::new(Triple::new(
                Term::iri("http://example.org/s"),
                Term::iri("http://example.org/tiny"),
                Term::typed_literal("1e-5", Literal::XSD_DOUBLE),
            )),
        ];

        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&triples).unwrap();

        // Double shorthand: bare value without datatype annotation.
        assert!(output.contains("-3.14"));
        assert!(output.contains("1e-5"));
        assert!(
            !output.contains("^^"),
            "double shorthand should not have ^^ annotation"
        );
    }

    #[test]
    fn test_serialize_boolean_shorthand() {
        let triples = vec![Arc::new(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/active"),
            Term::typed_literal("true", Literal::XSD_BOOLEAN),
        ))];

        let serializer = TurtleSerializer::new();
        let output = serializer.to_string(&triples).unwrap();
        // Boolean shorthand: bare `true` without datatype annotation.
        assert!(output.contains(" true"));
        assert!(!output.contains("^^"));
    }
}
