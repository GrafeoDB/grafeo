//! N-Quads serializer.
//!
//! Extends N-Triples with an optional fourth element for the graph name.
//! See [W3C N-Quads](https://www.w3.org/TR/n-quads/).
//!
//! Format: `<subject> <predicate> <object> <graphName> .`
//! When the graph name is omitted, the triple belongs to the default graph.

use crate::graph::rdf::store::RdfStore;
use crate::graph::rdf::triple::Triple;
use std::io::{self, Write};
use std::sync::Arc;

/// Serializes an RDF store (default graph + named graphs) to N-Quads format.
///
/// # Errors
///
/// Returns an I/O error if writing fails.
pub fn write_nquads<W: Write>(writer: &mut W, store: &RdfStore) -> io::Result<()> {
    // Default graph triples (no graph name).
    for triple in store.triples() {
        write_nquad_line(writer, &triple, None)?;
    }

    // Named graph triples.
    for name in store.graph_names() {
        if let Some(graph) = store.graph(&name) {
            for triple in graph.triples() {
                write_nquad_line(writer, &triple, Some(&name))?;
            }
        }
    }

    Ok(())
}

/// Serializes an RDF store to an N-Quads string.
///
/// # Errors
///
/// Returns an I/O error if serialization fails.
///
/// # Panics
///
/// Panics if the N-Quads output is not valid UTF-8 (should not occur with well-formed data).
pub fn to_nquads_string(store: &RdfStore) -> io::Result<String> {
    let mut buf = Vec::new();
    write_nquads(&mut buf, store)?;
    Ok(String::from_utf8(buf).expect("N-Quads output should be valid UTF-8"))
}

/// Writes a single N-Quads line.
fn write_nquad_line<W: Write>(
    writer: &mut W,
    triple: &Arc<Triple>,
    graph_name: Option<&str>,
) -> io::Result<()> {
    // N-Triples representation of s p o.
    write!(
        writer,
        "{} {} {}",
        triple.subject(),
        triple.predicate(),
        triple.object()
    )?;

    // Optional graph name.
    if let Some(name) = graph_name {
        write!(writer, " <{name}>")?;
    }

    writeln!(writer, " .")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::store::RdfStore;
    use crate::graph::rdf::term::Term;
    use crate::graph::rdf::triple::Triple;

    #[test]
    fn test_nquads_default_graph_only() {
        let store = RdfStore::new();
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let output = to_nquads_string(&store).unwrap();
        assert!(output.contains("<http://example.org/alix>"));
        assert!(output.contains("<http://xmlns.com/foaf/0.1/name>"));
        assert!(output.contains("\"Alix\""));
        assert!(output.ends_with(" .\n"));
        // Default graph: no fourth element.
        assert_eq!(output.matches('<').count(), 2); // only subject and predicate IRIs
    }

    #[test]
    fn test_nquads_with_named_graph() {
        let store = RdfStore::new();
        store.insert(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("default"),
        ));

        let graph = store.graph_or_create("http://example.org/g1");
        graph.insert(Triple::new(
            Term::iri("http://example.org/s"),
            Term::iri("http://example.org/p"),
            Term::literal("named"),
        ));

        let output = to_nquads_string(&store).unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);

        // One line without graph name, one with.
        let has_graph = lines.iter().any(|l| l.contains("<http://example.org/g1>"));
        let has_default = lines
            .iter()
            .any(|l| l.contains("\"default\"") && !l.contains("<http://example.org/g1>"));
        assert!(has_graph, "should have named graph quad");
        assert!(has_default, "should have default graph triple");
    }

    #[test]
    fn test_nquads_empty() {
        let store = RdfStore::new();
        let output = to_nquads_string(&store).unwrap();
        assert!(output.is_empty(), "{output:?}");
    }
}
