//! RDF (Resource Description Framework) Graph Model.
//!
//! This module implements an RDF triple store following the W3C RDF 1.1 specification.
//!
//! RDF represents data as triples (Subject, Predicate, Object) forming a directed graph
//! where edges are predicates connecting subject and object nodes.
//!
//! # Key Concepts
//!
//! - **IRI**: Internationalized Resource Identifier, uniquely identifies resources
//! - **Blank Node**: Anonymous node within a graph
//! - **Literal**: Data value (string, number, date, etc.) with optional datatype/language
//! - **Triple**: The fundamental unit (subject, predicate, object)
//! - **Named Graph**: A set of triples identified by an IRI
//!
//! # Example
//!
//! ```ignore
//! use graphos_core::graph::rdf::{RdfStore, Term, Triple};
//!
//! let store = RdfStore::new();
//!
//! // Add a triple: <http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice"
//! store.insert(Triple::new(
//!     Term::iri("http://example.org/alice"),
//!     Term::iri("http://xmlns.com/foaf/0.1/name"),
//!     Term::literal("Alice"),
//! ));
//!
//! // Query triples with a specific subject
//! for triple in store.triples_with_subject(Term::iri("http://example.org/alice")) {
//!     println!("{:?}", triple);
//! }
//! ```

mod term;
mod triple;
mod store;

pub use term::{BlankNode, Iri, Literal, Term};
pub use triple::Triple;
pub use store::{RdfStore, RdfStoreConfig};
