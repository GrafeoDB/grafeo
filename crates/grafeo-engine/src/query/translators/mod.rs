//! Query language translators.
//!
//! Each submodule translates a parsed AST from a specific query language
//! (GQL, Cypher, SPARQL, etc.) into the shared [`LogicalPlan`](crate::query::plan::LogicalPlan) IR.

#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "sparql",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
pub(crate) mod common;

#[cfg(feature = "gql")]
pub mod gql;

#[cfg(feature = "cypher")]
pub mod cypher;

#[cfg(feature = "sparql")]
pub mod sparql;

#[cfg(feature = "gremlin")]
pub mod gremlin;

#[cfg(feature = "graphql")]
pub mod graphql;

#[cfg(feature = "sql-pgq")]
pub mod sql_pgq;

#[cfg(all(feature = "graphql", feature = "triple-store"))]
pub mod graphql_rdf;
