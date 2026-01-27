//! Query language parsers.
//!
//! This module provides parsers for different query languages:
//!
//! - [`gql`] - GQL parser (ISO/IEC 39075:2024)
//! - [`cypher`] - Cypher parser (openCypher 9.0, feature-gated)
//! - [`sparql`] - SPARQL parser (W3C SPARQL 1.1, feature-gated)
//! - [`gremlin`] - Gremlin parser (Apache TinkerPop, feature-gated)
//! - [`graphql`] - GraphQL parser (spec-compliant, feature-gated)

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
