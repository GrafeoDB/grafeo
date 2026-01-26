//! Query language parsers.
//!
//! This module provides parsers for different query languages:
//!
//! - [`gql`] - GQL parser (ISO/IEC 39075:2024)
//! - [`cypher`] - Cypher parser (openCypher 9.0, feature-gated)

#[cfg(feature = "gql")]
pub mod gql;

#[cfg(feature = "cypher")]
pub mod cypher;
