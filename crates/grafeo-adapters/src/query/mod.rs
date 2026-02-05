//! Query language parsers - speak your preferred graph language.
//!
//! Grafeo speaks GQL natively (the new ISO standard), but we also understand
//! Cypher, SPARQL, Gremlin, and GraphQL. Enable what you need via features.
//!
//! | Language | Standard | Feature flag | Notes |
//! | -------- | -------- | ------------ | ----- |
//! | GQL | ISO/IEC 39075:2024 | `gql` (default) | The ISO standard, our native tongue |
//! | Cypher | openCypher 9.0 | `cypher` | Neo4j's query language |
//! | SPARQL | W3C SPARQL 1.1 | `sparql` | For RDF triple stores |
//! | Gremlin | Apache TinkerPop | `gremlin` | Graph traversal DSL |
//! | GraphQL | June 2018 spec | `graphql` | API query language |

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
