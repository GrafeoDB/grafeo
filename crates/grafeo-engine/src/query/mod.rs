//! The complete query processing pipeline.
//!
//! Your query goes through several stages before results come back:
//!
//! 1. **Translator** - Parses GQL/Cypher/SPARQL into a logical plan
//! 2. **Binder** - Validates that variables and properties exist
//! 3. **Optimizer** - Pushes filters down, reorders joins for speed
//! 4. **Planner** - Converts the logical plan to physical operators
//! 5. **Executor** - Actually runs the operators and streams results
//!
//! Most users don't interact with these directly - just call
//! [`Session::execute()`](crate::Session::execute). But if you're building
//! custom query processing, [`QueryProcessor`] is the unified interface.

pub mod binder;
pub mod cache;
pub mod executor;
pub mod optimizer;
pub mod plan;
pub mod planner;
pub mod processor;

#[cfg(feature = "rdf")]
pub mod planner_rdf;

#[cfg(feature = "gql")]
pub mod gql_translator;

#[cfg(feature = "cypher")]
pub mod cypher_translator;

#[cfg(feature = "sparql")]
pub mod sparql_translator;

#[cfg(feature = "gremlin")]
pub mod gremlin_translator;

#[cfg(feature = "graphql")]
pub mod graphql_translator;

#[cfg(all(feature = "graphql", feature = "rdf"))]
pub mod graphql_rdf_translator;

// Core exports
pub use cache::{CacheKey, CacheStats, CachingQueryProcessor, QueryCache};
pub use executor::Executor;
pub use optimizer::{CardinalityEstimator, Optimizer};
pub use plan::{LogicalExpression, LogicalOperator, LogicalPlan};
pub use planner::{
    PhysicalPlan, Planner, convert_aggregate_function, convert_binary_op,
    convert_filter_expression, convert_unary_op,
};
pub use processor::{QueryLanguage, QueryParams, QueryProcessor};

#[cfg(feature = "rdf")]
pub use planner_rdf::RdfPlanner;

// Translator exports
#[cfg(feature = "gql")]
pub use gql_translator::translate as translate_gql;

#[cfg(feature = "cypher")]
pub use cypher_translator::translate as translate_cypher;

#[cfg(feature = "sparql")]
pub use sparql_translator::translate as translate_sparql;

#[cfg(feature = "gremlin")]
pub use gremlin_translator::translate as translate_gremlin;

#[cfg(feature = "graphql")]
pub use graphql_translator::translate as translate_graphql;

#[cfg(all(feature = "graphql", feature = "rdf"))]
pub use graphql_rdf_translator::translate as translate_graphql_rdf;
