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
pub mod limits;
pub mod optimizer;
pub mod plan;
pub mod planner;
pub mod processor;
pub mod profile;
pub mod translators;

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

#[cfg(feature = "triple-store")]
pub use planner::rdf::RdfPlanner;

// Translator exports
#[cfg(feature = "gql")]
pub use translators::gql::translate as translate_gql;

#[cfg(feature = "cypher")]
pub use translators::cypher::translate as translate_cypher;

#[cfg(feature = "sparql")]
pub use translators::sparql::translate as translate_sparql;

#[cfg(feature = "gremlin")]
pub use translators::gremlin::translate as translate_gremlin;

#[cfg(feature = "graphql")]
pub use translators::graphql::translate as translate_graphql;

#[cfg(feature = "sql-pgq")]
pub use translators::sql_pgq::translate as translate_sql_pgq;

#[cfg(all(feature = "graphql", feature = "triple-store"))]
pub use translators::graphql_rdf::translate as translate_graphql_rdf;
