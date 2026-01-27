//! Statistics collection for cost-based query optimization.
//!
//! This module provides statistics about:
//! - Tables/Labels: cardinality, size
//! - Columns/Properties: distinct values, min/max, null fraction, histograms
//! - Relationships: edge type statistics, degree distributions
//! - RDF: triple patterns, predicates, join selectivity

mod collector;
mod histogram;
mod rdf;

pub use collector::{
    ColumnStatistics, EdgeTypeStatistics, LabelStatistics, PropertyKey, Statistics,
    TableStatistics,
};
pub use histogram::{Histogram, HistogramBucket};
pub use rdf::{
    IndexStatistics, PredicateStatistics, RdfStatistics, RdfStatisticsCollector, TriplePosition,
};
