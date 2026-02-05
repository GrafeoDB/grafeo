//! Statistics for cost-based query optimization.
//!
//! The query optimizer uses these statistics to pick the best execution plan.
//! Without stats, it has to guess - with stats, it knows that filtering by
//! "status = 'active'" returns 90% of rows while "label = 'Admin'" returns 0.1%.
//!
//! | Statistic | What it tells the optimizer |
//! | --------- | --------------------------- |
//! | Label cardinality | How many nodes have this label |
//! | Property histograms | Distribution of values for range predicates |
//! | Degree stats | How many edges per node (affects traversal cost) |
//! | Distinct counts | Selectivity of equality predicates |

mod collector;
mod histogram;
mod rdf;

pub use collector::{
    ColumnStatistics, EdgeTypeStatistics, LabelStatistics, PropertyKey, Statistics, TableStatistics,
};
pub use histogram::{Histogram, HistogramBucket};
pub use rdf::{
    IndexStatistics, PredicateStatistics, RdfStatistics, RdfStatisticsCollector, TriplePosition,
};
