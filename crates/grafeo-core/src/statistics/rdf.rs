//! RDF-specific statistics for SPARQL query optimization.
//!
//! SPARQL queries are built from triple patterns like `?person :knows ?friend`.
//! To pick the best join order, the optimizer needs to estimate how many results
//! each pattern will produce. This module tracks the distribution of subjects,
//! predicates, and objects to make those estimates.

use super::histogram::Histogram;
use grafeo_common::types::Value;
use std::collections::HashMap;

/// Everything the SPARQL optimizer knows about your RDF data.
///
/// Use [`estimate_triple_pattern_cardinality()`](Self::estimate_triple_pattern_cardinality)
/// to predict how many triples match a pattern like `?s :knows ?o`.
#[derive(Debug, Clone, Default)]
pub struct RdfStatistics {
    /// Total number of triples.
    pub total_triples: u64,
    /// Number of unique subjects.
    pub subject_count: u64,
    /// Number of unique predicates.
    pub predicate_count: u64,
    /// Number of unique objects.
    pub object_count: u64,

    /// Per-predicate statistics.
    pub predicates: HashMap<String, PredicateStatistics>,

    /// Subject frequency histogram (for join selectivity).
    pub subject_histogram: Option<Histogram>,
    /// Object frequency histogram.
    pub object_histogram: Option<Histogram>,

    /// Index access pattern statistics (for cost model).
    pub index_stats: IndexStatistics,
}

impl RdfStatistics {
    /// Creates new empty RDF statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates statistics from an RDF store.
    pub fn update_from_counts(
        &mut self,
        total_triples: u64,
        subject_count: u64,
        predicate_count: u64,
        object_count: u64,
    ) {
        self.total_triples = total_triples;
        self.subject_count = subject_count;
        self.predicate_count = predicate_count;
        self.object_count = object_count;
    }

    /// Adds or updates predicate statistics.
    pub fn update_predicate(&mut self, predicate: &str, stats: PredicateStatistics) {
        self.predicates.insert(predicate.to_string(), stats);
    }

    /// Gets predicate statistics.
    pub fn get_predicate(&self, predicate: &str) -> Option<&PredicateStatistics> {
        self.predicates.get(predicate)
    }

    /// Estimates cardinality for a triple pattern.
    ///
    /// # Arguments
    /// * `subject_bound` - Whether the subject is a constant
    /// * `predicate_bound` - Whether the predicate is a constant (and its value if so)
    /// * `object_bound` - Whether the object is a constant
    pub fn estimate_triple_pattern_cardinality(
        &self,
        subject_bound: bool,
        predicate_bound: Option<&str>,
        object_bound: bool,
    ) -> f64 {
        if self.total_triples == 0 {
            return 0.0;
        }

        let base = self.total_triples as f64;

        match (subject_bound, predicate_bound, object_bound) {
            // Fully bound pattern - either 0 or 1
            (true, Some(_), true) => 1.0,

            // Subject and predicate bound
            (true, Some(pred), false) => {
                if let Some(stats) = self.predicates.get(pred) {
                    // Average objects per subject for this predicate
                    stats.avg_objects_per_subject()
                } else {
                    // Default: assume 10 objects per subject
                    10.0
                }
            }

            // Subject and object bound (any predicate)
            (true, None, true) => {
                // Relatively rare - use predicate count as estimate
                self.predicate_count as f64
            }

            // Only subject bound
            (true, None, false) => {
                // Average triples per subject
                base / self.subject_count.max(1) as f64
            }

            // Only predicate bound
            (false, Some(pred), false) => {
                if let Some(stats) = self.predicates.get(pred) {
                    stats.triple_count as f64
                } else {
                    base / self.predicate_count.max(1) as f64
                }
            }

            // Predicate and object bound
            (false, Some(pred), true) => {
                if let Some(stats) = self.predicates.get(pred) {
                    // Average subjects per object for this predicate
                    stats.avg_subjects_per_object()
                } else {
                    10.0
                }
            }

            // Only object bound
            (false, None, true) => {
                // Average triples per object
                base / self.object_count.max(1) as f64
            }

            // No bindings - full scan
            (false, None, false) => base,
        }
    }

    /// Estimates join selectivity between two patterns sharing a variable.
    pub fn estimate_join_selectivity(
        &self,
        var_position1: TriplePosition,
        var_position2: TriplePosition,
    ) -> f64 {
        let domain_size = match (var_position1, var_position2) {
            (TriplePosition::Subject, TriplePosition::Subject) => self.subject_count,
            (TriplePosition::Subject, TriplePosition::Object)
            | (TriplePosition::Object, TriplePosition::Subject) => {
                // Subject-object join - use larger domain
                self.subject_count.max(self.object_count)
            }
            (TriplePosition::Object, TriplePosition::Object) => self.object_count,
            _ => {
                // Joins involving predicates are rare and highly selective
                self.predicate_count
            }
        };

        if domain_size == 0 {
            return 1.0;
        }

        // Join selectivity ≈ 1 / domain_size
        1.0 / domain_size as f64
    }

    /// Estimates the cardinality after a FILTER operation.
    pub fn estimate_filter_selectivity(&self, predicate_iri: Option<&str>) -> f64 {
        // Default filter selectivity
        if let Some(pred) = predicate_iri {
            if let Some(stats) = self.predicates.get(pred) {
                // Use predicate's object statistics for filter estimation
                if let Some(ref _hist) = stats.object_histogram {
                    // Assume filters reduce to ~33% of values
                    return 0.33;
                }
            }
        }
        0.33 // Default filter selectivity
    }
}

/// Which position in a triple pattern - subject, predicate, or object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriplePosition {
    /// Subject position.
    Subject,
    /// Predicate position.
    Predicate,
    /// Object position.
    Object,
}

/// Statistics for a single predicate (like `:knows` or `:name`).
#[derive(Debug, Clone)]
pub struct PredicateStatistics {
    /// Number of triples with this predicate.
    pub triple_count: u64,
    /// Number of unique subjects using this predicate.
    pub distinct_subjects: u64,
    /// Number of unique objects for this predicate.
    pub distinct_objects: u64,
    /// Whether this predicate is functional (1 object per subject).
    pub is_functional: bool,
    /// Whether this predicate is inverse functional (1 subject per object).
    pub is_inverse_functional: bool,
    /// Object type statistics (for typed literals).
    pub object_type_distribution: HashMap<String, u64>,
    /// Histogram of object values (for selective filters).
    pub object_histogram: Option<Histogram>,
}

impl PredicateStatistics {
    /// Creates new predicate statistics.
    pub fn new(triple_count: u64, distinct_subjects: u64, distinct_objects: u64) -> Self {
        let is_functional = triple_count > 0 && triple_count == distinct_subjects;
        let is_inverse_functional = triple_count > 0 && triple_count == distinct_objects;

        Self {
            triple_count,
            distinct_subjects,
            distinct_objects,
            is_functional,
            is_inverse_functional,
            object_type_distribution: HashMap::new(),
            object_histogram: None,
        }
    }

    /// Marks the predicate as functional.
    pub fn with_functional(mut self, functional: bool) -> Self {
        self.is_functional = functional;
        self
    }

    /// Adds object histogram.
    pub fn with_object_histogram(mut self, histogram: Histogram) -> Self {
        self.object_histogram = Some(histogram);
        self
    }

    /// Adds object type distribution.
    pub fn with_object_types(mut self, types: HashMap<String, u64>) -> Self {
        self.object_type_distribution = types;
        self
    }

    /// Average number of objects per subject.
    pub fn avg_objects_per_subject(&self) -> f64 {
        if self.distinct_subjects == 0 {
            return 0.0;
        }
        self.triple_count as f64 / self.distinct_subjects as f64
    }

    /// Average number of subjects per object.
    pub fn avg_subjects_per_object(&self) -> f64 {
        if self.distinct_objects == 0 {
            return 0.0;
        }
        self.triple_count as f64 / self.distinct_objects as f64
    }

    /// Selectivity of a value equality filter on objects.
    pub fn object_equality_selectivity(&self, _value: &Value) -> f64 {
        if let Some(ref hist) = self.object_histogram {
            // Use histogram for better estimate
            return hist.estimate_equality_selectivity(_value);
        }

        // Fall back to uniform distribution
        if self.distinct_objects == 0 {
            return 1.0;
        }
        1.0 / self.distinct_objects as f64
    }
}

/// Cost estimates for different index access patterns.
///
/// RDF stores typically have multiple indexes (SPO, POS, OSP). This tracks
/// how expensive each one is to use, so the optimizer can pick the cheapest.
#[derive(Debug, Clone, Default)]
pub struct IndexStatistics {
    /// Average cost of SPO index lookup (subject first).
    pub spo_lookup_cost: f64,
    /// Average cost of POS index lookup (predicate first).
    pub pos_lookup_cost: f64,
    /// Average cost of OSP index lookup (object first).
    pub osp_lookup_cost: f64,
    /// Whether OSP index is available.
    pub has_osp_index: bool,
}

impl IndexStatistics {
    /// Creates default index statistics.
    pub fn new() -> Self {
        Self {
            spo_lookup_cost: 1.0,
            pos_lookup_cost: 1.5,
            osp_lookup_cost: 2.0,
            has_osp_index: true,
        }
    }

    /// Estimates the cost of executing a triple pattern.
    pub fn estimate_pattern_cost(
        &self,
        subject_bound: bool,
        predicate_bound: bool,
        object_bound: bool,
    ) -> f64 {
        match (subject_bound, predicate_bound, object_bound) {
            // Use SPO index
            (true, _, _) => self.spo_lookup_cost,
            // Use POS index
            (false, true, _) => self.pos_lookup_cost,
            // Use OSP index if available
            (false, false, true) if self.has_osp_index => self.osp_lookup_cost,
            // Full scan
            _ => 10.0, // High cost for full scan
        }
    }
}

/// Streams triples through to build RDF statistics automatically.
///
/// Call [`record_triple()`](Self::record_triple) for each triple, then
/// [`build()`](Self::build) to get the final [`RdfStatistics`].
#[derive(Default)]
pub struct RdfStatisticsCollector {
    /// Total triple count.
    triple_count: u64,
    /// Subject occurrences.
    subjects: HashMap<String, u64>,
    /// Predicate occurrences.
    predicates: HashMap<String, PredicateCollector>,
    /// Object occurrences.
    objects: HashMap<String, u64>,
}

/// Collector for per-predicate statistics.
#[derive(Default)]
struct PredicateCollector {
    count: u64,
    subjects: HashMap<String, u64>,
    objects: HashMap<String, u64>,
}

impl RdfStatisticsCollector {
    /// Creates a new statistics collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a triple.
    pub fn record_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        self.triple_count += 1;

        *self.subjects.entry(subject.to_string()).or_insert(0) += 1;
        *self.objects.entry(object.to_string()).or_insert(0) += 1;

        let pred_stats = self.predicates.entry(predicate.to_string()).or_default();
        pred_stats.count += 1;
        *pred_stats.subjects.entry(subject.to_string()).or_insert(0) += 1;
        *pred_stats.objects.entry(object.to_string()).or_insert(0) += 1;
    }

    /// Builds the final statistics.
    pub fn build(self) -> RdfStatistics {
        let mut stats = RdfStatistics::new();

        stats.total_triples = self.triple_count;
        stats.subject_count = self.subjects.len() as u64;
        stats.predicate_count = self.predicates.len() as u64;
        stats.object_count = self.objects.len() as u64;

        for (pred, collector) in self.predicates {
            let pred_stats = PredicateStatistics::new(
                collector.count,
                collector.subjects.len() as u64,
                collector.objects.len() as u64,
            );
            stats.predicates.insert(pred, pred_stats);
        }

        stats.index_stats = IndexStatistics::new();

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdf_statistics_basic() {
        let mut stats = RdfStatistics::new();
        stats.update_from_counts(1000, 100, 50, 200);

        // Test triple pattern cardinality estimation
        let card = stats.estimate_triple_pattern_cardinality(true, None, false);
        assert!(card > 0.0);

        // Fully unbound should return total
        let full_card = stats.estimate_triple_pattern_cardinality(false, None, false);
        assert_eq!(full_card, 1000.0);
    }

    #[test]
    fn test_predicate_statistics() {
        let pred_stats = PredicateStatistics::new(100, 50, 80);

        assert_eq!(pred_stats.avg_objects_per_subject(), 2.0);
        assert!(!pred_stats.is_functional);
    }

    #[test]
    fn test_functional_predicate() {
        let pred_stats = PredicateStatistics::new(100, 100, 100);

        assert!(pred_stats.is_functional);
        assert!(pred_stats.is_inverse_functional);
        assert_eq!(pred_stats.avg_objects_per_subject(), 1.0);
    }

    #[test]
    fn test_join_selectivity() {
        let mut stats = RdfStatistics::new();
        stats.update_from_counts(1000, 100, 50, 200);

        let sel = stats.estimate_join_selectivity(TriplePosition::Subject, TriplePosition::Subject);
        assert_eq!(sel, 0.01); // 1/100

        let sel = stats.estimate_join_selectivity(TriplePosition::Subject, TriplePosition::Object);
        assert_eq!(sel, 1.0 / 200.0); // 1/max(100, 200)
    }

    #[test]
    fn test_statistics_collector() {
        let mut collector = RdfStatisticsCollector::new();

        collector.record_triple("alice", "knows", "bob");
        collector.record_triple("alice", "name", "Alice");
        collector.record_triple("bob", "name", "Bob");
        collector.record_triple("bob", "knows", "charlie");

        let stats = collector.build();

        assert_eq!(stats.total_triples, 4);
        assert_eq!(stats.subject_count, 2); // alice, bob
        assert_eq!(stats.predicate_count, 2); // knows, name
        assert_eq!(stats.object_count, 4); // bob, Alice, Bob, charlie
    }

    #[test]
    fn test_pattern_cost_estimation() {
        let index_stats = IndexStatistics::new();

        // Subject bound - cheapest
        let cost = index_stats.estimate_pattern_cost(true, false, false);
        assert_eq!(cost, 1.0);

        // Predicate bound
        let cost = index_stats.estimate_pattern_cost(false, true, false);
        assert_eq!(cost, 1.5);

        // Full scan - most expensive
        let cost = index_stats.estimate_pattern_cost(false, false, false);
        assert_eq!(cost, 10.0);
    }
}
