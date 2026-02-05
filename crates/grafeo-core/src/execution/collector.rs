//! Generic collector trait for parallel aggregation.
//!
//! Collectors provide a clean separation between what data to aggregate
//! and how to execute the aggregation in parallel. The pattern is inspired
//! by Tantivy's collector architecture.
//!
//! # Pattern
//!
//! 1. Create partition-local collectors (one per worker thread)
//! 2. Each collector processes its partition independently (no shared state)
//! 3. Merge all partition results into a final result
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::execution::collector::{Collector, PartitionCollector, CountCollector};
//!
//! let collector = CountCollector;
//!
//! // In parallel execution:
//! let mut partition_collectors: Vec<_> = (0..4)
//!     .map(|id| collector.for_partition(id))
//!     .collect();
//!
//! // Each partition processes its chunks
//! for (partition, chunks) in partitions.into_iter().enumerate() {
//!     for chunk in chunks {
//!         partition_collectors[partition].collect(&chunk)?;
//!     }
//! }
//!
//! // Merge results
//! let fruits: Vec<_> = partition_collectors.into_iter()
//!     .map(|c| c.harvest())
//!     .collect();
//! let total = collector.merge(fruits);
//! ```

use super::chunk::DataChunk;
use super::operators::OperatorError;

/// A collector that aggregates results from parallel execution.
///
/// Pattern: Create partition-local collectors, process independently,
/// then merge results. No shared mutable state during collection.
pub trait Collector: Sync {
    /// Final result type after merging all partitions.
    type Fruit: Send;

    /// Partition-local collector type.
    type PartitionCollector: PartitionCollector<Fruit = Self::Fruit>;

    /// Creates a collector for a single partition (called per-thread).
    fn for_partition(&self, partition_id: usize) -> Self::PartitionCollector;

    /// Merges results from all partitions (called once at the end).
    fn merge(&self, fruits: Vec<Self::Fruit>) -> Self::Fruit;
}

/// Per-partition collector - processes chunks locally.
///
/// Each partition collector is created by [`Collector::for_partition`]
/// and processes data independently. This enables lock-free parallel
/// execution.
pub trait PartitionCollector: Send {
    /// Result type produced by this partition.
    type Fruit: Send;

    /// Processes a batch of data.
    ///
    /// Called repeatedly with chunks from this partition.
    fn collect(&mut self, chunk: &DataChunk) -> Result<(), OperatorError>;

    /// Finalizes and returns the result for this partition.
    ///
    /// Called once after all chunks have been processed.
    fn harvest(self) -> Self::Fruit;
}

// ============================================================================
// Built-in Collectors
// ============================================================================

/// Counts rows across all partitions.
///
/// # Example
///
/// ```ignore
/// use grafeo_core::execution::collector::{Collector, CountCollector};
///
/// let collector = CountCollector;
/// let mut pc = collector.for_partition(0);
///
/// pc.collect(&chunk1)?;
/// pc.collect(&chunk2)?;
///
/// let count = pc.harvest();
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CountCollector;

impl Collector for CountCollector {
    type Fruit = u64;
    type PartitionCollector = CountPartitionCollector;

    fn for_partition(&self, _partition_id: usize) -> Self::PartitionCollector {
        CountPartitionCollector { count: 0 }
    }

    fn merge(&self, fruits: Vec<u64>) -> u64 {
        fruits.into_iter().sum()
    }
}

/// Partition-local counter.
pub struct CountPartitionCollector {
    count: u64,
}

impl PartitionCollector for CountPartitionCollector {
    type Fruit = u64;

    fn collect(&mut self, chunk: &DataChunk) -> Result<(), OperatorError> {
        self.count += chunk.len() as u64;
        Ok(())
    }

    fn harvest(self) -> u64 {
        self.count
    }
}

/// Collects all chunks (materializes the entire result).
///
/// Use this when you need all the data, not just an aggregate.
/// Be careful with large datasets - this can consume significant memory.
#[derive(Debug, Clone, Default)]
pub struct MaterializeCollector;

impl Collector for MaterializeCollector {
    type Fruit = Vec<DataChunk>;
    type PartitionCollector = MaterializePartitionCollector;

    fn for_partition(&self, _partition_id: usize) -> Self::PartitionCollector {
        MaterializePartitionCollector { chunks: Vec::new() }
    }

    fn merge(&self, mut fruits: Vec<Vec<DataChunk>>) -> Vec<DataChunk> {
        let total_chunks: usize = fruits.iter().map(|f| f.len()).sum();
        let mut result = Vec::with_capacity(total_chunks);
        for fruit in &mut fruits {
            result.append(fruit);
        }
        result
    }
}

/// Partition-local materializer.
pub struct MaterializePartitionCollector {
    chunks: Vec<DataChunk>,
}

impl PartitionCollector for MaterializePartitionCollector {
    type Fruit = Vec<DataChunk>;

    fn collect(&mut self, chunk: &DataChunk) -> Result<(), OperatorError> {
        self.chunks.push(chunk.clone());
        Ok(())
    }

    fn harvest(self) -> Vec<DataChunk> {
        self.chunks
    }
}

/// Collects first N rows across all partitions.
///
/// Stops collecting once the limit is reached (per partition).
/// Final merge ensures exactly `limit` rows are returned.
#[derive(Debug, Clone)]
pub struct LimitCollector {
    limit: usize,
}

impl LimitCollector {
    /// Creates a collector that limits output to `limit` rows.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl Collector for LimitCollector {
    type Fruit = (Vec<DataChunk>, usize);
    type PartitionCollector = LimitPartitionCollector;

    fn for_partition(&self, _partition_id: usize) -> Self::PartitionCollector {
        LimitPartitionCollector {
            chunks: Vec::new(),
            limit: self.limit,
            collected: 0,
        }
    }

    fn merge(&self, fruits: Vec<(Vec<DataChunk>, usize)>) -> (Vec<DataChunk>, usize) {
        let mut result = Vec::new();
        let mut total = 0;

        for (chunks, _) in fruits {
            for chunk in chunks {
                if total >= self.limit {
                    break;
                }
                let take = (self.limit - total).min(chunk.len());
                if take < chunk.len() {
                    result.push(chunk.slice(0, take));
                } else {
                    result.push(chunk);
                }
                total += take;
            }
            if total >= self.limit {
                break;
            }
        }

        (result, total)
    }
}

/// Partition-local limiter.
pub struct LimitPartitionCollector {
    chunks: Vec<DataChunk>,
    limit: usize,
    collected: usize,
}

impl PartitionCollector for LimitPartitionCollector {
    type Fruit = (Vec<DataChunk>, usize);

    fn collect(&mut self, chunk: &DataChunk) -> Result<(), OperatorError> {
        if self.collected >= self.limit {
            return Ok(());
        }

        let take = (self.limit - self.collected).min(chunk.len());
        if take < chunk.len() {
            self.chunks.push(chunk.slice(0, take));
        } else {
            self.chunks.push(chunk.clone());
        }
        self.collected += take;

        Ok(())
    }

    fn harvest(self) -> (Vec<DataChunk>, usize) {
        (self.chunks, self.collected)
    }
}

/// Collects statistics (count, sum, min, max) for a column.
#[derive(Debug, Clone)]
pub struct StatsCollector {
    column_idx: usize,
}

impl StatsCollector {
    /// Creates a collector that computes statistics for the given column.
    #[must_use]
    pub fn new(column_idx: usize) -> Self {
        Self { column_idx }
    }
}

/// Statistics result from [`StatsCollector`].
#[derive(Debug, Clone, Default)]
pub struct CollectorStats {
    /// Number of non-null values.
    pub count: u64,
    /// Sum of values (if numeric).
    pub sum: f64,
    /// Minimum value (if ordered).
    pub min: Option<f64>,
    /// Maximum value (if ordered).
    pub max: Option<f64>,
}

impl CollectorStats {
    /// Merges another stats into this one.
    pub fn merge(&mut self, other: CollectorStats) {
        self.count += other.count;
        self.sum += other.sum;
        self.min = match (self.min, other.min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(v), None) | (None, Some(v)) => Some(v),
            (None, None) => None,
        };
        self.max = match (self.max, other.max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(v), None) | (None, Some(v)) => Some(v),
            (None, None) => None,
        };
    }

    /// Computes the average (mean) value.
    #[must_use]
    pub fn avg(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.sum / self.count as f64)
        } else {
            None
        }
    }
}

impl Collector for StatsCollector {
    type Fruit = CollectorStats;
    type PartitionCollector = StatsPartitionCollector;

    fn for_partition(&self, _partition_id: usize) -> Self::PartitionCollector {
        StatsPartitionCollector {
            column_idx: self.column_idx,
            stats: CollectorStats::default(),
        }
    }

    fn merge(&self, fruits: Vec<CollectorStats>) -> CollectorStats {
        let mut result = CollectorStats::default();
        for fruit in fruits {
            result.merge(fruit);
        }
        result
    }
}

/// Partition-local stats collector.
pub struct StatsPartitionCollector {
    column_idx: usize,
    stats: CollectorStats,
}

impl PartitionCollector for StatsPartitionCollector {
    type Fruit = CollectorStats;

    fn collect(&mut self, chunk: &DataChunk) -> Result<(), OperatorError> {
        let column = chunk.column(self.column_idx).ok_or_else(|| {
            OperatorError::ColumnNotFound(format!(
                "column index {} out of bounds (width={})",
                self.column_idx,
                chunk.column_count()
            ))
        })?;

        for i in 0..chunk.len() {
            // Try typed access first (for specialized vectors), then fall back to generic
            let val = if let Some(f) = column.get_float64(i) {
                Some(f)
            } else if let Some(i) = column.get_int64(i) {
                Some(i as f64)
            } else if let Some(value) = column.get_value(i) {
                // Handle Generic vectors - extract numeric value
                match value {
                    grafeo_common::types::Value::Int64(i) => Some(i as f64),
                    grafeo_common::types::Value::Float64(f) => Some(f),
                    _ => None,
                }
            } else {
                None
            };

            if let Some(v) = val {
                self.stats.count += 1;
                self.stats.sum += v;
                self.stats.min = Some(match self.stats.min {
                    Some(m) => m.min(v),
                    None => v,
                });
                self.stats.max = Some(match self.stats.max {
                    Some(m) => m.max(v),
                    None => v,
                });
            }
        }

        Ok(())
    }

    fn harvest(self) -> CollectorStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ValueVector;
    use grafeo_common::types::Value;

    fn make_test_chunk(size: usize) -> DataChunk {
        let values: Vec<Value> = (0..size).map(|i| Value::from(i as i64)).collect();
        let column = ValueVector::from_values(&values);
        DataChunk::new(vec![column])
    }

    #[test]
    fn test_count_collector() {
        let collector = CountCollector;

        let mut pc = collector.for_partition(0);
        pc.collect(&make_test_chunk(10)).unwrap();
        pc.collect(&make_test_chunk(5)).unwrap();
        let count1 = pc.harvest();

        let mut pc2 = collector.for_partition(1);
        pc2.collect(&make_test_chunk(7)).unwrap();
        let count2 = pc2.harvest();

        let total = collector.merge(vec![count1, count2]);
        assert_eq!(total, 22);
    }

    #[test]
    fn test_materialize_collector() {
        let collector = MaterializeCollector;

        let mut pc = collector.for_partition(0);
        pc.collect(&make_test_chunk(10)).unwrap();
        pc.collect(&make_test_chunk(5)).unwrap();
        let chunks1 = pc.harvest();

        let mut pc2 = collector.for_partition(1);
        pc2.collect(&make_test_chunk(7)).unwrap();
        let chunks2 = pc2.harvest();

        let result = collector.merge(vec![chunks1, chunks2]);
        assert_eq!(result.len(), 3);
        assert_eq!(result.iter().map(|c| c.len()).sum::<usize>(), 22);
    }

    #[test]
    fn test_limit_collector() {
        let collector = LimitCollector::new(12);

        let mut pc = collector.for_partition(0);
        pc.collect(&make_test_chunk(10)).unwrap();
        pc.collect(&make_test_chunk(5)).unwrap(); // Only 2 more should be taken
        let result1 = pc.harvest();

        let mut pc2 = collector.for_partition(1);
        pc2.collect(&make_test_chunk(20)).unwrap();
        let result2 = pc2.harvest();

        let (chunks, total) = collector.merge(vec![result1, result2]);
        assert_eq!(total, 12);

        let actual_rows: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(actual_rows, 12);
    }

    #[test]
    fn test_stats_collector() {
        let collector = StatsCollector::new(0);

        let mut pc = collector.for_partition(0);

        // Create chunk with values 0..10
        let values: Vec<Value> = (0..10).map(|i| Value::from(i as i64)).collect();
        let column = ValueVector::from_values(&values);
        let chunk = DataChunk::new(vec![column]);

        pc.collect(&chunk).unwrap();
        let stats = pc.harvest();

        assert_eq!(stats.count, 10);
        assert!((stats.sum - 45.0).abs() < 0.001); // 0+1+2+...+9 = 45
        assert!((stats.min.unwrap() - 0.0).abs() < 0.001);
        assert!((stats.max.unwrap() - 9.0).abs() < 0.001);
        assert!((stats.avg().unwrap() - 4.5).abs() < 0.001);
    }

    #[test]
    fn test_stats_merge() {
        let collector = StatsCollector::new(0);

        // Partition 1: values 0..5
        let mut pc1 = collector.for_partition(0);
        let values1: Vec<Value> = (0..5).map(|i| Value::from(i as i64)).collect();
        let chunk1 = DataChunk::new(vec![ValueVector::from_values(&values1)]);
        pc1.collect(&chunk1).unwrap();

        // Partition 2: values 5..10
        let mut pc2 = collector.for_partition(1);
        let values2: Vec<Value> = (5..10).map(|i| Value::from(i as i64)).collect();
        let chunk2 = DataChunk::new(vec![ValueVector::from_values(&values2)]);
        pc2.collect(&chunk2).unwrap();

        let stats = collector.merge(vec![pc1.harvest(), pc2.harvest()]);

        assert_eq!(stats.count, 10);
        assert!((stats.min.unwrap() - 0.0).abs() < 0.001);
        assert!((stats.max.unwrap() - 9.0).abs() < 0.001);
    }
}
