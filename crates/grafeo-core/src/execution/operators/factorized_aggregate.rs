//! Factorized aggregate operator for computing aggregates without flattening.
//!
//! This module provides operators that compute COUNT, SUM, AVG, MIN, MAX directly
//! on factorized data, avoiding the O(n²) or worse complexity of flattening first.
//!
//! # Performance
//!
//! For a 2-hop query with 100 sources, 10 neighbors each hop:
//!
//! - **Regular aggregate (flatten then count)**: Materialize 10,000 rows, then count
//! - **Factorized aggregate (count from multiplicities)**: O(1,100) operations
//!
//! This gives 10-100x speedups on aggregate queries.

use std::sync::Arc;

use super::{FactorizedResult, LazyFactorizedChainOperator, Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::factorized_chunk::FactorizedChunk;
use crate::execution::vector::ValueVector;
use grafeo_common::types::{LogicalType, Value};

/// Types of aggregates that can be computed on factorized data.
#[derive(Debug, Clone)]
pub enum FactorizedAggregate {
    /// COUNT(*) - total logical row count.
    Count,
    /// COUNT(column) - count non-null values in a column at deepest level.
    CountColumn {
        /// Column index within the deepest level.
        column_idx: usize,
    },
    /// SUM(column) - sum of values at deepest level, weighted by multiplicity.
    Sum {
        /// Column index within the deepest level.
        column_idx: usize,
    },
    /// AVG(column) - average of values at deepest level.
    Avg {
        /// Column index within the deepest level.
        column_idx: usize,
    },
    /// MIN(column) - minimum value at deepest level.
    Min {
        /// Column index within the deepest level.
        column_idx: usize,
    },
    /// MAX(column) - maximum value at deepest level.
    Max {
        /// Column index within the deepest level.
        column_idx: usize,
    },
}

impl FactorizedAggregate {
    /// Creates a COUNT(*) aggregate.
    #[must_use]
    pub fn count() -> Self {
        Self::Count
    }

    /// Creates a COUNT(column) aggregate.
    #[must_use]
    pub fn count_column(column_idx: usize) -> Self {
        Self::CountColumn { column_idx }
    }

    /// Creates a SUM aggregate.
    #[must_use]
    pub fn sum(column_idx: usize) -> Self {
        Self::Sum { column_idx }
    }

    /// Creates an AVG aggregate.
    #[must_use]
    pub fn avg(column_idx: usize) -> Self {
        Self::Avg { column_idx }
    }

    /// Creates a MIN aggregate.
    #[must_use]
    pub fn min(column_idx: usize) -> Self {
        Self::Min { column_idx }
    }

    /// Creates a MAX aggregate.
    #[must_use]
    pub fn max(column_idx: usize) -> Self {
        Self::Max { column_idx }
    }

    /// Computes this aggregate on a factorized chunk.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The factorized chunk to aggregate
    ///
    /// # Returns
    ///
    /// The aggregate result as a Value.
    ///
    /// # Note
    ///
    /// For multiple aggregates on the same chunk, prefer using
    /// [`compute_with_multiplicities`](Self::compute_with_multiplicities) with
    /// precomputed multiplicities to avoid O(levels) recomputation per aggregate.
    pub fn compute(&self, chunk: &FactorizedChunk) -> Value {
        // For aggregates that need multiplicities, compute them once
        let multiplicities = match self {
            Self::CountColumn { .. } | Self::Sum { .. } | Self::Avg { .. } => {
                Some(chunk.compute_path_multiplicities())
            }
            _ => None,
        };
        self.compute_with_multiplicities(chunk, multiplicities.as_deref())
    }

    /// Computes this aggregate using precomputed multiplicities.
    ///
    /// This is more efficient when computing multiple aggregates on the same chunk,
    /// as multiplicities only need to be computed once.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The factorized chunk to aggregate
    /// * `multiplicities` - Precomputed path multiplicities (from `chunk.path_multiplicities_cached()`)
    ///
    /// # Returns
    ///
    /// The aggregate result as a Value.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut chunk = get_factorized_chunk();
    /// let mults = chunk.path_multiplicities_cached();
    ///
    /// // Compute multiple aggregates with O(1) cached lookup each
    /// let count = FactorizedAggregate::count().compute_with_multiplicities(&chunk, Some(&mults));
    /// let sum = FactorizedAggregate::sum(0).compute_with_multiplicities(&chunk, Some(&mults));
    /// ```
    pub fn compute_with_multiplicities(
        &self,
        chunk: &FactorizedChunk,
        multiplicities: Option<&[usize]>,
    ) -> Value {
        match self {
            Self::Count => Value::Int64(chunk.count_rows() as i64),

            Self::CountColumn { column_idx } => {
                // Count non-null values at deepest level, weighted by multiplicity
                if chunk.level_count() == 0 {
                    return Value::Int64(0);
                }

                let deepest_idx = chunk.level_count() - 1;
                let deepest = match chunk.level(deepest_idx) {
                    Some(l) => l,
                    None => return Value::Int64(0),
                };

                let col = match deepest.column(*column_idx) {
                    Some(c) => c,
                    None => return Value::Int64(0),
                };

                // Use provided multiplicities or compute them
                let computed;
                let mults = match multiplicities {
                    Some(m) => m,
                    None => {
                        computed = chunk.compute_path_multiplicities();
                        &computed
                    }
                };

                let mut count: i64 = 0;
                for (phys_idx, mult) in mults.iter().enumerate() {
                    if let Some(value) = col.get_physical(phys_idx) {
                        if !matches!(value, Value::Null) {
                            count += *mult as i64;
                        }
                    }
                }

                Value::Int64(count)
            }

            Self::Sum { column_idx } => match chunk.sum_deepest(*column_idx) {
                Some(sum) => Value::Float64(sum),
                None => Value::Null,
            },

            Self::Avg { column_idx } => match chunk.avg_deepest(*column_idx) {
                Some(avg) => Value::Float64(avg),
                None => Value::Null,
            },

            Self::Min { column_idx } => chunk.min_deepest(*column_idx).unwrap_or(Value::Null),

            Self::Max { column_idx } => chunk.max_deepest(*column_idx).unwrap_or(Value::Null),
        }
    }

    /// Returns the output type for this aggregate.
    #[must_use]
    pub fn output_type(&self) -> LogicalType {
        match self {
            Self::Count | Self::CountColumn { .. } => LogicalType::Int64,
            Self::Sum { .. } | Self::Avg { .. } => LogicalType::Float64,
            // MIN/MAX preserve the input type, but we default to Any since we don't know
            Self::Min { .. } | Self::Max { .. } => LogicalType::Any,
        }
    }
}

/// An aggregate operator that works directly on factorized data.
///
/// This operator takes a `LazyFactorizedChainOperator` as input and computes
/// aggregates without flattening. This is the key to achieving 10-100x speedups
/// for aggregate queries on multi-hop traversals.
///
/// # Example
///
/// ```ignore
/// // Query: MATCH (a)->(b)->(c) RETURN COUNT(*)
/// let expand_chain = LazyFactorizedChainOperator::new(store, scan, steps);
/// let agg = FactorizedAggregateOperator::new(expand_chain, vec![FactorizedAggregate::count()]);
/// // Returns a single row with the count, computed in O(n) instead of O(n²)
/// ```
pub struct FactorizedAggregateOperator {
    /// The input operator providing factorized data.
    input: LazyFactorizedChainOperator,
    /// The aggregates to compute.
    aggregates: Vec<FactorizedAggregate>,
    /// Whether the operator has been executed.
    executed: bool,
}

impl FactorizedAggregateOperator {
    /// Creates a new factorized aggregate operator.
    ///
    /// # Arguments
    ///
    /// * `input` - The input operator (typically a `LazyFactorizedChainOperator`)
    /// * `aggregates` - The aggregates to compute
    pub fn new(input: LazyFactorizedChainOperator, aggregates: Vec<FactorizedAggregate>) -> Self {
        Self {
            input,
            aggregates,
            executed: false,
        }
    }

    /// Executes the aggregation on factorized input.
    fn execute(&mut self) -> OperatorResult {
        // Get the factorized result WITHOUT flattening
        let mut factorized = match self.input.next_factorized() {
            Ok(Some(chunk)) => chunk,
            Ok(None) => {
                // No input - return zeros/nulls for aggregates
                return Ok(Some(self.create_empty_result()));
            }
            Err(e) => return Err(e),
        };

        // Compute multiplicities once for all aggregates (cached internally)
        // This is the key optimization: O(levels) once instead of O(levels * num_aggregates)
        let multiplicities: Arc<[usize]> = factorized.path_multiplicities_cached();

        // Compute each aggregate using the cached multiplicities
        let output_cols: Vec<ValueVector> = self
            .aggregates
            .iter()
            .map(|agg| {
                let mut col = ValueVector::with_type(agg.output_type());
                col.push_value(agg.compute_with_multiplicities(&factorized, Some(&multiplicities)));
                col
            })
            .collect();

        // Create single-row result chunk
        let mut chunk = DataChunk::new(output_cols);
        chunk.set_count(1);

        Ok(Some(chunk))
    }

    /// Creates an empty result (for when input is empty).
    fn create_empty_result(&self) -> DataChunk {
        let cols: Vec<ValueVector> = self
            .aggregates
            .iter()
            .map(|agg| {
                let mut col = ValueVector::with_type(agg.output_type());
                match agg {
                    FactorizedAggregate::Count | FactorizedAggregate::CountColumn { .. } => {
                        col.push_value(Value::Int64(0));
                    }
                    _ => {
                        col.push_value(Value::Null);
                    }
                }
                col
            })
            .collect();

        let mut chunk = DataChunk::new(cols);
        chunk.set_count(1);
        chunk
    }
}

impl Operator for FactorizedAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        if self.executed {
            return Ok(None);
        }

        self.executed = true;
        self.execute()
    }

    fn reset(&mut self) {
        self.input.reset();
        self.executed = false;
    }

    fn name(&self) -> &'static str {
        "FactorizedAggregate"
    }
}

/// Trait for operators that can provide factorized output.
///
/// This trait allows the planner to check if an operator can provide
/// factorized data for factorized aggregation.
pub trait FactorizedOperator {
    /// Returns the next chunk as factorized data.
    fn next_factorized(&mut self) -> FactorizedResult;
}

impl FactorizedOperator for LazyFactorizedChainOperator {
    fn next_factorized(&mut self) -> FactorizedResult {
        LazyFactorizedChainOperator::next_factorized(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::factorized_chunk::FactorizationLevel;
    use crate::execution::factorized_vector::FactorizedVector;

    /// Creates a test factorized chunk with known structure for testing aggregates.
    ///
    /// Structure:
    /// - Level 0: 2 sources with values [10, 20]
    /// - Level 1: source 0 has 3 children [1, 2, 3], source 1 has 2 children [4, 5]
    ///
    /// Logical rows: (10, 1), (10, 2), (10, 3), (20, 4), (20, 5) = 5 rows
    fn create_test_factorized_chunk() -> FactorizedChunk {
        // Level 0: 2 sources
        let mut source_data = ValueVector::with_type(LogicalType::Int64);
        source_data.push_int64(10);
        source_data.push_int64(20);
        let level0 = FactorizationLevel::flat(
            vec![FactorizedVector::flat(source_data)],
            vec!["source".to_string()],
        );

        // Level 1: 5 children (3 for source 0, 2 for source 1)
        let mut child_data = ValueVector::with_type(LogicalType::Int64);
        child_data.push_int64(1);
        child_data.push_int64(2);
        child_data.push_int64(3);
        child_data.push_int64(4);
        child_data.push_int64(5);

        let offsets = vec![0u32, 3, 5];
        let child_vec = FactorizedVector::unflat(child_data, offsets, 2);
        let level1 =
            FactorizationLevel::unflat(vec![child_vec], vec!["child".to_string()], vec![3, 2]);

        // Build the chunk
        let mut chunk = FactorizedChunk::empty();
        chunk.add_factorized_level(level0);
        chunk.add_factorized_level(level1);
        chunk
    }

    #[test]
    fn test_count_aggregate() {
        let chunk = create_test_factorized_chunk();

        let agg = FactorizedAggregate::count();
        let result = agg.compute(&chunk);

        assert_eq!(result, Value::Int64(5));
    }

    #[test]
    fn test_sum_aggregate() {
        let chunk = create_test_factorized_chunk();

        // Sum of deepest level: 1 + 2 + 3 + 4 + 5 = 15
        let agg = FactorizedAggregate::sum(0);
        let result = agg.compute(&chunk);

        assert_eq!(result, Value::Float64(15.0));
    }

    #[test]
    fn test_avg_aggregate() {
        let chunk = create_test_factorized_chunk();

        // Avg of deepest level: 15 / 5 = 3.0
        let agg = FactorizedAggregate::avg(0);
        let result = agg.compute(&chunk);

        assert_eq!(result, Value::Float64(3.0));
    }

    #[test]
    fn test_min_aggregate() {
        let chunk = create_test_factorized_chunk();

        let agg = FactorizedAggregate::min(0);
        let result = agg.compute(&chunk);

        assert_eq!(result, Value::Int64(1));
    }

    #[test]
    fn test_max_aggregate() {
        let chunk = create_test_factorized_chunk();

        let agg = FactorizedAggregate::max(0);
        let result = agg.compute(&chunk);

        assert_eq!(result, Value::Int64(5));
    }

    #[test]
    fn test_multiplicity_weighted_sum() {
        // Create a 3-level chunk where multiplicities matter
        // Level 0: 1 source
        // Level 1: 2 children for the source
        // Level 2: child 0 has 3 grandchildren, child 1 has 2 grandchildren
        //
        // Logical rows = 1 * 2 * (weighted) = 5 (3 from child0, 2 from child1)
        // But the source value (10) appears in ALL 5 paths

        let mut source = ValueVector::with_type(LogicalType::Int64);
        source.push_int64(10);
        let level0 =
            FactorizationLevel::flat(vec![FactorizedVector::flat(source)], vec!["a".to_string()]);

        let mut children = ValueVector::with_type(LogicalType::Int64);
        children.push_int64(100);
        children.push_int64(200);
        let child_vec = FactorizedVector::unflat(children, vec![0, 2], 1);
        let level1 = FactorizationLevel::unflat(
            vec![child_vec],
            vec!["b".to_string()],
            vec![2], // 2 children for the 1 parent
        );

        // Grandchildren: child 0 (100) has [1, 2, 3], child 1 (200) has [4, 5]
        let mut grandchildren = ValueVector::with_type(LogicalType::Int64);
        grandchildren.push_int64(1);
        grandchildren.push_int64(2);
        grandchildren.push_int64(3);
        grandchildren.push_int64(4);
        grandchildren.push_int64(5);
        let gc_vec = FactorizedVector::unflat(grandchildren, vec![0, 3, 5], 2);
        let level2 = FactorizationLevel::unflat(
            vec![gc_vec],
            vec!["c".to_string()],
            vec![3, 2], // child 0 has 3, child 1 has 2
        );

        let mut chunk = FactorizedChunk::empty();
        chunk.add_factorized_level(level0);
        chunk.add_factorized_level(level1);
        chunk.add_factorized_level(level2);

        // Verify logical row count
        assert_eq!(chunk.logical_row_count(), 5);

        // SUM at deepest level (each grandchild has multiplicity 1)
        // = 1 + 2 + 3 + 4 + 5 = 15
        let sum_agg = FactorizedAggregate::sum(0);
        let sum_result = sum_agg.compute(&chunk);
        assert_eq!(sum_result, Value::Float64(15.0));

        // COUNT should be 5
        let count_agg = FactorizedAggregate::count();
        let count_result = count_agg.compute(&chunk);
        assert_eq!(count_result, Value::Int64(5));
    }

    #[test]
    fn test_empty_chunk_aggregates() {
        let chunk = FactorizedChunk::empty();

        assert_eq!(
            FactorizedAggregate::count().compute(&chunk),
            Value::Int64(0)
        );
        assert_eq!(FactorizedAggregate::sum(0).compute(&chunk), Value::Null);
        assert_eq!(FactorizedAggregate::min(0).compute(&chunk), Value::Null);
    }
}
