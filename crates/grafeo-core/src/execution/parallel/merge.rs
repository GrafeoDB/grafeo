//! Merge utilities for parallel pipeline breakers.
//!
//! When parallel pipelines have pipeline breakers (Sort, Aggregate, Distinct),
//! each worker produces partial results that must be merged into final output.

use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::vector::ValueVector;
use grafeo_common::types::Value;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Trait for operators that support parallel merge.
///
/// Pipeline breakers must implement this to enable parallel execution.
pub trait MergeableOperator: Send + Sync {
    /// Merges partial results from another operator instance.
    fn merge_from(&mut self, other: Self)
    where
        Self: Sized;

    /// Returns whether this operator supports parallel merge.
    fn supports_parallel_merge(&self) -> bool {
        true
    }
}

/// Accumulator state that supports merging.
///
/// Used by aggregate operators to merge partial aggregations.
#[derive(Debug, Clone)]
pub struct MergeableAccumulator {
    /// Count of values.
    pub count: i64,
    /// Sum of values.
    pub sum: f64,
    /// Minimum value.
    pub min: Option<Value>,
    /// Maximum value.
    pub max: Option<Value>,
    /// First value encountered.
    pub first: Option<Value>,
    /// For AVG: sum of squared values (for variance if needed).
    pub sum_squared: f64,
}

impl MergeableAccumulator {
    /// Creates a new empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: None,
            max: None,
            first: None,
            sum_squared: 0.0,
        }
    }

    /// Adds a value to the accumulator.
    pub fn add(&mut self, value: &Value) {
        if matches!(value, Value::Null) {
            return;
        }

        self.count += 1;

        if let Some(n) = value_to_f64(value) {
            self.sum += n;
            self.sum_squared += n * n;
        }

        // Min
        if self.min.is_none() || compare_for_min(&self.min, value) {
            self.min = Some(value.clone());
        }

        // Max
        if self.max.is_none() || compare_for_max(&self.max, value) {
            self.max = Some(value.clone());
        }

        // First
        if self.first.is_none() {
            self.first = Some(value.clone());
        }
    }

    /// Merges another accumulator into this one.
    pub fn merge(&mut self, other: &MergeableAccumulator) {
        self.count += other.count;
        self.sum += other.sum;
        self.sum_squared += other.sum_squared;

        // Merge min
        if let Some(ref other_min) = other.min {
            if compare_for_min(&self.min, other_min) {
                self.min = Some(other_min.clone());
            }
        }

        // Merge max
        if let Some(ref other_max) = other.max {
            if compare_for_max(&self.max, other_max) {
                self.max = Some(other_max.clone());
            }
        }

        // Keep our first (we processed earlier)
        // If we have no first, take theirs
        if self.first.is_none() {
            self.first = other.first.clone();
        }
    }

    /// Finalizes COUNT aggregate.
    #[must_use]
    pub fn finalize_count(&self) -> Value {
        Value::Int64(self.count)
    }

    /// Finalizes SUM aggregate.
    #[must_use]
    pub fn finalize_sum(&self) -> Value {
        if self.count == 0 {
            Value::Null
        } else {
            Value::Float64(self.sum)
        }
    }

    /// Finalizes MIN aggregate.
    #[must_use]
    pub fn finalize_min(&self) -> Value {
        self.min.clone().unwrap_or(Value::Null)
    }

    /// Finalizes MAX aggregate.
    #[must_use]
    pub fn finalize_max(&self) -> Value {
        self.max.clone().unwrap_or(Value::Null)
    }

    /// Finalizes AVG aggregate.
    #[must_use]
    pub fn finalize_avg(&self) -> Value {
        if self.count == 0 {
            Value::Null
        } else {
            Value::Float64(self.sum / self.count as f64)
        }
    }

    /// Finalizes FIRST aggregate.
    #[must_use]
    pub fn finalize_first(&self) -> Value {
        self.first.clone().unwrap_or(Value::Null)
    }
}

impl Default for MergeableAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        _ => None,
    }
}

fn compare_for_min(current: &Option<Value>, new: &Value) -> bool {
    match (current, new) {
        (None, _) => true,
        (Some(Value::Int64(a)), Value::Int64(b)) => b < a,
        (Some(Value::Float64(a)), Value::Float64(b)) => b < a,
        (Some(Value::String(a)), Value::String(b)) => b < a,
        _ => false,
    }
}

fn compare_for_max(current: &Option<Value>, new: &Value) -> bool {
    match (current, new) {
        (None, _) => true,
        (Some(Value::Int64(a)), Value::Int64(b)) => b > a,
        (Some(Value::Float64(a)), Value::Float64(b)) => b > a,
        (Some(Value::String(a)), Value::String(b)) => b > a,
        _ => false,
    }
}

/// Sort key for k-way merge.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Column index to sort by.
    pub column: usize,
    /// Sort direction (ascending = true).
    pub ascending: bool,
    /// Nulls first (true) or last (false).
    pub nulls_first: bool,
}

impl SortKey {
    /// Creates an ascending sort key.
    #[must_use]
    pub fn ascending(column: usize) -> Self {
        Self {
            column,
            ascending: true,
            nulls_first: false,
        }
    }

    /// Creates a descending sort key.
    #[must_use]
    pub fn descending(column: usize) -> Self {
        Self {
            column,
            ascending: false,
            nulls_first: true,
        }
    }
}

/// Entry in the k-way merge heap.
struct MergeEntry {
    /// Row data.
    row: Vec<Value>,
    /// Source run index.
    run_index: usize,
    /// Row index within the run (for debugging/tracing).
    #[allow(dead_code)]
    row_index: usize,
    /// Sort keys for comparison.
    keys: Vec<SortKey>,
}

impl MergeEntry {
    fn compare_to(&self, other: &Self) -> Ordering {
        for key in &self.keys {
            let a = self.row.get(key.column);
            let b = other.row.get(key.column);

            let ordering = compare_values_for_sort(a, b, key.nulls_first);

            let ordering = if key.ascending {
                ordering
            } else {
                ordering.reverse()
            };

            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        Ordering::Equal
    }
}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.compare_to(other) == Ordering::Equal
    }
}

impl Eq for MergeEntry {}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap behavior (we want smallest first)
        other.compare_to(self)
    }
}

fn compare_values_for_sort(a: Option<&Value>, b: Option<&Value>, nulls_first: bool) -> Ordering {
    match (a, b) {
        (None, None) | (Some(Value::Null), Some(Value::Null)) => Ordering::Equal,
        (None, _) | (Some(Value::Null), _) => {
            if nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (_, None) | (_, Some(Value::Null)) => {
            if nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (Some(a), Some(b)) => compare_values(a, b),
    }
}

fn compare_values(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// Merges multiple sorted runs into a single sorted output.
///
/// Uses a min-heap for efficient k-way merge.
pub fn merge_sorted_runs(
    runs: Vec<Vec<Vec<Value>>>,
    keys: &[SortKey],
) -> Result<Vec<Vec<Value>>, OperatorError> {
    if runs.is_empty() {
        return Ok(Vec::new());
    }

    if runs.len() == 1 {
        // Invariant: runs.len() == 1 guarantees exactly one element
        return Ok(runs
            .into_iter()
            .next()
            .expect("runs has exactly one element: checked on previous line"));
    }

    // Count total rows
    let total_rows: usize = runs.iter().map(|r| r.len()).sum();
    let mut result = Vec::with_capacity(total_rows);

    // Track position in each run
    let mut positions: Vec<usize> = vec![0; runs.len()];

    // Initialize heap with first row from each non-empty run
    let mut heap = BinaryHeap::new();
    for (run_index, run) in runs.iter().enumerate() {
        if !run.is_empty() {
            heap.push(MergeEntry {
                row: run[0].clone(),
                run_index,
                row_index: 0,
                keys: keys.to_vec(),
            });
            positions[run_index] = 1;
        }
    }

    // Extract rows in order
    while let Some(entry) = heap.pop() {
        result.push(entry.row);

        // Add next row from same run if available
        let pos = positions[entry.run_index];
        if pos < runs[entry.run_index].len() {
            heap.push(MergeEntry {
                row: runs[entry.run_index][pos].clone(),
                run_index: entry.run_index,
                row_index: pos,
                keys: keys.to_vec(),
            });
            positions[entry.run_index] += 1;
        }
    }

    Ok(result)
}

/// Converts sorted rows to DataChunks.
pub fn rows_to_chunks(
    rows: Vec<Vec<Value>>,
    chunk_size: usize,
) -> Result<Vec<DataChunk>, OperatorError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let num_columns = rows[0].len();
    let num_chunks = (rows.len() + chunk_size - 1) / chunk_size;
    let mut chunks = Vec::with_capacity(num_chunks);

    for chunk_rows in rows.chunks(chunk_size) {
        let mut columns: Vec<ValueVector> = (0..num_columns).map(|_| ValueVector::new()).collect();

        for row in chunk_rows {
            for (col_idx, col) in columns.iter_mut().enumerate() {
                let val = row.get(col_idx).cloned().unwrap_or(Value::Null);
                col.push(val);
            }
        }

        chunks.push(DataChunk::new(columns));
    }

    Ok(chunks)
}

/// Merges multiple sorted DataChunk streams into a single sorted stream.
pub fn merge_sorted_chunks(
    runs: Vec<Vec<DataChunk>>,
    keys: &[SortKey],
    chunk_size: usize,
) -> Result<Vec<DataChunk>, OperatorError> {
    // Convert chunks to row format for merging
    let row_runs: Vec<Vec<Vec<Value>>> = runs.into_iter().map(chunks_to_rows).collect();

    let merged_rows = merge_sorted_runs(row_runs, keys)?;
    rows_to_chunks(merged_rows, chunk_size)
}

/// Converts DataChunks to row format.
fn chunks_to_rows(chunks: Vec<DataChunk>) -> Vec<Vec<Value>> {
    let mut rows = Vec::new();

    for chunk in chunks {
        let num_columns = chunk.num_columns();
        for i in 0..chunk.len() {
            let mut row = Vec::with_capacity(num_columns);
            for col_idx in 0..num_columns {
                let val = chunk
                    .column(col_idx)
                    .and_then(|c| c.get(i))
                    .unwrap_or(Value::Null);
                row.push(val);
            }
            rows.push(row);
        }
    }

    rows
}

/// Concatenates multiple DataChunk results (for non-sorted parallel results).
pub fn concat_parallel_results(results: Vec<Vec<DataChunk>>) -> Vec<DataChunk> {
    results.into_iter().flatten().collect()
}

/// Merges parallel DISTINCT results by deduplication.
pub fn merge_distinct_results(
    results: Vec<Vec<DataChunk>>,
) -> Result<Vec<DataChunk>, OperatorError> {
    use std::collections::HashSet;

    // Simple row-based deduplication using hash
    let mut seen: HashSet<u64> = HashSet::new();
    let mut unique_rows: Vec<Vec<Value>> = Vec::new();

    for chunks in results {
        for chunk in chunks {
            let num_columns = chunk.num_columns();
            for i in 0..chunk.len() {
                let mut row = Vec::with_capacity(num_columns);
                for col_idx in 0..num_columns {
                    let val = chunk
                        .column(col_idx)
                        .and_then(|c| c.get(i))
                        .unwrap_or(Value::Null);
                    row.push(val);
                }

                let hash = hash_row(&row);
                if seen.insert(hash) {
                    unique_rows.push(row);
                }
            }
        }
    }

    rows_to_chunks(unique_rows, 2048)
}

fn hash_row(row: &[Value]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for value in row {
        match value {
            Value::Null => 0u8.hash(&mut hasher),
            Value::Bool(b) => b.hash(&mut hasher),
            Value::Int64(i) => i.hash(&mut hasher),
            Value::Float64(f) => f.to_bits().hash(&mut hasher),
            Value::String(s) => s.hash(&mut hasher),
            _ => 0u8.hash(&mut hasher),
        }
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mergeable_accumulator() {
        let mut acc1 = MergeableAccumulator::new();
        acc1.add(&Value::Int64(10));
        acc1.add(&Value::Int64(20));

        let mut acc2 = MergeableAccumulator::new();
        acc2.add(&Value::Int64(30));
        acc2.add(&Value::Int64(40));

        acc1.merge(&acc2);

        assert_eq!(acc1.count, 4);
        assert_eq!(acc1.sum, 100.0);
        assert_eq!(acc1.finalize_min(), Value::Int64(10));
        assert_eq!(acc1.finalize_max(), Value::Int64(40));
        assert_eq!(acc1.finalize_avg(), Value::Float64(25.0));
    }

    #[test]
    fn test_merge_sorted_runs_empty() {
        let runs: Vec<Vec<Vec<Value>>> = Vec::new();
        let result = merge_sorted_runs(runs, &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_sorted_runs_single() {
        let runs = vec![vec![
            vec![Value::Int64(1)],
            vec![Value::Int64(2)],
            vec![Value::Int64(3)],
        ]];
        let keys = vec![SortKey::ascending(0)];

        let result = merge_sorted_runs(runs, &keys).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_merge_sorted_runs_multiple() {
        // Run 1: [1, 4, 7]
        // Run 2: [2, 5, 8]
        // Run 3: [3, 6, 9]
        let runs = vec![
            vec![
                vec![Value::Int64(1)],
                vec![Value::Int64(4)],
                vec![Value::Int64(7)],
            ],
            vec![
                vec![Value::Int64(2)],
                vec![Value::Int64(5)],
                vec![Value::Int64(8)],
            ],
            vec![
                vec![Value::Int64(3)],
                vec![Value::Int64(6)],
                vec![Value::Int64(9)],
            ],
        ];
        let keys = vec![SortKey::ascending(0)];

        let result = merge_sorted_runs(runs, &keys).unwrap();
        assert_eq!(result.len(), 9);

        // Verify sorted order
        for i in 0..9 {
            assert_eq!(result[i][0], Value::Int64((i + 1) as i64));
        }
    }

    #[test]
    fn test_merge_sorted_runs_descending() {
        let runs = vec![
            vec![
                vec![Value::Int64(7)],
                vec![Value::Int64(4)],
                vec![Value::Int64(1)],
            ],
            vec![
                vec![Value::Int64(8)],
                vec![Value::Int64(5)],
                vec![Value::Int64(2)],
            ],
        ];
        let keys = vec![SortKey::descending(0)];

        let result = merge_sorted_runs(runs, &keys).unwrap();
        assert_eq!(result.len(), 6);

        // Verify descending order
        assert_eq!(result[0][0], Value::Int64(8));
        assert_eq!(result[1][0], Value::Int64(7));
        assert_eq!(result[5][0], Value::Int64(1));
    }

    #[test]
    fn test_rows_to_chunks() {
        let rows = (0..10).map(|i| vec![Value::Int64(i)]).collect();
        let chunks = rows_to_chunks(rows, 3).unwrap();

        assert_eq!(chunks.len(), 4); // 10 rows / 3 = 4 chunks
        assert_eq!(chunks[0].len(), 3);
        assert_eq!(chunks[1].len(), 3);
        assert_eq!(chunks[2].len(), 3);
        assert_eq!(chunks[3].len(), 1);
    }

    #[test]
    fn test_merge_distinct_results() {
        let chunk1 = DataChunk::new(vec![ValueVector::from_values(&[
            Value::Int64(1),
            Value::Int64(2),
            Value::Int64(3),
        ])]);

        let chunk2 = DataChunk::new(vec![ValueVector::from_values(&[
            Value::Int64(2),
            Value::Int64(3),
            Value::Int64(4),
        ])]);

        let results = vec![vec![chunk1], vec![chunk2]];
        let merged = merge_distinct_results(results).unwrap();

        let total_rows: usize = merged.iter().map(DataChunk::len).sum();
        assert_eq!(total_rows, 4); // 1, 2, 3, 4 (no duplicates)
    }

    #[test]
    fn test_concat_parallel_results() {
        let chunk1 = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(1)])]);
        let chunk2 = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(2)])]);
        let chunk3 = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(3)])]);

        let results = vec![vec![chunk1], vec![chunk2, chunk3]];
        let concatenated = concat_parallel_results(results);

        assert_eq!(concatenated.len(), 3);
    }
}
