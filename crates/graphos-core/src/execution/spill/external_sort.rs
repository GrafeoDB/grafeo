//! External merge sort for out-of-core sorting.
//!
//! This module implements external sorting using sorted runs on disk.
//! When memory is exhausted, sorted buffers are written as runs to disk.
//! On finalization, all runs are merged using k-way merge.

use super::file::{SpillFile, SpillFileReader};
use super::manager::SpillManager;
use super::serializer::{deserialize_row, serialize_row};
use graphos_common::types::Value;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    /// Ascending order.
    Ascending,
    /// Descending order.
    Descending,
}

/// Null handling in sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullOrder {
    /// NULLs come first.
    First,
    /// NULLs come last.
    Last,
}

/// Sort key specification.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Column index to sort by.
    pub column: usize,
    /// Sort direction.
    pub direction: SortDirection,
    /// Null handling.
    pub null_order: NullOrder,
}

impl SortKey {
    /// Creates a new ascending sort key.
    #[must_use]
    pub fn ascending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Ascending,
            null_order: NullOrder::Last,
        }
    }

    /// Creates a new descending sort key.
    #[must_use]
    pub fn descending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Descending,
            null_order: NullOrder::First,
        }
    }
}

/// External merge sort for out-of-core sorting.
///
/// Manages sorted runs on disk and provides k-way merge.
pub struct ExternalSort {
    /// Spill manager for file creation.
    manager: Arc<SpillManager>,
    /// Sorted runs on disk.
    sorted_runs: Vec<SpillFile>,
    /// Number of rows in each run.
    run_row_counts: Vec<usize>,
    /// Number of columns per row.
    num_columns: usize,
    /// Sort keys.
    sort_keys: Vec<SortKey>,
}

impl ExternalSort {
    /// Creates a new external sort.
    #[must_use]
    pub fn new(manager: Arc<SpillManager>, num_columns: usize, sort_keys: Vec<SortKey>) -> Self {
        Self {
            manager,
            sorted_runs: Vec::new(),
            run_row_counts: Vec::new(),
            num_columns,
            sort_keys,
        }
    }

    /// Returns the number of runs on disk.
    #[must_use]
    pub fn num_runs(&self) -> usize {
        self.sorted_runs.len()
    }

    /// Returns the total number of rows across all runs.
    #[must_use]
    pub fn total_rows(&self) -> usize {
        self.run_row_counts.iter().sum()
    }

    /// Spills an already-sorted buffer as a run to disk.
    ///
    /// The buffer must already be sorted according to the sort keys.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to disk fails.
    pub fn spill_sorted_run(&mut self, rows: Vec<Vec<Value>>) -> std::io::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }

        let row_count = rows.len();
        let mut spill_file = self.manager.create_file("sort_run")?;

        // Write row count header
        spill_file.write_u64_le(row_count as u64)?;

        // Write all rows
        for row in &rows {
            serialize_row(row, &mut SpillFileWriter(&mut spill_file))?;
        }

        spill_file.finish_write()?;
        self.manager.register_spilled_bytes(spill_file.bytes_written());

        self.sorted_runs.push(spill_file);
        self.run_row_counts.push(row_count);

        Ok(())
    }

    /// Merges all runs and an optional in-memory buffer into sorted output.
    ///
    /// Uses k-way merge with a min-heap.
    ///
    /// # Errors
    ///
    /// Returns an error if reading from disk fails.
    pub fn merge_all(&mut self, in_memory_buffer: Vec<Vec<Value>>) -> std::io::Result<Vec<Vec<Value>>> {
        let num_runs = self.sorted_runs.len();
        let has_memory = !in_memory_buffer.is_empty();

        // Special case: no runs and no memory buffer
        if num_runs == 0 && !has_memory {
            return Ok(Vec::new());
        }

        // Special case: only in-memory buffer, no disk runs
        if num_runs == 0 {
            let mut sorted_buffer = in_memory_buffer;
            let keys = self.sort_keys.clone();
            sorted_buffer.sort_by(|a, b| compare_rows(a, b, &keys));
            return Ok(sorted_buffer);
        }

        // Special case: single run and no memory buffer
        if num_runs == 1 && !has_memory {
            return self.read_single_run(0);
        }

        // General case: k-way merge
        self.k_way_merge(in_memory_buffer)
    }

    /// Reads a single run from disk.
    fn read_single_run(&mut self, run_index: usize) -> std::io::Result<Vec<Vec<Value>>> {
        let spill_file = &self.sorted_runs[run_index];
        let mut reader = spill_file.reader()?;

        let row_count = reader.read_u64_le()? as usize;
        let mut rows = Vec::with_capacity(row_count);

        for _ in 0..row_count {
            let row = deserialize_row(&mut SpillFileReaderAdapter(&mut reader), self.num_columns)?;
            rows.push(row);
        }

        Ok(rows)
    }

    /// Performs k-way merge of all runs and an optional in-memory buffer.
    fn k_way_merge(&mut self, mut in_memory_buffer: Vec<Vec<Value>>) -> std::io::Result<Vec<Vec<Value>>> {
        let total_rows = self.total_rows() + in_memory_buffer.len();
        let mut result = Vec::with_capacity(total_rows);

        // Sort the in-memory buffer first
        if !in_memory_buffer.is_empty() {
            let keys = self.sort_keys.clone();
            in_memory_buffer.sort_by(|a, b| compare_rows(a, b, &keys));
        }

        // Create readers for all runs
        let mut run_readers: Vec<RunReader> = Vec::with_capacity(self.sorted_runs.len());
        for (idx, spill_file) in self.sorted_runs.iter().enumerate() {
            let mut reader = spill_file.reader()?;
            let row_count = reader.read_u64_le()? as usize;

            if row_count > 0 {
                // Read first row
                let first_row =
                    deserialize_row(&mut SpillFileReaderAdapter(&mut reader), self.num_columns)?;
                run_readers.push(RunReader {
                    reader,
                    remaining: row_count - 1,
                    current_row: Some(first_row),
                    run_index: idx,
                    num_columns: self.num_columns,
                });
            }
        }

        // Create in-memory iterator
        let mut memory_iter = in_memory_buffer.into_iter().peekable();
        let memory_run_index = self.sorted_runs.len();

        // Build initial heap
        let mut heap = BinaryHeap::new();

        for run_reader in &run_readers {
            if let Some(row) = &run_reader.current_row {
                heap.push(HeapEntry {
                    row: row.clone(),
                    run_index: run_reader.run_index,
                    sort_keys: self.sort_keys.clone(),
                });
            }
        }

        // Add first memory row if present
        if let Some(row) = memory_iter.peek() {
            heap.push(HeapEntry {
                row: row.clone(),
                run_index: memory_run_index,
                sort_keys: self.sort_keys.clone(),
            });
        }

        // Merge loop
        while let Some(entry) = heap.pop() {
            result.push(entry.row);

            if entry.run_index == memory_run_index {
                // Advance memory iterator
                memory_iter.next();
                if let Some(row) = memory_iter.peek() {
                    heap.push(HeapEntry {
                        row: row.clone(),
                        run_index: memory_run_index,
                        sort_keys: self.sort_keys.clone(),
                    });
                }
            } else {
                // Advance file run
                let run_reader = &mut run_readers[entry.run_index];
                if let Some(next_row) = run_reader.next_row()? {
                    heap.push(HeapEntry {
                        row: next_row,
                        run_index: entry.run_index,
                        sort_keys: self.sort_keys.clone(),
                    });
                }
            }
        }

        Ok(result)
    }

    /// Cleans up all spill files.
    pub fn cleanup(&mut self) {
        for file in self.sorted_runs.drain(..) {
            let bytes = file.bytes_written();
            let _ = file.delete();
            self.manager.unregister_spilled_bytes(bytes);
        }
        self.run_row_counts.clear();
    }
}

impl Drop for ExternalSort {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Helper struct for reading from a run.
struct RunReader {
    reader: SpillFileReader,
    remaining: usize,
    current_row: Option<Vec<Value>>,
    run_index: usize,
    num_columns: usize,
}

impl RunReader {
    fn next_row(&mut self) -> std::io::Result<Option<Vec<Value>>> {
        if self.remaining == 0 {
            self.current_row = None;
            return Ok(None);
        }

        let row = deserialize_row(&mut SpillFileReaderAdapter(&mut self.reader), self.num_columns)?;
        self.remaining -= 1;
        self.current_row = Some(row.clone());
        Ok(Some(row))
    }
}

/// Entry in the merge heap.
struct HeapEntry {
    row: Vec<Value>,
    run_index: usize,
    sort_keys: Vec<SortKey>,
}

impl Eq for HeapEntry {}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse comparison because BinaryHeap is a max-heap but we want min
        compare_rows(&other.row, &self.row, &self.sort_keys)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compares two rows by sort keys.
fn compare_rows(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for key in keys {
        let a_val = a.get(key.column);
        let b_val = b.get(key.column);

        let ordering = match (a_val, b_val) {
            (Some(Value::Null), Some(Value::Null)) => Ordering::Equal,
            (Some(Value::Null), _) => match key.null_order {
                NullOrder::First => Ordering::Less,
                NullOrder::Last => Ordering::Greater,
            },
            (_, Some(Value::Null)) => match key.null_order {
                NullOrder::First => Ordering::Greater,
                NullOrder::Last => Ordering::Less,
            },
            (Some(a), Some(b)) => compare_values(a, b),
            _ => Ordering::Equal,
        };

        let ordering = match key.direction {
            SortDirection::Ascending => ordering,
            SortDirection::Descending => ordering.reverse(),
        };

        if ordering != Ordering::Equal {
            return ordering;
        }
    }

    Ordering::Equal
}

/// Compares two values.
fn compare_values(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// Adapter to write to SpillFile through std::io::Write.
struct SpillFileWriter<'a>(&'a mut SpillFile);

impl<'a> std::io::Write for SpillFileWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Adapter to read from SpillFileReader through std::io::Read.
struct SpillFileReaderAdapter<'a>(&'a mut SpillFileReader);

impl<'a> std::io::Read for SpillFileReaderAdapter<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read_exact(buf)?;
        Ok(buf.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Returns (TempDir, SpillManager). TempDir must be kept alive as long as manager is used.
    fn create_manager() -> (TempDir, Arc<SpillManager>) {
        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());
        (temp_dir, manager)
    }

    fn row(values: &[i64]) -> Vec<Value> {
        values.iter().map(|&v| Value::Int64(v)).collect()
    }

    #[test]
    fn test_external_sort_empty() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        let result = sort.merge_all(Vec::new()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_external_sort_memory_only() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        let buffer = vec![row(&[3]), row(&[1]), row(&[2])];
        let result = sort.merge_all(buffer).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], row(&[1]));
        assert_eq!(result[1], row(&[2]));
        assert_eq!(result[2], row(&[3]));
    }

    #[test]
    fn test_external_sort_single_run() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        // Spill a sorted run
        let sorted_run = vec![row(&[1]), row(&[2]), row(&[3])];
        sort.spill_sorted_run(sorted_run).unwrap();

        assert_eq!(sort.num_runs(), 1);
        assert_eq!(sort.total_rows(), 3);

        let result = sort.merge_all(Vec::new()).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], row(&[1]));
        assert_eq!(result[1], row(&[2]));
        assert_eq!(result[2], row(&[3]));
    }

    #[test]
    fn test_external_sort_two_runs() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        // Spill two sorted runs
        sort.spill_sorted_run(vec![row(&[1]), row(&[3]), row(&[5])]).unwrap();
        sort.spill_sorted_run(vec![row(&[2]), row(&[4]), row(&[6])]).unwrap();

        assert_eq!(sort.num_runs(), 2);

        let result = sort.merge_all(Vec::new()).unwrap();
        assert_eq!(result.len(), 6);
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r, &row(&[(i + 1) as i64]));
        }
    }

    #[test]
    fn test_external_sort_runs_with_memory() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        // Spill a run
        sort.spill_sorted_run(vec![row(&[1]), row(&[4]), row(&[7])]).unwrap();

        // Merge with in-memory buffer
        let buffer = vec![row(&[6]), row(&[3]), row(&[5]), row(&[2])];
        let result = sort.merge_all(buffer).unwrap();

        assert_eq!(result.len(), 7);
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r, &row(&[(i + 1) as i64]));
        }
    }

    #[test]
    fn test_external_sort_descending() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::descending(0)]);

        sort.spill_sorted_run(vec![row(&[5]), row(&[3]), row(&[1])]).unwrap();
        sort.spill_sorted_run(vec![row(&[6]), row(&[4]), row(&[2])]).unwrap();

        let result = sort.merge_all(Vec::new()).unwrap();
        assert_eq!(result.len(), 6);
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r, &row(&[(6 - i) as i64]));
        }
    }

    #[test]
    fn test_external_sort_multi_column() {
        let (_temp_dir, manager) = create_manager();
        let sort_keys = vec![SortKey::ascending(0), SortKey::descending(1)];
        let mut sort = ExternalSort::new(manager, 2, sort_keys);

        // Rows: (group, value)
        sort.spill_sorted_run(vec![
            vec![Value::Int64(1), Value::Int64(30)],
            vec![Value::Int64(1), Value::Int64(10)],
            vec![Value::Int64(2), Value::Int64(20)],
        ]).unwrap();

        sort.spill_sorted_run(vec![
            vec![Value::Int64(1), Value::Int64(20)],
            vec![Value::Int64(2), Value::Int64(30)],
            vec![Value::Int64(2), Value::Int64(10)],
        ]).unwrap();

        let result = sort.merge_all(Vec::new()).unwrap();

        // Expected: sorted by col0 asc, then col1 desc
        // (1,30), (1,20), (1,10), (2,30), (2,20), (2,10)
        assert_eq!(result.len(), 6);
        assert_eq!(result[0], vec![Value::Int64(1), Value::Int64(30)]);
        assert_eq!(result[1], vec![Value::Int64(1), Value::Int64(20)]);
        assert_eq!(result[2], vec![Value::Int64(1), Value::Int64(10)]);
        assert_eq!(result[3], vec![Value::Int64(2), Value::Int64(30)]);
        assert_eq!(result[4], vec![Value::Int64(2), Value::Int64(20)]);
        assert_eq!(result[5], vec![Value::Int64(2), Value::Int64(10)]);
    }

    #[test]
    fn test_external_sort_with_nulls() {
        let (_temp_dir, manager) = create_manager();
        let sort_keys = vec![SortKey {
            column: 0,
            direction: SortDirection::Ascending,
            null_order: NullOrder::Last,
        }];
        let mut sort = ExternalSort::new(manager, 1, sort_keys);

        sort.spill_sorted_run(vec![
            vec![Value::Int64(1)],
            vec![Value::Int64(3)],
            vec![Value::Null],
        ]).unwrap();

        sort.spill_sorted_run(vec![
            vec![Value::Int64(2)],
            vec![Value::Null],
        ]).unwrap();

        let result = sort.merge_all(Vec::new()).unwrap();

        // Nulls should be last
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], vec![Value::Int64(1)]);
        assert_eq!(result[1], vec![Value::Int64(2)]);
        assert_eq!(result[2], vec![Value::Int64(3)]);
        assert_eq!(result[3], vec![Value::Null]);
        assert_eq!(result[4], vec![Value::Null]);
    }

    #[test]
    fn test_external_sort_many_runs() {
        let (_temp_dir, manager) = create_manager();
        let mut sort = ExternalSort::new(manager, 1, vec![SortKey::ascending(0)]);

        // Create 10 runs with interleaved values
        for i in 0..10 {
            let run: Vec<Vec<Value>> = (0..10).map(|j| row(&[i + j * 10])).collect();
            sort.spill_sorted_run(run).unwrap();
        }

        assert_eq!(sort.num_runs(), 10);
        assert_eq!(sort.total_rows(), 100);

        let result = sort.merge_all(Vec::new()).unwrap();
        assert_eq!(result.len(), 100);

        // Verify sorted order
        for (i, r) in result.iter().enumerate() {
            assert_eq!(r, &row(&[i as i64]));
        }
    }

    #[test]
    fn test_external_sort_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        {
            let mut sort = ExternalSort::new(Arc::clone(&manager), 1, vec![SortKey::ascending(0)]);
            sort.spill_sorted_run(vec![row(&[1]), row(&[2])]).unwrap();
            sort.spill_sorted_run(vec![row(&[3]), row(&[4])]).unwrap();

            assert!(manager.spilled_bytes() > 0);
            // sort dropped here
        }

        // After drop, spilled bytes should be cleaned up
        // (The manager still exists, but files are deleted)
    }
}
