//! Improved iteration over factorized data.
//!
//! This module provides cache-friendly iteration over [`FactorizedChunk`] data
//! using pre-computed indices for better performance on multi-hop traversals.
//!
//! # Performance
//!
//! The original `FactorizedRowIterator` computes ranges on-the-fly, which can
//! be cache-unfriendly for deep traversals. This improved iterator:
//!
//! - **Pre-computes all indices** before iteration starts
//! - **Uses SmallVec** for stack allocation when level count is small
//! - **Provides `RowView`** for zero-copy value access
//!
//! # Example
//!
//! ```ignore
//! let chunk = get_factorized_chunk();
//!
//! // Pre-computed iteration (better for repeated access)
//! let iter = PrecomputedIter::new(&chunk);
//! for row in iter {
//!     let source_id = row.get_node_id(0, 0);
//!     let dest_id = row.get_node_id(1, 0);
//!     // ... process row
//! }
//! ```

use smallvec::SmallVec;

use super::factorized_chunk::FactorizedChunk;
use grafeo_common::types::{EdgeId, NodeId, Value};

/// Maximum number of levels to stack-allocate.
/// 4 levels covers most practical graph queries (source -> hop1 -> hop2 -> hop3).
const STACK_LEVELS: usize = 4;

/// Pre-computed indices for a single logical row.
///
/// Stores the physical index at each level for fast access.
#[derive(Debug, Clone)]
pub struct RowIndices {
    /// Physical indices at each level.
    /// Index 0 is the source level, higher indices are deeper levels.
    indices: SmallVec<[usize; STACK_LEVELS]>,
}

impl RowIndices {
    /// Creates new row indices from a slice.
    #[must_use]
    pub fn new(indices: &[usize]) -> Self {
        Self {
            indices: SmallVec::from_slice(indices),
        }
    }

    /// Returns the physical index at a given level.
    #[must_use]
    pub fn get(&self, level: usize) -> Option<usize> {
        self.indices.get(level).copied()
    }

    /// Returns the number of levels.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.indices.len()
    }

    /// Returns the indices as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.indices
    }
}

/// A view into a single logical row of a factorized chunk.
///
/// Provides zero-copy access to values at each level without
/// materializing the entire row.
#[derive(Debug, Clone)]
pub struct RowView<'a> {
    chunk: &'a FactorizedChunk,
    indices: RowIndices,
}

impl<'a> RowView<'a> {
    /// Creates a new row view.
    #[must_use]
    pub fn new(chunk: &'a FactorizedChunk, indices: RowIndices) -> Self {
        Self { chunk, indices }
    }

    /// Creates a new row view from a reference to indices (clones the indices).
    #[must_use]
    pub fn from_ref(chunk: &'a FactorizedChunk, indices: &RowIndices) -> Self {
        Self {
            chunk,
            indices: indices.clone(),
        }
    }

    /// Gets a value at a specific level and column.
    #[must_use]
    pub fn get(&self, level: usize, column: usize) -> Option<Value> {
        let physical_idx = self.indices.get(level)?;
        let level_data = self.chunk.level(level)?;
        let col = level_data.column(column)?;
        col.get_physical(physical_idx)
    }

    /// Gets a NodeId at a specific level and column.
    #[must_use]
    pub fn get_node_id(&self, level: usize, column: usize) -> Option<NodeId> {
        let physical_idx = self.indices.get(level)?;
        let level_data = self.chunk.level(level)?;
        let col = level_data.column(column)?;
        col.get_node_id_physical(physical_idx)
    }

    /// Gets an EdgeId at a specific level and column.
    #[must_use]
    pub fn get_edge_id(&self, level: usize, column: usize) -> Option<EdgeId> {
        let physical_idx = self.indices.get(level)?;
        let level_data = self.chunk.level(level)?;
        let col = level_data.column(column)?;
        col.get_edge_id_physical(physical_idx)
    }

    /// Returns the number of levels in this row.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.indices.level_count()
    }

    /// Returns an iterator over all values in this row.
    ///
    /// Values are yielded in order: all columns at level 0, then level 1, etc.
    pub fn values(&self) -> impl Iterator<Item = Value> + '_ {
        (0..self.level_count()).flat_map(move |level| {
            let physical_idx = self.indices.get(level).unwrap_or(0);
            let level_data = self.chunk.level(level);
            let col_count = level_data.map_or(0, |l| l.column_count());

            (0..col_count).filter_map(move |col| {
                level_data
                    .and_then(|l| l.column(col))
                    .and_then(|c| c.get_physical(physical_idx))
            })
        })
    }

    /// Materializes this row into a vector of values.
    #[must_use]
    pub fn to_vec(&self) -> Vec<Value> {
        self.values().collect()
    }
}

/// Pre-computed iterator over logical rows in a factorized chunk.
///
/// This iterator pre-computes all row indices upfront, providing:
/// - Better cache locality during iteration
/// - O(1) random access to any row
/// - Efficient parallel processing (indices can be partitioned)
///
/// # Trade-offs
///
/// - **Memory**: O(logical_rows * levels) for pre-computed indices
/// - **Startup**: O(logical_rows) to compute all indices
/// - **Iteration**: O(1) per row (just index lookup)
///
/// Use this when:
/// - You'll iterate multiple times
/// - You need random access to rows
/// - You want to parallelize processing
///
/// Use the original `FactorizedRowIterator` when:
/// - You'll iterate once
/// - Memory is constrained
/// - You don't need random access
#[derive(Debug)]
pub struct PrecomputedIter<'a> {
    chunk: &'a FactorizedChunk,
    /// Pre-computed row indices.
    rows: Vec<RowIndices>,
    /// Current position in the iteration.
    position: usize,
}

impl<'a> PrecomputedIter<'a> {
    /// Creates a new pre-computed iterator.
    ///
    /// This pre-computes all row indices upfront, which takes O(logical_rows)
    /// time but enables O(1) access per row during iteration.
    #[must_use]
    pub fn new(chunk: &'a FactorizedChunk) -> Self {
        let rows = Self::compute_all_indices(chunk);
        Self {
            chunk,
            rows,
            position: 0,
        }
    }

    /// Pre-computes all row indices for the chunk.
    fn compute_all_indices(chunk: &FactorizedChunk) -> Vec<RowIndices> {
        let level_count = chunk.level_count();
        if level_count == 0 {
            return Vec::new();
        }

        let logical_rows = chunk.logical_row_count();
        let mut rows = Vec::with_capacity(logical_rows);

        // Use a stack-based approach to avoid recursion
        let mut indices: SmallVec<[usize; STACK_LEVELS]> = SmallVec::new();
        indices.resize(level_count, 0);

        Self::enumerate_rows_iterative(chunk, &mut indices, &mut rows);

        rows
    }

    /// Iteratively enumerate all valid row index combinations.
    fn enumerate_rows_iterative(
        chunk: &FactorizedChunk,
        initial_indices: &mut SmallVec<[usize; STACK_LEVELS]>,
        rows: &mut Vec<RowIndices>,
    ) {
        let level_count = chunk.level_count();
        if level_count == 0 {
            return;
        }

        // Initialize all levels to their starting positions
        for level in 0..level_count {
            if level == 0 {
                initial_indices[0] = 0;
            } else {
                let parent_idx = initial_indices[level - 1];
                if let Some(col) = chunk.level(level).and_then(|l| l.column(0)) {
                    let (start, _) = col.range_for_parent(parent_idx);
                    initial_indices[level] = start;
                }
            }
        }

        // Check if initial position is valid
        if !Self::is_valid_position(chunk, initial_indices) {
            if !Self::advance_to_next_valid(chunk, initial_indices) {
                return; // No valid rows
            }
        }

        // Enumerate all rows
        loop {
            // Record current valid row
            rows.push(RowIndices::new(initial_indices));

            // Advance to next row
            if !Self::advance_to_next_valid(chunk, initial_indices) {
                break;
            }
        }
    }

    /// Checks if the current position represents a valid row.
    fn is_valid_position(
        chunk: &FactorizedChunk,
        indices: &SmallVec<[usize; STACK_LEVELS]>,
    ) -> bool {
        let level_count = chunk.level_count();

        for level in 0..level_count {
            if level == 0 {
                let level_data = match chunk.level(0) {
                    Some(l) => l,
                    None => return false,
                };
                if indices[0] >= level_data.group_count() {
                    return false;
                }
            } else {
                let parent_idx = indices[level - 1];
                if let Some(col) = chunk.level(level).and_then(|l| l.column(0)) {
                    let (start, end) = col.range_for_parent(parent_idx);
                    if start >= end || indices[level] < start || indices[level] >= end {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        true
    }

    /// Advances to the next valid position.
    fn advance_to_next_valid(
        chunk: &FactorizedChunk,
        indices: &mut SmallVec<[usize; STACK_LEVELS]>,
    ) -> bool {
        let level_count = chunk.level_count();
        if level_count == 0 {
            return false;
        }

        loop {
            // Try to advance, starting from deepest level
            let mut advanced = false;

            for level in (0..level_count).rev() {
                let (_start, end) = if level == 0 {
                    let level_data = match chunk.level(0) {
                        Some(l) => l,
                        None => return false,
                    };
                    (0, level_data.group_count())
                } else {
                    let parent_idx = indices[level - 1];
                    if let Some(col) = chunk.level(level).and_then(|l| l.column(0)) {
                        col.range_for_parent(parent_idx)
                    } else {
                        (0, 0)
                    }
                };

                if indices[level] + 1 < end {
                    // Can advance at this level
                    indices[level] += 1;

                    // Reset all deeper levels to their start positions
                    for deeper in (level + 1)..level_count {
                        let deeper_parent = indices[deeper - 1];
                        if let Some(col) = chunk.level(deeper).and_then(|l| l.column(0)) {
                            let (deeper_start, _) = col.range_for_parent(deeper_parent);
                            indices[deeper] = deeper_start;
                        }
                    }

                    advanced = true;
                    break;
                }
                // Can't advance at this level, try parent
            }

            if !advanced {
                return false; // Exhausted all levels
            }

            // Check if new position is valid
            if Self::is_valid_position(chunk, indices) {
                return true;
            }
            // Otherwise, keep advancing
        }
    }

    /// Returns the total number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns true if there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Gets row indices by position (random access).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&RowIndices> {
        self.rows.get(index)
    }

    /// Gets a row view by position (random access).
    #[must_use]
    pub fn row(&self, index: usize) -> Option<RowView<'a>> {
        self.rows
            .get(index)
            .map(|indices| RowView::from_ref(self.chunk, indices))
    }

    /// Returns an iterator over row views.
    pub fn rows(&self) -> impl Iterator<Item = RowView<'a>> + '_ {
        self.rows
            .iter()
            .map(|indices| RowView::from_ref(self.chunk, indices))
    }

    /// Resets the iterator to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

impl<'a> Iterator for PrecomputedIter<'a> {
    type Item = RowView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.rows.len() {
            return None;
        }

        let indices = &self.rows[self.position];
        self.position += 1;
        Some(RowView::from_ref(self.chunk, indices))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rows.len() - self.position;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PrecomputedIter<'_> {}

/// Streaming iterator that doesn't pre-compute indices.
///
/// This is more memory-efficient than [`PrecomputedIter`] but slower
/// for repeated iteration.
#[derive(Debug)]
pub struct StreamingIter<'a> {
    chunk: &'a FactorizedChunk,
    indices: SmallVec<[usize; STACK_LEVELS]>,
    exhausted: bool,
    started: bool,
}

impl<'a> StreamingIter<'a> {
    /// Creates a new streaming iterator.
    #[must_use]
    pub fn new(chunk: &'a FactorizedChunk) -> Self {
        let level_count = chunk.level_count();
        let mut indices = SmallVec::new();
        indices.resize(level_count, 0);

        let mut iter = Self {
            chunk,
            indices,
            exhausted: level_count == 0,
            started: false,
        };

        // Initialize to first valid position
        if !iter.exhausted {
            iter.initialize_indices();
            if !PrecomputedIter::is_valid_position(chunk, &iter.indices) {
                iter.exhausted = !PrecomputedIter::advance_to_next_valid(chunk, &mut iter.indices);
            }
        }

        iter
    }

    /// Initializes all level indices to their starting positions.
    fn initialize_indices(&mut self) {
        let level_count = self.chunk.level_count();
        for level in 0..level_count {
            if level == 0 {
                self.indices[0] = 0;
            } else {
                let parent_idx = self.indices[level - 1];
                if let Some(col) = self.chunk.level(level).and_then(|l| l.column(0)) {
                    let (start, _) = col.range_for_parent(parent_idx);
                    self.indices[level] = start;
                }
            }
        }
    }

    /// Returns the current indices.
    #[must_use]
    pub fn current_indices(&self) -> Option<RowIndices> {
        if self.exhausted {
            None
        } else {
            Some(RowIndices::new(&self.indices))
        }
    }

    /// Resets to the beginning.
    pub fn reset(&mut self) {
        self.started = false;
        self.exhausted = self.chunk.level_count() == 0;
        if !self.exhausted {
            self.initialize_indices();
            if !PrecomputedIter::is_valid_position(self.chunk, &self.indices) {
                self.exhausted =
                    !PrecomputedIter::advance_to_next_valid(self.chunk, &mut self.indices);
            }
        }
    }
}

impl<'a> Iterator for StreamingIter<'a> {
    type Item = RowIndices;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        if self.started {
            // Advance to next valid position
            if !PrecomputedIter::advance_to_next_valid(self.chunk, &mut self.indices) {
                self.exhausted = true;
                return None;
            }
        } else {
            self.started = true;
        }

        Some(RowIndices::new(&self.indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::factorized_chunk::FactorizationLevel;
    use crate::execution::factorized_vector::FactorizedVector;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::{LogicalType, NodeId};

    fn create_test_chunk() -> FactorizedChunk {
        // Level 0: 2 sources with values [10, 20]
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

        let mut chunk = FactorizedChunk::empty();
        chunk.add_factorized_level(level0);
        chunk.add_factorized_level(level1);
        chunk
    }

    fn create_node_chunk() -> FactorizedChunk {
        let mut source_data = ValueVector::with_type(LogicalType::Node);
        source_data.push_node_id(NodeId::new(100));
        source_data.push_node_id(NodeId::new(200));
        let level0 = FactorizationLevel::flat(
            vec![FactorizedVector::flat(source_data)],
            vec!["source".to_string()],
        );

        let mut chunk = FactorizedChunk::empty();
        chunk.add_factorized_level(level0);
        chunk
    }

    #[test]
    fn test_row_indices_new() {
        let indices = RowIndices::new(&[0, 1, 2]);
        assert_eq!(indices.level_count(), 3);
        assert_eq!(indices.get(0), Some(0));
        assert_eq!(indices.get(1), Some(1));
        assert_eq!(indices.get(2), Some(2));
        assert_eq!(indices.get(3), None);
    }

    #[test]
    fn test_row_indices_as_slice() {
        let indices = RowIndices::new(&[5, 10, 15]);
        assert_eq!(indices.as_slice(), &[5, 10, 15]);
    }

    #[test]
    fn test_row_view_new() {
        let chunk = create_test_chunk();
        let indices = RowIndices::new(&[0, 0]);

        let view = RowView::new(&chunk, indices);
        assert_eq!(view.level_count(), 2);
    }

    #[test]
    fn test_row_view_from_ref() {
        let chunk = create_test_chunk();
        let indices = RowIndices::new(&[0, 1]);

        let view = RowView::from_ref(&chunk, &indices);
        assert_eq!(view.get(0, 0), Some(Value::Int64(10)));
        assert_eq!(view.get(1, 0), Some(Value::Int64(2)));
    }

    #[test]
    fn test_row_view_get_node_id() {
        let chunk = create_node_chunk();
        let indices = RowIndices::new(&[0]);

        let view = RowView::new(&chunk, indices);
        assert_eq!(view.get_node_id(0, 0), Some(NodeId::new(100)));
    }

    #[test]
    fn test_row_view_get_invalid() {
        let chunk = create_test_chunk();
        let indices = RowIndices::new(&[0, 0]);

        let view = RowView::new(&chunk, indices);

        // Invalid level
        assert_eq!(view.get(10, 0), None);

        // Invalid column
        assert_eq!(view.get(0, 10), None);
    }

    #[test]
    fn test_row_view_values() {
        let chunk = create_test_chunk();
        let indices = RowIndices::new(&[0, 0]);

        let view = RowView::new(&chunk, indices);
        let values: Vec<Value> = view.values().collect();

        assert_eq!(values.len(), 2);
        assert_eq!(values[0], Value::Int64(10));
        assert_eq!(values[1], Value::Int64(1));
    }

    #[test]
    fn test_row_view_to_vec() {
        let chunk = create_test_chunk();
        let indices = RowIndices::new(&[1, 4]);

        let view = RowView::new(&chunk, indices);
        let vec = view.to_vec();

        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], Value::Int64(20));
        assert_eq!(vec[1], Value::Int64(5));
    }

    #[test]
    fn test_precomputed_iter_count() {
        let chunk = create_test_chunk();
        let iter = PrecomputedIter::new(&chunk);

        // Should have 5 logical rows: (10,1), (10,2), (10,3), (20,4), (20,5)
        assert_eq!(iter.len(), 5);
    }

    #[test]
    fn test_precomputed_iter_values() {
        let chunk = create_test_chunk();
        let iter = PrecomputedIter::new(&chunk);

        let rows: Vec<Vec<Value>> = iter.map(|row| row.to_vec()).collect();

        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0], vec![Value::Int64(10), Value::Int64(1)]);
        assert_eq!(rows[1], vec![Value::Int64(10), Value::Int64(2)]);
        assert_eq!(rows[2], vec![Value::Int64(10), Value::Int64(3)]);
        assert_eq!(rows[3], vec![Value::Int64(20), Value::Int64(4)]);
        assert_eq!(rows[4], vec![Value::Int64(20), Value::Int64(5)]);
    }

    #[test]
    fn test_precomputed_iter_get() {
        let chunk = create_test_chunk();
        let iter = PrecomputedIter::new(&chunk);

        let indices = iter.get(2).unwrap();
        assert_eq!(indices.as_slice(), &[0, 2]);

        assert!(iter.get(10).is_none());
    }

    #[test]
    fn test_precomputed_iter_rows() {
        let chunk = create_test_chunk();
        let iter = PrecomputedIter::new(&chunk);

        let rows: Vec<RowView> = iter.rows().collect();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn test_precomputed_iter_reset() {
        let chunk = create_test_chunk();
        let mut iter = PrecomputedIter::new(&chunk);

        // Consume some items
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert_eq!(iter.size_hint().0, 3);

        // Reset
        iter.reset();
        assert_eq!(iter.size_hint().0, 5);
    }

    #[test]
    fn test_row_view_get() {
        let chunk = create_test_chunk();
        let iter = PrecomputedIter::new(&chunk);

        let first_row = iter.row(0).unwrap();
        assert_eq!(first_row.get(0, 0), Some(Value::Int64(10)));
        assert_eq!(first_row.get(1, 0), Some(Value::Int64(1)));

        let last_row = iter.row(4).unwrap();
        assert_eq!(last_row.get(0, 0), Some(Value::Int64(20)));
        assert_eq!(last_row.get(1, 0), Some(Value::Int64(5)));

        assert!(iter.row(10).is_none());
    }

    #[test]
    fn test_streaming_iter() {
        let chunk = create_test_chunk();
        let iter = StreamingIter::new(&chunk);

        let indices: Vec<RowIndices> = iter.collect();

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0].as_slice(), &[0, 0]);
        assert_eq!(indices[1].as_slice(), &[0, 1]);
        assert_eq!(indices[2].as_slice(), &[0, 2]);
        assert_eq!(indices[3].as_slice(), &[1, 3]);
        assert_eq!(indices[4].as_slice(), &[1, 4]);
    }

    #[test]
    fn test_streaming_iter_current_indices() {
        let chunk = create_test_chunk();
        let iter = StreamingIter::new(&chunk);

        let current = iter.current_indices();
        assert!(current.is_some());
        assert_eq!(current.unwrap().as_slice(), &[0, 0]);
    }

    #[test]
    fn test_streaming_iter_reset() {
        let chunk = create_test_chunk();
        let mut iter = StreamingIter::new(&chunk);

        // Consume some items
        iter.next();
        iter.next();

        // Reset
        iter.reset();

        // Should start from beginning again
        let first = iter.next().unwrap();
        assert_eq!(first.as_slice(), &[0, 0]);
    }

    #[test]
    fn test_streaming_iter_exhausted() {
        let chunk = create_test_chunk();
        let mut iter = StreamingIter::new(&chunk);

        // Consume all items
        while iter.next().is_some() {}

        // Should return None for current indices
        assert!(iter.current_indices().is_none());
    }

    #[test]
    fn test_empty_chunk() {
        let chunk = FactorizedChunk::empty();
        let iter = PrecomputedIter::new(&chunk);

        assert!(iter.is_empty());
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn test_empty_chunk_streaming() {
        let chunk = FactorizedChunk::empty();
        let mut iter = StreamingIter::new(&chunk);

        assert!(iter.next().is_none());
        assert!(iter.current_indices().is_none());
    }

    #[test]
    fn test_random_access() {
        let chunk = create_test_chunk();
        let iter = PrecomputedIter::new(&chunk);

        // Random access should work
        let row2 = iter.row(2).unwrap();
        let row4 = iter.row(4).unwrap();
        let row0 = iter.row(0).unwrap();

        assert_eq!(row2.get(1, 0), Some(Value::Int64(3)));
        assert_eq!(row4.get(1, 0), Some(Value::Int64(5)));
        assert_eq!(row0.get(1, 0), Some(Value::Int64(1)));
    }

    #[test]
    fn test_exact_size_iterator() {
        let chunk = create_test_chunk();
        let iter = PrecomputedIter::new(&chunk);

        assert_eq!(iter.len(), 5);
        assert_eq!(iter.size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_single_level_chunk() {
        let mut source_data = ValueVector::with_type(LogicalType::Int64);
        source_data.push_int64(1);
        source_data.push_int64(2);
        source_data.push_int64(3);
        let level0 = FactorizationLevel::flat(
            vec![FactorizedVector::flat(source_data)],
            vec!["value".to_string()],
        );

        let mut chunk = FactorizedChunk::empty();
        chunk.add_factorized_level(level0);

        let iter = PrecomputedIter::new(&chunk);
        assert_eq!(iter.len(), 3);

        let streaming = StreamingIter::new(&chunk);
        let indices: Vec<RowIndices> = streaming.collect();
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_row_indices_clone() {
        let indices = RowIndices::new(&[1, 2, 3]);
        let cloned = indices.clone();

        assert_eq!(indices.as_slice(), cloned.as_slice());
    }

    #[test]
    fn test_row_view_level_count() {
        let chunk = create_test_chunk();
        let indices = RowIndices::new(&[0, 0]);
        let view = RowView::new(&chunk, indices);

        assert_eq!(view.level_count(), 2);
    }
}
