//! DataChunk for batched tuple processing.

use super::selection::SelectionVector;
use super::vector::ValueVector;
use graphos_common::types::LogicalType;

/// Default chunk size (number of tuples).
pub const DEFAULT_CHUNK_SIZE: usize = 2048;

/// A chunk of data containing multiple columns.
///
/// DataChunk is the fundamental unit of data processing in vectorized execution.
/// It holds multiple ValueVectors (columns) and an optional SelectionVector
/// for filtering without copying.
#[derive(Debug)]
pub struct DataChunk {
    /// Column vectors.
    columns: Vec<ValueVector>,
    /// Selection vector (None means all rows are selected).
    selection: Option<SelectionVector>,
    /// Number of rows in this chunk.
    count: usize,
    /// Capacity of this chunk.
    capacity: usize,
}

impl DataChunk {
    /// Creates a new empty data chunk with the given schema.
    #[must_use]
    pub fn new(column_types: &[LogicalType]) -> Self {
        Self::with_capacity(column_types, DEFAULT_CHUNK_SIZE)
    }

    /// Creates a new data chunk with the given schema and capacity.
    #[must_use]
    pub fn with_capacity(column_types: &[LogicalType], capacity: usize) -> Self {
        let columns = column_types
            .iter()
            .map(|t| ValueVector::with_capacity(t.clone(), capacity))
            .collect();

        Self {
            columns,
            selection: None,
            count: 0,
            capacity,
        }
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns the number of rows (considering selection).
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.selection.as_ref().map_or(self.count, |s| s.len())
    }

    /// Returns the total number of rows (ignoring selection).
    #[must_use]
    pub fn total_row_count(&self) -> usize {
        self.count
    }

    /// Returns true if the chunk is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.row_count() == 0
    }

    /// Returns the capacity of this chunk.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if the chunk is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.count >= self.capacity
    }

    /// Gets a column by index.
    #[must_use]
    pub fn column(&self, index: usize) -> Option<&ValueVector> {
        self.columns.get(index)
    }

    /// Gets a mutable column by index.
    pub fn column_mut(&mut self, index: usize) -> Option<&mut ValueVector> {
        self.columns.get_mut(index)
    }

    /// Returns the selection vector.
    #[must_use]
    pub fn selection(&self) -> Option<&SelectionVector> {
        self.selection.as_ref()
    }

    /// Sets the selection vector.
    pub fn set_selection(&mut self, selection: SelectionVector) {
        self.selection = Some(selection);
    }

    /// Clears the selection vector (selects all rows).
    pub fn clear_selection(&mut self) {
        self.selection = None;
    }

    /// Sets the row count.
    pub fn set_count(&mut self, count: usize) {
        self.count = count;
    }

    /// Resets the chunk for reuse.
    pub fn reset(&mut self) {
        for col in &mut self.columns {
            col.clear();
        }
        self.selection = None;
        self.count = 0;
    }

    /// Flattens the selection by copying only selected rows.
    ///
    /// After this operation, selection is None and count equals the
    /// previously selected row count.
    pub fn flatten(&mut self) {
        if self.selection.is_none() {
            return;
        }

        // TODO: Implement actual flattening by copying selected rows
        // For now, just clear selection
        self.selection = None;
    }

    /// Returns an iterator over selected row indices.
    pub fn selected_indices(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match &self.selection {
            Some(sel) => Box::new(sel.iter()),
            None => Box::new(0..self.count),
        }
    }
}

/// Builder for creating DataChunks row by row.
pub struct DataChunkBuilder {
    chunk: DataChunk,
}

impl DataChunkBuilder {
    /// Creates a new builder with the given schema.
    #[must_use]
    pub fn new(column_types: &[LogicalType]) -> Self {
        Self {
            chunk: DataChunk::new(column_types),
        }
    }

    /// Creates a new builder with the given schema and capacity.
    #[must_use]
    pub fn with_capacity(column_types: &[LogicalType], capacity: usize) -> Self {
        Self {
            chunk: DataChunk::with_capacity(column_types, capacity),
        }
    }

    /// Returns the current row count.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.chunk.count
    }

    /// Returns true if the builder is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.chunk.is_full()
    }

    /// Gets a mutable column for appending values.
    pub fn column_mut(&mut self, index: usize) -> Option<&mut ValueVector> {
        self.chunk.column_mut(index)
    }

    /// Increments the row count.
    pub fn advance_row(&mut self) {
        self.chunk.count += 1;
    }

    /// Finishes building and returns the chunk.
    #[must_use]
    pub fn finish(self) -> DataChunk {
        self.chunk
    }

    /// Resets the builder for reuse.
    pub fn reset(&mut self) {
        self.chunk.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_creation() {
        let schema = [LogicalType::Int64, LogicalType::String];
        let chunk = DataChunk::new(&schema);

        assert_eq!(chunk.column_count(), 2);
        assert_eq!(chunk.row_count(), 0);
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_chunk_builder() {
        let schema = [LogicalType::Int64, LogicalType::String];
        let mut builder = DataChunkBuilder::new(&schema);

        // Add first row
        builder.column_mut(0).unwrap().push_int64(1);
        builder.column_mut(1).unwrap().push_string("hello");
        builder.advance_row();

        // Add second row
        builder.column_mut(0).unwrap().push_int64(2);
        builder.column_mut(1).unwrap().push_string("world");
        builder.advance_row();

        let chunk = builder.finish();

        assert_eq!(chunk.row_count(), 2);
        assert_eq!(chunk.column(0).unwrap().get_int64(0), Some(1));
        assert_eq!(chunk.column(1).unwrap().get_string(1), Some("world"));
    }

    #[test]
    fn test_chunk_selection() {
        let schema = [LogicalType::Int64];
        let mut builder = DataChunkBuilder::new(&schema);

        for i in 0..10 {
            builder.column_mut(0).unwrap().push_int64(i);
            builder.advance_row();
        }

        let mut chunk = builder.finish();
        assert_eq!(chunk.row_count(), 10);

        // Apply selection for even numbers
        let selection = SelectionVector::from_predicate(10, |i| i % 2 == 0);
        chunk.set_selection(selection);

        assert_eq!(chunk.row_count(), 5); // 0, 2, 4, 6, 8
        assert_eq!(chunk.total_row_count(), 10);
    }

    #[test]
    fn test_chunk_reset() {
        let schema = [LogicalType::Int64];
        let mut builder = DataChunkBuilder::new(&schema);

        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();

        let mut chunk = builder.finish();
        assert_eq!(chunk.row_count(), 1);

        chunk.reset();
        assert_eq!(chunk.row_count(), 0);
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_selected_indices() {
        let schema = [LogicalType::Int64];
        let mut chunk = DataChunk::new(&schema);
        chunk.set_count(5);

        // No selection - should iterate 0..5
        let indices: Vec<_> = chunk.selected_indices().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);

        // With selection
        let selection = SelectionVector::from_predicate(5, |i| i == 1 || i == 3);
        chunk.set_selection(selection);

        let indices: Vec<_> = chunk.selected_indices().collect();
        assert_eq!(indices, vec![1, 3]);
    }
}
