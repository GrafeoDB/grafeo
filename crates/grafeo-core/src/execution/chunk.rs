//! DataChunk - the fundamental unit of vectorized execution.
//!
//! A DataChunk holds a batch of rows in columnar format. Processing data in
//! batches (typically 1024-2048 rows) lets the CPU stay busy and enables SIMD.

use super::selection::SelectionVector;
use super::vector::ValueVector;
use crate::index::ZoneMapEntry;
use grafeo_common::types::LogicalType;
use grafeo_common::utils::hash::FxHashMap;

/// Default chunk size (number of tuples).
pub const DEFAULT_CHUNK_SIZE: usize = 2048;

/// Zone map hints for chunk-level predicate pruning.
///
/// When a scan operator loads a chunk, it can attach zone map statistics
/// (min/max values) for each column. The filter operator can then skip
/// the entire chunk without evaluating rows when the zone map proves
/// no rows can match the predicate.
///
/// # Example
///
/// If filtering `age > 100` and the chunk's max age is 80, the entire
/// chunk can be skipped because no row can possibly match.
#[derive(Debug, Clone, Default)]
pub struct ChunkZoneHints {
    /// Zone map entries keyed by column index.
    pub column_hints: FxHashMap<usize, ZoneMapEntry>,
}

/// A batch of rows stored column-wise for vectorized processing.
///
/// Instead of storing rows like `[(a1,b1), (a2,b2), ...]`, we store columns
/// like `[a1,a2,...], [b1,b2,...]`. This is cache-friendly for analytical
/// queries that touch few columns but many rows.
///
/// The optional `SelectionVector` lets you filter rows without copying data -
/// just mark which row indices are "selected".
///
/// # Example
///
/// ```
/// use grafeo_core::execution::DataChunk;
/// use grafeo_core::execution::ValueVector;
/// use grafeo_common::types::Value;
///
/// // Create columns
/// let names = ValueVector::from_values(&[Value::from("Alice"), Value::from("Bob")]);
/// let ages = ValueVector::from_values(&[Value::from(30i64), Value::from(25i64)]);
///
/// // Bundle into a chunk
/// let chunk = DataChunk::new(vec![names, ages]);
/// assert_eq!(chunk.len(), 2);
/// ```
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
    /// Zone map hints for chunk-level filtering (optional).
    zone_hints: Option<ChunkZoneHints>,
}

impl DataChunk {
    /// Creates an empty data chunk with no columns.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            selection: None,
            count: 0,
            capacity: 0,
            zone_hints: None,
        }
    }

    /// Creates a new data chunk from existing vectors.
    #[must_use]
    pub fn new(columns: Vec<ValueVector>) -> Self {
        let count = columns.first().map_or(0, ValueVector::len);
        let capacity = columns.first().map_or(DEFAULT_CHUNK_SIZE, |c| c.len());
        Self {
            columns,
            selection: None,
            count,
            capacity,
            zone_hints: None,
        }
    }

    /// Creates a new empty data chunk with the given schema.
    #[must_use]
    pub fn with_schema(column_types: &[LogicalType]) -> Self {
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
            zone_hints: None,
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

    /// Alias for row_count().
    #[must_use]
    pub fn len(&self) -> usize {
        self.row_count()
    }

    /// Returns all columns.
    #[must_use]
    pub fn columns(&self) -> &[ValueVector] {
        &self.columns
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

    /// Sets zone map hints for this chunk.
    ///
    /// Zone map hints enable the filter operator to skip entire chunks
    /// when predicates can't possibly match based on min/max statistics.
    pub fn set_zone_hints(&mut self, hints: ChunkZoneHints) {
        self.zone_hints = Some(hints);
    }

    /// Returns zone map hints if available.
    ///
    /// Used by the filter operator for chunk-level predicate pruning.
    #[must_use]
    pub fn zone_hints(&self) -> Option<&ChunkZoneHints> {
        self.zone_hints.as_ref()
    }

    /// Clears zone map hints.
    pub fn clear_zone_hints(&mut self) {
        self.zone_hints = None;
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
        self.zone_hints = None;
        self.count = 0;
    }

    /// Flattens the selection by copying only selected rows.
    ///
    /// After this operation, selection is None and count equals the
    /// previously selected row count.
    pub fn flatten(&mut self) {
        let selection = match self.selection.take() {
            Some(sel) => sel,
            None => return,
        };

        let selected_count = selection.len();

        // Create new columns with only selected rows, preserving data types
        let mut new_columns = Vec::with_capacity(self.columns.len());

        for col in &self.columns {
            // Create new vector with same data type as original
            let mut new_col = ValueVector::with_type(col.data_type().clone());
            for idx in selection.iter() {
                if let Some(val) = col.get(idx) {
                    new_col.push(val);
                }
            }
            new_columns.push(new_col);
        }

        self.columns = new_columns;
        self.count = selected_count;
        self.capacity = selected_count;
    }

    /// Returns an iterator over selected row indices.
    pub fn selected_indices(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match &self.selection {
            Some(sel) => Box::new(sel.iter()),
            None => Box::new(0..self.count),
        }
    }

    /// Concatenates multiple chunks into a single chunk.
    ///
    /// All chunks must have the same schema (same number and types of columns).
    pub fn concat(chunks: &[DataChunk]) -> DataChunk {
        if chunks.is_empty() {
            return DataChunk::empty();
        }

        if chunks.len() == 1 {
            // Clone the single chunk
            return DataChunk {
                columns: chunks[0].columns.clone(),
                selection: chunks[0].selection.clone(),
                count: chunks[0].count,
                capacity: chunks[0].capacity,
                zone_hints: chunks[0].zone_hints.clone(),
            };
        }

        let num_columns = chunks[0].column_count();
        if num_columns == 0 {
            return DataChunk::empty();
        }

        let total_rows: usize = chunks.iter().map(|c| c.row_count()).sum();

        // Concatenate each column
        let mut result_columns = Vec::with_capacity(num_columns);

        for col_idx in 0..num_columns {
            let mut concat_vector = ValueVector::new();

            for chunk in chunks {
                if let Some(col) = chunk.column(col_idx) {
                    // Append all values from this column
                    for i in chunk.selected_indices() {
                        if let Some(val) = col.get(i) {
                            concat_vector.push(val);
                        }
                    }
                }
            }

            result_columns.push(concat_vector);
        }

        DataChunk {
            columns: result_columns,
            selection: None,
            count: total_rows,
            capacity: total_rows,
            zone_hints: None,
        }
    }

    /// Applies a filter predicate and returns a new chunk with selected rows.
    pub fn filter(&self, predicate: &SelectionVector) -> DataChunk {
        // Combine existing selection with predicate
        let selected: Vec<usize> = predicate
            .iter()
            .filter(|&idx| self.selection.as_ref().map_or(true, |s| s.contains(idx)))
            .collect();

        let mut result_columns = Vec::with_capacity(self.columns.len());

        for col in &self.columns {
            let mut new_col = ValueVector::new();
            for &idx in &selected {
                if let Some(val) = col.get(idx) {
                    new_col.push(val);
                }
            }
            result_columns.push(new_col);
        }

        DataChunk {
            columns: result_columns,
            selection: None,
            count: selected.len(),
            capacity: selected.len(),
            zone_hints: None,
        }
    }

    /// Returns a slice of this chunk.
    ///
    /// Returns a new DataChunk containing rows [offset, offset + count).
    #[must_use]
    pub fn slice(&self, offset: usize, count: usize) -> DataChunk {
        if offset >= self.len() || count == 0 {
            return DataChunk::empty();
        }

        let actual_count = count.min(self.len() - offset);
        let mut result_columns = Vec::with_capacity(self.columns.len());

        for col in &self.columns {
            let mut new_col = ValueVector::new();
            for i in offset..(offset + actual_count) {
                let actual_idx = if let Some(sel) = &self.selection {
                    sel.get(i).unwrap_or(i)
                } else {
                    i
                };
                if let Some(val) = col.get(actual_idx) {
                    new_col.push(val);
                }
            }
            result_columns.push(new_col);
        }

        DataChunk {
            columns: result_columns,
            selection: None,
            count: actual_count,
            capacity: actual_count,
            zone_hints: None,
        }
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }
}

impl Clone for DataChunk {
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.clone(),
            selection: self.selection.clone(),
            count: self.count,
            capacity: self.capacity,
            zone_hints: self.zone_hints.clone(),
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
    pub fn with_schema(column_types: &[LogicalType]) -> Self {
        Self {
            chunk: DataChunk::with_schema(column_types),
        }
    }

    /// Creates a new builder with the given schema and capacity.
    #[must_use]
    pub fn with_capacity(column_types: &[LogicalType], capacity: usize) -> Self {
        Self {
            chunk: DataChunk::with_capacity(column_types, capacity),
        }
    }

    /// Alias for with_schema for backward compatibility.
    #[must_use]
    pub fn new(column_types: &[LogicalType]) -> Self {
        Self::with_schema(column_types)
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
        let chunk = DataChunk::with_schema(&schema);

        assert_eq!(chunk.column_count(), 2);
        assert_eq!(chunk.row_count(), 0);
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_chunk_builder() {
        let schema = [LogicalType::Int64, LogicalType::String];
        let mut builder = DataChunkBuilder::with_schema(&schema);

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
        let mut builder = DataChunkBuilder::with_schema(&schema);

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
        let mut builder = DataChunkBuilder::with_schema(&schema);

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
        let mut chunk = DataChunk::with_schema(&schema);
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

    #[test]
    fn test_chunk_flatten() {
        let schema = [LogicalType::Int64, LogicalType::String];
        let mut builder = DataChunkBuilder::with_schema(&schema);

        // Add rows: (0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")
        let letters = ["a", "b", "c", "d", "e"];
        for i in 0..5 {
            builder.column_mut(0).unwrap().push_int64(i);
            builder
                .column_mut(1)
                .unwrap()
                .push_string(letters[i as usize]);
            builder.advance_row();
        }

        let mut chunk = builder.finish();

        // Select only odd rows: (1, "b"), (3, "d")
        let selection = SelectionVector::from_predicate(5, |i| i % 2 == 1);
        chunk.set_selection(selection);

        assert_eq!(chunk.row_count(), 2);
        assert_eq!(chunk.total_row_count(), 5);

        // Flatten should copy selected rows
        chunk.flatten();

        // After flatten, total_row_count should equal row_count
        assert_eq!(chunk.row_count(), 2);
        assert_eq!(chunk.total_row_count(), 2);
        assert!(chunk.selection().is_none());

        // Verify the data is correct
        assert_eq!(chunk.column(0).unwrap().get_int64(0), Some(1));
        assert_eq!(chunk.column(0).unwrap().get_int64(1), Some(3));
        assert_eq!(chunk.column(1).unwrap().get_string(0), Some("b"));
        assert_eq!(chunk.column(1).unwrap().get_string(1), Some("d"));
    }

    #[test]
    fn test_chunk_flatten_no_selection() {
        let schema = [LogicalType::Int64];
        let mut builder = DataChunkBuilder::with_schema(&schema);

        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();

        let mut chunk = builder.finish();
        let original_count = chunk.row_count();

        // Flatten with no selection should be a no-op
        chunk.flatten();

        assert_eq!(chunk.row_count(), original_count);
        assert_eq!(chunk.column(0).unwrap().get_int64(0), Some(42));
    }
}
