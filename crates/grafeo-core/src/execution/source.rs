//! Source implementations for push-based execution.
//!
//! Sources produce chunks of data that flow through push-based pipelines.

use super::chunk::DataChunk;
use super::operators::{Operator, OperatorError};
use super::pipeline::Source;
use super::vector::ValueVector;
use grafeo_common::types::{NodeId, Value};

/// Adapts a pull-based operator to work as a Source.
///
/// This allows gradual migration from pull to push model by wrapping
/// existing operators as sources for push pipelines.
pub struct OperatorSource {
    operator: Box<dyn Operator>,
}

impl OperatorSource {
    /// Create a new operator source.
    pub fn new(operator: Box<dyn Operator>) -> Self {
        Self { operator }
    }
}

impl Source for OperatorSource {
    fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        // Pull-based operators produce their own chunk sizes,
        // so we ignore the requested chunk_size
        self.operator.next()
    }

    fn reset(&mut self) {
        self.operator.reset();
    }

    fn name(&self) -> &'static str {
        "OperatorSource"
    }
}

/// Source that produces chunks from a vector of values.
///
/// Useful for testing and for materializing intermediate results.
pub struct VectorSource {
    values: Vec<Vec<Value>>,
    position: usize,
    num_columns: usize,
}

impl VectorSource {
    /// Create a new vector source from column data.
    pub fn new(columns: Vec<Vec<Value>>) -> Self {
        let num_columns = columns.len();
        Self {
            values: columns,
            position: 0,
            num_columns,
        }
    }

    /// Create a single-column source.
    pub fn single_column(values: Vec<Value>) -> Self {
        Self::new(vec![values])
    }

    /// Create from node IDs.
    pub fn from_node_ids(ids: Vec<NodeId>) -> Self {
        let values: Vec<Value> = ids
            .into_iter()
            .map(|id| Value::Int64(id.0 as i64))
            .collect();
        Self::single_column(values)
    }
}

impl Source for VectorSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.num_columns == 0 || self.values[0].is_empty() {
            return Ok(None);
        }

        let total_rows = self.values[0].len();
        if self.position >= total_rows {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(total_rows);
        let mut columns = Vec::with_capacity(self.num_columns);

        for col_values in &self.values {
            let slice = &col_values[self.position..end];
            let vector = ValueVector::from_values(slice);
            columns.push(vector);
        }

        self.position = end;
        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "VectorSource"
    }
}

/// Source that produces a single empty chunk (for testing).
pub struct EmptySource;

impl EmptySource {
    /// Create a new empty source.
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmptySource {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for EmptySource {
    fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        Ok(None)
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "EmptySource"
    }
}

/// Source that produces chunks from a pre-built collection.
///
/// Takes ownership of DataChunks and produces them one at a time.
pub struct ChunkSource {
    chunks: Vec<DataChunk>,
    position: usize,
}

impl ChunkSource {
    /// Create a new chunk source.
    pub fn new(chunks: Vec<DataChunk>) -> Self {
        Self {
            chunks,
            position: 0,
        }
    }

    /// Create from a single chunk.
    pub fn single(chunk: DataChunk) -> Self {
        Self::new(vec![chunk])
    }
}

impl Source for ChunkSource {
    fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.position >= self.chunks.len() {
            return Ok(None);
        }

        let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
        self.position += 1;
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        // Cannot reset since chunks were moved out
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "ChunkSource"
    }
}

/// Source that generates values using a closure.
///
/// Useful for generating test data or for lazy evaluation.
pub struct GeneratorSource<F>
where
    F: FnMut(usize) -> Option<Vec<Value>> + Send + Sync,
{
    generator: F,
    row_index: usize,
    exhausted: bool,
}

impl<F> GeneratorSource<F>
where
    F: FnMut(usize) -> Option<Vec<Value>> + Send + Sync,
{
    /// Create a new generator source.
    pub fn new(generator: F) -> Self {
        Self {
            generator,
            row_index: 0,
            exhausted: false,
        }
    }
}

impl<F> Source for GeneratorSource<F>
where
    F: FnMut(usize) -> Option<Vec<Value>> + Send + Sync,
{
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.exhausted {
            return Ok(None);
        }

        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(chunk_size);

        for _ in 0..chunk_size {
            if let Some(row) = (self.generator)(self.row_index) {
                rows.push(row);
                self.row_index += 1;
            } else {
                self.exhausted = true;
                break;
            }
        }

        if rows.is_empty() {
            return Ok(None);
        }

        // Transpose rows into columns
        let num_columns = rows[0].len();
        let mut columns: Vec<ValueVector> = (0..num_columns).map(|_| ValueVector::new()).collect();

        for row in rows {
            for (col_idx, val) in row.into_iter().enumerate() {
                if col_idx < columns.len() {
                    columns[col_idx].push(val);
                }
            }
        }

        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.row_index = 0;
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "GeneratorSource"
    }
}

/// Source that scans RDF triples matching a pattern.
///
/// Produces chunks with columns for subject, predicate, and object.
#[cfg(feature = "rdf")]
pub struct TripleScanSource {
    /// The triples to scan (materialized for simplicity).
    triples: Vec<(Value, Value, Value)>,
    /// Current position in the triples.
    position: usize,
    /// Variable names for output columns.
    output_vars: Vec<String>,
}

#[cfg(feature = "rdf")]
impl TripleScanSource {
    /// Create a new triple scan source.
    ///
    /// # Arguments
    /// * `triples` - The triples to scan (subject, predicate, object as Values)
    /// * `output_vars` - Names of variables to bind (typically ["s", "p", "o"] or a subset)
    pub fn new(triples: Vec<(Value, Value, Value)>, output_vars: Vec<String>) -> Self {
        Self {
            triples,
            position: 0,
            output_vars,
        }
    }

    /// Create from an RDF store query result.
    pub fn from_triples<I>(iter: I, output_vars: Vec<String>) -> Self
    where
        I: IntoIterator<Item = (Value, Value, Value)>,
    {
        Self::new(iter.into_iter().collect(), output_vars)
    }

    /// Returns the number of remaining triples.
    pub fn remaining(&self) -> usize {
        self.triples.len().saturating_sub(self.position)
    }
}

#[cfg(feature = "rdf")]
impl Source for TripleScanSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.position >= self.triples.len() {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(self.triples.len());
        let slice = &self.triples[self.position..end];

        // Create columns for subject, predicate, object
        let mut subjects = Vec::with_capacity(slice.len());
        let mut predicates = Vec::with_capacity(slice.len());
        let mut objects = Vec::with_capacity(slice.len());

        for (s, p, o) in slice {
            subjects.push(s.clone());
            predicates.push(p.clone());
            objects.push(o.clone());
        }

        let mut columns = Vec::with_capacity(3);

        // Only include columns for requested variables
        for var in &self.output_vars {
            match var.as_str() {
                "s" | "subject" => columns.push(ValueVector::from_values(&subjects)),
                "p" | "predicate" => columns.push(ValueVector::from_values(&predicates)),
                "o" | "object" => columns.push(ValueVector::from_values(&objects)),
                _ => {
                    // For other variable names, we need to determine which position
                    // they refer to based on the query pattern
                    // For now, include all three columns if unknown
                    if columns.is_empty() {
                        columns.push(ValueVector::from_values(&subjects));
                        columns.push(ValueVector::from_values(&predicates));
                        columns.push(ValueVector::from_values(&objects));
                    }
                }
            }
        }

        // If no columns were added, include all
        if columns.is_empty() {
            columns.push(ValueVector::from_values(&subjects));
            columns.push(ValueVector::from_values(&predicates));
            columns.push(ValueVector::from_values(&objects));
        }

        self.position = end;
        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "TripleScanSource"
    }
}

/// Push operator that performs a hash join on RDF triple patterns.
///
/// Used when joining two triple patterns that share a variable.
/// Note: This is a stub for future implementation.
#[cfg(feature = "rdf")]
#[allow(dead_code)]
pub struct TripleJoinOperator {
    /// Name of the join variable.
    join_var: String,
    /// Left input column index for join key.
    left_key_col: usize,
    /// Right input column index for join key.
    right_key_col: usize,
    /// Buffered right side for hash join.
    right_buffer: Vec<DataChunk>,
    /// Whether we've seen the right side completely.
    right_complete: bool,
}

#[cfg(feature = "rdf")]
impl TripleJoinOperator {
    /// Create a new triple join operator.
    pub fn new(join_var: String, left_key_col: usize, right_key_col: usize) -> Self {
        Self {
            join_var,
            left_key_col,
            right_key_col,
            right_buffer: Vec::new(),
            right_complete: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_source_single_chunk() {
        let values = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        let mut source = VectorSource::single_column(values);

        let chunk = source.next_chunk(10).unwrap().unwrap();
        assert_eq!(chunk.len(), 3);

        let next = source.next_chunk(10).unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_vector_source_chunked() {
        let values: Vec<Value> = (0..10).map(Value::Int64).collect();
        let mut source = VectorSource::single_column(values);

        let chunk1 = source.next_chunk(3).unwrap().unwrap();
        assert_eq!(chunk1.len(), 3);

        let chunk2 = source.next_chunk(3).unwrap().unwrap();
        assert_eq!(chunk2.len(), 3);

        let chunk3 = source.next_chunk(3).unwrap().unwrap();
        assert_eq!(chunk3.len(), 3);

        let chunk4 = source.next_chunk(3).unwrap().unwrap();
        assert_eq!(chunk4.len(), 1); // Remaining row

        let none = source.next_chunk(3).unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn test_vector_source_reset() {
        let values = vec![Value::Int64(1), Value::Int64(2)];
        let mut source = VectorSource::single_column(values);

        let _ = source.next_chunk(10).unwrap();
        assert!(source.next_chunk(10).unwrap().is_none());

        source.reset();
        let chunk = source.next_chunk(10).unwrap().unwrap();
        assert_eq!(chunk.len(), 2);
    }

    #[test]
    fn test_empty_source() {
        let mut source = EmptySource::new();
        assert!(source.next_chunk(100).unwrap().is_none());
    }

    #[test]
    fn test_chunk_source() {
        let v1 = ValueVector::from_values(&[Value::Int64(1), Value::Int64(2)]);
        let chunk1 = DataChunk::new(vec![v1]);

        let v2 = ValueVector::from_values(&[Value::Int64(3), Value::Int64(4)]);
        let chunk2 = DataChunk::new(vec![v2]);

        let mut source = ChunkSource::new(vec![chunk1, chunk2]);

        let c1 = source.next_chunk(100).unwrap().unwrap();
        assert_eq!(c1.len(), 2);

        let c2 = source.next_chunk(100).unwrap().unwrap();
        assert_eq!(c2.len(), 2);

        assert!(source.next_chunk(100).unwrap().is_none());
    }

    #[test]
    fn test_generator_source() {
        let mut source = GeneratorSource::new(|i| {
            if i < 5 {
                Some(vec![Value::Int64(i as i64)])
            } else {
                None
            }
        });

        let chunk1 = source.next_chunk(2).unwrap().unwrap();
        assert_eq!(chunk1.len(), 2);

        let chunk2 = source.next_chunk(2).unwrap().unwrap();
        assert_eq!(chunk2.len(), 2);

        let chunk3 = source.next_chunk(2).unwrap().unwrap();
        assert_eq!(chunk3.len(), 1);

        assert!(source.next_chunk(2).unwrap().is_none());
    }
}
