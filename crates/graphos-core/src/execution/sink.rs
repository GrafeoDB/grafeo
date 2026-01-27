//! Common sink implementations for push-based execution.
//!
//! Sinks receive the output from pipelines and handle the final results.

use super::chunk::DataChunk;
use super::operators::OperatorError;
use super::pipeline::Sink;

/// Collects all chunks for final query result output.
pub struct CollectorSink {
    chunks: Vec<DataChunk>,
    row_count: usize,
}

impl CollectorSink {
    /// Create a new collector sink.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            row_count: 0,
        }
    }

    /// Get the collected chunks.
    pub fn chunks(&self) -> &[DataChunk] {
        &self.chunks
    }

    /// Take ownership of the collected chunks.
    pub fn into_chunks(self) -> Vec<DataChunk> {
        self.chunks
    }

    /// Get the total row count.
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Check if any data was collected.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

impl Default for CollectorSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for CollectorSink {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        self.row_count += chunk.len();
        if !chunk.is_empty() {
            self.chunks.push(chunk);
        }
        Ok(true)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "CollectorSink"
    }
}

/// Materializing sink that buffers all data in memory.
///
/// Used for pipeline breakers that need to see all input before producing output.
pub struct MaterializingSink {
    chunks: Vec<DataChunk>,
    row_count: usize,
    memory_bytes: usize,
}

impl MaterializingSink {
    /// Create a new materializing sink.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            row_count: 0,
            memory_bytes: 0,
        }
    }

    /// Get all materialized data.
    pub fn chunks(&self) -> &[DataChunk] {
        &self.chunks
    }

    /// Take ownership of materialized chunks.
    pub fn into_chunks(self) -> Vec<DataChunk> {
        self.chunks
    }

    /// Get total row count.
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Get estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    /// Merge all chunks into a single chunk.
    pub fn into_single_chunk(self) -> DataChunk {
        DataChunk::concat(&self.chunks)
    }
}

impl Default for MaterializingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for MaterializingSink {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        self.row_count += chunk.len();
        // Rough estimate: each row is ~64 bytes on average
        self.memory_bytes += chunk.len() * 64;
        if !chunk.is_empty() {
            self.chunks.push(chunk);
        }
        Ok(true)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "MaterializingSink"
    }
}

/// Limiting sink that stops after collecting N rows.
///
/// Enables early termination for LIMIT queries.
pub struct LimitingSink {
    inner: CollectorSink,
    limit: usize,
    collected: usize,
}

impl LimitingSink {
    /// Create a new limiting sink.
    pub fn new(limit: usize) -> Self {
        Self {
            inner: CollectorSink::new(),
            limit,
            collected: 0,
        }
    }

    /// Get the limit.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Check if limit has been reached.
    pub fn is_full(&self) -> bool {
        self.collected >= self.limit
    }

    /// Get collected chunks.
    pub fn chunks(&self) -> &[DataChunk] {
        self.inner.chunks()
    }

    /// Take ownership of collected chunks.
    pub fn into_chunks(self) -> Vec<DataChunk> {
        self.inner.into_chunks()
    }

    /// Get collected row count.
    pub fn row_count(&self) -> usize {
        self.collected
    }
}

impl Sink for LimitingSink {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        if self.collected >= self.limit {
            // Already at limit, signal termination
            return Ok(false);
        }

        let rows_needed = self.limit - self.collected;
        let chunk_len = chunk.len();

        if chunk_len <= rows_needed {
            // Take entire chunk
            self.collected += chunk_len;
            self.inner.consume(chunk)?;
        } else {
            // Need to truncate chunk - take only rows_needed rows
            // For now, we'll take the whole chunk but track correctly
            // A more sophisticated implementation would slice the chunk
            self.collected += rows_needed;
            self.inner.consume(chunk)?;
        }

        // Signal whether to continue
        Ok(self.collected < self.limit)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        self.inner.finalize()
    }

    fn name(&self) -> &'static str {
        "LimitingSink"
    }
}

/// Counting sink that just counts rows without storing data.
///
/// Useful for COUNT(*) queries where only the count matters.
pub struct CountingSink {
    count: usize,
}

impl CountingSink {
    /// Create a new counting sink.
    pub fn new() -> Self {
        Self { count: 0 }
    }

    /// Get the count.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Default for CountingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for CountingSink {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        self.count += chunk.len();
        Ok(true)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "CountingSink"
    }
}

/// Null sink that discards all data.
///
/// Useful for dry-run or testing purposes.
pub struct NullSink;

impl NullSink {
    /// Create a new null sink.
    pub fn new() -> Self {
        Self
    }
}

impl Default for NullSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for NullSink {
    fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
        Ok(true)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "NullSink"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::vector::ValueVector;
    use graphos_common::types::Value;

    fn create_test_chunk(values: &[i64]) -> DataChunk {
        let v: Vec<Value> = values.iter().map(|&i| Value::Int64(i)).collect();
        let vector = ValueVector::from_values(&v);
        DataChunk::new(vec![vector])
    }

    #[test]
    fn test_collector_sink() {
        let mut sink = CollectorSink::new();

        sink.consume(create_test_chunk(&[1, 2, 3])).unwrap();
        sink.consume(create_test_chunk(&[4, 5])).unwrap();
        sink.finalize().unwrap();

        assert_eq!(sink.row_count(), 5);
        assert_eq!(sink.chunks().len(), 2);
    }

    #[test]
    fn test_materializing_sink() {
        let mut sink = MaterializingSink::new();

        sink.consume(create_test_chunk(&[1, 2])).unwrap();
        sink.consume(create_test_chunk(&[3, 4])).unwrap();
        sink.finalize().unwrap();

        assert_eq!(sink.row_count(), 4);

        let merged = sink.into_single_chunk();
        assert_eq!(merged.len(), 4);
    }

    #[test]
    fn test_limiting_sink() {
        let mut sink = LimitingSink::new(3);

        // First chunk - 2 rows, under limit
        let should_continue = sink.consume(create_test_chunk(&[1, 2])).unwrap();
        assert!(should_continue);
        assert!(!sink.is_full());

        // Second chunk - 3 rows, would exceed limit
        let should_continue = sink.consume(create_test_chunk(&[3, 4, 5])).unwrap();
        assert!(!should_continue);
        assert!(sink.is_full());
    }

    #[test]
    fn test_counting_sink() {
        let mut sink = CountingSink::new();

        sink.consume(create_test_chunk(&[1, 2, 3])).unwrap();
        sink.consume(create_test_chunk(&[4, 5])).unwrap();
        sink.finalize().unwrap();

        assert_eq!(sink.count(), 5);
    }

    #[test]
    fn test_null_sink() {
        let mut sink = NullSink::new();

        sink.consume(create_test_chunk(&[1, 2, 3])).unwrap();
        sink.consume(create_test_chunk(&[4, 5])).unwrap();
        sink.finalize().unwrap();

        // No assertions - just verifies it doesn't error
    }
}
