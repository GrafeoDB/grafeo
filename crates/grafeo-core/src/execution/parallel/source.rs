//! Parallel source trait for partitionable data sources.
//!
//! Extends the Source trait with capabilities needed for parallel execution:
//! knowing total row count, creating partitions for morsels, etc.

use super::morsel::{Morsel, generate_morsels};
use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::pipeline::Source;
use crate::execution::vector::ValueVector;
use grafeo_common::types::Value;
use std::sync::Arc;

/// Trait for sources that support parallel partitioning.
///
/// Parallel sources can:
/// - Report their total row count (if known)
/// - Be partitioned into independent morsels
/// - Create partition sources for specific morsels
pub trait ParallelSource: Source + Send + Sync {
    /// Returns the total number of rows in this source, if known.
    ///
    /// Returns `None` if the total is unknown (e.g., for streaming sources).
    fn total_rows(&self) -> Option<usize>;

    /// Returns whether this source can be partitioned.
    ///
    /// Some sources (like streaming or network sources) cannot be partitioned.
    fn is_partitionable(&self) -> bool {
        self.total_rows().is_some()
    }

    /// Creates a partition source for the given morsel.
    ///
    /// The returned source produces data only for the row range specified
    /// in the morsel.
    fn create_partition(&self, morsel: &Morsel) -> Box<dyn Source>;

    /// Generates morsels that cover all rows in this source.
    ///
    /// Returns an empty vector if the source has no rows or cannot be partitioned.
    fn generate_morsels(&self, morsel_size: usize, source_id: usize) -> Vec<Morsel> {
        match self.total_rows() {
            Some(total) => generate_morsels(total, morsel_size, source_id),
            None => Vec::new(),
        }
    }

    /// Returns the number of columns in this source.
    fn num_columns(&self) -> usize;
}

/// Parallel source wrapper for vector data.
///
/// Wraps columnar data in a parallel source that can be partitioned.
pub struct ParallelVectorSource {
    /// Column data (shared across partitions).
    columns: Arc<Vec<Vec<Value>>>,
    /// Current read position.
    position: usize,
}

impl ParallelVectorSource {
    /// Creates a new parallel vector source.
    #[must_use]
    pub fn new(columns: Vec<Vec<Value>>) -> Self {
        Self {
            columns: Arc::new(columns),
            position: 0,
        }
    }

    /// Creates a single-column source.
    #[must_use]
    pub fn single_column(values: Vec<Value>) -> Self {
        Self::new(vec![values])
    }
}

impl Source for ParallelVectorSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.columns.is_empty() || self.columns[0].is_empty() {
            return Ok(None);
        }

        let total_rows = self.columns[0].len();
        if self.position >= total_rows {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(total_rows);
        let mut vectors = Vec::with_capacity(self.columns.len());

        for col_values in self.columns.iter() {
            let slice = &col_values[self.position..end];
            vectors.push(ValueVector::from_values(slice));
        }

        self.position = end;
        Ok(Some(DataChunk::new(vectors)))
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "ParallelVectorSource"
    }
}

impl ParallelSource for ParallelVectorSource {
    fn total_rows(&self) -> Option<usize> {
        if self.columns.is_empty() {
            Some(0)
        } else {
            Some(self.columns[0].len())
        }
    }

    fn create_partition(&self, morsel: &Morsel) -> Box<dyn Source> {
        Box::new(PartitionedVectorSource::new(
            Arc::clone(&self.columns),
            morsel.start_row,
            morsel.end_row,
        ))
    }

    fn num_columns(&self) -> usize {
        self.columns.len()
    }
}

/// A partitioned view into a vector source.
///
/// Only produces data for a specific row range.
struct PartitionedVectorSource {
    columns: Arc<Vec<Vec<Value>>>,
    start_row: usize,
    end_row: usize,
    position: usize,
}

impl PartitionedVectorSource {
    fn new(columns: Arc<Vec<Vec<Value>>>, start_row: usize, end_row: usize) -> Self {
        Self {
            columns,
            start_row,
            end_row,
            position: start_row,
        }
    }
}

impl Source for PartitionedVectorSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.columns.is_empty() || self.position >= self.end_row {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(self.end_row);
        let mut vectors = Vec::with_capacity(self.columns.len());

        for col_values in self.columns.iter() {
            let slice = &col_values[self.position..end];
            vectors.push(ValueVector::from_values(slice));
        }

        self.position = end;
        Ok(Some(DataChunk::new(vectors)))
    }

    fn reset(&mut self) {
        self.position = self.start_row;
    }

    fn name(&self) -> &'static str {
        "PartitionedVectorSource"
    }
}

/// Parallel source for pre-built chunks.
///
/// Wraps a collection of DataChunks in a parallel source.
pub struct ParallelChunkSource {
    chunks: Arc<Vec<DataChunk>>,
    /// Row count in each chunk (cached for fast morsel generation).
    #[allow(dead_code)]
    chunk_row_counts: Vec<usize>,
    /// Cumulative row count at each chunk start.
    cumulative_rows: Vec<usize>,
    /// Total row count.
    total_rows: usize,
    /// Current chunk index.
    chunk_index: usize,
    /// Number of columns.
    num_columns: usize,
}

impl ParallelChunkSource {
    /// Creates a new parallel chunk source.
    #[must_use]
    pub fn new(chunks: Vec<DataChunk>) -> Self {
        let chunk_row_counts: Vec<usize> = chunks.iter().map(DataChunk::len).collect();

        let mut cumulative_rows = Vec::with_capacity(chunks.len() + 1);
        let mut sum = 0;
        cumulative_rows.push(0);
        for &count in &chunk_row_counts {
            sum += count;
            cumulative_rows.push(sum);
        }

        let num_columns = chunks.first().map(|c| c.num_columns()).unwrap_or(0);

        Self {
            chunks: Arc::new(chunks),
            chunk_row_counts,
            cumulative_rows,
            total_rows: sum,
            chunk_index: 0,
            num_columns,
        }
    }
}

impl Source for ParallelChunkSource {
    fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.chunk_index >= self.chunks.len() {
            return Ok(None);
        }

        let chunk = self.chunks[self.chunk_index].clone();
        self.chunk_index += 1;
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.chunk_index = 0;
    }

    fn name(&self) -> &'static str {
        "ParallelChunkSource"
    }
}

impl ParallelSource for ParallelChunkSource {
    fn total_rows(&self) -> Option<usize> {
        Some(self.total_rows)
    }

    fn create_partition(&self, morsel: &Morsel) -> Box<dyn Source> {
        Box::new(PartitionedChunkSource::new(
            Arc::clone(&self.chunks),
            self.cumulative_rows.clone(),
            morsel.start_row,
            morsel.end_row,
        ))
    }

    fn num_columns(&self) -> usize {
        self.num_columns
    }
}

/// A partitioned view into a chunk source.
struct PartitionedChunkSource {
    chunks: Arc<Vec<DataChunk>>,
    cumulative_rows: Vec<usize>,
    start_row: usize,
    end_row: usize,
    current_row: usize,
}

impl PartitionedChunkSource {
    fn new(
        chunks: Arc<Vec<DataChunk>>,
        cumulative_rows: Vec<usize>,
        start_row: usize,
        end_row: usize,
    ) -> Self {
        Self {
            chunks,
            cumulative_rows,
            start_row,
            end_row,
            current_row: start_row,
        }
    }

    /// Finds the chunk index containing the given row.
    fn find_chunk_index(&self, row: usize) -> Option<usize> {
        // Binary search for the chunk containing this row
        match self
            .cumulative_rows
            .binary_search_by(|&cumul| cumul.cmp(&row))
        {
            Ok(idx) => Some(idx.min(self.chunks.len().saturating_sub(1))),
            Err(idx) => {
                if idx == 0 {
                    Some(0)
                } else {
                    Some((idx - 1).min(self.chunks.len().saturating_sub(1)))
                }
            }
        }
    }
}

impl Source for PartitionedChunkSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.current_row >= self.end_row || self.chunks.is_empty() {
            return Ok(None);
        }

        // Find the chunk containing current_row
        let chunk_idx = match self.find_chunk_index(self.current_row) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        if chunk_idx >= self.chunks.len() {
            return Ok(None);
        }

        let chunk_start = self.cumulative_rows[chunk_idx];
        let chunk = &self.chunks[chunk_idx];
        let offset_in_chunk = self.current_row - chunk_start;

        // Calculate how many rows to extract
        let rows_in_chunk = chunk.len().saturating_sub(offset_in_chunk);
        let rows_to_end = self.end_row.saturating_sub(self.current_row);
        let rows_to_extract = rows_in_chunk.min(rows_to_end).min(chunk_size);

        if rows_to_extract == 0 {
            return Ok(None);
        }

        // Extract slice from chunk
        let sliced = chunk.slice(offset_in_chunk, rows_to_extract);
        self.current_row += rows_to_extract;

        Ok(Some(sliced))
    }

    fn reset(&mut self) {
        self.current_row = self.start_row;
    }

    fn name(&self) -> &'static str {
        "PartitionedChunkSource"
    }
}

/// Generates a range source for parallel execution testing.
///
/// Produces integers from 0 to n-1 in a single column.
pub struct RangeSource {
    total: usize,
    position: usize,
}

impl RangeSource {
    /// Creates a new range source.
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self { total, position: 0 }
    }
}

impl Source for RangeSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.position >= self.total {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(self.total);
        let values: Vec<Value> = (self.position..end)
            .map(|i| Value::Int64(i as i64))
            .collect();

        self.position = end;
        Ok(Some(DataChunk::new(vec![ValueVector::from_values(
            &values,
        )])))
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "RangeSource"
    }
}

impl ParallelSource for RangeSource {
    fn total_rows(&self) -> Option<usize> {
        Some(self.total)
    }

    fn create_partition(&self, morsel: &Morsel) -> Box<dyn Source> {
        Box::new(RangePartition::new(morsel.start_row, morsel.end_row))
    }

    fn num_columns(&self) -> usize {
        1
    }
}

/// A partition of a range source.
struct RangePartition {
    start: usize,
    end: usize,
    position: usize,
}

impl RangePartition {
    fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            position: start,
        }
    }
}

impl Source for RangePartition {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.position >= self.end {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(self.end);
        let values: Vec<Value> = (self.position..end)
            .map(|i| Value::Int64(i as i64))
            .collect();

        self.position = end;
        Ok(Some(DataChunk::new(vec![ValueVector::from_values(
            &values,
        )])))
    }

    fn reset(&mut self) {
        self.position = self.start;
    }

    fn name(&self) -> &'static str {
        "RangePartition"
    }
}

/// Parallel source for RDF triple scanning.
///
/// Wraps triple data in a parallel source that can be partitioned for
/// morsel-driven execution of SPARQL queries.
#[cfg(feature = "rdf")]
pub struct ParallelTripleScanSource {
    /// Triple data: (subject, predicate, object) tuples.
    triples: Arc<Vec<(Value, Value, Value)>>,
    /// Current read position.
    position: usize,
    /// Variable names for output columns (e.g., ["s", "p", "o"]).
    output_vars: Vec<String>,
}

#[cfg(feature = "rdf")]
impl ParallelTripleScanSource {
    /// Creates a new parallel triple scan source.
    #[must_use]
    pub fn new(triples: Vec<(Value, Value, Value)>, output_vars: Vec<String>) -> Self {
        Self {
            triples: Arc::new(triples),
            position: 0,
            output_vars,
        }
    }

    /// Creates from an iterator of triples.
    pub fn from_triples<I>(iter: I, output_vars: Vec<String>) -> Self
    where
        I: IntoIterator<Item = (Value, Value, Value)>,
    {
        Self::new(iter.into_iter().collect(), output_vars)
    }
}

#[cfg(feature = "rdf")]
impl Source for ParallelTripleScanSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.position >= self.triples.len() {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(self.triples.len());
        let slice = &self.triples[self.position..end];

        let mut subjects = Vec::with_capacity(slice.len());
        let mut predicates = Vec::with_capacity(slice.len());
        let mut objects = Vec::with_capacity(slice.len());

        for (s, p, o) in slice {
            subjects.push(s.clone());
            predicates.push(p.clone());
            objects.push(o.clone());
        }

        let columns = vec![
            ValueVector::from_values(&subjects),
            ValueVector::from_values(&predicates),
            ValueVector::from_values(&objects),
        ];

        self.position = end;
        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "ParallelTripleScanSource"
    }
}

#[cfg(feature = "rdf")]
impl ParallelSource for ParallelTripleScanSource {
    fn total_rows(&self) -> Option<usize> {
        Some(self.triples.len())
    }

    fn create_partition(&self, morsel: &Morsel) -> Box<dyn Source> {
        Box::new(PartitionedTripleScanSource::new(
            Arc::clone(&self.triples),
            self.output_vars.clone(),
            morsel.start_row,
            morsel.end_row,
        ))
    }

    fn num_columns(&self) -> usize {
        3 // subject, predicate, object
    }
}

/// A partitioned view into a triple scan source.
#[cfg(feature = "rdf")]
struct PartitionedTripleScanSource {
    triples: Arc<Vec<(Value, Value, Value)>>,
    #[allow(dead_code)]
    output_vars: Vec<String>,
    start_row: usize,
    end_row: usize,
    position: usize,
}

#[cfg(feature = "rdf")]
impl PartitionedTripleScanSource {
    fn new(
        triples: Arc<Vec<(Value, Value, Value)>>,
        output_vars: Vec<String>,
        start_row: usize,
        end_row: usize,
    ) -> Self {
        Self {
            triples,
            output_vars,
            start_row,
            end_row,
            position: start_row,
        }
    }
}

#[cfg(feature = "rdf")]
impl Source for PartitionedTripleScanSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.position >= self.end_row || self.position >= self.triples.len() {
            return Ok(None);
        }

        let end = (self.position + chunk_size)
            .min(self.end_row)
            .min(self.triples.len());
        let slice = &self.triples[self.position..end];

        let mut subjects = Vec::with_capacity(slice.len());
        let mut predicates = Vec::with_capacity(slice.len());
        let mut objects = Vec::with_capacity(slice.len());

        for (s, p, o) in slice {
            subjects.push(s.clone());
            predicates.push(p.clone());
            objects.push(o.clone());
        }

        let columns = vec![
            ValueVector::from_values(&subjects),
            ValueVector::from_values(&predicates),
            ValueVector::from_values(&objects),
        ];

        self.position = end;
        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.position = self.start_row;
    }

    fn name(&self) -> &'static str {
        "PartitionedTripleScanSource"
    }
}

// ---------------------------------------------------------------------------
// Parallel Node Scan Source (LPG)
// ---------------------------------------------------------------------------

use crate::graph::lpg::LpgStore;
use grafeo_common::types::NodeId;

/// Parallel source for scanning nodes from the LPG store.
///
/// Enables morsel-driven parallel execution of node scans by label.
/// Each partition independently scans a range of node IDs, enabling
/// linear scaling on multi-core systems for large datasets.
///
/// # Example
///
/// ```ignore
/// use grafeo_core::execution::parallel::{ParallelNodeScanSource, ParallelPipeline};
/// use std::sync::Arc;
///
/// let store = Arc::new(LpgStore::new());
/// // ... populate store ...
///
/// // Scan all Person nodes in parallel
/// let source = ParallelNodeScanSource::with_label(store, "Person");
/// let morsels = source.generate_morsels(4096, 0);
/// ```
pub struct ParallelNodeScanSource {
    /// The store to scan from.
    store: Arc<LpgStore>,
    /// Cached node IDs for the scan.
    node_ids: Arc<Vec<NodeId>>,
    /// Current read position.
    position: usize,
}

impl ParallelNodeScanSource {
    /// Creates a parallel source for all nodes in the store.
    #[must_use]
    pub fn new(store: Arc<LpgStore>) -> Self {
        let node_ids = Arc::new(store.node_ids());
        Self {
            store,
            node_ids,
            position: 0,
        }
    }

    /// Creates a parallel source for nodes with a specific label.
    #[must_use]
    pub fn with_label(store: Arc<LpgStore>, label: &str) -> Self {
        let node_ids = Arc::new(store.nodes_by_label(label));
        Self {
            store,
            node_ids,
            position: 0,
        }
    }

    /// Creates from pre-computed node IDs.
    ///
    /// Useful when node IDs are already available from a previous operation.
    #[must_use]
    pub fn from_node_ids(store: Arc<LpgStore>, node_ids: Vec<NodeId>) -> Self {
        Self {
            store,
            node_ids: Arc::new(node_ids),
            position: 0,
        }
    }

    /// Returns the underlying store reference.
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        &self.store
    }
}

impl Source for ParallelNodeScanSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.position >= self.node_ids.len() {
            return Ok(None);
        }

        let end = (self.position + chunk_size).min(self.node_ids.len());
        let slice = &self.node_ids[self.position..end];

        // Create a NodeId vector
        let mut vector = ValueVector::with_type(grafeo_common::types::LogicalType::Node);
        for &id in slice {
            vector.push_node_id(id);
        }

        self.position = end;
        Ok(Some(DataChunk::new(vec![vector])))
    }

    fn reset(&mut self) {
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "ParallelNodeScanSource"
    }
}

impl ParallelSource for ParallelNodeScanSource {
    fn total_rows(&self) -> Option<usize> {
        Some(self.node_ids.len())
    }

    fn create_partition(&self, morsel: &Morsel) -> Box<dyn Source> {
        Box::new(PartitionedNodeScanSource::new(
            Arc::clone(&self.node_ids),
            morsel.start_row,
            morsel.end_row,
        ))
    }

    fn num_columns(&self) -> usize {
        1 // Node ID column
    }
}

/// A partitioned view into a node scan source.
struct PartitionedNodeScanSource {
    node_ids: Arc<Vec<NodeId>>,
    start_row: usize,
    end_row: usize,
    position: usize,
}

impl PartitionedNodeScanSource {
    fn new(node_ids: Arc<Vec<NodeId>>, start_row: usize, end_row: usize) -> Self {
        Self {
            node_ids,
            start_row,
            end_row,
            position: start_row,
        }
    }
}

impl Source for PartitionedNodeScanSource {
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
        if self.position >= self.end_row || self.position >= self.node_ids.len() {
            return Ok(None);
        }

        let end = (self.position + chunk_size)
            .min(self.end_row)
            .min(self.node_ids.len());
        let slice = &self.node_ids[self.position..end];

        // Create a NodeId vector
        let mut vector = ValueVector::with_type(grafeo_common::types::LogicalType::Node);
        for &id in slice {
            vector.push_node_id(id);
        }

        self.position = end;
        Ok(Some(DataChunk::new(vec![vector])))
    }

    fn reset(&mut self) {
        self.position = self.start_row;
    }

    fn name(&self) -> &'static str {
        "PartitionedNodeScanSource"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_vector_source() {
        let values: Vec<Value> = (0..100).map(Value::Int64).collect();
        let source = ParallelVectorSource::single_column(values);

        assert_eq!(source.total_rows(), Some(100));
        assert!(source.is_partitionable());
        assert_eq!(source.num_columns(), 1);

        let morsels = source.generate_morsels(30, 0);
        assert_eq!(morsels.len(), 4); // 100 / 30 = 3 full + 1 partial
    }

    #[test]
    fn test_parallel_vector_source_partition() {
        let values: Vec<Value> = (0..100).map(Value::Int64).collect();
        let source = ParallelVectorSource::single_column(values);

        let morsel = Morsel::new(0, 0, 20, 50);
        let mut partition = source.create_partition(&morsel);

        // Should produce 30 rows total
        let mut total = 0;
        while let Ok(Some(chunk)) = partition.next_chunk(10) {
            total += chunk.len();
        }
        assert_eq!(total, 30);
    }

    #[test]
    fn test_range_source() {
        let source = RangeSource::new(100);

        assert_eq!(source.total_rows(), Some(100));
        assert!(source.is_partitionable());

        let morsels = source.generate_morsels(25, 0);
        assert_eq!(morsels.len(), 4);
    }

    #[test]
    fn test_range_source_partition() {
        let source = RangeSource::new(100);

        let morsel = Morsel::new(0, 0, 10, 30);
        let mut partition = source.create_partition(&morsel);

        let chunk = partition.next_chunk(100).unwrap().unwrap();
        assert_eq!(chunk.len(), 20);

        // Verify values are in range [10, 30)
        let col = chunk.column(0).unwrap();
        assert_eq!(col.get(0), Some(Value::Int64(10)));
        assert_eq!(col.get(19), Some(Value::Int64(29)));
    }

    #[test]
    fn test_parallel_chunk_source() {
        let chunks: Vec<DataChunk> = (0..5)
            .map(|i| {
                let values: Vec<Value> = (i * 10..(i + 1) * 10).map(Value::Int64).collect();
                DataChunk::new(vec![ValueVector::from_values(&values)])
            })
            .collect();

        let source = ParallelChunkSource::new(chunks);
        assert_eq!(source.total_rows(), Some(50));
        assert_eq!(source.num_columns(), 1);
    }

    #[test]
    fn test_parallel_chunk_source_partition() {
        let chunks: Vec<DataChunk> = (0..5)
            .map(|i| {
                let values: Vec<Value> = (i * 10..(i + 1) * 10).map(Value::Int64).collect();
                DataChunk::new(vec![ValueVector::from_values(&values)])
            })
            .collect();

        let source = ParallelChunkSource::new(chunks);

        // Partition spanning parts of chunks 1 and 2 (rows 15-35)
        let morsel = Morsel::new(0, 0, 15, 35);
        let mut partition = source.create_partition(&morsel);

        let mut total = 0;
        let mut first_value: Option<i64> = None;
        let mut last_value: Option<i64> = None;

        while let Ok(Some(chunk)) = partition.next_chunk(10) {
            if first_value.is_none() {
                if let Some(Value::Int64(v)) = chunk.column(0).and_then(|c| c.get(0)) {
                    first_value = Some(v);
                }
            }
            if let Some(Value::Int64(v)) = chunk
                .column(0)
                .and_then(|c| c.get(chunk.len().saturating_sub(1)))
            {
                last_value = Some(v);
            }
            total += chunk.len();
        }

        assert_eq!(total, 20);
        assert_eq!(first_value, Some(15));
        assert_eq!(last_value, Some(34));
    }

    #[test]
    fn test_partitioned_source_reset() {
        let source = RangeSource::new(100);
        let morsel = Morsel::new(0, 0, 0, 50);
        let mut partition = source.create_partition(&morsel);

        // Exhaust partition
        while partition.next_chunk(100).unwrap().is_some() {}

        // Reset and read again
        partition.reset();
        let chunk = partition.next_chunk(100).unwrap().unwrap();
        assert_eq!(chunk.len(), 50);
    }

    #[cfg(feature = "rdf")]
    #[test]
    fn test_parallel_triple_scan_source() {
        let triples = vec![
            (
                Value::String("s1".into()),
                Value::String("p1".into()),
                Value::String("o1".into()),
            ),
            (
                Value::String("s2".into()),
                Value::String("p2".into()),
                Value::String("o2".into()),
            ),
            (
                Value::String("s3".into()),
                Value::String("p3".into()),
                Value::String("o3".into()),
            ),
        ];
        let source =
            ParallelTripleScanSource::new(triples, vec!["s".into(), "p".into(), "o".into()]);

        assert_eq!(source.total_rows(), Some(3));
        assert!(source.is_partitionable());
        assert_eq!(source.num_columns(), 3);
    }

    #[cfg(feature = "rdf")]
    #[test]
    fn test_parallel_triple_scan_partition() {
        let triples: Vec<(Value, Value, Value)> = (0..100)
            .map(|i| {
                (
                    Value::String(format!("s{}", i).into()),
                    Value::String(format!("p{}", i).into()),
                    Value::String(format!("o{}", i).into()),
                )
            })
            .collect();
        let source =
            ParallelTripleScanSource::new(triples, vec!["s".into(), "p".into(), "o".into()]);

        let morsel = Morsel::new(0, 0, 20, 50);
        let mut partition = source.create_partition(&morsel);

        let mut total = 0;
        while let Ok(Some(chunk)) = partition.next_chunk(10) {
            total += chunk.len();
        }
        assert_eq!(total, 30);
    }

    #[test]
    fn test_parallel_node_scan_source() {
        let store = Arc::new(LpgStore::new());

        // Add some nodes with labels
        for i in 0..100 {
            if i % 2 == 0 {
                store.create_node(&["Person", "Employee"]);
            } else {
                store.create_node(&["Person"]);
            }
        }

        // Test scan all nodes
        let source = ParallelNodeScanSource::new(Arc::clone(&store));
        assert_eq!(source.total_rows(), Some(100));
        assert!(source.is_partitionable());
        assert_eq!(source.num_columns(), 1);

        // Test scan by label
        let source_person = ParallelNodeScanSource::with_label(Arc::clone(&store), "Person");
        assert_eq!(source_person.total_rows(), Some(100));

        let source_employee = ParallelNodeScanSource::with_label(Arc::clone(&store), "Employee");
        assert_eq!(source_employee.total_rows(), Some(50));
    }

    #[test]
    fn test_parallel_node_scan_partition() {
        let store = Arc::new(LpgStore::new());

        // Add 100 nodes
        for _ in 0..100 {
            store.create_node(&[]);
        }

        let source = ParallelNodeScanSource::new(Arc::clone(&store));

        // Create partition for rows 20-50
        let morsel = Morsel::new(0, 0, 20, 50);
        let mut partition = source.create_partition(&morsel);

        // Should produce 30 rows total
        let mut total = 0;
        while let Ok(Some(chunk)) = partition.next_chunk(10) {
            total += chunk.len();
        }
        assert_eq!(total, 30);
    }

    #[test]
    fn test_parallel_node_scan_morsels() {
        let store = Arc::new(LpgStore::new());

        // Add 1000 nodes
        for _ in 0..1000 {
            store.create_node(&[]);
        }

        let source = ParallelNodeScanSource::new(Arc::clone(&store));

        // Generate morsels with size 256
        let morsels = source.generate_morsels(256, 0);
        assert_eq!(morsels.len(), 4); // 1000 / 256 = 3 full + 1 partial

        // Verify morsels cover all rows
        let mut total_rows = 0;
        for morsel in &morsels {
            total_rows += morsel.end_row - morsel.start_row;
        }
        assert_eq!(total_rows, 1000);
    }
}
