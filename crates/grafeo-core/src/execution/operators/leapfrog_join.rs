//! Leapfrog TrieJoin operator for worst-case optimal joins.
//!
//! This operator wraps the `LeapfrogJoin` algorithm from the trie index module
//! to provide efficient multi-way joins for cyclic patterns like triangles.
//!
//! Traditional binary hash joins cascade O(N²) for triangle patterns; leapfrog
//! achieves O(N^1.5) by processing all relations simultaneously.

use grafeo_common::types::{EdgeId, LogicalType, NodeId, Value};

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use crate::execution::DataChunk;
use crate::index::trie::{LeapfrogJoin, TrieIndex};

/// Row identifier for reconstructing output: (input_index, chunk_index, row_index).
type RowId = (usize, usize, usize);

/// A multi-way join intersection result.
struct JoinResult {
    /// The matched key value (intersection point).
    /// Kept for debugging and potential future use in result verification.
    #[allow(dead_code)]
    key: NodeId,
    /// Row identifiers from each input that participated in this match.
    row_ids: Vec<Vec<RowId>>,
}

/// Leapfrog TrieJoin operator for worst-case optimal multi-way joins.
///
/// Uses the leapfrog algorithm to efficiently find intersections across
/// multiple sorted inputs without materializing intermediate Cartesian products.
pub struct LeapfrogJoinOperator {
    /// Input operators (one per relation in the join).
    inputs: Vec<Box<dyn Operator>>,

    /// Column indices for join keys in each input.
    /// Each inner Vec maps to one join variable.
    join_key_indices: Vec<Vec<usize>>,

    /// Output schema (combined columns from all inputs).
    output_schema: Vec<LogicalType>,

    /// Mapping from output column index to (input_idx, column_idx).
    output_column_mapping: Vec<(usize, usize)>,

    // === Materialization state ===
    /// Materialized input chunks (built once during first next() call).
    materialized_inputs: Vec<Vec<DataChunk>>,

    /// TrieIndex structures built from materialized inputs.
    tries: Vec<TrieIndex>,

    /// Whether materialization is complete.
    materialized: bool,

    // === Iteration state ===
    /// Pre-computed join results.
    results: Vec<JoinResult>,

    /// Current position in results.
    result_position: usize,

    /// Current expansion position within current result's cross product.
    expansion_indices: Vec<usize>,

    /// Whether iteration is exhausted.
    exhausted: bool,
}

impl LeapfrogJoinOperator {
    /// Creates a new leapfrog join operator.
    ///
    /// # Arguments
    /// * `inputs` - Input operators (one per relation).
    /// * `join_key_indices` - Column indices for join keys in each input.
    /// * `output_schema` - Schema of the output columns.
    /// * `output_column_mapping` - Maps output columns to (input_idx, column_idx).
    #[must_use]
    pub fn new(
        inputs: Vec<Box<dyn Operator>>,
        join_key_indices: Vec<Vec<usize>>,
        output_schema: Vec<LogicalType>,
        output_column_mapping: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            inputs,
            join_key_indices,
            output_schema,
            output_column_mapping,
            materialized_inputs: Vec::new(),
            tries: Vec::new(),
            materialized: false,
            results: Vec::new(),
            result_position: 0,
            expansion_indices: Vec::new(),
            exhausted: false,
        }
    }

    /// Materializes all inputs and builds trie indexes.
    fn materialize_inputs(&mut self) -> Result<(), OperatorError> {
        // Phase 1: Collect all chunks from each input
        for input in &mut self.inputs {
            let mut chunks = Vec::new();
            while let Some(chunk) = input.next()? {
                chunks.push(chunk);
            }
            self.materialized_inputs.push(chunks);
        }

        // Phase 2: Build TrieIndex for each input
        for (input_idx, chunks) in self.materialized_inputs.iter().enumerate() {
            let mut trie = TrieIndex::new();
            let key_indices = &self.join_key_indices[input_idx];

            for (chunk_idx, chunk) in chunks.iter().enumerate() {
                for row in 0..chunk.row_count() {
                    // Extract join key values and convert to path
                    if let Some(path) = self.extract_join_keys(chunk, row, key_indices) {
                        // Encode row location as EdgeId for trie storage
                        let row_id = Self::encode_row_id(input_idx, chunk_idx, row);
                        trie.insert(&path, row_id);
                    }
                }
            }
            self.tries.push(trie);
        }

        self.materialized = true;
        Ok(())
    }

    /// Extracts join key values from a row and converts to NodeId path.
    fn extract_join_keys(
        &self,
        chunk: &DataChunk,
        row: usize,
        key_indices: &[usize],
    ) -> Option<Vec<NodeId>> {
        let mut path = Vec::with_capacity(key_indices.len());

        for &col_idx in key_indices {
            let col = chunk.column(col_idx)?;
            let node_id = match col.data_type() {
                LogicalType::Node => col.get_node_id(row),
                LogicalType::Edge => col.get_edge_id(row).map(|e| NodeId::new(e.as_u64())),
                LogicalType::Int64 => col.get_int64(row).map(|i| NodeId::new(i as u64)),
                _ => return None, // Unsupported join key type
            }?;
            path.push(node_id);
        }

        Some(path)
    }

    /// Encodes a row location as an EdgeId for trie storage.
    fn encode_row_id(input_idx: usize, chunk_idx: usize, row: usize) -> EdgeId {
        // Pack: input (8 bits) | chunk (24 bits) | row (32 bits)
        let encoded = ((input_idx as u64) << 56)
            | ((chunk_idx as u64 & 0xFFFFFF) << 32)
            | (row as u64 & 0xFFFFFFFF);
        EdgeId::new(encoded)
    }

    /// Decodes a row location from an EdgeId.
    fn decode_row_id(edge_id: EdgeId) -> RowId {
        let encoded = edge_id.as_u64();
        let input_idx = (encoded >> 56) as usize;
        let chunk_idx = ((encoded >> 32) & 0xFFFFFF) as usize;
        let row = (encoded & 0xFFFFFFFF) as usize;
        (input_idx, chunk_idx, row)
    }

    /// Executes the leapfrog join to find all intersections.
    fn execute_leapfrog(&mut self) -> Result<(), OperatorError> {
        if self.tries.is_empty() {
            return Ok(());
        }

        // Create iterators for each trie at the first level
        let iters: Vec<_> = self.tries.iter().map(|t| t.iter()).collect();

        // Create leapfrog join
        let mut join = LeapfrogJoin::new(iters);

        // Find all intersections at the first level
        while let Some(key) = join.key() {
            // Collect all row IDs from each input that match this key
            let mut row_ids_per_input: Vec<Vec<RowId>> = vec![Vec::new(); self.tries.len()];

            // For each trie, collect all row IDs at this key
            if let Some(child_iters) = join.open() {
                for (input_idx, _child_iter) in child_iters.into_iter().enumerate() {
                    // The child iterator points to the second level of the trie
                    // We need to collect the edge IDs (our encoded row IDs) at this position
                    self.collect_row_ids_at_key(&self.tries[input_idx], key, input_idx, &mut row_ids_per_input[input_idx]);
                }
            }

            // Only add result if all inputs have matching rows
            if row_ids_per_input.iter().all(|ids| !ids.is_empty()) {
                self.results.push(JoinResult { key, row_ids: row_ids_per_input });
            }

            if !join.next() {
                break;
            }
        }

        // Initialize expansion indices if we have results
        if !self.results.is_empty() {
            self.expansion_indices = vec![0; self.inputs.len()];
        }

        Ok(())
    }

    /// Collects all row IDs from a trie at a specific key.
    fn collect_row_ids_at_key(
        &self,
        trie: &TrieIndex,
        key: NodeId,
        input_idx: usize,
        row_ids: &mut Vec<RowId>,
    ) {
        // Get iterator at the key's path
        if let Some(edges) = trie.get(&[key]) {
            for &edge_id in edges {
                let decoded = Self::decode_row_id(edge_id);
                // Verify input index matches (should always match)
                if decoded.0 == input_idx {
                    row_ids.push(decoded);
                }
            }
        }

        // Also check children (for multi-level tries)
        if let Some(iter) = trie.iter_at(&[key]) {
            let mut iter = iter;
            loop {
                if let Some(child_key) = iter.key() {
                    if let Some(edges) = trie.get(&[key, child_key]) {
                        for &edge_id in edges {
                            row_ids.push(Self::decode_row_id(edge_id));
                        }
                    }
                }
                if !iter.next() {
                    break;
                }
            }
        }
    }

    /// Advances to the next combination in the current result's cross product.
    fn advance_expansion(&mut self) -> bool {
        if self.result_position >= self.results.len() {
            return false;
        }

        let result = &self.results[self.result_position];

        // Try to advance from the rightmost input
        for i in (0..self.expansion_indices.len()).rev() {
            self.expansion_indices[i] += 1;
            if self.expansion_indices[i] < result.row_ids[i].len() {
                return true;
            }
            self.expansion_indices[i] = 0;
        }

        // All combinations exhausted for this result, move to next
        self.result_position += 1;
        if self.result_position < self.results.len() {
            self.expansion_indices = vec![0; self.inputs.len()];
            true
        } else {
            false
        }
    }

    /// Builds an output row from the current expansion position.
    fn build_output_row(&self, builder: &mut DataChunkBuilder) -> Result<(), OperatorError> {
        let result = &self.results[self.result_position];

        for (out_col, &(input_idx, in_col)) in self.output_column_mapping.iter().enumerate() {
            let expansion_idx = self.expansion_indices[input_idx];
            let (_, chunk_idx, row) = result.row_ids[input_idx][expansion_idx];

            let chunk = &self.materialized_inputs[input_idx][chunk_idx];
            let col = chunk
                .column(in_col)
                .ok_or_else(|| OperatorError::ColumnNotFound(in_col.to_string()))?;

            let out_col_vec = builder
                .column_mut(out_col)
                .ok_or_else(|| OperatorError::ColumnNotFound(out_col.to_string()))?;

            // Copy value from input to output
            if let Some(value) = col.get_value(row) {
                out_col_vec.push_value(value);
            } else {
                out_col_vec.push_value(Value::Null);
            }
        }

        builder.advance_row();
        Ok(())
    }
}

impl Operator for LeapfrogJoinOperator {
    fn next(&mut self) -> OperatorResult {
        // First call: materialize inputs and execute leapfrog
        if !self.materialized {
            self.materialize_inputs()?;
            self.execute_leapfrog()?;
        }

        if self.exhausted || self.results.is_empty() {
            return Ok(None);
        }

        // Check if we've exhausted all results
        if self.result_position >= self.results.len() {
            self.exhausted = true;
            return Ok(None);
        }

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

        while !builder.is_full() {
            self.build_output_row(&mut builder)?;

            if !self.advance_expansion() {
                self.exhausted = true;
                break;
            }
        }

        if builder.row_count() > 0 {
            Ok(Some(builder.finish()))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        for input in &mut self.inputs {
            input.reset();
        }
        self.materialized_inputs.clear();
        self.tries.clear();
        self.materialized = false;
        self.results.clear();
        self.result_position = 0;
        self.expansion_indices.clear();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "LeapfrogJoin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::vector::ValueVector;

    /// Creates a simple scan operator that returns a single chunk.
    struct MockScanOperator {
        chunk: Option<DataChunk>,
        returned: bool,
    }

    impl MockScanOperator {
        fn new(chunk: DataChunk) -> Self {
            Self {
                chunk: Some(chunk),
                returned: false,
            }
        }
    }

    impl Operator for MockScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.returned {
                Ok(None)
            } else {
                self.returned = true;
                Ok(self.chunk.take())
            }
        }

        fn reset(&mut self) {
            self.returned = false;
        }

        fn name(&self) -> &'static str {
            "MockScan"
        }
    }

    fn create_node_chunk(node_ids: &[i64]) -> DataChunk {
        let mut col = ValueVector::with_type(LogicalType::Int64);
        for &id in node_ids {
            col.push_int64(id);
        }
        DataChunk::new(vec![col])
    }

    #[test]
    fn test_leapfrog_binary_intersection() {
        // Input 1: nodes [1, 2, 3, 5]
        // Input 2: nodes [2, 3, 4, 5]
        // Expected intersection: [2, 3, 5]

        let chunk1 = create_node_chunk(&[1, 2, 3, 5]);
        let chunk2 = create_node_chunk(&[2, 3, 4, 5]);

        let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
        let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));

        let mut leapfrog = LeapfrogJoinOperator::new(
            vec![op1, op2],
            vec![vec![0], vec![0]], // Join on first column of each
            vec![LogicalType::Int64, LogicalType::Int64],
            vec![(0, 0), (1, 0)], // Output both columns
        );

        let mut all_results = Vec::new();
        while let Some(chunk) = leapfrog.next().unwrap() {
            for row in 0..chunk.row_count() {
                let val1 = chunk.column(0).unwrap().get_int64(row).unwrap();
                let val2 = chunk.column(1).unwrap().get_int64(row).unwrap();
                all_results.push((val1, val2));
            }
        }

        // Should find 3 matches: (2,2), (3,3), (5,5)
        assert_eq!(all_results.len(), 3);
        assert!(all_results.contains(&(2, 2)));
        assert!(all_results.contains(&(3, 3)));
        assert!(all_results.contains(&(5, 5)));
    }

    #[test]
    fn test_leapfrog_empty_intersection() {
        // Input 1: nodes [1, 2, 3]
        // Input 2: nodes [4, 5, 6]
        // Expected: empty

        let chunk1 = create_node_chunk(&[1, 2, 3]);
        let chunk2 = create_node_chunk(&[4, 5, 6]);

        let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1));
        let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2));

        let mut leapfrog = LeapfrogJoinOperator::new(
            vec![op1, op2],
            vec![vec![0], vec![0]],
            vec![LogicalType::Int64, LogicalType::Int64],
            vec![(0, 0), (1, 0)],
        );

        let result = leapfrog.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_leapfrog_reset() {
        let chunk1 = create_node_chunk(&[1, 2, 3]);
        let chunk2 = create_node_chunk(&[2, 3, 4]);

        let op1: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk1.clone()));
        let op2: Box<dyn Operator> = Box::new(MockScanOperator::new(chunk2.clone()));

        let mut leapfrog = LeapfrogJoinOperator::new(
            vec![op1, op2],
            vec![vec![0], vec![0]],
            vec![LogicalType::Int64, LogicalType::Int64],
            vec![(0, 0), (1, 0)],
        );

        // First iteration - consume all results
        let mut _count = 0;
        while leapfrog.next().unwrap().is_some() {
            _count += 1;
        }

        // Reset won't work with MockScanOperator since the chunk is taken
        // but the reset logic itself should work
        leapfrog.reset();
        assert!(!leapfrog.materialized);
        assert!(leapfrog.results.is_empty());
    }

    #[test]
    fn test_encode_decode_row_id() {
        let test_cases = [
            (0, 0, 0),
            (1, 2, 3),
            (255, 16777215, 4294967295), // Max values for each field
        ];

        for (input_idx, chunk_idx, row) in test_cases {
            let encoded = LeapfrogJoinOperator::encode_row_id(input_idx, chunk_idx, row);
            let decoded = LeapfrogJoinOperator::decode_row_id(encoded);
            assert_eq!(decoded, (input_idx, chunk_idx, row));
        }
    }
}
