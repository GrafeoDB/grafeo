//! Join operators for combining data from two sources.
//!
//! This module provides:
//! - `HashJoinOperator`: Efficient hash-based join for equality conditions
//! - `NestedLoopJoinOperator`: General-purpose join for any condition

use std::collections::HashMap;

use grafeo_common::types::{LogicalType, Value};

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use crate::execution::{DataChunk, ValueVector};

/// The type of join to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join: only matching rows from both sides.
    Inner,
    /// Left outer join: all rows from left, matching from right (nulls if no match).
    Left,
    /// Right outer join: all rows from right, matching from left (nulls if no match).
    Right,
    /// Full outer join: all rows from both sides.
    Full,
    /// Cross join: cartesian product of both sides.
    Cross,
    /// Semi join: rows from left that have a match in right.
    Semi,
    /// Anti join: rows from left that have no match in right.
    Anti,
}

/// A hash key that can be hashed and compared for join operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HashKey {
    /// Null key.
    Null,
    /// Boolean key.
    Bool(bool),
    /// Integer key.
    Int64(i64),
    /// String key (using the string content for hashing).
    String(String),
    /// Composite key for multi-column joins.
    Composite(Vec<HashKey>),
}

impl HashKey {
    /// Creates a hash key from a Value.
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Null => HashKey::Null,
            Value::Bool(b) => HashKey::Bool(*b),
            Value::Int64(i) => HashKey::Int64(*i),
            Value::Float64(f) => {
                // Convert float to bits for consistent hashing
                HashKey::Int64(f.to_bits() as i64)
            }
            Value::String(s) => HashKey::String(s.to_string()),
            Value::Bytes(b) => {
                // Use byte content for hashing
                HashKey::String(format!("{b:?}"))
            }
            Value::Timestamp(t) => HashKey::Int64(t.as_micros()),
            Value::List(items) => {
                HashKey::Composite(items.iter().map(HashKey::from_value).collect())
            }
            Value::Map(map) => {
                let mut keys: Vec<_> = map
                    .iter()
                    .map(|(k, v)| {
                        HashKey::Composite(vec![
                            HashKey::String(k.to_string()),
                            HashKey::from_value(v),
                        ])
                    })
                    .collect();
                keys.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
                HashKey::Composite(keys)
            }
            Value::Vector(v) => {
                // Hash vectors by converting each f32 to its bit representation
                HashKey::Composite(
                    v.iter()
                        .map(|f| HashKey::Int64(f.to_bits() as i64))
                        .collect(),
                )
            }
        }
    }

    /// Creates a hash key from a column value at a given row.
    pub fn from_column(column: &ValueVector, row: usize) -> Option<Self> {
        column.get_value(row).map(|v| Self::from_value(&v))
    }
}

/// Hash join operator.
///
/// Builds a hash table from the build side (right) and probes with the probe side (left).
/// Efficient for equality joins on one or more columns.
pub struct HashJoinOperator {
    /// Left (probe) side operator.
    probe_side: Box<dyn Operator>,
    /// Right (build) side operator.
    build_side: Box<dyn Operator>,
    /// Column indices on the probe side for join keys.
    probe_keys: Vec<usize>,
    /// Column indices on the build side for join keys.
    build_keys: Vec<usize>,
    /// Join type.
    join_type: JoinType,
    /// Output schema (combined from both sides).
    output_schema: Vec<LogicalType>,
    /// Hash table: key -> list of (chunk_index, row_index).
    hash_table: HashMap<HashKey, Vec<(usize, usize)>>,
    /// Materialized build side chunks.
    build_chunks: Vec<DataChunk>,
    /// Whether the build phase is complete.
    build_complete: bool,
    /// Current probe chunk being processed.
    current_probe_chunk: Option<DataChunk>,
    /// Current row in the probe chunk.
    current_probe_row: usize,
    /// Current position in the hash table matches for the current probe row.
    current_match_position: usize,
    /// Current matches for the current probe row.
    current_matches: Vec<(usize, usize)>,
    /// For left/full outer joins: track which probe rows had matches.
    probe_matched: Vec<bool>,
    /// For right/full outer joins: track which build rows were matched.
    build_matched: Vec<Vec<bool>>,
    /// Whether we're in the emit unmatched phase (for outer joins).
    emitting_unmatched: bool,
    /// Current chunk index when emitting unmatched rows.
    unmatched_chunk_idx: usize,
    /// Current row index when emitting unmatched rows.
    unmatched_row_idx: usize,
}

impl HashJoinOperator {
    /// Creates a new hash join operator.
    ///
    /// # Arguments
    /// * `probe_side` - Left side operator (will be probed).
    /// * `build_side` - Right side operator (will build hash table).
    /// * `probe_keys` - Column indices on probe side for join keys.
    /// * `build_keys` - Column indices on build side for join keys.
    /// * `join_type` - Type of join to perform.
    /// * `output_schema` - Schema of the output (probe columns + build columns).
    pub fn new(
        probe_side: Box<dyn Operator>,
        build_side: Box<dyn Operator>,
        probe_keys: Vec<usize>,
        build_keys: Vec<usize>,
        join_type: JoinType,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            probe_side,
            build_side,
            probe_keys,
            build_keys,
            join_type,
            output_schema,
            hash_table: HashMap::new(),
            build_chunks: Vec::new(),
            build_complete: false,
            current_probe_chunk: None,
            current_probe_row: 0,
            current_match_position: 0,
            current_matches: Vec::new(),
            probe_matched: Vec::new(),
            build_matched: Vec::new(),
            emitting_unmatched: false,
            unmatched_chunk_idx: 0,
            unmatched_row_idx: 0,
        }
    }

    /// Builds the hash table from the build side.
    fn build_hash_table(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.build_side.next()? {
            let chunk_idx = self.build_chunks.len();

            // Initialize match tracking for outer joins
            if matches!(self.join_type, JoinType::Right | JoinType::Full) {
                self.build_matched.push(vec![false; chunk.row_count()]);
            }

            // Add each row to the hash table
            for row in chunk.selected_indices() {
                let key = self.extract_key(&chunk, row, &self.build_keys)?;

                // Skip null keys for inner/semi/anti joins
                if matches!(key, HashKey::Null)
                    && !matches!(
                        self.join_type,
                        JoinType::Left | JoinType::Right | JoinType::Full
                    )
                {
                    continue;
                }

                self.hash_table
                    .entry(key)
                    .or_default()
                    .push((chunk_idx, row));
            }

            self.build_chunks.push(chunk);
        }

        self.build_complete = true;
        Ok(())
    }

    /// Extracts a hash key from a chunk row.
    fn extract_key(
        &self,
        chunk: &DataChunk,
        row: usize,
        key_columns: &[usize],
    ) -> Result<HashKey, OperatorError> {
        if key_columns.len() == 1 {
            let col = chunk.column(key_columns[0]).ok_or_else(|| {
                OperatorError::ColumnNotFound(format!("column {}", key_columns[0]))
            })?;
            Ok(HashKey::from_column(col, row).unwrap_or(HashKey::Null))
        } else {
            let keys: Vec<HashKey> = key_columns
                .iter()
                .map(|&col_idx| {
                    chunk
                        .column(col_idx)
                        .and_then(|col| HashKey::from_column(col, row))
                        .unwrap_or(HashKey::Null)
                })
                .collect();
            Ok(HashKey::Composite(keys))
        }
    }

    /// Produces an output row from a probe row and build row.
    fn produce_output_row(
        &self,
        builder: &mut DataChunkBuilder,
        probe_chunk: &DataChunk,
        probe_row: usize,
        build_chunk: Option<&DataChunk>,
        build_row: Option<usize>,
    ) -> Result<(), OperatorError> {
        let probe_col_count = probe_chunk.column_count();

        // Copy probe side columns
        for col_idx in 0..probe_col_count {
            let src_col = probe_chunk
                .column(col_idx)
                .ok_or_else(|| OperatorError::ColumnNotFound(format!("probe column {col_idx}")))?;
            let dst_col = builder
                .column_mut(col_idx)
                .ok_or_else(|| OperatorError::ColumnNotFound(format!("output column {col_idx}")))?;

            if let Some(value) = src_col.get_value(probe_row) {
                dst_col.push_value(value);
            } else {
                dst_col.push_value(Value::Null);
            }
        }

        // Copy build side columns
        match (build_chunk, build_row) {
            (Some(chunk), Some(row)) => {
                for col_idx in 0..chunk.column_count() {
                    let src_col = chunk.column(col_idx).ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("build column {col_idx}"))
                    })?;
                    let dst_col =
                        builder
                            .column_mut(probe_col_count + col_idx)
                            .ok_or_else(|| {
                                OperatorError::ColumnNotFound(format!(
                                    "output column {}",
                                    probe_col_count + col_idx
                                ))
                            })?;

                    if let Some(value) = src_col.get_value(row) {
                        dst_col.push_value(value);
                    } else {
                        dst_col.push_value(Value::Null);
                    }
                }
            }
            _ => {
                // Emit nulls for build side (left outer join case)
                if !self.build_chunks.is_empty() {
                    let build_col_count = self.build_chunks[0].column_count();
                    for col_idx in 0..build_col_count {
                        let dst_col =
                            builder
                                .column_mut(probe_col_count + col_idx)
                                .ok_or_else(|| {
                                    OperatorError::ColumnNotFound(format!(
                                        "output column {}",
                                        probe_col_count + col_idx
                                    ))
                                })?;
                        dst_col.push_value(Value::Null);
                    }
                }
            }
        }

        builder.advance_row();
        Ok(())
    }

    /// Gets the next probe chunk.
    fn get_next_probe_chunk(&mut self) -> Result<bool, OperatorError> {
        let chunk = self.probe_side.next()?;
        if let Some(ref c) = chunk {
            // Initialize match tracking for outer joins
            if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                self.probe_matched = vec![false; c.row_count()];
            }
        }
        let has_chunk = chunk.is_some();
        self.current_probe_chunk = chunk;
        self.current_probe_row = 0;
        Ok(has_chunk)
    }

    /// Emits unmatched build rows for right/full outer joins.
    fn emit_unmatched_build(&mut self) -> OperatorResult {
        if self.build_matched.is_empty() {
            return Ok(None);
        }

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

        // Determine probe column count from schema or first probe chunk
        let probe_col_count = if !self.build_chunks.is_empty() {
            self.output_schema.len() - self.build_chunks[0].column_count()
        } else {
            0
        };

        while self.unmatched_chunk_idx < self.build_chunks.len() {
            let chunk = &self.build_chunks[self.unmatched_chunk_idx];
            let matched = &self.build_matched[self.unmatched_chunk_idx];

            while self.unmatched_row_idx < matched.len() {
                if !matched[self.unmatched_row_idx] {
                    // This row was not matched - emit with nulls on probe side

                    // Emit nulls for probe side
                    for col_idx in 0..probe_col_count {
                        if let Some(dst_col) = builder.column_mut(col_idx) {
                            dst_col.push_value(Value::Null);
                        }
                    }

                    // Copy build side values
                    for col_idx in 0..chunk.column_count() {
                        if let (Some(src_col), Some(dst_col)) = (
                            chunk.column(col_idx),
                            builder.column_mut(probe_col_count + col_idx),
                        ) {
                            if let Some(value) = src_col.get_value(self.unmatched_row_idx) {
                                dst_col.push_value(value);
                            } else {
                                dst_col.push_value(Value::Null);
                            }
                        }
                    }

                    builder.advance_row();

                    if builder.is_full() {
                        self.unmatched_row_idx += 1;
                        return Ok(Some(builder.finish()));
                    }
                }

                self.unmatched_row_idx += 1;
            }

            self.unmatched_chunk_idx += 1;
            self.unmatched_row_idx = 0;
        }

        if builder.row_count() > 0 {
            Ok(Some(builder.finish()))
        } else {
            Ok(None)
        }
    }
}

impl Operator for HashJoinOperator {
    fn next(&mut self) -> OperatorResult {
        // Phase 1: Build hash table
        if !self.build_complete {
            self.build_hash_table()?;
        }

        // Phase 3: Emit unmatched build rows (right/full outer join)
        if self.emitting_unmatched {
            return self.emit_unmatched_build();
        }

        // Phase 2: Probe
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

        loop {
            // Get current probe chunk or fetch new one
            if self.current_probe_chunk.is_none() {
                if !self.get_next_probe_chunk()? {
                    // No more probe data
                    if matches!(self.join_type, JoinType::Right | JoinType::Full) {
                        self.emitting_unmatched = true;
                        return self.emit_unmatched_build();
                    }
                    return if builder.row_count() > 0 {
                        Ok(Some(builder.finish()))
                    } else {
                        Ok(None)
                    };
                }
            }

            // Invariant: current_probe_chunk is Some here - the guard at line 396 either
            // populates it via get_next_probe_chunk() or returns from the function
            let probe_chunk = self
                .current_probe_chunk
                .as_ref()
                .expect("probe chunk is Some: guard at line 396 ensures this");
            let probe_rows: Vec<usize> = probe_chunk.selected_indices().collect();

            while self.current_probe_row < probe_rows.len() {
                let probe_row = probe_rows[self.current_probe_row];

                // If we don't have current matches, look them up
                if self.current_matches.is_empty() && self.current_match_position == 0 {
                    let key = self.extract_key(probe_chunk, probe_row, &self.probe_keys)?;

                    // Handle semi/anti joins differently
                    match self.join_type {
                        JoinType::Semi => {
                            if self.hash_table.contains_key(&key) {
                                // Emit probe row only
                                for col_idx in 0..probe_chunk.column_count() {
                                    if let (Some(src_col), Some(dst_col)) =
                                        (probe_chunk.column(col_idx), builder.column_mut(col_idx))
                                    {
                                        if let Some(value) = src_col.get_value(probe_row) {
                                            dst_col.push_value(value);
                                        }
                                    }
                                }
                                builder.advance_row();
                            }
                            self.current_probe_row += 1;
                            continue;
                        }
                        JoinType::Anti => {
                            if !self.hash_table.contains_key(&key) {
                                // Emit probe row only
                                for col_idx in 0..probe_chunk.column_count() {
                                    if let (Some(src_col), Some(dst_col)) =
                                        (probe_chunk.column(col_idx), builder.column_mut(col_idx))
                                    {
                                        if let Some(value) = src_col.get_value(probe_row) {
                                            dst_col.push_value(value);
                                        }
                                    }
                                }
                                builder.advance_row();
                            }
                            self.current_probe_row += 1;
                            continue;
                        }
                        _ => {
                            self.current_matches =
                                self.hash_table.get(&key).cloned().unwrap_or_default();
                        }
                    }
                }

                // Process matches
                if self.current_matches.is_empty() {
                    // No matches - for left/full outer join, emit with nulls
                    if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                        self.produce_output_row(&mut builder, probe_chunk, probe_row, None, None)?;
                    }
                    self.current_probe_row += 1;
                    self.current_match_position = 0;
                } else {
                    // Process each match
                    while self.current_match_position < self.current_matches.len() {
                        let (build_chunk_idx, build_row) =
                            self.current_matches[self.current_match_position];
                        let build_chunk = &self.build_chunks[build_chunk_idx];

                        // Mark as matched for outer joins
                        if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                            if probe_row < self.probe_matched.len() {
                                self.probe_matched[probe_row] = true;
                            }
                        }
                        if matches!(self.join_type, JoinType::Right | JoinType::Full) {
                            if build_chunk_idx < self.build_matched.len()
                                && build_row < self.build_matched[build_chunk_idx].len()
                            {
                                self.build_matched[build_chunk_idx][build_row] = true;
                            }
                        }

                        self.produce_output_row(
                            &mut builder,
                            probe_chunk,
                            probe_row,
                            Some(build_chunk),
                            Some(build_row),
                        )?;

                        self.current_match_position += 1;

                        if builder.is_full() {
                            return Ok(Some(builder.finish()));
                        }
                    }

                    // Done with this probe row
                    self.current_probe_row += 1;
                    self.current_matches.clear();
                    self.current_match_position = 0;
                }

                if builder.is_full() {
                    return Ok(Some(builder.finish()));
                }
            }

            // Done with current probe chunk
            self.current_probe_chunk = None;
            self.current_probe_row = 0;

            if builder.row_count() > 0 {
                return Ok(Some(builder.finish()));
            }
        }
    }

    fn reset(&mut self) {
        self.probe_side.reset();
        self.build_side.reset();
        self.hash_table.clear();
        self.build_chunks.clear();
        self.build_complete = false;
        self.current_probe_chunk = None;
        self.current_probe_row = 0;
        self.current_match_position = 0;
        self.current_matches.clear();
        self.probe_matched.clear();
        self.build_matched.clear();
        self.emitting_unmatched = false;
        self.unmatched_chunk_idx = 0;
        self.unmatched_row_idx = 0;
    }

    fn name(&self) -> &'static str {
        "HashJoin"
    }
}

/// Nested loop join operator.
///
/// Performs a cartesian product of both sides, filtering by the join condition.
/// Less efficient than hash join but supports any join condition.
pub struct NestedLoopJoinOperator {
    /// Left side operator.
    left: Box<dyn Operator>,
    /// Right side operator.
    right: Box<dyn Operator>,
    /// Join condition predicate (if any).
    condition: Option<Box<dyn JoinCondition>>,
    /// Join type.
    join_type: JoinType,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Materialized right side chunks.
    right_chunks: Vec<DataChunk>,
    /// Whether the right side is materialized.
    right_materialized: bool,
    /// Current left chunk.
    current_left_chunk: Option<DataChunk>,
    /// Current row in the left chunk.
    current_left_row: usize,
    /// Current chunk index in the right side.
    current_right_chunk: usize,
    /// Whether the current left row has been matched (for Left Join).
    current_left_matched: bool,
    /// Current row in the current right chunk.
    current_right_row: usize,
}

/// Trait for join conditions.
pub trait JoinCondition: Send + Sync {
    /// Evaluates the condition for a pair of rows.
    fn evaluate(
        &self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> bool;
}

/// A simple equality condition for nested loop joins.
pub struct EqualityCondition {
    /// Column index on the left side.
    left_column: usize,
    /// Column index on the right side.
    right_column: usize,
}

impl EqualityCondition {
    /// Creates a new equality condition.
    pub fn new(left_column: usize, right_column: usize) -> Self {
        Self {
            left_column,
            right_column,
        }
    }
}

impl JoinCondition for EqualityCondition {
    fn evaluate(
        &self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> bool {
        let left_val = left_chunk
            .column(self.left_column)
            .and_then(|c| c.get_value(left_row));
        let right_val = right_chunk
            .column(self.right_column)
            .and_then(|c| c.get_value(right_row));

        match (left_val, right_val) {
            (Some(l), Some(r)) => l == r,
            _ => false,
        }
    }
}

impl NestedLoopJoinOperator {
    /// Creates a new nested loop join operator.
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        condition: Option<Box<dyn JoinCondition>>,
        join_type: JoinType,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            left,
            right,
            condition,
            join_type,
            output_schema,
            right_chunks: Vec::new(),
            right_materialized: false,
            current_left_chunk: None,
            current_left_row: 0,
            current_right_chunk: 0,
            current_right_row: 0,
            current_left_matched: false,
        }
    }

    /// Materializes the right side.
    fn materialize_right(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.right.next()? {
            self.right_chunks.push(chunk);
        }
        self.right_materialized = true;
        Ok(())
    }

    /// Produces an output row.
    fn produce_row(
        &self,
        builder: &mut DataChunkBuilder,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) {
        // Copy left columns
        for col_idx in 0..left_chunk.column_count() {
            if let (Some(src), Some(dst)) =
                (left_chunk.column(col_idx), builder.column_mut(col_idx))
            {
                if let Some(val) = src.get_value(left_row) {
                    dst.push_value(val);
                } else {
                    dst.push_value(Value::Null);
                }
            }
        }

        // Copy right columns
        let left_col_count = left_chunk.column_count();
        for col_idx in 0..right_chunk.column_count() {
            if let (Some(src), Some(dst)) = (
                right_chunk.column(col_idx),
                builder.column_mut(left_col_count + col_idx),
            ) {
                if let Some(val) = src.get_value(right_row) {
                    dst.push_value(val);
                } else {
                    dst.push_value(Value::Null);
                }
            }
        }

        builder.advance_row();
    }

    /// Produces an output row with NULLs for the right side (for unmatched left rows in Left Join).
    fn produce_left_unmatched_row(
        &self,
        builder: &mut DataChunkBuilder,
        left_chunk: &DataChunk,
        left_row: usize,
        right_col_count: usize,
    ) {
        // Copy left columns
        for col_idx in 0..left_chunk.column_count() {
            if let (Some(src), Some(dst)) =
                (left_chunk.column(col_idx), builder.column_mut(col_idx))
            {
                if let Some(val) = src.get_value(left_row) {
                    dst.push_value(val);
                } else {
                    dst.push_value(Value::Null);
                }
            }
        }

        // Fill right columns with NULLs
        let left_col_count = left_chunk.column_count();
        for col_idx in 0..right_col_count {
            if let Some(dst) = builder.column_mut(left_col_count + col_idx) {
                dst.push_value(Value::Null);
            }
        }

        builder.advance_row();
    }
}

impl Operator for NestedLoopJoinOperator {
    fn next(&mut self) -> OperatorResult {
        // Materialize right side
        if !self.right_materialized {
            self.materialize_right()?;
        }

        // If right side is empty and not a left outer join, return nothing
        if self.right_chunks.is_empty() && !matches!(self.join_type, JoinType::Left) {
            return Ok(None);
        }

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

        loop {
            // Get current left chunk
            if self.current_left_chunk.is_none() {
                self.current_left_chunk = self.left.next()?;
                self.current_left_row = 0;
                self.current_right_chunk = 0;
                self.current_right_row = 0;

                if self.current_left_chunk.is_none() {
                    // No more left data
                    return if builder.row_count() > 0 {
                        Ok(Some(builder.finish()))
                    } else {
                        Ok(None)
                    };
                }
            }

            let left_chunk = self.current_left_chunk.as_ref().unwrap();
            let left_rows: Vec<usize> = left_chunk.selected_indices().collect();

            // Calculate right column count for potential unmatched rows
            let right_col_count = if !self.right_chunks.is_empty() {
                self.right_chunks[0].column_count()
            } else {
                // Infer from output schema
                self.output_schema
                    .len()
                    .saturating_sub(left_chunk.column_count())
            };

            // Process current left row against all right rows
            while self.current_left_row < left_rows.len() {
                let left_row = left_rows[self.current_left_row];

                // Reset match tracking for this left row
                if self.current_right_chunk == 0 && self.current_right_row == 0 {
                    self.current_left_matched = false;
                }

                // Cross join or inner/other join
                while self.current_right_chunk < self.right_chunks.len() {
                    let right_chunk = &self.right_chunks[self.current_right_chunk];
                    let right_rows: Vec<usize> = right_chunk.selected_indices().collect();

                    while self.current_right_row < right_rows.len() {
                        let right_row = right_rows[self.current_right_row];

                        // Check condition
                        let matches = match &self.condition {
                            Some(cond) => {
                                cond.evaluate(left_chunk, left_row, right_chunk, right_row)
                            }
                            None => true, // Cross join
                        };

                        if matches {
                            self.current_left_matched = true;
                            self.produce_row(
                                &mut builder,
                                left_chunk,
                                left_row,
                                right_chunk,
                                right_row,
                            );

                            if builder.is_full() {
                                self.current_right_row += 1;
                                return Ok(Some(builder.finish()));
                            }
                        }

                        self.current_right_row += 1;
                    }

                    self.current_right_chunk += 1;
                    self.current_right_row = 0;
                }

                // Done processing all right rows for this left row
                // For Left Join, emit unmatched left row with NULLs
                if matches!(self.join_type, JoinType::Left) && !self.current_left_matched {
                    self.produce_left_unmatched_row(
                        &mut builder,
                        left_chunk,
                        left_row,
                        right_col_count,
                    );

                    if builder.is_full() {
                        self.current_left_row += 1;
                        self.current_right_chunk = 0;
                        self.current_right_row = 0;
                        return Ok(Some(builder.finish()));
                    }
                }

                // Move to next left row
                self.current_left_row += 1;
                self.current_right_chunk = 0;
                self.current_right_row = 0;
            }

            // Done with current left chunk
            self.current_left_chunk = None;

            if builder.row_count() > 0 {
                return Ok(Some(builder.finish()));
            }
        }
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.right_chunks.clear();
        self.right_materialized = false;
        self.current_left_chunk = None;
        self.current_left_row = 0;
        self.current_right_chunk = 0;
        self.current_right_row = 0;
        self.current_left_matched = false;
    }

    fn name(&self) -> &'static str {
        "NestedLoopJoin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;

    /// Mock operator for testing.
    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    fn create_int_chunk(values: &[i64]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for &v in values {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        builder.finish()
    }

    #[test]
    fn test_hash_join_inner() {
        // Left: [1, 2, 3, 4]
        // Right: [2, 3, 4, 5]
        // Inner join on column 0 should produce: [2, 3, 4]

        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 3, 4])]);
        let right = MockOperator::new(vec![create_int_chunk(&[2, 3, 4, 5])]);

        let output_schema = vec![LogicalType::Int64, LogicalType::Int64];
        let mut join = HashJoinOperator::new(
            Box::new(left),
            Box::new(right),
            vec![0],
            vec![0],
            JoinType::Inner,
            output_schema,
        );

        let mut results = Vec::new();
        while let Some(chunk) = join.next().unwrap() {
            for row in chunk.selected_indices() {
                let left_val = chunk.column(0).unwrap().get_int64(row).unwrap();
                let right_val = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((left_val, right_val));
            }
        }

        results.sort();
        assert_eq!(results, vec![(2, 2), (3, 3), (4, 4)]);
    }

    #[test]
    fn test_hash_join_left_outer() {
        // Left: [1, 2, 3]
        // Right: [2, 3]
        // Left outer join should produce: [(1, null), (2, 2), (3, 3)]

        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 3])]);
        let right = MockOperator::new(vec![create_int_chunk(&[2, 3])]);

        let output_schema = vec![LogicalType::Int64, LogicalType::Int64];
        let mut join = HashJoinOperator::new(
            Box::new(left),
            Box::new(right),
            vec![0],
            vec![0],
            JoinType::Left,
            output_schema,
        );

        let mut results = Vec::new();
        while let Some(chunk) = join.next().unwrap() {
            for row in chunk.selected_indices() {
                let left_val = chunk.column(0).unwrap().get_int64(row).unwrap();
                let right_val = chunk.column(1).unwrap().get_int64(row);
                results.push((left_val, right_val));
            }
        }

        results.sort_by_key(|(l, _)| *l);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (1, None)); // No match
        assert_eq!(results[1], (2, Some(2)));
        assert_eq!(results[2], (3, Some(3)));
    }

    #[test]
    fn test_nested_loop_cross_join() {
        // Left: [1, 2]
        // Right: [10, 20]
        // Cross join should produce: [(1,10), (1,20), (2,10), (2,20)]

        let left = MockOperator::new(vec![create_int_chunk(&[1, 2])]);
        let right = MockOperator::new(vec![create_int_chunk(&[10, 20])]);

        let output_schema = vec![LogicalType::Int64, LogicalType::Int64];
        let mut join = NestedLoopJoinOperator::new(
            Box::new(left),
            Box::new(right),
            None,
            JoinType::Cross,
            output_schema,
        );

        let mut results = Vec::new();
        while let Some(chunk) = join.next().unwrap() {
            for row in chunk.selected_indices() {
                let left_val = chunk.column(0).unwrap().get_int64(row).unwrap();
                let right_val = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((left_val, right_val));
            }
        }

        results.sort();
        assert_eq!(results, vec![(1, 10), (1, 20), (2, 10), (2, 20)]);
    }

    #[test]
    fn test_hash_join_semi() {
        // Left: [1, 2, 3, 4]
        // Right: [2, 4]
        // Semi join should produce: [2, 4] (only left rows that have matches)

        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 3, 4])]);
        let right = MockOperator::new(vec![create_int_chunk(&[2, 4])]);

        // Semi join only outputs probe (left) columns
        let output_schema = vec![LogicalType::Int64];
        let mut join = HashJoinOperator::new(
            Box::new(left),
            Box::new(right),
            vec![0],
            vec![0],
            JoinType::Semi,
            output_schema,
        );

        let mut results = Vec::new();
        while let Some(chunk) = join.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        results.sort();
        assert_eq!(results, vec![2, 4]);
    }

    #[test]
    fn test_hash_join_anti() {
        // Left: [1, 2, 3, 4]
        // Right: [2, 4]
        // Anti join should produce: [1, 3] (left rows with no matches)

        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 3, 4])]);
        let right = MockOperator::new(vec![create_int_chunk(&[2, 4])]);

        let output_schema = vec![LogicalType::Int64];
        let mut join = HashJoinOperator::new(
            Box::new(left),
            Box::new(right),
            vec![0],
            vec![0],
            JoinType::Anti,
            output_schema,
        );

        let mut results = Vec::new();
        while let Some(chunk) = join.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        results.sort();
        assert_eq!(results, vec![1, 3]);
    }
}
