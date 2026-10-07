//! Join operators for combining data from two sources.
//!
//! This module provides:
//! - `HashJoinOperator`: Efficient hash-based join for equality conditions
//! - `NestedLoopJoinOperator`: General-purpose join for any condition

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use arcstr::ArcStr;
use grafeo_common::types::{HashableValue, LogicalType, Value};

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::chunk::{ColumnTypes, DataChunkBuilder, copied_column_types};
use crate::execution::{DataChunk, ValueVector};

/// The type of join to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
pub enum HashKey {
    /// Null key.
    Null,
    /// Boolean key.
    Bool(bool),
    /// Integer key.
    Int64(i64),
    /// String key (cheap clone via ArcStr refcount).
    String(ArcStr),
    /// Byte content key.
    Bytes(Vec<u8>),
    /// Composite key for multi-column joins.
    Composite(Vec<HashKey>),
}

impl Ord for HashKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (HashKey::Null, HashKey::Null) => Ordering::Equal,
            (HashKey::Null, _) => Ordering::Less,
            (_, HashKey::Null) => Ordering::Greater,
            (HashKey::Bool(a), HashKey::Bool(b)) => a.cmp(b),
            (HashKey::Bool(_), _) => Ordering::Less,
            (_, HashKey::Bool(_)) => Ordering::Greater,
            (HashKey::Int64(a), HashKey::Int64(b)) => a.cmp(b),
            (HashKey::Int64(_), _) => Ordering::Less,
            (_, HashKey::Int64(_)) => Ordering::Greater,
            (HashKey::String(a), HashKey::String(b)) => a.cmp(b),
            (HashKey::String(_), _) => Ordering::Less,
            (_, HashKey::String(_)) => Ordering::Greater,
            (HashKey::Bytes(a), HashKey::Bytes(b)) => a.cmp(b),
            (HashKey::Bytes(_), _) => Ordering::Less,
            (_, HashKey::Bytes(_)) => Ordering::Greater,
            (HashKey::Composite(a), HashKey::Composite(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for HashKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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
                // reason: intentional bit-level reinterpretation for hashing
                #[allow(clippy::cast_possible_wrap)]
                HashKey::Int64(f.to_bits() as i64)
            }
            Value::String(s) => HashKey::String(s.clone()),
            Value::Bytes(b) => HashKey::Bytes(b.to_vec()),
            Value::Timestamp(t) => HashKey::Int64(t.as_micros()),
            // reason: date days and time nanos fit i64
            #[allow(clippy::cast_possible_wrap)]
            Value::Date(d) => HashKey::Int64(d.as_days() as i64),
            // reason: date days and time nanos are small, fit i64
            #[allow(clippy::cast_possible_wrap)]
            Value::Time(t) => HashKey::Int64(t.as_nanos() as i64),
            Value::Duration(d) => HashKey::Composite(vec![
                HashKey::Int64(d.months()),
                HashKey::Int64(d.days()),
                HashKey::Int64(d.nanos()),
            ]),
            Value::ZonedDatetime(zdt) => HashKey::Int64(zdt.as_timestamp().as_micros()),
            Value::List(items) => {
                HashKey::Composite(items.iter().map(HashKey::from_value).collect())
            }
            Value::Map(map) => {
                // BTreeMap::iter() visits entries in ascending key order, so no sort needed.
                let keys: Vec<_> = map
                    .iter()
                    .map(|(k, v)| {
                        HashKey::Composite(vec![
                            HashKey::String(ArcStr::from(k.as_str())),
                            HashKey::from_value(v),
                        ])
                    })
                    .collect();
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
            Value::Path { nodes, edges } => {
                let mut parts: Vec<_> = nodes.iter().map(HashKey::from_value).collect();
                parts.extend(edges.iter().map(HashKey::from_value));
                HashKey::Composite(parts)
            }
            // CRDT counters are opaque keys; hash by total logical value.
            Value::GCounter(counts) => {
                // reason: CRDT counter values are practically small, wrap is acceptable for hashing
                #[allow(clippy::cast_possible_wrap)]
                HashKey::Int64(counts.values().copied().map(|v| v as i64).sum())
            }
            Value::OnCounter { pos, neg } => {
                // reason: CRDT counter values are practically small, wrap is acceptable for hashing
                #[allow(clippy::cast_possible_wrap)]
                // reason: GCounter values are small, sum fits i64
                let p: i64 = pos.values().copied().map(|v| v as i64).sum();
                // reason: GCounter values are small, sum fits i64
                #[allow(clippy::cast_possible_wrap)]
                let n: i64 = neg.values().copied().map(|v| v as i64).sum();
                HashKey::Int64(p - n)
            }
            _ => HashKey::Null,
        }
    }

    /// Creates a hash key from a column value at a given row.
    pub fn from_column(column: &ValueVector, row: usize) -> Option<Self> {
        column.get_value(row).map(|v| Self::from_value(&v))
    }

    /// The key of `value` for a join on `=` (see
    /// [`HashJoinOperator::with_value_equality_keys`]), or `None` for NULL,
    /// which `=` finds equal to nothing. Values that `=` finds equal have
    /// equal keys; values it tells apart may share one.
    #[must_use]
    pub fn for_equality(value: &Value) -> Option<Self> {
        (!value.is_null()).then(|| Self::equality_key(value))
    }

    /// The key of a value, by the rules of `=` (a filter's `values_equal`).
    fn equality_key(value: &Value) -> Self {
        match value {
            // Inside a list or map, NULL equals NULL.
            Value::Null => HashKey::Null,
            Value::Bool(b) => HashKey::Bool(*b),
            // `=` compares an integer with a float as this `f64`.
            Value::Int64(i) => Self::number_key(*i as f64),
            Value::Float64(f) => Self::number_key(*f),
            // A string equals a float it parses as (`'5.0' = 5.0`) and an
            // integer it parses as (`'05' = 5`, and such a string parses as
            // the float of that integer too), and otherwise only itself.
            Value::String(s) => match s.parse::<f64>() {
                Ok(number) if !number.is_nan() => Self::number_key(number),
                _ => HashKey::String(s.clone()),
            },
            Value::List(items) => {
                HashKey::Composite(items.iter().map(Self::equality_key).collect())
            }
            Value::Map(map) => HashKey::Composite(
                map.iter()
                    .map(|(key, value)| {
                        HashKey::Composite(vec![
                            HashKey::String(ArcStr::from(key.as_str())),
                            Self::equality_key(value),
                        ])
                    })
                    .collect(),
            ),
            Value::Path { nodes, edges } => HashKey::Composite(vec![
                HashKey::Composite(nodes.iter().map(Self::equality_key).collect()),
                HashKey::Composite(edges.iter().map(Self::equality_key).collect()),
            ]),
            // The elements compare as `f32`: zero equals negative zero.
            Value::Vector(items) => HashKey::Composite(
                items
                    .iter()
                    .map(|&item| {
                        let item = if item == 0.0 { 0.0_f32 } else { item };
                        HashKey::Int64(i64::from(item.to_bits()))
                    })
                    .collect(),
            ),
            // Temporal values, bytes and counters: `=` is their own equality,
            // which their hash agrees with (a time compares by its UTC
            // instant, a zoned datetime too).
            other => {
                let mut hasher = std::hash::DefaultHasher::new();
                HashableValue(other.clone()).hash(&mut hasher);
                HashKey::Int64(hasher.finish().cast_signed())
            }
        }
    }

    /// The key of a number: `=` compares numbers as `f64` within
    /// `f64::EPSILON`. Below 2 in magnitude distinct values can be that close,
    /// so they share a key; from 2 up, distinct values differ by at least
    /// `f64::EPSILON` and are never equal. NaN and the infinities equal
    /// nothing.
    fn number_key(number: f64) -> Self {
        let number = if number.abs() < 2.0 { 0.0 } else { number };
        HashKey::Int64(number.to_bits().cast_signed())
    }
}

/// How the keys of a [`HashJoinOperator`] match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyMatch {
    /// The same value of the same kind ([`HashKey::from_value`]).
    Exact,
    /// As `=` compares values ([`HashKey::for_equality`]).
    ValueEquality,
}

/// Hash join operator.
///
/// Builds a hash table from the build side (right) and probes with the probe side (left).
/// Efficient for equality joins on one or more columns. A row keeps the column
/// types of the rows it joins: a node or edge stays one.
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
    /// Output schema (combined from both sides). Used only for a side that has
    /// no rows: the build side of a left join that matched nothing, the probe
    /// side of the unmatched build rows of a right join.
    output_schema: Vec<LogicalType>,
    /// The column types of the probe chunks read so far.
    probe_types: ColumnTypes,
    /// The column types of the materialized build side.
    build_types: ColumnTypes,
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
    /// A condition on each pair of rows the keys match, beyond the keys (the
    /// WHERE of an OPTIONAL MATCH that reads both sides): a pair that fails it
    /// is no match, and a left row without one keeps nulls.
    residual: Option<Box<dyn JoinCondition>>,
    /// Whether a pair of the current probe row passed the residual condition.
    current_probe_kept: bool,
    /// How keys match.
    keys: KeyMatch,
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
            probe_types: ColumnTypes::default(),
            build_types: ColumnTypes::default(),
            hash_table: HashMap::new(),
            build_chunks: Vec::new(),
            build_complete: false,
            current_probe_chunk: None,
            current_probe_row: 0,
            current_match_position: 0,
            current_matches: Vec::new(),
            probe_matched: Vec::new(),
            build_matched: Vec::new(),
            residual: None,
            current_probe_kept: false,
            keys: KeyMatch::Exact,
            emitting_unmatched: false,
            unmatched_chunk_idx: 0,
            unmatched_row_idx: 0,
        }
    }

    /// Adds a condition each pair of rows the keys match must also meet: a
    /// pair that fails it is no match (so in a left join, a probe row none of
    /// whose pairs pass it keeps nulls).
    #[must_use]
    pub fn with_residual(mut self, condition: Box<dyn JoinCondition>) -> Self {
        self.residual = Some(condition);
        self
    }

    /// Matches the keys by value equality, as `=` compares them (see
    /// [`HashKey::for_equality`]): `5` meets `5.0` and `'5'`. Keys `=` tells
    /// apart may meet too (`5` meets `'5.0'`), so a filter above the join
    /// decides each pair. A row with NULL in a key column meets no row.
    ///
    /// For a probe side that does not write: an inner join without a build
    /// row to meet ends without reading it.
    #[must_use]
    pub fn with_value_equality_keys(mut self) -> Self {
        self.keys = KeyMatch::ValueEquality;
        self
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

            self.build_types.add(&chunk);
            self.build_chunks.push(chunk);
        }

        self.build_complete = true;
        Ok(())
    }

    /// The column types of the rows joined from `probe_chunk`: its own, then
    /// the build side's (only its own for a semi- or anti-join), so the copied
    /// values keep their types (see [`ColumnTypes`]). The declared schema gives
    /// the build side's types while it has no rows.
    fn output_types(&self, probe_chunk: &DataChunk) -> Vec<LogicalType> {
        let mut types = probe_chunk.column_types();
        if matches!(self.join_type, JoinType::Semi | JoinType::Anti) {
            return copied_column_types(&types, &self.output_schema);
        }
        if self.build_chunks.is_empty() {
            types.extend(
                self.output_schema
                    .iter()
                    .skip(probe_chunk.column_count())
                    .cloned(),
            );
        } else {
            types.extend_from_slice(self.build_types.types());
        }
        copied_column_types(&types, &self.output_schema)
    }

    /// The column types of the unmatched build rows of a right or full join:
    /// the probe side's (declared while it had no rows), then the build side's.
    fn unmatched_build_types(&self, probe_col_count: usize) -> Vec<LogicalType> {
        let mut types = if self.probe_types.types().is_empty() {
            self.output_schema
                .iter()
                .take(probe_col_count)
                .cloned()
                .collect()
        } else {
            self.probe_types.types().to_vec()
        };
        types.extend_from_slice(self.build_types.types());
        copied_column_types(&types, &self.output_schema)
    }

    /// Extracts a hash key from a chunk row. With value-equality keys, a row
    /// with NULL in a key column has the key `Null`, which meets no row.
    fn extract_key(
        &self,
        chunk: &DataChunk,
        row: usize,
        key_columns: &[usize],
    ) -> Result<HashKey, OperatorError> {
        if self.keys == KeyMatch::ValueEquality {
            let mut keys = Vec::with_capacity(key_columns.len());
            for &column in key_columns {
                let value = chunk
                    .column(column)
                    .ok_or_else(|| OperatorError::ColumnNotFound(format!("column {column}")))?
                    .get_value(row);
                match value.as_ref().and_then(HashKey::for_equality) {
                    Some(key) => keys.push(key),
                    None => return Ok(HashKey::Null),
                }
            }
            return Ok(if keys.len() == 1 {
                keys.swap_remove(0)
            } else {
                HashKey::Composite(keys)
            });
        }
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
                // Emit nulls for build side (left outer join case), in every
                // build column: the declared ones while the build side is empty
                let build_col_count = self.build_chunks.first().map_or_else(
                    || self.output_schema.len().saturating_sub(probe_col_count),
                    DataChunk::column_count,
                );
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
            self.probe_types.add(c);
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

        // Determine probe column count from schema or first probe chunk
        let probe_col_count = if !self.build_chunks.is_empty() {
            self.output_schema.len() - self.build_chunks[0].column_count()
        } else {
            0
        };
        let mut builder =
            DataChunkBuilder::with_capacity(&self.unmatched_build_types(probe_col_count), 2048);

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
        if self.keys == KeyMatch::ValueEquality
            && self.join_type == JoinType::Inner
            && self.hash_table.is_empty()
        {
            return Ok(None);
        }

        // Phase 3: Emit unmatched build rows (right/full outer join)
        if self.emitting_unmatched {
            return self.emit_unmatched_build();
        }

        // Phase 2: Probe. Each returned chunk holds rows of one probe chunk,
        // built in its column types.
        let mut chunk_builder: Option<DataChunkBuilder> = None;

        loop {
            // Get current probe chunk or fetch new one
            if self.current_probe_chunk.is_none() && !self.get_next_probe_chunk()? {
                // No more probe data
                if matches!(self.join_type, JoinType::Right | JoinType::Full) {
                    self.emitting_unmatched = true;
                    return self.emit_unmatched_build();
                }
                // A probe chunk's rows are returned when the chunk ends.
                return Ok(None);
            }

            // Invariant: current_probe_chunk is Some here - the guard at line 396 either
            // populates it via get_next_probe_chunk() or returns from the function
            let probe_chunk = self
                .current_probe_chunk
                .as_ref()
                .expect("probe chunk is Some: guard at line 396 ensures this");
            let builder = chunk_builder.get_or_insert_with(|| {
                DataChunkBuilder::with_capacity(&self.output_types(probe_chunk), 2048)
            });
            let probe_rows: Vec<usize> = probe_chunk.selected_indices().collect();

            while self.current_probe_row < probe_rows.len() {
                let probe_row = probe_rows[self.current_probe_row];

                // If we don't have current matches, look them up
                if self.current_matches.is_empty() && self.current_match_position == 0 {
                    let key = self.extract_key(probe_chunk, probe_row, &self.probe_keys)?;
                    // A NULL value-equality key meets nothing, not even the
                    // NULL keys an outer join keeps on the build side.
                    let candidates = if self.keys == KeyMatch::ValueEquality && key == HashKey::Null
                    {
                        None
                    } else {
                        self.hash_table.get(&key)
                    };

                    // Handle semi/anti joins differently: a probe row has a match
                    // when a pair with the same key passes the residual condition.
                    let has_match = || {
                        candidates.is_some_and(|candidates| {
                            self.residual.as_ref().is_none_or(|residual| {
                                candidates.iter().any(|&(chunk_idx, row)| {
                                    residual.evaluate(
                                        probe_chunk,
                                        probe_row,
                                        &self.build_chunks[chunk_idx],
                                        row,
                                    )
                                })
                            })
                        })
                    };
                    match self.join_type {
                        JoinType::Semi => {
                            if has_match() {
                                // Emit probe row only
                                for col_idx in 0..probe_chunk.column_count() {
                                    if let (Some(src_col), Some(dst_col)) =
                                        (probe_chunk.column(col_idx), builder.column_mut(col_idx))
                                        && let Some(value) = src_col.get_value(probe_row)
                                    {
                                        dst_col.push_value(value);
                                    }
                                }
                                builder.advance_row();
                            }
                            self.current_probe_row += 1;
                            continue;
                        }
                        JoinType::Anti => {
                            if !has_match() {
                                // Emit probe row only
                                for col_idx in 0..probe_chunk.column_count() {
                                    if let (Some(src_col), Some(dst_col)) =
                                        (probe_chunk.column(col_idx), builder.column_mut(col_idx))
                                        && let Some(value) = src_col.get_value(probe_row)
                                    {
                                        dst_col.push_value(value);
                                    }
                                }
                                builder.advance_row();
                            }
                            self.current_probe_row += 1;
                            continue;
                        }
                        _ => {
                            self.current_matches = candidates.cloned().unwrap_or_default();
                            self.current_probe_kept = false;
                        }
                    }
                }

                // Process matches
                if self.current_matches.is_empty() {
                    // No matches - for left/full outer join, emit with nulls
                    if matches!(self.join_type, JoinType::Left | JoinType::Full) {
                        self.produce_output_row(builder, probe_chunk, probe_row, None, None)?;
                    }
                    self.current_probe_row += 1;
                    self.current_match_position = 0;
                } else {
                    // Process each match
                    while self.current_match_position < self.current_matches.len() {
                        let (build_chunk_idx, build_row) =
                            self.current_matches[self.current_match_position];
                        let build_chunk = &self.build_chunks[build_chunk_idx];

                        if let Some(residual) = &self.residual
                            && !residual.evaluate(probe_chunk, probe_row, build_chunk, build_row)
                        {
                            self.current_match_position += 1;
                            continue;
                        }
                        self.current_probe_kept = true;

                        // Mark as matched for outer joins
                        if matches!(self.join_type, JoinType::Left | JoinType::Full)
                            && probe_row < self.probe_matched.len()
                        {
                            self.probe_matched[probe_row] = true;
                        }
                        if matches!(self.join_type, JoinType::Right | JoinType::Full)
                            && build_chunk_idx < self.build_matched.len()
                            && build_row < self.build_matched[build_chunk_idx].len()
                        {
                            self.build_matched[build_chunk_idx][build_row] = true;
                        }

                        self.produce_output_row(
                            builder,
                            probe_chunk,
                            probe_row,
                            Some(build_chunk),
                            Some(build_row),
                        )?;

                        self.current_match_position += 1;

                        if builder.is_full() {
                            return Ok(chunk_builder.take().map(DataChunkBuilder::finish));
                        }
                    }

                    // Done with this probe row: without a pair that passed the
                    // residual condition, a left or full join keeps it with nulls.
                    if self.residual.is_some()
                        && !self.current_probe_kept
                        && matches!(self.join_type, JoinType::Left | JoinType::Full)
                    {
                        self.produce_output_row(builder, probe_chunk, probe_row, None, None)?;
                    }
                    self.current_probe_row += 1;
                    self.current_matches.clear();
                    self.current_match_position = 0;
                }

                if builder.is_full() {
                    return Ok(chunk_builder.take().map(DataChunkBuilder::finish));
                }
            }

            // Done with current probe chunk
            self.current_probe_chunk = None;
            self.current_probe_row = 0;

            if let Some(done) = chunk_builder.take()
                && done.row_count() > 0
            {
                return Ok(Some(done.finish()));
            }
        }
    }

    fn reset(&mut self) {
        self.probe_side.reset();
        self.build_side.reset();
        self.probe_types = ColumnTypes::default();
        self.build_types = ColumnTypes::default();
        self.hash_table.clear();
        self.build_chunks.clear();
        self.build_complete = false;
        self.current_probe_chunk = None;
        self.current_probe_row = 0;
        self.current_match_position = 0;
        self.current_matches.clear();
        self.probe_matched.clear();
        self.build_matched.clear();
        self.current_probe_kept = false;
        self.emitting_unmatched = false;
        self.unmatched_chunk_idx = 0;
        self.unmatched_row_idx = 0;
    }

    fn name(&self) -> &'static str {
        "HashJoin"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Nested loop join operator.
///
/// Performs a cartesian product of both sides, filtering by the join condition.
/// Less efficient than hash join but supports any join condition. A row keeps
/// the column types of the rows it joins: a node or edge stays one.
pub struct NestedLoopJoinOperator {
    /// Left side operator.
    left: Box<dyn Operator>,
    /// Right side operator.
    right: Box<dyn Operator>,
    /// Join condition predicate (if any).
    condition: Option<Box<dyn JoinCondition>>,
    /// Join type.
    join_type: JoinType,
    /// Output schema. Only its right-side part is used: for the right side's
    /// columns of a left join when the right side has no rows.
    output_schema: Vec<LogicalType>,
    /// The column types of the materialized right side.
    right_types: ColumnTypes,
    /// Materialized right side chunks.
    right_chunks: Vec<DataChunk>,
    /// Whether the right side is materialized.
    right_materialized: bool,
    /// Whether the whole left side is read before the right side.
    left_first: bool,
    /// The left side's chunks, when it is read first.
    left_chunks: Option<VecDeque<DataChunk>>,
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

/// A condition given by a predicate over the joined row: the left row's
/// columns, then the right row's, numbered as the predicate's variable
/// columns number them.
pub struct JoinedRowCondition {
    predicate: Box<dyn super::filter::Predicate>,
    /// The joined row being checked, reused from pair to pair while the
    /// column types stay the same.
    scratch: parking_lot::Mutex<Option<DataChunk>>,
}

impl JoinedRowCondition {
    /// Creates the condition from a predicate over the joined row.
    pub fn new(predicate: Box<dyn super::filter::Predicate>) -> Self {
        Self {
            predicate,
            scratch: parking_lot::Mutex::new(None),
        }
    }
}

impl JoinCondition for JoinedRowCondition {
    fn evaluate(
        &self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> bool {
        let sources: Vec<(&ValueVector, usize)> =
            [(left_chunk, left_row), (right_chunk, right_row)]
                .into_iter()
                .flat_map(|(chunk, row)| chunk.columns().iter().map(move |column| (column, row)))
                .collect();
        let mut scratch = self.scratch.lock();
        let fits = scratch.as_ref().is_some_and(|joined| {
            joined.column_count() == sources.len()
                && joined
                    .columns()
                    .iter()
                    .zip(&sources)
                    .all(|(column, (source, _))| column.data_type() == source.data_type())
        });
        if !fits {
            *scratch = Some(DataChunk::new(
                sources
                    .iter()
                    .map(|(source, _)| ValueVector::with_capacity(source.data_type().clone(), 1))
                    .collect(),
            ));
        }
        let joined = scratch.as_mut().expect("the joined row was just built");
        for (index, (source, row)) in sources.iter().enumerate() {
            let column = joined
                .column_mut(index)
                .expect("the joined row has a column per source column");
            column.clear();
            source.copy_row_to(*row, column);
        }
        joined.set_count(1);
        self.predicate.evaluate(joined, 0)
    }
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
            right_types: ColumnTypes::default(),
            right_chunks: Vec::new(),
            right_materialized: false,
            left_first: false,
            left_chunks: None,
            current_left_chunk: None,
            current_left_row: 0,
            current_right_chunk: 0,
            current_right_row: 0,
            current_left_matched: false,
        }
    }

    /// Reads the whole left side before the right side, so that the right
    /// side sees what the left side wrote: in `INSERT (:N) WITH 1 AS x
    /// MATCH (n:N)` the scan of `n` runs after the insert.
    #[must_use]
    pub fn with_left_first(mut self) -> Self {
        self.left_first = true;
        self
    }

    /// The next left chunk, from the buffer when the left side was read first.
    fn next_left(&mut self) -> OperatorResult {
        match &mut self.left_chunks {
            Some(chunks) => Ok(chunks.pop_front()),
            None => self.left.next(),
        }
    }

    /// Materializes the right side.
    fn materialize_right(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.right.next()? {
            self.right_types.add(&chunk);
            self.right_chunks.push(chunk);
        }
        self.right_materialized = true;
        Ok(())
    }

    /// The column types of the rows joined from `left_chunk`: its own, then
    /// the right side's, so the copied values keep their types (see
    /// [`ColumnTypes`]). The declared schema gives the right side's types
    /// while it has no rows.
    fn output_types(&self, left_chunk: &DataChunk) -> Vec<LogicalType> {
        let mut types = left_chunk.column_types();
        if self.right_chunks.is_empty() {
            types.extend(
                self.output_schema
                    .iter()
                    .skip(left_chunk.column_count())
                    .cloned(),
            );
        } else {
            types.extend_from_slice(self.right_types.types());
        }
        copied_column_types(&types, &self.output_schema)
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
        if self.left_first && self.left_chunks.is_none() {
            let mut chunks = VecDeque::new();
            while let Some(chunk) = self.left.next()? {
                chunks.push_back(chunk);
            }
            self.left_chunks = Some(chunks);
        }

        // Materialize right side
        if !self.right_materialized {
            self.materialize_right()?;
        }

        // If right side is empty and not a left outer join, return nothing
        if self.right_chunks.is_empty() && !matches!(self.join_type, JoinType::Left) {
            return Ok(None);
        }

        loop {
            // Get current left chunk
            if self.current_left_chunk.is_none() {
                self.current_left_chunk = self.next_left()?;
                self.current_left_row = 0;
                self.current_right_chunk = 0;
                self.current_right_row = 0;

                if self.current_left_chunk.is_none() {
                    // No more left data
                    return Ok(None);
                }
            }

            let left_chunk = self
                .current_left_chunk
                .as_ref()
                .expect("left chunk is Some: loaded in loop above");
            let left_rows: Vec<usize> = left_chunk.selected_indices().collect();
            // Each returned chunk holds rows of one left chunk, built in its
            // column types.
            let mut builder = DataChunkBuilder::with_capacity(&self.output_types(left_chunk), 2048);

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
        self.right_types = ColumnTypes::default();
        self.right_chunks.clear();
        self.right_materialized = false;
        self.left_chunks = None;
        self.current_left_chunk = None;
        self.current_left_row = 0;
        self.current_right_chunk = 0;
        self.current_right_row = 0;
        self.current_left_matched = false;
    }

    fn name(&self) -> &'static str {
        "NestedLoopJoin"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
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

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
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

        results.sort_unstable();
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

    /// A residual condition decides which key matches count: a left row none
    /// of whose pairs pass it keeps nulls (once), the others keep the pairs
    /// that pass. The condition reads the joined row: left columns, then right.
    #[test]
    fn test_hash_join_left_outer_with_a_residual() {
        struct LeftIsNot2;
        impl super::super::filter::Predicate for LeftIsNot2 {
            fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
                assert_eq!(
                    chunk.column_count(),
                    2,
                    "the left column, then the right one"
                );
                chunk.column(0).unwrap().get_int64(row) != Some(2)
            }
        }

        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 3])]);
        let right = MockOperator::new(vec![create_int_chunk(&[1, 2, 2, 3])]);
        let mut join = HashJoinOperator::new(
            Box::new(left),
            Box::new(right),
            vec![0],
            vec![0],
            JoinType::Left,
            vec![LogicalType::Int64, LogicalType::Int64],
        )
        .with_residual(Box::new(JoinedRowCondition::new(Box::new(LeftIsNot2))));

        let mut results = Vec::new();
        while let Some(chunk) = join.next().unwrap() {
            for row in chunk.selected_indices() {
                let left_val = chunk.column(0).unwrap().get_int64(row).unwrap();
                let right_val = chunk.column(1).unwrap().get_int64(row);
                results.push((left_val, right_val));
            }
        }
        results.sort_by_key(|(l, _)| *l);
        assert_eq!(results, [(1, Some(1)), (2, None), (3, Some(3))]);

        // A semi-join keeps the rows with a pair that passes it, an anti-join
        // the others.
        for (join_type, expected) in [(JoinType::Semi, vec![1, 3]), (JoinType::Anti, vec![2])] {
            let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 3])]);
            let right = MockOperator::new(vec![create_int_chunk(&[1, 2, 2, 3])]);
            let mut join = HashJoinOperator::new(
                Box::new(left),
                Box::new(right),
                vec![0],
                vec![0],
                join_type,
                vec![LogicalType::Int64],
            )
            .with_residual(Box::new(JoinedRowCondition::new(Box::new(LeftIsNot2))));
            let mut kept = Vec::new();
            while let Some(chunk) = join.next().unwrap() {
                for row in chunk.selected_indices() {
                    kept.push(chunk.column(0).unwrap().get_int64(row).unwrap());
                }
            }
            kept.sort_unstable();
            assert_eq!(kept, expected, "{join_type:?}");
        }
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

        results.sort_unstable();
        assert_eq!(results, vec![(1, 10), (1, 20), (2, 10), (2, 20)]);
    }

    /// A cross join keeps the column types of both sides, whatever schema it
    /// declares: an edge stays an edge, so its properties are not read from
    /// the node with the same ID.
    #[test]
    fn a_cross_join_keeps_the_column_types() {
        use grafeo_common::types::{EdgeId, NodeId};

        let mut left = DataChunkBuilder::new(&[LogicalType::Edge, LogicalType::Node]);
        for id in [3_u64, 5] {
            left.column_mut(0).unwrap().push_edge_id(EdgeId::new(id));
            left.column_mut(1)
                .unwrap()
                .push_node_id(NodeId::new(id + 100));
            left.advance_row();
        }
        let mut right = DataChunkBuilder::new(&[LogicalType::Node]);
        right.column_mut(0).unwrap().push_node_id(NodeId::new(7));
        right.advance_row();

        let mut join = NestedLoopJoinOperator::new(
            Box::new(MockOperator::new(vec![left.finish()])),
            Box::new(MockOperator::new(vec![right.finish()])),
            None,
            JoinType::Cross,
            vec![LogicalType::Any, LogicalType::Any, LogicalType::Node],
        );

        let chunk = join.next().unwrap().unwrap();
        assert_eq!(
            chunk.column_types(),
            [LogicalType::Edge, LogicalType::Node, LogicalType::Node]
        );
        let rows: Vec<(u64, u64, u64)> = chunk
            .selected_indices()
            .map(|row| {
                (
                    chunk.column(0).unwrap().get_edge_id(row).unwrap().as_u64(),
                    chunk.column(1).unwrap().get_node_id(row).unwrap().as_u64(),
                    chunk.column(2).unwrap().get_node_id(row).unwrap().as_u64(),
                )
            })
            .collect();
        assert_eq!(rows, [(3, 103, 7), (5, 105, 7)]);
        assert!(join.next().unwrap().is_none());
    }

    /// Probe rows (node, key) for keys 1 and 2, nodes 101 and 102.
    fn node_probe_chunk() -> DataChunk {
        use grafeo_common::types::NodeId;

        let mut probe = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Int64]);
        for key in [1_u64, 2] {
            probe
                .column_mut(0)
                .unwrap()
                .push_node_id(NodeId::new(key + 100));
            probe
                .column_mut(1)
                .unwrap()
                .push_int64(i64::try_from(key).unwrap());
            probe.advance_row();
        }
        probe.finish()
    }

    /// A hash join keeps the column types of both sides (the declared schema
    /// says `Any`): a node stays a node and an edge an edge.
    #[test]
    fn a_hash_join_keeps_the_column_types() {
        use grafeo_common::types::EdgeId;

        let mut build = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Edge]);
        build.column_mut(0).unwrap().push_int64(2);
        build.column_mut(1).unwrap().push_edge_id(EdgeId::new(7));
        build.advance_row();

        let mut join = HashJoinOperator::new(
            Box::new(MockOperator::new(vec![node_probe_chunk()])),
            Box::new(MockOperator::new(vec![build.finish()])),
            vec![1],
            vec![0],
            JoinType::Inner,
            vec![LogicalType::Any; 4],
        );

        let chunk = join.next().unwrap().unwrap();
        assert_eq!(
            chunk.column_types(),
            [
                LogicalType::Node,
                LogicalType::Int64,
                LogicalType::Int64,
                LogicalType::Edge
            ]
        );
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(
            chunk.column(0).unwrap().get_node_id(0).unwrap().as_u64(),
            102
        );
        assert!(chunk.column(0).unwrap().get_edge_id(0).is_none());
        assert_eq!(chunk.column(3).unwrap().get_edge_id(0).unwrap().as_u64(), 7);
        assert!(chunk.column(3).unwrap().get_node_id(0).is_none());
        assert!(join.next().unwrap().is_none());
    }

    /// A semi-join returns its probe rows in their own types.
    #[test]
    fn a_semi_join_keeps_the_probe_types() {
        let mut build = DataChunkBuilder::new(&[LogicalType::Int64]);
        build.column_mut(0).unwrap().push_int64(1);
        build.advance_row();

        let mut join = HashJoinOperator::new(
            Box::new(MockOperator::new(vec![node_probe_chunk()])),
            Box::new(MockOperator::new(vec![build.finish()])),
            vec![1],
            vec![0],
            JoinType::Semi,
            vec![LogicalType::Any; 2],
        );

        let chunk = join.next().unwrap().unwrap();
        assert_eq!(
            chunk.column_types(),
            [LogicalType::Node, LogicalType::Int64]
        );
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(
            chunk.column(0).unwrap().get_node_id(0).unwrap().as_u64(),
            101
        );
    }

    /// A left join with no build rows takes the build side's types from the
    /// declared schema and the probe side's from its rows.
    #[test]
    fn a_left_join_without_build_rows_keeps_the_probe_types() {
        let mut join = HashJoinOperator::new(
            Box::new(MockOperator::new(vec![node_probe_chunk()])),
            Box::new(MockOperator::new(vec![])),
            vec![1],
            vec![0],
            JoinType::Left,
            vec![
                LogicalType::Any,
                LogicalType::Any,
                LogicalType::Int64,
                LogicalType::Edge,
            ],
        );

        let chunk = join.next().unwrap().unwrap();
        assert_eq!(
            chunk.column_types(),
            [
                LogicalType::Node,
                LogicalType::Int64,
                LogicalType::Int64,
                LogicalType::Edge
            ]
        );
        assert_eq!(chunk.row_count(), 2);
        assert!(chunk.column(3).unwrap().is_null(0));
        assert!(chunk.column(3).unwrap().is_null(1));
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

        results.sort_unstable();
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

        results.sort_unstable();
        assert_eq!(results, vec![1, 3]);
    }

    #[test]
    fn test_hash_key_from_map() {
        use grafeo_common::types::{PropertyKey, Value};
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::new("key"), Value::Int64(42));
        let v = Value::Map(Arc::new(map));
        let key = HashKey::from_value(&v);
        // BTreeMap iterates in ascending key order, result is a Composite
        assert!(matches!(key, HashKey::Composite(_)));

        // Two maps with the same content produce the same hash key
        let mut map2 = BTreeMap::new();
        map2.insert(PropertyKey::new("key"), Value::Int64(42));
        let v2 = Value::Map(Arc::new(map2));
        assert_eq!(HashKey::from_value(&v), HashKey::from_value(&v2));
    }

    #[test]
    fn test_hash_key_from_map_empty() {
        use grafeo_common::types::Value;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let v = Value::Map(Arc::new(BTreeMap::new()));
        let key = HashKey::from_value(&v);
        assert_eq!(key, HashKey::Composite(vec![]));
    }

    #[test]
    fn test_hash_key_from_gcounter() {
        use grafeo_common::types::Value;
        use std::collections::HashMap;
        use std::sync::Arc;

        let mut counts = HashMap::new();
        counts.insert("node-a".to_string(), 5u64);
        counts.insert("node-b".to_string(), 3u64);
        let v = Value::GCounter(Arc::new(counts));
        // GCounter hashes to sum of all values (5 + 3 = 8)
        assert_eq!(HashKey::from_value(&v), HashKey::Int64(8));
    }

    #[test]
    fn test_hash_key_from_gcounter_empty() {
        use grafeo_common::types::Value;
        use std::collections::HashMap;
        use std::sync::Arc;

        let v = Value::GCounter(Arc::new(HashMap::new()));
        assert_eq!(HashKey::from_value(&v), HashKey::Int64(0));
    }

    #[test]
    fn test_hash_key_from_oncounter() {
        use grafeo_common::types::Value;
        use std::collections::HashMap;
        use std::sync::Arc;

        let mut pos = HashMap::new();
        pos.insert("node-a".to_string(), 10u64);
        let mut neg = HashMap::new();
        neg.insert("node-a".to_string(), 3u64);
        let v = Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        };
        // OnCounter hashes to pos_sum - neg_sum = 10 - 3 = 7
        assert_eq!(HashKey::from_value(&v), HashKey::Int64(7));
    }

    #[test]
    fn test_hash_key_from_oncounter_balanced() {
        use grafeo_common::types::Value;
        use std::collections::HashMap;
        use std::sync::Arc;

        let mut pos = HashMap::new();
        pos.insert("r".to_string(), 5u64);
        let mut neg = HashMap::new();
        neg.insert("r".to_string(), 5u64);
        let v = Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        };
        assert_eq!(HashKey::from_value(&v), HashKey::Int64(0));
    }

    #[test]
    fn test_hash_join_into_any() {
        let left = MockOperator::new(vec![]);
        let right = MockOperator::new(vec![]);
        let op = HashJoinOperator::new(
            Box::new(left),
            Box::new(right),
            vec![0],
            vec![0],
            JoinType::Inner,
            vec![LogicalType::Int64, LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<HashJoinOperator>().is_ok());
    }

    #[test]
    fn test_nested_loop_join_into_any() {
        let left = MockOperator::new(vec![]);
        let right = MockOperator::new(vec![]);
        let op = NestedLoopJoinOperator::new(
            Box::new(left),
            Box::new(right),
            None,
            JoinType::Cross,
            vec![LogicalType::Int64, LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<NestedLoopJoinOperator>().is_ok());
    }

    #[test]
    fn test_hash_key_ord_same_variant() {
        use std::cmp::Ordering;

        assert_eq!(HashKey::Null.cmp(&HashKey::Null), Ordering::Equal);
        assert_eq!(
            HashKey::Bool(false).cmp(&HashKey::Bool(true)),
            Ordering::Less
        );
        assert_eq!(
            HashKey::Bool(true).cmp(&HashKey::Bool(false)),
            Ordering::Greater
        );
        assert_eq!(HashKey::Int64(1).cmp(&HashKey::Int64(2)), Ordering::Less);
        assert_eq!(HashKey::Int64(5).cmp(&HashKey::Int64(5)), Ordering::Equal);
        assert_eq!(
            HashKey::String(arcstr::literal!("a")).cmp(&HashKey::String(arcstr::literal!("b"))),
            Ordering::Less,
        );
        assert_eq!(
            HashKey::Bytes(vec![1, 2]).cmp(&HashKey::Bytes(vec![1, 3])),
            Ordering::Less,
        );
        assert_eq!(
            HashKey::Composite(vec![HashKey::Int64(1)])
                .cmp(&HashKey::Composite(vec![HashKey::Int64(2)])),
            Ordering::Less,
        );
    }

    #[test]
    fn test_hash_key_ord_cross_variant() {
        use std::cmp::Ordering;

        // Null < Bool < Int64 < String < Bytes < Composite
        assert_eq!(HashKey::Null.cmp(&HashKey::Bool(false)), Ordering::Less);
        assert_eq!(HashKey::Bool(true).cmp(&HashKey::Null), Ordering::Greater);
        assert_eq!(HashKey::Bool(false).cmp(&HashKey::Int64(0)), Ordering::Less);
        assert_eq!(
            HashKey::Int64(0).cmp(&HashKey::Bool(false)),
            Ordering::Greater
        );
        assert_eq!(
            HashKey::Int64(0).cmp(&HashKey::String(arcstr::literal!("a"))),
            Ordering::Less,
        );
        assert_eq!(
            HashKey::String(arcstr::literal!("a")).cmp(&HashKey::Int64(0)),
            Ordering::Greater,
        );
        assert_eq!(
            HashKey::String(arcstr::literal!("a")).cmp(&HashKey::Bytes(vec![1])),
            Ordering::Less,
        );
        assert_eq!(
            HashKey::Bytes(vec![1]).cmp(&HashKey::String(arcstr::literal!("a"))),
            Ordering::Greater,
        );
        assert_eq!(
            HashKey::Bytes(vec![1]).cmp(&HashKey::Composite(vec![])),
            Ordering::Less,
        );
        assert_eq!(
            HashKey::Composite(vec![]).cmp(&HashKey::Bytes(vec![1])),
            Ordering::Greater,
        );
    }

    #[test]
    fn test_hash_key_partial_ord() {
        // PartialOrd delegates to Ord, just verify it returns Some
        assert!(HashKey::Null.partial_cmp(&HashKey::Int64(1)).is_some());
        assert!(HashKey::Int64(1).partial_cmp(&HashKey::Int64(2)).is_some());
    }

    // `=` is evaluated by a filter over an LPG store.
    #[cfg(feature = "lpg")]
    mod value_equality {
        use std::collections::BTreeMap;
        use std::sync::Arc;

        use grafeo_common::types::{
            Date, Duration, PropertyKey, Time, Timestamp, Value, ZonedDatetime,
        };

        use super::*;
        use crate::execution::operators::filter::{
            BinaryFilterOp, ExpressionPredicate, FilterExpression,
        };
        use crate::graph::GraphStoreSearch;
        use crate::graph::lpg::LpgStore;

        fn list(items: Vec<Value>) -> Value {
            Value::List(items.into())
        }

        fn map(key: &str, value: Value) -> Value {
            Value::Map(Arc::new(BTreeMap::from([(PropertyKey::new(key), value)])))
        }

        fn time(text: &str) -> Value {
            Value::Time(Time::parse(text).expect("a time"))
        }

        fn zoned(text: &str) -> Value {
            Value::ZonedDatetime(ZonedDatetime::parse(text).expect("a zoned datetime"))
        }

        /// Values that `=` compares in more than one way: integers and floats
        /// near zero, near 2 and past 2^53, strings that read as numbers,
        /// nulls inside lists, maps, times and datetimes at one instant in
        /// different offsets, and vectors with a negative zero.
        fn values() -> Vec<Value> {
            let two_to_53 = 1_i64 << 53;
            let mut values: Vec<Value> = [0, 1, -1, 2, 5, 7, 1000, two_to_53, two_to_53 + 1]
                .into_iter()
                .chain([i64::MAX, i64::MIN])
                .map(Value::Int64)
                .collect();
            values.extend(
                [
                    1.0,
                    0.0,
                    -0.0,
                    1e-17,
                    -1e-17,
                    0.5,
                    1.5,
                    2.0,
                    1.999_999_999_999_999_8,
                    0.999_999_999_999_999_9,
                    -0.999_999_999_999_999_9,
                    5.0,
                    1000.0,
                    // 2^53 and 2^53 + 1 round to the same float
                    two_to_53 as f64,
                    (two_to_53 + 1) as f64,
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                ]
                .map(Value::Float64),
            );
            values.extend(
                [
                    "5", "05", "+5", "5.0", "-0", "0", "1e3", "NaN", "inf", "abc", "", " 5",
                ]
                .map(Value::from),
            );
            values.extend([
                Value::Bool(true),
                Value::Bool(false),
                list(vec![Value::Int64(1)]),
                list(vec![Value::Float64(1.0)]),
                list(vec![Value::from("1")]),
                list(vec![Value::Null]),
                list(vec![Value::Int64(1), Value::Null]),
                list(vec![Value::Float64(1.0), Value::Null]),
                list(Vec::new()),
                map("a", Value::Int64(1)),
                map("a", Value::Float64(1.0)),
                map("b", Value::Int64(1)),
                Value::Date(Date::parse("2026-10-07").expect("a date")),
                Value::Date(Date::parse("2026-10-08").expect("a date")),
                Value::Timestamp(Timestamp::from_micros(1_000_000)),
                time("14:00:00+01:00"),
                time("13:00:00Z"),
                time("13:00:00"),
                time("14:00:00"),
                zoned("2026-10-07T14:00:00+01:00"),
                zoned("2026-10-07T13:00:00Z"),
                zoned("2026-10-07T13:00:00+01:00"),
                Value::Duration(Duration::new(0, 1, 0)),
                Value::Duration(Duration::new(0, 0, 86_400_000_000_000)),
                Value::Bytes(vec![1_u8, 2].into()),
                Value::Bytes(vec![1_u8, 3].into()),
                Value::Vector(vec![0.0_f32].into()),
                Value::Vector(vec![-0.0_f32].into()),
                Value::Vector(vec![1.0_f32].into()),
                Value::Path {
                    nodes: vec![Value::Int64(1), Value::Float64(2.0)].into(),
                    edges: vec![Value::Int64(3)].into(),
                },
                Value::Path {
                    nodes: vec![Value::Float64(1.0), Value::Int64(2)].into(),
                    edges: vec![Value::Float64(3.0)].into(),
                },
            ]);
            values
        }

        /// Whether `a = b` is TRUE, evaluated as a filter evaluates it.
        fn equal(a: &Value, b: &Value) -> bool {
            let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
            let predicate = ExpressionPredicate::new(
                FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(a.clone())),
                    op: BinaryFilterOp::Eq,
                    right: Box::new(FilterExpression::Literal(b.clone())),
                },
                std::collections::HashMap::new(),
                store,
            );
            let chunk = DataChunkBuilder::new(&[LogicalType::Int64]).finish();
            predicate.eval_at(&chunk, 0) == Some(Value::Bool(true))
        }

        #[test]
        fn values_that_compare_equal_have_equal_keys() {
            let values = values();
            for a in &values {
                for b in &values {
                    if equal(a, b) {
                        assert_eq!(
                            HashKey::for_equality(a),
                            HashKey::for_equality(b),
                            "{a:?} = {b:?}"
                        );
                    }
                }
            }
        }

        /// The pairs that make the keys above more than exact: `=` finds
        /// them equal, so the invariant covers them.
        #[test]
        fn equality_holds_across_kinds_and_forms() {
            let two_to_53 = 1_i64 << 53;
            let pairs = [
                (Value::from("5"), Value::Int64(5)),
                (Value::from("05"), Value::Int64(5)),
                (Value::from("+5"), Value::Int64(5)),
                (Value::from("5"), Value::Float64(5.0)),
                (Value::from("5.0"), Value::Float64(5.0)),
                (Value::from("1e3"), Value::Float64(1000.0)),
                (Value::from("-0"), Value::Float64(1e-17)),
                (Value::from("NaN"), Value::from("NaN")),
                (Value::Int64(5), Value::Float64(5.0)),
                (Value::Int64(0), Value::Float64(1e-17)),
                (Value::Float64(1e-17), Value::Float64(-1e-17)),
                (Value::Float64(0.0), Value::Float64(-0.0)),
                (Value::Float64(0.999_999_999_999_999_9), Value::Int64(1)),
                (Value::Float64(0.999_999_999_999_999_9), Value::Float64(1.0)),
                (Value::Float64(-0.999_999_999_999_999_9), Value::Int64(-1)),
                (
                    Value::Int64(two_to_53 + 1),
                    Value::Float64(two_to_53 as f64),
                ),
                (list(vec![Value::Null]), list(vec![Value::Null])),
                (
                    list(vec![Value::Int64(1), Value::Null]),
                    list(vec![Value::Float64(1.0), Value::Null]),
                ),
                (list(vec![Value::from("1")]), list(vec![Value::Int64(1)])),
                (map("a", Value::Int64(1)), map("a", Value::Float64(1.0))),
                (time("14:00:00+01:00"), time("13:00:00Z")),
                (
                    zoned("2026-10-07T14:00:00+01:00"),
                    zoned("2026-10-07T13:00:00Z"),
                ),
                (
                    Value::Vector(vec![0.0_f32].into()),
                    Value::Vector(vec![-0.0_f32].into()),
                ),
            ];
            for (a, b) in pairs {
                assert!(equal(&a, &b), "{a:?} = {b:?}");
            }
            // And these are not: a key may still be shared (see the keys).
            for (a, b) in [
                (Value::from("5.0"), Value::Int64(5)),
                (Value::Int64(two_to_53), Value::Int64(two_to_53 + 1)),
                (Value::Float64(f64::NAN), Value::Float64(f64::NAN)),
                (Value::Float64(f64::INFINITY), Value::Float64(f64::INFINITY)),
                (Value::Float64(2.0), Value::Float64(1.999_999_999_999_999_8)),
                (time("13:00:00"), time("13:00:00Z")),
                (Value::Null, Value::Null),
            ] {
                assert!(!equal(&a, &b), "{a:?} <> {b:?}");
            }
        }

        /// NULL equals nothing, so it has no key; a NULL inside a list is a
        /// value like any other.
        #[test]
        fn null_has_no_key() {
            assert_eq!(HashKey::for_equality(&Value::Null), None);
            assert!(HashKey::for_equality(&list(vec![Value::Null])).is_some());
        }

        /// Values `=` tells apart get different keys here, so the join does
        /// not pair every row with every other.
        #[test]
        fn values_far_apart_have_different_keys() {
            for (a, b) in [
                (Value::Int64(2), Value::Int64(3)),
                (Value::Int64(2), Value::Float64(2.5)),
                (Value::Float64(1e3), Value::Float64(1e3 + 1.0)),
                (Value::from("abc"), Value::from("abd")),
                (Value::from("abc"), Value::Int64(0)),
                (Value::Bool(true), Value::Bool(false)),
                (list(vec![Value::Int64(1)]), list(vec![Value::Int64(2)])),
                (map("a", Value::Int64(1)), map("b", Value::Int64(1))),
                (time("13:00:00Z"), time("14:00:00Z")),
                (
                    zoned("2026-10-07T13:00:00+01:00"),
                    zoned("2026-10-07T13:00:00Z"),
                ),
                (
                    Value::Vector(vec![0.0_f32].into()),
                    Value::Vector(vec![1.0_f32].into()),
                ),
            ] {
                assert_ne!(
                    HashKey::for_equality(&a),
                    HashKey::for_equality(&b),
                    "{a:?} and {b:?}"
                );
            }
        }

        /// A chunk with one column of `Any` values, one per row.
        fn values_chunk(values: &[Value]) -> DataChunk {
            let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
            for value in values {
                builder.column_mut(0).unwrap().push_value(value.clone());
                builder.advance_row();
            }
            builder.finish()
        }

        /// The pairs of key values an inner join with value-equality keys
        /// returns, in order.
        fn joined(probe: &[Value], build: &[Value]) -> Vec<(Value, Value)> {
            let mut join = HashJoinOperator::new(
                Box::new(MockOperator::new(vec![values_chunk(probe)])),
                Box::new(MockOperator::new(vec![values_chunk(build)])),
                vec![0],
                vec![0],
                JoinType::Inner,
                vec![LogicalType::Any, LogicalType::Any],
            )
            .with_value_equality_keys();
            let mut pairs = Vec::new();
            while let Some(chunk) = join.next().unwrap() {
                for row in chunk.selected_indices() {
                    pairs.push((
                        chunk.column(0).unwrap().get_value(row).unwrap(),
                        chunk.column(1).unwrap().get_value(row).unwrap(),
                    ));
                }
            }
            pairs
        }

        /// 5 meets 5.0 and '5', and '5.0', which `=` rejects (the filter
        /// above the join does): the keys pick the candidates, each probe row
        /// with its build rows in build order. NULL meets nothing.
        #[test]
        fn a_join_on_value_equality_keys_pairs_candidates_in_order() {
            let probe = [Value::Int64(5), Value::Null, Value::Int64(7)];
            let build = [
                Value::Float64(5.0),
                Value::from("5"),
                Value::Null,
                Value::from("5.0"),
                Value::Int64(7),
                Value::from("x"),
            ];
            assert_eq!(
                joined(&probe, &build),
                [
                    (Value::Int64(5), Value::Float64(5.0)),
                    (Value::Int64(5), Value::from("5")),
                    (Value::Int64(5), Value::from("5.0")),
                    (Value::Int64(7), Value::Int64(7)),
                ]
            );
        }

        /// A probe side that must not be read.
        struct Unread;

        impl Operator for Unread {
            fn next(&mut self) -> OperatorResult {
                panic!("the probe side was read")
            }

            fn reset(&mut self) {}

            fn name(&self) -> &'static str {
                "Unread"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        /// Without a build row to meet, or with only NULL keys, an inner join
        /// on value-equality keys ends without reading its probe side.
        #[test]
        fn an_empty_build_side_ends_the_join_unprobed() {
            for build in [
                Vec::new(),
                vec![values_chunk(&[]), values_chunk(&[Value::Null, Value::Null])],
            ] {
                let mut join = HashJoinOperator::new(
                    Box::new(Unread),
                    Box::new(MockOperator::new(build)),
                    vec![0],
                    vec![0],
                    JoinType::Inner,
                    vec![LogicalType::Any, LogicalType::Any],
                )
                .with_value_equality_keys();
                assert!(join.next().unwrap().is_none());
                assert!(join.next().unwrap().is_none());
            }
        }

        /// A NULL in any key column of a row matches nothing, on either side.
        #[test]
        fn a_null_in_any_key_column_matches_nothing() {
            let pairs = |values: &[[Value; 2]]| {
                let mut builder = DataChunkBuilder::new(&[LogicalType::Any, LogicalType::Any]);
                for row in values {
                    for (column, value) in row.iter().enumerate() {
                        builder
                            .column_mut(column)
                            .unwrap()
                            .push_value(value.clone());
                    }
                    builder.advance_row();
                }
                MockOperator::new(vec![builder.finish()])
            };
            let probe = pairs(&[
                [Value::Int64(1), Value::Null],
                [Value::Int64(1), Value::Int64(2)],
            ]);
            let build = pairs(&[
                [Value::Int64(1), Value::Null],
                [Value::Float64(1.0), Value::Float64(2.0)],
                [Value::Null, Value::Int64(2)],
            ]);
            let mut join = HashJoinOperator::new(
                Box::new(probe),
                Box::new(build),
                vec![0, 1],
                vec![0, 1],
                JoinType::Inner,
                vec![LogicalType::Any; 4],
            )
            .with_value_equality_keys();
            let mut rows = Vec::new();
            while let Some(chunk) = join.next().unwrap() {
                for row in chunk.selected_indices() {
                    rows.push(
                        (0..4)
                            .map(|column| chunk.column(column).unwrap().get_value(row).unwrap())
                            .collect::<Vec<_>>(),
                    );
                }
            }
            assert_eq!(
                rows,
                [vec![
                    Value::Int64(1),
                    Value::Int64(2),
                    Value::Float64(1.0),
                    Value::Float64(2.0)
                ]]
            );
        }
    }
}
