//! DataChunk - the fundamental unit of vectorized execution.
//!
//! A DataChunk holds a batch of rows in columnar format. Processing data in
//! batches (typically 1024-2048 rows) lets the CPU stay busy and enables SIMD.

#[cfg(feature = "spill")]
use super::memory::{GroupedMemoryOwner, grouped_memory_limit};
#[cfg(feature = "spill")]
use super::operators::OperatorError;
use super::selection::SelectionVector;
use super::vector::ValueVector;
#[cfg(feature = "spill")]
use super::vector::{logical_type_retained_bytes, try_admitted_vec};
use crate::index::ZoneMapEntry;
use grafeo_common::types::LogicalType;
#[cfg(feature = "spill")]
use grafeo_common::types::Value;
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
/// let names = ValueVector::from_values(&[Value::from("Alix"), Value::from("Gus")]);
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
    /// Column containers admitted through the controlled grouped-output route.
    #[cfg(feature = "spill")]
    owned_column_bytes: usize,
    /// Keeps admission live until every chunk container has been dropped.
    #[cfg(feature = "spill")]
    owner: Option<GroupedMemoryOwner>,
}

/// Rows together with the reservation that owns their executor buffers.
#[cfg(feature = "spill")]
#[derive(Debug)]
pub struct AccountedRows {
    rows: Vec<Vec<Value>>,
    column_types: Option<Vec<LogicalType>>,
    owner: Option<GroupedMemoryOwner>,
}

#[cfg(feature = "spill")]
impl AccountedRows {
    /// Transfers rows and their reservation together to the next consumer.
    #[must_use]
    pub fn into_parts(self) -> (Vec<Vec<Value>>, Option<GroupedMemoryOwner>) {
        (self.rows, self.owner)
    }

    /// Transfers captured schema and rows before their shared reservation.
    #[must_use]
    pub fn into_parts_with_types(
        self,
    ) -> (
        Vec<Vec<Value>>,
        Option<Vec<LogicalType>>,
        Option<GroupedMemoryOwner>,
    ) {
        (self.rows, self.column_types, self.owner)
    }
}

impl DataChunk {
    /// Creates an admitted chunk for grouped output.
    ///
    /// Use [`Self::try_push_owned_value`] on this route. The existing manual
    /// mutation APIs retain their unaccounted behavior for compatibility.
    ///
    /// # Errors
    ///
    /// Returns a resource error if its buffers cannot be admitted.
    #[cfg(feature = "spill")]
    pub fn try_with_owner(
        column_types: &[LogicalType],
        capacity: usize,
        mut owner: GroupedMemoryOwner,
    ) -> Result<Self, OperatorError> {
        if capacity > DEFAULT_CHUNK_SIZE {
            return Err(grouped_memory_limit());
        }
        let original = owner.size();
        let mut columns = try_admitted_vec(column_types.len(), &mut owner)?;
        for data_type in column_types {
            columns.push(ValueVector::try_with_owner(
                data_type, capacity, &mut owner,
            )?);
        }
        let owned_column_bytes = owner.size() - original;
        Ok(Self {
            columns,
            selection: None,
            count: 0,
            capacity,
            zone_hints: None,
            owned_column_bytes,
            owner: Some(owner),
        })
    }

    /// Borrows the reservation carried by an admitted chunk.
    #[cfg(feature = "spill")]
    pub fn owner_mut(&mut self) -> Option<&mut GroupedMemoryOwner> {
        self.owner.as_mut()
    }

    /// Pushes a value whose payload has already been admitted by the caller.
    ///
    /// # Errors
    ///
    /// Returns an invalid-column or resource error before growing a buffer.
    #[cfg(feature = "spill")]
    pub fn try_push_owned_value(
        &mut self,
        column: usize,
        value: Value,
    ) -> Result<(), OperatorError> {
        let owner = self.owner.as_mut().ok_or_else(|| {
            OperatorError::Internal("grouped output chunk has no memory owner".into())
        })?;
        let column = self.columns.get_mut(column).ok_or_else(|| {
            OperatorError::Internal("grouped output column is out of bounds".into())
        })?;
        let before = owner.size();
        column.try_push_owned_value(value, owner)?;
        // Payloads are admitted before this call. Only the vector's arrays
        // change its reservation here, including scalar-to-generic conversion.
        if owner.size() >= before {
            self.owned_column_bytes += owner.size() - before;
        } else {
            self.owned_column_bytes -= before - owner.size();
        }
        Ok(())
    }

    /// Converts columns to rows while carrying their executor reservation.
    ///
    /// # Errors
    ///
    /// Returns a resource error if the live copy overlap cannot be admitted.
    #[cfg(feature = "spill")]
    pub fn try_into_accounted_rows(self) -> Result<AccountedRows, OperatorError> {
        self.try_into_accounted_rows_with_types(false)
    }

    /// Captures result types while admitting their live copy overlap with columns.
    ///
    /// # Errors
    ///
    /// Returns a resource error before growing schema or row buffers.
    #[cfg(feature = "spill")]
    pub fn try_into_accounted_rows_with_types(
        self,
        capture_types: bool,
    ) -> Result<AccountedRows, OperatorError> {
        self.try_into_accounted_rows_with_types_limited(capture_types, usize::MAX)
    }

    /// Takes at most `limit` selected rows while retaining their reservation.
    ///
    /// # Errors
    ///
    /// Returns a resource error before allocating schema or row buffers.
    #[cfg(feature = "spill")]
    pub fn try_into_accounted_rows_with_types_limited(
        mut self,
        capture_types: bool,
        limit: usize,
    ) -> Result<AccountedRows, OperatorError> {
        let column_types = if capture_types {
            let mut types = match self.owner.as_mut() {
                Some(owner) => try_admitted_vec(self.columns.len(), owner)?,
                None => Vec::with_capacity(self.columns.len()),
            };
            for column in &self.columns {
                if let Some(owner) = self.owner.as_mut() {
                    let bytes = logical_type_retained_bytes(column.data_type(), owner)?;
                    owner.try_grow(bytes)?;
                }
                types.push(column.data_type().clone());
            }
            Some(types)
        } else {
            None
        };
        let count = self.row_count().min(limit);
        let mut rows = match self.owner.as_mut() {
            Some(owner) => try_admitted_vec(count, owner)?,
            None => Vec::with_capacity(count),
        };
        for position in 0..count {
            let row = self
                .selection
                .as_ref()
                .map_or(position, |s| s.get(position).unwrap_or(position));
            let mut values = match self.owner.as_mut() {
                Some(owner) => try_admitted_vec(self.columns.len(), owner)?,
                None => Vec::with_capacity(self.columns.len()),
            };
            for column in &self.columns {
                // Value clones share their payloads; no second payload heap
                // is allocated while the column and row representations overlap.
                values.push(column.get_value(row).unwrap_or(Value::Null));
            }
            rows.push(values);
        }
        self.columns = Vec::new();
        self.selection = None;
        self.zone_hints = None;
        if let Some(owner) = self.owner.as_mut() {
            let retained = owner
                .size()
                .checked_sub(self.owned_column_bytes)
                .ok_or_else(grouped_memory_limit)?;
            owner.try_resize(retained)?;
        }
        Ok(AccountedRows {
            rows,
            column_types,
            owner: self.owner.take(),
        })
    }

    /// Creates an empty data chunk with no columns.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            selection: None,
            count: 0,
            capacity: 0,
            zone_hints: None,
            #[cfg(feature = "spill")]
            owned_column_bytes: 0,
            #[cfg(feature = "spill")]
            owner: None,
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
            #[cfg(feature = "spill")]
            owned_column_bytes: 0,
            #[cfg(feature = "spill")]
            owner: None,
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
            #[cfg(feature = "spill")]
            owned_column_bytes: 0,
            #[cfg(feature = "spill")]
            owner: None,
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

    /// Returns the types of the columns.
    #[must_use]
    pub fn column_types(&self) -> Vec<LogicalType> {
        self.columns
            .iter()
            .map(|column| column.data_type().clone())
            .collect()
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

    /// Appends `column`, which holds a value for each row of a chunk without
    /// a selection.
    pub fn push_column(&mut self, column: ValueVector) {
        debug_assert!(self.selection.is_none(), "a column for every row");
        debug_assert_eq!(column.len(), self.count, "a value for every row");
        self.columns.push(column);
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
        let Some(selection) = self.selection.take() else {
            return;
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

    /// Returns a new chunk with rows sorted by the values in the given column.
    ///
    /// For NodeId columns, sorts by the raw node ID. For other types, sorts by
    /// the Value's natural ordering. Rows with null values in the sort column
    /// are placed last.
    ///
    /// This is used for locality optimization: sorting by source node ID before
    /// an expand operator groups nearby nodes together, improving cache locality
    /// during adjacency index lookups.
    #[must_use]
    pub fn sort_by_column(&self, col_idx: usize) -> DataChunk {
        let row_count = self.len();
        if row_count <= 1 {
            return self.clone();
        }

        let Some(sort_col) = self.column(col_idx) else {
            return self.clone();
        };

        // Build permutation index sorted by the column values.
        // NodeId columns sort by raw ID; Int64 columns sort numerically
        // (offset-encoded so negative values sort correctly); nulls sort last.
        let mut indices: Vec<usize> = self.selected_indices().collect();
        indices.sort_by_key(|&i| {
            // Returns (is_null, key): nulls get (1, 0) so they sort strictly
            // after all non-null values (0, _), regardless of the key value.
            sort_col
                .get_node_id(i)
                .map(|id| (0u8, id.as_u64()))
                .or_else(|| {
                    sort_col.get_value(i).and_then(|v| match v {
                        // XOR with sign bit flips the ordering so that
                        // i64::MIN maps to 0 and i64::MAX maps to u64::MAX,
                        // preserving the natural signed ordering.
                        grafeo_common::types::Value::Int64(n) => {
                            // reason: intentional bit-level reinterpretation for sort ordering
                            #[allow(clippy::cast_sign_loss)]
                            Some((0u8, (n as u64) ^ (1u64 << 63)))
                        }
                        _ => None,
                    })
                })
                .unwrap_or((1u8, 0u64))
        });

        // Apply permutation to all columns
        let mut result_columns = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            let mut new_col = ValueVector::with_type(col.data_type().clone());
            for &idx in &indices {
                if let Some(val) = col.get(idx) {
                    new_col.push(val);
                }
            }
            result_columns.push(new_col);
        }

        DataChunk {
            columns: result_columns,
            selection: None,
            count: row_count,
            capacity: row_count,
            zone_hints: None,
            #[cfg(feature = "spill")]
            owned_column_bytes: 0,
            #[cfg(feature = "spill")]
            owner: None,
        }
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
                #[cfg(feature = "spill")]
                owned_column_bytes: 0,
                #[cfg(feature = "spill")]
                owner: None,
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
            #[cfg(feature = "spill")]
            owned_column_bytes: 0,
            #[cfg(feature = "spill")]
            owner: None,
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
            #[cfg(feature = "spill")]
            owned_column_bytes: 0,
            #[cfg(feature = "spill")]
            owner: None,
        }
    }

    /// Returns a slice of this chunk.
    ///
    /// Returns a new DataChunk containing the selected rows [offset, offset +
    /// count). Each column keeps its type, so every value is copied as it is.
    #[must_use]
    pub fn slice(&self, offset: usize, count: usize) -> DataChunk {
        if offset >= self.len() || count == 0 {
            return DataChunk::empty();
        }

        let actual_count = count.min(self.len() - offset);
        let mut result_columns = Vec::with_capacity(self.columns.len());

        for col in &self.columns {
            let mut new_col = ValueVector::with_capacity(col.data_type().clone(), actual_count);
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
            #[cfg(feature = "spill")]
            owned_column_bytes: 0,
            #[cfg(feature = "spill")]
            owner: None,
        }
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }
}

/// The column types for rows taken from the chunks of one input.
///
/// A column keeps its type while every chunk with rows has that type there and
/// becomes [`LogicalType::Any`] where they differ; a chunk without rows counts
/// only until one with rows comes. Rows copied into columns of these types keep
/// every value: a typed column stores a value of another type as that type's
/// default (see [`ValueVector::push_value`]), so an operator that only reorders,
/// cuts or deduplicates rows must not copy them by a declared schema.
#[derive(Debug, Default, Clone)]
pub(crate) struct ColumnTypes {
    /// The types so far (none before the first chunk).
    types: Option<Vec<LogicalType>>,
    /// Whether they come from chunks with rows.
    from_rows: bool,
}

impl ColumnTypes {
    /// Takes column types into account with a dedicated schema reservation.
    ///
    /// The owner must cover only this schema and outlive its metadata storage.
    ///
    /// # Errors
    ///
    /// Returns a resource or reservation error without changing the schema.
    #[cfg(feature = "spill")]
    pub(crate) fn add_accounted(
        &mut self,
        chunk: &DataChunk,
        owner: &mut GroupedMemoryOwner,
    ) -> Result<(), OperatorError> {
        let has_rows = chunk.row_count() > 0;
        match &mut self.types {
            Some(types) if self.from_rows && has_rows => {
                if !types.iter().zip(chunk.columns()).any(|(known, column)| {
                    known != column.data_type() && !matches!(known, LogicalType::Any)
                }) {
                    return Ok(());
                }
                let mut retained = types
                    .capacity()
                    .checked_mul(std::mem::size_of::<LogicalType>())
                    .ok_or_else(grouped_memory_limit)?;
                // Measure before mutating. The traversal helper admits its
                // scratch while the old metadata is still covered by owner.
                for (index, known) in types.iter().enumerate() {
                    if chunk
                        .column(index)
                        .is_none_or(|column| known == column.data_type())
                    {
                        retained = retained
                            .checked_add(logical_type_retained_bytes(known, owner)?)
                            .ok_or_else(grouped_memory_limit)?;
                    }
                }
                if retained > owner.size() {
                    return Err(OperatorError::Internal(
                        "grouped input schema has unowned metadata".into(),
                    ));
                }
                for (known, column) in types.iter_mut().zip(chunk.columns()) {
                    if known != column.data_type() {
                        // Assignment drops nested metadata before its bytes
                        // are released from the dedicated reservation below.
                        *known = LogicalType::Any;
                    }
                }
                owner.try_resize(retained)
            }
            Some(_) if self.from_rows || !has_rows => Ok(()),
            _ => {
                let original = owner.size();
                let replacement: Result<_, OperatorError> = (|| {
                    let mut types = try_admitted_vec(chunk.column_count(), owner)?;
                    let mut retained = types
                        .capacity()
                        .checked_mul(std::mem::size_of::<LogicalType>())
                        .ok_or_else(grouped_memory_limit)?;
                    for column in chunk.columns() {
                        let data_type = column.data_type();
                        // Retained source capacities bound the clone's heap.
                        // Admit that bound before invoking LogicalType::clone.
                        let bytes = logical_type_retained_bytes(data_type, owner)?;
                        owner.try_grow(bytes)?;
                        let cloned = data_type.clone();
                        retained = retained
                            .checked_add(logical_type_retained_bytes(&cloned, owner)?)
                            .ok_or_else(grouped_memory_limit)?;
                        types.push(cloned);
                    }
                    if retained > owner.size() - original {
                        return Err(OperatorError::Internal(
                            "grouped input schema clone exceeds its admission".into(),
                        ));
                    }
                    Ok((types, retained))
                })();
                let (types, retained) = match replacement {
                    Ok(replacement) => replacement,
                    Err(error) => {
                        // The unsuccessful replacement was dropped before
                        // releasing any of the admission that covered it.
                        owner.try_resize(original)?;
                        return Err(error);
                    }
                };
                let previous = self.types.replace(types);
                drop(previous);
                self.from_rows = has_rows;
                owner.try_resize(retained)
            }
        }
    }

    /// Takes the column types of `chunk` into account.
    pub(crate) fn add(&mut self, chunk: &DataChunk) {
        let has_rows = chunk.row_count() > 0;
        match &mut self.types {
            Some(types) if self.from_rows && has_rows => {
                for (known, column) in types.iter_mut().zip(chunk.columns()) {
                    if known != column.data_type() {
                        *known = LogicalType::Any;
                    }
                }
            }
            // A chunk without rows says nothing about the values: its types
            // count only while no other chunk gave any.
            Some(_) if self.from_rows || !has_rows => {}
            // The first chunk, or the first with rows after chunks without.
            _ => {
                self.types = Some(chunk.column_types());
                self.from_rows = has_rows;
            }
        }
    }

    /// The column types of the chunks seen so far (none before the first).
    pub(crate) fn types(&self) -> &[LogicalType] {
        self.types.as_deref().unwrap_or(&[])
    }
}

/// The type for a column copied from an input column of type `input`, which
/// the planner declared as `declared`: the input's, so that every value is
/// copied as it is, except where the input's is [`LogicalType::Any`] and the
/// planner declared a node or edge: an ID whose kind the input lost is that
/// entity again.
pub(crate) fn copied_column_type(
    input: &LogicalType,
    declared: Option<&LogicalType>,
) -> LogicalType {
    match (input, declared) {
        (LogicalType::Any, Some(entity @ (LogicalType::Node | LogicalType::Edge))) => {
            entity.clone()
        }
        _ => input.clone(),
    }
}

/// [`copied_column_type`] for each of the `input` types, declared as the
/// `declared` types at the same positions.
pub(crate) fn copied_column_types(
    input: &[LogicalType],
    declared: &[LogicalType],
) -> Vec<LogicalType> {
    input
        .iter()
        .enumerate()
        .map(|(i, input_type)| copied_column_type(input_type, declared.get(i)))
        .collect()
}

impl Clone for DataChunk {
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.clone(),
            selection: self.selection.clone(),
            count: self.count,
            capacity: self.capacity,
            zone_hints: self.zone_hints.clone(),
            #[cfg(feature = "spill")]
            owned_column_bytes: 0,
            #[cfg(feature = "spill")]
            owner: None,
        }
    }
}

#[cfg(all(test, feature = "spill"))]
mod owned_tests {
    use super::*;
    use crate::execution::memory::GroupedMemoryOwner;
    use crate::execution::operators::OperatorError;
    use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig, MemoryRegion};
    use grafeo_common::types::{EdgeId, NodeId, Value};
    use std::sync::Arc;

    fn manager(budget: usize) -> Arc<BufferManager> {
        BufferManager::new(BufferManagerConfig {
            budget,
            soft_limit_fraction: 1.0,
            evict_limit_fraction: 1.0,
            hard_limit_fraction: 1.0,
            background_eviction: false,
            spill_path: None,
        })
    }

    fn owner(manager: &Arc<BufferManager>) -> GroupedMemoryOwner {
        GroupedMemoryOwner::try_new(Arc::clone(manager), 0).unwrap()
    }

    #[test]
    fn owned_chunk_denies_buffers_before_allocation() {
        let manager = manager(64);
        let result = DataChunk::try_with_owner(&[LogicalType::Any], 16, owner(&manager));
        assert!(matches!(result, Err(OperatorError::LimitExceeded(_))));
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_chunk_keeps_reservation_until_drop() {
        let manager = manager(16 * 1024);
        let chunk = DataChunk::try_with_owner(&[LogicalType::Int64], 2, owner(&manager)).unwrap();
        assert!(manager.allocated() > 0);
        drop(chunk);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_rows_deny_copy_overlap_and_release_input() {
        let manager = manager(16 * 1024);
        let mut chunk = DataChunk::try_with_owner(&[LogicalType::Any], 1, owner(&manager)).unwrap();
        chunk.try_push_owned_value(0, Value::Int64(7)).unwrap();
        chunk.set_count(1);
        let held = manager
            .try_allocate(
                16 * 1024 - manager.allocated(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        assert!(matches!(
            chunk.try_into_accounted_rows(),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(manager.allocated(), held.size());
        drop(held);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_schema_capture_requires_headroom() {
        let budget = 16 * 1024;
        let manager = manager(budget);
        let schema = LogicalType::List(Box::new(LogicalType::Struct(vec![(
            "field".repeat(512),
            LogicalType::Int64,
        )])));
        let mut chunk = DataChunk::try_with_owner(&[schema], 1, owner(&manager)).unwrap();
        chunk.try_push_owned_value(0, Value::Null).unwrap();
        chunk.set_count(1);
        let held = manager
            .try_allocate(budget - manager.allocated(), MemoryRegion::ExecutionBuffers)
            .unwrap();
        assert!(matches!(
            chunk.try_into_accounted_rows_with_types(true),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(manager.allocated(), held.size());
        drop(held);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_rows_transfer_shared_payload_and_reservation() {
        let manager = manager(16 * 1024);
        let payload: Arc<[u8]> = Arc::from([11, 22, 33]);
        let reservation = GroupedMemoryOwner::try_new(Arc::clone(&manager), 64).unwrap();
        let mut chunk = DataChunk::try_with_owner(&[LogicalType::Any], 2, reservation).unwrap();
        chunk
            .try_push_owned_value(0, Value::Bytes(Arc::clone(&payload)))
            .unwrap();
        chunk.try_push_owned_value(0, Value::Null).unwrap();
        chunk.set_count(2);
        let (rows, reservation) = chunk.try_into_accounted_rows().unwrap().into_parts();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1], vec![Value::Null]);
        let Value::Bytes(copied) = &rows[0][0] else {
            panic!("the byte value must retain its type");
        };
        assert!(Arc::ptr_eq(&payload, copied));
        assert!(reservation.is_some());
        assert!(manager.allocated() > 64);
        drop(rows);
        assert!(manager.allocated() > 0);
        drop(reservation);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_rows_keep_entity_types_and_nulls() {
        let manager = manager(16 * 1024);
        let mut chunk =
            DataChunk::try_with_owner(&[LogicalType::Node, LogicalType::Edge], 2, owner(&manager))
                .unwrap();
        chunk
            .try_push_owned_value(0, Value::Int64(i64::MIN))
            .unwrap();
        chunk.try_push_owned_value(1, Value::Int64(-1)).unwrap();
        chunk.try_push_owned_value(0, Value::Null).unwrap();
        chunk.try_push_owned_value(1, Value::Int64(7)).unwrap();
        chunk.set_count(2);
        assert_eq!(chunk.column_types(), [LogicalType::Node, LogicalType::Edge]);
        assert_eq!(
            chunk.column(0).unwrap().get_node_id(0),
            Some(NodeId::new(1 << 63))
        );
        assert_eq!(
            chunk.column(1).unwrap().get_edge_id(0),
            Some(EdgeId::new(u64::MAX))
        );
        let rows = chunk.try_into_accounted_rows().unwrap();
        let (values, reservation) = rows.into_parts();
        assert_eq!(
            values,
            vec![
                vec![Value::Int64(i64::MIN), Value::Int64(-1)],
                vec![Value::Null, Value::Int64(7)]
            ]
        );
        drop(values);
        drop(reservation);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_push_denies_growth_without_changing_column() {
        let manager = manager(16 * 1024);
        let mut chunk = DataChunk::try_with_owner(&[LogicalType::Any], 1, owner(&manager)).unwrap();
        chunk.try_push_owned_value(0, Value::Int64(7)).unwrap();
        let held = manager
            .try_allocate(
                16 * 1024 - manager.allocated(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        assert!(matches!(
            chunk.try_push_owned_value(0, Value::Int64(9)),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(chunk.column(0).unwrap().len(), 1);
        assert_eq!(chunk.column(0).unwrap().get_value(0), Some(Value::Int64(7)));
        assert_eq!(manager.allocated(), 16 * 1024);
        drop(chunk);
        assert_eq!(manager.allocated(), held.size());
        drop(held);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_push_denies_null_validity_before_mutation() {
        let manager = manager(16 * 1024);
        let mut chunk =
            DataChunk::try_with_owner(&[LogicalType::Int64], 1, owner(&manager)).unwrap();
        let held = manager
            .try_allocate(
                16 * 1024 - manager.allocated(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        assert!(matches!(
            chunk.try_push_owned_value(0, Value::Null),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(chunk.column(0).unwrap().len(), 0);
        drop(chunk);
        assert_eq!(manager.allocated(), held.size());
    }

    #[test]
    fn owned_push_denies_type_conversion_before_mutation() {
        let manager = manager(16 * 1024);
        let mut chunk =
            DataChunk::try_with_owner(&[LogicalType::Int64], 1, owner(&manager)).unwrap();
        chunk.try_push_owned_value(0, Value::Int64(7)).unwrap();
        let held = manager
            .try_allocate(
                16 * 1024 - manager.allocated(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        assert!(matches!(
            chunk.try_push_owned_value(0, Value::Bool(true)),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(chunk.column(0).unwrap().data_type(), &LogicalType::Int64);
        assert_eq!(chunk.column(0).unwrap().get_value(0), Some(Value::Int64(7)));
        assert_eq!(chunk.column(0).unwrap().len(), 1);
        drop(chunk);
        assert_eq!(manager.allocated(), held.size());
    }

    #[test]
    fn owned_push_preserves_values_across_growth_and_conversion() {
        let manager = manager(16 * 1024);
        let mut chunk =
            DataChunk::try_with_owner(&[LogicalType::Int64], 1, owner(&manager)).unwrap();
        chunk.try_push_owned_value(0, Value::Int64(7)).unwrap();
        chunk.try_push_owned_value(0, Value::Bool(true)).unwrap();
        chunk.try_push_owned_value(0, Value::Null).unwrap();
        chunk.set_count(3);
        assert_eq!(chunk.column_types(), [LogicalType::Any]);
        let (rows, reservation) = chunk.try_into_accounted_rows().unwrap().into_parts();
        assert_eq!(
            rows,
            vec![
                vec![Value::Int64(7)],
                vec![Value::Bool(true)],
                vec![Value::Null]
            ]
        );
        drop(rows);
        drop(reservation);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn owned_chunk_denies_nested_type_metadata_before_clone() {
        let manager = manager(4096);
        let data_type = LogicalType::Tuple(vec![LogicalType::Int64; 512]);
        let result = DataChunk::try_with_owner(&[data_type], 0, owner(&manager));
        assert!(matches!(result, Err(OperatorError::LimitExceeded(_))));
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn accounted_column_types_denies_nested_schema_before_clone() {
        let manager = manager(4096);
        let mut reservation = owner(&manager);
        let mut types = ColumnTypes::default();
        let chunk =
            DataChunk::with_capacity(&[LogicalType::Tuple(vec![LogicalType::Int64; 512])], 0);
        assert!(matches!(
            types.add_accounted(&chunk, &mut reservation),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(types.types(), &[]);
        assert_eq!(reservation.size(), 0);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn accounted_column_types_retains_schema_until_drop() {
        let manager = manager(16 * 1024);
        let mut reservation = owner(&manager);
        let mut types = ColumnTypes::default();
        let schema = LogicalType::Struct(vec![(
            "field".repeat(16),
            LogicalType::List(Box::new(LogicalType::Int64)),
        )]);
        let chunk = DataChunk::with_capacity(std::slice::from_ref(&schema), 0);
        types.add_accounted(&chunk, &mut reservation).unwrap();
        assert_eq!(types.types(), &[schema]);
        assert!(reservation.size() > std::mem::size_of::<LogicalType>());
        assert_eq!(manager.allocated(), reservation.size());
        drop(types);
        assert!(manager.allocated() > 0);
        drop(reservation);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn accounted_column_types_denies_replacement_overlap_and_preserves_schema() {
        let manager = manager(16 * 1024);
        let mut reservation = owner(&manager);
        let mut types = ColumnTypes::default();
        let schema = LogicalType::List(Box::new(LogicalType::Int64));
        let empty = DataChunk::with_capacity(std::slice::from_ref(&schema), 0);
        types.add_accounted(&empty, &mut reservation).unwrap();
        let original = reservation.size();
        let mut with_rows = DataChunk::with_capacity(&[LogicalType::Bool], 1);
        with_rows
            .column_mut(0)
            .unwrap()
            .push_value(Value::Bool(true));
        with_rows.set_count(1);
        let held = manager
            .try_allocate(
                16 * 1024 - manager.allocated(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        assert!(matches!(
            types.add_accounted(&with_rows, &mut reservation),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(types.types(), &[schema]);
        assert_eq!(reservation.size(), original);
        assert!(!types.from_rows);
        assert_eq!(manager.allocated(), 16 * 1024);
        drop(held);
        types.add_accounted(&with_rows, &mut reservation).unwrap();
        assert_eq!(types.types(), &[LogicalType::Bool]);
        assert!(types.from_rows);
        assert_eq!(reservation.size(), std::mem::size_of::<LogicalType>());
        drop(types);
        drop(reservation);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn accounted_column_types_releases_differing_heap_and_ignores_empty_rows() {
        let manager = manager(16 * 1024);
        let mut reservation = owner(&manager);
        let mut types = ColumnTypes::default();
        let mut first =
            DataChunk::with_capacity(&[LogicalType::List(Box::new(LogicalType::Int64))], 1);
        first.column_mut(0).unwrap().push_value(Value::Int64(7));
        first.set_count(1);
        types.add_accounted(&first, &mut reservation).unwrap();
        let original = reservation.size();
        let mut different = DataChunk::with_capacity(&[LogicalType::Bool], 1);
        different
            .column_mut(0)
            .unwrap()
            .push_value(Value::Bool(true));
        different.set_count(1);
        types.add_accounted(&different, &mut reservation).unwrap();
        assert_eq!(types.types(), &[LogicalType::Any]);
        assert!(reservation.size() < original);
        assert_eq!(reservation.size(), std::mem::size_of::<LogicalType>());
        let held = manager
            .try_allocate(
                16 * 1024 - manager.allocated(),
                MemoryRegion::ExecutionBuffers,
            )
            .unwrap();
        let empty = DataChunk::with_capacity(&[LogicalType::String], 0);
        types.add_accounted(&empty, &mut reservation).unwrap();
        assert_eq!(types.types(), &[LogicalType::Any]);
        assert_eq!(manager.allocated(), 16 * 1024);
        drop(types);
        drop(reservation);
        assert_eq!(manager.allocated(), held.size());
        drop(held);
        assert_eq!(manager.allocated(), 0);
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
    use grafeo_common::types::Value;

    /// Chunks without rows do not turn a node column into `Any`: their types
    /// count only until a chunk with rows comes. Chunks with rows of different
    /// types do.
    #[test]
    fn column_types_ignore_chunks_without_rows() {
        let empty = || DataChunk::with_capacity(&[LogicalType::Any], 0);
        let nodes = || {
            let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
            builder
                .column_mut(0)
                .unwrap()
                .push_node_id(grafeo_common::types::NodeId::new(1));
            builder.advance_row();
            builder.finish()
        };

        let mut types = ColumnTypes::default();
        types.add(&empty());
        assert_eq!(types.types(), [LogicalType::Any]);
        types.add(&nodes());
        types.add(&empty());
        assert_eq!(types.types(), [LogicalType::Node]);

        let mut edges = DataChunkBuilder::new(&[LogicalType::Edge]);
        edges
            .column_mut(0)
            .unwrap()
            .push_edge_id(grafeo_common::types::EdgeId::new(1));
        edges.advance_row();
        types.add(&edges.finish());
        assert_eq!(types.types(), [LogicalType::Any]);
    }

    /// A copied column keeps its input type; a declared node or edge only
    /// gives an `Any` column back its entity kind, never a scalar type.
    #[test]
    fn copied_columns_keep_the_input_type_or_a_declared_entity() {
        use LogicalType::{Any, Edge, Int64, Node};
        let text = LogicalType::String;

        assert_eq!(
            copied_column_types(
                &[Any, Any, Any, Node, Int64, text.clone(), Any],
                &[Edge, Node, Int64, Edge, Any, Node],
            ),
            [Edge, Node, Any, Node, Int64, text, Any]
        );
    }

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
    // reason: loop index 0..5 fits usize
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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

    #[test]
    fn test_chunk_zone_hints_default() {
        let hints = ChunkZoneHints::default();
        assert!(hints.column_hints.is_empty());
    }

    #[test]
    fn test_chunk_zone_hints_set_and_get() {
        let schema = [LogicalType::Int64];
        let mut chunk = DataChunk::with_schema(&schema);

        // Initially no zone hints
        assert!(chunk.zone_hints().is_none());

        // Set zone hints
        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(
                grafeo_common::types::Value::Int64(10),
                grafeo_common::types::Value::Int64(100),
                0,
                10,
            ),
        );
        chunk.set_zone_hints(hints);

        // Zone hints should now be present
        assert!(chunk.zone_hints().is_some());
        let retrieved = chunk.zone_hints().unwrap();
        assert_eq!(retrieved.column_hints.len(), 1);
        assert!(retrieved.column_hints.contains_key(&0));
    }

    #[test]
    fn test_chunk_zone_hints_clear() {
        let schema = [LogicalType::Int64];
        let mut chunk = DataChunk::with_schema(&schema);

        // Set zone hints
        let hints = ChunkZoneHints::default();
        chunk.set_zone_hints(hints);
        assert!(chunk.zone_hints().is_some());

        // Clear zone hints
        chunk.clear_zone_hints();
        assert!(chunk.zone_hints().is_none());
    }

    #[test]
    fn test_chunk_zone_hints_preserved_on_clone() {
        let schema = [LogicalType::Int64];
        let mut chunk = DataChunk::with_schema(&schema);

        // Set zone hints
        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(
                grafeo_common::types::Value::Int64(1),
                grafeo_common::types::Value::Int64(10),
                0,
                10,
            ),
        );
        chunk.set_zone_hints(hints);

        // Clone and verify zone hints are preserved
        let cloned = chunk.clone();
        assert!(cloned.zone_hints().is_some());
        assert_eq!(cloned.zone_hints().unwrap().column_hints.len(), 1);
    }

    #[test]
    fn test_chunk_reset_clears_zone_hints() {
        let schema = [LogicalType::Int64];
        let mut chunk = DataChunk::with_schema(&schema);

        // Set zone hints
        let hints = ChunkZoneHints::default();
        chunk.set_zone_hints(hints);
        assert!(chunk.zone_hints().is_some());

        // Reset should clear zone hints
        chunk.reset();
        assert!(chunk.zone_hints().is_none());
    }

    #[test]
    fn test_sort_by_column_int64() {
        let v = ValueVector::from_values(&[Value::Int64(30), Value::Int64(10), Value::Int64(20)]);
        let names = ValueVector::from_values(&[
            Value::from("Alix"),
            Value::from("Gus"),
            Value::from("Jules"),
        ]);
        let chunk = DataChunk::new(vec![v, names]);
        let sorted = chunk.sort_by_column(0);

        assert_eq!(sorted.len(), 3);
        // Values should be sorted: 10, 20, 30
        assert_eq!(
            sorted.column(0).unwrap().get_value(0),
            Some(Value::Int64(10))
        );
        assert_eq!(
            sorted.column(0).unwrap().get_value(1),
            Some(Value::Int64(20))
        );
        assert_eq!(
            sorted.column(0).unwrap().get_value(2),
            Some(Value::Int64(30))
        );
        // Names should follow the permutation: Gus, Jules, Alix
        assert_eq!(
            sorted.column(1).unwrap().get_value(0),
            Some(Value::from("Gus"))
        );
        assert_eq!(
            sorted.column(1).unwrap().get_value(1),
            Some(Value::from("Jules"))
        );
        assert_eq!(
            sorted.column(1).unwrap().get_value(2),
            Some(Value::from("Alix"))
        );
    }

    #[test]
    fn test_sort_by_column_node_ids() {
        use grafeo_common::types::NodeId;

        let mut id_col = ValueVector::new();
        id_col.push_node_id(NodeId::new(100));
        id_col.push_node_id(NodeId::new(5));
        id_col.push_node_id(NodeId::new(50));

        let data_col = ValueVector::from_values(&[
            Value::from("hundred"),
            Value::from("five"),
            Value::from("fifty"),
        ]);

        let chunk = DataChunk::new(vec![id_col, data_col]);
        let sorted = chunk.sort_by_column(0);

        assert_eq!(sorted.len(), 3);
        // Node IDs should be sorted: 5, 50, 100
        assert_eq!(
            sorted.column(0).unwrap().get_node_id(0),
            Some(NodeId::new(5))
        );
        assert_eq!(
            sorted.column(0).unwrap().get_node_id(1),
            Some(NodeId::new(50))
        );
        assert_eq!(
            sorted.column(0).unwrap().get_node_id(2),
            Some(NodeId::new(100))
        );
        // Data should follow: five, fifty, hundred
        assert_eq!(
            sorted.column(1).unwrap().get_value(0),
            Some(Value::from("five"))
        );
    }

    #[test]
    fn test_sort_by_column_empty() {
        let chunk = DataChunk::empty();
        let sorted = chunk.sort_by_column(0);
        assert_eq!(sorted.len(), 0);
    }

    #[test]
    fn test_sort_by_column_single_row() {
        let v = ValueVector::from_values(&[Value::Int64(42)]);
        let chunk = DataChunk::new(vec![v]);
        let sorted = chunk.sort_by_column(0);
        assert_eq!(sorted.len(), 1);
        assert_eq!(
            sorted.column(0).unwrap().get_value(0),
            Some(Value::Int64(42))
        );
    }
}
