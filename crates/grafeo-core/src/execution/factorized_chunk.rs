//! FactorizedChunk - multi-level factorized data representation.
//!
//! A `FactorizedChunk` organizes columns into levels, where each level can have
//! different factorization (multiplicity). This avoids materializing the full
//! Cartesian product during multi-hop graph traversals.
//!
//! # Example
//!
//! For a 2-hop query `MATCH (a)-[e1]->(b)-[e2]->(c)`:
//!
//! ```text
//! Level 0 (flat):   [a1, a2]           (2 source nodes)
//! Level 1 (unflat): [b1, b2, b3, b4]   (4 first-hop neighbors)
//!                   offsets: [0, 2, 4]  (a1 has 2 neighbors, a2 has 2)
//! Level 2 (unflat): [c1, c2, ..., c8]  (8 second-hop neighbors)
//!                   offsets: [0, 2, 4, 6, 8]
//!
//! Logical rows = 2 * 2 * 2 = 8, but physical storage = 2 + 4 + 8 = 14 values
//! vs flat storage = 8 * 5 columns = 40 values
//! ```

use std::sync::Arc;

use super::chunk::DataChunk;
use super::chunk_state::ChunkState;
use super::factorized_vector::FactorizedVector;
use super::vector::ValueVector;

/// A chunk that supports factorized representation across multiple levels.
///
/// Columns are organized in groups by their factorization level:
/// - Level 0 (flat): Base columns, one value per logical row
/// - Level 1 (unflat): First expansion, grouped by level 0
/// - Level 2 (unflat): Second expansion, grouped by level 1
/// - And so on...
///
/// # State Management
///
/// The chunk maintains a [`ChunkState`] that provides:
/// - Cached multiplicities for O(1) aggregate access
/// - Selection vector support for lazy filtering
/// - Generation tracking for cache invalidation
#[derive(Debug, Clone)]
pub struct FactorizedChunk {
    /// Column groups organized by factorization level.
    levels: Vec<FactorizationLevel>,
    /// Total logical row count (product of all multiplicities).
    logical_row_count: usize,
    /// Unified state tracking (caching, selection, etc.).
    state: ChunkState,
}

/// A factorization level containing columns at the same nesting depth.
#[derive(Debug, Clone)]
pub struct FactorizationLevel {
    /// Columns at this level.
    columns: Vec<FactorizedVector>,
    /// Column names or identifiers (for schema mapping).
    column_names: Vec<String>,
    /// Number of groups at this level.
    group_count: usize,
    /// Multiplicities for each group (how many children per parent).
    /// For level 0, this is vec![1; group_count].
    /// For level N, multiplicities[i] = number of values for parent i.
    multiplicities: Vec<usize>,
}

impl FactorizationLevel {
    /// Creates a new flat level (level 0) from columns.
    #[must_use]
    pub fn flat(columns: Vec<FactorizedVector>, column_names: Vec<String>) -> Self {
        let group_count = columns.first().map_or(0, FactorizedVector::physical_len);
        let multiplicities = vec![1; group_count];
        Self {
            columns,
            column_names,
            group_count,
            multiplicities,
        }
    }

    /// Creates a new unflat level with the given multiplicities.
    ///
    /// Note: `multiplicities[i]` is the number of values for parent i.
    /// The total number of values (group_count) is the sum of all multiplicities.
    #[must_use]
    pub fn unflat(
        columns: Vec<FactorizedVector>,
        column_names: Vec<String>,
        multiplicities: Vec<usize>,
    ) -> Self {
        // group_count is the total number of values at this level (sum of multiplicities)
        let group_count = multiplicities.iter().sum();
        Self {
            columns,
            column_names,
            group_count,
            multiplicities,
        }
    }

    /// Returns the number of columns at this level.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns the number of groups at this level.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// Returns the total physical value count across all columns.
    #[must_use]
    pub fn physical_value_count(&self) -> usize {
        self.columns
            .iter()
            .map(FactorizedVector::physical_len)
            .sum()
    }

    /// Returns the multiplicities for this level.
    #[must_use]
    pub fn multiplicities(&self) -> &[usize] {
        &self.multiplicities
    }

    /// Returns a column by index.
    #[must_use]
    pub fn column(&self, index: usize) -> Option<&FactorizedVector> {
        self.columns.get(index)
    }

    /// Returns a mutable column by index.
    pub fn column_mut(&mut self, index: usize) -> Option<&mut FactorizedVector> {
        self.columns.get_mut(index)
    }

    /// Returns the column names.
    #[must_use]
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }
}

impl FactorizedChunk {
    /// Creates an empty factorized chunk.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            levels: Vec::new(),
            logical_row_count: 0,
            state: ChunkState::flat(0),
        }
    }

    /// Creates a factorized chunk from a flat `DataChunk`.
    ///
    /// The resulting chunk has a single level (level 0) with all columns flat.
    #[must_use]
    pub fn from_flat(chunk: &DataChunk, column_names: Vec<String>) -> Self {
        let columns: Vec<FactorizedVector> = chunk
            .columns()
            .iter()
            .map(|c| FactorizedVector::flat(c.clone()))
            .collect();

        let row_count = chunk.row_count();
        let level = FactorizationLevel::flat(columns, column_names);

        Self {
            levels: vec![level],
            logical_row_count: row_count,
            state: ChunkState::unflat(1, row_count),
        }
    }

    /// Creates a factorized chunk with a single flat level.
    #[must_use]
    pub fn with_flat_level(columns: Vec<ValueVector>, column_names: Vec<String>) -> Self {
        let row_count = columns.first().map_or(0, ValueVector::len);
        let factorized_columns: Vec<FactorizedVector> =
            columns.into_iter().map(FactorizedVector::flat).collect();

        let level = FactorizationLevel::flat(factorized_columns, column_names);

        Self {
            levels: vec![level],
            logical_row_count: row_count,
            state: ChunkState::unflat(1, row_count),
        }
    }

    /// Returns the number of factorization levels.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Returns the logical row count (full Cartesian product size).
    #[must_use]
    pub fn logical_row_count(&self) -> usize {
        self.logical_row_count
    }

    /// Returns the physical storage size (actual values stored).
    #[must_use]
    pub fn physical_size(&self) -> usize {
        self.levels
            .iter()
            .map(FactorizationLevel::physical_value_count)
            .sum()
    }

    /// Returns the chunk state.
    #[must_use]
    pub fn chunk_state(&self) -> &ChunkState {
        &self.state
    }

    /// Returns mutable access to the chunk state.
    pub fn chunk_state_mut(&mut self) -> &mut ChunkState {
        &mut self.state
    }

    /// Returns path multiplicities, computing once and caching.
    ///
    /// This is the key optimization for aggregation: multiplicities are
    /// computed once and reused for all aggregates (COUNT, SUM, AVG, etc.).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mults = chunk.path_multiplicities_cached();
    /// let sum = chunk.sum_deepest_with_multiplicities(0, &mults);
    /// let avg = chunk.avg_deepest_with_multiplicities(0, &mults);
    /// ```
    pub fn path_multiplicities_cached(&mut self) -> Arc<[usize]> {
        // Check if already cached
        if let Some(cached) = self.state.cached_multiplicities() {
            return Arc::clone(cached);
        }

        // Compute and cache
        let mults = self.compute_path_multiplicities();
        let arc_mults: Arc<[usize]> = mults.into();
        self.state.set_cached_multiplicities(Arc::clone(&arc_mults));
        arc_mults
    }

    /// Returns a level by index.
    #[must_use]
    pub fn level(&self, index: usize) -> Option<&FactorizationLevel> {
        self.levels.get(index)
    }

    /// Returns a mutable level by index.
    pub fn level_mut(&mut self, index: usize) -> Option<&mut FactorizationLevel> {
        self.levels.get_mut(index)
    }

    /// Adds a new factorization level for expansion results.
    ///
    /// The new level's multiplicities determine how many values each parent
    /// in the previous level expands to.
    ///
    /// # Arguments
    ///
    /// * `columns` - Columns at the new level
    /// * `column_names` - Names for the new columns
    /// * `offsets` - Offset array where `offsets[i]` is the start index for parent `i`
    pub fn add_level(
        &mut self,
        columns: Vec<ValueVector>,
        column_names: Vec<String>,
        offsets: &[u32],
    ) {
        let parent_count = offsets.len().saturating_sub(1);

        // Compute multiplicities from offsets
        let multiplicities: Vec<usize> = (0..parent_count)
            .map(|i| (offsets[i + 1] - offsets[i]) as usize)
            .collect();

        // Create unflat factorized vectors
        let factorized_columns: Vec<FactorizedVector> = columns
            .into_iter()
            .map(|data| FactorizedVector::unflat(data, offsets.to_vec(), parent_count))
            .collect();

        let level =
            FactorizationLevel::unflat(factorized_columns, column_names, multiplicities.clone());
        self.levels.push(level);

        // Update logical row count
        // New count = previous count * sum of new multiplicities / parent_count
        // Actually: each parent's contribution is multiplied by its multiplicity
        if self.levels.len() == 1 {
            // First level - logical count is just the sum of multiplicities (or total values)
            self.logical_row_count = multiplicities.iter().sum();
        } else {
            // For subsequent levels, we need to compute based on parent multiplicities
            self.recompute_logical_row_count();
        }

        // Update state (invalidates cached multiplicities)
        self.update_state();
    }

    /// Adds a level with pre-computed factorized vectors.
    pub fn add_factorized_level(&mut self, level: FactorizationLevel) {
        self.levels.push(level);
        self.recompute_logical_row_count();
        self.update_state();
    }

    /// Updates the ChunkState to reflect current structure.
    fn update_state(&mut self) {
        self.state = ChunkState::unflat(self.levels.len(), self.logical_row_count);
    }

    /// Recomputes the logical row count from all levels.
    fn recompute_logical_row_count(&mut self) {
        if self.levels.is_empty() {
            self.logical_row_count = 0;
            return;
        }

        // Start with level 0 count
        let level0_count = self.levels[0].group_count;
        if self.levels.len() == 1 {
            self.logical_row_count = level0_count;
            return;
        }

        // For multi-level: compute recursively
        // Each parent at level N-1 contributes its multiplicity to level N
        let mut counts = vec![1usize; level0_count];

        for level_idx in 1..self.levels.len() {
            let level = &self.levels[level_idx];
            let mut new_counts = Vec::new();

            for (parent_idx, &parent_count) in counts.iter().enumerate() {
                // This parent expands to level.multiplicities[parent_idx] children
                if parent_idx < level.multiplicities.len() {
                    let child_mult = level.multiplicities[parent_idx];
                    for _ in 0..child_mult {
                        new_counts.push(parent_count);
                    }
                }
            }

            counts = new_counts;
        }

        self.logical_row_count = counts.len();
    }

    /// Flattens to a regular `DataChunk` (materializes the Cartesian product).
    ///
    /// All levels are expanded into flat rows.
    #[must_use]
    pub fn flatten(&self) -> DataChunk {
        if self.levels.is_empty() {
            return DataChunk::empty();
        }

        // Collect all column types across all levels
        let mut all_columns: Vec<ValueVector> = Vec::new();

        // For a single level, just flatten each column
        if self.levels.len() == 1 {
            let level = &self.levels[0];
            for col in &level.columns {
                all_columns.push(col.flatten(None));
            }
            return DataChunk::new(all_columns);
        }

        // Multi-level: need to expand according to multiplicities
        // Build column data by iterating through logical rows
        let row_iter = self.logical_row_iter();
        let total_cols: usize = self.levels.iter().map(|l| l.column_count()).sum();

        // Pre-allocate output columns
        let mut output_columns: Vec<ValueVector> = Vec::with_capacity(total_cols);
        for level in &self.levels {
            for col in &level.columns {
                output_columns.push(ValueVector::with_capacity(
                    col.data_type(),
                    self.logical_row_count,
                ));
            }
        }

        // Iterate through all logical rows
        for indices in row_iter {
            let mut col_offset = 0;
            for (level_idx, level) in self.levels.iter().enumerate() {
                let level_idx_value = indices.get(level_idx).copied().unwrap_or(0);
                for (col_idx, col) in level.columns.iter().enumerate() {
                    if let Some(value) = col.get_physical(level_idx_value) {
                        output_columns[col_offset + col_idx].push_value(value);
                    }
                }
                col_offset += level.column_count();
            }
        }

        DataChunk::new(output_columns)
    }

    /// Returns an iterator over logical rows without materializing.
    ///
    /// Each iteration yields a vector of physical indices, one per level.
    pub fn logical_row_iter(&self) -> FactorizedRowIterator<'_> {
        FactorizedRowIterator::new(self)
    }

    /// Gets the total number of columns across all levels.
    #[must_use]
    pub fn total_column_count(&self) -> usize {
        self.levels.iter().map(|l| l.column_count()).sum()
    }

    /// Gets all column names in order across all levels.
    #[must_use]
    pub fn all_column_names(&self) -> Vec<String> {
        self.levels
            .iter()
            .flat_map(|l| l.column_names.iter().cloned())
            .collect()
    }

    /// Filters the deepest level in-place using a predicate on column values.
    ///
    /// This is the key optimization: instead of flattening and filtering all rows,
    /// we filter only at the deepest level and update parent multiplicities.
    ///
    /// # Arguments
    ///
    /// * `column_idx` - Column index within the deepest level to filter on
    /// * `predicate` - Function that returns true for values to keep
    ///
    /// # Returns
    ///
    /// A new FactorizedChunk with filtered values, or None if all rows are filtered out.
    #[must_use]
    pub fn filter_deepest<F>(&self, column_idx: usize, predicate: F) -> Option<Self>
    where
        F: Fn(&grafeo_common::types::Value) -> bool,
    {
        if self.levels.is_empty() {
            return None;
        }

        let deepest_idx = self.levels.len() - 1;
        let deepest = &self.levels[deepest_idx];

        // Get the column to filter on
        let filter_col = deepest.column(column_idx)?;

        // Build filtered columns for the deepest level
        let mut new_columns: Vec<ValueVector> = (0..deepest.column_count())
            .map(|i| ValueVector::with_type(deepest.column(i).unwrap().data_type()))
            .collect();

        // Track new multiplicities for each parent
        let parent_count = filter_col.parent_count();
        let mut new_multiplicities: Vec<usize> = vec![0; parent_count];
        let mut new_offsets: Vec<u32> = vec![0];

        // Filter each parent's children
        for parent_idx in 0..parent_count {
            let (start, end) = filter_col.range_for_parent(parent_idx);

            for phys_idx in start..end {
                // Check if this value passes the filter
                if let Some(value) = filter_col.get_physical(phys_idx) {
                    if predicate(&value) {
                        // Copy all columns for this row
                        for col_idx in 0..deepest.column_count() {
                            if let Some(col) = deepest.column(col_idx) {
                                if let Some(v) = col.get_physical(phys_idx) {
                                    new_columns[col_idx].push_value(v);
                                }
                            }
                        }
                        new_multiplicities[parent_idx] += 1;
                    }
                }
            }

            new_offsets.push(new_columns[0].len() as u32);
        }

        // Check if we have any rows left
        let total_remaining: usize = new_multiplicities.iter().sum();
        if total_remaining == 0 {
            return Some(Self::empty());
        }

        // Build the new factorized vectors
        let new_factorized_cols: Vec<FactorizedVector> = new_columns
            .into_iter()
            .map(|data| FactorizedVector::unflat(data, new_offsets.clone(), parent_count))
            .collect();

        let new_level = FactorizationLevel::unflat(
            new_factorized_cols,
            deepest.column_names().to_vec(),
            new_multiplicities,
        );

        // Build the result chunk
        let mut result = Self {
            levels: self.levels[..deepest_idx].to_vec(),
            logical_row_count: 0,
            state: ChunkState::flat(0),
        };
        result.levels.push(new_level);
        result.recompute_logical_row_count();
        result.update_state();

        Some(result)
    }

    /// Filters the deepest level using a multi-column predicate.
    ///
    /// This allows filtering based on values from multiple columns in the deepest level.
    ///
    /// # Arguments
    ///
    /// * `predicate` - Function that takes a slice of values (one per column) and returns true to keep
    #[must_use]
    pub fn filter_deepest_multi<F>(&self, predicate: F) -> Option<Self>
    where
        F: Fn(&[grafeo_common::types::Value]) -> bool,
    {
        if self.levels.is_empty() {
            return None;
        }

        let deepest_idx = self.levels.len() - 1;
        let deepest = &self.levels[deepest_idx];
        let col_count = deepest.column_count();

        if col_count == 0 {
            return None;
        }

        let first_col = deepest.column(0)?;
        let parent_count = first_col.parent_count();

        // Build filtered columns
        let mut new_columns: Vec<ValueVector> = (0..col_count)
            .map(|i| ValueVector::with_type(deepest.column(i).unwrap().data_type()))
            .collect();

        let mut new_multiplicities: Vec<usize> = vec![0; parent_count];
        let mut new_offsets: Vec<u32> = vec![0];
        let mut row_values: Vec<grafeo_common::types::Value> = Vec::with_capacity(col_count);

        for parent_idx in 0..parent_count {
            let (start, end) = first_col.range_for_parent(parent_idx);

            for phys_idx in start..end {
                // Collect values from all columns
                row_values.clear();
                for col_idx in 0..col_count {
                    if let Some(col) = deepest.column(col_idx) {
                        if let Some(v) = col.get_physical(phys_idx) {
                            row_values.push(v);
                        }
                    }
                }

                // Apply predicate
                if predicate(&row_values) {
                    for (col_idx, v) in row_values.iter().enumerate() {
                        new_columns[col_idx].push_value(v.clone());
                    }
                    new_multiplicities[parent_idx] += 1;
                }
            }

            new_offsets.push(new_columns[0].len() as u32);
        }

        // Check if any rows remain
        let total: usize = new_multiplicities.iter().sum();
        if total == 0 {
            return Some(Self::empty());
        }

        // Build new level
        let new_factorized_cols: Vec<FactorizedVector> = new_columns
            .into_iter()
            .map(|data| FactorizedVector::unflat(data, new_offsets.clone(), parent_count))
            .collect();

        let new_level = FactorizationLevel::unflat(
            new_factorized_cols,
            deepest.column_names().to_vec(),
            new_multiplicities,
        );

        let mut result = Self {
            levels: self.levels[..deepest_idx].to_vec(),
            logical_row_count: 0,
            state: ChunkState::flat(0),
        };
        result.levels.push(new_level);
        result.recompute_logical_row_count();
        result.update_state();

        Some(result)
    }

    // ========================================================================
    // Factorized Aggregation Methods
    // ========================================================================

    /// Computes COUNT(*) without flattening - returns the logical row count.
    ///
    /// This is O(n) where n is the number of physical values, instead of
    /// O(m) where m is the number of logical rows (which can be exponentially larger).
    ///
    /// # Example
    ///
    /// For a 3-level chunk:
    /// - Level 0: 100 sources
    /// - Level 1: 10 neighbors each = 1,000 physical
    /// - Level 2: 10 neighbors each = 10,000 physical
    /// - Logical rows = 100 * 10 * 10 = 10,000
    ///
    /// `count_rows()` returns 10,000 by computing from multiplicities, not by
    /// iterating through all logical rows.
    #[must_use]
    pub fn count_rows(&self) -> usize {
        self.logical_row_count()
    }

    /// Computes the effective multiplicity for each value at the deepest level.
    ///
    /// This is how many times each value would appear in the flattened result.
    /// For example, if a source has 3 first-hop neighbors and each has 2 second-hop
    /// neighbors, each first-hop value has multiplicity 2 (appearing in 2 paths).
    ///
    /// # Returns
    ///
    /// A vector where `result[i]` is the multiplicity of physical value `i` at the
    /// deepest level. The sum of all multiplicities equals `logical_row_count()`.
    ///
    /// # Note
    ///
    /// For repeated access (e.g., computing multiple aggregates), prefer using
    /// [`path_multiplicities_cached`](Self::path_multiplicities_cached) which
    /// caches the result and avoids O(levels) recomputation.
    #[must_use]
    pub fn compute_path_multiplicities(&self) -> Vec<usize> {
        if self.levels.is_empty() {
            return Vec::new();
        }

        // For a single level, each value has multiplicity 1
        if self.levels.len() == 1 {
            return vec![1; self.levels[0].group_count];
        }

        // Start with multiplicity 1 for each value at level 0
        let mut parent_multiplicities = vec![1usize; self.levels[0].group_count];

        // Propagate multiplicities through each level
        for level_idx in 1..self.levels.len() {
            let level = &self.levels[level_idx];
            let mut child_multiplicities = Vec::with_capacity(level.group_count);

            // For each parent, its children inherit its multiplicity
            for (parent_idx, &parent_mult) in parent_multiplicities.iter().enumerate() {
                let child_count = if parent_idx < level.multiplicities.len() {
                    level.multiplicities[parent_idx]
                } else {
                    0
                };

                // Each child of this parent inherits the parent's multiplicity
                for _ in 0..child_count {
                    child_multiplicities.push(parent_mult);
                }
            }

            parent_multiplicities = child_multiplicities;
        }

        parent_multiplicities
    }

    /// Computes SUM on a numeric column at the deepest level without flattening.
    ///
    /// Each value is multiplied by its effective multiplicity (how many times
    /// it would appear in the flattened result).
    ///
    /// # Arguments
    ///
    /// * `column_idx` - Column index within the deepest level
    ///
    /// # Returns
    ///
    /// The sum as f64, or None if the column doesn't exist or contains non-numeric values.
    #[must_use]
    pub fn sum_deepest(&self, column_idx: usize) -> Option<f64> {
        if self.levels.is_empty() {
            return None;
        }

        let deepest_idx = self.levels.len() - 1;
        let deepest = &self.levels[deepest_idx];
        let col = deepest.column(column_idx)?;

        // Compute multiplicity for each physical value
        let multiplicities = self.compute_path_multiplicities();

        let mut sum = 0.0;
        for (phys_idx, mult) in multiplicities.iter().enumerate() {
            if let Some(value) = col.get_physical(phys_idx) {
                // Try to convert to f64
                let num_value = match &value {
                    grafeo_common::types::Value::Int64(v) => *v as f64,
                    grafeo_common::types::Value::Float64(v) => *v,
                    _ => continue, // Skip non-numeric values
                };
                sum += num_value * (*mult as f64);
            }
        }
        Some(sum)
    }

    /// Computes AVG on a numeric column at the deepest level without flattening.
    ///
    /// This is equivalent to `sum_deepest() / count_rows()`.
    ///
    /// # Arguments
    ///
    /// * `column_idx` - Column index within the deepest level
    ///
    /// # Returns
    ///
    /// The average as f64, or None if the column doesn't exist or the chunk is empty.
    #[must_use]
    pub fn avg_deepest(&self, column_idx: usize) -> Option<f64> {
        let count = self.logical_row_count();
        if count == 0 {
            return None;
        }

        let sum = self.sum_deepest(column_idx)?;
        Some(sum / count as f64)
    }

    /// Computes MIN on a column at the deepest level without flattening.
    ///
    /// Unlike SUM/AVG, MIN doesn't need multiplicities - we just find the minimum
    /// among all physical values.
    ///
    /// # Arguments
    ///
    /// * `column_idx` - Column index within the deepest level
    ///
    /// # Returns
    ///
    /// The minimum value, or None if the column doesn't exist or is empty.
    #[must_use]
    pub fn min_deepest(&self, column_idx: usize) -> Option<grafeo_common::types::Value> {
        if self.levels.is_empty() {
            return None;
        }

        let deepest_idx = self.levels.len() - 1;
        let deepest = &self.levels[deepest_idx];
        let col = deepest.column(column_idx)?;

        let mut min_value: Option<grafeo_common::types::Value> = None;

        for phys_idx in 0..col.physical_len() {
            if let Some(value) = col.get_physical(phys_idx) {
                min_value = Some(match min_value {
                    None => value,
                    Some(current) => {
                        if Self::value_less_than(&value, &current) {
                            value
                        } else {
                            current
                        }
                    }
                });
            }
        }

        min_value
    }

    /// Computes MAX on a column at the deepest level without flattening.
    ///
    /// Unlike SUM/AVG, MAX doesn't need multiplicities - we just find the maximum
    /// among all physical values.
    ///
    /// # Arguments
    ///
    /// * `column_idx` - Column index within the deepest level
    ///
    /// # Returns
    ///
    /// The maximum value, or None if the column doesn't exist or is empty.
    #[must_use]
    pub fn max_deepest(&self, column_idx: usize) -> Option<grafeo_common::types::Value> {
        if self.levels.is_empty() {
            return None;
        }

        let deepest_idx = self.levels.len() - 1;
        let deepest = &self.levels[deepest_idx];
        let col = deepest.column(column_idx)?;

        let mut max_value: Option<grafeo_common::types::Value> = None;

        for phys_idx in 0..col.physical_len() {
            if let Some(value) = col.get_physical(phys_idx) {
                max_value = Some(match max_value {
                    None => value,
                    Some(current) => {
                        if Self::value_less_than(&current, &value) {
                            value
                        } else {
                            current
                        }
                    }
                });
            }
        }

        max_value
    }

    /// Compares two Values for ordering (a < b).
    ///
    /// Comparison rules:
    /// - Null is always less than non-null
    /// - Numeric types are compared by value
    /// - Strings are compared lexicographically
    /// - Other types use debug string comparison as fallback
    fn value_less_than(a: &grafeo_common::types::Value, b: &grafeo_common::types::Value) -> bool {
        use grafeo_common::types::Value;

        match (a, b) {
            // Null handling
            (Value::Null, Value::Null) => false,
            (Value::Null, _) => true,
            (_, Value::Null) => false,

            // Numeric comparisons
            (Value::Int64(x), Value::Int64(y)) => x < y,
            (Value::Float64(x), Value::Float64(y)) => x < y,
            (Value::Int64(x), Value::Float64(y)) => (*x as f64) < *y,
            (Value::Float64(x), Value::Int64(y)) => *x < (*y as f64),

            // String comparison
            (Value::String(x), Value::String(y)) => x.as_ref() < y.as_ref(),

            // Bool comparison (false < true)
            (Value::Bool(x), Value::Bool(y)) => !x && *y,

            // Fallback for incompatible types - not comparable
            // Return false to keep the current value (arbitrary but consistent)
            _ => false,
        }
    }

    // ========================================================================
    // Projection and Column Operations
    // ========================================================================

    /// Projects specific columns from the factorized chunk without flattening.
    ///
    /// # Arguments
    ///
    /// * `column_specs` - List of (level_idx, column_idx, new_name) tuples
    ///
    /// # Returns
    ///
    /// A new FactorizedChunk with only the specified columns.
    #[must_use]
    pub fn project(&self, column_specs: &[(usize, usize, String)]) -> Self {
        if self.levels.is_empty() || column_specs.is_empty() {
            return Self::empty();
        }

        // Group specs by level
        let mut level_specs: Vec<Vec<(usize, String)>> = vec![Vec::new(); self.levels.len()];
        for (level_idx, col_idx, name) in column_specs {
            if *level_idx < self.levels.len() {
                level_specs[*level_idx].push((*col_idx, name.clone()));
            }
        }

        // Build new levels with projected columns
        let mut new_levels = Vec::new();

        for (level_idx, specs) in level_specs.iter().enumerate() {
            if specs.is_empty() {
                continue;
            }

            let src_level = &self.levels[level_idx];

            let columns: Vec<FactorizedVector> = specs
                .iter()
                .filter_map(|(col_idx, _)| src_level.column(*col_idx).cloned())
                .collect();

            let names: Vec<String> = specs.iter().map(|(_, name)| name.clone()).collect();

            if level_idx == 0 {
                new_levels.push(FactorizationLevel::flat(columns, names));
            } else {
                let mults = src_level.multiplicities().to_vec();
                new_levels.push(FactorizationLevel::unflat(columns, names, mults));
            }
        }

        if new_levels.is_empty() {
            return Self::empty();
        }

        let mut result = Self {
            levels: new_levels,
            logical_row_count: 0,
            state: ChunkState::flat(0),
        };
        result.recompute_logical_row_count();
        result.update_state();
        result
    }
}

/// Iterator over logical rows in a factorized chunk.
///
/// Instead of materializing all rows, this iterator yields index tuples
/// that can be used to access values at each level.
///
/// # Alternatives
///
/// For better performance, consider using the iterators from [`factorized_iter`](super::factorized_iter):
///
/// - [`PrecomputedIter`](super::factorized_iter::PrecomputedIter) - Pre-computes all indices
///   for O(1) random access and better cache locality
/// - [`StreamingIter`](super::factorized_iter::StreamingIter) - More memory-efficient
///   streaming iteration with SmallVec stack allocation
/// - [`RowView`](super::factorized_iter::RowView) - Zero-copy access to row values
pub struct FactorizedRowIterator<'a> {
    chunk: &'a FactorizedChunk,
    /// Current physical indices at each level.
    indices: Vec<usize>,
    /// Maximum physical index at each level (per parent).
    /// This is updated as we traverse.
    exhausted: bool,
}

impl<'a> FactorizedRowIterator<'a> {
    fn new(chunk: &'a FactorizedChunk) -> Self {
        let indices = vec![0; chunk.level_count()];
        let mut exhausted = chunk.levels.is_empty() || chunk.levels[0].group_count == 0;

        let mut iter = Self {
            chunk,
            indices,
            exhausted,
        };

        // If initial position is invalid (e.g., first parent has 0 children), advance to valid position
        if !exhausted && !iter.has_valid_deepest_range() {
            if !iter.advance() {
                exhausted = true;
            }
            iter.exhausted = exhausted;
        }

        iter
    }

    /// Advances the indices like a mixed-radix counter.
    fn advance(&mut self) -> bool {
        if self.exhausted || self.chunk.levels.is_empty() {
            return false;
        }

        // Start from the deepest level and work backwards
        for level_idx in (0..self.chunk.levels.len()).rev() {
            let level = &self.chunk.levels[level_idx];

            // Get the parent index for this level
            let parent_idx = if level_idx == 0 {
                // Level 0 has no parent - just check bounds
                self.indices[0] + 1
            } else {
                // Get current parent's physical index
                self.indices[level_idx - 1]
            };

            // Get the range of valid indices for this parent
            let (_start, end) = if level_idx == 0 {
                (0, level.group_count)
            } else {
                // For unflat levels, get range from parent
                if let Some(col) = level.columns.first() {
                    col.range_for_parent(parent_idx)
                } else {
                    (0, 0)
                }
            };

            let current = self.indices[level_idx];
            if current + 1 < end {
                // Can advance at this level
                self.indices[level_idx] = current + 1;
                // Reset all deeper levels to their start positions
                for deeper_idx in (level_idx + 1)..self.chunk.levels.len() {
                    if let Some(deeper_col) = self.chunk.levels[deeper_idx].columns.first() {
                        let (deeper_start, _) =
                            deeper_col.range_for_parent(self.indices[deeper_idx - 1]);
                        self.indices[deeper_idx] = deeper_start;
                    }
                }

                // Check if the deepest level has valid range - if any parent has 0 children,
                // we need to keep advancing instead of returning this invalid row
                if self.has_valid_deepest_range() {
                    return true;
                }
                // Otherwise, recursively try to advance again from the new position
                // This handles sparse data where many parents have 0 children
                return self.advance();
            }
            // Can't advance at this level - try parent level
        }

        // Couldn't advance at any level - exhausted
        self.exhausted = true;
        false
    }

    /// Checks if the deepest level has a valid (non-empty) range for its current parent.
    fn has_valid_deepest_range(&self) -> bool {
        if self.chunk.levels.len() <= 1 {
            return true; // Single level or empty - always valid
        }

        let deepest_idx = self.chunk.levels.len() - 1;
        let parent_idx = self.indices[deepest_idx - 1];

        if let Some(col) = self.chunk.levels[deepest_idx].columns.first() {
            let (start, end) = col.range_for_parent(parent_idx);
            start < end // Valid if range is non-empty
        } else {
            false
        }
    }
}

impl Iterator for FactorizedRowIterator<'_> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        // Return current indices, then advance
        let result = self.indices.clone();
        self.advance();
        Some(result)
    }
}

/// A chunk that can be either flat (DataChunk) or factorized (FactorizedChunk).
#[derive(Debug, Clone)]
pub enum ChunkVariant {
    /// A flat chunk with all rows materialized.
    Flat(DataChunk),
    /// A factorized chunk with multi-level representation.
    Factorized(FactorizedChunk),
}

impl ChunkVariant {
    /// Creates a flat variant from a DataChunk.
    #[must_use]
    pub fn flat(chunk: DataChunk) -> Self {
        Self::Flat(chunk)
    }

    /// Creates a factorized variant from a FactorizedChunk.
    #[must_use]
    pub fn factorized(chunk: FactorizedChunk) -> Self {
        Self::Factorized(chunk)
    }

    /// Ensures the chunk is flat, flattening if necessary.
    #[must_use]
    pub fn ensure_flat(self) -> DataChunk {
        match self {
            Self::Flat(chunk) => chunk,
            Self::Factorized(chunk) => chunk.flatten(),
        }
    }

    /// Returns the logical row count.
    #[must_use]
    pub fn logical_row_count(&self) -> usize {
        match self {
            Self::Flat(chunk) => chunk.row_count(),
            Self::Factorized(chunk) => chunk.logical_row_count(),
        }
    }

    /// Returns true if this is a factorized chunk.
    #[must_use]
    pub fn is_factorized(&self) -> bool {
        matches!(self, Self::Factorized(_))
    }

    /// Returns true if this is a flat chunk.
    #[must_use]
    pub fn is_flat(&self) -> bool {
        matches!(self, Self::Flat(_))
    }

    /// Returns true if the chunk is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.logical_row_count() == 0
    }
}

impl From<DataChunk> for ChunkVariant {
    fn from(chunk: DataChunk) -> Self {
        Self::Flat(chunk)
    }
}

impl From<FactorizedChunk> for ChunkVariant {
    fn from(chunk: FactorizedChunk) -> Self {
        Self::Factorized(chunk)
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{LogicalType, NodeId, Value};

    use super::*;

    fn make_flat_chunk() -> DataChunk {
        let mut col = ValueVector::with_type(LogicalType::Int64);
        col.push_int64(1);
        col.push_int64(2);
        DataChunk::new(vec![col])
    }

    fn create_multi_level_chunk() -> FactorizedChunk {
        // 2 sources, each with 2 neighbors = 4 logical rows
        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(10);
        sources.push_int64(20);

        let mut chunk = FactorizedChunk::with_flat_level(vec![sources], vec!["src".to_string()]);

        let mut neighbors = ValueVector::with_type(LogicalType::Int64);
        neighbors.push_int64(1);
        neighbors.push_int64(2);
        neighbors.push_int64(3);
        neighbors.push_int64(4);

        let offsets = vec![0, 2, 4];
        chunk.add_level(vec![neighbors], vec!["nbr".to_string()], &offsets);
        chunk
    }

    #[test]
    fn test_from_flat() {
        let flat = make_flat_chunk();
        let factorized = FactorizedChunk::from_flat(&flat, vec!["col1".to_string()]);

        assert_eq!(factorized.level_count(), 1);
        assert_eq!(factorized.logical_row_count(), 2);
        assert_eq!(factorized.physical_size(), 2);
    }

    #[test]
    fn test_add_level() {
        // Start with 2 source nodes
        let mut col0 = ValueVector::with_type(LogicalType::Node);
        col0.push_node_id(NodeId::new(100));
        col0.push_node_id(NodeId::new(200));

        let mut chunk = FactorizedChunk::with_flat_level(vec![col0], vec!["source".to_string()]);

        assert_eq!(chunk.level_count(), 1);
        assert_eq!(chunk.logical_row_count(), 2);

        // Add level 1: source 0 has 3 neighbors, source 1 has 2 neighbors
        let mut neighbors = ValueVector::with_type(LogicalType::Node);
        neighbors.push_node_id(NodeId::new(10));
        neighbors.push_node_id(NodeId::new(11));
        neighbors.push_node_id(NodeId::new(12));
        neighbors.push_node_id(NodeId::new(20));
        neighbors.push_node_id(NodeId::new(21));

        let offsets = vec![0, 3, 5]; // source 0: 0..3, source 1: 3..5
        chunk.add_level(vec![neighbors], vec!["neighbor".to_string()], &offsets);

        assert_eq!(chunk.level_count(), 2);
        assert_eq!(chunk.logical_row_count(), 5); // 3 + 2 neighbors
        assert_eq!(chunk.physical_size(), 2 + 5); // 2 sources + 5 neighbors
    }

    #[test]
    fn test_flatten_single_level() {
        let flat = make_flat_chunk();
        let factorized = FactorizedChunk::from_flat(&flat, vec!["col1".to_string()]);

        let flattened = factorized.flatten();
        assert_eq!(flattened.row_count(), 2);
        assert_eq!(flattened.column(0).unwrap().get_int64(0), Some(1));
        assert_eq!(flattened.column(0).unwrap().get_int64(1), Some(2));
    }

    #[test]
    fn test_flatten_multi_level() {
        // 2 sources, each with 2 neighbors = 4 logical rows
        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(1);
        sources.push_int64(2);

        let mut chunk = FactorizedChunk::with_flat_level(vec![sources], vec!["src".to_string()]);

        let mut neighbors = ValueVector::with_type(LogicalType::Int64);
        neighbors.push_int64(10);
        neighbors.push_int64(11);
        neighbors.push_int64(20);
        neighbors.push_int64(21);

        let offsets = vec![0, 2, 4];
        chunk.add_level(vec![neighbors], vec!["nbr".to_string()], &offsets);

        let flat = chunk.flatten();
        assert_eq!(flat.row_count(), 4);
        assert_eq!(flat.column_count(), 2);

        // Check that sources are duplicated correctly
        // Row 0: (1, 10), Row 1: (1, 11), Row 2: (2, 20), Row 3: (2, 21)
        assert_eq!(flat.column(0).unwrap().get_int64(0), Some(1));
        assert_eq!(flat.column(0).unwrap().get_int64(1), Some(1));
        assert_eq!(flat.column(0).unwrap().get_int64(2), Some(2));
        assert_eq!(flat.column(0).unwrap().get_int64(3), Some(2));
        assert_eq!(flat.column(1).unwrap().get_int64(0), Some(10));
        assert_eq!(flat.column(1).unwrap().get_int64(1), Some(11));
        assert_eq!(flat.column(1).unwrap().get_int64(2), Some(20));
        assert_eq!(flat.column(1).unwrap().get_int64(3), Some(21));
    }

    #[test]
    fn test_logical_row_iter_single_level() {
        let flat = make_flat_chunk();
        let factorized = FactorizedChunk::from_flat(&flat, vec!["col1".to_string()]);

        let indices: Vec<_> = factorized.logical_row_iter().collect();
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], vec![0]);
        assert_eq!(indices[1], vec![1]);
    }

    #[test]
    fn test_chunk_variant() {
        let flat = make_flat_chunk();
        let variant = ChunkVariant::flat(flat.clone());

        assert!(variant.is_flat());
        assert!(!variant.is_factorized());
        assert_eq!(variant.logical_row_count(), 2);

        let ensured = variant.ensure_flat();
        assert_eq!(ensured.row_count(), 2);
    }

    #[test]
    fn test_chunk_variant_factorized() {
        let chunk = create_multi_level_chunk();
        let variant = ChunkVariant::factorized(chunk);

        assert!(variant.is_factorized());
        assert!(!variant.is_flat());
        assert_eq!(variant.logical_row_count(), 4);

        let flat = variant.ensure_flat();
        assert_eq!(flat.row_count(), 4);
    }

    #[test]
    fn test_chunk_variant_from() {
        let flat = make_flat_chunk();
        let variant: ChunkVariant = flat.into();
        assert!(variant.is_flat());

        let factorized = create_multi_level_chunk();
        let variant2: ChunkVariant = factorized.into();
        assert!(variant2.is_factorized());
    }

    #[test]
    fn test_chunk_variant_is_empty() {
        let empty_flat = DataChunk::empty();
        let variant = ChunkVariant::flat(empty_flat);
        assert!(variant.is_empty());

        let non_empty = make_flat_chunk();
        let variant2 = ChunkVariant::flat(non_empty);
        assert!(!variant2.is_empty());
    }

    #[test]
    fn test_empty_chunk() {
        let chunk = FactorizedChunk::empty();
        assert_eq!(chunk.level_count(), 0);
        assert_eq!(chunk.logical_row_count(), 0);
        assert_eq!(chunk.physical_size(), 0);

        let flat = chunk.flatten();
        assert!(flat.is_empty());
    }

    #[test]
    fn test_all_column_names() {
        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(1);

        let mut chunk = FactorizedChunk::with_flat_level(vec![sources], vec!["source".to_string()]);

        let mut neighbors = ValueVector::with_type(LogicalType::Int64);
        neighbors.push_int64(10);

        chunk.add_level(vec![neighbors], vec!["neighbor".to_string()], &[0, 1]);

        let names = chunk.all_column_names();
        assert_eq!(names, vec!["source", "neighbor"]);
    }

    #[test]
    fn test_level_mut() {
        let mut chunk = create_multi_level_chunk();

        // Access level mutably
        let level = chunk.level_mut(0).unwrap();
        assert_eq!(level.column_count(), 1);

        // Invalid level should return None
        assert!(chunk.level_mut(10).is_none());
    }

    #[test]
    fn test_factorization_level_column_mut() {
        let mut chunk = create_multi_level_chunk();

        let level = chunk.level_mut(0).unwrap();
        let col = level.column_mut(0);
        assert!(col.is_some());

        // Invalid column should return None
        assert!(level.column_mut(10).is_none());
    }

    #[test]
    fn test_factorization_level_physical_value_count() {
        let chunk = create_multi_level_chunk();

        let level0 = chunk.level(0).unwrap();
        assert_eq!(level0.physical_value_count(), 2); // 2 sources

        let level1 = chunk.level(1).unwrap();
        assert_eq!(level1.physical_value_count(), 4); // 4 neighbors
    }

    #[test]
    fn test_count_rows() {
        let chunk = create_multi_level_chunk();
        assert_eq!(chunk.count_rows(), 4);

        let empty = FactorizedChunk::empty();
        assert_eq!(empty.count_rows(), 0);
    }

    #[test]
    fn test_compute_path_multiplicities() {
        let chunk = create_multi_level_chunk();

        let mults = chunk.compute_path_multiplicities();
        // Each value at the deepest level has multiplicity 1 since each parent has 2 children
        assert_eq!(mults.len(), 4);
        assert!(mults.iter().all(|&m| m == 1));
    }

    #[test]
    fn test_compute_path_multiplicities_single_level() {
        let mut col = ValueVector::with_type(LogicalType::Int64);
        col.push_int64(1);
        col.push_int64(2);
        col.push_int64(3);

        let chunk = FactorizedChunk::with_flat_level(vec![col], vec!["val".to_string()]);
        let mults = chunk.compute_path_multiplicities();

        // Single level: each value has multiplicity 1
        assert_eq!(mults.len(), 3);
        assert!(mults.iter().all(|&m| m == 1));
    }

    #[test]
    fn test_compute_path_multiplicities_empty() {
        let chunk = FactorizedChunk::empty();
        let mults = chunk.compute_path_multiplicities();
        assert!(mults.is_empty());
    }

    #[test]
    fn test_path_multiplicities_cached() {
        let mut chunk = create_multi_level_chunk();

        // First call computes and caches
        let mults1 = chunk.path_multiplicities_cached();
        assert_eq!(mults1.len(), 4);

        // Second call should return cached value
        let mults2 = chunk.path_multiplicities_cached();
        assert_eq!(mults1.len(), mults2.len());
    }

    #[test]
    fn test_sum_deepest() {
        let chunk = create_multi_level_chunk();

        // Deepest level has values [1, 2, 3, 4]
        let sum = chunk.sum_deepest(0);
        assert_eq!(sum, Some(10.0)); // 1 + 2 + 3 + 4
    }

    #[test]
    fn test_sum_deepest_empty() {
        let chunk = FactorizedChunk::empty();
        assert!(chunk.sum_deepest(0).is_none());
    }

    #[test]
    fn test_sum_deepest_invalid_column() {
        let chunk = create_multi_level_chunk();
        assert!(chunk.sum_deepest(10).is_none());
    }

    #[test]
    fn test_avg_deepest() {
        let chunk = create_multi_level_chunk();

        // Deepest level has values [1, 2, 3, 4], avg = 2.5
        let avg = chunk.avg_deepest(0);
        assert_eq!(avg, Some(2.5));
    }

    #[test]
    fn test_avg_deepest_empty() {
        let chunk = FactorizedChunk::empty();
        assert!(chunk.avg_deepest(0).is_none());
    }

    #[test]
    fn test_min_deepest() {
        let chunk = create_multi_level_chunk();

        let min = chunk.min_deepest(0);
        assert_eq!(min, Some(Value::Int64(1)));
    }

    #[test]
    fn test_min_deepest_empty() {
        let chunk = FactorizedChunk::empty();
        assert!(chunk.min_deepest(0).is_none());
    }

    #[test]
    fn test_min_deepest_invalid_column() {
        let chunk = create_multi_level_chunk();
        assert!(chunk.min_deepest(10).is_none());
    }

    #[test]
    fn test_max_deepest() {
        let chunk = create_multi_level_chunk();

        let max = chunk.max_deepest(0);
        assert_eq!(max, Some(Value::Int64(4)));
    }

    #[test]
    fn test_max_deepest_empty() {
        let chunk = FactorizedChunk::empty();
        assert!(chunk.max_deepest(0).is_none());
    }

    #[test]
    fn test_value_less_than() {
        // Null handling
        assert!(FactorizedChunk::value_less_than(
            &Value::Null,
            &Value::Int64(1)
        ));
        assert!(!FactorizedChunk::value_less_than(
            &Value::Int64(1),
            &Value::Null
        ));
        assert!(!FactorizedChunk::value_less_than(
            &Value::Null,
            &Value::Null
        ));

        // Int64
        assert!(FactorizedChunk::value_less_than(
            &Value::Int64(1),
            &Value::Int64(2)
        ));
        assert!(!FactorizedChunk::value_less_than(
            &Value::Int64(2),
            &Value::Int64(1)
        ));

        // Float64
        assert!(FactorizedChunk::value_less_than(
            &Value::Float64(1.5),
            &Value::Float64(2.5)
        ));

        // Mixed Int/Float
        assert!(FactorizedChunk::value_less_than(
            &Value::Int64(1),
            &Value::Float64(1.5)
        ));
        assert!(FactorizedChunk::value_less_than(
            &Value::Float64(0.5),
            &Value::Int64(1)
        ));

        // String
        assert!(FactorizedChunk::value_less_than(
            &Value::String("apple".into()),
            &Value::String("banana".into())
        ));

        // Bool (false < true)
        assert!(FactorizedChunk::value_less_than(
            &Value::Bool(false),
            &Value::Bool(true)
        ));
        assert!(!FactorizedChunk::value_less_than(
            &Value::Bool(true),
            &Value::Bool(false)
        ));

        // Incompatible types return false
        assert!(!FactorizedChunk::value_less_than(
            &Value::Int64(1),
            &Value::String("hello".into())
        ));
    }

    #[test]
    fn test_filter_deepest() {
        let chunk = create_multi_level_chunk();

        // Filter to keep only values > 2
        let filtered = chunk.filter_deepest(0, |v| {
            if let Value::Int64(n) = v {
                *n > 2
            } else {
                false
            }
        });

        let filtered = filtered.unwrap();
        assert_eq!(filtered.logical_row_count(), 2); // Only 3 and 4 remain
    }

    #[test]
    fn test_filter_deepest_empty() {
        let chunk = FactorizedChunk::empty();
        assert!(chunk.filter_deepest(0, |_| true).is_none());
    }

    #[test]
    fn test_filter_deepest_all_filtered() {
        let chunk = create_multi_level_chunk();

        // Filter everything out
        let filtered = chunk.filter_deepest(0, |_| false);

        let filtered = filtered.unwrap();
        assert_eq!(filtered.logical_row_count(), 0);
    }

    #[test]
    fn test_filter_deepest_invalid_column() {
        let chunk = create_multi_level_chunk();
        assert!(chunk.filter_deepest(10, |_| true).is_none());
    }

    #[test]
    fn test_filter_deepest_multi() {
        // Create a chunk with 2 columns at the deepest level
        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(1);

        let mut chunk = FactorizedChunk::with_flat_level(vec![sources], vec!["src".to_string()]);

        let mut col1 = ValueVector::with_type(LogicalType::Int64);
        col1.push_int64(10);
        col1.push_int64(20);
        col1.push_int64(30);

        let mut col2 = ValueVector::with_type(LogicalType::Int64);
        col2.push_int64(1);
        col2.push_int64(2);
        col2.push_int64(3);

        let offsets = vec![0, 3];
        chunk.add_level(
            vec![col1, col2],
            vec!["a".to_string(), "b".to_string()],
            &offsets,
        );

        // Filter based on both columns
        let filtered = chunk.filter_deepest_multi(|values| {
            if values.len() == 2 {
                if let (Value::Int64(a), Value::Int64(b)) = (&values[0], &values[1]) {
                    return *a + *b > 15;
                }
            }
            false
        });

        assert!(filtered.is_some());
        let filtered = filtered.unwrap();
        assert_eq!(filtered.logical_row_count(), 2); // (20,2) and (30,3) pass
    }

    #[test]
    fn test_filter_deepest_multi_empty() {
        let chunk = FactorizedChunk::empty();
        assert!(chunk.filter_deepest_multi(|_| true).is_none());
    }

    #[test]
    fn test_filter_deepest_multi_no_columns() {
        // Create a chunk with no columns at level 1
        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(1);

        let mut chunk = FactorizedChunk::with_flat_level(vec![sources], vec!["src".to_string()]);

        // Add empty level (edge case)
        let empty_level = FactorizationLevel::unflat(vec![], vec![], vec![0]);
        chunk.add_factorized_level(empty_level);

        assert!(chunk.filter_deepest_multi(|_| true).is_none());
    }

    #[test]
    fn test_project() {
        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(1);
        sources.push_int64(2);

        let mut col2 = ValueVector::with_type(LogicalType::String);
        col2.push_string("a");
        col2.push_string("b");

        let chunk = FactorizedChunk::with_flat_level(
            vec![sources, col2],
            vec!["num".to_string(), "str".to_string()],
        );

        // Project only the first column
        let projected = chunk.project(&[(0, 0, "projected_num".to_string())]);

        assert_eq!(projected.total_column_count(), 1);
        let names = projected.all_column_names();
        assert_eq!(names, vec!["projected_num"]);
    }

    #[test]
    fn test_project_empty() {
        let chunk = FactorizedChunk::empty();
        let projected = chunk.project(&[(0, 0, "col".to_string())]);
        assert_eq!(projected.level_count(), 0);
    }

    #[test]
    fn test_project_empty_specs() {
        let chunk = create_multi_level_chunk();
        let projected = chunk.project(&[]);
        assert_eq!(projected.level_count(), 0);
    }

    #[test]
    fn test_project_invalid_level() {
        let chunk = create_multi_level_chunk();

        // Project from invalid level
        let projected = chunk.project(&[(10, 0, "col".to_string())]);
        assert_eq!(projected.level_count(), 0);
    }

    #[test]
    fn test_project_multi_level() {
        let chunk = create_multi_level_chunk();

        // Project from both levels
        let projected =
            chunk.project(&[(0, 0, "source".to_string()), (1, 0, "neighbor".to_string())]);

        assert_eq!(projected.level_count(), 2);
        assert_eq!(projected.total_column_count(), 2);
    }

    #[test]
    fn test_total_column_count() {
        let chunk = create_multi_level_chunk();
        assert_eq!(chunk.total_column_count(), 2); // 1 at level 0, 1 at level 1
    }

    #[test]
    fn test_chunk_state_access() {
        let mut chunk = create_multi_level_chunk();

        let state = chunk.chunk_state();
        assert!(state.is_factorized());

        let state_mut = chunk.chunk_state_mut();
        state_mut.invalidate_cache();
    }

    #[test]
    fn test_logical_row_iter_multi_level() {
        let chunk = create_multi_level_chunk();

        let indices: Vec<_> = chunk.logical_row_iter().collect();
        assert_eq!(indices.len(), 4);

        // Verify structure: [source_idx, neighbor_idx]
        assert_eq!(indices[0], vec![0, 0]);
        assert_eq!(indices[1], vec![0, 1]);
        assert_eq!(indices[2], vec![1, 2]);
        assert_eq!(indices[3], vec![1, 3]);
    }

    #[test]
    fn test_sum_deepest_with_float() {
        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(1);

        let mut chunk = FactorizedChunk::with_flat_level(vec![sources], vec!["src".to_string()]);

        let mut floats = ValueVector::with_type(LogicalType::Float64);
        floats.push_float64(1.5);
        floats.push_float64(2.5);
        floats.push_float64(3.0);

        chunk.add_level(vec![floats], vec!["val".to_string()], &[0, 3]);

        let sum = chunk.sum_deepest(0);
        assert_eq!(sum, Some(7.0)); // 1.5 + 2.5 + 3.0
    }

    #[test]
    fn test_min_max_with_strings() {
        let mut sources = ValueVector::with_type(LogicalType::Int64);
        sources.push_int64(1);

        let mut chunk = FactorizedChunk::with_flat_level(vec![sources], vec!["src".to_string()]);

        let mut strings = ValueVector::with_type(LogicalType::String);
        strings.push_string("banana");
        strings.push_string("apple");
        strings.push_string("cherry");

        chunk.add_level(vec![strings], vec!["fruit".to_string()], &[0, 3]);

        let min = chunk.min_deepest(0);
        assert_eq!(min, Some(Value::String("apple".into())));

        let max = chunk.max_deepest(0);
        assert_eq!(max, Some(Value::String("cherry".into())));
    }

    #[test]
    fn test_recompute_logical_row_count_empty() {
        let mut chunk = FactorizedChunk::empty();
        chunk.recompute_logical_row_count();
        assert_eq!(chunk.logical_row_count(), 0);
    }

    #[test]
    fn test_factorization_level_group_count() {
        let chunk = create_multi_level_chunk();

        let level0 = chunk.level(0).unwrap();
        assert_eq!(level0.group_count(), 2);

        let level1 = chunk.level(1).unwrap();
        assert_eq!(level1.group_count(), 4);
    }

    #[test]
    fn test_factorization_level_multiplicities() {
        let chunk = create_multi_level_chunk();

        let level1 = chunk.level(1).unwrap();
        let mults = level1.multiplicities();
        assert_eq!(mults, &[2, 2]); // Each source has 2 neighbors
    }

    #[test]
    fn test_factorization_level_column_names() {
        let chunk = create_multi_level_chunk();

        let level0 = chunk.level(0).unwrap();
        assert_eq!(level0.column_names(), &["src"]);

        let level1 = chunk.level(1).unwrap();
        assert_eq!(level1.column_names(), &["nbr"]);
    }
}
