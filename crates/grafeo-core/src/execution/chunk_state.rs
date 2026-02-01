//! Unified chunk state tracking for factorized execution.
//!
//! This module provides [`ChunkState`] for centralized state management,
//! inspired by LadybugDB's `FStateType` pattern. Key benefits:
//!
//! - **Cached multiplicities**: Computed once, reused for all aggregates
//! - **Selection integration**: Lazy filtering without data copying
//! - **O(1) logical row count**: Cached, not recomputed
//!
//! # Example
//!
//! ```ignore
//! let mut state = ChunkState::unflat(3, 1000);
//!
//! // First access computes, subsequent accesses use cache
//! let mults = state.get_or_compute_multiplicities(|| expensive_compute());
//! let mults2 = state.get_or_compute_multiplicities(|| unreachable!());
//! assert!(std::ptr::eq(mults.as_ptr(), mults2.as_ptr()));
//! ```

use std::sync::Arc;

use super::selection::SelectionVector;

/// Factorization state of a chunk (flat vs factorized).
///
/// Similar to LadybugDB's `FStateType`, this provides a single source
/// of truth for the chunk's factorization status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorizationState {
    /// All vectors are flat - one value per logical row.
    ///
    /// This is the state after flattening or for simple scans.
    Flat {
        /// Number of rows (physical = logical).
        row_count: usize,
    },
    /// One or more vectors are unflat - values grouped by parent.
    ///
    /// The chunk has multi-level structure.
    Unflat {
        /// Number of factorization levels.
        level_count: usize,
        /// Total logical row count (cached, not recomputed).
        logical_rows: usize,
    },
}

impl FactorizationState {
    /// Returns true if this is a flat state.
    #[must_use]
    pub fn is_flat(&self) -> bool {
        matches!(self, Self::Flat { .. })
    }

    /// Returns true if this is an unflat (factorized) state.
    #[must_use]
    pub fn is_unflat(&self) -> bool {
        matches!(self, Self::Unflat { .. })
    }

    /// Returns the logical row count.
    #[must_use]
    pub fn logical_row_count(&self) -> usize {
        match self {
            Self::Flat { row_count } => *row_count,
            Self::Unflat { logical_rows, .. } => *logical_rows,
        }
    }

    /// Returns the number of factorization levels.
    #[must_use]
    pub fn level_count(&self) -> usize {
        match self {
            Self::Flat { .. } => 1,
            Self::Unflat { level_count, .. } => *level_count,
        }
    }
}

/// Selection state for a single factorization level.
///
/// Supports both sparse (for low selectivity) and dense (for high selectivity)
/// representations to optimize memory usage.
#[derive(Debug, Clone)]
pub enum LevelSelection {
    /// All values at this level are selected.
    All {
        /// Total count of values at this level.
        count: usize,
    },
    /// Only specific indices are selected (for low selectivity).
    ///
    /// Uses `SelectionVector` which stores indices as `u16`.
    Sparse(SelectionVector),
}

impl LevelSelection {
    /// Creates a selection that selects all values.
    #[must_use]
    pub fn all(count: usize) -> Self {
        Self::All { count }
    }

    /// Creates a sparse selection from a predicate.
    #[must_use]
    pub fn from_predicate<F>(count: usize, predicate: F) -> Self
    where
        F: Fn(usize) -> bool,
    {
        let selected = SelectionVector::from_predicate(count, predicate);
        if selected.len() == count {
            Self::All { count }
        } else {
            Self::Sparse(selected)
        }
    }

    /// Returns the number of selected values.
    #[must_use]
    pub fn selected_count(&self) -> usize {
        match self {
            Self::All { count } => *count,
            Self::Sparse(sel) => sel.len(),
        }
    }

    /// Returns true if a physical index is selected.
    #[must_use]
    pub fn is_selected(&self, physical_idx: usize) -> bool {
        match self {
            Self::All { count } => physical_idx < *count,
            Self::Sparse(sel) => sel.contains(physical_idx),
        }
    }

    /// Filters this selection with a predicate, returning a new selection.
    #[must_use]
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(usize) -> bool,
    {
        match self {
            Self::All { count } => Self::from_predicate(*count, predicate),
            Self::Sparse(sel) => {
                let filtered = sel.filter(predicate);
                Self::Sparse(filtered)
            }
        }
    }

    /// Returns an iterator over selected indices.
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match self {
            Self::All { count } => Box::new(0..*count),
            Self::Sparse(sel) => Box::new(sel.iter()),
        }
    }
}

/// Hierarchical selection for factorized data.
///
/// Tracks selections at each factorization level, enabling filtering
/// without flattening or copying data.
#[derive(Debug, Clone)]
pub struct FactorizedSelection {
    /// Selection state per level (level 0 = sources, higher = more nested).
    level_selections: Vec<LevelSelection>,
    /// Cached logical row count after selection (lazily computed).
    cached_selected_count: Option<usize>,
}

impl FactorizedSelection {
    /// Creates a selection that selects all values at all levels.
    #[must_use]
    pub fn all(level_counts: &[usize]) -> Self {
        let level_selections = level_counts
            .iter()
            .map(|&count| LevelSelection::all(count))
            .collect();
        Self {
            level_selections,
            cached_selected_count: None,
        }
    }

    /// Creates a selection from level selections.
    #[must_use]
    pub fn new(level_selections: Vec<LevelSelection>) -> Self {
        Self {
            level_selections,
            cached_selected_count: None,
        }
    }

    /// Returns the number of levels.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.level_selections.len()
    }

    /// Gets the selection for a specific level.
    #[must_use]
    pub fn level(&self, level: usize) -> Option<&LevelSelection> {
        self.level_selections.get(level)
    }

    /// Filters at a specific level using a predicate.
    ///
    /// Returns a new selection with the filter applied.
    /// This is O(n_physical) where n is the physical size of that level,
    /// not O(n_logical) where n is the logical row count.
    #[must_use]
    pub fn filter_level<F>(&self, level: usize, predicate: F) -> Self
    where
        F: Fn(usize) -> bool,
    {
        let mut new_selections = self.level_selections.clone();

        if let Some(sel) = new_selections.get_mut(level) {
            *sel = sel.filter(predicate);
        }

        Self {
            level_selections: new_selections,
            cached_selected_count: None, // Invalidate cache
        }
    }

    /// Checks if a physical index at a level is selected.
    #[must_use]
    pub fn is_selected(&self, level: usize, physical_idx: usize) -> bool {
        self.level_selections
            .get(level)
            .is_some_and(|sel| sel.is_selected(physical_idx))
    }

    /// Computes and caches the selected logical row count.
    ///
    /// The computation considers parent-child relationships:
    /// a child is only counted if its parent is selected.
    pub fn selected_count(&mut self, multiplicities: &[Vec<usize>]) -> usize {
        if let Some(count) = self.cached_selected_count {
            return count;
        }

        let count = self.compute_selected_count(multiplicities);
        self.cached_selected_count = Some(count);
        count
    }

    /// Computes the selected count without caching.
    fn compute_selected_count(&self, multiplicities: &[Vec<usize>]) -> usize {
        if self.level_selections.is_empty() {
            return 0;
        }

        // For single level, just count selected
        if self.level_selections.len() == 1 {
            return self.level_selections[0].selected_count();
        }

        // For multi-level, we need to propagate selection through levels
        // Start with level 0 selection
        let mut parent_selected: Vec<bool> = match &self.level_selections[0] {
            LevelSelection::All { count } => vec![true; *count],
            LevelSelection::Sparse(sel) => {
                let max_idx = sel.iter().max().unwrap_or(0);
                let mut selected = vec![false; max_idx + 1];
                for idx in sel.iter() {
                    selected[idx] = true;
                }
                selected
            }
        };

        // Propagate through subsequent levels
        for (level_sel, level_mults) in self
            .level_selections
            .iter()
            .skip(1)
            .zip(multiplicities.iter().skip(1))
        {
            let mut child_selected = Vec::new();
            let mut child_idx = 0;

            for (parent_idx, &mult) in level_mults.iter().enumerate() {
                let parent_is_selected = parent_selected.get(parent_idx).copied().unwrap_or(false);

                for _ in 0..mult {
                    let child_is_selected = parent_is_selected && level_sel.is_selected(child_idx);
                    child_selected.push(child_is_selected);
                    child_idx += 1;
                }
            }

            parent_selected = child_selected;
        }

        // Count final selected
        parent_selected.iter().filter(|&&s| s).count()
    }

    /// Invalidates the cached selected count.
    pub fn invalidate_cache(&mut self) {
        self.cached_selected_count = None;
    }
}

/// Unified chunk state tracking metadata for factorized execution.
///
/// This replaces scattered state tracking with a centralized structure
/// that is updated incrementally rather than recomputed.
///
/// # Key Features
///
/// - **Cached multiplicities**: Computed once per chunk, reused for all aggregates
/// - **Selection integration**: Supports lazy filtering without data copying
/// - **Generation tracking**: Enables cache invalidation on structure changes
#[derive(Debug, Clone)]
pub struct ChunkState {
    /// Factorization state of this chunk.
    state: FactorizationState,
    /// Selection for filtering without data copying.
    /// When Some, only selected indices are "active".
    selection: Option<FactorizedSelection>,
    /// Cached path multiplicities (invalidated on structure change).
    /// Key optimization: computed once, reused for all aggregates.
    cached_multiplicities: Option<Arc<[usize]>>,
    /// Generation counter for cache invalidation.
    generation: u64,
}

impl ChunkState {
    /// Creates a new flat chunk state.
    #[must_use]
    pub fn flat(row_count: usize) -> Self {
        Self {
            state: FactorizationState::Flat { row_count },
            selection: None,
            cached_multiplicities: None,
            generation: 0,
        }
    }

    /// Creates an unflat (factorized) chunk state.
    #[must_use]
    pub fn unflat(level_count: usize, logical_rows: usize) -> Self {
        Self {
            state: FactorizationState::Unflat {
                level_count,
                logical_rows,
            },
            selection: None,
            cached_multiplicities: None,
            generation: 0,
        }
    }

    /// Returns the factorization state.
    #[must_use]
    pub fn factorization_state(&self) -> FactorizationState {
        self.state
    }

    /// Returns true if this chunk is flat.
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.state.is_flat()
    }

    /// Returns true if this chunk is factorized (unflat).
    #[must_use]
    pub fn is_factorized(&self) -> bool {
        self.state.is_unflat()
    }

    /// Returns the logical row count.
    ///
    /// If a selection is active, this returns the base logical row count
    /// (selection count must be computed separately with multiplicities).
    #[must_use]
    pub fn logical_row_count(&self) -> usize {
        self.state.logical_row_count()
    }

    /// Returns the number of factorization levels.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.state.level_count()
    }

    /// Returns the current generation (for cache validation).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the selection, if any.
    #[must_use]
    pub fn selection(&self) -> Option<&FactorizedSelection> {
        self.selection.as_ref()
    }

    /// Returns mutable access to the selection.
    pub fn selection_mut(&mut self) -> &mut Option<FactorizedSelection> {
        &mut self.selection
    }

    /// Sets the selection.
    pub fn set_selection(&mut self, selection: FactorizedSelection) {
        self.selection = Some(selection);
        // Don't invalidate multiplicity cache - selection is orthogonal
    }

    /// Clears the selection.
    pub fn clear_selection(&mut self) {
        self.selection = None;
    }

    /// Updates the state (e.g., after adding a level).
    pub fn set_state(&mut self, state: FactorizationState) {
        self.state = state;
        self.invalidate_cache();
    }

    /// Invalidates cached data (call when structure changes).
    pub fn invalidate_cache(&mut self) {
        self.cached_multiplicities = None;
        self.generation += 1;
    }

    /// Gets cached multiplicities, or computes and caches them.
    ///
    /// This is the key optimization: multiplicities are computed once
    /// and reused for all aggregates (COUNT, SUM, AVG, etc.).
    ///
    /// # Arguments
    ///
    /// * `compute` - Function to compute multiplicities if not cached
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mults = state.get_or_compute_multiplicities(|| {
    ///     chunk.compute_path_multiplicities_impl()
    /// });
    /// ```
    pub fn get_or_compute_multiplicities<F>(&mut self, compute: F) -> Arc<[usize]>
    where
        F: FnOnce() -> Vec<usize>,
    {
        if let Some(ref cached) = self.cached_multiplicities {
            return Arc::clone(cached);
        }

        let mults: Arc<[usize]> = compute().into();
        self.cached_multiplicities = Some(Arc::clone(&mults));
        mults
    }

    /// Returns cached multiplicities without computing.
    ///
    /// Returns None if not yet computed.
    #[must_use]
    pub fn cached_multiplicities(&self) -> Option<&Arc<[usize]>> {
        self.cached_multiplicities.as_ref()
    }

    /// Sets the cached multiplicities directly.
    ///
    /// Useful when multiplicities are computed externally.
    pub fn set_cached_multiplicities(&mut self, mults: Arc<[usize]>) {
        self.cached_multiplicities = Some(mults);
    }
}

impl Default for ChunkState {
    fn default() -> Self {
        Self::flat(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorization_state_flat() {
        let state = FactorizationState::Flat { row_count: 100 };
        assert!(state.is_flat());
        assert!(!state.is_unflat());
        assert_eq!(state.logical_row_count(), 100);
        assert_eq!(state.level_count(), 1);
    }

    #[test]
    fn test_factorization_state_unflat() {
        let state = FactorizationState::Unflat {
            level_count: 3,
            logical_rows: 1000,
        };
        assert!(!state.is_flat());
        assert!(state.is_unflat());
        assert_eq!(state.logical_row_count(), 1000);
        assert_eq!(state.level_count(), 3);
    }

    #[test]
    fn test_level_selection_all() {
        let sel = LevelSelection::all(10);
        assert_eq!(sel.selected_count(), 10);
        for i in 0..10 {
            assert!(sel.is_selected(i));
        }
        assert!(!sel.is_selected(10));
    }

    #[test]
    fn test_level_selection_filter() {
        let sel = LevelSelection::all(10);
        let filtered = sel.filter(|i| i % 2 == 0);
        assert_eq!(filtered.selected_count(), 5);
        assert!(filtered.is_selected(0));
        assert!(!filtered.is_selected(1));
        assert!(filtered.is_selected(2));
    }

    #[test]
    fn test_level_selection_filter_sparse() {
        // Start with sparse selection
        let sel = LevelSelection::from_predicate(10, |i| i < 5);
        assert_eq!(sel.selected_count(), 5);

        // Filter the sparse selection further
        let filtered = sel.filter(|i| i % 2 == 0);
        // Only 0, 2, 4 pass both filters
        assert_eq!(filtered.selected_count(), 3);
        assert!(filtered.is_selected(0));
        assert!(!filtered.is_selected(1));
        assert!(filtered.is_selected(2));
    }

    #[test]
    fn test_level_selection_iter_all() {
        let sel = LevelSelection::all(5);
        let indices: Vec<usize> = sel.iter().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_level_selection_iter_sparse() {
        let sel = LevelSelection::from_predicate(10, |i| i % 3 == 0);
        let indices: Vec<usize> = sel.iter().collect();
        assert_eq!(indices, vec![0, 3, 6, 9]);
    }

    #[test]
    fn test_level_selection_from_predicate_all_selected() {
        // When predicate selects everything, should return All variant
        let sel = LevelSelection::from_predicate(5, |_| true);
        assert_eq!(sel.selected_count(), 5);
        match sel {
            LevelSelection::All { count } => assert_eq!(count, 5),
            LevelSelection::Sparse(_) => panic!("Expected All variant"),
        }
    }

    #[test]
    fn test_level_selection_from_predicate_partial() {
        // When predicate selects some, should return Sparse variant
        let sel = LevelSelection::from_predicate(10, |i| i < 3);
        assert_eq!(sel.selected_count(), 3);
        match sel {
            LevelSelection::Sparse(_) => {}
            LevelSelection::All { .. } => panic!("Expected Sparse variant"),
        }
    }

    #[test]
    fn test_factorized_selection_all() {
        let sel = FactorizedSelection::all(&[10, 100, 1000]);
        assert_eq!(sel.level_count(), 3);
        assert!(sel.is_selected(0, 5));
        assert!(sel.is_selected(1, 50));
        assert!(sel.is_selected(2, 500));
    }

    #[test]
    fn test_factorized_selection_new() {
        let level_sels = vec![
            LevelSelection::all(5),
            LevelSelection::from_predicate(10, |i| i < 3),
        ];
        let sel = FactorizedSelection::new(level_sels);

        assert_eq!(sel.level_count(), 2);
        assert!(sel.is_selected(0, 4));
        assert!(sel.is_selected(1, 2));
        assert!(!sel.is_selected(1, 5));
    }

    #[test]
    fn test_factorized_selection_filter_level() {
        let sel = FactorizedSelection::all(&[10, 100]);
        let filtered = sel.filter_level(1, |i| i < 50);

        assert!(filtered.is_selected(0, 5)); // Level 0 unchanged
        assert!(filtered.is_selected(1, 25)); // Level 1: 25 < 50
        assert!(!filtered.is_selected(1, 75)); // Level 1: 75 >= 50
    }

    #[test]
    fn test_factorized_selection_filter_level_invalid() {
        let sel = FactorizedSelection::all(&[10, 100]);

        // Filtering a non-existent level should not panic
        let filtered = sel.filter_level(5, |_| true);
        assert_eq!(filtered.level_count(), 2);
    }

    #[test]
    fn test_factorized_selection_is_selected_invalid_level() {
        let sel = FactorizedSelection::all(&[10]);
        assert!(!sel.is_selected(5, 0)); // Non-existent level
    }

    #[test]
    fn test_factorized_selection_level() {
        let sel = FactorizedSelection::all(&[10, 20]);

        let level0 = sel.level(0);
        assert!(level0.is_some());
        assert_eq!(level0.unwrap().selected_count(), 10);

        let level1 = sel.level(1);
        assert!(level1.is_some());
        assert_eq!(level1.unwrap().selected_count(), 20);

        assert!(sel.level(5).is_none());
    }

    #[test]
    fn test_factorized_selection_selected_count_single_level() {
        let mut sel = FactorizedSelection::all(&[10]);
        let multiplicities: Vec<Vec<usize>> = vec![vec![1; 10]];

        let count = sel.selected_count(&multiplicities);
        assert_eq!(count, 10);
    }

    #[test]
    fn test_factorized_selection_selected_count_multi_level() {
        let level_sels = vec![
            LevelSelection::all(2),                            // 2 parents
            LevelSelection::from_predicate(4, |i| i % 2 == 0), // Select indices 0, 2
        ];
        let mut sel = FactorizedSelection::new(level_sels);

        // Parent 0 has 2 children (indices 0, 1)
        // Parent 1 has 2 children (indices 2, 3)
        let multiplicities = vec![
            vec![1, 1], // Level 0 multiplicities
            vec![2, 2], // Level 1 multiplicities (children per parent)
        ];

        let count = sel.selected_count(&multiplicities);
        // Parent 0 is selected, children 0 is selected (1 passes)
        // Parent 1 is selected, child 2 is selected (1 passes)
        assert_eq!(count, 2);
    }

    #[test]
    fn test_factorized_selection_selected_count_cached() {
        let mut sel = FactorizedSelection::all(&[5]);
        let multiplicities: Vec<Vec<usize>> = vec![vec![1; 5]];

        // First call computes
        let count1 = sel.selected_count(&multiplicities);
        assert_eq!(count1, 5);

        // Second call uses cache
        let count2 = sel.selected_count(&multiplicities);
        assert_eq!(count2, 5);
    }

    #[test]
    fn test_factorized_selection_selected_count_empty() {
        let mut sel = FactorizedSelection::all(&[]);
        let multiplicities: Vec<Vec<usize>> = vec![];

        assert_eq!(sel.selected_count(&multiplicities), 0);
    }

    #[test]
    fn test_factorized_selection_invalidate_cache() {
        let mut sel = FactorizedSelection::all(&[5]);
        let multiplicities: Vec<Vec<usize>> = vec![vec![1; 5]];

        // Compute and cache
        let _ = sel.selected_count(&multiplicities);

        // Invalidate
        sel.invalidate_cache();

        // Should recompute (no way to verify, but shouldn't crash)
        let _ = sel.selected_count(&multiplicities);
    }

    #[test]
    fn test_chunk_state_flat() {
        let state = ChunkState::flat(100);
        assert!(state.is_flat());
        assert!(!state.is_factorized());
        assert_eq!(state.logical_row_count(), 100);
        assert_eq!(state.level_count(), 1);
    }

    #[test]
    fn test_chunk_state_unflat() {
        let state = ChunkState::unflat(3, 1000);
        assert!(!state.is_flat());
        assert!(state.is_factorized());
        assert_eq!(state.logical_row_count(), 1000);
        assert_eq!(state.level_count(), 3);
    }

    #[test]
    fn test_chunk_state_factorization_state() {
        let state = ChunkState::flat(50);
        let fs = state.factorization_state();
        assert!(fs.is_flat());
    }

    #[test]
    fn test_chunk_state_selection() {
        let mut state = ChunkState::unflat(2, 100);

        // Initially no selection
        assert!(state.selection().is_none());

        // Set selection
        let sel = FactorizedSelection::all(&[10, 100]);
        state.set_selection(sel);

        assert!(state.selection().is_some());
        assert_eq!(state.selection().unwrap().level_count(), 2);
    }

    #[test]
    fn test_chunk_state_selection_mut() {
        let mut state = ChunkState::unflat(2, 100);

        // Set selection
        let sel = FactorizedSelection::all(&[10, 100]);
        state.set_selection(sel);

        // Get mutable access
        let sel_mut = state.selection_mut();
        assert!(sel_mut.is_some());

        // Clear via mutable access
        *sel_mut = None;
        assert!(state.selection().is_none());
    }

    #[test]
    fn test_chunk_state_clear_selection() {
        let mut state = ChunkState::unflat(2, 100);

        let sel = FactorizedSelection::all(&[10, 100]);
        state.set_selection(sel);
        assert!(state.selection().is_some());

        state.clear_selection();
        assert!(state.selection().is_none());
    }

    #[test]
    fn test_chunk_state_set_state() {
        let mut state = ChunkState::flat(100);
        assert!(state.is_flat());
        assert_eq!(state.generation(), 0);

        state.set_state(FactorizationState::Unflat {
            level_count: 2,
            logical_rows: 200,
        });

        assert!(state.is_factorized());
        assert_eq!(state.logical_row_count(), 200);
        assert_eq!(state.generation(), 1); // Cache invalidated
    }

    #[test]
    fn test_chunk_state_caching() {
        let mut state = ChunkState::unflat(2, 100);

        // First call should compute
        let mut computed = false;
        let mults1 = state.get_or_compute_multiplicities(|| {
            computed = true;
            vec![1, 2, 3, 4, 5]
        });
        assert!(computed);
        assert_eq!(mults1.len(), 5);

        // Second call should use cache
        computed = false;
        let mults2 = state.get_or_compute_multiplicities(|| {
            computed = true;
            vec![99, 99, 99]
        });
        assert!(!computed);
        assert_eq!(mults2.len(), 5); // Same as before

        // After invalidation, should recompute
        state.invalidate_cache();
        let mults3 = state.get_or_compute_multiplicities(|| {
            computed = true;
            vec![10, 20, 30]
        });
        assert!(computed);
        assert_eq!(mults3.len(), 3);
    }

    #[test]
    fn test_chunk_state_cached_multiplicities() {
        let mut state = ChunkState::unflat(2, 100);

        // Initially not cached
        assert!(state.cached_multiplicities().is_none());

        // Compute multiplicities
        let _ = state.get_or_compute_multiplicities(|| vec![1, 2, 3]);

        // Now cached
        assert!(state.cached_multiplicities().is_some());
        assert_eq!(state.cached_multiplicities().unwrap().len(), 3);
    }

    #[test]
    fn test_chunk_state_set_cached_multiplicities() {
        let mut state = ChunkState::unflat(2, 100);

        let mults: Arc<[usize]> = vec![5, 10, 15].into();
        state.set_cached_multiplicities(mults);

        assert!(state.cached_multiplicities().is_some());
        assert_eq!(state.cached_multiplicities().unwrap().len(), 3);
    }

    #[test]
    fn test_chunk_state_generation() {
        let mut state = ChunkState::flat(100);
        assert_eq!(state.generation(), 0);

        state.invalidate_cache();
        assert_eq!(state.generation(), 1);

        state.set_state(FactorizationState::Unflat {
            level_count: 2,
            logical_rows: 200,
        });
        assert_eq!(state.generation(), 2);
    }

    #[test]
    fn test_chunk_state_default() {
        let state = ChunkState::default();
        assert!(state.is_flat());
        assert_eq!(state.logical_row_count(), 0);
        assert_eq!(state.generation(), 0);
    }
}
