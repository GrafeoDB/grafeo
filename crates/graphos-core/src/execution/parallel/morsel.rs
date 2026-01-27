//! Morsel type for parallel execution units.
//!
//! A morsel represents a chunk of work (rows) to be processed by a worker thread.
//! Morsels are larger than DataChunks (64K vs 2K rows) to amortize scheduling overhead.

use graphos_common::memory::buffer::PressureLevel;

/// Default morsel size (64K rows).
///
/// This is larger than the typical DataChunk size to amortize scheduling overhead
/// while still providing enough parallelism opportunities.
pub const DEFAULT_MORSEL_SIZE: usize = 65536;

/// Minimum morsel size under memory pressure.
pub const MIN_MORSEL_SIZE: usize = 1024;

/// Morsel size under moderate memory pressure.
pub const MODERATE_PRESSURE_MORSEL_SIZE: usize = 32768;

/// Morsel size under high memory pressure.
pub const HIGH_PRESSURE_MORSEL_SIZE: usize = 16384;

/// Morsel size under critical memory pressure.
pub const CRITICAL_PRESSURE_MORSEL_SIZE: usize = MIN_MORSEL_SIZE;

/// A morsel represents a unit of work for parallel execution.
///
/// Each morsel identifies a range of rows from a source to be processed
/// by a single worker thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Morsel {
    /// Unique identifier for this morsel within a pipeline execution.
    pub id: usize,
    /// Source partition identifier (for multi-source queries).
    pub source_id: usize,
    /// Starting row index (inclusive).
    pub start_row: usize,
    /// Ending row index (exclusive).
    pub end_row: usize,
}

impl Morsel {
    /// Creates a new morsel.
    #[must_use]
    pub fn new(id: usize, source_id: usize, start_row: usize, end_row: usize) -> Self {
        Self {
            id,
            source_id,
            start_row,
            end_row,
        }
    }

    /// Returns the number of rows in this morsel.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.end_row.saturating_sub(self.start_row)
    }

    /// Returns whether this morsel is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.row_count() == 0
    }

    /// Splits this morsel into two at the given row offset.
    ///
    /// Returns `None` if the split point is outside the morsel range.
    #[must_use]
    pub fn split_at(&self, offset: usize) -> Option<(Morsel, Morsel)> {
        let split_row = self.start_row + offset;
        if split_row <= self.start_row || split_row >= self.end_row {
            return None;
        }

        let first = Morsel {
            id: self.id,
            source_id: self.source_id,
            start_row: self.start_row,
            end_row: split_row,
        };

        let second = Morsel {
            id: self.id + 1, // New ID for split morsel
            source_id: self.source_id,
            start_row: split_row,
            end_row: self.end_row,
        };

        Some((first, second))
    }
}

/// Computes the optimal morsel size based on memory pressure.
///
/// Under memory pressure, smaller morsels allow more fine-grained
/// control over memory usage and enable earlier spilling.
#[must_use]
pub fn compute_morsel_size(pressure_level: PressureLevel) -> usize {
    match pressure_level {
        PressureLevel::Normal => DEFAULT_MORSEL_SIZE,
        PressureLevel::Moderate => MODERATE_PRESSURE_MORSEL_SIZE,
        PressureLevel::High => HIGH_PRESSURE_MORSEL_SIZE,
        PressureLevel::Critical => CRITICAL_PRESSURE_MORSEL_SIZE,
    }
}

/// Computes the optimal morsel size with a custom base size.
#[must_use]
pub fn compute_morsel_size_with_base(base_size: usize, pressure_level: PressureLevel) -> usize {
    let factor = match pressure_level {
        PressureLevel::Normal => 1.0,
        PressureLevel::Moderate => 0.5,
        PressureLevel::High => 0.25,
        PressureLevel::Critical => MIN_MORSEL_SIZE as f64 / base_size as f64,
    };

    ((base_size as f64 * factor) as usize).max(MIN_MORSEL_SIZE)
}

/// Generates morsels for a given total row count.
///
/// Returns a vector of morsels that together cover all rows.
#[must_use]
pub fn generate_morsels(total_rows: usize, morsel_size: usize, source_id: usize) -> Vec<Morsel> {
    if total_rows == 0 || morsel_size == 0 {
        return Vec::new();
    }

    let num_morsels = (total_rows + morsel_size - 1) / morsel_size;
    let mut morsels = Vec::with_capacity(num_morsels);

    for (id, start) in (0..total_rows).step_by(morsel_size).enumerate() {
        let end = (start + morsel_size).min(total_rows);
        morsels.push(Morsel::new(id, source_id, start, end));
    }

    morsels
}

/// Generates morsels with adaptive sizing based on memory pressure.
#[must_use]
pub fn generate_adaptive_morsels(
    total_rows: usize,
    pressure_level: PressureLevel,
    source_id: usize,
) -> Vec<Morsel> {
    let morsel_size = compute_morsel_size(pressure_level);
    generate_morsels(total_rows, morsel_size, source_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morsel_creation() {
        let morsel = Morsel::new(0, 1, 0, 1000);
        assert_eq!(morsel.id, 0);
        assert_eq!(morsel.source_id, 1);
        assert_eq!(morsel.start_row, 0);
        assert_eq!(morsel.end_row, 1000);
        assert_eq!(morsel.row_count(), 1000);
        assert!(!morsel.is_empty());
    }

    #[test]
    fn test_morsel_empty() {
        let morsel = Morsel::new(0, 0, 100, 100);
        assert!(morsel.is_empty());
        assert_eq!(morsel.row_count(), 0);
    }

    #[test]
    fn test_morsel_split() {
        let morsel = Morsel::new(0, 0, 0, 1000);

        // Valid split
        let (first, second) = morsel.split_at(400).unwrap();
        assert_eq!(first.start_row, 0);
        assert_eq!(first.end_row, 400);
        assert_eq!(second.start_row, 400);
        assert_eq!(second.end_row, 1000);

        // Invalid splits
        assert!(morsel.split_at(0).is_none());
        assert!(morsel.split_at(1000).is_none());
        assert!(morsel.split_at(1500).is_none());
    }

    #[test]
    fn test_compute_morsel_size() {
        assert_eq!(compute_morsel_size(PressureLevel::Normal), DEFAULT_MORSEL_SIZE);
        assert_eq!(compute_morsel_size(PressureLevel::Moderate), MODERATE_PRESSURE_MORSEL_SIZE);
        assert_eq!(compute_morsel_size(PressureLevel::High), HIGH_PRESSURE_MORSEL_SIZE);
        assert_eq!(compute_morsel_size(PressureLevel::Critical), CRITICAL_PRESSURE_MORSEL_SIZE);
    }

    #[test]
    fn test_compute_morsel_size_with_base() {
        let base = 10000;

        assert_eq!(compute_morsel_size_with_base(base, PressureLevel::Normal), 10000);
        assert_eq!(compute_morsel_size_with_base(base, PressureLevel::Moderate), 5000);
        assert_eq!(compute_morsel_size_with_base(base, PressureLevel::High), 2500);
        assert_eq!(
            compute_morsel_size_with_base(base, PressureLevel::Critical),
            MIN_MORSEL_SIZE
        );
    }

    #[test]
    fn test_generate_morsels() {
        let morsels = generate_morsels(1000, 300, 0);

        assert_eq!(morsels.len(), 4);
        assert_eq!(morsels[0].start_row, 0);
        assert_eq!(morsels[0].end_row, 300);
        assert_eq!(morsels[1].start_row, 300);
        assert_eq!(morsels[1].end_row, 600);
        assert_eq!(morsels[2].start_row, 600);
        assert_eq!(morsels[2].end_row, 900);
        assert_eq!(morsels[3].start_row, 900);
        assert_eq!(morsels[3].end_row, 1000);
    }

    #[test]
    fn test_generate_morsels_empty() {
        assert!(generate_morsels(0, 100, 0).is_empty());
        assert!(generate_morsels(100, 0, 0).is_empty());
    }

    #[test]
    fn test_generate_morsels_exact_fit() {
        let morsels = generate_morsels(1000, 250, 0);

        assert_eq!(morsels.len(), 4);
        for (i, morsel) in morsels.iter().enumerate() {
            assert_eq!(morsel.row_count(), 250);
            assert_eq!(morsel.id, i);
        }
    }

    #[test]
    fn test_generate_adaptive_morsels() {
        let total = 100000;

        let normal_morsels = generate_adaptive_morsels(total, PressureLevel::Normal, 0);
        let high_morsels = generate_adaptive_morsels(total, PressureLevel::High, 0);

        // More morsels under pressure (smaller size)
        assert!(high_morsels.len() > normal_morsels.len());
    }
}
