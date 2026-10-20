//! The rows a statement reads from storage.

use std::sync::atomic::{AtomicU64, Ordering};

/// Counts the rows a statement's access operators read from storage, before
/// any filter: the nodes a scan, a label scan, an index seek or a range scan
/// reads, the hits of a text or vector search, and the edges an expand
/// reads. The planner gives every access operator of a statement the same
/// counter; the result's `rows_scanned` is its count.
///
/// ```
/// use grafeo_core::execution::operators::ReadCounter;
///
/// let reads = ReadCounter::default();
/// reads.add(2048);
/// reads.add(3);
/// assert_eq!(reads.get(), 2051);
/// ```
#[derive(Debug, Default)]
pub struct ReadCounter(AtomicU64);

impl ReadCounter {
    /// Counts `rows` more rows read.
    #[inline]
    pub fn add(&self, rows: usize) {
        if rows > 0 {
            self.0.fetch_add(rows as u64, Ordering::Relaxed);
        }
    }

    /// The rows read so far.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Counts `rows` in `counter`, when an operator has one.
#[inline]
pub(super) fn count(counter: Option<&ReadCounter>, rows: usize) {
    if let Some(counter) = counter {
        counter.add(rows);
    }
}
