//! Query profiling infrastructure.
//!
//! Provides [`ProfiledOperator`], a wrapper that collects runtime statistics
//! (row counts, timing, call counts) around any pull-based [`Operator`].
//! Used by the `PROFILE` statement to annotate each operator with actual
//! execution metrics.

use std::sync::Arc;

use parking_lot::Mutex;

use super::operators::{Operator, OperatorResult};

/// Runtime statistics for a single operator in a profiled query.
#[derive(Debug, Clone, Default)]
pub struct ProfileStats {
    /// Total rows produced as output.
    pub rows_out: u64,
    /// Wall-clock time spent in this operator (nanoseconds), including children.
    pub time_ns: u64,
    /// Number of times `next()` was called on this operator.
    pub calls: u64,
}

/// Shared handle to profile stats, written by `ProfiledOperator` during
/// execution and read afterwards for formatting.
pub type SharedProfileStats = Arc<Mutex<ProfileStats>>;

/// Wraps a pull-based [`Operator`] to collect runtime statistics.
///
/// Each call to [`next()`](Operator::next) is timed and the output rows
/// are counted. Statistics are written into a [`SharedProfileStats`] handle
/// so they can be collected after execution completes.
pub struct ProfiledOperator {
    inner: Box<dyn Operator>,
    stats: SharedProfileStats,
}

impl ProfiledOperator {
    /// Creates a new profiled wrapper around the given operator.
    pub fn new(inner: Box<dyn Operator>, stats: SharedProfileStats) -> Self {
        Self { inner, stats }
    }
}

impl Operator for ProfiledOperator {
    fn next(&mut self) -> OperatorResult {
        {
            let mut s = self.stats.lock();
            s.calls += 1;
        }

        #[cfg(not(target_arch = "wasm32"))]
        let start = std::time::Instant::now();

        let result = self.inner.next();

        #[cfg(not(target_arch = "wasm32"))]
        {
            // reason: per-call elapsed nanos fits u64 for any practical duration
            #[allow(clippy::cast_possible_truncation)]
            let elapsed = start.elapsed().as_nanos() as u64;
            self.stats.lock().time_ns += elapsed;
        }

        if let Ok(Some(ref chunk)) = result {
            self.stats.lock().rows_out += chunk.row_count() as u64;
        }

        result
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

// ProfiledOperator is Send + Sync because:
// - inner: Box<dyn Operator> is Send + Sync (trait bound)
// - stats: Arc<parking_lot::Mutex<ProfileStats>> is Send + Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    // Called at compile time to verify the bounds hold.
    #[allow(dead_code)]
    fn check() {
        assert_send_sync::<ProfiledOperator>();
    }
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunk;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::LogicalType;

    /// A mock operator that yields a fixed number of chunks, each with `rows_per_chunk` rows.
    struct MockOperator {
        chunks_remaining: usize,
        rows_per_chunk: usize,
    }

    impl MockOperator {
        fn new(chunks: usize, rows_per_chunk: usize) -> Self {
            Self {
                chunks_remaining: chunks,
                rows_per_chunk,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.chunks_remaining == 0 {
                return Ok(None);
            }
            self.chunks_remaining -= 1;
            let mut col = ValueVector::with_capacity(LogicalType::Int64, self.rows_per_chunk);
            // reason: test chunk rows are small, fit i64
            #[allow(clippy::cast_possible_wrap)]
            for i in 0..self.rows_per_chunk {
                col.push(grafeo_common::types::Value::Int64(i as i64));
            }
            let chunk = DataChunk::new(vec![col]);
            Ok(Some(chunk))
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "MockOperator"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn profile_stats_default_is_zero() {
        let stats = ProfileStats::default();
        assert_eq!(stats.rows_out, 0);
        assert_eq!(stats.time_ns, 0);
        assert_eq!(stats.calls, 0);
    }

    #[test]
    fn profiled_operator_counts_rows_and_calls() {
        let mock = MockOperator::new(3, 10);
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut profiled = ProfiledOperator::new(Box::new(mock), Arc::clone(&stats));

        // Drain operator (3 chunks + 1 None = 4 calls)
        while profiled.next().unwrap().is_some() {}

        let s = stats.lock();
        assert_eq!(s.rows_out, 30); // 3 chunks x 10 rows
        assert_eq!(s.calls, 4); // 3 data + 1 None
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn profiled_operator_measures_time() {
        let mock = MockOperator::new(1, 5);
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut profiled = ProfiledOperator::new(Box::new(mock), Arc::clone(&stats));

        let _ = profiled.next();
        assert!(stats.lock().time_ns > 0);
    }

    #[test]
    fn profiled_operator_delegates_name() {
        let mock = MockOperator::new(0, 0);
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let profiled = ProfiledOperator::new(Box::new(mock), Arc::clone(&stats));
        assert_eq!(profiled.name(), "MockOperator");
    }
}
