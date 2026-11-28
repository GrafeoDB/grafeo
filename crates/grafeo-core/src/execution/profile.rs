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
    /// The work its own code counted, without its children's (#794); empty in a build without
    /// `--cfg grafeo_work_counters`.
    pub work: grafeo_common::work::WorkCounts,
    /// The operator's note on a choice it made while it ran (see
    /// [`Operator::profile_note`]), as of its last call.
    pub note: Option<String>,
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

/// The outer operator's tally, held while a profiled operator runs. Dropping it, also when the
/// operator unwinds, puts that tally back and adds what the operator counted to its stats, so a
/// panic cannot leave the outer operator's work swapped out.
struct OwnWork<'a> {
    outer: grafeo_common::work::WorkCounts,
    stats: &'a SharedProfileStats,
}

impl Drop for OwnWork<'_> {
    fn drop(&mut self) {
        let own = grafeo_common::work::swap(self.outer);
        self.stats.lock().work.add(&own);
    }
}

impl Operator for ProfiledOperator {
    fn next(&mut self) -> OperatorResult {
        {
            let mut s = self.stats.lock();
            s.calls += 1;
        }

        // The work counted while the operator runs is its own: a child's wrapper swaps in a
        // tally of its own around the child (see `grafeo_common::work`).
        let own_work = OwnWork {
            outer: grafeo_common::work::take(),
            stats: &self.stats,
        };
        let start = grafeo_common::clock::Instant::now();

        let result = self.inner.next();

        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        drop(own_work);
        let mut stats = self.stats.lock();
        stats.time_ns += elapsed;
        if let Ok(Some(ref chunk)) = result {
            stats.rows_out += chunk.row_count() as u64;
        }
        stats.note = self.inner.profile_note();
        drop(stats);

        result
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn profile_note(&self) -> Option<String> {
        self.inner.profile_note()
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

    #[test]
    fn profiled_operator_measures_time() {
        let mock = MockOperator::new(1, 5);
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut profiled = ProfiledOperator::new(Box::new(mock), Arc::clone(&stats));

        let _ = profiled.next();
        assert!(stats.lock().time_ns > 0);
    }

    /// An operator that counts `own` rows built per call, on top of its child's.
    #[cfg(grafeo_work_counters)]
    struct Counting {
        child: Option<Box<dyn Operator>>,
        own: u64,
    }

    #[cfg(grafeo_work_counters)]
    impl Operator for Counting {
        fn next(&mut self) -> OperatorResult {
            grafeo_common::work::count(grafeo_common::work::Work::RowBuilt, self.own);
            if let Some(child) = &mut self.child {
                let _ = child.next()?;
            }
            Ok(None)
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "Counting"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// Each profiled operator gets the work of its own code, not its child's, and the statement's
    /// total is their sum; without `PROFILE` the thread's tally holds the total.
    #[cfg(grafeo_work_counters)]
    #[test]
    fn each_operator_gets_its_own_work() {
        use grafeo_common::work::{Work, take};

        let _ = take();
        let child_stats = Arc::new(Mutex::new(ProfileStats::default()));
        let parent_stats = Arc::new(Mutex::new(ProfileStats::default()));
        let child = ProfiledOperator::new(
            Box::new(Counting {
                child: None,
                own: 3,
            }),
            Arc::clone(&child_stats),
        );
        let mut parent = ProfiledOperator::new(
            Box::new(Counting {
                child: Some(Box::new(child)),
                own: 2,
            }),
            Arc::clone(&parent_stats),
        );
        grafeo_common::work::count(Work::Parse, 1);
        let _ = parent.next().unwrap();
        assert_eq!(parent_stats.lock().work.get(Work::RowBuilt), 2);
        assert_eq!(child_stats.lock().work.get(Work::RowBuilt), 3);
        let left = take();
        assert_eq!(left.get(Work::RowBuilt), 0, "the wrappers took every row");
        assert_eq!(left.get(Work::Parse), 1, "the work outside operators stays");

        let mut plain = Counting {
            child: Some(Box::new(Counting {
                child: None,
                own: 3,
            })),
            own: 2,
        };
        let _ = plain.next().unwrap();
        assert_eq!(
            take().get(Work::RowBuilt),
            5,
            "unprofiled, the thread holds the total"
        );
    }

    /// An operator that counts `own` rows built, then panics.
    #[cfg(grafeo_work_counters)]
    struct Panicking {
        own: u64,
    }

    #[cfg(grafeo_work_counters)]
    impl Operator for Panicking {
        fn next(&mut self) -> OperatorResult {
            grafeo_common::work::count(grafeo_common::work::Work::RowBuilt, self.own);
            panic!("an operator panics");
        }
        fn reset(&mut self) {}
        fn name(&self) -> &'static str {
            "Panicking"
        }
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// A child that panics gives the outer tally back as it unwinds: what was counted before it
    /// ran stays where it was, and the child's own work goes to the child.
    #[cfg(grafeo_work_counters)]
    #[test]
    fn a_panicking_child_leaves_the_outer_tally_intact() {
        use grafeo_common::work::{Work, count, take};

        let _ = take();
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let mut child = ProfiledOperator::new(Box::new(Panicking { own: 4 }), Arc::clone(&stats));
        count(Work::RowBuilt, 2);
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| child.next()));
        assert!(unwound.is_err());
        assert_eq!(
            take().get(Work::RowBuilt),
            2,
            "the outer tally is back, without the child's work"
        );
        assert_eq!(stats.lock().work.get(Work::RowBuilt), 4);
    }

    #[test]
    fn profiled_operator_delegates_name() {
        let mock = MockOperator::new(0, 0);
        let stats = Arc::new(Mutex::new(ProfileStats::default()));
        let profiled = ProfiledOperator::new(Box::new(mock), Arc::clone(&stats));
        assert_eq!(profiled.name(), "MockOperator");
    }
}
