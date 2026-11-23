//! Work counters (#794): the engine's own count of the work a statement does, per cost class.
//!
//! Timings vary 5 to 15% between runs, so a 2% effect cannot be seen in them, and they cannot
//! say where inside an operator the time goes. These counts are exact and the same on every
//! machine: a performance change states its effect in them, and a bench gates CI on them.
//!
//! Only a build with `--cfg grafeo_work_counters` counts: a cfg, not a feature, so that no
//! `--all-features` build, the timed CodSpeed and bench builds included, ever counts. Without it,
//! [`WorkCounts`] is empty and [`count`] does nothing, so release builds and every published crate
//! compile the counting away: nothing is stored, and the calls inline to nothing. The cfg also
//! fixes the hash seed (see `utils::hash`), so the counts that follow a hash table's layout are the
//! same in every run.
//!
//! The counts of a thread accumulate in one tally. `PROFILE` takes it apart per operator by
//! swapping in a fresh tally around each operator's `next()` ([`swap`]), so each operator gets the
//! work its own code did, as its self-time leaves out its children; what is left on the thread is
//! the statement's work outside operators. Without `PROFILE`, [`take`] returns the whole
//! statement's work.

/// A class of work the engine counts (see the [module](self) documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Work {
    /// A row's version, or a word of 64 rows that read alike, tested against a view (MVCC).
    VisibilityTest,
    /// A vector read from a stamped (non-flat) store, which carries versions.
    StampedRead,
    /// A node's or an edge's row group looked up.
    RowGroupLookup,
    /// An adjacency list read.
    AdjacencyRead,
    /// A property value read from its column.
    ColumnRead,
    /// A chunk of rows built.
    ChunkBuilt,
    /// A row built into a chunk.
    RowBuilt,
    /// A column value copied from one row to another.
    ColumnCopied,
    /// A value hashed.
    ValueHash,
    /// Two values compared.
    ValueCompare,
    /// A string value cloned out of a column.
    StringClone,
    /// An allocation (counted by a counting allocator).
    Allocation,
    /// The bytes of the allocations.
    AllocatedBytes,
    /// A check against the memory budget.
    BudgetCheck,
    /// A memory grant grown.
    GrantGrowth,
    /// A deadline checked (a timeout or a cancellation).
    DeadlineCheck,
    /// A statement parsed.
    Parse,
    /// A plan found in the plan cache.
    PlanCacheHit,
    /// A plan not found in the plan cache.
    PlanCacheMiss,
    /// A physical plan built.
    PlanBuilt,
    /// A correlated subplan planned again or reset.
    SubplanReplanned,
}

impl Work {
    /// The number of classes.
    pub const COUNT: usize = 21;

    /// Every class, in the order of its discriminant.
    pub const ALL: [Self; Self::COUNT] = [
        Self::VisibilityTest,
        Self::StampedRead,
        Self::RowGroupLookup,
        Self::AdjacencyRead,
        Self::ColumnRead,
        Self::ChunkBuilt,
        Self::RowBuilt,
        Self::ColumnCopied,
        Self::ValueHash,
        Self::ValueCompare,
        Self::StringClone,
        Self::Allocation,
        Self::AllocatedBytes,
        Self::BudgetCheck,
        Self::GrantGrowth,
        Self::DeadlineCheck,
        Self::Parse,
        Self::PlanCacheHit,
        Self::PlanCacheMiss,
        Self::PlanBuilt,
        Self::SubplanReplanned,
    ];

    /// The class's name in `PROFILE` and in the bench's baseline.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::VisibilityTest => "visibility_tests",
            Self::StampedRead => "stamped_reads",
            Self::RowGroupLookup => "rowgroup_lookups",
            Self::AdjacencyRead => "adjacency_reads",
            Self::ColumnRead => "column_reads",
            Self::ChunkBuilt => "chunks_built",
            Self::RowBuilt => "rows_built",
            Self::ColumnCopied => "columns_copied",
            Self::ValueHash => "value_hashes",
            Self::ValueCompare => "value_compares",
            Self::StringClone => "string_clones",
            Self::Allocation => "allocations",
            Self::AllocatedBytes => "allocated_bytes",
            Self::BudgetCheck => "budget_checks",
            Self::GrantGrowth => "grant_growths",
            Self::DeadlineCheck => "deadline_checks",
            Self::Parse => "parses",
            Self::PlanCacheHit => "plan_cache_hits",
            Self::PlanCacheMiss => "plan_cache_misses",
            Self::PlanBuilt => "plans_built",
            Self::SubplanReplanned => "subplans_replanned",
        }
    }
}

/// Counts per class (see [`Work`]); empty in a build without `--cfg grafeo_work_counters`.
#[cfg(grafeo_work_counters)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkCounts([u64; Work::COUNT]);

/// Counts per class (see [`Work`]); empty in a build without `--cfg grafeo_work_counters`.
#[cfg(not(grafeo_work_counters))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkCounts;

// Without the cfg the tally holds nothing, so a stats struct that carries one grows by
// nothing and a swap of it moves nothing.
#[cfg(not(grafeo_work_counters))]
const _: () = assert!(std::mem::size_of::<WorkCounts>() == 0);

#[cfg(grafeo_work_counters)]
impl Default for WorkCounts {
    fn default() -> Self {
        Self::empty()
    }
}

impl WorkCounts {
    /// No work.
    #[must_use]
    pub const fn empty() -> Self {
        #[cfg(grafeo_work_counters)]
        {
            Self([0; Work::COUNT])
        }
        #[cfg(not(grafeo_work_counters))]
        {
            Self
        }
    }

    /// The count of `work`; always 0 without the cfg.
    #[must_use]
    pub fn get(&self, work: Work) -> u64 {
        #[cfg(grafeo_work_counters)]
        {
            self.0[work as usize]
        }
        #[cfg(not(grafeo_work_counters))]
        {
            let _ = work;
            0
        }
    }

    /// Adds `other`'s counts to these.
    pub fn add(&mut self, other: &Self) {
        #[cfg(grafeo_work_counters)]
        for (count, more) in self.0.iter_mut().zip(other.0) {
            *count = count.saturating_add(more);
        }
        #[cfg(not(grafeo_work_counters))]
        let _ = other;
    }

    /// The classes with a count above 0, with their counts, in [`Work::ALL`]'s order.
    pub fn nonzero(&self) -> impl Iterator<Item = (Work, u64)> + '_ {
        Work::ALL
            .into_iter()
            .map(|work| (work, self.get(work)))
            .filter(|&(_, count)| count > 0)
    }

    /// The counts as `PROFILE` prints them: `name=count` for each class above 0, space apart;
    /// empty when nothing was counted.
    #[must_use]
    pub fn describe(&self) -> String {
        self.nonzero()
            .map(|(work, count)| format!("{}={count}", work.name()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Whether this build counts (`--cfg grafeo_work_counters`).
pub const ENABLED: bool = cfg!(grafeo_work_counters);

#[cfg(grafeo_work_counters)]
std::thread_local! {
    // A const initializer allocates nothing, so a counting allocator can count into it.
    static TALLY: [std::cell::Cell<u64>; Work::COUNT] =
        const { [const { std::cell::Cell::new(0) }; Work::COUNT] };
}

/// Counts `n` of `work` on this thread's tally; nothing without the cfg. A counting allocator may
/// call it: it allocates nothing, and once the thread's locals are torn down it counts nothing.
#[inline]
pub fn count(work: Work, n: u64) {
    #[cfg(grafeo_work_counters)]
    let _ = TALLY.try_with(|tally| {
        let cell = &tally[work as usize];
        cell.set(cell.get().saturating_add(n));
    });
    #[cfg(not(grafeo_work_counters))]
    let _ = (work, n);
}

/// Replaces this thread's tally with `counts` and returns what it held.
#[inline]
#[must_use]
pub fn swap(counts: WorkCounts) -> WorkCounts {
    #[cfg(grafeo_work_counters)]
    {
        TALLY.with(|tally| {
            let mut held = [0; Work::COUNT];
            for ((cell, old), new) in tally.iter().zip(held.iter_mut()).zip(counts.0) {
                *old = cell.replace(new);
            }
            WorkCounts(held)
        })
    }
    #[cfg(not(grafeo_work_counters))]
    {
        counts
    }
}

/// This thread's tally, which starts again from 0.
#[inline]
#[must_use]
pub fn take() -> WorkCounts {
    swap(WorkCounts::empty())
}

#[cfg(all(test, grafeo_work_counters))]
mod tests {
    use super::*;

    /// Every variant's place in [`Work::ALL`], by an exhaustive match: a new class fails to
    /// compile here until it is given its place, so `ALL` and `COUNT` cannot miss it.
    fn place(work: Work) -> usize {
        match work {
            Work::VisibilityTest => 0,
            Work::StampedRead => 1,
            Work::RowGroupLookup => 2,
            Work::AdjacencyRead => 3,
            Work::ColumnRead => 4,
            Work::ChunkBuilt => 5,
            Work::RowBuilt => 6,
            Work::ColumnCopied => 7,
            Work::ValueHash => 8,
            Work::ValueCompare => 9,
            Work::StringClone => 10,
            Work::Allocation => 11,
            Work::AllocatedBytes => 12,
            Work::BudgetCheck => 13,
            Work::GrantGrowth => 14,
            Work::DeadlineCheck => 15,
            Work::Parse => 16,
            Work::PlanCacheHit => 17,
            Work::PlanCacheMiss => 18,
            Work::PlanBuilt => 19,
            Work::SubplanReplanned => 20,
        }
    }

    #[test]
    fn every_class_has_its_place_in_all() {
        for (index, work) in Work::ALL.into_iter().enumerate() {
            assert_eq!(place(work), index, "{work:?}");
        }
        assert_eq!(
            place(Work::SubplanReplanned) + 1,
            Work::COUNT,
            "COUNT is the last place plus one"
        );
    }

    #[test]
    fn every_class_is_listed_once_in_discriminant_order() {
        for (index, work) in Work::ALL.into_iter().enumerate() {
            assert_eq!(work as usize, index, "{work:?}");
        }
        let names: std::collections::HashSet<&str> =
            Work::ALL.into_iter().map(Work::name).collect();
        assert_eq!(names.len(), Work::COUNT, "every name is its own");
    }

    #[test]
    fn counts_add_up_and_take_starts_again() {
        let _ = take();
        count(Work::RowBuilt, 3);
        count(Work::RowBuilt, 2);
        count(Work::Parse, 1);
        let counts = take();
        assert_eq!(counts.get(Work::RowBuilt), 5);
        assert_eq!(counts.get(Work::Parse), 1);
        assert_eq!(counts.get(Work::ValueHash), 0);
        assert_eq!(counts.describe(), "rows_built=5 parses=1");
        assert_eq!(take(), WorkCounts::empty(), "take starts again from 0");
    }

    #[test]
    fn a_swap_keeps_the_outer_tally_apart() {
        let _ = take();
        count(Work::ValueHash, 7);
        let outer = take();
        count(Work::ValueHash, 2);
        let inner = swap(outer);
        assert_eq!(inner.get(Work::ValueHash), 2);
        assert_eq!(take().get(Work::ValueHash), 7, "the outer tally is back");
    }

    #[test]
    fn each_thread_counts_on_its_own() {
        let _ = take();
        count(Work::ColumnRead, 4);
        let other = std::thread::spawn(|| {
            count(Work::ColumnRead, 9);
            take().get(Work::ColumnRead)
        })
        .join()
        .unwrap();
        assert_eq!(other, 9);
        assert_eq!(take().get(Work::ColumnRead), 4);
    }
}

#[cfg(all(test, not(grafeo_work_counters)))]
mod tests_without_the_cfg {
    use super::*;

    #[test]
    fn nothing_is_counted() {
        count(Work::RowBuilt, 3);
        assert_eq!(take().get(Work::RowBuilt), 0);
        assert_eq!(WorkCounts::empty().describe(), "");
    }
}
