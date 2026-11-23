//! The memory a path search holds, and its deadline: one counting rule for
//! every path search (the RDF property paths, the LPG variable-length and
//! shortest-path searches). Each search has a budget of its own
//! (`Config::path_search_budget`), and what it holds is a grant of the
//! statement's memory too, under the query memory limit and the memory limit.

use grafeo_common::clock::Instant;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::hash::Hash;
use std::mem::{align_of, size_of};

use crate::execution::deadline::Ticker;
use crate::execution::memory::OperatorMemory;
use crate::execution::operators::OperatorError;
use crate::execution::path_queue::SMALL_BUDGET_BYTES;

/// The entries a vector or queue's reservation counts beyond its doubled
/// capacity: the allocator may round a buffer up by that much. Within a
/// path search budget under 16 KiB it shrinks with the budget, as the
/// queue's blocks do.
pub const RESERVE_SLACK_ENTRIES: usize = 16;

/// The entries a collection grows to at least.
const MIN_GROWTH_ENTRIES: usize = 4;

/// The buckets a table's reservation counts at least.
const MIN_TABLE_BUCKETS: usize = 16;

/// `full`, a size every collection starts from, within a budget of `limit`
/// bytes: as it is from [`SMALL_BUDGET_BYTES`], and below it the same share
/// of the budget as a block of the queue, so a small budget is not taken by
/// sizes no path asked for yet.
fn scaled_within(limit: usize, full: usize) -> usize {
    if limit >= SMALL_BUDGET_BYTES {
        full
    } else {
        full * limit / SMALL_BUDGET_BYTES
    }
}

/// The entries a collection grows to at least within a budget of `limit`.
fn least_growth(limit: usize) -> usize {
    scaled_within(limit, MIN_GROWTH_ENTRIES).max(1)
}

/// The capacity a collection grown one entry at a time within a budget of
/// `limit` bytes has once it holds `len` entries, by the rule of
/// [`PathBudget::reserve_vec`] and [`PathBudget::reserve_queue`]: each growth
/// doubles it, to at least the budget's least growth.
pub(crate) fn capacity_within(limit: usize, len: usize) -> usize {
    let mut capacity = 0_usize;
    for held in 0..len {
        if held == capacity {
            capacity = (held + 1)
                .max(capacity.saturating_mul(2))
                .max(least_growth(limit));
        }
    }
    capacity
}

/// What a refusal of a path search's budget says: the search, and how to need
/// less.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PathRefusal {
    /// The search, as the message's subject (`An RDF property path`).
    pub(crate) what: &'static str,
    /// What to do instead.
    pub(crate) advice: &'static str,
}

/// An RDF property path's refusal.
pub(crate) const RDF_PATH_REFUSAL: PathRefusal = PathRefusal {
    what: "An RDF property path",
    advice: "bound the path's repetition, or start it from fewer terms",
};

/// What a path search holds: its budget, and a grant of the statement's memory
/// as large as what it charges (see [`with_memory`](Self::with_memory)). A
/// charge checks the budget first and the grant second, and is refused by the
/// first that has no room; nothing is allocated before it is charged. The
/// grant goes back when the budget drops (the search ended, failed or timed
/// out), with no release call on that path.
///
/// A deadline is read two ways, through a [`Ticker`]. A wait or a retry (a lock wait, an
/// overlay's poll) reads the clock every time ([`check`](Self::check)). Small work (a visited
/// state, a scanned candidate, an origin, a row, a charge) counts a unit and reads it on the first
/// and every [`CLOCK_EVERY`](crate::execution::deadline::CLOCK_EVERY)th ([`tick`](Self::tick)),
/// so a search that emits nothing still meets its deadline.
pub(crate) struct PathBudget {
    limit: usize,
    used: usize,
    /// The most it held since the last [`clear`](Self::clear).
    #[cfg(test)]
    peak: usize,
    /// The search's deadline, which the RDF searches' waits read too.
    pub(crate) ticker: Ticker,
    /// The statement's grant of what the search holds (unbudgeted outside a
    /// budgeted statement).
    memory: OperatorMemory,
    /// What a refusal of the budget says.
    refusal: PathRefusal,
}

impl PathBudget {
    /// A budget of `limit` bytes with `deadline`, refused as an RDF property
    /// path, and no grant (see [`with_memory`](Self::with_memory)).
    pub(crate) fn new(limit: usize, deadline: Option<Instant>) -> Self {
        Self {
            limit,
            used: 0,
            #[cfg(test)]
            peak: 0,
            ticker: Ticker::with_deadline(deadline),
            memory: OperatorMemory::unbudgeted(),
            refusal: RDF_PATH_REFUSAL,
        }
    }

    /// The budget whose charges are a grant of `memory` too: its statement's.
    #[must_use]
    pub(crate) fn with_memory(mut self, memory: OperatorMemory) -> Self {
        self.memory = memory;
        self
    }

    /// The budget refused as `refusal` says.
    #[must_use]
    pub(crate) fn with_refusal(mut self, refusal: PathRefusal) -> Self {
        self.refusal = refusal;
        self
    }

    /// Lets go of everything charged, for the next search: the grant is kept,
    /// so the next search grows within it.
    pub(crate) fn clear(&mut self) {
        self.used = 0;
        #[cfg(test)]
        {
            self.peak = 0;
        }
        self.memory.clear();
    }

    /// Counts a unit of small work and reads the clock on the first and every
    /// [`CLOCK_EVERY`](crate::execution::deadline::CLOCK_EVERY)th.
    pub(crate) fn tick(&mut self) -> Result<(), OperatorError> {
        self.ticker.tick()
    }

    /// The refusal of `requested` more bytes over what the search holds:
    /// `GRAFEO-S001` naming the budget and its setting (#492), worded as the
    /// search's [`PathRefusal`].
    pub(crate) fn exceeded(&self, requested: usize) -> OperatorError {
        crate::execution::operators::path_search_refused(
            format!(
                "{} would hold more than the {} of memory a path search may use: {}",
                self.refusal.what,
                crate::execution::operators::budget_text(self.limit),
                self.refusal.advice
            ),
            requested,
            self.used,
            self.limit,
        )
    }

    /// Charges `bytes` more: refused past the budget (its own refusal), then
    /// past the statement's grant (the query memory limit's or the memory
    /// limit's); either way nothing is counted.
    pub(crate) fn charge(&mut self, bytes: usize) -> Result<(), OperatorError> {
        // A bounded walk charges nothing per path: skip the grant's bookkeeping.
        if bytes == 0 {
            return Ok(());
        }
        self.tick()?;
        let used = self
            .used
            .checked_add(bytes)
            .ok_or_else(|| self.exceeded(bytes))?;
        if used > self.limit {
            return Err(self.exceeded(bytes));
        }
        self.memory.set(used)?;
        self.used = used;
        #[cfg(test)]
        {
            self.peak = self.peak.max(used);
        }
        Ok(())
    }

    /// Gives back `bytes` charged before, to the budget and to the count of the
    /// grant (the grant itself is kept until the budget drops).
    pub(crate) fn release(&mut self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        debug_assert!(bytes <= self.used, "path search charge released twice");
        self.used = self.used.saturating_sub(bytes);
        self.memory.lower(self.used);
    }

    /// The bytes it holds.
    pub(crate) fn used(&self) -> usize {
        self.used
    }

    /// The most it held since the last [`clear`](Self::clear).
    #[cfg(test)]
    pub(crate) fn peak(&self) -> usize {
        self.peak
    }

    /// The bytes the search may hold.
    pub(crate) fn limit(&self) -> usize {
        self.limit
    }

    fn growth(
        &self,
        len: usize,
        capacity: usize,
        additional: usize,
    ) -> Result<usize, OperatorError> {
        let needed = len
            .checked_add(additional)
            .ok_or_else(|| self.exceeded(usize::MAX))?;
        let doubled = capacity
            .checked_mul(2)
            .ok_or_else(|| self.exceeded(usize::MAX))?;
        Ok(needed.max(doubled).max(least_growth(self.limit)))
    }

    /// `full` within this budget (see [`scaled_within`]).
    fn scaled(&self, full: usize) -> usize {
        scaled_within(self.limit, full)
    }

    // Hashbrown 0.17 keeps at least one/eighth of the buckets empty and
    // pads its control array. Reserve a conservative layout before it can
    // allocate, including the old table during a reallocation; afterwards
    // keep only the actual allocation_size charge.
    //
    // A table of fewer than 8 entries of 4 bytes or more has 4 or 8 buckets;
    // below the small budget that, not the floor, is what it reserves.
    fn table_reservation<T>(&self, target: usize) -> Result<usize, OperatorError> {
        let buckets = match target {
            0..4 if size_of::<T>() >= 4 => 4,
            4..8 if size_of::<T>() >= 4 => 8,
            _ => (target
                .checked_mul(8)
                .ok_or_else(|| self.exceeded(usize::MAX))?
                / 7)
            .checked_next_power_of_two()
            .ok_or_else(|| self.exceeded(usize::MAX))?,
        }
        .max(self.scaled(MIN_TABLE_BUCKETS));
        buckets
            .checked_mul(size_of::<T>())
            .and_then(|data| data.checked_add(buckets))
            .and_then(|data| data.checked_add(align_of::<T>().max(16) + 16))
            .ok_or_else(|| self.exceeded(usize::MAX))
    }

    fn reconcile(
        &mut self,
        reserved: usize,
        old: usize,
        actual: usize,
    ) -> Result<(), OperatorError> {
        debug_assert!(
            actual <= reserved,
            "an allocation of {actual} bytes passed its reservation of {reserved}"
        );
        if actual > reserved {
            return Err(self.exceeded(actual - reserved));
        }
        self.release(reserved - actual + old);
        Ok(())
    }

    /// Allocates the empty `block` to hold `items` items of `T`, charged before
    /// it allocates: a fixed-size block of a queue that grows by blocks (see
    /// [`PathQueue`](crate::execution::path_queue::PathQueue)), which never
    /// holds an old and a new buffer at once. A refusal leaves it unallocated.
    pub(crate) fn reserve_block<T>(
        &mut self,
        block: &mut Vec<T>,
        items: usize,
    ) -> Result<(), OperatorError> {
        debug_assert_eq!(block.capacity(), 0, "a block is allocated once");
        let reservation = items
            .checked_mul(size_of::<T>())
            .ok_or_else(|| self.exceeded(usize::MAX))?;
        self.charge(reservation)?;
        if block.try_reserve_exact(items).is_err() {
            self.release(reservation);
            return Err(self.exceeded(reservation));
        }
        self.reconcile(
            reservation,
            0,
            block.capacity().saturating_mul(size_of::<T>()),
        )
    }

    /// Makes room for `additional` more in `values`, charged before it
    /// grows (see [`grow_map`](Self::grow_map)). Inline: most calls find
    /// room and return at once.
    #[inline]
    pub(crate) fn reserve_map<K: Eq + Hash, V>(
        &mut self,
        values: &mut FxHashMap<K, V>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        if additional <= values.capacity().saturating_sub(values.len()) {
            return Ok(());
        }
        self.grow_map(values, additional)
    }

    /// Grows `values` for `additional` more, to the doubled capacity, charged
    /// first: a refusal leaves it as it was, and a failed allocation gives the
    /// charge back. Each growth counts as a unit of the search's work.
    #[inline(never)]
    fn grow_map<K: Eq + Hash, V>(
        &mut self,
        values: &mut FxHashMap<K, V>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        self.tick()?;
        let target = self.growth(values.len(), values.capacity(), additional)?;
        let reservation = self.table_reservation::<(K, V)>(target)?;
        let old = values.allocation_size();
        self.charge(reservation)?;
        if values.try_reserve(target - values.len()).is_err() {
            self.release(reservation);
            return Err(self.exceeded(reservation));
        }
        self.reconcile(reservation, old, values.allocation_size())
    }

    /// Makes room for `additional` more in `values`, charged before it
    /// grows (see [`grow_set`](Self::grow_set)). Inline: most calls find
    /// room and return at once.
    #[inline]
    pub(crate) fn reserve_set<T: Eq + Hash>(
        &mut self,
        values: &mut FxHashSet<T>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        if additional <= values.capacity().saturating_sub(values.len()) {
            return Ok(());
        }
        self.grow_set(values, additional)
    }

    /// Grows `values` for `additional` more, to the doubled capacity, charged
    /// first: a refusal leaves it as it was, and a failed allocation gives the
    /// charge back. Each growth counts as a unit of the search's work.
    #[inline(never)]
    fn grow_set<T: Eq + Hash>(
        &mut self,
        values: &mut FxHashSet<T>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        self.tick()?;
        let target = self.growth(values.len(), values.capacity(), additional)?;
        let reservation = self.table_reservation::<T>(target)?;
        let old = values.allocation_size();
        self.charge(reservation)?;
        if values.try_reserve(target - values.len()).is_err() {
            self.release(reservation);
            return Err(self.exceeded(reservation));
        }
        self.reconcile(reservation, old, values.allocation_size())
    }

    /// Makes room for `additional` more in `values`, charged before it
    /// grows (see [`grow_vec`](Self::grow_vec)). Inline: most calls find
    /// room and return at once.
    #[inline]
    pub(crate) fn reserve_vec<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        if additional <= values.capacity().saturating_sub(values.len()) {
            return Ok(());
        }
        self.grow_vec(values, additional)
    }

    /// Grows `values` for `additional` more, to the doubled capacity, charged
    /// first: a refusal leaves it as it was, and a failed allocation gives the
    /// charge back. Each growth counts as a unit of the search's work.
    #[inline(never)]
    fn grow_vec<T>(&mut self, values: &mut Vec<T>, additional: usize) -> Result<(), OperatorError> {
        self.tick()?;
        let target = self.growth(values.len(), values.capacity(), additional)?;
        let reservation = target
            .checked_add(self.scaled(RESERVE_SLACK_ENTRIES))
            .and_then(|n| n.checked_mul(size_of::<T>()))
            .ok_or_else(|| self.exceeded(usize::MAX))?;
        let old = values.capacity().saturating_mul(size_of::<T>());
        self.charge(reservation)?;
        if values.try_reserve_exact(target - values.len()).is_err() {
            self.release(reservation);
            return Err(self.exceeded(reservation));
        }
        self.reconcile(
            reservation,
            old,
            values.capacity().saturating_mul(size_of::<T>()),
        )
    }

    /// Makes room for `additional` more in `values`, charged before it
    /// grows (see [`grow_queue`](Self::grow_queue)). Inline: most calls find
    /// room and return at once.
    #[inline]
    pub(crate) fn reserve_queue<T>(
        &mut self,
        values: &mut VecDeque<T>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        if additional <= values.capacity().saturating_sub(values.len()) {
            return Ok(());
        }
        self.grow_queue(values, additional)
    }

    /// Grows `values` for `additional` more, to the doubled capacity, charged
    /// first: a refusal leaves it as it was, and a failed allocation gives the
    /// charge back. Each growth counts as a unit of the search's work.
    #[inline(never)]
    fn grow_queue<T>(
        &mut self,
        values: &mut VecDeque<T>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        self.tick()?;
        let target = self.growth(values.len(), values.capacity(), additional)?;
        let reservation = target
            .checked_add(self.scaled(RESERVE_SLACK_ENTRIES))
            .and_then(|n| n.checked_mul(size_of::<T>()))
            .ok_or_else(|| self.exceeded(usize::MAX))?;
        let old = values.capacity().saturating_mul(size_of::<T>());
        self.charge(reservation)?;
        if values.try_reserve_exact(target - values.len()).is_err() {
            self.release(reservation);
            return Err(self.exceeded(reservation));
        }
        self.reconcile(
            reservation,
            old,
            values.capacity().saturating_mul(size_of::<T>()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::memory::{self, OperatorMemoryContext, StatementMemory};
    use grafeo_common::memory::buffer::BufferManager;

    #[test]
    fn rejected_growth_does_not_allocate_or_charge_a_hash_table() {
        let mut budget = PathBudget::new(0, None);
        let mut values = grafeo_common::utils::hash::FxHashSet::<usize>::default();
        assert!(budget.reserve_set(&mut values, 1).is_err_and(|error| {
            crate::execution::operators::path_budget_message(&error).is_some()
        }));
        assert_eq!(values.allocation_size(), 0);
        assert_eq!(budget.used(), 0);
    }

    /// A charge is a grant of the statement's memory too: past the query memory
    /// limit it is refused naming that limit, with the budget's room left, and
    /// nothing stays counted; the grant goes back when the budget drops.
    #[test]
    fn a_charge_past_the_statement_cap_is_refused_and_given_back() {
        let ctx = OperatorMemoryContext::new(BufferManager::with_budget(256 << 20))
            .with_query_limit(64 << 10);
        let memory = {
            let _scope = memory::enter(StatementMemory::Budgeted(ctx.clone()));
            memory::capture()
        };
        let mut budget = PathBudget::new(1 << 30, None).with_memory(memory);
        budget.charge(32 << 10).unwrap();
        assert!(ctx.query().used() >= 32 << 10, "counted in the statement");
        let refused = budget.charge(64 << 10).unwrap_err();
        assert!(
            crate::execution::operators::path_budget_message(&refused).is_none(),
            "the statement's cap refused, not the budget: {refused}"
        );
        assert!(
            refused.to_string().contains("query memory limit"),
            "{refused}"
        );
        assert_eq!(budget.used(), 32 << 10, "the refused charge counts nothing");
        budget.release(32 << 10);
        budget.clear();
        drop(budget);
        assert_eq!(ctx.query().used(), 0, "the grant went back with the budget");
    }
}
