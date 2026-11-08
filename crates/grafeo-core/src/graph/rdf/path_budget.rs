//! Query-local accounting and deadline checks for native RDF paths.

use std::collections::VecDeque;
use std::hash::Hash;
use std::mem::{align_of, size_of};
use std::time::Duration;

use grafeo_common::clock::Instant;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::{RwLock, RwLockReadGuard};

use crate::execution::deadline::Ticker;
use crate::execution::operators::OperatorError;

/// Local only: charging never calls an allocator manager or evicts a store.
///
/// A deadline is read two ways, through a [`Ticker`]. A wait or a retry (a lock wait, an
/// overlay's poll) reads the clock every time ([`check`](Self::check)). Small work (a visited
/// state, a scanned candidate, an origin, a row, a charge) counts a unit and reads it on the first
/// and every [`CLOCK_EVERY`](crate::execution::deadline::CLOCK_EVERY)th ([`tick`](Self::tick)),
/// so a search that emits nothing still meets its deadline.
pub(crate) struct PathBudget {
    limit: usize,
    used: usize,
    ticker: Ticker,
}

impl PathBudget {
    pub(crate) fn new(limit: usize, deadline: Option<Instant>) -> Self {
        Self {
            limit,
            used: 0,
            ticker: Ticker::with_deadline(deadline),
        }
    }

    /// Reads the clock: for a wait or a retry, which may take any time.
    pub(crate) fn check(&mut self) -> Result<(), OperatorError> {
        self.ticker.check()
    }

    /// Counts a unit of small work and reads the clock on the first and every
    /// [`CLOCK_EVERY`](crate::execution::deadline::CLOCK_EVERY)th.
    pub(crate) fn tick(&mut self) -> Result<(), OperatorError> {
        self.ticker.tick()
    }

    fn exceeded(&self) -> OperatorError {
        OperatorError::LimitExceeded(format!(
            "RDF property path exceeds its {}-byte memory budget",
            self.limit
        ))
    }

    pub(crate) fn charge(&mut self, bytes: usize) -> Result<(), OperatorError> {
        self.tick()?;
        let used = self
            .used
            .checked_add(bytes)
            .ok_or_else(|| self.exceeded())?;
        if used > self.limit {
            return Err(self.exceeded());
        }
        self.used = used;
        Ok(())
    }

    pub(crate) fn release(&mut self, bytes: usize) {
        debug_assert!(bytes <= self.used, "RDF path charge released twice");
        self.used = self.used.saturating_sub(bytes);
    }

    /// Wait in bounded intervals, including when the lock is contended.
    pub(crate) fn lock_wait(&mut self) -> Result<Duration, OperatorError> {
        self.check()?;
        let interval = Duration::from_millis(10);
        Ok(self.ticker.deadline().map_or(interval, |deadline| {
            deadline
                .saturating_duration_since(Instant::now())
                .min(interval)
        }))
    }

    /// Wait in bounded intervals, including when the lock is contended.
    pub(crate) fn read<'a, T>(
        &mut self,
        lock: &'a RwLock<T>,
    ) -> Result<RwLockReadGuard<'a, T>, OperatorError> {
        loop {
            let wait = self.lock_wait()?;
            if let Some(guard) = lock.try_read_for(wait) {
                self.check()?;
                return Ok(guard);
            }
        }
    }

    fn growth(
        &self,
        len: usize,
        capacity: usize,
        additional: usize,
    ) -> Result<usize, OperatorError> {
        let needed = len.checked_add(additional).ok_or_else(|| self.exceeded())?;
        let doubled = capacity.checked_mul(2).ok_or_else(|| self.exceeded())?;
        Ok(needed.max(doubled).max(4))
    }

    // Hashbrown 0.17 keeps at least one/eighth of the buckets empty and
    // pads its control array. Reserve a conservative layout before it can
    // allocate, including the old table during a reallocation; afterwards
    // keep only the actual allocation_size charge.
    fn table_reservation<T>(&self, target: usize) -> Result<usize, OperatorError> {
        let buckets = (target.checked_mul(8).ok_or_else(|| self.exceeded())? / 7)
            .checked_next_power_of_two()
            .ok_or_else(|| self.exceeded())?
            .max(16);
        buckets
            .checked_mul(size_of::<T>())
            .and_then(|data| data.checked_add(buckets))
            .and_then(|data| data.checked_add(align_of::<T>().max(16) + 16))
            .ok_or_else(|| self.exceeded())
    }

    fn reconcile(
        &mut self,
        reserved: usize,
        old: usize,
        actual: usize,
    ) -> Result<(), OperatorError> {
        if actual > reserved {
            return Err(self.exceeded());
        }
        self.release(reserved - actual + old);
        Ok(())
    }

    pub(crate) fn reserve_map<K: Eq + Hash, V>(
        &mut self,
        values: &mut FxHashMap<K, V>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        self.tick()?;
        if additional <= values.capacity().saturating_sub(values.len()) {
            return Ok(());
        }
        let target = self.growth(values.len(), values.capacity(), additional)?;
        let reservation = self.table_reservation::<(K, V)>(target)?;
        let old = values.allocation_size();
        self.charge(reservation)?;
        if values.try_reserve(target - values.len()).is_err() {
            self.release(reservation);
            return Err(self.exceeded());
        }
        self.reconcile(reservation, old, values.allocation_size())
    }

    pub(crate) fn reserve_set<T: Eq + Hash>(
        &mut self,
        values: &mut FxHashSet<T>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        self.tick()?;
        if additional <= values.capacity().saturating_sub(values.len()) {
            return Ok(());
        }
        let target = self.growth(values.len(), values.capacity(), additional)?;
        let reservation = self.table_reservation::<T>(target)?;
        let old = values.allocation_size();
        self.charge(reservation)?;
        if values.try_reserve(target - values.len()).is_err() {
            self.release(reservation);
            return Err(self.exceeded());
        }
        self.reconcile(reservation, old, values.allocation_size())
    }

    pub(crate) fn reserve_vec<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        self.tick()?;
        if additional <= values.capacity().saturating_sub(values.len()) {
            return Ok(());
        }
        let target = self.growth(values.len(), values.capacity(), additional)?;
        let reservation = target
            .checked_add(16)
            .and_then(|n| n.checked_mul(size_of::<T>()))
            .ok_or_else(|| self.exceeded())?;
        let old = values.capacity().saturating_mul(size_of::<T>());
        self.charge(reservation)?;
        if values.try_reserve_exact(target - values.len()).is_err() {
            self.release(reservation);
            return Err(self.exceeded());
        }
        self.reconcile(
            reservation,
            old,
            values.capacity().saturating_mul(size_of::<T>()),
        )
    }

    pub(crate) fn reserve_queue<T>(
        &mut self,
        values: &mut VecDeque<T>,
        additional: usize,
    ) -> Result<(), OperatorError> {
        self.tick()?;
        if additional <= values.capacity().saturating_sub(values.len()) {
            return Ok(());
        }
        let target = self.growth(values.len(), values.capacity(), additional)?;
        let reservation = target
            .checked_add(16)
            .and_then(|n| n.checked_mul(size_of::<T>()))
            .ok_or_else(|| self.exceeded())?;
        let old = values.capacity().saturating_mul(size_of::<T>());
        self.charge(reservation)?;
        if values.try_reserve_exact(target - values.len()).is_err() {
            self.release(reservation);
            return Err(self.exceeded());
        }
        self.reconcile(
            reservation,
            old,
            values.capacity().saturating_mul(size_of::<T>()),
        )
    }

    /// Lets the deadline pass after `polls` reads of the clock (see
    /// [`Ticker::expire_after_polls`]).
    #[cfg(test)]
    pub(crate) fn expire_after_polls(&mut self, polls: u32) {
        self.ticker.expire_after_polls(polls);
    }

    #[cfg(test)]
    pub(crate) fn used(&self) -> usize {
        self.used
    }
}

/// Counts retained Arc payloads conservatively, including their headers.
///
/// # Errors
///
/// [`UnsupportedTermKind`](grafeo_common::types::rdf::UnsupportedTermKind)
/// for a term of a kind it does not know.
pub(crate) fn term_payload(
    term: &super::Term,
) -> Result<usize, grafeo_common::types::rdf::UnsupportedTermKind> {
    let arc = |len: usize| len.saturating_add(2 * size_of::<usize>());
    Ok(match term {
        super::Term::Iri(iri) => arc(iri.as_str().len()),
        super::Term::BlankNode(node) => arc(node.id().len()),
        super::Term::Literal(literal) => arc(literal.value().len())
            .saturating_add(arc(literal.datatype().len()))
            .saturating_add(literal.language().map_or(0, |lang| arc(lang.len()))),
        other => return Err(grafeo_common::types::rdf::UnsupportedTermKind::of(other)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_growth_does_not_allocate_or_charge_a_hash_table() {
        let mut budget = PathBudget::new(0, None);
        let mut values = grafeo_common::utils::hash::FxHashSet::<usize>::default();
        assert!(matches!(
            budget.reserve_set(&mut values, 1),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert_eq!(values.allocation_size(), 0);
        assert_eq!(budget.used(), 0);
    }
}
