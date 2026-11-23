//! The retained bytes of RDF terms, which the path searches charge; the budget
//! itself is [`PathBudget`](crate::execution::path_budget::PathBudget), shared
//! by every path search.

use std::mem::size_of;
use std::time::Duration;

use grafeo_common::clock::Instant;
use parking_lot::{RecursiveRwLock, RecursiveRwLockReadGuard, RwLock, RwLockReadGuard};

use crate::execution::operators::OperatorError;

pub(crate) use crate::execution::path_budget::PathBudget;

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

/// The waits of the RDF path searches, which read under the dataset's locks.
impl PathBudget {
    /// Reads the clock: for a wait or a retry, which may take any time.
    pub(crate) fn check(&mut self) -> Result<(), OperatorError> {
        self.ticker.check()
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
    /// The shared side of a dataset's apply guard, waited for in bounded
    /// intervals as [`read`](Self::read) waits: a path search behind a long
    /// commit meets its deadline. A free guard, the common case (no commit
    /// applies), is taken at once, with no clock read: the search polls its
    /// deadline at every node anyway.
    pub(crate) fn read_guard<'a>(
        &mut self,
        guard: &'a RecursiveRwLock<()>,
    ) -> Result<RecursiveRwLockReadGuard<'a, ()>, OperatorError> {
        if let Some(held) = guard.try_read() {
            return Ok(held);
        }
        loop {
            let wait = self.lock_wait()?;
            if let Some(held) = guard.try_read_for(wait) {
                self.check()?;
                return Ok(held);
            }
        }
    }
    /// Lets the deadline pass after `polls` reads of the clock (see
    /// [`Ticker::expire_after_polls`]).
    #[cfg(test)]
    pub(crate) fn expire_after_polls(&mut self, polls: u32) {
        self.ticker.expire_after_polls(polls);
    }
}
