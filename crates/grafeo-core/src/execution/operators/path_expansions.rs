//! The path expansions a statement's path searches may make.

use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::utils::error::{Error, LimitBound, QueryError, QueryErrorKind};

use super::OperatorError;

/// The path expansions one statement may make by default: 100 million.
pub const DEFAULT_PATH_EXPANSION_LIMIT: u64 = 100_000_000;

/// The path expansions one statement may still make (the path expansion limit): shared by
/// every variable-length and shortest-path search of the statement, its subplans and the
/// bodies of the procedures it calls included, as the [`ReadCounter`](super::ReadCounter) is.
///
/// An expansion is one edge a search reads to extend a path, and one node or edge id it
/// copies out of a path into a row (a path variable, an edge list). A reachability search,
/// which reaches each node once, and an RDF property path are not charged.
///
/// ```
/// use grafeo_core::execution::operators::PathExpansions;
///
/// let expansions = PathExpansions::new(1_000);
/// assert_eq!(expansions.limit(), 1_000);
/// assert_eq!(expansions.used(), 0);
/// ```
#[derive(Debug)]
pub struct PathExpansions {
    /// The expansions the statement may make.
    limit: u64,
    /// The expansions its searches made so far.
    used: AtomicU64,
}

impl PathExpansions {
    /// A statement's count, with `limit` expansions to make.
    #[must_use]
    pub const fn new(limit: u64) -> Self {
        Self {
            limit,
            used: AtomicU64::new(0),
        }
    }

    /// The expansions the statement may make.
    #[must_use]
    pub const fn limit(&self) -> u64 {
        self.limit
    }

    /// The expansions its searches made so far.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// The expansions left now.
    fn left(&self) -> u64 {
        self.limit.saturating_sub(self.used())
    }
}

impl Default for PathExpansions {
    fn default() -> Self {
        Self::new(DEFAULT_PATH_EXPANSION_LIMIT)
    }
}

/// A search's view of the statement's expansions between two reads of the shared count: a
/// take is a subtraction and a compare, and the search publishes what it took once, when it
/// returns.
#[derive(Debug, Clone, Copy)]
pub(super) struct Allowance {
    /// The expansions the statement had left when the search read its count, less what the
    /// search took since.
    left: u64,
    /// The expansions the search took since.
    taken: u64,
}

impl Allowance {
    /// What the statement has left now; no limit for a search outside a statement (`None`).
    pub(super) fn of(expansions: Option<&PathExpansions>) -> Self {
        Self {
            left: expansions.map_or(u64::MAX, PathExpansions::left),
            taken: 0,
        }
    }

    /// Takes `n` expansions; `false` when the statement has fewer left, and then none are left,
    /// so a search that goes on before it fails takes nothing more.
    #[inline]
    #[must_use]
    pub(super) fn take(&mut self, n: usize) -> bool {
        let n = u64::try_from(n).unwrap_or(u64::MAX);
        if n > self.left {
            self.left = 0;
            return false;
        }
        self.left -= n;
        self.taken += n;
        true
    }

    /// Adds what the search took to the statement's count, and reads what the statement has
    /// left now: after the operator below a search ran, whose own searches took from the
    /// count.
    pub(super) fn refresh(&mut self, expansions: Option<&PathExpansions>) {
        self.settle(expansions);
        *self = Self::of(expansions);
    }

    /// Adds what the search took to the statement's count.
    pub(super) fn settle(self, expansions: Option<&PathExpansions>) {
        if let Some(expansions) = expansions
            && self.taken > 0
        {
            expansions.used.fetch_add(self.taken, Ordering::Relaxed);
        }
    }
}

/// The error of `what`, a path search of a statement that would pass its path expansion
/// limit of `expansions`: `advice` says how to need fewer expansions.
#[cold]
pub(super) fn path_expansion_error(
    what: &str,
    expansions: Option<&PathExpansions>,
    advice: &str,
) -> OperatorError {
    let bound = LimitBound::PathExpansionLimit;
    let limit = expansions.map_or(u64::MAX, PathExpansions::limit);
    OperatorError::Wrapped(Box::new(Error::from(QueryError::new(
        QueryErrorKind::Limit,
        format!(
            "{what} made more than {limit} path expansions (edges followed and path elements \
             returned), the {} of one statement: {advice}; or raise it with {}",
            bound.name(),
            bound.setting()
        ),
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_allowance_takes_up_to_what_is_left_and_publishes_once() {
        let expansions = PathExpansions::new(10);
        let expansions = Some(&expansions);
        let mut first = Allowance::of(expansions);
        assert!(first.take(4));
        assert!(first.take(6), "exactly the limit");
        assert!(!first.take(1), "one more");
        assert_eq!(
            expansions.map(PathExpansions::used),
            Some(0),
            "nothing published yet"
        );
        first.settle(expansions);
        assert_eq!(expansions.map(PathExpansions::used), Some(10));
        let mut second = Allowance::of(expansions);
        assert!(second.take(0));
        assert!(!second.take(1), "the statement's count is shared");
    }
}
