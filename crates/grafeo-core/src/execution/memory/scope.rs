//! The memory of the statement this thread plans and converts: the scope its
//! operators capture their memory from at construction.

use std::cell::RefCell;

use super::StatementMemory;

thread_local! {
    /// The memory of the statement this thread plans and converts, if any
    /// (see [`enter`]).
    static STATEMENT_MEMORY: RefCell<Option<StatementMemory>> = const { RefCell::new(None) };
}

/// Sets the memory of the statement this thread plans and converts until the
/// returned scope drops, which restores the one before (also on a panic).
/// Operators built while the scope lives capture it at construction (see
/// [`capture`](super::capture), and `current` in builds with spilling), so
/// they count wherever they run.
pub fn enter(memory: StatementMemory) -> MemoryScope {
    let previous = STATEMENT_MEMORY.with(|cell| cell.replace(Some(memory)));
    MemoryScope {
        previous,
        _not_send: std::marker::PhantomData,
    }
}

/// Whether this thread plans a statement now, budgeted or not (see
/// [`enter`]): a session's planner checks it in debug builds, so a plan path
/// that sets no scope fails its tests instead of counting nothing.
#[must_use]
pub fn in_scope() -> bool {
    STATEMENT_MEMORY.with(|cell| cell.borrow().is_some())
}

/// The memory of the statement planned on this thread, to enter again where
/// part of it is planned later (a stored procedure's body, planned when it
/// runs).
#[must_use]
pub fn statement() -> Option<StatementMemory> {
    STATEMENT_MEMORY.with(|cell| cell.borrow().clone())
}

/// The statement memory [`enter`] set, until it drops.
#[must_use = "the statement's memory holds only while the scope lives"]
pub struct MemoryScope {
    /// The memory before [`enter`], restored on drop.
    previous: Option<StatementMemory>,
    /// Thread-local state restores on drop, so the scope must not move across
    /// threads.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for MemoryScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        STATEMENT_MEMORY.with(|cell| cell.replace(previous));
    }
}
