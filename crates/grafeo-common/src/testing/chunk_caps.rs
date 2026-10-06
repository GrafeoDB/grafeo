//! Smaller chunk caps for tests, so the paths that cut sections into many
//! chunks run on small data.
//!
//! Sections read [`ChunkCaps::current`] when they are built: the caps
//! [`with_chunk_caps`] set on this thread, or [`ChunkCaps::DEFAULT`]. The
//! override is thread-local, so concurrent tests never see each other's caps.
//! Always compiled: outside tests the override is never set, and reading it
//! costs one thread-local access per section built.

use std::cell::Cell;

use crate::storage::chunk::ChunkCaps;

thread_local! {
    /// The caps set by [`with_chunk_caps`] on this thread, if any.
    static CAPS: Cell<Option<ChunkCaps>> = const { Cell::new(None) };
}

/// The caps [`with_chunk_caps`] set on this thread, or `None` outside it.
pub(crate) fn overridden() -> Option<ChunkCaps> {
    CAPS.with(Cell::get)
}

/// Puts the caps it holds back on this thread when dropped, also during a
/// panic.
struct Restore(Option<ChunkCaps>);

impl Drop for Restore {
    fn drop(&mut self) {
        CAPS.with(|caps| caps.set(self.0));
    }
}

/// Runs `f` with `caps` as this thread's chunk caps, restoring the previous
/// caps afterwards, also when `f` panics.
///
/// The caps are not validated here: writers refuse caps of zero (see
/// [`ChunkCaps::validate`]).
pub fn with_chunk_caps<T>(caps: ChunkCaps, f: impl FnOnce() -> T) -> T {
    let _restore = Restore(CAPS.with(|current| current.replace(Some(caps))));
    f()
}
