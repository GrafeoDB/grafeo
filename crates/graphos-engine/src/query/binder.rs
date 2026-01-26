//! Semantic binding and type checking.
//!
//! The binder resolves names, checks types, and produces a bound AST.

/// Placeholder for binder implementation.
pub struct Binder {
    // TODO: Add catalog reference, scope stack, etc.
}

impl Binder {
    /// Creates a new binder.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for Binder {
    fn default() -> Self {
        Self::new()
    }
}
