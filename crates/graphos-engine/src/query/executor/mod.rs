//! Query executor.
//!
//! Executes physical plans and produces results.

/// Placeholder for executor implementation.
pub struct Executor {
    // TODO: Add implementation
}

impl Executor {
    /// Creates a new executor.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}
