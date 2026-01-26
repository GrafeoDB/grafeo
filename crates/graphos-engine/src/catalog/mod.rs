//! Database catalog.
//!
//! Manages schema definitions and index metadata.

/// Placeholder for catalog implementation.
pub struct Catalog {
    // TODO: Add implementation
}

impl Catalog {
    /// Creates a new catalog.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}
