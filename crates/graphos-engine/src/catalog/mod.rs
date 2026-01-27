//! Database catalog.
//!
//! Manages schema definitions and index metadata.

/// Placeholder for catalog implementation.
pub struct Catalog {
    // FIXME(catalog): Implement schema definitions and index metadata
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
