//! Pure in-memory storage backend.

use grafeo_common::memory::arena::AllocError;
use grafeo_core::graph::rowgroup::RowGroupStore;
use std::sync::Arc;

/// In-memory storage backend.
///
/// This is the default storage backend that keeps all data in memory.
/// Data is lost when the process exits unless WAL is enabled.
pub struct MemoryBackend {
    /// The underlying row-group store.
    store: Arc<RowGroupStore>,
}

impl MemoryBackend {
    /// Creates a new in-memory backend.
    ///
    /// # Errors
    ///
    /// None since 0.6.0: the row-group store allocates as it grows. The
    /// [`AllocError`] stays in the signature, which the arena-based store
    /// of earlier releases needed.
    pub fn new() -> Result<Self, AllocError> {
        Ok(Self {
            store: Arc::new(RowGroupStore::new()),
        })
    }

    /// Returns a reference to the underlying store.
    #[must_use]
    pub fn store(&self) -> &Arc<RowGroupStore> {
        &self.store
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self {
            store: Arc::new(RowGroupStore::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use grafeo_core::graph::{GraphStore, GraphStoreMut};

    use super::*;

    #[test]
    fn test_memory_backend() {
        let backend = MemoryBackend::new().unwrap();
        let store = backend.store();

        let id = store.create_node(&["Test"]);
        assert!(id.is_valid());
        assert_eq!(store.node_count(), 1);
    }
}
