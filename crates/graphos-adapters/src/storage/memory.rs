//! Pure in-memory storage backend.

use graphos_core::graph::lpg::LpgStore;
use std::sync::Arc;

/// In-memory storage backend.
///
/// This is the default storage backend that keeps all data in memory.
/// Data is lost when the process exits unless WAL is enabled.
pub struct MemoryBackend {
    /// The underlying LPG store.
    store: Arc<LpgStore>,
}

impl MemoryBackend {
    /// Creates a new in-memory backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: Arc::new(LpgStore::new()),
        }
    }

    /// Returns a reference to the underlying store.
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        &self.store
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_backend() {
        let backend = MemoryBackend::new();
        let store = backend.store();

        let id = store.create_node(&["Test"]);
        assert!(id.is_valid());
    }
}
