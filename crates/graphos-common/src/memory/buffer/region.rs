//! Memory region definitions for the unified buffer manager.

/// Memory region identifiers for budget partitioning.
///
/// The buffer manager divides its budget across these regions,
/// allowing for differentiated eviction policies and pressure tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRegion {
    /// Graph storage: nodes, edges, properties, adjacency lists.
    GraphStorage,
    /// Index structures: btree, hash, trie indexes.
    IndexBuffers,
    /// Query execution: DataChunks, hash tables, sort buffers.
    ExecutionBuffers,
    /// Spill staging area for operators under memory pressure.
    SpillStaging,
}

impl MemoryRegion {
    /// Returns the array index for this region (for per-region tracking).
    #[must_use]
    pub const fn index(&self) -> usize {
        match self {
            Self::GraphStorage => 0,
            Self::IndexBuffers => 1,
            Self::ExecutionBuffers => 2,
            Self::SpillStaging => 3,
        }
    }

    /// Returns a human-readable name for this region.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::GraphStorage => "Graph Storage",
            Self::IndexBuffers => "Index Buffers",
            Self::ExecutionBuffers => "Execution Buffers",
            Self::SpillStaging => "Spill Staging",
        }
    }

    /// Returns all memory regions.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [
            Self::GraphStorage,
            Self::IndexBuffers,
            Self::ExecutionBuffers,
            Self::SpillStaging,
        ]
    }
}

impl std::fmt::Display for MemoryRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_indices() {
        assert_eq!(MemoryRegion::GraphStorage.index(), 0);
        assert_eq!(MemoryRegion::IndexBuffers.index(), 1);
        assert_eq!(MemoryRegion::ExecutionBuffers.index(), 2);
        assert_eq!(MemoryRegion::SpillStaging.index(), 3);
    }

    #[test]
    fn test_region_names() {
        assert_eq!(MemoryRegion::GraphStorage.name(), "Graph Storage");
        assert_eq!(MemoryRegion::ExecutionBuffers.name(), "Execution Buffers");
    }

    #[test]
    fn test_region_all() {
        let all = MemoryRegion::all();
        assert_eq!(all.len(), 4);
    }
}
