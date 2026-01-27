//! Memory consumer trait for subsystems that use managed memory.

use thiserror::Error;

use super::region::MemoryRegion;

/// Error type for spilling operations.
#[derive(Error, Debug, Clone)]
pub enum SpillError {
    /// Spilling is not supported by this consumer.
    #[error("spilling not supported")]
    NotSupported,
    /// I/O error during spill.
    #[error("I/O error during spill: {0}")]
    IoError(String),
    /// Spill directory not configured.
    #[error("spill directory not configured")]
    NoSpillDirectory,
    /// Insufficient disk space.
    #[error("insufficient disk space for spill")]
    InsufficientDiskSpace,
}

/// Trait for subsystems that consume managed memory.
///
/// Memory consumers register with the buffer manager and participate
/// in eviction when memory pressure is detected. Lower priority consumers
/// are evicted first.
pub trait MemoryConsumer: Send + Sync {
    /// Returns a unique name for this consumer (for debugging/logging).
    fn name(&self) -> &str;

    /// Returns current memory usage in bytes.
    fn memory_usage(&self) -> usize;

    /// Returns eviction priority (0 = lowest priority, evict first; 255 = highest, evict last).
    fn eviction_priority(&self) -> u8;

    /// Returns which memory region this consumer belongs to.
    fn region(&self) -> MemoryRegion;

    /// Attempts to evict/release memory to reach target usage.
    ///
    /// Returns the number of bytes actually freed.
    fn evict(&self, target_bytes: usize) -> usize;

    /// Returns whether this consumer supports spilling to disk.
    fn can_spill(&self) -> bool {
        false
    }

    /// Spills data to disk to free memory.
    ///
    /// Returns the number of bytes freed on success.
    ///
    /// # Errors
    ///
    /// Returns an error if spilling fails or is not supported.
    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        Err(SpillError::NotSupported)
    }

    /// Reloads spilled data from disk (called when memory becomes available).
    ///
    /// # Errors
    ///
    /// Returns an error if reloading fails.
    fn reload(&self) -> Result<(), SpillError> {
        Ok(())
    }
}

/// Standard priority levels for common consumer types.
///
/// Lower values = evict first, higher values = evict last.
pub mod priorities {
    /// Spill staging buffers - evict first (already written to disk).
    pub const SPILL_STAGING: u8 = 10;

    /// Cached query results - relatively cheap to recompute.
    pub const QUERY_CACHE: u8 = 30;

    /// Index buffers - can be rebuilt from primary data.
    pub const INDEX_BUFFERS: u8 = 50;

    /// Idle transaction buffers - not actively in use.
    pub const IDLE_TRANSACTION: u8 = 70;

    /// Graph storage - persistent data, expensive to reload.
    pub const GRAPH_STORAGE: u8 = 100;

    /// Active transaction data - highest priority, evict last.
    pub const ACTIVE_TRANSACTION: u8 = 200;
}

/// Statistics about a memory consumer.
#[derive(Debug, Clone)]
pub struct ConsumerStats {
    /// Consumer name.
    pub name: String,
    /// Memory region.
    pub region: MemoryRegion,
    /// Current memory usage in bytes.
    pub usage_bytes: usize,
    /// Eviction priority.
    pub priority: u8,
    /// Whether spilling is supported.
    pub can_spill: bool,
}

impl ConsumerStats {
    /// Creates stats from a consumer.
    pub fn from_consumer(consumer: &dyn MemoryConsumer) -> Self {
        Self {
            name: consumer.name().to_string(),
            region: consumer.region(),
            usage_bytes: consumer.memory_usage(),
            priority: consumer.eviction_priority(),
            can_spill: consumer.can_spill(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestConsumer {
        name: String,
        usage: AtomicUsize,
        priority: u8,
        region: MemoryRegion,
    }

    impl TestConsumer {
        fn new(name: &str, usage: usize, priority: u8, region: MemoryRegion) -> Self {
            Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
            }
        }
    }

    impl MemoryConsumer for TestConsumer {
        fn name(&self) -> &str {
            &self.name
        }

        fn memory_usage(&self) -> usize {
            self.usage.load(Ordering::Relaxed)
        }

        fn eviction_priority(&self) -> u8 {
            self.priority
        }

        fn region(&self) -> MemoryRegion {
            self.region
        }

        fn evict(&self, target_bytes: usize) -> usize {
            let current = self.usage.load(Ordering::Relaxed);
            let to_evict = target_bytes.min(current);
            self.usage.fetch_sub(to_evict, Ordering::Relaxed);
            to_evict
        }
    }

    #[test]
    fn test_consumer_stats() {
        let consumer = TestConsumer::new(
            "test",
            1024,
            priorities::INDEX_BUFFERS,
            MemoryRegion::IndexBuffers,
        );

        let stats = ConsumerStats::from_consumer(&consumer);
        assert_eq!(stats.name, "test");
        assert_eq!(stats.usage_bytes, 1024);
        assert_eq!(stats.priority, priorities::INDEX_BUFFERS);
        assert_eq!(stats.region, MemoryRegion::IndexBuffers);
        assert!(!stats.can_spill);
    }

    #[test]
    fn test_consumer_eviction() {
        let consumer = TestConsumer::new(
            "test",
            1000,
            priorities::INDEX_BUFFERS,
            MemoryRegion::IndexBuffers,
        );

        let freed = consumer.evict(500);
        assert_eq!(freed, 500);
        assert_eq!(consumer.memory_usage(), 500);

        // Try to evict more than available
        let freed = consumer.evict(1000);
        assert_eq!(freed, 500);
        assert_eq!(consumer.memory_usage(), 0);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(priorities::SPILL_STAGING < priorities::QUERY_CACHE);
        assert!(priorities::QUERY_CACHE < priorities::INDEX_BUFFERS);
        assert!(priorities::INDEX_BUFFERS < priorities::IDLE_TRANSACTION);
        assert!(priorities::IDLE_TRANSACTION < priorities::GRAPH_STORAGE);
        assert!(priorities::GRAPH_STORAGE < priorities::ACTIVE_TRANSACTION);
    }
}
