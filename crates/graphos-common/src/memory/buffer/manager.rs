//! Unified buffer manager implementation.

use super::consumer::MemoryConsumer;
use super::grant::{GrantReleaser, MemoryGrant};
use super::region::MemoryRegion;
use super::stats::{BufferStats, PressureLevel};
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Default memory budget as a fraction of system memory.
const DEFAULT_MEMORY_FRACTION: f64 = 0.75;

/// Configuration for the buffer manager.
#[derive(Debug, Clone)]
pub struct BufferManagerConfig {
    /// Total memory budget in bytes.
    pub budget: usize,
    /// Soft limit threshold (default: 70%).
    pub soft_limit_fraction: f64,
    /// Eviction threshold (default: 85%).
    pub evict_limit_fraction: f64,
    /// Hard limit threshold (default: 95%).
    pub hard_limit_fraction: f64,
    /// Enable background eviction thread.
    pub background_eviction: bool,
    /// Directory for spilling data to disk.
    pub spill_path: Option<PathBuf>,
}

impl BufferManagerConfig {
    /// Detects system memory size.
    ///
    /// Returns a conservative estimate if detection fails.
    #[must_use]
    pub fn detect_system_memory() -> usize {
        // Try to detect system memory
        // On failure, return a conservative 1GB default
        #[cfg(target_os = "windows")]
        {
            // Windows: Use GetPhysicallyInstalledSystemMemory or GlobalMemoryStatusEx
            // For now, use a fallback
            Self::fallback_system_memory()
        }

        #[cfg(target_os = "linux")]
        {
            // Linux: Read from /proc/meminfo
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                for line in contents.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024;
                            }
                        }
                    }
                }
            }
            Self::fallback_system_memory()
        }

        #[cfg(target_os = "macos")]
        {
            // macOS: Use sysctl
            Self::fallback_system_memory()
        }

        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            Self::fallback_system_memory()
        }
    }

    fn fallback_system_memory() -> usize {
        // Default to 1GB if detection fails
        1024 * 1024 * 1024
    }

    /// Creates a config with the given budget.
    #[must_use]
    pub fn with_budget(budget: usize) -> Self {
        Self {
            budget,
            ..Default::default()
        }
    }
}

impl Default for BufferManagerConfig {
    fn default() -> Self {
        let system_memory = Self::detect_system_memory();
        Self {
            budget: (system_memory as f64 * DEFAULT_MEMORY_FRACTION) as usize,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false, // Disabled by default for simplicity
            spill_path: None,
        }
    }
}

/// The central unified buffer manager.
///
/// Manages memory allocation across all subsystems with pressure-aware
/// eviction and optional spilling support.
pub struct BufferManager {
    /// Configuration.
    config: BufferManagerConfig,
    /// Total allocated bytes.
    allocated: AtomicUsize,
    /// Per-region allocated bytes.
    region_allocated: [AtomicUsize; 4],
    /// Registered memory consumers.
    consumers: RwLock<Vec<Arc<dyn MemoryConsumer>>>,
    /// Computed soft limit in bytes.
    soft_limit: usize,
    /// Computed eviction limit in bytes.
    evict_limit: usize,
    /// Computed hard limit in bytes.
    hard_limit: usize,
    /// Shutdown flag.
    shutdown: AtomicBool,
}

impl BufferManager {
    /// Creates a new buffer manager with the given configuration.
    #[must_use]
    pub fn new(config: BufferManagerConfig) -> Arc<Self> {
        let soft_limit = (config.budget as f64 * config.soft_limit_fraction) as usize;
        let evict_limit = (config.budget as f64 * config.evict_limit_fraction) as usize;
        let hard_limit = (config.budget as f64 * config.hard_limit_fraction) as usize;

        Arc::new(Self {
            config,
            allocated: AtomicUsize::new(0),
            region_allocated: [
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
            ],
            consumers: RwLock::new(Vec::new()),
            soft_limit,
            evict_limit,
            hard_limit,
            shutdown: AtomicBool::new(false),
        })
    }

    /// Creates a buffer manager with default configuration.
    #[must_use]
    pub fn with_defaults() -> Arc<Self> {
        Self::new(BufferManagerConfig::default())
    }

    /// Creates a buffer manager with a specific budget.
    #[must_use]
    pub fn with_budget(budget: usize) -> Arc<Self> {
        Self::new(BufferManagerConfig::with_budget(budget))
    }

    /// Attempts to allocate memory for the given region.
    ///
    /// Returns `None` if allocation would exceed the hard limit after
    /// eviction attempts.
    pub fn try_allocate(self: &Arc<Self>, size: usize, region: MemoryRegion) -> Option<MemoryGrant> {
        // Check if we can allocate
        let current = self.allocated.load(Ordering::Relaxed);

        if current + size > self.hard_limit {
            // Try eviction first
            self.run_eviction_cycle(true);

            // Check again
            let current = self.allocated.load(Ordering::Relaxed);
            if current + size > self.hard_limit {
                return None;
            }
        }

        // Perform allocation
        self.allocated.fetch_add(size, Ordering::Relaxed);
        self.region_allocated[region.index()].fetch_add(size, Ordering::Relaxed);

        // Check pressure and potentially trigger background eviction
        self.check_pressure();

        Some(MemoryGrant::new(
            Arc::clone(self) as Arc<dyn GrantReleaser>,
            size,
            region,
        ))
    }

    /// Returns the current pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        let current = self.allocated.load(Ordering::Relaxed);
        self.compute_pressure_level(current)
    }

    /// Returns current buffer statistics.
    #[must_use]
    pub fn stats(&self) -> BufferStats {
        let total_allocated = self.allocated.load(Ordering::Relaxed);
        BufferStats {
            budget: self.config.budget,
            total_allocated,
            region_allocated: [
                self.region_allocated[0].load(Ordering::Relaxed),
                self.region_allocated[1].load(Ordering::Relaxed),
                self.region_allocated[2].load(Ordering::Relaxed),
                self.region_allocated[3].load(Ordering::Relaxed),
            ],
            pressure_level: self.compute_pressure_level(total_allocated),
            consumer_count: self.consumers.read().len(),
        }
    }

    /// Registers a memory consumer for eviction callbacks.
    pub fn register_consumer(&self, consumer: Arc<dyn MemoryConsumer>) {
        self.consumers.write().push(consumer);
    }

    /// Unregisters a memory consumer by name.
    pub fn unregister_consumer(&self, name: &str) {
        self.consumers.write().retain(|c| c.name() != name);
    }

    /// Forces eviction to reach the target usage.
    ///
    /// Returns the number of bytes actually freed.
    pub fn evict_to_target(&self, target_bytes: usize) -> usize {
        let current = self.allocated.load(Ordering::Relaxed);
        if current <= target_bytes {
            return 0;
        }

        let to_free = current - target_bytes;
        self.run_eviction_internal(to_free)
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &BufferManagerConfig {
        &self.config
    }

    /// Returns the memory budget.
    #[must_use]
    pub fn budget(&self) -> usize {
        self.config.budget
    }

    /// Returns currently allocated bytes.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Returns available bytes.
    #[must_use]
    pub fn available(&self) -> usize {
        self.config.budget.saturating_sub(self.allocated.load(Ordering::Relaxed))
    }

    /// Shuts down the buffer manager.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    // === Internal methods ===

    fn compute_pressure_level(&self, current: usize) -> PressureLevel {
        if current >= self.hard_limit {
            PressureLevel::Critical
        } else if current >= self.evict_limit {
            PressureLevel::High
        } else if current >= self.soft_limit {
            PressureLevel::Moderate
        } else {
            PressureLevel::Normal
        }
    }

    fn check_pressure(&self) {
        let level = self.pressure_level();
        if level.requires_eviction() {
            // In a more complete implementation, this would signal
            // a background thread. For now, do synchronous eviction.
            let aggressive = level >= PressureLevel::High;
            self.run_eviction_cycle(aggressive);
        }
    }

    fn run_eviction_cycle(&self, aggressive: bool) -> usize {
        let target = if aggressive {
            self.soft_limit
        } else {
            self.evict_limit
        };

        let current = self.allocated.load(Ordering::Relaxed);
        if current <= target {
            return 0;
        }

        let to_free = current - target;
        self.run_eviction_internal(to_free)
    }

    fn run_eviction_internal(&self, to_free: usize) -> usize {
        let consumers = self.consumers.read();

        // Sort consumers by priority (lowest first = evict first)
        let mut sorted: Vec<_> = consumers.iter().collect();
        sorted.sort_by_key(|c| c.eviction_priority());

        let mut total_freed = 0;
        for consumer in sorted {
            if total_freed >= to_free {
                break;
            }

            let remaining = to_free - total_freed;
            let consumer_usage = consumer.memory_usage();

            // Ask consumer to evict up to half its usage or remaining needed
            let target_evict = remaining.min(consumer_usage / 2);
            if target_evict > 0 {
                let freed = consumer.evict(target_evict);
                total_freed += freed;
                // Note: consumers should call release through their grants,
                // so we don't double-decrement here.
            }
        }

        total_freed
    }
}

impl GrantReleaser for BufferManager {
    fn release(&self, size: usize, region: MemoryRegion) {
        self.allocated.fetch_sub(size, Ordering::Relaxed);
        self.region_allocated[region.index()].fetch_sub(size, Ordering::Relaxed);
    }

    fn try_allocate_raw(&self, size: usize, region: MemoryRegion) -> bool {
        let current = self.allocated.load(Ordering::Relaxed);

        if current + size > self.hard_limit {
            // Try eviction
            self.run_eviction_cycle(true);

            let current = self.allocated.load(Ordering::Relaxed);
            if current + size > self.hard_limit {
                return false;
            }
        }

        self.allocated.fetch_add(size, Ordering::Relaxed);
        self.region_allocated[region.index()].fetch_add(size, Ordering::Relaxed);
        true
    }
}

impl Drop for BufferManager {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::buffer::consumer::priorities;
    use std::sync::atomic::AtomicUsize;

    struct TestConsumer {
        name: String,
        usage: AtomicUsize,
        priority: u8,
        region: MemoryRegion,
        evicted: AtomicUsize,
    }

    impl TestConsumer {
        fn new(name: &str, usage: usize, priority: u8, region: MemoryRegion) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
                evicted: AtomicUsize::new(0),
            })
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
            self.evicted.fetch_add(to_evict, Ordering::Relaxed);
            to_evict
        }
    }

    #[test]
    fn test_basic_allocation() {
        let config = BufferManagerConfig {
            budget: 1024 * 1024, // 1MB
            ..Default::default()
        };
        let manager = BufferManager::new(config);

        let grant = manager.try_allocate(1024, MemoryRegion::ExecutionBuffers);
        assert!(grant.is_some());
        assert_eq!(manager.stats().total_allocated, 1024);
    }

    #[test]
    fn test_grant_raii_release() {
        let config = BufferManagerConfig {
            budget: 1024,
            ..Default::default()
        };
        let manager = BufferManager::new(config);

        {
            let _grant = manager.try_allocate(512, MemoryRegion::ExecutionBuffers);
            assert_eq!(manager.stats().total_allocated, 512);
        }

        // Grant dropped, memory should be released
        assert_eq!(manager.stats().total_allocated, 0);
    }

    #[test]
    fn test_pressure_levels() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        assert_eq!(manager.pressure_level(), PressureLevel::Normal);

        // Allocate to 70% (soft limit)
        let _g1 = manager.try_allocate(700, MemoryRegion::ExecutionBuffers);
        assert_eq!(manager.pressure_level(), PressureLevel::Moderate);

        // Allocate to 85% (evict limit)
        let _g2 = manager.try_allocate(150, MemoryRegion::ExecutionBuffers);
        assert_eq!(manager.pressure_level(), PressureLevel::High);

        // Note: Can't easily test Critical without blocking
    }

    #[test]
    fn test_region_tracking() {
        let config = BufferManagerConfig {
            budget: 10000,
            ..Default::default()
        };
        let manager = BufferManager::new(config);

        let _g1 = manager.try_allocate(100, MemoryRegion::GraphStorage);
        let _g2 = manager.try_allocate(200, MemoryRegion::IndexBuffers);
        let _g3 = manager.try_allocate(300, MemoryRegion::ExecutionBuffers);

        let stats = manager.stats();
        assert_eq!(stats.region_usage(MemoryRegion::GraphStorage), 100);
        assert_eq!(stats.region_usage(MemoryRegion::IndexBuffers), 200);
        assert_eq!(stats.region_usage(MemoryRegion::ExecutionBuffers), 300);
        assert_eq!(stats.total_allocated, 600);
    }

    #[test]
    fn test_consumer_registration() {
        let manager = BufferManager::with_budget(10000);

        let consumer = TestConsumer::new(
            "test",
            1000,
            priorities::INDEX_BUFFERS,
            MemoryRegion::IndexBuffers,
        );

        manager.register_consumer(consumer);
        assert_eq!(manager.stats().consumer_count, 1);

        manager.unregister_consumer("test");
        assert_eq!(manager.stats().consumer_count, 0);
    }

    #[test]
    fn test_eviction_ordering() {
        let manager = BufferManager::with_budget(10000);

        // Low priority consumer (evict first)
        let low_priority = TestConsumer::new(
            "low",
            500,
            priorities::SPILL_STAGING,
            MemoryRegion::SpillStaging,
        );

        // High priority consumer (evict last)
        let high_priority = TestConsumer::new(
            "high",
            500,
            priorities::ACTIVE_TRANSACTION,
            MemoryRegion::ExecutionBuffers,
        );

        manager.register_consumer(Arc::clone(&low_priority) as Arc<dyn MemoryConsumer>);
        manager.register_consumer(Arc::clone(&high_priority) as Arc<dyn MemoryConsumer>);

        // Manually set allocated to simulate memory usage
        // (consumers track their own usage separately from manager's allocation tracking)
        manager.allocated.store(1000, Ordering::Relaxed);

        // Request eviction to target 700 (need to free 300 bytes)
        let freed = manager.evict_to_target(700);

        // Low priority should be evicted first (up to half = 250)
        assert!(low_priority.evicted.load(Ordering::Relaxed) > 0);
        assert!(freed > 0);
    }

    #[test]
    fn test_hard_limit_blocking() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Allocate up to hard limit (950 bytes)
        let _g1 = manager.try_allocate(950, MemoryRegion::ExecutionBuffers);

        // This should fail (would exceed hard limit)
        let g2 = manager.try_allocate(100, MemoryRegion::ExecutionBuffers);
        assert!(g2.is_none());
    }

    #[test]
    fn test_available_memory() {
        let manager = BufferManager::with_budget(1000);

        assert_eq!(manager.available(), 1000);

        let _g = manager.try_allocate(300, MemoryRegion::ExecutionBuffers);
        assert_eq!(manager.available(), 700);
    }
}
