//! Unified buffer manager implementation.

use super::consumer::MemoryConsumer;
use super::grant::{GrantReleaser, MemoryGrant};
use super::region::MemoryRegion;
use super::stats::{BufferStats, PressureLevel};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
        // Under Miri, file I/O is blocked by isolation: use fallback directly
        #[cfg(miri)]
        {
            return Self::fallback_system_memory();
        }

        // Try to detect system memory
        // On failure, return a conservative 1GB default
        #[cfg(not(miri))]
        {
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
                        if line.starts_with("MemTotal:")
                            && let Some(kb_str) = line.split_whitespace().nth(1)
                            && let Ok(kb) = kb_str.parse::<usize>()
                        {
                            return kb * 1024;
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
            // reason: memory fraction (0.0..1.0) of a positive usize is always a valid positive usize
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
    /// Set of consumer names pinned to RAM via `TierOverride::ForceRam`.
    ///
    /// Phase 8g enforcement: any spill loop ([`Self::run_eviction_internal`],
    /// [`Self::spill_all`], [`Self::spill_consumer_by_name`]) skips consumers
    /// whose name is in this set. Populated by the engine at startup based on
    /// `Config.section_configs`.
    force_ram_consumers: RwLock<HashSet<String>>,
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
        // reason: limit fractions (0.0..1.0) of a positive usize are always valid positive usizes
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let soft_limit = (config.budget as f64 * config.soft_limit_fraction) as usize;
        // reason: limit fractions (0.0..1.0) of a positive usize are always valid positive usizes
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let evict_limit = (config.budget as f64 * config.evict_limit_fraction) as usize;
        // reason: limit fractions (0.0..1.0) of a positive usize are always valid positive usizes
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
            force_ram_consumers: RwLock::new(HashSet::new()),
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
    pub fn try_allocate(
        self: &Arc<Self>,
        size: usize,
        region: MemoryRegion,
    ) -> Option<MemoryGrant> {
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
        // Also drop any ForceRam pin so re-registering with a different
        // tier doesn't carry over stale state.
        self.force_ram_consumers.write().remove(name);
    }

    /// Pins a consumer to RAM via [`crate::storage::TierOverride::ForceRam`].
    ///
    /// After this call, no spill loop in the buffer manager will spill the
    /// consumer with the given name. Allocation pressure that would otherwise
    /// have spilled this consumer instead falls through to other consumers,
    /// or fails if no other spillable consumer can free enough.
    ///
    /// Idempotent: calling twice with the same name is a no-op.
    pub fn mark_force_ram(&self, name: &str) {
        self.force_ram_consumers.write().insert(name.to_string());
    }

    /// Removes the [`crate::storage::TierOverride::ForceRam`] pin from a
    /// consumer (Phase 8g). After this call, the consumer participates in
    /// spill again like any other.
    pub fn clear_force_ram(&self, name: &str) {
        self.force_ram_consumers.write().remove(name);
    }

    /// Returns `true` if a consumer is currently pinned via ForceRam.
    #[must_use]
    pub fn is_force_ram(&self, name: &str) -> bool {
        self.force_ram_consumers.read().contains(name)
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

    /// Spills all consumers that support it, regardless of memory pressure.
    ///
    /// Used when `TierOverride::ForceDisk` is configured. Returns total bytes freed.
    /// Consumers pinned via [`Self::mark_force_ram`] are skipped.
    pub fn spill_all(&self) -> usize {
        let consumers = self.consumers.read();
        let force_ram = self.force_ram_consumers.read();
        let mut total_freed = 0;
        for consumer in consumers.iter() {
            if force_ram.contains(consumer.name()) {
                continue;
            }
            if consumer.can_spill()
                && let Ok(freed) = consumer.spill(usize::MAX)
            {
                total_freed += freed;
            }
        }
        total_freed
    }

    /// Spills all consumers whose [`MemoryConsumer::name`] equals `name`.
    ///
    /// Used for targeted [`crate::storage::TierOverride::ForceDisk`] enforcement
    /// at database open: each section type configured as `ForceDisk` triggers a
    /// spill on its matching consumer only, leaving other consumers untouched.
    ///
    /// Best-effort: a failure on one consumer does not stop the others. Returns
    /// total bytes freed across all matching consumers.
    ///
    /// If `name` is pinned via [`Self::mark_force_ram`], this call is a no-op
    /// and returns `0`. The pin is honored even on explicit-by-name spill
    /// requests: `ForceRam` is a hard contract.
    pub fn spill_consumer_by_name(&self, name: &str) -> usize {
        if self.is_force_ram(name) {
            #[cfg(feature = "tracing")]
            tracing::debug!(
                target: "grafeo::buffer",
                consumer = name,
                "spill skipped: consumer pinned ForceRam"
            );
            return 0;
        }
        let consumers = self.consumers.read();
        let mut total_freed = 0;
        for consumer in consumers.iter() {
            if consumer.name() == name
                && consumer.can_spill()
                && let Ok(freed) = consumer.spill(usize::MAX)
            {
                total_freed += freed;
            }
        }
        #[cfg(feature = "tracing")]
        tracing::info!(
            target: "grafeo::buffer",
            consumer = name,
            freed_bytes = total_freed,
            "tier transition: spill"
        );
        total_freed
    }

    /// Reloads OnDisk consumers back into RAM, in priority order (highest
    /// priority first), as long as projected usage stays below
    /// `target_fraction` of the budget.
    ///
    /// Phase 9a: closes the loop on the spill / reload lifecycle. Today
    /// consumers spill on memory pressure and stay OnDisk forever; this
    /// method gives users (or a future background thread) an explicit
    /// trigger to bring spilled state back into RAM after pressure drops.
    ///
    /// The walk visits consumers whose
    /// [`MemoryConsumer::current_tier`] is
    /// [`super::tiered::StorageTier::OnDisk`] and calls
    /// [`MemoryConsumer::reload`] on each. After each reload, if current
    /// allocation exceeds `target_fraction * budget`, the loop stops and
    /// leaves remaining consumers on disk. `reload()` errors are
    /// logged-and-skipped: the operation is best-effort.
    ///
    /// Returns the number of consumers successfully reloaded.
    ///
    /// `target_fraction` is clamped to `[0.0, 1.0]`. A value of 0.7 means
    /// "stop bringing things back when we'd hit 70% of the budget" —
    /// matching the soft-limit threshold default.
    pub fn reload_eligible(&self, target_fraction: f64) -> usize {
        let target_fraction = target_fraction.clamp(0.0, 1.0);
        // reason: target byte count from a bounded fraction is non-negative and bounded by budget
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target_bytes = (self.config.budget as f64 * target_fraction) as usize;

        let candidates: Vec<Arc<dyn MemoryConsumer>> = {
            let consumers = self.consumers.read();
            let mut out: Vec<_> = consumers
                .iter()
                .filter(|c| c.current_tier() == super::tiered::StorageTier::OnDisk)
                .map(Arc::clone)
                .collect();
            // Highest priority first: graph storage > active txn > index > query cache.
            out.sort_by_key(|c| std::cmp::Reverse(c.eviction_priority()));
            out
        };

        let mut reloaded = 0;
        for consumer in candidates {
            let current = self.allocated.load(Ordering::Relaxed);
            if current >= target_bytes {
                break;
            }
            match consumer.reload() {
                Ok(()) => {
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        target: "grafeo::buffer",
                        consumer = consumer.name(),
                        "tier transition: reload"
                    );
                    reloaded += 1;
                }
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        target: "grafeo::buffer",
                        consumer = consumer.name(),
                        error = %_e,
                        "tier reload failed"
                    );
                    continue;
                }
            }
        }
        #[cfg(feature = "tracing")]
        tracing::debug!(
            target: "grafeo::buffer",
            reloaded_count = reloaded,
            target_fraction = target_fraction,
            "reload_eligible cycle complete"
        );
        reloaded
    }

    /// Returns the current tier reported by each registered consumer that
    /// wraps a section.
    ///
    /// Tier is sourced from [`MemoryConsumer::current_tier`]. Consumers whose
    /// names don't follow the `"section:<TypeName>"` convention are skipped
    /// (e.g. CDC, overlay).
    #[must_use]
    pub fn snapshot_consumer_tiers(&self) -> Vec<(String, super::tiered::StorageTier)> {
        let consumers = self.consumers.read();
        consumers
            .iter()
            .filter_map(|c| {
                let name = c.name();
                if !name.starts_with("section:") {
                    return None;
                }
                Some((name.to_string(), c.current_tier()))
            })
            .collect()
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
        self.config
            .budget
            .saturating_sub(self.allocated.load(Ordering::Relaxed))
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
        let force_ram = self.force_ram_consumers.read();

        // Sort consumers by priority (lowest first = evict first)
        let mut sorted: Vec<_> = consumers.iter().collect();
        sorted.sort_by_key(|c| c.eviction_priority());

        let mut total_freed = 0;
        for consumer in &sorted {
            if total_freed >= to_free {
                break;
            }

            // Phase 8g: respect ForceRam pin even on the evict path.
            // evict() is typically a no-op for these consumers anyway,
            // but skipping is the principled choice.
            if force_ram.contains(consumer.name()) {
                continue;
            }

            let remaining = to_free - total_freed;
            let consumer_usage = consumer.memory_usage();

            // Ask consumer to evict up to half its usage or remaining needed
            let target_evict = remaining.min(consumer_usage / 2);
            if target_evict > 0 {
                let freed = consumer.evict(target_evict);
                total_freed += freed;
            }
        }

        // If eviction was not enough, try spilling to disk for consumers
        // that support it (e.g., vector indexes with mmap storage).
        // Phase 8g: ForceRam consumers are skipped here too — that's the
        // hard contract ("never move me to disk").
        if total_freed < to_free {
            for consumer in &sorted {
                if total_freed >= to_free {
                    break;
                }
                if force_ram.contains(consumer.name()) {
                    continue;
                }
                if !consumer.can_spill() {
                    continue;
                }
                let remaining = to_free - total_freed;
                match consumer.spill(remaining) {
                    Ok(freed) => total_freed += freed,
                    Err(_) => continue,
                }
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

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            if self.memory_usage() == 0 {
                super::super::tiered::StorageTier::Uninitialized
            } else {
                super::super::tiered::StorageTier::InMemory
            }
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

    // --- Spill-aware test consumer ---

    struct SpillableConsumer {
        name: String,
        usage: AtomicUsize,
        priority: u8,
        region: MemoryRegion,
        evicted: AtomicUsize,
        spilled: AtomicUsize,
        spillable: bool,
        evict_returns_zero: bool,
    }

    impl SpillableConsumer {
        fn new(
            name: &str,
            usage: usize,
            priority: u8,
            region: MemoryRegion,
            spillable: bool,
        ) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
                evicted: AtomicUsize::new(0),
                spilled: AtomicUsize::new(0),
                spillable,
                evict_returns_zero: false,
            })
        }

        fn new_evict_fails(
            name: &str,
            usage: usize,
            priority: u8,
            region: MemoryRegion,
            spillable: bool,
        ) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
                evicted: AtomicUsize::new(0),
                spilled: AtomicUsize::new(0),
                spillable,
                evict_returns_zero: true,
            })
        }
    }

    impl MemoryConsumer for SpillableConsumer {
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
            if self.evict_returns_zero {
                return 0;
            }
            let current = self.usage.load(Ordering::Relaxed);
            let to_evict = target_bytes.min(current);
            self.usage.fetch_sub(to_evict, Ordering::Relaxed);
            self.evicted.fetch_add(to_evict, Ordering::Relaxed);
            to_evict
        }

        fn can_spill(&self) -> bool {
            self.spillable
        }

        fn spill(
            &self,
            target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            if !self.spillable {
                return Err(crate::memory::buffer::consumer::SpillError::NotSupported);
            }
            let current = self.usage.load(Ordering::Relaxed);
            let to_spill = target_bytes.min(current);
            self.usage.fetch_sub(to_spill, Ordering::Relaxed);
            self.spilled.fetch_add(to_spill, Ordering::Relaxed);
            Ok(to_spill)
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            if self.spilled.load(Ordering::Relaxed) > 0 {
                super::super::tiered::StorageTier::OnDisk
            } else if self.memory_usage() == 0 {
                super::super::tiered::StorageTier::Uninitialized
            } else {
                super::super::tiered::StorageTier::InMemory
            }
        }
    }

    #[test]
    fn test_spill_all_calls_spillable_consumers() {
        let manager = BufferManager::with_budget(10000);
        let spillable = SpillableConsumer::new(
            "spillable",
            500,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            true,
        );
        let non_spillable = SpillableConsumer::new(
            "non_spillable",
            500,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            false,
        );
        manager.register_consumer(Arc::clone(&spillable) as Arc<dyn MemoryConsumer>);
        manager.register_consumer(Arc::clone(&non_spillable) as Arc<dyn MemoryConsumer>);

        let freed = manager.spill_all();
        assert_eq!(freed, 500);
        assert_eq!(spillable.spilled.load(Ordering::Relaxed), 500);
        assert_eq!(non_spillable.spilled.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_spill_all_skips_non_spillable() {
        let manager = BufferManager::with_budget(10000);
        let consumer = SpillableConsumer::new(
            "no_spill",
            1000,
            priorities::INDEX_BUFFERS,
            MemoryRegion::IndexBuffers,
            false,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);

        assert_eq!(manager.spill_all(), 0);
        assert_eq!(consumer.memory_usage(), 1000);
    }

    #[test]
    fn test_eviction_falls_back_to_spill() {
        let manager = BufferManager::with_budget(10000);
        let consumer = SpillableConsumer::new_evict_fails(
            "spill_fallback",
            1000,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            true,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(2000, Ordering::Relaxed);

        let freed = manager.evict_to_target(1500);
        assert_eq!(consumer.evicted.load(Ordering::Relaxed), 0);
        assert!(consumer.spilled.load(Ordering::Relaxed) > 0);
        assert!(freed > 0);
    }

    #[test]
    fn test_eviction_no_spill_when_sufficient() {
        let manager = BufferManager::with_budget(10000);
        let consumer = SpillableConsumer::new(
            "eviction_enough",
            1000,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            true,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(1200, Ordering::Relaxed);

        let freed = manager.evict_to_target(1000);
        assert_eq!(freed, 200);
        assert_eq!(consumer.spilled.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_eviction_spill_skips_non_spillable() {
        let manager = BufferManager::with_budget(10000);
        let consumer = SpillableConsumer::new_evict_fails(
            "no_spill",
            1000,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            false,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(2000, Ordering::Relaxed);

        let freed = manager.evict_to_target(1500);
        assert_eq!(freed, 0);
        assert_eq!(consumer.memory_usage(), 1000);
    }

    #[test]
    fn alix_with_defaults_creates_manager() {
        let manager = BufferManager::with_defaults();
        // with_defaults uses system memory detection, budget should be > 0
        assert!(manager.budget() > 0);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(manager.available(), manager.budget());
    }

    #[test]
    fn gus_config_accessor_returns_budget() {
        let manager = BufferManager::with_budget(4096);
        let config = manager.config();
        assert_eq!(config.budget, 4096);
        assert!(!config.background_eviction);
        assert!(config.spill_path.is_none());
    }

    #[test]
    fn vincent_shutdown_sets_flag() {
        let manager = BufferManager::with_budget(1000);
        manager.shutdown();
        // shutdown stores true; drop also stores true, so this just verifies
        // the method runs without error and the manager remains usable
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    fn jules_critical_pressure_level() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Manually set allocated above hard limit to test Critical level
        manager.allocated.store(960, Ordering::Relaxed);
        assert_eq!(manager.pressure_level(), PressureLevel::Critical);
    }

    #[test]
    fn mia_evict_to_target_already_below() {
        let manager = BufferManager::with_budget(10000);
        // allocated is 0, target is 5000: already below target
        let freed = manager.evict_to_target(5000);
        assert_eq!(freed, 0);
    }

    #[test]
    fn butch_try_allocate_raw_success() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // GrantReleaser::try_allocate_raw succeeds when under hard limit
        let success = manager.try_allocate_raw(100, MemoryRegion::GraphStorage);
        assert!(success);
        assert_eq!(manager.allocated(), 100);
        assert_eq!(
            manager.stats().region_usage(MemoryRegion::GraphStorage),
            100
        );
    }

    #[test]
    fn django_try_allocate_raw_fails_at_hard_limit() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Fill up to hard limit
        manager.allocated.store(940, Ordering::Relaxed);

        // This exceeds hard limit (940 + 100 = 1040 > 950), no consumers to evict
        let success = manager.try_allocate_raw(100, MemoryRegion::ExecutionBuffers);
        assert!(!success);
    }

    #[test]
    fn shosanna_drop_sets_shutdown() {
        // Create and immediately drop to exercise the Drop impl
        let manager = BufferManager::with_budget(512);
        drop(manager);
        // If we get here without panic, the Drop impl ran successfully.
    }

    #[test]
    fn hans_eviction_with_zero_usage_consumer() {
        let manager = BufferManager::with_budget(10000);
        // Consumer with zero usage: target_evict will be 0, so evict is skipped
        let consumer = TestConsumer::new(
            "empty",
            0,
            priorities::SPILL_STAGING,
            MemoryRegion::SpillStaging,
        );
        manager.register_consumer(Arc::clone(&consumer) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(500, Ordering::Relaxed);

        let freed = manager.evict_to_target(200);
        // Consumer has 0 usage, so target_evict = min(300, 0/2) = 0, evict skipped
        assert_eq!(consumer.evicted.load(Ordering::Relaxed), 0);
        assert_eq!(freed, 0);
    }

    #[test]
    fn beatrix_grant_releaser_release_decrements() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Allocate via try_allocate_raw, then release via GrantReleaser trait
        assert!(manager.try_allocate_raw(200, MemoryRegion::IndexBuffers));
        assert_eq!(manager.allocated(), 200);

        manager.release(200, MemoryRegion::IndexBuffers);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(manager.stats().region_usage(MemoryRegion::IndexBuffers), 0);
    }

    /// Consumer whose spill() returns an error to exercise the Err(_) => continue path.
    struct FailingSpillConsumer {
        name: String,
        usage: AtomicUsize,
        priority: u8,
        region: MemoryRegion,
    }

    impl FailingSpillConsumer {
        fn new(name: &str, usage: usize, priority: u8, region: MemoryRegion) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                usage: AtomicUsize::new(usage),
                priority,
                region,
            })
        }
    }

    impl MemoryConsumer for FailingSpillConsumer {
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

        fn evict(&self, _target_bytes: usize) -> usize {
            0 // eviction always fails
        }

        fn can_spill(&self) -> bool {
            true
        }

        fn spill(
            &self,
            _target_bytes: usize,
        ) -> Result<usize, crate::memory::buffer::consumer::SpillError> {
            Err(crate::memory::buffer::consumer::SpillError::IoError(
                "disk full".to_string(),
            ))
        }

        fn current_tier(&self) -> super::super::tiered::StorageTier {
            if self.memory_usage() == 0 {
                super::super::tiered::StorageTier::Uninitialized
            } else {
                super::super::tiered::StorageTier::InMemory
            }
        }
    }

    #[test]
    fn vincent_spill_error_continues_to_next_consumer() {
        let manager = BufferManager::with_budget(10000);

        // First consumer: spill fails
        let failing = FailingSpillConsumer::new(
            "failing_spill",
            500,
            priorities::SPILL_STAGING,
            MemoryRegion::SpillStaging,
        );

        // Second consumer: spill succeeds
        let working = SpillableConsumer::new_evict_fails(
            "working_spill",
            500,
            priorities::QUERY_CACHE,
            MemoryRegion::ExecutionBuffers,
            true,
        );

        manager.register_consumer(Arc::clone(&failing) as Arc<dyn MemoryConsumer>);
        manager.register_consumer(Arc::clone(&working) as Arc<dyn MemoryConsumer>);
        manager.allocated.store(2000, Ordering::Relaxed);

        let freed = manager.evict_to_target(1500);
        // failing consumer's spill errors out, working consumer's spill succeeds
        assert!(working.spilled.load(Ordering::Relaxed) > 0);
        assert!(freed > 0);
    }

    #[test]
    fn django_detect_system_memory_returns_positive() {
        let mem = BufferManagerConfig::detect_system_memory();
        assert!(mem > 0);
    }

    #[test]
    fn shosanna_spill_path_config() {
        let config = BufferManagerConfig {
            budget: 1024,
            spill_path: Some(PathBuf::from("/tmp/grafeo-spill")),
            ..Default::default()
        };
        assert_eq!(
            config.spill_path.as_ref().unwrap().to_str().unwrap(),
            "/tmp/grafeo-spill"
        );
        let manager = BufferManager::new(config);
        assert!(manager.config().spill_path.is_some());
    }
}
