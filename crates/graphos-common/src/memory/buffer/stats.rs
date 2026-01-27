//! Buffer manager statistics and pressure levels.

use super::region::MemoryRegion;

/// Memory pressure level thresholds.
///
/// The buffer manager uses these levels to determine when to
/// trigger eviction and spilling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PressureLevel {
    /// < 70% used - Normal operation, no action needed.
    Normal,
    /// 70-85% used - Begin evicting cold data proactively.
    Moderate,
    /// 85-95% used - Aggressive eviction, trigger spilling.
    High,
    /// > 95% used - Critical, block new allocations until memory freed.
    Critical,
}

impl PressureLevel {
    /// Returns a human-readable description of this pressure level.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Normal => "Normal operation",
            Self::Moderate => "Proactive eviction",
            Self::High => "Aggressive eviction/spilling",
            Self::Critical => "Blocking allocations",
        }
    }

    /// Returns whether this level requires eviction action.
    #[must_use]
    pub const fn requires_eviction(&self) -> bool {
        matches!(self, Self::Moderate | Self::High | Self::Critical)
    }

    /// Returns whether this level should trigger spilling.
    #[must_use]
    pub const fn should_spill(&self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }

    /// Returns whether allocations should be blocked at this level.
    #[must_use]
    pub const fn blocks_allocations(&self) -> bool {
        matches!(self, Self::Critical)
    }
}

impl std::fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::Moderate => write!(f, "Moderate"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Statistics about buffer manager state.
#[derive(Debug, Clone)]
pub struct BufferStats {
    /// Total memory budget in bytes.
    pub budget: usize,
    /// Total allocated bytes across all regions.
    pub total_allocated: usize,
    /// Per-region allocation in bytes.
    pub region_allocated: [usize; 4],
    /// Current pressure level.
    pub pressure_level: PressureLevel,
    /// Number of registered consumers.
    pub consumer_count: usize,
}

impl BufferStats {
    /// Returns the utilization as a fraction (0.0 to 1.0).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.budget == 0 {
            return 0.0;
        }
        self.total_allocated as f64 / self.budget as f64
    }

    /// Returns the utilization as a percentage (0 to 100).
    #[must_use]
    pub fn utilization_percent(&self) -> f64 {
        self.utilization() * 100.0
    }

    /// Returns allocated bytes for a specific region.
    #[must_use]
    pub fn region_usage(&self, region: MemoryRegion) -> usize {
        self.region_allocated[region.index()]
    }

    /// Returns available bytes (budget - allocated).
    #[must_use]
    pub fn available(&self) -> usize {
        self.budget.saturating_sub(self.total_allocated)
    }
}

impl Default for BufferStats {
    fn default() -> Self {
        Self {
            budget: 0,
            total_allocated: 0,
            region_allocated: [0; 4],
            pressure_level: PressureLevel::Normal,
            consumer_count: 0,
        }
    }
}

impl std::fmt::Display for BufferStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Buffer Manager Stats:")?;
        writeln!(
            f,
            "  Budget: {} ({:.2}% used)",
            format_bytes(self.budget),
            self.utilization_percent()
        )?;
        writeln!(
            f,
            "  Allocated: {} / {}",
            format_bytes(self.total_allocated),
            format_bytes(self.budget)
        )?;
        writeln!(f, "  Pressure: {}", self.pressure_level)?;
        writeln!(f, "  Consumers: {}", self.consumer_count)?;
        writeln!(f, "  Per-region:")?;
        for region in MemoryRegion::all() {
            writeln!(
                f,
                "    {}: {}",
                region.name(),
                format_bytes(self.region_usage(region))
            )?;
        }
        Ok(())
    }
}

/// Formats bytes in human-readable form.
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_level_ordering() {
        assert!(PressureLevel::Normal < PressureLevel::Moderate);
        assert!(PressureLevel::Moderate < PressureLevel::High);
        assert!(PressureLevel::High < PressureLevel::Critical);
    }

    #[test]
    fn test_pressure_level_properties() {
        assert!(!PressureLevel::Normal.requires_eviction());
        assert!(PressureLevel::Moderate.requires_eviction());
        assert!(PressureLevel::High.requires_eviction());
        assert!(PressureLevel::Critical.requires_eviction());

        assert!(!PressureLevel::Normal.should_spill());
        assert!(!PressureLevel::Moderate.should_spill());
        assert!(PressureLevel::High.should_spill());
        assert!(PressureLevel::Critical.should_spill());

        assert!(!PressureLevel::Normal.blocks_allocations());
        assert!(!PressureLevel::High.blocks_allocations());
        assert!(PressureLevel::Critical.blocks_allocations());
    }

    #[test]
    fn test_buffer_stats_utilization() {
        let stats = BufferStats {
            budget: 1000,
            total_allocated: 750,
            region_allocated: [250, 250, 200, 50],
            pressure_level: PressureLevel::Moderate,
            consumer_count: 3,
        };

        assert!((stats.utilization() - 0.75).abs() < 0.001);
        assert!((stats.utilization_percent() - 75.0).abs() < 0.1);
        assert_eq!(stats.available(), 250);
    }

    #[test]
    fn test_buffer_stats_region_usage() {
        let stats = BufferStats {
            budget: 1000,
            total_allocated: 600,
            region_allocated: [100, 200, 250, 50],
            pressure_level: PressureLevel::Normal,
            consumer_count: 2,
        };

        assert_eq!(stats.region_usage(MemoryRegion::GraphStorage), 100);
        assert_eq!(stats.region_usage(MemoryRegion::IndexBuffers), 200);
        assert_eq!(stats.region_usage(MemoryRegion::ExecutionBuffers), 250);
        assert_eq!(stats.region_usage(MemoryRegion::SpillStaging), 50);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
