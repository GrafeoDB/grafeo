//! Unified buffer manager for memory management.
//!
//! This module provides centralized memory management across all subsystems
//! with pressure-aware allocation, eviction, and spilling support.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                    BufferManager                           │
//! │  ┌──────────────┬──────────────┬──────────────┬──────────┐ │
//! │  │ GraphStorage │ IndexBuffers │ Execution    │ Spill    │ │
//! │  │              │              │ Buffers      │ Staging  │ │
//! │  └──────────────┴──────────────┴──────────────┴──────────┘ │
//! │                         │                                  │
//! │  Pressure Thresholds:   │                                  │
//! │  < 70%  Normal          │                                  │
//! │  70-85% Moderate (evict cold)                              │
//! │  85-95% High (aggressive evict/spill)                      │
//! │  > 95%  Critical (block allocations)                       │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use graphos_common::memory::buffer::{BufferManager, MemoryRegion};
//!
//! // Create with default config (75% of system RAM)
//! let manager = BufferManager::with_defaults();
//!
//! // Or with specific budget
//! let manager = BufferManager::with_budget(1024 * 1024 * 100); // 100MB
//!
//! // Allocate memory
//! if let Some(grant) = manager.try_allocate(1024, MemoryRegion::ExecutionBuffers) {
//!     // Use the memory...
//!     // Memory is automatically released when grant is dropped
//! }
//!
//! // Check pressure level
//! let level = manager.pressure_level();
//! if level.should_spill() {
//!     // Trigger spilling for spillable operators
//! }
//! ```

mod consumer;
mod grant;
mod manager;
mod region;
mod stats;

pub use consumer::{priorities, ConsumerStats, MemoryConsumer, SpillError};
pub use grant::{CompositeGrant, GrantReleaser, MemoryGrant};
pub use manager::{BufferManager, BufferManagerConfig};
pub use region::MemoryRegion;
pub use stats::{BufferStats, PressureLevel};
