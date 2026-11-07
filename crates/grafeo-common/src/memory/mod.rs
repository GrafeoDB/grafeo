//! Custom allocators tuned for graph database workloads.
//!
//! These allocators give you better performance than the global allocator
//! for specific patterns. Pick the right one for your use case:
//!
//! | Allocator | Best for | Trade-off |
//! | --------- | -------- | --------- |
//! | [`arena`] | MVCC versioning, bulk alloc/dealloc | Can't free individual items |
//! | [`bump`] | Temporary data within a query | Must reset to free anything |
//! | [`pool`] | Frequently reused objects | Fixed-size objects only |
//! | [`buffer`] | Large data, memory pressure | More complex API |
//!
//! [`system`] detects the memory this process may use, the base of the default memory limit.

pub mod arena;
pub mod buffer;
pub mod bump;
pub mod pool;
pub mod reporter;
pub mod system;
pub mod usage;

pub use arena::{AllocError, Arena, ArenaAllocator};
pub use buffer::{
    BufferManager, BufferManagerConfig, BufferStats, GRANT_STEP, MemoryConsumer, MemoryGrant,
    MemoryRegion, PressureLevel, StorageTier,
};
pub use bump::BumpAllocator;
pub use pool::ObjectPool;
pub use reporter::MemoryReporter;
pub use system::{
    DEFAULT_MEMORY_FRACTION, DetectedMemory, FALLBACK_SYSTEM_MEMORY, MemorySource,
    WASM_PLATFORM_MEMORY, detect_memory,
};
pub use usage::{IndexMemory, MvccMemory, NamedMemory, StoreMemory, StringPoolMemory};
