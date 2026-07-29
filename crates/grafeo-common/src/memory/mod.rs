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

pub mod arena;
pub mod buffer;
pub mod bump;
pub mod pool;
pub mod reporter;
pub mod usage;

pub use arena::{AllocError, Arena, ArenaAllocator};
#[allow(deprecated)]
pub use buffer::TieredStore;
pub use buffer::{
    BufferManager, BufferManagerConfig, BufferStats, MemoryConsumer, MemoryGrant, MemoryRegion,
    PressureLevel, StorageTier,
};
pub use bump::BumpAllocator;
pub use pool::ObjectPool;
pub use reporter::MemoryReporter;
pub use usage::{
    AdjacencyCapacityMemory, IndexMemory, LpgResidencyMemory, MvccMemory, NamedMemory,
    PropertyColumnMemory, PropertyStorageMemory, StoreMemory, StringPoolMemory,
};
