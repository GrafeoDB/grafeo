//! Disk spilling for queries that exceed available memory.
//!
//! When a sort or aggregation grows too large for RAM, we spill partitions to
//! disk and merge them back later. This lets queries complete even with limited
//! memory - just slower.
//!
//! | Component | Purpose |
//! | --------- | ------- |
//! | [`SpillManager`] | Manages spill file lifecycle with automatic cleanup |
//! | [`SpillFile`] | Read/write individual spill files |
//! | [`ExternalSort`] | External merge sort for big ORDER BY |
//! | [`PartitionedState`] | Hash partitioning for spillable GROUP BY |
//!
//! Async variants (`AsyncSpillManager`, `AsyncSpillFile`) use tokio for
//! non-blocking I/O when running in async contexts.

mod async_file;
mod async_manager;
mod external_sort;
mod file;
mod manager;
mod partition;
mod serializer;

pub use async_file::{AsyncSpillFile, AsyncSpillFileReader};
pub use async_manager::AsyncSpillManager;
pub use external_sort::{ExternalSort, NullOrder, SortDirection, SortKey};
pub use file::{SpillFile, SpillFileReader};
pub use manager::SpillManager;
pub use partition::{DEFAULT_NUM_PARTITIONS, PartitionedState};
pub use serializer::{deserialize_row, deserialize_value, serialize_row, serialize_value};
