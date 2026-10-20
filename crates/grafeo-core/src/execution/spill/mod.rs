//! Disk spilling for queries that exceed available memory.
//!
//! When a sort or aggregation grows too large for RAM, we spill partitions to
//! disk and merge them back later. This lets queries complete even with limited
//! memory, just slower.
//!
//! | Component | Purpose |
//! | --------- | ------- |
//! | [`SpillManager`] | Manages spill file lifecycle with automatic cleanup |
//! | [`SpillFile`] | Read/write individual spill files |
//! | [`ExternalSort`] | External merge sort for big ORDER BY |
//! | [`PartitionedState`] | Hash partitioning for spillable GROUP BY |
//!
//! Async variants (`AsyncSpillManager`, `AsyncSpillFile`) live in
//! `grafeo-engine::execution::spill` (requires tokio runtime).

mod external_sort;
mod file;
mod manager;
mod partition;
mod serializer;

pub use external_sort::{ExternalSort, NullOrder, SortDirection, SortKey};
pub use file::{SpillFile, SpillFileReader};
pub use manager::SpillManager;
pub(crate) use partition::OwnedPartitionDrainCursor;
pub use partition::{DEFAULT_NUM_PARTITIONS, PartitionedState};
pub use serializer::{deserialize_row, deserialize_value, serialize_row, serialize_value};
pub(crate) use serializer::{
    deserialize_value_admitted, deserialize_value_representative_admitted,
    value_hash_scratch_bytes, value_representative_admitted, value_retained_bytes,
};
