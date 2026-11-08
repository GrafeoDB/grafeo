//! Disk spilling for queries that exceed available memory.
//!
//! When a sort or aggregation grows too large for RAM, we spill partitions to
//! disk and merge them back later. This lets queries complete even with limited
//! memory, just slower.
//!
//! | Component | Purpose |
//! | --------- | ------- |
//! | [`SpillAccount`] | One open's spilled bytes against its cap and the room it keeps |
//! | [`SpillManager`] | Manages spill file lifecycle with automatic cleanup |
//! | [`SpillFile`] | Read/write individual spill files |
//! | [`ExternalSort`] | External merge sort for big ORDER BY |
//! | [`PartitionedState`] | Hash partitioning for spillable GROUP BY |
//!
//! Async variants (`AsyncSpillManager`, `AsyncSpillFile`) live in
//! `grafeo-engine::execution::spill` (requires tokio runtime).
//!
//! The account is in every build (the vector columns' spill reserves in it
//! too); the rest needs the `spill` feature.

mod account;
#[cfg(feature = "spill")]
mod external_sort;
#[cfg(feature = "spill")]
mod file;
#[cfg(feature = "spill")]
mod manager;
#[cfg(feature = "spill")]
mod partition;
#[cfg(feature = "spill")]
mod serializer;

pub use account::{
    SPILL_PROBE_STEP, SpillAccount, SpillCap, SpillOwner, SpillReservation, SpillRoom,
};
#[cfg(feature = "spill")]
pub use external_sort::{ExternalSort, NullOrder, SortDirection, SortKey};
#[cfg(feature = "spill")]
pub use file::{SpillFile, SpillFileReader};
#[cfg(feature = "spill")]
pub use manager::{SPILL_FILE_STEP, SpillManager};
#[cfg(feature = "spill")]
pub use partition::{DEFAULT_NUM_PARTITIONS, PartitionedState};
#[cfg(feature = "spill")]
pub use serializer::{deserialize_row, deserialize_value, serialize_row, serialize_value};

/// The spill I/O error `error` as an operator's error: a refusal of the
/// spill account (carried by the `io::Error` of a write it refused) becomes
/// the refusal's own error (S001, naming the bound), any other the I/O
/// error.
#[cfg(feature = "spill")]
pub(crate) fn spill_error(error: std::io::Error) -> crate::execution::operators::OperatorError {
    use grafeo_common::utils::error::Error;

    match refusal_of(&error) {
        Some(refusal) => Error::from(refusal.clone()).into(),
        None => Error::Io(error).into(),
    }
}

/// A refusal of the spill account as the `io::Error` a spill write returns:
/// of kind `StorageFull`, carrying the refusal (see [`refusal_of`]).
pub fn refused(refusal: grafeo_common::utils::error::LimitRefusal) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::StorageFull, refusal)
}

/// The refusal of the spill account an `io::Error` from [`refused`] carries.
#[must_use]
pub fn refusal_of(error: &std::io::Error) -> Option<&grafeo_common::utils::error::LimitRefusal> {
    error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<grafeo_common::utils::error::LimitRefusal>())
}
