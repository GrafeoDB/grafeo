//! Async spill file management (moved from grafeo-core to keep grafeo-core free of tokio).
//!
//! The sync spill infrastructure (`SpillManager`, `SpillFile`, `ExternalSort`)
//! remains in `grafeo-core::execution::spill`. These async wrappers live here
//! because they require a tokio runtime.
//!
//! Deprecated since 0.6.1 with the `async-storage` feature, and removed in
//! 0.7.0: the engine spills through the sync types of grafeo-core.

#[cfg(feature = "async-storage")]
#[allow(
    deprecated,
    reason = "the module defines and re-exports its deprecated async spill types until 0.7.0 removes them"
)]
pub mod async_file;
#[cfg(feature = "async-storage")]
#[allow(
    deprecated,
    reason = "the module defines and re-exports its deprecated async spill types until 0.7.0 removes them"
)]
pub mod async_manager;

#[cfg(feature = "async-storage")]
#[allow(
    deprecated,
    reason = "the module defines and re-exports its deprecated async spill types until 0.7.0 removes them"
)]
pub use async_file::{AsyncSpillFile, AsyncSpillFileReader};
#[cfg(feature = "async-storage")]
#[allow(
    deprecated,
    reason = "the module defines and re-exports its deprecated async spill types until 0.7.0 removes them"
)]
pub use async_manager::AsyncSpillManager;
