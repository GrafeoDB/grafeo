//! Write-Ahead Log (WAL) for durability.

mod log;
mod record;
mod recovery;

pub use log::WalManager;
pub use record::WalRecord;
pub use recovery::WalRecovery;
