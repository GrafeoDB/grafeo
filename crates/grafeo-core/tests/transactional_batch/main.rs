//! Storage-level transactional batch node/edge creation tests (W1-GRAFEO-BATCH).
//!
//! ```bash
//! cargo test -p grafeo-core --test transactional_batch --features lpg
//! ```

#![cfg(feature = "lpg")]

mod common;
mod create_commit;
mod rollback;
