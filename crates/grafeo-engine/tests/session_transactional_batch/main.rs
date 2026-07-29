//! Session-level transactional batch mutation tests (W1-GRAFEO-BATCH).
//!
//! ```bash
//! cargo test -p grafeo-engine --test session_transactional_batch --features 'lpg,wal,vector-index,cdc'
//! ```

#![allow(missing_docs)]

mod commit_visibility;
mod common;
mod rollback;
