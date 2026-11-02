//! Helpers shared by integration tests: `tests/main.rs` declares this module,
//! and a test file takes it with `use crate::common;`.
//!
//! The separate test binaries (the `[[test]]` entries in `Cargo.toml`) do not
//! have `crate::common`: they include a helper with
//! `#[path = "common/<name>.rs"]`.

pub mod replay;
#[cfg(all(feature = "testing-statement-injection", feature = "gql"))]
pub mod started;
