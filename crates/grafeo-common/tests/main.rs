//! The integration tests of `grafeo-common`, built as one test binary: each file
//! in this directory is a module here, so the crate is linked into one binary, not
//! one per file.
//!
//! ```bash
//! cargo test -p grafeo-common --all-features --test main                  # all of them
//! cargo test -p grafeo-common --all-features --test main some_file::      # one file
//! ```
//!
//! A new test file needs its `mod` line here: Cargo does not look for test files in
//! this crate. A file that needs a process of its own, or a build with few features,
//! is a `[[test]]` entry in `Cargo.toml` instead, with a comment saying why.

mod decimal_reference;
mod memory_grants;
