//! The integration tests of `grafeo-storage`, built as one test binary: each file
//! in this directory is a module here, so the crate is linked into one binary, not
//! one per file.
//!
//! ```bash
//! cargo test -p grafeo-storage --all-features --test main                  # all of them
//! cargo test -p grafeo-storage --all-features --test main some_file::      # one file
//! ```
//!
//! A new test file needs its `mod` line here: Cargo does not look for test files in
//! this crate. A file that needs a process of its own, or a build with few features,
//! is a `[[test]]` entry in `Cargo.toml` instead, with a comment saying why.

mod golden_grafeo_file;
mod golden_wal_frames;
mod offsets_past_4_gib;
mod wal_v2_crash;
mod wal_v2_guarantees;
mod wal_v2_open_questions;
