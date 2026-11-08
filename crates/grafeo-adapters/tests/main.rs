//! The integration tests of `grafeo-adapters`, built as one test binary: each file
//! in this directory is a module here, so the crate is linked into one binary, not
//! one per file.
//!
//! ```bash
//! cargo test -p grafeo-adapters --all-features --test main                  # all of them
//! cargo test -p grafeo-adapters --all-features --test main some_file::      # one file
//! ```
//!
//! A new test file needs its `mod` line here: Cargo does not look for test files in
//! this crate. A file that needs a process of its own, or a build with few features,
//! is a `[[test]]` entry in `Cargo.toml` instead, with a comment saying why.

mod algorithm_determinism;
mod algorithm_key_order;
mod algorithms_read_visible_graph;
mod gql_fuzz_test;
mod pattern_edges_rules;
mod sparql_parser_tests;
mod sparql_w3c_parser_tests;
