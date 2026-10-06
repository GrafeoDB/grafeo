//! Testing utilities for Grafeo internals.
//!
//! `crash`, `pause`, `statement_failure` and `commit_hook` are feature-gated
//! and compile to no-ops in production builds. `child_process` is always
//! compiled: the storage layer takes its database locks through it,
//! uncontended outside tests. `chunk_caps` is always compiled too: sections
//! read their chunk caps through it when they are built, unset outside tests.

pub mod child_process;
pub mod chunk_caps;
pub mod commit_hook;
pub mod crash;
pub mod pause;
pub mod statement_failure;
