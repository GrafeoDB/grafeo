//! Testing utilities for Grafeo internals.
//!
//! `crash`, `pause`, `statement_failure` and `commit_hook` are feature-gated
//! and compile to no-ops in production builds. `child_process` is always
//! compiled: the storage layer takes its database locks through it,
//! uncontended outside tests.

pub mod child_process;
pub mod commit_hook;
pub mod crash;
pub mod pause;
pub mod statement_failure;
