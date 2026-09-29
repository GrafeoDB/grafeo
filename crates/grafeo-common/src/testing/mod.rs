//! Testing utilities for Grafeo internals.
//!
//! `crash` and `statement_failure` are feature-gated and compile to no-ops in
//! production builds. `child_process` is always compiled: the storage layer
//! takes its database locks through it, uncontended outside tests.

pub mod child_process;
pub mod crash;
pub mod statement_failure;
