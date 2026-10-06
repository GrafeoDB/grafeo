//! Helpers shared by integration tests, included with `mod common;`.
//!
//! A file that includes this module compiles all of it, and items it does not
//! use warn as dead code. When a helper is added that not every includer uses,
//! include the submodules one by one with `#[path = "common/<name>.rs"]`.

pub mod replay;
