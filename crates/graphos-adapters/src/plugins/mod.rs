//! Plugin system for Graphos.
//!
//! This module provides the plugin infrastructure and bridges to
//! external libraries.

mod registry;
mod traits;

pub use registry::PluginRegistry;
pub use traits::{Algorithm, Plugin};
