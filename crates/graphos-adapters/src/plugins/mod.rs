//! Plugin system for Graphos.
//!
//! This module provides the plugin infrastructure and bridges to
//! external libraries.
//!
//! ## Modules
//!
//! - [`algorithms`] - Graph algorithms (BFS, DFS, components, centrality, etc.)

mod registry;
mod traits;
pub mod algorithms;

pub use registry::PluginRegistry;
pub use traits::{
    Algorithm, AlgorithmResult, ParameterDef, ParameterType, Parameters, Plugin,
};
