//! Extend Grafeo with custom functions and algorithms.
//!
//! Plugins let you add new capabilities without modifying Grafeo's core. Register
//! your custom algorithms with [`PluginRegistry`] to make them available to queries.
//!
//! The [`algorithms`] module includes ready-to-use implementations of classic
//! graph algorithms - traversals, shortest paths, centrality measures, and more.

pub mod algorithms;
mod registry;
mod traits;

pub use registry::PluginRegistry;
pub use traits::{Algorithm, AlgorithmResult, ParameterDef, ParameterType, Parameters, Plugin};
