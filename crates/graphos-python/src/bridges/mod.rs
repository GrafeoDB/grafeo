//! Python bridges for external graph libraries.
//!
//! This module provides adapters that allow Graphos to work with
//! common Python graph analysis libraries like NetworkX and solvOR.

pub mod algorithms;
pub mod networkx;
pub mod solvor;

pub use algorithms::PyAlgorithms;
pub use networkx::PyNetworkXAdapter;
pub use solvor::PySolvORAdapter;
