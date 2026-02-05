//! Connect Grafeo to Python's graph analysis ecosystem.
//!
//! | Adapter | Python library | When to use |
//! | ------- | -------------- | ----------- |
//! | [`PyNetworkXAdapter`] | [NetworkX](https://networkx.org/) | Visualization, graph algorithms |
//! | [`PySolvORAdapter`] | [solvOR](https://pypi.org/project/solvor/) | Operations Research problems |
//! | [`PyAlgorithms`] | (native) | Best performance, no dependencies |

pub mod algorithms;
pub mod networkx;
pub mod solvor;

pub use algorithms::PyAlgorithms;
pub use networkx::PyNetworkXAdapter;
pub use solvor::PySolvORAdapter;
