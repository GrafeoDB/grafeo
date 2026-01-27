//! Graph algorithms for Graphos.
//!
//! This module provides high-performance graph algorithm implementations
//! inspired by rustworkx and GRAPE. Algorithms are designed to work with
//! the Graphos LPG store and can be exposed to Python via bridges.
//!
//! ## Algorithm Categories
//!
//! - [`traversal`] - BFS, DFS with visitor pattern
//! - [`components`] - Connected components, SCC, topological sort
//! - [`shortest_path`] - Dijkstra, A*, Bellman-Ford, Floyd-Warshall
//!
//! ## Usage
//!
//! ```ignore
//! use graphos_adapters::plugins::algorithms::{bfs, dfs, connected_components, dijkstra};
//! use graphos_core::graph::lpg::LpgStore;
//! use graphos_common::types::NodeId;
//!
//! let store = LpgStore::new();
//! // ... populate graph ...
//!
//! // Run BFS from node 0
//! let visited = bfs(&store, NodeId::new(0));
//!
//! // Find connected components
//! let components = connected_components(&store);
//!
//! // Run Dijkstra's shortest path
//! let result = dijkstra(&store, NodeId::new(0), Some("weight"));
//! ```

mod traits;
mod traversal;
mod components;
mod shortest_path;
mod centrality;
mod community;
mod mst;
mod flow;
mod structure;

// Core traits
pub use traits::{
    Control, DistanceMap, GraphAlgorithm, MinScored, ParallelGraphAlgorithm, TraversalEvent,
};

// Traversal algorithms
pub use traversal::{bfs, bfs_layers, bfs_with_visitor, dfs, dfs_all, dfs_with_visitor};

// Component algorithms
pub use components::{
    connected_components, connected_component_count,
    is_dag,
    strongly_connected_components, strongly_connected_component_count,
    topological_sort,
    UnionFind,
};

// Shortest path algorithms
pub use shortest_path::{
    astar, bellman_ford, dijkstra, dijkstra_path, floyd_warshall,
    BellmanFordResult, DijkstraResult, FloydWarshallResult,
};

// Centrality algorithms
pub use centrality::{
    betweenness_centrality, closeness_centrality, degree_centrality,
    degree_centrality_normalized, pagerank, DegreeCentralityResult,
};

// Community detection algorithms
pub use community::{
    label_propagation, louvain, community_count, LouvainResult,
};

// Minimum Spanning Tree algorithms
pub use mst::{kruskal, prim, MstResult};

// Network Flow algorithms
pub use flow::{max_flow, min_cost_max_flow, MaxFlowResult, MinCostFlowResult};

// Structure analysis algorithms
pub use structure::{
    articulation_points, bridges, k_core, kcore_decomposition, KCoreResult,
};

// Algorithm wrappers (for future registry integration)
pub use traversal::{BfsAlgorithm, DfsAlgorithm};
pub use components::{
    ConnectedComponentsAlgorithm,
    StronglyConnectedComponentsAlgorithm,
    TopologicalSortAlgorithm,
};
pub use shortest_path::{
    BellmanFordAlgorithm, DijkstraAlgorithm, FloydWarshallAlgorithm,
};
pub use centrality::{
    BetweennessCentralityAlgorithm, ClosenessCentralityAlgorithm,
    DegreeCentralityAlgorithm, PageRankAlgorithm,
};
pub use community::{
    LabelPropagationAlgorithm, LouvainAlgorithm,
};
pub use mst::{KruskalAlgorithm, PrimAlgorithm};
pub use flow::{MaxFlowAlgorithm, MinCostFlowAlgorithm};
pub use structure::{ArticulationPointsAlgorithm, BridgesAlgorithm, KCoreAlgorithm};
