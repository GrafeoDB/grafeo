//! Classic graph algorithms - traversals, paths, centrality, communities.
//!
//! Everything you'd expect from a graph analytics library, designed to work
//! seamlessly with Grafeo's LPG store. All algorithms are available from Python too.
//!
//! | Category | Algorithms |
//! | -------- | ---------- |
//! | Traversal | BFS, DFS with visitor pattern |
//! | Components | Connected, strongly connected, topological sort |
//! | Shortest paths | Dijkstra, A*, Bellman-Ford, Floyd-Warshall |
//! | Centrality | PageRank, betweenness, closeness, degree |
//! | Community | Louvain, label propagation |
//! | Structure | K-core, bridges, articulation points |
//!
//! ## Usage
//!
//! ```no_run
//! use grafeo_adapters::plugins::algorithms::{bfs, connected_components, dijkstra};
//! use grafeo_core::graph::lpg::LpgStore;
//! use grafeo_common::types::NodeId;
//!
//! let store = LpgStore::new().unwrap();
//! let n0 = store.create_node(&["Node"]);
//! let n1 = store.create_node(&["Node"]);
//! store.create_edge(n0, n1, "CONNECTS");
//!
//! // Run BFS from the first node
//! let visited = bfs(&store, n0);
//!
//! // Find connected components
//! let components = connected_components(&store);
//!
//! // Run Dijkstra's shortest path
//! let result = dijkstra(&store, n0, Some("weight"));
//! ```

mod centrality;
mod clustering;
mod community;
mod components;
mod flow;
mod isomorphism;
mod key_order;
pub mod metrics;
mod mst;
mod shortest_path;
mod structure;
mod traits;
mod traversal;

// Core traits
pub use traits::{
    Control, DistanceMap, GraphAlgorithm, MinScored, ParallelGraphAlgorithm, TraversalEvent,
};

// Traversal algorithms
pub use traversal::{bfs, bfs_layers, bfs_with_visitor, dfs, dfs_all, dfs_with_visitor};

// Component algorithms
pub use components::{
    UnionFind, connected_component_count, connected_components, is_dag,
    strongly_connected_component_count, strongly_connected_components, topological_sort,
};

// Shortest path algorithms
pub use shortest_path::{
    BellmanFordResult, DijkstraResult, FloydWarshallResult, astar, bellman_ford, dijkstra,
    dijkstra_path, floyd_warshall,
};

// Centrality algorithms
pub use centrality::{
    DegreeCentralityResult, betweenness_centrality, closeness_centrality, degree_centrality,
    degree_centrality_normalized, pagerank, pagerank_in_order,
};

// Node order by a key (#566 `key=`)
pub use key_order::{KeyOrderError, order_by_key};

// Clustering algorithms
pub use clustering::{
    ClusteringCoefficientResult, clustering_coefficient, global_clustering_coefficient,
    local_clustering_coefficient, total_triangles, triangle_count,
};
#[cfg(feature = "parallel")]
pub use clustering::{clustering_coefficient_parallel, total_triangles_parallel};

// Community detection algorithms
pub use community::{
    LouvainResult, StochasticBlockPartitionResult, community_count, label_propagation,
    label_propagation_in_order, louvain, louvain_in_order, stochastic_block_partition,
    stochastic_block_partition_incremental,
};

// Minimum Spanning Tree algorithms
pub use mst::{MstResult, kruskal, prim};

// Subgraph isomorphism
pub use isomorphism::{
    subgraph_isomorphism, subgraph_isomorphism_count, subgraph_isomorphism_count_from_edges,
};

// Network Flow algorithms
pub use flow::{MaxFlowResult, MinCostFlowResult, max_flow, min_cost_max_flow};

// Structure analysis algorithms
pub use structure::{
    KCoreResult, KTrussResult, articulation_points, bridges, edge_triangle_support, k_core,
    k_truss, kcore_decomposition, ktruss_decomposition,
};

// Algorithm wrappers (for future registry integration)
pub use centrality::{
    BetweennessCentralityAlgorithm, ClosenessCentralityAlgorithm, DegreeCentralityAlgorithm,
    PageRankAlgorithm,
};
pub use clustering::{ClusteringCoefficientAlgorithm, TotalTrianglesAlgorithm};
pub use community::{
    LabelPropagationAlgorithm, LouvainAlgorithm, StochasticBlockPartitionAlgorithm,
};
pub use components::{
    ConnectedComponentsAlgorithm, StronglyConnectedComponentsAlgorithm, TopologicalSortAlgorithm,
};
pub use flow::{MaxFlowAlgorithm, MinCostFlowAlgorithm};
pub use isomorphism::SubgraphIsomorphismAlgorithm;
pub use mst::{KruskalAlgorithm, PrimAlgorithm};
pub use shortest_path::{
    BellmanFordAlgorithm, DijkstraAlgorithm, FloydWarshallAlgorithm, SsspAlgorithm,
};
pub use structure::{
    ArticulationPointsAlgorithm, BridgesAlgorithm, KCoreAlgorithm, KTrussAlgorithm,
};
pub use traversal::{BfsAlgorithm, DfsAlgorithm};
