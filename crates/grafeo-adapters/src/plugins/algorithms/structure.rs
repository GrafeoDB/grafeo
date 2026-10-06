//! Structure analysis algorithms: Articulation Points, Bridges, K-Core decomposition.
//!
//! These algorithms identify critical structural elements in graphs.

use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(all(test, feature = "lpg"))]
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::{GraphAlgorithm, impl_algorithm};

// ============================================================================
// Articulation Points (Cut Vertices)
// ============================================================================

/// Finds articulation points (cut vertices) in the graph.
///
/// An articulation point is a vertex whose removal disconnects the graph.
/// Uses Tarjan's algorithm with low-link values.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Set of node IDs that are articulation points.
///
/// # Panics
///
/// Panics if the DFS stack is inconsistent (internal invariant).
///
/// # Complexity
///
/// O(V + E)
pub fn articulation_points(store: &dyn GraphStore) -> FxHashSet<NodeId> {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return FxHashSet::default();
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    // Build undirected adjacency list
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    for (i, &node) in nodes.iter().enumerate() {
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                adj[i].insert(j);
                adj[j].insert(i); // Undirected
            }
        }
    }

    let mut visited = vec![false; n];
    let mut disc = vec![0usize; n]; // Discovery time
    let mut low = vec![0usize; n]; // Low-link value
    let mut parent = vec![None::<usize>; n];
    let mut ap = vec![false; n]; // Is articulation point
    let mut time = 0usize;

    // DFS from each unvisited node (handles disconnected graphs)
    for start in 0..n {
        if visited[start] {
            continue;
        }

        // Iterative DFS using explicit stack
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)]; // (node, neighbor_idx)
        let mut children_count: FxHashMap<usize, usize> = FxHashMap::default();

        while let Some(&(u, idx)) = stack.last() {
            if !visited[u] {
                visited[u] = true;
                disc[u] = time;
                low[u] = time;
                time += 1;
                children_count.insert(u, 0);
            }

            let neighbors: Vec<usize> = adj[u].iter().copied().collect();

            if idx < neighbors.len() {
                let v = neighbors[idx];
                stack.last_mut().expect("DFS: stack non-empty").1 += 1;

                if !visited[v] {
                    parent[v] = Some(u);
                    *children_count.entry(u).or_insert(0) += 1;
                    stack.push((v, 0));
                } else if parent[u] != Some(v) {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                stack.pop();

                if let Some(p) = parent[u] {
                    low[p] = low[p].min(low[u]);

                    // Check articulation point condition
                    if parent[p].is_some() && low[u] >= disc[p] {
                        ap[p] = true;
                    }
                }

                // Root is articulation point if it has more than one child
                if parent[u].is_none() && *children_count.get(&u).unwrap_or(&0) > 1 {
                    ap[u] = true;
                }
            }
        }
    }

    ap.iter()
        .enumerate()
        .filter(|&(_, is_ap)| *is_ap)
        .map(|(idx, _)| idx_to_node[idx])
        .collect()
}

// ============================================================================
// Bridges (Cut Edges)
// ============================================================================

/// Finds bridges (cut edges) in the graph.
///
/// A bridge is an edge whose removal disconnects the graph.
/// Uses Tarjan's algorithm with low-link values.
///
/// # Arguments
///
/// * `store` - The graph store, treated as simple and undirected: edge
///   direction is ignored, parallel edges count as one and self-loops are
///   ignored
///
/// # Returns
///
/// List of bridges as (source, target) pairs, the smaller node id first, in
/// node-id order.
///
/// # Panics
///
/// Panics if the DFS stack is inconsistent (internal invariant).
///
/// # Complexity
///
/// O(V + E), the bridges put in node-id order by counting sort.
pub fn bridges(store: &dyn GraphStore) -> Vec<(NodeId, NodeId)> {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return Vec::new();
    }

    let adj = simple_undirected_adjacency(store, &nodes);

    let mut visited = vec![false; n];
    let mut disc = vec![0usize; n];
    let mut low = vec![0usize; n];
    let mut parent = vec![None::<usize>; n];
    let mut time = 0usize;
    let mut bridge_list: Vec<(usize, usize)> = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }

        // (node, index of its next neighbour to visit)
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];

        while let Some(&(u, idx)) = stack.last() {
            if !visited[u] {
                visited[u] = true;
                disc[u] = time;
                low[u] = time;
                time += 1;
            }

            if let Some(&v) = adj[u].get(idx) {
                stack.last_mut().expect("DFS: stack non-empty").1 += 1;

                if !visited[v] {
                    parent[v] = Some(u);
                    stack.push((v, 0));
                } else if parent[u] != Some(v) {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                stack.pop();

                if let Some(p) = parent[u] {
                    low[p] = low[p].min(low[u]);

                    // Bridge condition: low[u] > disc[p]
                    if low[u] > disc[p] {
                        bridge_list.push((p.min(u), p.max(u)));
                    }
                }
            }
        }
    }

    // `node_ids` is in node-id order, so the index pairs in index order are
    // the bridges in node-id order.
    sort_index_pairs(&mut bridge_list, n);
    bridge_list
        .into_iter()
        .map(|(i, j)| (nodes[i], nodes[j]))
        .collect()
}

/// The simple undirected graph of `nodes` (in the order of `node_ids`) as
/// adjacency lists of indices into `nodes`: edge direction is ignored, each
/// neighbour is listed once however many edges join the two, and self-loops
/// are left out. O(V + E).
fn simple_undirected_adjacency(store: &dyn GraphStore, nodes: &[NodeId]) -> Vec<Vec<usize>> {
    let n = nodes.len();
    let node_to_idx: FxHashMap<NodeId, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, &node) in nodes.iter().enumerate() {
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor)
                && j != i
            {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    // Drop parallel edges in linear time: `last_seen[u]` is the last vertex
    // whose list kept u, so a second u in the same list is a duplicate.
    let mut last_seen = vec![usize::MAX; n];
    for (v, neighbors) in adj.iter_mut().enumerate() {
        neighbors.retain(|&u| {
            let first = last_seen[u] != v;
            last_seen[u] = v;
            first
        });
    }
    adj
}

/// Sorts pairs of indices below `n` by first, then second index, in
/// O(n + pairs) with two stable counting-sort passes (least significant index
/// first), where a comparison sort would take O(pairs log pairs).
fn sort_index_pairs(pairs: &mut Vec<(usize, usize)>, n: usize) {
    fn by_index(
        pairs: &[(usize, usize)],
        n: usize,
        index: impl Fn(&(usize, usize)) -> usize,
    ) -> Vec<(usize, usize)> {
        let mut start = vec![0usize; n + 1];
        for pair in pairs {
            start[index(pair) + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut sorted = vec![(0, 0); pairs.len()];
        for pair in pairs {
            let slot = &mut start[index(pair)];
            sorted[*slot] = *pair;
            *slot += 1;
        }
        sorted
    }
    let by_second = by_index(pairs, n, |&(_, second)| second);
    *pairs = by_index(&by_second, n, |&(first, _)| first);
}

// ============================================================================
// K-Core Decomposition
// ============================================================================

/// Result of k-core decomposition.
#[derive(Debug, Clone)]
pub struct KCoreResult {
    /// Core number for each node.
    pub core_numbers: FxHashMap<NodeId, usize>,
    /// Maximum core number (degeneracy).
    pub max_core: usize,
}

impl KCoreResult {
    /// Returns nodes in the k-core (nodes with core number >= k), in node-id
    /// order.
    pub fn k_core(&self, k: usize) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self
            .core_numbers
            .iter()
            .filter(|&(_, core)| *core >= k)
            .map(|(&node, _)| node)
            .collect();
        nodes.sort_unstable();
        nodes
    }

    /// Returns the k-shell (nodes with core number exactly k), in node-id
    /// order.
    pub fn k_shell(&self, k: usize) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self
            .core_numbers
            .iter()
            .filter(|&(_, core)| *core == k)
            .map(|(&node, _)| node)
            .collect();
        nodes.sort_unstable();
        nodes
    }
}

/// Computes the k-core decomposition of the graph.
///
/// The k-core is the maximal subgraph where every vertex has degree at least k.
/// The core number of a vertex is the largest k such that it belongs to the k-core.
///
/// The graph is treated as simple and undirected: edge direction is ignored, parallel
/// edges between two nodes count once, and self-loops are ignored. The decomposition is
/// unique, so the result does not depend on node or edge order.
///
/// # Arguments
///
/// * `store` - The graph store
///
/// # Returns
///
/// Core numbers for all nodes and the maximum core number.
///
/// # Complexity
///
/// O(V + E), the bin-sort peeling of Batagelj and Zaversnik.
pub fn kcore_decomposition(store: &dyn GraphStore) -> KCoreResult {
    let nodes = store.node_ids();
    let n = nodes.len();

    // Simple undirected adjacency: no self-loops, each neighbour once.
    let adj = simple_undirected_adjacency(store, &nodes);

    // `degree[v]` starts as the degree and ends as the core number of v.
    let mut degree: Vec<usize> = adj.iter().map(Vec::len).collect();
    let max_degree = degree.iter().copied().max().unwrap_or(0);

    // Bin sort: `order` holds the vertices by current degree, `bin_start[d]` is where
    // degree d starts in `order`, `position[v]` is where v sits in `order`.
    let mut bin_start = vec![0usize; max_degree + 1];
    for &d in &degree {
        bin_start[d] += 1;
    }
    let mut start = 0;
    for count in &mut bin_start {
        let size = *count;
        *count = start;
        start += size;
    }
    let mut position = vec![0usize; n];
    let mut order = vec![0usize; n];
    let mut next_free = bin_start.clone();
    for v in 0..n {
        position[v] = next_free[degree[v]];
        order[position[v]] = v;
        next_free[degree[v]] += 1;
    }

    // Peel vertices in order of current degree. Moving a neighbour u from degree d to
    // d - 1 swaps it with the first vertex of bin d and shifts that bin's start by one.
    for i in 0..n {
        let v = order[i];
        for &u in &adj[v] {
            if degree[u] > degree[v] {
                let du = degree[u];
                let first = bin_start[du];
                let w = order[first];
                if u != w {
                    order[position[u]] = w;
                    position[w] = position[u];
                    order[first] = u;
                    position[u] = first;
                }
                bin_start[du] += 1;
                degree[u] -= 1;
            }
        }
    }

    let max_core = degree.iter().copied().max().unwrap_or(0);
    let core_numbers: FxHashMap<NodeId, usize> = nodes
        .iter()
        .zip(&degree)
        .map(|(&node, &core)| (node, core))
        .collect();

    KCoreResult {
        core_numbers,
        max_core,
    }
}

/// Extracts the k-core subgraph (nodes with core number >= k).
pub fn k_core(store: &dyn GraphStore, k: usize) -> Vec<NodeId> {
    let result = kcore_decomposition(store);
    result.k_core(k)
}

// ============================================================================
// K-Truss Decomposition
// ============================================================================

/// Result of k-truss decomposition.
#[derive(Debug, Clone)]
pub struct KTrussResult {
    /// Maps each edge (as ordered node pair) to its truss number.
    /// The truss number is the maximum k for which the edge belongs to the k-truss.
    pub truss_numbers: FxHashMap<(NodeId, NodeId), usize>,
    /// The maximum truss number found in the graph.
    pub max_truss: usize,
}

impl KTrussResult {
    /// Returns edges in the k-truss (edges with truss number >= k), in node-id
    /// order.
    pub fn k_truss(&self, k: usize) -> Vec<(NodeId, NodeId)> {
        let mut edges: Vec<(NodeId, NodeId)> = self
            .truss_numbers
            .iter()
            .filter(|&(_, &truss)| truss >= k)
            .map(|(&edge, _)| edge)
            .collect();
        edges.sort_unstable();
        edges
    }
}

/// Computes the number of triangles containing each edge (edge support).
///
/// For each edge (u, v), the support is the number of common neighbors of u and v.
/// This is a standalone metric useful for edge importance analysis.
///
/// # Complexity
///
/// O(m * d_max) where m is the number of edges and d_max is the maximum degree
pub fn edge_triangle_support(store: &dyn GraphStore) -> FxHashMap<(NodeId, NodeId), u64> {
    let nodes = store.node_ids();
    if nodes.is_empty() {
        return FxHashMap::default();
    }

    let neighbors = build_undirected_neighbors_set(store);
    let mut support: FxHashMap<(NodeId, NodeId), u64> = FxHashMap::default();

    // For each node u, for each pair of neighbors (v, w), if v-w is also
    // an edge, increment support for all three edges of the triangle.
    for (&node, node_neighbors) in &neighbors {
        let nb_list: Vec<NodeId> = node_neighbors.iter().copied().collect();
        for i in 0..nb_list.len() {
            for j in (i + 1)..nb_list.len() {
                let v = nb_list[i];
                let w = nb_list[j];
                if neighbors.get(&v).is_some_and(|s| s.contains(&w)) {
                    // Triangle found: {node, v, w}. Increment support for all three edges.
                    let edges = [
                        ordered_pair(node, v),
                        ordered_pair(node, w),
                        ordered_pair(v, w),
                    ];
                    for edge in edges {
                        *support.entry(edge).or_default() += 1;
                    }
                }
            }
        }
    }

    // Each triangle is found 3 times (once from each vertex), so divide by 3.
    for val in support.values_mut() {
        *val /= 3;
    }

    support
}

/// Full k-truss decomposition: compute the truss number of every edge.
///
/// An edge belongs to the k-truss if it is supported by at least k-2 triangles
/// in every iteration of the peeling process. The truss number is the maximum k
/// for which this holds.
///
/// Uses iterative edge peeling: repeatedly remove edges with the lowest support,
/// recording their truss number, and updating support counts for affected edges.
///
/// # Complexity
///
/// O(m^1.5) total across all peeling iterations
pub fn ktruss_decomposition(store: &dyn GraphStore) -> KTrussResult {
    let nodes = store.node_ids();
    if nodes.is_empty() {
        return KTrussResult {
            truss_numbers: FxHashMap::default(),
            max_truss: 0,
        };
    }

    // Build undirected adjacency using HashSets for fast membership tests.
    let mut neighbors: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();
    for &node in &nodes {
        neighbors.insert(node, FxHashSet::default());
    }
    for &node in &nodes {
        for (nb, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(set) = neighbors.get_mut(&node) {
                set.insert(nb);
            }
            if let Some(set) = neighbors.get_mut(&nb) {
                set.insert(node);
            }
        }
    }

    // Collect all undirected edges.
    let mut live_edges: FxHashSet<(NodeId, NodeId)> = FxHashSet::default();
    for (&u, u_neighbors) in &neighbors {
        for &v in u_neighbors {
            live_edges.insert(ordered_pair(u, v));
        }
    }

    if live_edges.is_empty() {
        return KTrussResult {
            truss_numbers: FxHashMap::default(),
            max_truss: 0,
        };
    }

    // Compute initial edge support (triangle count per edge).
    let mut support: FxHashMap<(NodeId, NodeId), usize> = FxHashMap::default();
    for &(u, v) in &live_edges {
        let u_nb = &neighbors[&u];
        let v_nb = &neighbors[&v];
        let common = u_nb.intersection(v_nb).count();
        support.insert((u, v), common);
    }

    // Iterative peeling.
    let mut truss_numbers: FxHashMap<(NodeId, NodeId), usize> = FxHashMap::default();
    let mut max_truss = 0usize;
    let mut k: usize = 2;

    while !live_edges.is_empty() {
        // Find all edges with support < k - 2.
        let threshold = k.saturating_sub(2);
        let mut to_remove: Vec<(NodeId, NodeId)> = live_edges
            .iter()
            .filter(|e| support.get(*e).copied().unwrap_or(0) < threshold)
            .copied()
            .collect();

        if to_remove.is_empty() {
            // All remaining edges have support >= threshold.
            // These edges have truss number >= k, try next k.
            k += 1;
            // Safety: if k exceeds possible support, everything will be removed next iteration.
            if k > live_edges.len() + 2 {
                // Remaining edges all belong to truss k-1.
                for &edge in &live_edges {
                    truss_numbers.insert(edge, k - 1);
                    max_truss = max_truss.max(k - 1);
                }
                break;
            }
            continue;
        }

        // Remove edges and update support.
        while let Some((u, v)) = to_remove.pop() {
            if !live_edges.contains(&(u, v)) {
                continue;
            }
            live_edges.remove(&(u, v));
            truss_numbers.insert((u, v), k - 1);
            max_truss = max_truss.max(k - 1);

            // Update adjacency.
            if let Some(set) = neighbors.get_mut(&u) {
                set.remove(&v);
            }
            if let Some(set) = neighbors.get_mut(&v) {
                set.remove(&u);
            }

            // For each common neighbor w of (u, v) that still exists:
            // decrement support of edges (u, w) and (v, w).
            let u_nb: Vec<NodeId> = neighbors
                .get(&u)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            let v_nb: FxHashSet<NodeId> = neighbors.get(&v).cloned().unwrap_or_default();

            for w in u_nb {
                if v_nb.contains(&w) {
                    let uw = ordered_pair(u, w);
                    let vw = ordered_pair(v, w);
                    if let Some(s) = support.get_mut(&uw) {
                        *s = s.saturating_sub(1);
                        if *s < threshold && live_edges.contains(&uw) {
                            to_remove.push(uw);
                        }
                    }
                    if let Some(s) = support.get_mut(&vw) {
                        *s = s.saturating_sub(1);
                        if *s < threshold && live_edges.contains(&vw) {
                            to_remove.push(vw);
                        }
                    }
                }
            }

            support.remove(&(u, v));
        }
    }

    KTrussResult {
        truss_numbers,
        max_truss,
    }
}

/// Extracts edges in the k-truss subgraph.
pub fn k_truss(store: &dyn GraphStore, k: usize) -> Vec<(NodeId, NodeId)> {
    let result = ktruss_decomposition(store);
    result.k_truss(k)
}

/// Orders a node pair so the smaller ID comes first (canonical edge representation).
fn ordered_pair(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Builds undirected neighbor sets for all nodes (same as clustering.rs helper).
fn build_undirected_neighbors_set(store: &dyn GraphStore) -> FxHashMap<NodeId, FxHashSet<NodeId>> {
    let nodes = store.node_ids();
    let mut neighbors: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();
    for &node in &nodes {
        neighbors.insert(node, FxHashSet::default());
    }
    for &node in &nodes {
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(set) = neighbors.get_mut(&node) {
                set.insert(neighbor);
            }
            if let Some(set) = neighbors.get_mut(&neighbor) {
                set.insert(node);
            }
        }
    }
    neighbors
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for Articulation Points algorithm.
static ARTICULATION_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn articulation_params() -> &'static [ParameterDef] {
    ARTICULATION_PARAMS.get_or_init(Vec::new)
}

/// Articulation Points algorithm wrapper.
pub struct ArticulationPointsAlgorithm;

impl_algorithm! {
    ArticulationPointsAlgorithm,
    name: "articulation_points",
    description: "Find articulation points (cut vertices) in the graph",
    params: articulation_params,
    execute(store, _params) {
        let points = articulation_points(store);

        let mut result = AlgorithmResult::new(vec!["node_id".to_string()]);

        for node in points {
            // reason: Node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            result.add_row(vec![Value::Int64(node.0 as i64)]);
        }

        result.sort_by_id_columns(1);
        Ok(result)
    }
}

/// Static parameter definitions for Bridges algorithm.
static BRIDGES_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn bridges_params() -> &'static [ParameterDef] {
    BRIDGES_PARAMS.get_or_init(Vec::new)
}

/// Bridges algorithm wrapper.
pub struct BridgesAlgorithm;

impl_algorithm! {
    BridgesAlgorithm,
    name: "bridges",
    description: "Find bridges (cut edges) in the graph",
    params: bridges_params,
    execute(store, _params) {
        let bridge_list = bridges(store);

        let mut result = AlgorithmResult::new(vec!["source".to_string(), "target".to_string()]);

        for (src, dst) in bridge_list {
            // reason: Node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            result.add_row(vec![Value::Int64(src.0 as i64), Value::Int64(dst.0 as i64)]);
        }

        result.sort_by_id_columns(2);
        Ok(result)
    }
}

/// Static parameter definitions for K-Core algorithm.
static KCORE_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn kcore_params() -> &'static [ParameterDef] {
    KCORE_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "k".to_string(),
            description: "Core number threshold (optional, returns decomposition if not set)"
                .to_string(),
            param_type: ParameterType::Integer,
            required: false,
            default: None,
        }]
    })
}

/// K-Core decomposition algorithm wrapper.
pub struct KCoreAlgorithm;

impl GraphAlgorithm for KCoreAlgorithm {
    fn name(&self) -> &str {
        "kcore"
    }

    fn description(&self) -> &str {
        "K-core decomposition of the graph"
    }

    fn parameters(&self) -> &[ParameterDef] {
        kcore_params()
    }

    // reason: node IDs and core numbers are bounded by graph size
    #[allow(clippy::cast_possible_wrap)]
    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let decomposition = kcore_decomposition(store);

        if let Some(k) = params.get_int("k") {
            if k < 0 {
                return Err(grafeo_common::utils::error::Error::InvalidValue(format!(
                    "k-core requires a non-negative k value, got {k}"
                )));
            }
            let k_usize = usize::try_from(k).map_err(|_| {
                grafeo_common::utils::error::Error::InvalidValue(format!(
                    "k-core k value {k} exceeds maximum supported size"
                ))
            })?;
            let k_core_nodes = decomposition.k_core(k_usize);

            let mut result =
                AlgorithmResult::new(vec!["node_id".to_string(), "in_k_core".to_string()]);

            for node in k_core_nodes {
                result.add_row(vec![Value::Int64(node.0 as i64), Value::Bool(true)]);
            }

            result.sort_by_id_columns(1);
            Ok(result)
        } else {
            // Return full decomposition
            let mut result = AlgorithmResult::new(vec![
                "node_id".to_string(),
                "core_number".to_string(),
                "max_core".to_string(),
            ]);

            for (node, core) in decomposition.core_numbers {
                result.add_row(vec![
                    Value::Int64(node.0 as i64),
                    Value::Int64(core as i64),
                    Value::Int64(decomposition.max_core as i64),
                ]);
            }

            result.sort_by_id_columns(1);
            Ok(result)
        }
    }
}

/// Static parameter definitions for K-Truss algorithm.
static KTRUSS_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn ktruss_params() -> &'static [ParameterDef] {
    KTRUSS_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "k".to_string(),
            description: "Truss number threshold (optional, returns decomposition if not set)"
                .to_string(),
            param_type: ParameterType::Integer,
            required: false,
            default: None,
        }]
    })
}

/// K-Truss decomposition algorithm wrapper.
pub struct KTrussAlgorithm;

impl GraphAlgorithm for KTrussAlgorithm {
    fn name(&self) -> &str {
        "ktruss"
    }

    fn description(&self) -> &str {
        "K-truss decomposition: finds dense subgraphs where every edge \
         is supported by at least k-2 triangles"
    }

    fn parameters(&self) -> &[ParameterDef] {
        ktruss_params()
    }

    // reason: node IDs and truss numbers are bounded by graph size
    #[allow(clippy::cast_possible_wrap)]
    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let decomposition = ktruss_decomposition(store);

        if let Some(k) = params.get_int("k") {
            if k < 0 {
                return Err(grafeo_common::utils::error::Error::InvalidValue(format!(
                    "k-truss requires a non-negative k value, got {k}"
                )));
            }
            let k_usize = usize::try_from(k).map_err(|_| {
                grafeo_common::utils::error::Error::InvalidValue(format!(
                    "k-truss k value {k} exceeds maximum supported size"
                ))
            })?;
            let edges = decomposition.k_truss(k_usize);

            let mut result = AlgorithmResult::new(vec!["source".to_string(), "target".to_string()]);

            for (src, dst) in edges {
                result.add_row(vec![Value::Int64(src.0 as i64), Value::Int64(dst.0 as i64)]);
            }

            result.sort_by_id_columns(2);
            Ok(result)
        } else {
            // Return full decomposition
            let mut result = AlgorithmResult::new(vec![
                "source".to_string(),
                "target".to_string(),
                "truss_number".to_string(),
            ]);

            for ((src, dst), truss) in &decomposition.truss_numbers {
                result.add_row(vec![
                    Value::Int64(src.0 as i64),
                    Value::Int64(dst.0 as i64),
                    Value::Int64(*truss as i64),
                ]);
            }

            result.sort_by_id_columns(2);
            Ok(result)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;

    fn create_simple_path() -> LpgStore {
        // Path: 0 - 1 - 2 - 3 (all are articulation points except endpoints)
        let store = LpgStore::new().unwrap();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n3, "EDGE");
        store.create_edge(n3, n2, "EDGE");

        store
    }

    fn create_diamond() -> LpgStore {
        // Diamond: 0 - 1 - 3
        //          |   |
        //          +---2
        // No articulation points in a diamond
        let store = LpgStore::new().unwrap();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        // 0-1, 0-2, 1-3, 2-3
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n1, n3, "EDGE");
        store.create_edge(n3, n1, "EDGE");
        store.create_edge(n2, n3, "EDGE");
        store.create_edge(n3, n2, "EDGE");

        store
    }

    fn create_tree() -> LpgStore {
        // Tree:     0
        //          / \
        //         1   2
        //        /
        //       3
        let store = LpgStore::new().unwrap();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n1, n3, "EDGE");
        store.create_edge(n3, n1, "EDGE");

        store
    }

    #[test]
    fn test_articulation_points_path() {
        let store = create_simple_path();
        let ap = articulation_points(&store);

        // In a path, middle nodes are articulation points
        // 0-1-2-3: nodes 1 and 2 are articulation points
        assert!(ap.len() >= 2);
        assert!(ap.contains(&NodeId::new(1)) || ap.contains(&NodeId::new(2)));
    }

    #[test]
    fn test_articulation_points_diamond() {
        let store = create_diamond();
        let ap = articulation_points(&store);

        // Diamond has no articulation points (it's 2-connected)
        assert!(ap.is_empty());
    }

    #[test]
    fn test_articulation_points_tree() {
        let store = create_tree();
        let ap = articulation_points(&store);

        // In a tree, all non-leaf nodes are articulation points
        // Tree: 0 has children 1, 2. Node 1 has child 3.
        // Articulation points: 0, 1
        assert!(ap.contains(&NodeId::new(0)) || ap.contains(&NodeId::new(1)));
    }

    #[test]
    fn test_articulation_points_empty() {
        let store = LpgStore::new().unwrap();
        let ap = articulation_points(&store);
        assert!(ap.is_empty());
    }

    #[test]
    fn test_bridges_path() {
        let store = create_simple_path();
        let br = bridges(&store);

        // In a path, all edges are bridges
        assert_eq!(br.len(), 3);
    }

    #[test]
    fn test_bridges_diamond() {
        let store = create_diamond();
        let br = bridges(&store);

        // Diamond has no bridges (every edge is part of a cycle)
        assert!(br.is_empty(), "{br:?}");
    }

    #[test]
    fn test_bridges_empty() {
        let store = LpgStore::new().unwrap();
        let br = bridges(&store);
        assert!(br.is_empty(), "{br:?}");
    }

    #[test]
    fn test_kcore_path() {
        let store = create_simple_path();
        let result = kcore_decomposition(&store);

        // Every node of a path is in the 1-core and none is in the 2-core.
        assert_eq!(result.core_numbers.len(), 4);
        assert!(result.core_numbers.values().all(|&core| core == 1));
        assert_eq!(result.max_core, 1);
    }

    #[test]
    fn test_kcore_triangle() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");

        let result = kcore_decomposition(&store);

        // Edges in both directions count once: every node of a triangle has core number 2.
        assert_eq!(core_numbers_in_order(&result, &[n0, n1, n2]), vec![2, 2, 2]);
        assert_eq!(result.max_core, 2);
    }

    #[test]
    fn test_kcore_empty() {
        let store = LpgStore::new().unwrap();
        let result = kcore_decomposition(&store);

        assert!(result.core_numbers.is_empty());
        assert_eq!(result.max_core, 0);
    }

    #[test]
    fn test_kcore_isolated() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);

        let result = kcore_decomposition(&store);

        // Isolated nodes have core number 0
        for (_, &core) in &result.core_numbers {
            assert_eq!(core, 0);
        }
    }

    #[test]
    fn test_k_core_extraction() {
        let store = create_simple_path();
        let result = kcore_decomposition(&store);

        // k_core(0) should return all nodes
        let k0_core = result.k_core(0);
        assert_eq!(k0_core.len(), 4);

        // Higher k-cores have fewer or equal nodes
        let k1_core = result.k_core(1);
        assert!(k1_core.len() <= 4);

        let k2_core = result.k_core(2);
        assert!(k2_core.len() <= k1_core.len());
    }

    #[test]
    fn test_k_shell() {
        let store = create_simple_path();
        let result = kcore_decomposition(&store);

        // Total nodes in all shells should equal total nodes
        let total_in_shells: usize = (0..=result.max_core).map(|k| result.k_shell(k).len()).sum();
        assert_eq!(total_in_shells, 4);
    }

    /// Builds a store with `n` nodes and one directed edge per pair in `edges`.
    fn store_from_edges(n: usize, edges: &[(usize, usize)]) -> (LpgStore, Vec<NodeId>) {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..n).map(|_| store.create_node(&["Node"])).collect();
        for &(u, v) in edges {
            store.create_edge(nodes[u], nodes[v], "EDGE");
        }
        (store, nodes)
    }

    fn core_numbers_in_order(result: &KCoreResult, nodes: &[NodeId]) -> Vec<usize> {
        nodes.iter().map(|node| result.core_numbers[node]).collect()
    }

    #[test]
    fn test_kcore_core_numbers_on_small_shapes() {
        // (node count, one directed edge per pair, expected core number per node)
        let cases: [(usize, Vec<(usize, usize)>, Vec<usize>); 4] = [
            (3, vec![(0, 1), (1, 2), (2, 0)], vec![2, 2, 2]),
            (4, vec![(0, 1), (1, 2), (2, 0), (2, 3)], vec![2, 2, 2, 1]),
            (4, vec![(0, 1), (0, 2), (0, 3)], vec![1, 1, 1, 1]),
            (
                4,
                vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                vec![3, 3, 3, 3],
            ),
        ];
        for (n, edges, expected) in cases {
            let (store, nodes) = store_from_edges(n, &edges);
            for _ in 0..3 {
                let result = kcore_decomposition(&store);
                assert_eq!(
                    core_numbers_in_order(&result, &nodes),
                    expected,
                    "{edges:?}"
                );
                assert_eq!(
                    result.max_core,
                    *expected.iter().max().unwrap(),
                    "{edges:?}"
                );
            }
        }
    }

    #[test]
    fn test_kcore_ignores_self_loops_and_parallel_edges() {
        // A triangle with a self-loop on node 0 and every edge doubled in both directions,
        // plus a node whose only edge is a self-loop.
        let edges = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1),
            (1, 2),
            (2, 1),
            (2, 0),
            (0, 2),
            (3, 3),
        ];
        let (store, nodes) = store_from_edges(4, &edges);
        let result = kcore_decomposition(&store);
        assert_eq!(core_numbers_in_order(&result, &nodes), vec![2, 2, 2, 0]);
        assert_eq!(result.max_core, 2);
    }

    /// Core numbers by definition: the largest k for which the node survives repeatedly
    /// removing every node with fewer than k neighbours (self-loops and parallel edges ignored).
    fn reference_core_numbers(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
        let mut adjacency: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
        for &(u, v) in edges {
            if u != v {
                adjacency[u].insert(v);
                adjacency[v].insert(u);
            }
        }
        let mut core = vec![0; n];
        for k in 1..=n {
            let mut alive = vec![true; n];
            loop {
                let doomed: Vec<usize> = (0..n)
                    .filter(|&v| alive[v] && adjacency[v].iter().filter(|&&u| alive[u]).count() < k)
                    .collect();
                if doomed.is_empty() {
                    break;
                }
                for v in doomed {
                    alive[v] = false;
                }
            }
            if !alive.contains(&true) {
                break;
            }
            for v in (0..n).filter(|&v| alive[v]) {
                core[v] = k;
            }
        }
        core
    }

    #[test]
    fn test_kcore_matches_reference_on_random_graphs() {
        // xorshift64: a fixed seed keeps the graphs the same on every run.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            usize::try_from(state % bound as u64).unwrap()
        };
        for case in 0..200 {
            let n = 1 + next(40);
            let edge_count = next(n * 4 + 1);
            let edges: Vec<(usize, usize)> = (0..edge_count).map(|_| (next(n), next(n))).collect();
            let (store, nodes) = store_from_edges(n, &edges);
            let result = kcore_decomposition(&store);
            let expected = reference_core_numbers(n, &edges);
            assert_eq!(
                core_numbers_in_order(&result, &nodes),
                expected,
                "case {case}: n={n} edges={edges:?}"
            );
            assert_eq!(result.max_core, expected.iter().copied().max().unwrap_or(0));
        }
    }

    // ---- K-Truss tests ----

    fn create_complete(n: usize) -> LpgStore {
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..n).map(|_| store.create_node(&["Node"])).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                store.create_edge(nodes[i], nodes[j], "EDGE");
                store.create_edge(nodes[j], nodes[i], "EDGE");
            }
        }
        store
    }

    #[test]
    fn test_ktruss_empty_graph() {
        let store = LpgStore::new().unwrap();
        let result = ktruss_decomposition(&store);
        assert!(result.truss_numbers.is_empty());
        assert_eq!(result.max_truss, 0);
    }

    #[test]
    fn test_ktruss_path_graph() {
        let store = create_simple_path();
        let result = ktruss_decomposition(&store);

        // Path has no triangles, so all edges have truss number 2.
        for (_, &truss) in &result.truss_numbers {
            assert_eq!(truss, 2, "Path edges should have truss number 2");
        }
    }

    #[test]
    fn test_ktruss_triangle() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");

        let result = ktruss_decomposition(&store);

        // Triangle (K_3): each edge has support 1, so truss number = 3.
        assert_eq!(result.max_truss, 3);
        for (_, &truss) in &result.truss_numbers {
            assert_eq!(truss, 3);
        }
    }

    #[test]
    fn test_ktruss_complete_k4() {
        let store = create_complete(4);
        let result = ktruss_decomposition(&store);

        // K_4: each edge has support 2 (2 common neighbors), truss number = 4.
        assert_eq!(result.max_truss, 4);
        for (_, &truss) in &result.truss_numbers {
            assert_eq!(truss, 4);
        }
    }

    #[test]
    fn test_ktruss_complete_k5() {
        let store = create_complete(5);
        let result = ktruss_decomposition(&store);

        // K_5: each edge has 3 common neighbors, truss number = 5.
        assert_eq!(result.max_truss, 5);
        for (_, &truss) in &result.truss_numbers {
            assert_eq!(truss, 5);
        }
    }

    #[test]
    fn test_ktruss_extraction() {
        let store = create_complete(4);
        let edges_4 = k_truss(&store, 4);
        // All 6 edges of K_4 should be in 4-truss
        assert_eq!(edges_4.len(), 6);

        let edges_5 = k_truss(&store, 5);
        // No edges should be in 5-truss for K_4
        assert!(edges_5.is_empty(), "{edges_5:?}");
    }

    #[test]
    fn test_edge_triangle_support() {
        let store = create_complete(4);
        let support = edge_triangle_support(&store);

        // K_4: each edge is in 2 triangles
        for (_, &count) in &support {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_ktruss_algorithm_wrapper() {
        let store = create_complete(4);
        let algo = KTrussAlgorithm;

        assert_eq!(algo.name(), "ktruss");

        // Full decomposition
        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 3); // source, target, truss_number
        assert_eq!(result.row_count(), 6); // 6 edges in K_4

        // Extract k=4 truss
        let mut params = Parameters::new();
        params.set_int("k", 4);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 2); // source, target
        assert_eq!(result.row_count(), 6);
    }

    // ---- KCoreAlgorithm wrapper tests ----

    #[test]
    fn test_kcore_algorithm_wrapper_full_decomposition() {
        let store = create_complete(4);
        let algo = KCoreAlgorithm;

        assert_eq!(algo.name(), "kcore");

        // Full decomposition (no k parameter)
        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 3); // node_id, core_number, max_core
        assert_eq!(result.row_count(), 4); // 4 nodes in K_4
    }

    #[test]
    fn test_kcore_algorithm_wrapper_with_k() {
        let store = create_complete(4);
        let algo = KCoreAlgorithm;

        // Extract k=2 core: with peeling, only some nodes retain core >= 2
        let mut params = Parameters::new();
        params.set_int("k", 2);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 2); // node_id, in_k_core
        // At least some nodes should be in the 2-core
        assert!(result.row_count() > 0);
        assert!(result.row_count() <= 4);
    }

    #[test]
    fn test_kcore_negative_k() {
        // Negative k should be rejected with a validation error
        let store = create_complete(4);
        let algo = KCoreAlgorithm;

        let mut params = Parameters::new();
        params.set_int("k", -1);
        let result = algo.execute(&store, &params);
        assert!(result.is_err(), "negative k should return an error");
    }

    #[test]
    fn test_kcore_valid_execution() {
        // Alix, Gus, and Vincent form a triangle
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);

        // Bidirectional triangle
        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, alix, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");
        store.create_edge(vincent, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");
        store.create_edge(vincent, alix, "KNOWS");

        let algo = KCoreAlgorithm;

        // Full decomposition: returns all 3 nodes with core numbers
        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 3); // node_id, core_number, max_core
        assert_eq!(result.row_count(), 3); // all 3 nodes decomposed

        // With k=1: at least some nodes should have core >= 1
        let mut params = Parameters::new();
        params.set_int("k", 1);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 2); // node_id, in_k_core
        assert!(result.row_count() > 0);
    }

    #[test]
    fn test_ktruss_negative_k() {
        // Negative k should be rejected with a validation error
        let store = create_complete(4);
        let algo = KTrussAlgorithm;

        let mut params = Parameters::new();
        params.set_int("k", -1);
        let result = algo.execute(&store, &params);
        assert!(result.is_err(), "negative k should return an error");
    }

    #[test]
    fn test_ktruss_valid_execution() {
        // Triangle between Alix, Gus, Vincent: 3-truss
        let store = LpgStore::new().unwrap();
        let alix = store.create_node(&["Person"]);
        let gus = store.create_node(&["Person"]);
        let vincent = store.create_node(&["Person"]);

        store.create_edge(alix, gus, "KNOWS");
        store.create_edge(gus, alix, "KNOWS");
        store.create_edge(gus, vincent, "KNOWS");
        store.create_edge(vincent, gus, "KNOWS");
        store.create_edge(alix, vincent, "KNOWS");
        store.create_edge(vincent, alix, "KNOWS");

        let algo = KTrussAlgorithm;
        let mut params = Parameters::new();
        params.set_int("k", 3);
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns.len(), 2); // source, target
        assert_eq!(result.row_count(), 3); // 3 edges in 3-truss
    }

    // ---- Cross-model: RDF adapter produces same results as LPG ----

    #[cfg(feature = "triple-store")]
    #[test]
    fn test_ktruss_rdf_matches_lpg() {
        use grafeo_core::graph::rdf::{RdfGraphStoreAdapter, RdfStore, Term, Triple};

        // Build K_4 in RDF
        let rdf = RdfStore::new();
        let names = ["a", "b", "c", "d"];
        let pred = Term::iri("http://example.org/knows");
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let u = Term::iri(format!("http://example.org/{}", names[i]));
                let v = Term::iri(format!("http://example.org/{}", names[j]));
                rdf.insert(Triple::new(u.clone(), pred.clone(), v.clone()));
                rdf.insert(Triple::new(v, pred.clone(), u));
            }
        }

        let adapter = RdfGraphStoreAdapter::new(&rdf);
        let rdf_result = ktruss_decomposition(&adapter);

        // K_4 via LPG
        let lpg = create_complete(4);
        let lpg_result = ktruss_decomposition(&lpg);

        assert_eq!(
            rdf_result.max_truss, lpg_result.max_truss,
            "RDF ({}) and LPG ({}) max_truss must match for K_4",
            rdf_result.max_truss, lpg_result.max_truss
        );
        assert_eq!(rdf_result.max_truss, 4);
        assert_eq!(
            rdf_result.truss_numbers.len(),
            lpg_result.truss_numbers.len()
        );
    }

    #[cfg(feature = "triple-store")]
    #[test]
    fn test_kcore_rdf_matches_lpg() {
        use grafeo_core::graph::rdf::{RdfGraphStoreAdapter, RdfStore, Term, Triple};

        // Build K_4 in RDF
        let rdf = RdfStore::new();
        let names = ["a", "b", "c", "d"];
        let pred = Term::iri("http://example.org/knows");
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let u = Term::iri(format!("http://example.org/{}", names[i]));
                let v = Term::iri(format!("http://example.org/{}", names[j]));
                rdf.insert(Triple::new(u.clone(), pred.clone(), v.clone()));
                rdf.insert(Triple::new(v, pred.clone(), u));
            }
        }

        let adapter = RdfGraphStoreAdapter::new(&rdf);
        let rdf_result = kcore_decomposition(&adapter);

        let lpg = create_complete(4);
        let lpg_result = kcore_decomposition(&lpg);

        // Every node of K_4 has core number 3, in both models.
        assert_eq!(rdf_result.core_numbers.len(), 4);
        assert!(rdf_result.core_numbers.values().all(|&core| core == 3));
        assert!(lpg_result.core_numbers.values().all(|&core| core == 3));
        assert_eq!(rdf_result.max_core, 3);
        assert_eq!(lpg_result.max_core, 3);
    }

    /// The bridges of `store` by definition: the simple undirected edges whose
    /// removal disconnects their two ends, in node-id order.
    fn bridges_by_definition(store: &LpgStore) -> Vec<(NodeId, NodeId)> {
        let mut edges: std::collections::BTreeSet<(NodeId, NodeId)> =
            std::collections::BTreeSet::new();
        for node in store.node_ids() {
            for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
                if neighbor != node {
                    edges.insert((node.min(neighbor), node.max(neighbor)));
                }
            }
        }
        let connected_without = |removed: (NodeId, NodeId)| {
            let mut seen = std::collections::BTreeSet::from([removed.0]);
            let mut stack = vec![removed.0];
            while let Some(u) = stack.pop() {
                for &(a, b) in &edges {
                    if (a, b) == removed {
                        continue;
                    }
                    let next = if a == u {
                        b
                    } else if b == u {
                        a
                    } else {
                        continue;
                    };
                    if seen.insert(next) {
                        stack.push(next);
                    }
                }
            }
            seen.contains(&removed.1)
        };
        edges
            .iter()
            .copied()
            .filter(|&edge| !connected_without(edge))
            .collect()
    }

    /// Bridges are the edges whose removal disconnects the graph, in node-id
    /// order, on graphs with parallel edges (which count once), self-loops
    /// and several components.
    #[test]
    fn bridges_match_the_definition_in_node_id_order() {
        let mut state: u64 = 0x0003_0019_0088;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        for round in 0..19 {
            let store = LpgStore::new().unwrap();
            let nodes: Vec<NodeId> = (0..40).map(|_| store.create_node(&["Node"])).collect();
            for _ in 0..(30 + round * 2) {
                let a = nodes[usize::try_from(next(40)).unwrap()];
                let b = nodes[usize::try_from(next(40)).unwrap()];
                store.create_edge(a, b, "EDGE");
                if next(19) == 0 {
                    store.create_edge(b, a, "EDGE");
                }
            }
            assert_eq!(
                bridges(&store),
                bridges_by_definition(&store),
                "round {round}"
            );
        }
    }

    /// Bridges stay O(V + E) at a node of high degree: a star of 50,000
    /// leaves (every edge a bridge) takes milliseconds. A search that copied a
    /// node's neighbours at every step took O(degree squared) there.
    #[test]
    fn bridges_of_a_high_degree_star_take_milliseconds() {
        let store = LpgStore::new().unwrap();
        let hub = store.create_node(&["Node"]);
        for leaf in 0..50_000 {
            // With `tiered-storage` (on under the workspace's --all-features),
            // the records of one epoch must fit its arena chunk.
            if leaf % 1_000 == 0 {
                store.new_epoch();
            }
            let leaf = store.create_node(&["Node"]);
            store.create_edge(hub, leaf, "EDGE");
        }
        let started = std::time::Instant::now();
        let found = bridges(&store);
        let elapsed = started.elapsed();
        assert_eq!(found.len(), 50_000);
        assert!(found.windows(2).all(|w| w[0] < w[1]), "in node-id order");
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "bridges of a 50,000-leaf star took {elapsed:?}"
        );
    }
}
