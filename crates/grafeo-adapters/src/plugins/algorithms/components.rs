//! Graph component algorithms.
//!
//! This module provides algorithms for finding connected components,
//! strongly connected components, and topological ordering.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(test)]
#[cfg(test)]
use grafeo_core::graph::rowgroup::RowGroupStore;

use super::super::{AlgorithmResult, ParameterDef, Prepared};
use super::traits::{ComponentResultBuilder, GraphAlgorithm, Ticker, impl_algorithm, int_value};

// ============================================================================
// Union-Find Data Structure
// ============================================================================

/// Union-Find (Disjoint Set Union) with path compression and union by rank.
///
/// This is the optimal data structure for incremental connectivity queries,
/// providing nearly O(1) amortized operations.
pub struct UnionFind {
    /// Parent pointers (self-loop means root)
    parent: Vec<usize>,
    /// Rank for union by rank
    rank: Vec<usize>,
}

impl UnionFind {
    /// Creates a new Union-Find with `n` elements, each in its own set.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Finds the representative (root) of the set containing `x`.
    ///
    /// Uses path compression for amortized O(α(n)) complexity.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    /// Unions the sets containing `x` and `y`.
    ///
    /// Returns `true` if they were in different sets (union performed).
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        // Union by rank
        match self.rank[root_x].cmp(&self.rank[root_y]) {
            std::cmp::Ordering::Less => {
                self.parent[root_x] = root_y;
            }
            std::cmp::Ordering::Greater => {
                self.parent[root_y] = root_x;
            }
            std::cmp::Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }

        true
    }

    /// Returns `true` if `x` and `y` are in the same set.
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

// ============================================================================
// Connected Components (Undirected/Weakly Connected)
// ============================================================================

/// Finds connected components in an undirected graph (or weakly connected
/// components in a directed graph).
///
/// Uses Union-Find for optimal performance.
///
/// # Returns
///
/// A map from node ID to component ID.
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn connected_components(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
) -> grafeo_common::utils::error::Result<FxHashMap<NodeId, u64>> {
    let graph = super::traits::GraphView::new(store, view);
    let found = connected_components_on(&graph);
    graph.finish(found)?
}

/// [`connected_components`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn connected_components_on(
    store: &super::traits::GraphView<'_>,
) -> grafeo_common::utils::error::Result<FxHashMap<NodeId, u64>> {
    let mut ticker = Ticker::new();
    let node_ids = store.node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    // Map NodeId -> index
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &node) in node_ids.iter().enumerate() {
        node_to_idx.insert(node, idx);
    }

    let mut uf = UnionFind::new(n);

    // Process all edges (treating graph as undirected)
    for &node in &node_ids {
        let idx = node_to_idx[&node];

        // Outgoing edges
        for (neighbor, _) in store.edges(node, Direction::Outgoing) {
            ticker.tick()?;
            if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                uf.union(idx, neighbor_idx);
            }
        }

        // Incoming edges (for weakly connected)
        for (neighbor, _) in store.edges(node, Direction::Incoming) {
            ticker.tick()?;
            if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                uf.union(idx, neighbor_idx);
            }
        }
    }

    // Build result: map each node to its component
    let mut root_to_component: FxHashMap<usize, u64> = FxHashMap::default();
    let mut next_component = 0u64;

    let mut result: FxHashMap<NodeId, u64> = FxHashMap::default();
    for (idx, &node) in node_ids.iter().enumerate() {
        let root = uf.find(idx);
        let component_id = *root_to_component.entry(root).or_insert_with(|| {
            let id = next_component;
            next_component += 1;
            id
        });
        result.insert(node, component_id);
    }

    Ok(result)
}

/// Returns the number of connected components.
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn connected_component_count(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
) -> grafeo_common::utils::error::Result<usize> {
    let graph = super::traits::GraphView::new(store, view);
    let found = connected_component_count_on(&graph);
    graph.finish(found)?
}

/// [`connected_component_count`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn connected_component_count_on(
    store: &super::traits::GraphView<'_>,
) -> grafeo_common::utils::error::Result<usize> {
    let components = connected_components_on(store)?;
    let unique: FxHashSet<u64> = components.values().copied().collect();
    Ok(unique.len())
}

// ============================================================================
// Strongly Connected Components (Tarjan's Algorithm)
// ============================================================================

/// Tarjan's algorithm state for a single node.
struct TarjanState {
    index: usize,
    low_link: usize,
    on_stack: bool,
}

/// Finds strongly connected components in a directed graph using Tarjan's algorithm.
///
/// # Returns
///
/// A map from node ID to SCC ID.
///
/// # Panics
///
/// Panics if the internal DFS state is inconsistent (should not happen with a valid `GraphStore`).
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn strongly_connected_components(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
) -> grafeo_common::utils::error::Result<FxHashMap<NodeId, u64>> {
    let graph = super::traits::GraphView::new(store, view);
    let found = strongly_connected_components_on(&graph);
    graph.finish(found)?
}

/// [`strongly_connected_components`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn strongly_connected_components_on(
    store: &super::traits::GraphView<'_>,
) -> grafeo_common::utils::error::Result<FxHashMap<NodeId, u64>> {
    let mut ticker = Ticker::new();
    let node_ids = store.node_ids();

    if node_ids.is_empty() {
        return Ok(FxHashMap::default());
    }

    // State for Tarjan's algorithm
    let mut state: FxHashMap<NodeId, TarjanState> = FxHashMap::default();
    let mut stack: Vec<NodeId> = Vec::new();
    let mut index = 0usize;
    let mut scc_id = 0u64;
    let mut result: FxHashMap<NodeId, u64> = FxHashMap::default();

    // We need to visit all nodes, starting DFS from unvisited ones
    for &start in &node_ids {
        if state.contains_key(&start) {
            continue;
        }

        // Iterative Tarjan's using explicit stack
        // Each entry: (node, neighbor_iter_state, is_first_visit)
        let mut dfs_stack: Vec<(NodeId, Vec<NodeId>, usize, bool)> = Vec::new();

        // Initialize start node
        state.insert(
            start,
            TarjanState {
                index,
                low_link: index,
                on_stack: true,
            },
        );
        index += 1;
        stack.push(start);

        let neighbors: Vec<NodeId> = store
            .edges(start, Direction::Outgoing)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        dfs_stack.push((start, neighbors, 0, true));

        while let Some((node, neighbors, neighbor_idx, _first_visit)) = dfs_stack.last_mut() {
            ticker.tick()?;
            let node = *node;

            if *neighbor_idx >= neighbors.len() {
                // Done with all neighbors
                dfs_stack.pop();

                // Check if this is an SCC root
                let node_state = state.get(&node).expect("Tarjan: node visited");
                if node_state.low_link == node_state.index {
                    // Pop the SCC from stack
                    loop {
                        let w = stack.pop().expect("Tarjan: stack contains SCC members");
                        state
                            .get_mut(&w)
                            .expect("Tarjan: node on stack has state")
                            .on_stack = false;
                        result.insert(w, scc_id);
                        if w == node {
                            break;
                        }
                    }
                    scc_id += 1;
                }

                // Update parent's low_link
                if let Some((parent, _, _, _)) = dfs_stack.last() {
                    let node_low = state.get(&node).expect("Tarjan: node visited").low_link;
                    let parent_state = state.get_mut(parent).expect("Tarjan: parent visited");
                    if node_low < parent_state.low_link {
                        parent_state.low_link = node_low;
                    }
                }

                continue;
            }

            let neighbor = neighbors[*neighbor_idx];
            *neighbor_idx += 1;

            if let Some(neighbor_state) = state.get(&neighbor) {
                // Already visited
                if neighbor_state.on_stack {
                    // Back edge: update low_link
                    // Extract index before mutable borrow
                    let neighbor_index = neighbor_state.index;
                    let node_state = state.get_mut(&node).expect("Tarjan: node visited");
                    if neighbor_index < node_state.low_link {
                        node_state.low_link = neighbor_index;
                    }
                }
            } else {
                // Unvisited: recurse
                state.insert(
                    neighbor,
                    TarjanState {
                        index,
                        low_link: index,
                        on_stack: true,
                    },
                );
                index += 1;
                stack.push(neighbor);

                let neighbor_neighbors: Vec<NodeId> = store
                    .edges(neighbor, Direction::Outgoing)
                    .into_iter()
                    .map(|(n, _)| n)
                    .collect();
                dfs_stack.push((neighbor, neighbor_neighbors, 0, true));
            }
        }
    }

    Ok(result)
}

/// Returns the number of strongly connected components.
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn strongly_connected_component_count(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
) -> grafeo_common::utils::error::Result<usize> {
    let graph = super::traits::GraphView::new(store, view);
    let found = strongly_connected_component_count_on(&graph);
    graph.finish(found)?
}

/// [`strongly_connected_component_count`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn strongly_connected_component_count_on(
    store: &super::traits::GraphView<'_>,
) -> grafeo_common::utils::error::Result<usize> {
    let components = strongly_connected_components_on(store)?;
    let unique: FxHashSet<u64> = components.values().copied().collect();
    Ok(unique.len())
}

// ============================================================================
// Topological Sort (Kahn's Algorithm)
// ============================================================================

/// Performs topological sort on a directed acyclic graph using Kahn's algorithm.
///
/// The order is the lexicographically smallest by node id: of the nodes with no edge left into
/// them, the smallest id comes next (a min-heap, O(n log n) more), so it depends on the graph
/// alone.
///
/// # Returns
///
/// `Some(order)` if the graph is a DAG, `None` if there's a cycle.
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn topological_sort(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
) -> grafeo_common::utils::error::Result<Option<Vec<NodeId>>> {
    let graph = super::traits::GraphView::new(store, view);
    let found = topological_sort_on(&graph);
    graph.finish(found)?
}

/// [`topological_sort`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn topological_sort_on(
    store: &super::traits::GraphView<'_>,
) -> grafeo_common::utils::error::Result<Option<Vec<NodeId>>> {
    let (order, left) = kahn_on(store, &mut Ticker::new())?;
    Ok(left.is_empty().then_some(order))
}

/// Kahn's algorithm over `store`, the smallest ready id first: the order it reached, and the
/// nodes it left, those on or downstream of a cycle (none for a DAG).
fn kahn_on(
    store: &super::traits::GraphView<'_>,
    ticker: &mut Ticker,
) -> grafeo_common::utils::error::Result<(Vec<NodeId>, FxHashSet<NodeId>)> {
    let node_ids = store.node_ids();

    if node_ids.is_empty() {
        return Ok((Vec::new(), FxHashSet::default()));
    }

    // Compute in-degrees
    let mut in_degree: FxHashMap<NodeId, usize> = FxHashMap::default();
    for &node in &node_ids {
        in_degree.entry(node).or_insert(0);
    }

    for &node in &node_ids {
        for (neighbor, _) in store.edges(node, Direction::Outgoing) {
            ticker.tick()?;
            *in_degree.entry(neighbor).or_default() += 1;
        }
    }

    // The nodes with no edge left into them, smallest id first.
    let mut queue: BinaryHeap<Reverse<NodeId>> = node_ids
        .iter()
        .filter(|&&node| in_degree.get(&node) == Some(&0))
        .map(|&node| Reverse(node))
        .collect();

    let mut result = Vec::with_capacity(node_ids.len());

    while let Some(Reverse(node)) = queue.pop() {
        ticker.tick()?;
        result.push(node);

        for (neighbor, _) in store.edges(node, Direction::Outgoing) {
            ticker.tick()?;
            if let Some(deg) = in_degree.get_mut(&neighbor) {
                *deg -= 1;
                if *deg == 0 {
                    queue.push(Reverse(neighbor));
                }
            }
        }
    }

    let left: FxHashSet<NodeId> = in_degree
        .into_iter()
        .filter(|&(_, degree)| degree > 0)
        .map(|(node, _)| node)
        .collect();
    Ok((result, left))
}

/// One cycle among `left`, the nodes Kahn's algorithm left, found the same way every time: from
/// the smallest of them, back along the smallest of each node's predecessors among them until a
/// node comes again (every node left has a predecessor left, and a walk back cannot dead-end below
/// a cycle as a walk forward can). The cycle's nodes in edge direction.
fn a_cycle(
    store: &super::traits::GraphView<'_>,
    left: &FxHashSet<NodeId>,
    ticker: &mut Ticker,
) -> Result<Vec<NodeId>> {
    let Some(&start) = left.iter().min() else {
        return Ok(Vec::new());
    };
    let mut walked: Vec<NodeId> = vec![start];
    let mut position: FxHashMap<NodeId, usize> = FxHashMap::default();
    position.insert(start, 0);
    let mut at = start;
    loop {
        ticker.tick()?;
        let previous = store
            .edges(at, Direction::Incoming)
            .into_iter()
            .map(|(source, _)| source)
            .filter(|source| left.contains(source))
            .min();
        let Some(previous) = previous else {
            return Ok(Vec::new());
        };
        if let Some(&first) = position.get(&previous) {
            // `walked[first..]` runs against the edges: reversed, it follows them.
            let mut cycle = walked.split_off(first);
            cycle.reverse();
            return Ok(cycle);
        }
        position.insert(previous, walked.len());
        walked.push(previous);
        at = previous;
    }
}

/// The text of a cycle in edge direction, back to its first node: `3 -> 5 -> 8 -> 3`, a
/// self-loop `4 -> 4`, and past 16 nodes the first 16 then the count.
fn cycle_text(cycle: &[NodeId]) -> String {
    const SHOWN: usize = 16;
    let mut parts: Vec<String> = cycle
        .iter()
        .take(SHOWN)
        .map(|node| node.as_u64().to_string())
        .collect();
    if cycle.len() > SHOWN {
        parts.push(format!("... ({} nodes)", cycle.len()));
    } else if let Some(first) = cycle.first() {
        parts.push(first.as_u64().to_string());
    }
    parts.join(" -> ")
}

/// Checks if the graph is a DAG (Directed Acyclic Graph).
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn is_dag(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
) -> grafeo_common::utils::error::Result<bool> {
    let graph = super::traits::GraphView::new(store, view);
    let found = is_dag_on(&graph);
    graph.finish(found)?
}

/// [`is_dag`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn is_dag_on(
    store: &super::traits::GraphView<'_>,
) -> grafeo_common::utils::error::Result<bool> {
    Ok(topological_sort_on(store)?.is_some())
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Connected components algorithm wrapper.
pub struct ConnectedComponentsAlgorithm;

impl_algorithm! {
    ConnectedComponentsAlgorithm,
    name: "connected_components",
    description: "Find connected components (undirected) or weakly connected components (directed)",
    params: &[],
    compute(store, _params) {
        let components = connected_components_on(store)?;

        let mut builder = ComponentResultBuilder::with_capacity(components.len());
        for (node, component) in components {
            builder.push(node, component);
        }

        let mut result = builder.build();
        result.sort_by_id_columns(1);
        Ok(result)
    }
}

/// Strongly connected components algorithm wrapper.
pub struct StronglyConnectedComponentsAlgorithm;

impl_algorithm! {
    StronglyConnectedComponentsAlgorithm,
    name: "strongly_connected_components",
    description: "Find strongly connected components using Tarjan's algorithm",
    params: &[],
    compute(store, _params) {
        let components = strongly_connected_components_on(store)?;

        let mut builder = ComponentResultBuilder::with_capacity(components.len());
        for (node, component) in components {
            builder.push(node, component);
        }

        let mut result = builder.build();
        result.sort_by_id_columns(1);
        Ok(result)
    }
}

/// Topological sort algorithm wrapper.
pub struct TopologicalSortAlgorithm;

impl GraphAlgorithm for TopologicalSortAlgorithm {
    fn name(&self) -> &str {
        "topological_sort"
    }

    fn description(&self) -> &str {
        "Topological ordering of a DAG using Kahn's algorithm"
    }

    fn parameters(&self) -> &[ParameterDef] {
        &[]
    }

    fn compute(
        &self,
        store: &dyn GraphStore,
        view: grafeo_common::types::ReadView,
        _params: &Prepared,
    ) -> Result<AlgorithmResult> {
        let graph = super::traits::GraphView::new(store, view);
        let store = &graph;
        let found = super::traits::evaluated(|| {
            // A graph with a cycle has no topological order: a data exception naming one cycle.
            let mut ticker = Ticker::new();
            let (order, left) = kahn_on(store, &mut ticker)?;
            if !left.is_empty() {
                let cycle = a_cycle(store, &left, &mut ticker)?;
                return Err(Error::from(QueryError::new(
                    QueryErrorKind::Execution,
                    format!(
                        "topological_sort: the graph has a cycle: {}",
                        cycle_text(&cycle)
                    ),
                )));
            }
            let mut result = AlgorithmResult::new(vec!["node_id".to_string(), "order".to_string()]);
            for (idx, node) in order.iter().enumerate() {
                result.add_row(vec![int_value(node.0)?, int_value(idx)?]);
            }
            Ok(result)
        });
        graph.finish(found)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::ReadView;

    fn create_dag() -> RowGroupStore {
        let store = RowGroupStore::new();

        // Create a DAG:
        //   0 -> 1 -> 3
        //   |    |
        //   v    v
        //   2 -> 4
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);
        let n4 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n1, n3, "EDGE");
        store.create_edge(n1, n4, "EDGE");
        store.create_edge(n2, n4, "EDGE");

        store
    }

    fn create_cyclic_graph() -> RowGroupStore {
        let store = RowGroupStore::new();

        // Create a cycle: 0 -> 1 -> 2 -> 0
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");

        store
    }

    fn create_disconnected_graph() -> RowGroupStore {
        let store = RowGroupStore::new();

        // Two disconnected components: {0, 1} and {2, 3}
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n2, n3, "EDGE");

        store
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);

        assert!(!uf.connected(0, 1));
        uf.union(0, 1);
        assert!(uf.connected(0, 1));

        uf.union(2, 3);
        assert!(uf.connected(2, 3));
        assert!(!uf.connected(0, 2));

        uf.union(1, 2);
        assert!(uf.connected(0, 3));
    }

    #[test]
    fn test_connected_components_single() {
        let store = create_dag();
        let count = connected_component_count(&store, ReadView::latest()).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_connected_components_disconnected() {
        let store = create_disconnected_graph();
        let count = connected_component_count(&store, ReadView::latest()).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_scc_dag() {
        let store = create_dag();
        // In a DAG, each node is its own SCC
        let count = strongly_connected_component_count(&store, ReadView::latest()).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_scc_cycle() {
        let store = create_cyclic_graph();
        // All nodes in a single cycle form one SCC
        let count = strongly_connected_component_count(&store, ReadView::latest()).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_topological_sort_dag() {
        let store = create_dag();
        let order = topological_sort(&store, ReadView::latest()).unwrap();
        assert!(order.is_some());

        let order = order.unwrap();
        assert_eq!(order.len(), 5);

        // Verify topological property: if there's an edge u -> v, u comes before v
        let position: FxHashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &n)| (n, i)).collect();

        for &node in &order {
            for (neighbor, _) in store
                .edges(node, Direction::Outgoing, &[], ReadView::latest())
                .unwrap()
            {
                assert!(position[&node] < position[&neighbor]);
            }
        }
    }

    #[test]
    fn test_topological_sort_cycle() {
        let store = create_cyclic_graph();
        let order = topological_sort(&store, ReadView::latest()).unwrap();
        assert!(order.is_none()); // Cycle detected
    }

    #[test]
    fn test_is_dag() {
        let dag = create_dag();
        assert!(is_dag(&dag, ReadView::latest()).unwrap());

        let cyclic = create_cyclic_graph();
        assert!(!is_dag(&cyclic, ReadView::latest()).unwrap());
    }
}
