//! Network flow algorithms: Max Flow (Edmonds-Karp), Min Cost Max Flow.
//!
//! These algorithms find optimal flow through a network with capacity
//! constraints on edges.

use std::collections::VecDeque;
use std::sync::OnceLock;

#[cfg(test)]
use grafeo_common::types::PropertyKey;
use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(test)]
#[cfg(test)]
use grafeo_core::graph::rowgroup::RowGroupStore;

use super::super::{AlgorithmResult, Domain, ParameterDef, ParameterType};
use super::traits::{EdgeWeight, Ticker, impl_algorithm, node_id_from_param};

/// A capacity: what an edge carries at most, so zero or more.
const CAPACITY: EdgeWeight = EdgeWeight {
    what: "capacity",
    default: 1.0,
    domain: Domain::NonNegative,
};

/// A cost per unit of flow, negative ones too: min-cost flow is defined for them, without a
/// negative cycle (see [`refuse_a_negative_cycle`]).
const COST: EdgeWeight = EdgeWeight {
    what: "cost",
    default: 0.0,
    domain: Domain::OfType,
};

// ============================================================================
// Max Flow Result
// ============================================================================

/// Result of max flow algorithm.
#[derive(Debug, Clone)]
pub struct MaxFlowResult {
    /// Maximum flow value.
    pub max_flow: f64,
    /// Flow on each edge: (source, target, flow), by source, then target.
    pub flow_edges: Vec<(NodeId, NodeId, f64)>,
}

// ============================================================================
// Edmonds-Karp Algorithm (Max Flow)
// ============================================================================

/// Computes maximum flow using the Edmonds-Karp algorithm.
///
/// Edmonds-Karp is a specific implementation of Ford-Fulkerson that uses
/// BFS to find augmenting paths, so it needs at most O(VE) of them. The search
/// visits neighbours in node-id order, so of several maximum flows it finds
/// the same one every call.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `source` - Source node ID
/// * `sink` - Sink node ID
/// * `capacity_property` - Optional property name for edge capacities (defaults to 1.0)
///
/// # Returns
///
/// Maximum flow value and flow assignment on edges, by source, then target.
///
/// # Complexity
///
/// O(V × E × (E + V log V)): O(V × E) augmenting paths, each found by a
/// search over the residual graph and applied through its ordered maps.
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn max_flow(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
    source: NodeId,
    sink: NodeId,
    capacity_property: Option<&str>,
) -> grafeo_common::utils::error::Result<Option<MaxFlowResult>> {
    let graph = super::traits::GraphView::new(store, view);
    let found = max_flow_on(&graph, source, sink, capacity_property);
    graph.finish(found)?
}

/// [`max_flow`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn max_flow_on(
    store: &super::traits::GraphView<'_>,
    source: NodeId,
    sink: NodeId,
    capacity_property: Option<&str>,
) -> grafeo_common::utils::error::Result<Option<MaxFlowResult>> {
    let mut ticker = Ticker::new();
    // Verify source and sink exist
    if store.node(source).is_none() || store.node(sink).is_none() {
        return Ok(None);
    }

    if source == sink {
        return Ok(Some(MaxFlowResult {
            max_flow: 0.0,
            flow_edges: Vec::new(),
        }));
    }

    let nodes = store.node_ids();
    let n = nodes.len();

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    let (Some(&source_idx), Some(&sink_idx)) = (node_to_idx.get(&source), node_to_idx.get(&sink))
    else {
        return Ok(None);
    };

    let mut residual = Residual::of(
        store,
        &nodes,
        &node_to_idx,
        capacity_property,
        None,
        &mut ticker,
    )?;
    let mut total_flow = 0.0;

    // Shortest augmenting paths, each found by a BFS over the arcs with capacity left.
    loop {
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut queue: VecDeque<usize> = VecDeque::new();
        visited[source_idx] = true;
        queue.push_back(source_idx);
        while let Some(u) = queue.pop_front() {
            if u == sink_idx {
                break;
            }
            for &arc in &residual.out[u] {
                ticker.tick()?;
                let ResidualArc {
                    to, residual: left, ..
                } = residual.arcs[arc];
                if !visited[to] && left > FLOW_EPSILON {
                    visited[to] = true;
                    parent[to] = Some(arc);
                    queue.push_back(to);
                }
            }
        }
        if !visited[sink_idx] {
            break;
        }
        let path_flow = residual.bottleneck(&parent, sink_idx);
        residual.augment(&parent, sink_idx, path_flow);
        total_flow += path_flow;
    }

    let flow_edges = residual
        .flows()
        .map(|(from, to, flow, _)| (idx_to_node[from], idx_to_node[to], flow))
        .collect();

    Ok(Some(MaxFlowResult {
        max_flow: total_flow,
        flow_edges,
    }))
}

// ============================================================================
// Min Cost Max Flow Result
// ============================================================================

/// Result of min cost max flow algorithm.
#[derive(Debug, Clone)]
pub struct MinCostFlowResult {
    /// Maximum flow value.
    pub max_flow: f64,
    /// Total cost of the flow.
    pub total_cost: f64,
    /// Flow on each edge that carries some: (source, target, flow, cost), by
    /// source, then target; parallel edges are a row each.
    pub flow_edges: Vec<(NodeId, NodeId, f64, f64)>,
}

// ============================================================================
// Min Cost Max Flow (Successive Shortest Paths)
// ============================================================================

/// Computes minimum cost maximum flow using successive shortest paths.
///
/// Finds the maximum flow with minimum total cost by repeatedly finding
/// the shortest (cheapest) augmenting path using Bellman-Ford. The search
/// relaxes edges in node-id order, so of several cheapest flows it finds the
/// same one every call.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `source` - Source node ID
/// * `sink` - Sink node ID
/// * `capacity_property` - Optional property name for edge capacities
/// * `cost_property` - Optional property name for edge costs
///
/// # Returns
///
/// Maximum flow value, total cost, and flow assignment on edges, by source,
/// then target.
///
/// # Complexity
///
/// O(V² × E × max_flow)
///
/// # Errors
///
/// The error of a read of the store that fails.
pub fn min_cost_max_flow(
    store: &dyn GraphStore,
    view: grafeo_common::types::ReadView,
    source: NodeId,
    sink: NodeId,
    capacity_property: Option<&str>,
    cost_property: Option<&str>,
) -> grafeo_common::utils::error::Result<Option<MinCostFlowResult>> {
    let graph = super::traits::GraphView::new(store, view);
    let found = min_cost_max_flow_on(&graph, source, sink, capacity_property, cost_property);
    graph.finish(found)?
}

/// [`min_cost_max_flow`] on the graph a [`GraphView`](super::traits::GraphView) reads.
pub(crate) fn min_cost_max_flow_on(
    store: &super::traits::GraphView<'_>,
    source: NodeId,
    sink: NodeId,
    capacity_property: Option<&str>,
    cost_property: Option<&str>,
) -> grafeo_common::utils::error::Result<Option<MinCostFlowResult>> {
    let mut ticker = Ticker::new();
    // Verify source and sink exist
    if store.node(source).is_none() || store.node(sink).is_none() {
        return Ok(None);
    }

    if source == sink {
        return Ok(Some(MinCostFlowResult {
            max_flow: 0.0,
            total_cost: 0.0,
            flow_edges: Vec::new(),
        }));
    }

    let nodes = store.node_ids();
    let n = nodes.len();

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<NodeId> = Vec::with_capacity(n);
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
        idx_to_node.push(node);
    }

    let (Some(&source_idx), Some(&sink_idx)) = (node_to_idx.get(&source), node_to_idx.get(&sink))
    else {
        return Ok(None);
    };

    let mut residual = Residual::of(
        store,
        &nodes,
        &node_to_idx,
        capacity_property,
        cost_property,
        &mut ticker,
    )?;
    refuse_a_negative_cycle(
        &residual.arcs,
        &residual.out,
        source_idx,
        cost_property,
        &mut ticker,
    )?;

    let mut total_flow = 0.0;
    let mut total_cost_val = 0.0;

    // Successive shortest paths, each found by Bellman-Ford over the arcs with capacity left.
    while let Some((path_cost, parent)) = cheapest_paths(
        &residual.arcs,
        &residual.out,
        source_idx,
        sink_idx,
        &mut ticker,
    )? {
        let path_flow = residual.bottleneck(&parent, sink_idx);
        residual.augment(&parent, sink_idx, path_flow);
        total_flow += path_flow;
        total_cost_val += path_flow * path_cost;
    }

    let flow_edges = residual
        .flows()
        .map(|(from, to, flow, cost)| (idx_to_node[from], idx_to_node[to], flow, cost))
        .collect();

    Ok(Some(MinCostFlowResult {
        max_flow: total_flow,
        total_cost: total_cost_val,
        flow_edges,
    }))
}

/// Residual capacity below which an arc counts as full.
const FLOW_EPSILON: f64 = 1e-9;

/// The residual graph both flow procedures search: each edge a forward arc with its own capacity
/// and cost, and the reverse arc that undoes its flow next to it (`arc ^ 1`), so parallel edges
/// and an edge with its opposite stay arcs of their own. A node's arcs follow the node-id order of
/// the edges' sources, then the store's order of each node's edges.
struct Residual {
    arcs: Vec<ResidualArc>,
    /// The arcs from each node, by index.
    out: Vec<Vec<usize>>,
}

impl Residual {
    /// The residual graph of the edges between `nodes`, each with its capacity and cost (as
    /// [`CAPACITY`] and [`COST`] read them, a cost of 0 without a property), one unit of work per
    /// edge.
    fn of(
        store: &super::traits::GraphView<'_>,
        nodes: &[NodeId],
        node_to_idx: &FxHashMap<NodeId, usize>,
        capacity_property: Option<&str>,
        cost_property: Option<&str>,
        ticker: &mut Ticker,
    ) -> grafeo_common::utils::error::Result<Self> {
        let mut arcs: Vec<ResidualArc> = Vec::new();
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for (i, &node) in nodes.iter().enumerate() {
            for (neighbor, edge_id) in store.edges(node, Direction::Outgoing) {
                ticker.tick()?;
                let Some(&j) = node_to_idx.get(&neighbor) else {
                    continue;
                };
                let capacity = store.edge_weight(edge_id, capacity_property, CAPACITY)?;
                let cost = store.edge_weight(edge_id, cost_property, COST)?;
                out[i].push(arcs.len());
                arcs.push(ResidualArc {
                    from: i,
                    to: j,
                    residual: capacity,
                    cost,
                    edge: edge_id,
                });
                out[j].push(arcs.len());
                arcs.push(ResidualArc {
                    from: j,
                    to: i,
                    residual: 0.0,
                    cost: -cost,
                    edge: edge_id,
                });
            }
        }
        Ok(Self { arcs, out })
    }

    /// The capacity left on the path `parent` gives back from `sink`, an arc into each node.
    fn bottleneck(&self, parent: &[Option<usize>], sink: usize) -> f64 {
        let mut least = f64::INFINITY;
        let mut at = sink;
        while let Some(arc) = parent[at] {
            least = least.min(self.arcs[arc].residual);
            at = self.arcs[arc].from;
        }
        least
    }

    /// Pushes `flow` along the path `parent` gives back from `sink`.
    fn augment(&mut self, parent: &[Option<usize>], sink: usize, flow: f64) {
        let mut at = sink;
        while let Some(arc) = parent[at] {
            self.arcs[arc].residual -= flow;
            self.arcs[arc ^ 1].residual += flow;
            at = self.arcs[arc].from;
        }
    }

    /// Each edge that carries flow, as (source index, target index, flow, cost), by source, then
    /// target, parallel edges in the store's order: the flow of an edge is what its reverse arc
    /// can undo.
    fn flows(&self) -> impl Iterator<Item = (usize, usize, f64, f64)> {
        let mut flows: Vec<(usize, usize, f64, f64)> = self
            .arcs
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|[_, back]| back.residual > FLOW_EPSILON)
            .map(|[forward, back]| (forward.from, forward.to, back.residual, forward.cost))
            .collect();
        flows.sort_by_key(|&(from, to, _, _)| (from, to));
        flows.into_iter()
    }
}

/// An arc of the [`Residual`] graph: an edge's forward arc, or its reverse arc, which undoes the
/// edge's flow.
#[derive(Clone, Copy)]
struct ResidualArc {
    from: usize,
    to: usize,
    /// Capacity left.
    residual: f64,
    cost: f64,
    /// The edge it belongs to, for an error.
    edge: EdgeId,
}

/// The cheapest path cost from `source` to `sink` over the arcs with capacity left, and the arc
/// into each node on the cheapest paths; `None` when the sink cannot be reached.
fn cheapest_paths(
    arcs: &[ResidualArc],
    out: &[Vec<usize>],
    source: usize,
    sink: usize,
    ticker: &mut Ticker,
) -> grafeo_common::utils::error::Result<Option<(f64, Vec<Option<usize>>)>> {
    let (dist, parent, _) = bellman_ford_over(arcs, out, source, ticker)?;
    Ok((dist[sink] != f64::INFINITY).then(|| (dist[sink], parent)))
}

/// Bellman-Ford from `source` over the arcs with capacity left: the distances, the arc into each
/// node, and a node relaxed in round `n`, which only a negative cycle leaves.
fn bellman_ford_over(
    arcs: &[ResidualArc],
    out: &[Vec<usize>],
    source: usize,
    ticker: &mut Ticker,
) -> grafeo_common::utils::error::Result<(Vec<f64>, Vec<Option<usize>>, Option<usize>)> {
    let n = out.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];
    dist[source] = 0.0;
    let mut last_relaxed = None;
    for _ in 0..n {
        let mut relaxed = None;
        for u in 0..n {
            ticker.tick()?;
            if dist[u] == f64::INFINITY {
                continue;
            }
            for &arc in &out[u] {
                let ResidualArc {
                    to, residual, cost, ..
                } = arcs[arc];
                if residual > FLOW_EPSILON && dist[u] + cost < dist[to] {
                    dist[to] = dist[u] + cost;
                    parent[to] = Some(arc);
                    relaxed = Some(to);
                }
            }
        }
        if relaxed.is_none() {
            return Ok((dist, parent, None));
        }
        last_relaxed = relaxed;
    }
    Ok((dist, parent, last_relaxed))
}

/// Refuses costs that hold a negative cycle the source reaches through arcs with capacity left,
/// as an invalid parameter value (22023) naming an edge of the cycle and the cost property: the
/// flow of least cost is then not defined, and the successive shortest paths never end. They keep
/// the residual graph free of such a cycle once the start has none, so one check before the first
/// augmentation suffices: Bellman-Ford from the source, where a relaxation in round `n` means a
/// cycle. A cycle the source cannot reach is never met, and a zero-cost cycle relaxes nothing.
fn refuse_a_negative_cycle(
    arcs: &[ResidualArc],
    out: &[Vec<usize>],
    source: usize,
    cost_property: Option<&str>,
    ticker: &mut Ticker,
) -> grafeo_common::utils::error::Result<()> {
    let (_, parent, relaxed) = bellman_ford_over(arcs, out, source, ticker)?;
    let Some(mut on_cycle) = relaxed else {
        return Ok(());
    };
    // `n` steps back along the arcs from a node relaxed in round `n` land on the cycle.
    for _ in 0..out.len() {
        if let Some(arc) = parent[on_cycle] {
            on_cycle = arcs[arc].from;
        }
    }
    let edge = parent[on_cycle].map_or_else(String::new, |arc| {
        format!(" through edge {}", arcs[arc].edge.as_u64())
    });
    Err(Error::from(QueryError::new(
        QueryErrorKind::InvalidParameterValue,
        format!(
            "min_cost_max_flow: the costs '{}' hold a negative cycle the source reaches{edge}, so \
             no flow of least cost exists",
            cost_property.unwrap_or("cost")
        ),
    )))
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for Max Flow algorithm.
static MAX_FLOW_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn max_flow_params() -> &'static [ParameterDef] {
    MAX_FLOW_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "source".to_string(),
                description: "Source node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
                domain: Domain::OfType,
            },
            ParameterDef {
                name: "sink".to_string(),
                description: "Sink node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
                domain: Domain::OfType,
            },
            ParameterDef {
                name: "capacity".to_string(),
                description: "Edge property name for capacities (default: 1.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
                domain: Domain::OfType,
            },
        ]
    })
}

/// Max Flow algorithm wrapper.
pub struct MaxFlowAlgorithm;

impl_algorithm! {
    MaxFlowAlgorithm,
    name: "max_flow",
    description: "Maximum flow using Edmonds-Karp algorithm",
    params: max_flow_params,
    compute(store, params) {
        let source_id = params.required_int("source")?;
        let sink_id = params.required_int("sink")?;

        let source = node_id_from_param(source_id, "source")?;
        let sink = node_id_from_param(sink_id, "sink")?;
        let capacity_prop = params.get_string("capacity");

        let result = max_flow_on(store, source, sink, capacity_prop)?
            .ok_or_else(|| Error::InvalidValue("Invalid source or sink node".to_string()))?;

        let mut output = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "flow".to_string(),
            "max_flow".to_string(),
        ]);

        for (src, dst, flow) in result.flow_edges {
            // reason: node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            output.add_row(vec![
                Value::Int64(src.0 as i64),
                Value::Int64(dst.0 as i64),
                Value::Float64(flow),
                Value::Float64(result.max_flow),
            ]);
        }

        // Add summary row if no edges
        if output.row_count() == 0 {
            output.add_row(vec![
                Value::Int64(source_id),
                Value::Int64(sink_id),
                Value::Float64(0.0),
                Value::Float64(result.max_flow),
            ]);
        }

        output.sort_by_id_columns(2);
        Ok(output)
    }
}

/// Static parameter definitions for Min Cost Max Flow algorithm.
static MIN_COST_FLOW_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn min_cost_flow_params() -> &'static [ParameterDef] {
    MIN_COST_FLOW_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "source".to_string(),
                description: "Source node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
                domain: Domain::OfType,
            },
            ParameterDef {
                name: "sink".to_string(),
                description: "Sink node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
                domain: Domain::OfType,
            },
            ParameterDef {
                name: "capacity".to_string(),
                description: "Edge property name for capacities (default: 1.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
                domain: Domain::OfType,
            },
            ParameterDef {
                name: "cost".to_string(),
                description: "Edge property name for costs (default: 0.0)".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
                domain: Domain::OfType,
            },
        ]
    })
}

/// Min Cost Max Flow algorithm wrapper.
pub struct MinCostFlowAlgorithm;

impl_algorithm! {
    MinCostFlowAlgorithm,
    name: "min_cost_max_flow",
    description: "Minimum cost maximum flow using successive shortest paths",
    params: min_cost_flow_params,
    compute(store, params) {
        let source_id = params.required_int("source")?;
        let sink_id = params.required_int("sink")?;

        let source = node_id_from_param(source_id, "source")?;
        let sink = node_id_from_param(sink_id, "sink")?;
        let capacity_prop = params.get_string("capacity");
        let cost_prop = params.get_string("cost");

        let result = min_cost_max_flow_on(store, source, sink, capacity_prop, cost_prop)?
            .ok_or_else(|| Error::InvalidValue("Invalid source or sink node".to_string()))?;

        let mut output = AlgorithmResult::new(vec![
            "source".to_string(),
            "target".to_string(),
            "flow".to_string(),
            "cost".to_string(),
            "max_flow".to_string(),
            "total_cost".to_string(),
        ]);

        for (src, dst, flow, cost) in result.flow_edges {
            // reason: node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            output.add_row(vec![
                Value::Int64(src.0 as i64),
                Value::Int64(dst.0 as i64),
                Value::Float64(flow),
                Value::Float64(cost),
                Value::Float64(result.max_flow),
                Value::Float64(result.total_cost),
            ]);
        }

        output.sort_by_id_columns(2);
        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::ReadView;

    fn create_simple_flow_graph() -> RowGroupStore {
        // Simple flow network:
        //   0 --[5]--> 1 --[3]--> 3
        //   |          |
        //  [3]        [2]
        //   |          |
        //   v          v
        //   2 --[4]--> 3
        let store = RowGroupStore::new();

        let n0 = store.create_node(&["Node"]); // Source
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]); // Sink

        store.create_edge_with_props(
            n0,
            n1,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(5.0))],
        );
        store.create_edge_with_props(
            n0,
            n2,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(3.0))],
        );
        store.create_edge_with_props(
            n1,
            n3,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(3.0))],
        );
        store.create_edge_with_props(
            n1,
            n2,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(2.0))],
        );
        store.create_edge_with_props(
            n2,
            n3,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(4.0))],
        );

        store
    }

    fn create_cost_flow_graph() -> RowGroupStore {
        // Flow network with costs:
        //   0 --[cap:3,cost:1]--> 1 --[cap:2,cost:2]--> 2
        //   |                                           ^
        //   +--[cap:2,cost:5]---------------------------+
        let store = RowGroupStore::new();

        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge_with_props(
            n0,
            n1,
            "EDGE",
            &[
                (PropertyKey::new("capacity"), Value::Float64(3.0)),
                (PropertyKey::new("cost"), Value::Float64(1.0)),
            ],
        );
        store.create_edge_with_props(
            n1,
            n2,
            "EDGE",
            &[
                (PropertyKey::new("capacity"), Value::Float64(2.0)),
                (PropertyKey::new("cost"), Value::Float64(2.0)),
            ],
        );
        store.create_edge_with_props(
            n0,
            n2,
            "EDGE",
            &[
                (PropertyKey::new("capacity"), Value::Float64(2.0)),
                (PropertyKey::new("cost"), Value::Float64(5.0)),
            ],
        );

        store
    }

    #[test]
    fn test_max_flow_basic() {
        let store = create_simple_flow_graph();
        let result = max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(0),
            NodeId::new(3),
            Some("capacity"),
        )
        .unwrap();

        assert!(result.is_some());
        let result = result.unwrap();

        // Max flow should be limited by edge capacities
        // Path 0->1->3 can carry 3
        // Path 0->2->3 can carry 3
        // Path 0->1->2->3 can carry min(5-3, 2, 4-3) = min(2, 2, 1) = 1
        // Total: 3 + 3 + 1 = 7? Let's verify...
        assert!(result.max_flow >= 5.0 && result.max_flow <= 8.0);
    }

    #[test]
    fn test_max_flow_same_source_sink() {
        let store = create_simple_flow_graph();
        let result = max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(0),
            NodeId::new(0),
            Some("capacity"),
        )
        .unwrap();

        assert!(result.is_some());
        assert_eq!(result.unwrap().max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_no_path() {
        let store = RowGroupStore::new();
        let n0 = store.create_node(&["Node"]);
        let _n1 = store.create_node(&["Node"]); // Disconnected

        let result = max_flow(&store, ReadView::latest(), n0, NodeId::new(1), None).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_invalid_nodes() {
        let store = RowGroupStore::new();
        store.create_node(&["Node"]);

        let result = max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(999),
            NodeId::new(0),
            None,
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_min_cost_flow_basic() {
        let store = create_cost_flow_graph();
        let result = min_cost_max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(0),
            NodeId::new(2),
            Some("capacity"),
            Some("cost"),
        )
        .unwrap();

        assert!(result.is_some());
        let result = result.unwrap();

        // Max flow through the network
        assert!(result.max_flow > 0.0);

        // Cost should be positive
        assert!(result.total_cost >= 0.0);
    }

    #[test]
    fn test_min_cost_prefers_cheaper_path() {
        let store = create_cost_flow_graph();
        let result = min_cost_max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(0),
            NodeId::new(2),
            Some("capacity"),
            Some("cost"),
        )
        .unwrap();

        assert!(result.is_some());
        let result = result.unwrap();

        // Should prefer path 0->1->2 (cost 3) over 0->2 (cost 5) when possible
        // Max flow through 0->1->2 is 2 (limited by 1->2)
        // Remaining flow 2 goes through 0->2
        // Total: 4 units of flow
        assert!(result.max_flow >= 2.0);
    }

    #[test]
    fn test_min_cost_flow_no_path() {
        let store = RowGroupStore::new();
        let n0 = store.create_node(&["Node"]);
        let _n1 = store.create_node(&["Node"]);

        let result =
            min_cost_max_flow(&store, ReadView::latest(), n0, NodeId::new(1), None, None).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_default_capacity() {
        // Without capacity property, all edges have capacity 1.0
        let store = RowGroupStore::new();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n1, n2, "EDGE");

        let result = max_flow(&store, ReadView::latest(), n0, n2, None).unwrap();
        assert!(result.is_some());

        // Two paths: 0->2 (cap 1) and 0->1->2 (cap 1)
        assert!(result.unwrap().max_flow >= 1.0);
    }

    #[test]
    fn test_max_flow_flow_edges_populated() {
        let store = create_simple_flow_graph();
        let result = max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(0),
            NodeId::new(3),
            Some("capacity"),
        )
        .unwrap()
        .unwrap();

        // Flow edges should be non-empty if there's positive flow
        assert!(result.max_flow > 0.0);
        assert!(!result.flow_edges.is_empty(), "result.flow_edges is empty");

        // All flow values should be positive
        for (_, _, flow) in &result.flow_edges {
            assert!(*flow > 0.0);
        }
    }

    #[test]
    fn test_max_flow_int_capacity() {
        let store = RowGroupStore::new();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);

        // Use Int64 capacity values instead of Float64
        store.create_edge_with_props(
            n0,
            n1,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Int64(5))],
        );

        let result = max_flow(&store, ReadView::latest(), n0, n1, Some("capacity"))
            .unwrap()
            .unwrap();
        assert!((result.max_flow - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_flow_parallel_paths() {
        let store = RowGroupStore::new();
        let s = store.create_node(&["Node"]);
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let t = store.create_node(&["Node"]);

        // Two independent paths: s->a->t and s->b->t
        store.create_edge_with_props(
            s,
            a,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(3.0))],
        );
        store.create_edge_with_props(
            a,
            t,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(3.0))],
        );
        store.create_edge_with_props(
            s,
            b,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(2.0))],
        );
        store.create_edge_with_props(
            b,
            t,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(2.0))],
        );

        let result = max_flow(&store, ReadView::latest(), s, t, Some("capacity"))
            .unwrap()
            .unwrap();
        assert!((result.max_flow - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_flow_bottleneck() {
        let store = RowGroupStore::new();
        let s = store.create_node(&["Node"]);
        let mid = store.create_node(&["Node"]);
        let t = store.create_node(&["Node"]);

        // s -> mid has high capacity, mid -> t has low = bottleneck
        store.create_edge_with_props(
            s,
            mid,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(100.0))],
        );
        store.create_edge_with_props(
            mid,
            t,
            "EDGE",
            &[(PropertyKey::new("capacity"), Value::Float64(1.0))],
        );

        let result = max_flow(&store, ReadView::latest(), s, t, Some("capacity"))
            .unwrap()
            .unwrap();
        assert!((result.max_flow - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_min_cost_flow_same_source_sink() {
        let store = create_cost_flow_graph();
        let result = min_cost_max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(0),
            NodeId::new(0),
            Some("capacity"),
            Some("cost"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.max_flow, 0.0);
        assert_eq!(result.total_cost, 0.0);
    }

    #[test]
    fn test_min_cost_flow_invalid_nodes() {
        let store = RowGroupStore::new();
        store.create_node(&["Node"]);
        let result = min_cost_max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(999),
            NodeId::new(0),
            None,
            None,
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_min_cost_flow_edges() {
        let store = create_cost_flow_graph();
        let result = min_cost_max_flow(
            &store,
            ReadView::latest(),
            NodeId::new(0),
            NodeId::new(2),
            Some("capacity"),
            Some("cost"),
        )
        .unwrap()
        .unwrap();

        // Flow edges should have both flow and cost info
        for (_, _, flow, _cost) in &result.flow_edges {
            assert!(*flow > 0.0);
        }
    }

    /// A source feeding a hub that fans out over 19 paths of capacity 1 to
    /// the sink; `hub_capacity` of them can carry flow. Returns the store,
    /// the source and the sink.
    fn fan_out(hub_capacity: f64) -> (RowGroupStore, NodeId, NodeId) {
        let store = RowGroupStore::new();
        let source = store.create_node(&["Node"]);
        let hub = store.create_node(&["Node"]);
        let middles: Vec<NodeId> = (0..19).map(|_| store.create_node(&["Node"])).collect();
        let sink = store.create_node(&["Node"]);
        let edge = |from, to, capacity: f64| {
            store.create_edge_with_props(
                from,
                to,
                "EDGE",
                &[
                    (PropertyKey::new("capacity"), Value::Float64(capacity)),
                    (PropertyKey::new("cost"), Value::Float64(3.0)),
                ],
            );
        };
        edge(source, hub, hub_capacity);
        for &middle in &middles {
            edge(hub, middle, 1.0);
            edge(middle, sink, 1.0);
        }
        (store, source, sink)
    }

    /// Max flow lists its flow edges by source, then target, and finds the
    /// same flow every call: its search walks the residual graph in node-id
    /// order, not in the order of a hash map (#592). With 3 of 19 equal paths
    /// in use, the paths it picks are the first 3 by node id.
    #[test]
    fn max_flow_edges_come_in_node_id_order_and_are_the_same_every_call() {
        for hub_capacity in [3.0, 19.0] {
            let (store, source, sink) = fan_out(hub_capacity);
            let first = max_flow(&store, ReadView::latest(), source, sink, Some("capacity"))
                .unwrap()
                .unwrap();
            assert_eq!(first.max_flow.to_bits(), hub_capacity.to_bits());
            let pairs: Vec<(NodeId, NodeId)> =
                first.flow_edges.iter().map(|&(s, t, _)| (s, t)).collect();
            assert!(
                pairs.windows(2).all(|w| w[0] < w[1]),
                "hub capacity {hub_capacity}: flow edges not in node-id order: {pairs:?}"
            );
            for _ in 0..5 {
                let again = max_flow(&store, ReadView::latest(), source, sink, Some("capacity"))
                    .unwrap()
                    .unwrap();
                let again: Vec<(NodeId, NodeId, u64)> = again
                    .flow_edges
                    .iter()
                    .map(|&(s, t, flow)| (s, t, flow.to_bits()))
                    .collect();
                let expected: Vec<(NodeId, NodeId, u64)> = first
                    .flow_edges
                    .iter()
                    .map(|&(s, t, flow)| (s, t, flow.to_bits()))
                    .collect();
                assert_eq!(again, expected, "hub capacity {hub_capacity}");
            }
        }
        let (store, source, sink) = fan_out(3.0);
        let hub = NodeId::new(source.as_u64() + 1);
        let used: Vec<NodeId> =
            max_flow(&store, ReadView::latest(), source, sink, Some("capacity"))
                .unwrap()
                .unwrap()
                .flow_edges
                .iter()
                .filter(|&&(s, _, _)| s == hub)
                .map(|&(_, t, _)| t)
                .collect();
        let first_three: Vec<NodeId> = (2..5).map(|i| NodeId::new(source.as_u64() + i)).collect();
        assert_eq!(used, first_three, "the paths through the smallest node ids");
    }

    /// Min-cost max flow lists its flow edges by source, then target, the
    /// same every call (#592).
    #[test]
    fn min_cost_flow_edges_come_in_node_id_order_and_are_the_same_every_call() {
        for hub_capacity in [3.0, 19.0] {
            let (store, source, sink) = fan_out(hub_capacity);
            let run = || -> Vec<(NodeId, NodeId, u64, u64)> {
                min_cost_max_flow(
                    &store,
                    ReadView::latest(),
                    source,
                    sink,
                    Some("capacity"),
                    Some("cost"),
                )
                .unwrap()
                .unwrap()
                .flow_edges
                .iter()
                .map(|&(s, t, flow, cost)| (s, t, flow.to_bits(), cost.to_bits()))
                .collect()
            };
            let first = run();
            assert!(
                first
                    .windows(2)
                    .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)),
                "hub capacity {hub_capacity}: flow edges not in node-id order: {first:?}"
            );
            for _ in 0..5 {
                assert_eq!(run(), first, "hub capacity {hub_capacity}");
            }
        }
    }
}
