//! Core traits and types for graph algorithms.
//!
//! This module provides the fundamental building blocks for implementing
//! high-performance graph algorithms, inspired by rustworkx patterns.

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::{Direction, GraphStore};
use std::cmp::Ordering;
use std::collections::HashMap;

use super::super::{AlgorithmResult, ParameterDef, Parameters};

/// The edges from `node` in `direction` that an algorithm reads, each with the
/// node at its other end: those visible at the store's current epoch whose
/// other end is visible too, so the graph an algorithm walks has the nodes of
/// [`GraphStore::node_ids`] and only edges between them.
///
/// A store's adjacency holds more: the edges of a node a store-level
/// `delete_node` removed without detaching it, and the edges a transaction
/// created and has not committed (to a node only it sees, or between two
/// visible nodes). Every algorithm reads adjacency through this function.
pub(crate) fn visible_edges_from(
    store: &dyn GraphStore,
    node: NodeId,
    direction: Direction,
) -> Vec<(NodeId, EdgeId)> {
    let epoch = store.current_epoch();
    let mut edges = store.edges_from(node, direction);
    edges.retain(|&(other, edge)| {
        store.is_edge_visible_at_epoch(edge, epoch) && store.is_node_visible_at_epoch(other, epoch)
    });
    edges
}

/// Safely converts a user-supplied `i64` parameter into a [`NodeId`].
///
/// Returns an error if the value is negative, since node IDs are unsigned.
pub fn node_id_from_param(value: i64, param_name: &str) -> Result<NodeId> {
    u64::try_from(value)
        .map(NodeId::new)
        .map_err(|_| Error::InvalidValue(format!("{param_name} must be non-negative, got {value}")))
}

// ============================================================================
// Control Flow
// ============================================================================

/// Control flow for algorithm visitors.
///
/// Returned by visitor callbacks to control traversal behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control<B = ()> {
    /// Continue the algorithm normally.
    Continue,
    /// Stop the algorithm and return the break value.
    Break(B),
    /// Skip the current subtree (for traversals) but continue the algorithm.
    Prune,
}

impl<B> Control<B> {
    /// Returns `true` if this is `Continue`.
    #[inline]
    pub fn is_continue(&self) -> bool {
        matches!(self, Control::Continue)
    }

    /// Returns `true` if this is `Break`.
    #[inline]
    pub fn is_break(&self) -> bool {
        matches!(self, Control::Break(_))
    }

    /// Returns `true` if this is `Prune`.
    #[inline]
    pub fn is_prune(&self) -> bool {
        matches!(self, Control::Prune)
    }

    /// Converts to an option containing the break value.
    pub fn break_value(self) -> Option<B> {
        match self {
            Control::Break(b) => Some(b),
            _ => None,
        }
    }
}

impl<B> Default for Control<B> {
    fn default() -> Self {
        Control::Continue
    }
}

// ============================================================================
// Traversal Events
// ============================================================================

/// Events emitted during graph traversal (BFS/DFS).
///
/// These events follow the visitor pattern, allowing algorithms to
/// react to different stages of the traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalEvent {
    /// A node is discovered for the first time.
    Discover(NodeId),

    /// A tree edge is traversed (edge to an undiscovered node).
    TreeEdge {
        /// The source node of the edge.
        source: NodeId,
        /// The target node of the edge.
        target: NodeId,
        /// The edge ID.
        edge: EdgeId,
    },

    /// A non-tree edge is traversed (edge to an already-discovered node).
    /// In BFS, this is a cross edge. In DFS, this could be a back edge or cross edge.
    NonTreeEdge {
        /// The source node of the edge.
        source: NodeId,
        /// The target node of the edge.
        target: NodeId,
        /// The edge ID.
        edge: EdgeId,
    },

    /// A back edge is traversed (DFS only - edge to an ancestor).
    BackEdge {
        /// The source node of the edge.
        source: NodeId,
        /// The target node of the edge.
        target: NodeId,
        /// The edge ID.
        edge: EdgeId,
    },

    /// Processing of a node is complete.
    Finish(NodeId),
}

impl TraversalEvent {
    /// Returns the source node of this event, if applicable.
    pub fn source(&self) -> Option<NodeId> {
        match self {
            TraversalEvent::Discover(n) | TraversalEvent::Finish(n) => Some(*n),
            TraversalEvent::TreeEdge { source, .. }
            | TraversalEvent::NonTreeEdge { source, .. }
            | TraversalEvent::BackEdge { source, .. } => Some(*source),
        }
    }

    /// Returns the target node of this event, if applicable.
    pub fn target(&self) -> Option<NodeId> {
        match self {
            TraversalEvent::Discover(n) | TraversalEvent::Finish(n) => Some(*n),
            TraversalEvent::TreeEdge { target, .. }
            | TraversalEvent::NonTreeEdge { target, .. }
            | TraversalEvent::BackEdge { target, .. } => Some(*target),
        }
    }
}

// ============================================================================
// Priority Queue Support
// ============================================================================

/// Wrapper for priority queue ordering (min-heap behavior).
///
/// Rust's `BinaryHeap` is a max-heap, so we reverse the ordering
/// to get min-heap behavior for Dijkstra and A*.
#[derive(Clone, Copy, Debug)]
pub struct MinScored<K, T>(pub K, pub T);

impl<K, T> MinScored<K, T> {
    /// Creates a new `MinScored` with the given score and value.
    pub fn new(score: K, value: T) -> Self {
        MinScored(score, value)
    }

    /// Returns the score.
    pub fn score(&self) -> &K {
        &self.0
    }

    /// Returns the value.
    pub fn value(&self) -> &T {
        &self.1
    }

    /// Consumes self and returns the value.
    pub fn into_value(self) -> T {
        self.1
    }
}

impl<K: PartialOrd, T> PartialEq for MinScored<K, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: PartialOrd, T> Eq for MinScored<K, T> {}

impl<K: PartialOrd, T> Ord for MinScored<K, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

impl<K: PartialOrd, T> PartialOrd for MinScored<K, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ============================================================================
// Graph Algorithm Traits
// ============================================================================

/// A graph algorithm that can be executed on any graph store.
///
/// This trait extends the base `Algorithm` trait with graph-specific
/// functionality, providing direct access to the graph store.
pub trait GraphAlgorithm: Send + Sync {
    /// Returns the name of the algorithm.
    fn name(&self) -> &str;

    /// Returns a description of the algorithm.
    fn description(&self) -> &str;

    /// Returns the parameter definitions for this algorithm.
    fn parameters(&self) -> &[ParameterDef];

    /// Executes the algorithm on the given graph store.
    ///
    /// # Errors
    ///
    /// Returns an error if the algorithm fails (e.g., invalid parameters or graph state).
    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult>;
}

/// A graph algorithm that supports parallel execution.
///
/// Algorithms implementing this trait can automatically switch between
/// sequential and parallel execution based on graph size.
pub trait ParallelGraphAlgorithm: GraphAlgorithm {
    /// Minimum node count to trigger parallelization.
    ///
    /// Below this threshold, sequential execution is used to avoid
    /// parallelization overhead.
    fn parallel_threshold(&self) -> usize {
        50
    }

    /// Executes the algorithm with explicit parallelism control.
    ///
    /// # Errors
    ///
    /// Returns an error if the algorithm fails (e.g., invalid parameters or graph state).
    fn execute_parallel(
        &self,
        store: &dyn GraphStore,
        params: &Parameters,
        num_threads: usize,
    ) -> Result<AlgorithmResult>;
}

// ============================================================================
// Distance Map
// ============================================================================

/// A map from nodes to distances/costs.
///
/// This trait abstracts over different distance map implementations,
/// allowing algorithms to work with various storage strategies.
pub trait DistanceMap<K> {
    /// Gets the distance to a node, if known.
    fn get(&self, node: NodeId) -> Option<&K>;

    /// Sets the distance to a node.
    fn insert(&mut self, node: NodeId, dist: K);

    /// Returns `true` if the node has a recorded distance.
    fn contains(&self, node: NodeId) -> bool {
        self.get(node).is_some()
    }
}

impl<V> DistanceMap<V> for HashMap<NodeId, V> {
    fn get(&self, node: NodeId) -> Option<&V> {
        HashMap::get(self, &node)
    }

    fn insert(&mut self, node: NodeId, dist: V) {
        HashMap::insert(self, node, dist);
    }
}

// ============================================================================
// impl_algorithm! Macro
// ============================================================================

/// Generates a `GraphAlgorithm` trait implementation with the standard boilerplate.
///
/// # Syntax
///
/// ```ignore
/// impl_algorithm! {
///     StructName,
///     name: "algorithm_name",
///     description: "What the algorithm does",
///     params: params_fn,            // fn() -> &'static [ParameterDef]
///     execute(store, params) { ... } // body that returns Result<AlgorithmResult>
/// }
/// ```
///
/// For algorithms with no parameters, use `params: &[]` instead of a function name.
/// If the execute body does not use `params`, prefix it with `_` to silence warnings:
///
/// ```ignore
/// impl_algorithm! {
///     StructName,
///     name: "algorithm_name",
///     description: "What the algorithm does",
///     params: &[],
///     execute(store, _params) { ... }
/// }
/// ```
macro_rules! impl_algorithm {
    (
        $struct_name:ty,
        name: $name:expr,
        description: $desc:expr,
        params: &[],
        execute($store:ident, $params:ident) $body:block
    ) => {
        impl $crate::plugins::algorithms::GraphAlgorithm for $struct_name {
            fn name(&self) -> &str {
                $name
            }

            fn description(&self) -> &str {
                $desc
            }

            fn parameters(&self) -> &[super::super::ParameterDef] {
                &[]
            }

            fn execute(
                &self,
                $store: &dyn grafeo_core::graph::GraphStore,
                $params: &super::super::Parameters,
            ) -> grafeo_common::utils::error::Result<super::super::AlgorithmResult> {
                $body
            }
        }
    };
    (
        $struct_name:ty,
        name: $name:expr,
        description: $desc:expr,
        params: $params_fn:expr,
        execute($store:ident, $params:ident) $body:block
    ) => {
        impl $crate::plugins::algorithms::GraphAlgorithm for $struct_name {
            fn name(&self) -> &str {
                $name
            }

            fn description(&self) -> &str {
                $desc
            }

            fn parameters(&self) -> &[super::super::ParameterDef] {
                $params_fn()
            }

            fn execute(
                &self,
                $store: &dyn grafeo_core::graph::GraphStore,
                $params: &super::super::Parameters,
            ) -> grafeo_common::utils::error::Result<super::super::AlgorithmResult> {
                $body
            }
        }
    };
}

pub(crate) use impl_algorithm;

// ============================================================================
// Result Builders
// ============================================================================

/// Helper for building algorithm results with node ID and value columns.
pub struct NodeValueResultBuilder {
    node_ids: Vec<u64>,
    values: Vec<Value>,
    value_column_name: String,
}

impl NodeValueResultBuilder {
    /// Creates a new builder with pre-allocated capacity.
    pub fn with_capacity(value_column_name: impl Into<String>, capacity: usize) -> Self {
        Self {
            node_ids: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            value_column_name: value_column_name.into(),
        }
    }

    /// Adds a node ID and value pair.
    pub fn push(&mut self, node: NodeId, value: impl Into<Value>) {
        self.node_ids.push(node.0);
        self.values.push(value.into());
    }

    /// Builds the algorithm result.
    pub fn build(self) -> AlgorithmResult {
        let mut result = AlgorithmResult::new(vec!["node_id".to_string(), self.value_column_name]);
        for (node_id, value) in self.node_ids.into_iter().zip(self.values) {
            // reason: node IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            result.add_row(vec![Value::Int64(node_id as i64), value]);
        }
        result
    }
}

/// Helper for building algorithm results with component information.
pub struct ComponentResultBuilder {
    node_ids: Vec<u64>,
    component_ids: Vec<u64>,
}

impl ComponentResultBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            node_ids: Vec::new(),
            component_ids: Vec::new(),
        }
    }

    /// Creates a new builder with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            node_ids: Vec::with_capacity(capacity),
            component_ids: Vec::with_capacity(capacity),
        }
    }

    /// Adds a node ID and component ID pair.
    pub fn push(&mut self, node: NodeId, component: u64) {
        self.node_ids.push(node.0);
        self.component_ids.push(component);
    }

    /// Builds the algorithm result.
    pub fn build(self) -> AlgorithmResult {
        let mut result =
            AlgorithmResult::new(vec!["node_id".to_string(), "component_id".to_string()]);
        for (node_id, component_id) in self.node_ids.into_iter().zip(self.component_ids) {
            // reason: node/component IDs are sequential counters, well within i64::MAX
            #[allow(clippy::cast_possible_wrap)]
            result.add_row(vec![
                Value::Int64(node_id as i64),
                Value::Int64(component_id as i64),
            ]);
        }
        result
    }
}

impl Default for ComponentResultBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{EdgeId, NodeId};

    use super::*;

    // ---- Control ----

    #[test]
    fn control_is_continue() {
        assert!(Control::<()>::Continue.is_continue());
        assert!(!Control::<()>::Prune.is_continue());
        assert!(!Control::Break(42).is_continue());
    }

    #[test]
    fn control_is_break() {
        assert!(Control::Break("stop").is_break());
        assert!(!Control::<()>::Continue.is_break());
        assert!(!Control::<()>::Prune.is_break());
    }

    #[test]
    fn control_is_prune() {
        assert!(Control::<()>::Prune.is_prune());
        assert!(!Control::<()>::Continue.is_prune());
        assert!(!Control::Break(0u8).is_prune());
    }

    #[test]
    fn control_break_value() {
        assert_eq!(Control::Break(99u32).break_value(), Some(99u32));
        assert_eq!(Control::<u32>::Continue.break_value(), None);
        assert_eq!(Control::<u32>::Prune.break_value(), None);
    }

    #[test]
    fn control_default_is_continue() {
        let c: Control<()> = Default::default();
        assert!(c.is_continue());
    }

    // ---- TraversalEvent ----

    #[test]
    fn traversal_event_discover_source_and_target() {
        let n = NodeId(7);
        let ev = TraversalEvent::Discover(n);
        assert_eq!(ev.source(), Some(n));
        assert_eq!(ev.target(), Some(n));
    }

    #[test]
    fn traversal_event_finish_source_and_target() {
        let n = NodeId(3);
        let ev = TraversalEvent::Finish(n);
        assert_eq!(ev.source(), Some(n));
        assert_eq!(ev.target(), Some(n));
    }

    #[test]
    fn traversal_event_tree_edge() {
        let src = NodeId(1);
        let dst = NodeId(2);
        let edge = EdgeId(10);
        let ev = TraversalEvent::TreeEdge {
            source: src,
            target: dst,
            edge,
        };
        assert_eq!(ev.source(), Some(src));
        assert_eq!(ev.target(), Some(dst));
    }

    #[test]
    fn traversal_event_non_tree_edge() {
        let src = NodeId(1);
        let dst = NodeId(3);
        let edge = EdgeId(20);
        let ev = TraversalEvent::NonTreeEdge {
            source: src,
            target: dst,
            edge,
        };
        assert_eq!(ev.source(), Some(src));
        assert_eq!(ev.target(), Some(dst));
    }

    #[test]
    fn traversal_event_back_edge() {
        let src = NodeId(5);
        let dst = NodeId(1);
        let edge = EdgeId(30);
        let ev = TraversalEvent::BackEdge {
            source: src,
            target: dst,
            edge,
        };
        assert_eq!(ev.source(), Some(src));
        assert_eq!(ev.target(), Some(dst));
    }

    // ---- MinScored ----

    #[test]
    fn min_scored_ordering_is_reversed() {
        // Lower score = higher priority in BinaryHeap (min-heap)
        let a = MinScored::new(1.0f64, "a");
        let b = MinScored::new(5.0f64, "b");
        assert!(a > b, "lower score should sort as Greater for min-heap");
    }

    #[test]
    fn min_scored_equal_scores() {
        let a = MinScored::new(3.0f64, "x");
        let b = MinScored::new(3.0f64, "y");
        assert_eq!(a, b);
    }

    #[test]
    fn min_scored_accessors() {
        let ms = MinScored::new(42.0f64, NodeId(7));
        assert_eq!(*ms.score(), 42.0);
        assert_eq!(*ms.value(), NodeId(7));
        assert_eq!(ms.into_value(), NodeId(7));
    }

    // ---- DistanceMap ----

    #[test]
    fn distance_map_contains_via_default_impl() {
        let mut map: std::collections::HashMap<NodeId, f64> = std::collections::HashMap::new();
        let n = NodeId(1);
        assert!(!map.contains(n));
        map.insert(n, 1.5);
        assert!(map.contains(n));
    }

    // ---- NodeValueResultBuilder ----

    #[test]
    fn node_value_result_builder_basic() {
        let mut b = NodeValueResultBuilder::with_capacity("score", 2);
        b.push(NodeId(1), Value::Float64(0.9));
        b.push(NodeId(2), Value::Float64(0.5));
        let result = b.build();
        assert_eq!(result.columns, vec!["node_id", "score"]);
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn node_value_result_builder_empty() {
        let b = NodeValueResultBuilder::with_capacity("val", 0);
        let result = b.build();
        assert!(result.rows.is_empty(), "{:?}", result.rows);
    }

    // ---- ComponentResultBuilder ----

    #[test]
    fn component_result_builder_basic() {
        let mut b = ComponentResultBuilder::new();
        b.push(NodeId(10), 0);
        b.push(NodeId(20), 1);
        b.push(NodeId(30), 0);
        let result = b.build();
        assert_eq!(result.columns, vec!["node_id", "component_id"]);
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn component_result_builder_with_capacity() {
        let mut b = ComponentResultBuilder::with_capacity(5);
        b.push(NodeId(1), 0);
        let result = b.build();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn component_result_builder_default() {
        let b: ComponentResultBuilder = Default::default();
        let result = b.build();
        assert!(result.rows.is_empty(), "{:?}", result.rows);
    }
}
