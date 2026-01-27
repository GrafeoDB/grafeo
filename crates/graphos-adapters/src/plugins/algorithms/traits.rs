//! Core traits and types for graph algorithms.
//!
//! This module provides the fundamental building blocks for implementing
//! high-performance graph algorithms, inspired by rustworkx patterns.

use graphos_common::types::{EdgeId, NodeId, Value};
use graphos_common::utils::error::Result;
use graphos_core::graph::lpg::LpgStore;
use std::cmp::Ordering;
use std::collections::HashMap;

use super::super::{AlgorithmResult, ParameterDef, Parameters};

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

impl<K: PartialOrd, T> PartialOrd for MinScored<K, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap
        other.0.partial_cmp(&self.0)
    }
}

impl<K: PartialOrd, T> Ord for MinScored<K, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// ============================================================================
// Graph Algorithm Traits
// ============================================================================

/// A graph algorithm that can be executed on an LPG store.
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
    fn execute(&self, store: &LpgStore, params: &Parameters) -> Result<AlgorithmResult>;
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
    fn execute_parallel(
        &self,
        store: &LpgStore,
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

impl<K> DistanceMap<K> for HashMap<NodeId, K> {
    fn get(&self, node: NodeId) -> Option<&K> {
        HashMap::get(self, &node)
    }

    fn insert(&mut self, node: NodeId, dist: K) {
        HashMap::insert(self, node, dist);
    }
}

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
    /// Creates a new builder with the given value column name.
    pub fn new(value_column_name: impl Into<String>) -> Self {
        Self {
            node_ids: Vec::new(),
            values: Vec::new(),
            value_column_name: value_column_name.into(),
        }
    }

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
