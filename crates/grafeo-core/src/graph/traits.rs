//! Storage traits for the graph engine.
//!
//! These traits capture the minimal surface that query operators need from
//! the graph store. The split is intentional:
//!
//! - [`GraphStore`]: Read-only operations (scans, lookups, traversal, statistics)
//! - [`GraphStoreMut`]: Write operations (create, delete, mutate)
//!
//! Admin operations (index management, MVCC internals, schema introspection,
//! statistics recomputation, WAL recovery) stay on the concrete store (the
//! row-group store) and are not part of these traits.
//!
//! ## Design rationale
//!
//! The traits work with typed graph objects (`Node`, `Edge`, `Value`) rather
//! than raw bytes. This preserves zero-overhead access for in-memory storage
//! while allowing future backends (SpilloverStore, disk-backed) to implement
//! the same interface with transparent serialization where needed.

use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
#[cfg(feature = "vector-index")]
use crate::index::vector::DistanceMetric;
use crate::statistics::Statistics;
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, ReadView, TransactionId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;

/// The labels of a node, as [`GraphStore::node_labels`] returns them.
pub type Labels = grafeo_common::change::Labels;

/// The reads of one graph, each at an explicit [`ReadView`].
///
/// Every read names who reads: a transaction at its snapshot, which sees
/// its own writes, or a committed reader at an epoch. The store answers as
/// of that view, so a read never depends on what the store happens to hold
/// between two calls. Every read returns a `Result`: a store that cannot
/// read a value (a cold chunk it cannot decode, a file it cannot read, a
/// remote store that does not answer) reports an error rather than an
/// absent value.
///
/// # What the engine relies on
///
/// - The view decides existence, values, labels and adjacency alike: a
///   node the view does not see has no values, labels or edges for it.
/// - [`edges`](Self::edges) lists only the edges the view sees whose other
///   end it sees too, and lists a self-loop once with [`Direction::Both`].
/// - Edge types and labels are matched exactly, as stored: the engine
///   resolves a query's names to the stored ones before it reads.
/// - The defaults are exact by construction: each answers what the required
///   reads answer, only slower. A store overrides them to answer faster,
///   never differently. A default that cannot answer returns an error.
///
/// # Object safety
///
/// The trait is object-safe: the engine holds a store as
/// `Arc<dyn GraphStore>`. Reads return `Vec`s rather than iterators for
/// that reason.
pub trait GraphStore: Send + Sync {
    // --- Point reads ---

    /// Node `id` as `view` sees it: `None` when the view does not see it.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node(&self, id: NodeId, view: ReadView) -> Result<Option<Node>>;

    /// Edge `id` as `view` sees it: `None` when the view does not see it.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn edge(&self, id: EdgeId, view: ReadView) -> Result<Option<Edge>>;

    /// Whether `view` sees node `id`.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn is_node_visible(&self, id: NodeId, view: ReadView) -> Result<bool> {
        Ok(self.node(id, view)?.is_some())
    }

    /// Whether `view` sees edge `id`.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn is_edge_visible(&self, id: EdgeId, view: ReadView) -> Result<bool> {
        Ok(self.edge(id, view)?.is_some())
    }

    /// The nodes of `ids` that `view` sees, in their order.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn filter_visible_node_ids(&self, ids: &[NodeId], view: ReadView) -> Result<Vec<NodeId>> {
        let mut seen = Vec::with_capacity(ids.len());
        for &id in ids {
            if self.is_node_visible(id, view)? {
                seen.push(id);
            }
        }
        Ok(seen)
    }

    /// The value of property `key` of node `id` as `view` sees the node:
    /// `None` when the view does not see it or it has no such value.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_property(&self, id: NodeId, key: &str, view: ReadView) -> Result<Option<Value>> {
        Ok(self
            .node(id, view)?
            .and_then(|node| node.get_property(key).cloned()))
    }

    /// The value of property `key` of edge `id` as `view` sees the edge
    /// (see [`node_property`](Self::node_property)).
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn edge_property(&self, id: EdgeId, key: &str, view: ReadView) -> Result<Option<Value>> {
        Ok(self
            .edge(id, view)?
            .and_then(|edge| edge.get_property(key).cloned()))
    }

    /// Calls `f` with the vector in property `key` of node `id` as `view`
    /// sees the node, and returns whether there was one. Vector search reads
    /// every distance through this, so a store lends the vector without
    /// copying it where it can.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn with_node_vector(
        &self,
        id: NodeId,
        key: &str,
        view: ReadView,
        f: &mut dyn FnMut(&[f32]),
    ) -> Result<bool> {
        match self.node_property(id, key, view)? {
            Some(Value::Vector(vector)) => {
                f(&vector);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// The labels of node `id` as `view` sees the node, `None` when the view
    /// does not see it.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_labels(&self, id: NodeId, view: ReadView) -> Result<Option<Labels>> {
        Ok(self.node(id, view)?.map(|node| node.labels))
    }

    /// Whether node `id` has `label` as `view` sees the node, `None` when
    /// the view does not see it.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_has_label(&self, id: NodeId, label: &str, view: ReadView) -> Result<Option<bool>> {
        Ok(self
            .node_labels(id, view)?
            .map(|labels| labels.iter().any(|name| name.as_str() == label)))
    }

    /// The type of edge `id` as `view` sees the edge, `None` when the view
    /// does not see it.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn edge_type(&self, id: EdgeId, view: ReadView) -> Result<Option<ArcStr>> {
        Ok(self.edge(id, view)?.map(|edge| edge.edge_type))
    }

    // --- Batch reads ---

    /// The value of `key` of each node of `ids` as `view` sees it, in their
    /// order: `None` for a node the view does not see or without the value.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
        view: ReadView,
    ) -> Result<Vec<Option<Value>>> {
        ids.iter()
            .map(|&id| self.node_property(id, key.as_str(), view))
            .collect()
    }

    /// Every value of each node of `ids` as `view` sees it, in their order:
    /// no values for a node the view does not see.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_properties_batch(
        &self,
        ids: &[NodeId],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        ids.iter()
            .map(|&id| {
                Ok(self
                    .node(id, view)?
                    .map(|node| node.properties.iter().map(clone_pair).collect())
                    .unwrap_or_default())
            })
            .collect()
    }

    /// The values of `keys` of each node of `ids` as `view` sees it, in
    /// their order (projection pushdown): no values for a node the view does
    /// not see.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        ids.iter()
            .map(|&id| {
                let mut values = FxHashMap::default();
                for key in keys {
                    if let Some(value) = self.node_property(id, key.as_str(), view)? {
                        values.insert(key.clone(), value);
                    }
                }
                Ok(values)
            })
            .collect()
    }

    /// The values of `keys` of each edge of `ids` as `view` sees it, in
    /// their order: no values for an edge the view does not see.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn edge_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Result<Vec<FxHashMap<PropertyKey, Value>>> {
        ids.iter()
            .map(|&id| {
                let mut values = FxHashMap::default();
                for key in keys {
                    if let Some(value) = self.edge_property(id, key.as_str(), view)? {
                        values.insert(key.clone(), value);
                    }
                }
                Ok(values)
            })
            .collect()
    }

    // --- Traversal ---

    /// The edges of `node` in `direction` of one of `types` (every type when
    /// empty) that `view` sees and whose other end it sees too, each as
    /// (other node, edge). [`Direction::Both`] lists a self-loop once. Types
    /// match exactly. Empty for a node the view does not see.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn edges(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        view: ReadView,
    ) -> Result<Vec<(NodeId, EdgeId)>>;

    /// How many edges [`edges`](Self::edges) lists for `node` in `direction`
    /// of every type.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn degree(&self, node: NodeId, direction: Direction, view: ReadView) -> Result<usize> {
        Ok(self.edges(node, direction, &[], view)?.len())
    }

    /// The nodes at the other end of the edges [`edges`](Self::edges) lists
    /// for `node` in `direction` of every type, in its order.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn neighbors(&self, node: NodeId, direction: Direction, view: ReadView) -> Result<Vec<NodeId>> {
        Ok(self
            .edges(node, direction, &[], view)?
            .into_iter()
            .map(|(other, _)| other)
            .collect())
    }

    /// Whether the store keeps each node's incoming edges: without them an
    /// incoming read is a scan, which a planner avoids. Not an answer: the
    /// reads answer alike either way.
    fn has_backward_adjacency(&self) -> bool;

    // --- Scans ---

    /// The nodes `view` sees, by ascending id.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_ids(&self, view: ReadView) -> Result<Vec<NodeId>>;

    /// The nodes with `label` that `view` sees, by ascending id.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn nodes_by_label(&self, label: &str, view: ReadView) -> Result<Vec<NodeId>> {
        let mut found = Vec::new();
        for id in self.node_ids(view)? {
            if self.node_has_label(id, label, view)? == Some(true) {
                found.push(id);
            }
        }
        Ok(found)
    }

    /// The nodes with at least one of `labels` that `view` sees, each once
    /// and by ascending id.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn nodes_with_any_label(&self, labels: &[&str], view: ReadView) -> Result<Vec<NodeId>> {
        let mut ids = Vec::new();
        for label in labels {
            ids.extend(self.nodes_by_label(label, view)?);
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    /// How many nodes with `label` `view` sees.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn nodes_by_label_count(&self, label: &str, view: ReadView) -> Result<usize> {
        Ok(self.nodes_by_label(label, view)?.len())
    }

    /// How many nodes `view` sees.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_count(&self, view: ReadView) -> Result<usize> {
        Ok(self.node_ids(view)?.len())
    }

    /// How many edges `view` sees: those [`edges`](Self::edges) lists from
    /// their source.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn edge_count(&self, view: ReadView) -> Result<usize> {
        let mut count = 0;
        for node in self.node_ids(view)? {
            count += self.degree(node, Direction::Outgoing, view)?;
        }
        Ok(count)
    }

    // --- Names ---

    /// Every label name the store holds: at least each one a view sees, and
    /// maybe names no node has any more.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn all_labels(&self) -> Result<Vec<String>>;

    /// Every edge type name the store holds (see
    /// [`all_labels`](Self::all_labels)). The engine resolves a query's edge
    /// types against it.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn all_edge_types(&self) -> Result<Vec<String>>;

    /// Every property key the store holds, of nodes and edges (see
    /// [`all_labels`](Self::all_labels)).
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn all_property_keys(&self) -> Result<Vec<String>>;

    // --- Lookups ---

    /// Whether the store keeps a property index on `property`: the planner
    /// then prefers a lookup to a scan. Not an answer: the lookups answer
    /// alike either way.
    fn has_property_index(&self, _property: &str) -> bool {
        false
    }

    /// The nodes `view` sees whose `property` equals `value`, by ascending
    /// id.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_nodes_by_property(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        self.find_nodes_by_properties(&[(property, value.clone())], view)
    }

    /// The nodes `view` sees whose values equal every one of `conditions`,
    /// by ascending id.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_nodes_by_properties(
        &self,
        conditions: &[(&str, Value)],
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        let mut found = Vec::new();
        'nodes: for id in self.node_ids(view)? {
            for (key, value) in conditions {
                if self.node_property(id, key, view)?.as_ref() != Some(value) {
                    continue 'nodes;
                }
            }
            found.push(id);
        }
        Ok(found)
    }

    /// The nodes whose `property` may be equal to `value` under `=` (the
    /// filter's equality, see
    /// [`HashKey::for_equality`](crate::execution::operators::HashKey::for_equality)),
    /// found through the property's index: every node `view` sees that `=`
    /// finds equal, and maybe others it sees, for a filter to decide. `None`
    /// without an index on `property`: the caller then scans.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_nodes_maybe_equal(
        &self,
        _property: &str,
        _value: &Value,
        _view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        Ok(None)
    }

    /// The nodes `view` sees whose `property` lies within the bounds, by
    /// ascending id. Values compare as a filter compares them: integers,
    /// floats and decimals by value, strings, booleans and temporal values
    /// each among their kind.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        let mut found = Vec::new();
        for id in self.node_ids(view)? {
            if let Some(value) = self.node_property(id, property, view)?
                && value_in_range(&value, min, max, min_inclusive, max_inclusive)
            {
                found.push(id);
            }
        }
        Ok(found)
    }

    /// Whether a node's `property` may meet `op value` for some view: `false`
    /// lets the planner skip a scan that finds nothing. Not an answer: `true`
    /// is always correct.
    fn node_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        true
    }

    /// As [`node_property_might_match`](Self::node_property_might_match),
    /// for edges.
    fn edge_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        true
    }

    // --- Statistics ---

    /// The statistics the cost-based optimizer plans with (a cheap `Arc`
    /// clone): estimates, not answers.
    fn statistics(&self) -> Arc<Statistics>;

    /// The latest epoch the store holds commits of. A database that adopts
    /// the store continues its epochs from it; the engine reads at the views
    /// of its own transactions, never at this epoch.
    fn committed_epoch(&self) -> EpochId;

    // --- History ---

    /// The versions of node `id` the store keeps, newest first: each with
    /// the epoch it began at, the epoch the next one or the delete ended it
    /// at (`None` for the one a reader sees now), and the node as a reader
    /// at its first epoch sees it.
    ///
    /// # Errors
    ///
    /// The default returns an unsupported error: a store without history
    /// says so rather than reporting none.
    fn node_history(&self, _id: NodeId) -> Result<Vec<(EpochId, Option<EpochId>, Node)>> {
        Err(unsupported("node history"))
    }

    /// The versions of edge `id` the store keeps (see
    /// [`node_history`](Self::node_history)).
    ///
    /// # Errors
    ///
    /// The default returns an unsupported error.
    fn edge_history(&self, _id: EdgeId) -> Result<Vec<(EpochId, Option<EpochId>, Edge)>> {
        Err(unsupported("edge history"))
    }

    // --- Text search (BM25) ---

    /// Whether the store keeps a text index on `label` and `property`. The
    /// text reads below answer through one; a store without text indexes
    /// keeps the defaults, which answer as a store without the index does.
    #[cfg(feature = "text-index")]
    fn has_text_index(&self, _label: &str, _property: &str) -> bool {
        false
    }

    /// The BM25 score of node `id`'s text in `property` for `query`, as
    /// `view` sees the node: 0 for a node the view does not see, `None`
    /// without a text index on `label` and `property`.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    #[cfg(feature = "text-index")]
    fn score_text(
        &self,
        _id: NodeId,
        _label: &str,
        _property: &str,
        _query: &str,
        _view: ReadView,
    ) -> Result<Option<f64>> {
        Ok(None)
    }

    /// The `k` documents `view` sees with the best BM25 scores for `query`,
    /// best first: empty without a text index on `label` and `property`.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        _label: &str,
        _property: &str,
        _query: &str,
        _k: usize,
        _view: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        Ok(Vec::new())
    }

    /// Every document `view` sees whose BM25 score for `query` is at least
    /// `threshold`, best first: empty without a text index.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        _label: &str,
        _property: &str,
        _query: &str,
        _threshold: f64,
        _view: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        Ok(Vec::new())
    }

    // --- Vector search ---

    /// Whether the store keeps a vector index on `label` and `property`.
    #[cfg(feature = "vector-index")]
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        self.vector_index_config(label, property).is_some()
    }

    /// The configuration of the vector index on `label` and `property`, if
    /// any: an index whose metric differs from a query's cannot serve it.
    #[cfg(feature = "vector-index")]
    fn vector_index_config(
        &self,
        _label: &str,
        _property: &str,
    ) -> Option<crate::index::vector::HnswConfig> {
        None
    }

    /// The `k` nodes `view` sees (with `label`, any node for `None`) whose
    /// vector in `property` is nearest to `query` by `metric`, nearest
    /// first, equal distances by ascending id. A vector of another length
    /// has no distance and is left out. The default measures every vector.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        view: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        let mut found = measured_vectors(self, label, property, query, metric, view)?;
        found.sort_by(crate::index::ranking::by_ascending_distance);
        found.truncate(k);
        Ok(found)
    }

    /// Every node `view` sees (with `label`) whose vector in `property` is
    /// within `threshold` of `query` by `metric`, nearest first (see
    /// [`vector_search`](Self::vector_search)).
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
        view: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        let mut found = measured_vectors(self, label, property, query, metric, view)?;
        found.retain(|(_, distance)| *distance <= threshold);
        found.sort_by(crate::index::ranking::by_ascending_distance);
        Ok(found)
    }
}

/// Clones a value pair of a property map.
fn clone_pair((key, value): (&PropertyKey, &Value)) -> (PropertyKey, Value) {
    (key.clone(), value.clone())
}

/// The error a read the store does not support returns.
fn unsupported(what: &str) -> grafeo_common::utils::error::Error {
    grafeo_common::utils::error::Error::Query(grafeo_common::utils::error::QueryError::unsupported(
        format!("this store does not support {what}"),
    ))
}

/// The distance from `query` of each vector in `property` of the nodes
/// `view` sees (with `label`), one with another length left out.
#[cfg(feature = "vector-index")]
fn measured_vectors<S: GraphStore + ?Sized>(
    store: &S,
    label: Option<&str>,
    property: &str,
    query: &[f32],
    metric: DistanceMetric,
    view: ReadView,
) -> Result<Vec<(NodeId, f64)>> {
    let ids = match label {
        Some(label) => store.nodes_by_label(label, view)?,
        None => store.node_ids(view)?,
    };
    let mut found = Vec::new();
    for id in ids {
        let mut distance = None;
        store.with_node_vector(id, property, view, &mut |vector| {
            if vector.len() == query.len() {
                distance = Some(f64::from(crate::index::vector::compute_distance(
                    query, vector, metric,
                )));
            }
        })?;
        if let Some(distance) = distance {
            found.push((id, distance));
        }
    }
    Ok(found)
}

/// Keeps one entry for each self-loop of `node` among `edges`, its edges in
/// both directions: a self-loop is in both adjacency lists of its node, and
/// [`GraphStore::edges`] lists it once, as an undirected pattern matches it
/// once (openCypher TCK Match3, cases 11 and 12). A store implementing
/// [`GraphStore::edges`] calls it on what it lists for `Both`.
pub fn list_self_loops_once(node: NodeId, edges: &mut Vec<(NodeId, EdgeId)>) {
    let mut loops: SmallVec<[EdgeId; 2]> = SmallVec::new();
    edges.retain(|&(other, edge)| {
        if other != node {
            return true;
        }
        if loops.contains(&edge) {
            return false;
        }
        loops.push(edge);
        true
    });
}

/// Compares two values for a range lookup, as a filter compares them.
pub(crate) fn compare_for_range(a: &Value, b: &Value) -> Option<CmpOrdering> {
    match (a, b) {
        // Decimals by value, also against integers and floats, as a filter
        // compares them (#610).
        (Value::Decimal(_), _) | (_, Value::Decimal(_)) => a.compare_decimal(b),
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        // An integer and a float compare exactly, as a filter does.
        (Value::Int64(a), Value::Float64(b)) => {
            crate::execution::operators::value_utils::partial_cmp_int_float(*a, *b)
        }
        (Value::Float64(a), Value::Int64(b)) => {
            crate::execution::operators::value_utils::partial_cmp_int_float(*b, *a)
                .map(CmpOrdering::reverse)
        }
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        _ => a.compare_instants(b),
    }
}

/// Whether `value` lies between the bounds of a range lookup.
pub(crate) fn value_in_range(
    value: &Value,
    min: Option<&Value>,
    max: Option<&Value>,
    min_inclusive: bool,
    max_inclusive: bool,
) -> bool {
    let above = min.is_none_or(|min| match compare_for_range(value, min) {
        Some(CmpOrdering::Greater) => true,
        Some(CmpOrdering::Equal) => min_inclusive,
        _ => false,
    });
    let below = max.is_none_or(|max| match compare_for_range(value, max) {
        Some(CmpOrdering::Less) => true,
        Some(CmpOrdering::Equal) => max_inclusive,
        _ => false,
    });
    above && below
}

/// Write operations for graph mutation.
///
/// Separated from [`GraphStore`] so read-only wrappers (snapshots, read
/// replicas) can implement only `GraphStore`. Any mutable store is also
/// readable via the supertrait bound.
pub trait GraphStoreMut: GraphStore {
    // --- Node creation ---

    /// Creates a new node with the given labels.
    fn create_node(&self, labels: &[&str]) -> NodeId;

    /// Creates a new node within a transaction context.
    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId;

    // --- Edge creation ---

    /// Creates a new edge between two nodes.
    ///
    /// The store does not check the endpoints: the caller makes sure both
    /// exist (the engine's writes go through
    /// [`create_edge_versioned`](Self::create_edge_versioned)).
    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId;

    /// Creates a new edge within a transaction context
    /// (`TransactionId::SYSTEM` outside one).
    ///
    /// The caller checks that its transaction sees both endpoints, and claims
    /// them against a concurrent delete (as `GraphWriter` does).
    ///
    /// # Errors
    ///
    /// Returns a write conflict, and creates nothing, when the store knows an
    /// endpoint is deleted, by a committed delete or one in progress: a
    /// compacted store's base node with a tombstone.
    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Result<EdgeId>;

    /// Creates multiple edges in batch (single lock acquisition). The
    /// endpoints are not checked, as for [`create_edge`](Self::create_edge).
    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId>;

    // --- Deletion ---

    /// Deletes a node. Returns `true` if the node existed.
    fn delete_node(&self, id: NodeId) -> bool;

    /// Deletes a node within a transaction context. Returns `true` if the node existed.
    ///
    /// # Errors
    ///
    /// Returns an error, and deletes nothing, when the store cannot read the
    /// node it would delete (a record or value it cannot read).
    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Result<bool>;

    /// Deletes all edges connected to a node (DETACH DELETE).
    fn delete_node_edges(&self, node_id: NodeId);

    /// Deletes an edge. Returns `true` if the edge existed.
    fn delete_edge(&self, id: EdgeId) -> bool;

    /// Deletes an edge within a transaction context. Returns `true` if the edge existed.
    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool;

    // --- Property mutation ---

    /// Sets a property on a node.
    fn set_node_property(&self, id: NodeId, key: &str, value: Value);

    /// Sets a property on an edge.
    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value);

    /// Sets a node property within a transaction context: a store that keeps
    /// versions writes the value as the transaction's.
    ///
    /// Default delegates to [`set_node_property`](Self::set_node_property).
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the previous value cannot
    /// be read (a spilled value whose file cannot be read): a rollback could
    /// not restore it.
    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        _transaction_id: TransactionId,
    ) -> Result<()> {
        self.set_node_property(id, key, value);
        Ok(())
    }

    /// Sets an edge property within a transaction context.
    ///
    /// Default delegates to [`set_edge_property`](Self::set_edge_property).
    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        _transaction_id: TransactionId,
    ) {
        self.set_edge_property(id, key, value);
    }

    /// Removes a property from a node. Returns the previous value if it existed.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the value cannot be read
    /// (a spilled value whose file cannot be read): the caller logs the
    /// removal, and a hidden value reported absent would not be logged.
    fn remove_node_property(&self, id: NodeId, key: &str) -> Result<Option<Value>>;

    /// Removes a property from an edge. Returns the previous value if it existed.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the value cannot be read,
    /// as [`remove_node_property`](Self::remove_node_property) does.
    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<Option<Value>>;

    /// Removes a node property within a transaction context. Returns the
    /// previous value if it existed.
    ///
    /// Default delegates to [`remove_node_property`](Self::remove_node_property).
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the previous value cannot
    /// be read (a spilled value whose file cannot be read).
    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        _transaction_id: TransactionId,
    ) -> Result<Option<Value>> {
        self.remove_node_property(id, key)
    }

    /// Removes an edge property within a transaction context.
    ///
    /// Default delegates to [`remove_edge_property`](Self::remove_edge_property).
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the previous value cannot
    /// be read.
    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        _transaction_id: TransactionId,
    ) -> Result<Option<Value>> {
        self.remove_edge_property(id, key)
    }

    // --- Label mutation ---

    /// Adds a label to a node. Returns `true` if the label was new.
    fn add_label(&self, node_id: NodeId, label: &str) -> bool;

    /// Removes a label from a node. Returns `true` if the label existed.
    fn remove_label(&self, node_id: NodeId, label: &str) -> bool;

    /// Adds a label within a transaction context.
    ///
    /// Default delegates to [`add_label`](Self::add_label).
    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        _transaction_id: TransactionId,
    ) -> bool {
        self.add_label(node_id, label)
    }

    /// Removes a label within a transaction context.
    ///
    /// Default delegates to [`remove_label`](Self::remove_label).
    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        _transaction_id: TransactionId,
    ) -> bool {
        self.remove_label(node_id, label)
    }

    // --- Convenience (with default implementations) ---

    /// Creates a new node with labels and properties in one call.
    ///
    /// The default implementation calls [`create_node`](Self::create_node)
    /// followed by [`set_node_property`](Self::set_node_property) for each
    /// property. Implementations may override for atomicity or performance.
    fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: &[(PropertyKey, Value)],
    ) -> NodeId {
        let id = self.create_node(labels);
        for (key, value) in properties {
            self.set_node_property(id, key.as_str(), value.clone());
        }
        id
    }

    /// Creates a new edge with properties in one call.
    ///
    /// The default implementation calls [`create_edge`](Self::create_edge)
    /// followed by [`set_edge_property`](Self::set_edge_property) for each
    /// property. Implementations may override for atomicity or performance.
    fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(PropertyKey, Value)],
    ) -> EdgeId {
        let id = self.create_edge(src, dst, edge_type);
        for (key, value) in properties {
            self.set_edge_property(id, key.as_str(), value.clone());
        }
        id
    }
}

/// A no-op [`GraphStore`] that returns empty results for all queries.
///
/// Used by the RDF planner to satisfy the expression evaluator's store
/// requirement. SPARQL expression functions (STR, LANG, DATATYPE, etc.)
/// operate on already-materialized values in DataChunk columns and never
/// call store methods.
pub struct NullGraphStore;

impl GraphStore for NullGraphStore {
    fn committed_epoch(&self) -> EpochId {
        EpochId::INITIAL
    }
    fn node(&self, _: NodeId, _: ReadView) -> Result<Option<Node>> {
        Ok(None)
    }
    fn edge(&self, _: EdgeId, _: ReadView) -> Result<Option<Edge>> {
        Ok(None)
    }
    fn edges(
        &self,
        _: NodeId,
        _: Direction,
        _: &[ArcStr],
        _: ReadView,
    ) -> Result<Vec<(NodeId, EdgeId)>> {
        Ok(Vec::new())
    }
    fn has_backward_adjacency(&self) -> bool {
        false
    }
    fn node_ids(&self, _: ReadView) -> Result<Vec<NodeId>> {
        Ok(Vec::new())
    }
    fn all_labels(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn all_edge_types(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn all_property_keys(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn statistics(&self) -> Arc<Statistics> {
        Arc::new(Statistics::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A view of a committed reader, for the stores below that keep no
    /// versions.
    const NOW: ReadView = ReadView::committed(EpochId(0));

    #[test]
    fn the_null_store_sees_nothing() {
        let store = NullGraphStore;
        let (node, edge) = (NodeId(1), EdgeId(1));
        let key = PropertyKey::from("name");
        assert!(GraphStore::node(&store, node, NOW).unwrap().is_none());
        assert!(GraphStore::edge(&store, edge, NOW).unwrap().is_none());
        assert!(!GraphStore::is_node_visible(&store, node, NOW).unwrap());
        assert!(!GraphStore::is_edge_visible(&store, edge, NOW).unwrap());
        assert_eq!(
            GraphStore::node_property_batch(&store, &[node, NodeId(2)], &key, NOW).unwrap(),
            vec![None, None]
        );
        let edges = GraphStore::edges(&store, node, Direction::Both, &[], NOW).unwrap();
        assert!(edges.is_empty(), "{edges:?}");
        assert_eq!(
            GraphStore::degree(&store, node, Direction::Outgoing, NOW).unwrap(),
            0
        );
        assert!(!GraphStore::has_backward_adjacency(&store));
        let ids = GraphStore::node_ids(&store, NOW).unwrap();
        assert!(ids.is_empty(), "{ids:?}");
        assert_eq!(GraphStore::node_count(&store, NOW).unwrap(), 0);
        assert_eq!(GraphStore::edge_count(&store, NOW).unwrap(), 0);
        for names in [
            GraphStore::all_labels(&store).unwrap(),
            GraphStore::all_edge_types(&store).unwrap(),
            GraphStore::all_property_keys(&store).unwrap(),
        ] {
            assert!(names.is_empty(), "{names:?}");
        }
        assert!(!GraphStore::has_property_index(&store, "name"));
        assert_eq!(
            GraphStore::find_nodes_maybe_equal(&store, "name", &Value::from("Gus"), NOW).unwrap(),
            None
        );
    }

    /// A store without history says so: an error, not an empty history.
    #[test]
    fn a_store_without_history_reports_it_unsupported() {
        let store = NullGraphStore;
        let error = GraphStore::node_history(&store, NodeId(1)).unwrap_err();
        assert!(error.to_string().contains("node history"), "{error}");
        assert!(GraphStore::edge_history(&store, EdgeId(1)).is_err());
    }

    /// A store that implements only the required reads, over fixed nodes
    /// and edges every view sees: the defaults must answer what the
    /// required reads answer.
    struct Required {
        nodes: Vec<Node>,
        edges: Vec<Edge>,
    }

    impl Required {
        fn new() -> Self {
            let person = |id: u64, name: &str, age: i64| {
                let mut node = Node::new(NodeId(id));
                node.add_label("Person");
                node.set_property("name", Value::from(name));
                node.set_property("age", Value::Int64(age));
                node
            };
            let mut city = Node::new(NodeId(4));
            city.add_label("City");
            city.set_property("name", Value::from("Amsterdam"));
            city.set_property("location", Value::Vector(Arc::from([1.0_f32, 0.0])));
            let mut knows = Edge::new(EdgeId(10), NodeId(1), NodeId(2), "KNOWS");
            knows.set_property("since", Value::Int64(1994));
            Self {
                nodes: vec![
                    person(1, "Alix", 19),
                    person(2, "Gus", 3),
                    person(3, "Mia", 88),
                    city,
                ],
                edges: vec![
                    knows,
                    Edge::new(EdgeId(11), NodeId(2), NodeId(3), "KNOWS"),
                    Edge::new(EdgeId(12), NodeId(1), NodeId(4), "LIVES_IN"),
                    Edge::new(EdgeId(13), NodeId(3), NodeId(3), "KNOWS"),
                ],
            }
        }
    }

    impl GraphStore for Required {
        fn node(&self, id: NodeId, _: ReadView) -> Result<Option<Node>> {
            Ok(self.nodes.iter().find(|node| node.id == id).cloned())
        }
        fn edge(&self, id: EdgeId, _: ReadView) -> Result<Option<Edge>> {
            Ok(self.edges.iter().find(|edge| edge.id == id).cloned())
        }
        fn edges(
            &self,
            node: NodeId,
            direction: Direction,
            types: &[ArcStr],
            _: ReadView,
        ) -> Result<Vec<(NodeId, EdgeId)>> {
            let mut found = Vec::new();
            for edge in &self.edges {
                if !types.is_empty() && !types.contains(&edge.edge_type) {
                    continue;
                }
                if edge.src == node && direction != Direction::Incoming {
                    found.push((edge.dst, edge.id));
                }
                if edge.dst == node && direction != Direction::Outgoing {
                    found.push((edge.src, edge.id));
                }
            }
            if direction == Direction::Both {
                list_self_loops_once(node, &mut found);
            }
            Ok(found)
        }
        fn has_backward_adjacency(&self) -> bool {
            true
        }
        fn node_ids(&self, _: ReadView) -> Result<Vec<NodeId>> {
            Ok(self.nodes.iter().map(|node| node.id).collect())
        }
        fn all_labels(&self) -> Result<Vec<String>> {
            Ok(vec!["Person".into(), "City".into()])
        }
        fn all_edge_types(&self) -> Result<Vec<String>> {
            Ok(vec!["KNOWS".into(), "LIVES_IN".into()])
        }
        fn all_property_keys(&self) -> Result<Vec<String>> {
            Ok(vec!["name".into(), "age".into(), "since".into()])
        }
        fn statistics(&self) -> Arc<Statistics> {
            Arc::new(Statistics::default())
        }
        fn committed_epoch(&self) -> EpochId {
            EpochId::INITIAL
        }
    }

    #[test]
    fn the_defaults_answer_what_the_required_reads_answer() {
        let store = Required::new();
        let (alix, gus, mia, amsterdam) = (NodeId(1), NodeId(2), NodeId(3), NodeId(4));
        assert_eq!(
            store.node_property(alix, "name", NOW).unwrap(),
            Some(Value::from("Alix"))
        );
        assert_eq!(store.node_property(alix, "city", NOW).unwrap(), None);
        assert_eq!(
            store.edge_property(EdgeId(10), "since", NOW).unwrap(),
            Some(Value::Int64(1994))
        );
        assert_eq!(
            store
                .edge_type(EdgeId(12), NOW)
                .unwrap()
                .map(|t| t.to_string()),
            Some("LIVES_IN".to_string())
        );
        assert_eq!(
            store.node_has_label(alix, "Person", NOW).unwrap(),
            Some(true)
        );
        assert_eq!(
            store.node_has_label(alix, "City", NOW).unwrap(),
            Some(false)
        );
        assert_eq!(store.node_has_label(NodeId(9), "City", NOW).unwrap(), None);
        assert_eq!(
            store
                .filter_visible_node_ids(&[NodeId(9), mia, alix], NOW)
                .unwrap(),
            [mia, alix]
        );
        assert_eq!(
            store.nodes_by_label("Person", NOW).unwrap(),
            [alix, gus, mia]
        );
        assert_eq!(
            store
                .nodes_with_any_label(&["City", "Person"], NOW)
                .unwrap(),
            [alix, gus, mia, amsterdam]
        );
        assert_eq!(store.nodes_by_label_count("City", NOW).unwrap(), 1);
        assert_eq!(store.node_count(NOW).unwrap(), 4);
        assert_eq!(store.edge_count(NOW).unwrap(), 4, "the self-loop once");
        assert_eq!(store.degree(mia, Direction::Both, NOW).unwrap(), 2);
        assert_eq!(
            store.neighbors(alix, Direction::Outgoing, NOW).unwrap(),
            [gus, amsterdam]
        );
        assert_eq!(
            store
                .find_nodes_by_property("name", &Value::from("Gus"), NOW)
                .unwrap(),
            [gus]
        );
        assert_eq!(
            store
                .find_nodes_by_properties(
                    &[("name", Value::from("Mia")), ("age", Value::Int64(88))],
                    NOW
                )
                .unwrap(),
            [mia]
        );
        assert_eq!(
            store
                .find_nodes_in_range(
                    "age",
                    Some(&Value::Float64(3.0)),
                    Some(&Value::Int64(88)),
                    false,
                    true,
                    NOW
                )
                .unwrap(),
            [alix, mia],
            "an integer and a float compare by value"
        );
        let selected = store
            .node_properties_selective_batch(&[alix, NodeId(9)], &[PropertyKey::from("age")], NOW)
            .unwrap();
        assert_eq!(selected[0].len(), 1);
        assert!(selected[1].is_empty());
        let all = store.node_properties_batch(&[gus], NOW).unwrap();
        assert_eq!(all[0].len(), 2);
        let mut seen = None;
        assert!(
            store
                .with_node_vector(amsterdam, "location", NOW, &mut |vector| {
                    seen = Some(vector.to_vec());
                })
                .unwrap()
        );
        assert_eq!(seen, Some(vec![1.0, 0.0]));
        assert!(
            !store
                .with_node_vector(alix, "location", NOW, &mut |_| {})
                .unwrap()
        );
    }

    /// Without an index the default vector search measures every vector:
    /// a store answers it, nearest first, never an empty "no match".
    #[cfg(feature = "vector-index")]
    #[test]
    fn the_default_vector_search_measures_every_vector() {
        let store = Required::new();
        let found = store
            .vector_search(
                Some("City"),
                "location",
                &[1.0, 0.0],
                3,
                DistanceMetric::Euclidean,
                NOW,
            )
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, NodeId(4));
        let within = store
            .vector_search_with_threshold(
                None,
                "location",
                &[0.0, 0.0],
                0.5,
                DistanceMetric::Euclidean,
                NOW,
            )
            .unwrap();
        assert!(within.is_empty(), "{within:?}");
    }

    #[test]
    fn a_self_loop_is_listed_once_both_ways() {
        let mut edges = vec![
            (NodeId(3), EdgeId(13)),
            (NodeId(2), EdgeId(11)),
            (NodeId(3), EdgeId(13)),
        ];
        list_self_loops_once(NodeId(3), &mut edges);
        assert_eq!(edges, [(NodeId(3), EdgeId(13)), (NodeId(2), EdgeId(11))]);
    }

    /// Minimal in-memory store used to exercise the default method bodies on
    /// `GraphStoreMut`. Concrete production stores override every default, so
    /// without this harness those default bodies would stay uncovered.
    #[derive(Default)]
    struct TestMutStore {
        inner: Mutex<TestMutInner>,
    }

    #[derive(Default)]
    struct TestMutInner {
        next_node: u64,
        next_edge: u64,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
    }

    impl TestMutStore {
        fn new() -> Self {
            Self::default()
        }

        fn find_node(&self, id: NodeId) -> Option<Node> {
            self.inner
                .lock()
                .unwrap()
                .nodes
                .iter()
                .find(|n| n.id == id)
                .cloned()
        }

        fn find_edge(&self, id: EdgeId) -> Option<Edge> {
            self.inner
                .lock()
                .unwrap()
                .edges
                .iter()
                .find(|e| e.id == id)
                .cloned()
        }
    }

    impl GraphStore for TestMutStore {
        fn node(&self, id: NodeId, _: ReadView) -> Result<Option<Node>> {
            Ok(self.find_node(id))
        }
        fn edge(&self, id: EdgeId, _: ReadView) -> Result<Option<Edge>> {
            Ok(self.find_edge(id))
        }
        fn edges(
            &self,
            _: NodeId,
            _: Direction,
            _: &[ArcStr],
            _: ReadView,
        ) -> Result<Vec<(NodeId, EdgeId)>> {
            Ok(Vec::new())
        }
        fn has_backward_adjacency(&self) -> bool {
            false
        }
        fn node_ids(&self, _: ReadView) -> Result<Vec<NodeId>> {
            Ok(self
                .inner
                .lock()
                .unwrap()
                .nodes
                .iter()
                .map(|n| n.id)
                .collect())
        }
        fn edge_count(&self, _: ReadView) -> Result<usize> {
            Ok(self.inner.lock().unwrap().edges.len())
        }
        fn all_labels(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn all_edge_types(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn all_property_keys(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
        fn statistics(&self) -> Arc<Statistics> {
            Arc::new(Statistics::default())
        }
        fn committed_epoch(&self) -> EpochId {
            EpochId::INITIAL
        }
    }

    impl GraphStoreMut for TestMutStore {
        fn create_node(&self, labels: &[&str]) -> NodeId {
            let mut inner = self.inner.lock().unwrap();
            inner.next_node += 1;
            let id = NodeId(inner.next_node);
            let mut node = Node::new(id);
            for label in labels {
                node.add_label(*label);
            }
            inner.nodes.push(node);
            id
        }
        fn create_node_versioned(&self, labels: &[&str], _: EpochId, _: TransactionId) -> NodeId {
            self.create_node(labels)
        }
        fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
            let mut inner = self.inner.lock().unwrap();
            inner.next_edge += 1;
            let id = EdgeId(inner.next_edge);
            inner.edges.push(Edge::new(id, src, dst, edge_type));
            id
        }
        fn create_edge_versioned(
            &self,
            src: NodeId,
            dst: NodeId,
            edge_type: &str,
            _: EpochId,
            _: TransactionId,
        ) -> Result<EdgeId> {
            Ok(self.create_edge(src, dst, edge_type))
        }
        fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
            edges
                .iter()
                .map(|(s, d, t)| self.create_edge(*s, *d, t))
                .collect()
        }
        fn delete_node(&self, id: NodeId) -> bool {
            let mut inner = self.inner.lock().unwrap();
            if let Some(pos) = inner.nodes.iter().position(|n| n.id == id) {
                inner.nodes.remove(pos);
                true
            } else {
                false
            }
        }
        fn delete_node_versioned(&self, id: NodeId, _: EpochId, _: TransactionId) -> Result<bool> {
            Ok(self.delete_node(id))
        }
        fn delete_node_edges(&self, node_id: NodeId) {
            let mut inner = self.inner.lock().unwrap();
            inner.edges.retain(|e| e.src != node_id && e.dst != node_id);
        }
        fn delete_edge(&self, id: EdgeId) -> bool {
            let mut inner = self.inner.lock().unwrap();
            if let Some(pos) = inner.edges.iter().position(|e| e.id == id) {
                inner.edges.remove(pos);
                true
            } else {
                false
            }
        }
        fn delete_edge_versioned(&self, id: EdgeId, _: EpochId, _: TransactionId) -> bool {
            self.delete_edge(id)
        }
        fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
            let mut inner = self.inner.lock().unwrap();
            if let Some(node) = inner.nodes.iter_mut().find(|n| n.id == id) {
                node.set_property(key, value);
            }
        }
        fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
            let mut inner = self.inner.lock().unwrap();
            if let Some(edge) = inner.edges.iter_mut().find(|e| e.id == id) {
                edge.set_property(key, value);
            }
        }
        fn remove_node_property(&self, id: NodeId, key: &str) -> Result<Option<Value>> {
            let mut inner = self.inner.lock().unwrap();
            Ok(inner
                .nodes
                .iter_mut()
                .find(|n| n.id == id)
                .and_then(|n| n.remove_property(key)))
        }
        fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<Option<Value>> {
            let mut inner = self.inner.lock().unwrap();
            Ok(inner
                .edges
                .iter_mut()
                .find(|e| e.id == id)
                .and_then(|e| e.remove_property(key)))
        }
        fn add_label(&self, node_id: NodeId, label: &str) -> bool {
            let mut inner = self.inner.lock().unwrap();
            if let Some(node) = inner.nodes.iter_mut().find(|n| n.id == node_id) {
                if node.has_label(label) {
                    false
                } else {
                    node.add_label(label);
                    true
                }
            } else {
                false
            }
        }
        fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
            let mut inner = self.inner.lock().unwrap();
            inner
                .nodes
                .iter_mut()
                .find(|n| n.id == node_id)
                .is_some_and(|n| n.remove_label(label))
        }
    }

    #[test]
    fn test_mut_store_default_set_versioned_property_delegates() {
        let store = TestMutStore::new();
        let id = store.create_node(&["Person"]);
        let key = PropertyKey::from("name");
        let txn = TransactionId(7);

        // Default impl of set_node_property_versioned calls set_node_property.
        store
            .set_node_property_versioned(id, "name", Value::from("Vincent"), txn)
            .unwrap();
        assert_eq!(
            store.node_property(id, key.as_str(), NOW).unwrap(),
            Some(Value::from("Vincent"))
        );

        let edge_id = {
            let src = store.create_node(&["Person"]);
            let dst = store.create_node(&["City"]);
            store.create_edge(src, dst, "LIVES_IN")
        };
        let since = PropertyKey::from("since");
        store.set_edge_property_versioned(edge_id, "since", Value::Int64(1994), txn);
        assert_eq!(
            store.edge_property(edge_id, since.as_str(), NOW).unwrap(),
            Some(Value::Int64(1994))
        );
    }

    #[test]
    fn test_mut_store_default_remove_versioned_property_delegates() {
        let store = TestMutStore::new();
        let txn = TransactionId(11);

        let node_id = store.create_node(&["Person"]);
        store.set_node_property(node_id, "city", Value::from("Amsterdam"));
        let removed = store
            .remove_node_property_versioned(node_id, "city", txn)
            .unwrap();
        assert_eq!(removed, Some(Value::from("Amsterdam")));
        assert!(store.node_property(node_id, "city", NOW).unwrap().is_none());

        let missing = store
            .remove_node_property_versioned(node_id, "absent", txn)
            .unwrap();
        assert!(missing.is_none());

        let src = store.create_node(&["Person"]);
        let dst = store.create_node(&["Person"]);
        let edge_id = store.create_edge(src, dst, "KNOWS");
        store.set_edge_property(edge_id, "weight", Value::Int64(42));
        let removed_edge = store
            .remove_edge_property_versioned(edge_id, "weight", txn)
            .unwrap();
        assert_eq!(removed_edge, Some(Value::Int64(42)));
        let removed_again = store
            .remove_edge_property_versioned(edge_id, "weight", txn)
            .unwrap();
        assert!(removed_again.is_none());
    }

    #[test]
    fn test_mut_store_default_label_versioned_delegates() {
        let store = TestMutStore::new();
        let txn = TransactionId(3);
        let id = store.create_node(&["Person"]);

        // Adding a new label returns true, re-adding the same returns false.
        assert!(store.add_label_versioned(id, "Director", txn));
        assert!(!store.add_label_versioned(id, "Director", txn));

        // Removing an existing label returns true, removing absent returns false.
        assert!(store.remove_label_versioned(id, "Director", txn));
        assert!(!store.remove_label_versioned(id, "Director", txn));

        // Unknown node id yields false on both add and remove paths.
        let unknown = NodeId(9999);
        assert!(!store.add_label_versioned(unknown, "Ghost", txn));
        assert!(!store.remove_label_versioned(unknown, "Ghost", txn));
    }

    #[test]
    fn test_mut_store_default_create_node_with_props() {
        let store = TestMutStore::new();
        let props = vec![
            (PropertyKey::from("name"), Value::from("Jules")),
            (PropertyKey::from("city"), Value::from("Paris")),
        ];

        let id = store.create_node_with_props(&["Person"], &props);
        let node = store.node(id, NOW).unwrap().expect("node should exist");
        assert!(node.has_label("Person"));
        assert_eq!(
            node.properties.get(&PropertyKey::from("name")),
            Some(&Value::from("Jules"))
        );
        assert_eq!(
            node.properties.get(&PropertyKey::from("city")),
            Some(&Value::from("Paris"))
        );

        // Empty properties slice still produces a valid node.
        let bare = store.create_node_with_props(&["Person"], &[]);
        let bare_node = store
            .node(bare, NOW)
            .unwrap()
            .expect("bare node should exist");
        assert!(bare_node.properties.is_empty());
    }

    #[test]
    fn test_mut_store_default_create_edge_with_props() {
        let store = TestMutStore::new();
        let src = store.create_node_with_props(
            &["Person"],
            &[(PropertyKey::from("name"), Value::from("Mia"))],
        );
        let dst = store.create_node_with_props(
            &["City"],
            &[(PropertyKey::from("name"), Value::from("Berlin"))],
        );
        let props = vec![
            (PropertyKey::from("since"), Value::Int64(2021)),
            (PropertyKey::from("role"), Value::from("resident")),
        ];

        let edge_id = store.create_edge_with_props(src, dst, "LIVES_IN", &props);
        let edge = store
            .edge(edge_id, NOW)
            .unwrap()
            .expect("edge should exist");
        assert_eq!(edge.src, src);
        assert_eq!(edge.dst, dst);
        assert_eq!(edge.edge_type.as_str(), "LIVES_IN");
        assert_eq!(
            edge.properties.get(&PropertyKey::from("since")),
            Some(&Value::Int64(2021))
        );
        assert_eq!(
            edge.properties.get(&PropertyKey::from("role")),
            Some(&Value::from("resident"))
        );

        // Confirm the edge type is also reachable through the read trait.
        assert_eq!(
            store
                .edge_type(edge_id, NOW)
                .unwrap()
                .as_ref()
                .map(arcstr::ArcStr::as_str),
            Some("LIVES_IN")
        );

        // With no properties, default still produces an edge.
        let bare = store.create_edge_with_props(src, dst, "VISITED", &[]);
        let bare_edge = store
            .edge(bare, NOW)
            .unwrap()
            .expect("bare edge should exist");
        assert!(bare_edge.properties.is_empty());
    }

    #[test]
    fn test_mut_store_object_safe_dyn_dispatch() {
        // Exercise the object-safe contract: GraphStore methods through `dyn`.
        let store: Arc<dyn GraphStore> = Arc::new(TestMutStore::new());
        assert_eq!(store.node_count(NOW).unwrap(), 0);
        assert_eq!(store.edge_count(NOW).unwrap(), 0);
        let ids = store.node_ids(NOW).unwrap();
        assert!(ids.is_empty(), "{ids:?}");
        assert!(store.node(NodeId(1), NOW).unwrap().is_none());
    }
}
