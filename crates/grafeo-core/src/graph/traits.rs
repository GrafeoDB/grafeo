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

/// The labels of a node, as [`GraphRead::node_labels`] returns them.
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
/// `Arc<dyn GraphRead>`. Reads return `Vec`s rather than iterators for
/// that reason.
pub trait GraphRead: Send + Sync {
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
fn measured_vectors<S: GraphRead + ?Sized>(
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
/// [`GraphRead::edges`] lists it once, as an undirected pattern matches it
/// once (openCypher TCK Match3, cases 11 and 12). A store implementing
/// [`GraphRead::edges`] calls it on what it lists for `Both`.
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

/// Read-only graph operations used by the query engine.
///
/// This trait captures the minimal surface that scan, expand, filter,
/// project, and shortest-path operators need. Implementations may serve
/// data from memory, disk, or a hybrid of both.
///
/// # Object safety
///
/// This trait is object-safe: you can use `Arc<dyn GraphStoreSearch>` for dynamic
/// dispatch. Traversal methods return `Vec` instead of `impl Iterator` to
/// enable this.
pub trait GraphStore: Send + Sync {
    // --- Point lookups ---

    /// Returns a node by ID (latest visible version at current epoch).
    fn get_node(&self, id: NodeId) -> Option<Node>;

    /// Returns an edge by ID (latest visible version at current epoch).
    fn get_edge(&self, id: EdgeId) -> Option<Edge>;

    /// Returns a node visible to a specific transaction.
    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node>;

    /// Returns an edge visible to a specific transaction.
    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge>;

    /// Returns a node using pure epoch-based visibility (no transaction context).
    ///
    /// The node is visible if `created_epoch <= epoch` and not deleted at or
    /// before `epoch`. Used for time-travel queries where transaction ownership
    /// must not bypass the epoch check.
    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node>;

    /// Returns an edge using pure epoch-based visibility (no transaction context).
    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge>;

    // --- Property access (fast path, avoids loading full entity) ---

    /// Gets a single property from a node without loading all properties.
    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value>;

    /// Calls `f` with the vector in a node's property and returns whether
    /// there was one. Vector search reads every distance through this, so a
    /// store should lend the vector without copying it; the default reads it
    /// through [`get_node_property`](Self::get_node_property).
    fn with_node_vector(&self, id: NodeId, key: &PropertyKey, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.get_node_property(id, key) {
            Some(Value::Vector(vector)) => {
                f(&vector);
                true
            }
            _ => false,
        }
    }

    /// Gets a single property from an edge without loading all properties.
    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value>;

    /// Gets a property for multiple nodes in a single batch operation.
    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>>;

    /// The value of property `key` of node `id` as
    /// [`get_node_versioned`](Self::get_node_versioned) sees the node: `None`
    /// when the node is not visible or has no such property. A query reads
    /// every `n.prop` through this; the default builds the node, a store
    /// that can read one value of a visible node overrides it.
    fn node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.get_node_versioned(id, epoch, transaction_id)?
            .get_property(key)
            .cloned()
    }

    /// The value of property `key` of node `id` as
    /// [`get_node_at_epoch`](Self::get_node_at_epoch) sees the node (see
    /// [`node_property_versioned`](Self::node_property_versioned)).
    fn node_property_at_epoch(&self, id: NodeId, key: &str, epoch: EpochId) -> Option<Value> {
        self.get_node_at_epoch(id, epoch)?
            .get_property(key)
            .cloned()
    }

    /// The value of property `key` of edge `id` as
    /// [`get_edge_versioned`](Self::get_edge_versioned) sees the edge (see
    /// [`node_property_versioned`](Self::node_property_versioned)).
    fn edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.get_edge_versioned(id, epoch, transaction_id)?
            .get_property(key)
            .cloned()
    }

    /// The value of property `key` of edge `id` as
    /// [`get_edge_at_epoch`](Self::get_edge_at_epoch) sees the edge (see
    /// [`node_property_versioned`](Self::node_property_versioned)).
    fn edge_property_at_epoch(&self, id: EdgeId, key: &str, epoch: EpochId) -> Option<Value> {
        self.get_edge_at_epoch(id, epoch)?
            .get_property(key)
            .cloned()
    }

    /// [`get_node_property_batch`](Self::get_node_property_batch) for the
    /// readers that must tell a value that cannot be read from no value (the
    /// key of an algorithm's `key=`): a spilled value that cannot be read is
    /// an error, where the batch read reads it as absent. A store with
    /// nothing to read from a file returns its batch read; a store that
    /// wraps another forwards to the inner store's fallible read.
    ///
    /// # Errors
    ///
    /// Returns the error of reading a spilled value.
    fn try_get_node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
    ) -> Result<Vec<Option<Value>>>;

    /// Gets all properties for multiple nodes in a single batch operation.
    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>>;

    /// Gets selected properties for multiple nodes (projection pushdown).
    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>>;

    /// Gets selected properties for multiple edges (projection pushdown).
    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>>;

    // --- Traversal ---

    /// Returns neighbor node IDs in the specified direction.
    ///
    /// Returns `Vec` instead of an iterator for object safety. The underlying
    /// `ChunkedAdjacency` already produces a `Vec` internally.
    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId>;

    /// Returns (target_node, edge_id) pairs for edges from a node.
    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)>;

    /// Returns the out-degree of a node (number of outgoing edges).
    fn out_degree(&self, node: NodeId) -> usize;

    /// Returns the in-degree of a node (number of incoming edges).
    fn in_degree(&self, node: NodeId) -> usize;

    /// Whether backward adjacency is available for incoming edge queries.
    fn has_backward_adjacency(&self) -> bool;

    // --- Scans ---

    /// Returns all non-deleted node IDs, sorted by ID.
    fn node_ids(&self) -> Vec<NodeId>;

    /// Returns all node IDs including uncommitted/PENDING versions.
    ///
    /// Unlike `node_ids()` which pre-filters by current epoch, this method
    /// returns every node that has a version chain entry. Used by scan operators
    /// that perform their own MVCC visibility filtering (e.g. with transaction context).
    fn all_node_ids(&self) -> Vec<NodeId> {
        // Default: fall back to node_ids() for stores without MVCC
        self.node_ids()
    }

    /// Returns node IDs with a specific label.
    fn nodes_by_label(&self, label: &str) -> Vec<NodeId>;

    /// Returns the IDs of the nodes with at least one of `labels`, in ID
    /// order and each once, as [`nodes_by_label`](Self::nodes_by_label)
    /// finds them.
    ///
    /// The default merges the label lists; a store that keeps labels as
    /// bitmaps can answer it with their union.
    fn nodes_with_any_label(&self, labels: &[&str]) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = labels
            .iter()
            .flat_map(|label| self.nodes_by_label(label))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Returns the number of non-deleted nodes with a specific label.
    ///
    /// Default falls back to `self.nodes_by_label(label).len()`; stores that
    /// maintain a per-label index should override for an O(1) count without
    /// allocating the full ID list.
    fn nodes_by_label_count(&self, label: &str) -> usize {
        self.nodes_by_label(label).len()
    }

    /// Returns the number of nodes visible at `epoch` that have `label` then:
    /// what a query counts, without the nodes of a transaction that has not
    /// committed (the label index holds those too).
    ///
    /// The default builds each node of the label index at `epoch`; stores
    /// override it with a visibility check and a label read, without building
    /// nodes and their properties.
    fn nodes_by_label_count_at_epoch(&self, label: &str, epoch: EpochId) -> usize {
        self.nodes_by_label(label)
            .into_iter()
            .filter(|&id| {
                self.get_node_at_epoch(id, epoch)
                    .is_some_and(|node| node.has_label(label))
            })
            .count()
    }

    /// Returns the total number of non-deleted nodes.
    fn node_count(&self) -> usize;

    /// Returns the total number of non-deleted edges.
    fn edge_count(&self) -> usize;

    // --- Entity metadata ---

    /// Returns the type string of an edge.
    fn edge_type(&self, id: EdgeId) -> Option<ArcStr>;

    /// Returns the type string of an edge visible to a specific transaction,
    /// including the edges it created and has not committed yet.
    ///
    /// The default reads the whole edge with
    /// [`get_edge_versioned`](Self::get_edge_versioned); stores override it
    /// with a cheaper lookup. It must not fall back to
    /// [`edge_type`](Self::edge_type), which reads the committed state and
    /// misses the transaction's own edges.
    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        self.get_edge_versioned(id, epoch, transaction_id)
            .map(|edge| edge.edge_type)
    }

    // --- Index introspection ---

    /// Returns `true` if a property index exists for the given property.
    ///
    /// The default returns `false`, which is correct for stores without indexes.
    fn has_property_index(&self, _property: &str) -> bool {
        false
    }

    // --- Filtered search ---

    /// Finds all nodes with a specific property value. Uses indexes when available.
    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId>;

    /// Finds nodes matching multiple property equality conditions.
    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId>;

    /// The nodes whose `property` may be equal to `value` under `=` (the
    /// filter's equality, see
    /// [`HashKey::for_equality`](crate::execution::operators::HashKey::for_equality)),
    /// found through the property's index: every node `=` finds equal, and
    /// maybe others, for a filter to decide.
    ///
    /// The default returns `None`, as does a store without an index on
    /// `property`: the caller then scans.
    fn find_nodes_maybe_equal(&self, _property: &str, _value: &Value) -> Option<Vec<NodeId>> {
        None
    }

    /// Finds nodes whose property value falls within a range.
    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId>;

    // --- Zone maps (skip pruning) ---

    /// Returns `true` if a node property predicate might match any nodes.
    /// Uses zone maps for early filtering.
    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool;

    /// Returns `true` if an edge property predicate might match any edges.
    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool;

    // --- Statistics (for cost-based optimizer) ---

    /// Returns the current statistics snapshot (cheap Arc clone).
    fn statistics(&self) -> Arc<Statistics>;

    /// Estimates cardinality for a label scan.
    fn estimate_label_cardinality(&self, label: &str) -> f64;

    /// Estimates average degree for an edge type.
    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64;

    // --- Epoch ---

    /// Returns the current MVCC epoch.
    fn current_epoch(&self) -> EpochId;

    // --- Schema introspection ---

    /// Returns all label names in the database.
    fn all_labels(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns all edge type names in the database.
    fn all_edge_types(&self) -> Vec<String> {
        Vec::new()
    }

    /// Returns all property key names used in the database.
    fn all_property_keys(&self) -> Vec<String> {
        Vec::new()
    }

    // --- Visibility checks (fast path, avoids building full entities) ---

    /// Checks if a node is visible at the given epoch without building the full Node.
    ///
    /// More efficient than `get_node_at_epoch(...).is_some()` because it skips
    /// label and property loading. Override in concrete stores for optimal
    /// performance.
    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        self.get_node_at_epoch(id, epoch).is_some()
    }

    /// Checks if a node is visible to a specific transaction without building
    /// the full Node.
    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.get_node_versioned(id, epoch, transaction_id).is_some()
    }

    /// Checks if an edge is visible at the given epoch without building the full Edge.
    ///
    /// More efficient than `get_edge_at_epoch(...).is_some()` because it skips
    /// type name resolution and property loading. Override in concrete stores
    /// for optimal performance.
    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        self.get_edge_at_epoch(id, epoch).is_some()
    }

    /// Checks if an edge is visible to a specific transaction without building
    /// the full Edge.
    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.get_edge_versioned(id, epoch, transaction_id).is_some()
    }

    /// Filters node IDs to only those visible at the given epoch (batch).
    ///
    /// More efficient than per-node calls because implementations can hold
    /// a single lock for the entire batch.
    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_at_epoch(*id, epoch))
            .collect()
    }

    /// Filters node IDs to only those visible to a transaction (batch).
    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_versioned(*id, epoch, transaction_id))
            .collect()
    }

    // --- Reads at a view (#412) ---
    //
    // Each read below answers as of a `ReadView`: a transaction at its
    // snapshot, its own writes included, or a committed reader at an epoch.
    // The defaults keep the answer of the same read without a view and drop
    // what the view does not see (existence only), so every store keeps
    // working as it did; a store that versions values and labels answers
    // exactly and says so with `answers_at_view`.

    /// Whether this store's lookups and label scans at a view answer what a
    /// scan of every node finds at that view, read through the store's own
    /// per-row reads at the view. The planner then lets a transaction or a
    /// read at a past epoch use them in place of a scan. Index entries may
    /// be a superset of what any one view sees, as long as the lookup
    /// rechecks each candidate at the view.
    ///
    /// The default is `false`: the default lookups give the answer of now,
    /// filtered by existence at the view.
    fn answers_at_view(&self) -> bool {
        false
    }

    /// Whether `view` sees node `id`.
    fn is_node_visible_at(&self, id: NodeId, view: ReadView) -> bool {
        match view.transaction {
            Some(transaction_id) => self.is_node_visible_versioned(id, view.epoch, transaction_id),
            None => self.is_node_visible_at_epoch(id, view.epoch),
        }
    }

    /// Whether `view` sees edge `id`.
    fn is_edge_visible_at(&self, id: EdgeId, view: ReadView) -> bool {
        match view.transaction {
            Some(transaction_id) => self.is_edge_visible_versioned(id, view.epoch, transaction_id),
            None => self.is_edge_visible_at_epoch(id, view.epoch),
        }
    }

    /// The nodes of `ids` that `view` sees, in their order.
    fn filter_visible_node_ids_at(&self, ids: &[NodeId], view: ReadView) -> Vec<NodeId> {
        match view.transaction {
            Some(transaction_id) => {
                self.filter_visible_node_ids_versioned(ids, view.epoch, transaction_id)
            }
            None => self.filter_visible_node_ids(ids, view.epoch),
        }
    }

    /// Node `id` as `view` sees it.
    fn get_node_at(&self, id: NodeId, view: ReadView) -> Option<Node> {
        match view.transaction {
            Some(transaction_id) => self.get_node_versioned(id, view.epoch, transaction_id),
            None => self.get_node_at_epoch(id, view.epoch),
        }
    }

    /// Edge `id` as `view` sees it.
    fn get_edge_at(&self, id: EdgeId, view: ReadView) -> Option<Edge> {
        match view.transaction {
            Some(transaction_id) => self.get_edge_versioned(id, view.epoch, transaction_id),
            None => self.get_edge_at_epoch(id, view.epoch),
        }
    }

    /// The value of property `key` of node `id` as `view` sees the node:
    /// `None` when the view does not see it or it has no such property.
    fn node_property_at(&self, id: NodeId, key: &str, view: ReadView) -> Option<Value> {
        match view.transaction {
            Some(transaction_id) => {
                self.node_property_versioned(id, key, view.epoch, transaction_id)
            }
            None => self.node_property_at_epoch(id, key, view.epoch),
        }
    }

    /// The value of property `key` of edge `id` as `view` sees the edge
    /// (see [`node_property_at`](Self::node_property_at)).
    fn edge_property_at(&self, id: EdgeId, key: &str, view: ReadView) -> Option<Value> {
        match view.transaction {
            Some(transaction_id) => {
                self.edge_property_versioned(id, key, view.epoch, transaction_id)
            }
            None => self.edge_property_at_epoch(id, key, view.epoch),
        }
    }

    /// The labels of node `id` as `view` sees the node, `None` when the view
    /// does not see it. The default builds the node.
    fn node_labels_at(&self, id: NodeId, view: ReadView) -> Option<SmallVec<[ArcStr; 2]>> {
        self.get_node_at(id, view).map(|node| node.labels)
    }

    /// Whether node `id` has `label` as `view` sees the node, `None` when
    /// the view does not see it: a label check.
    fn node_has_label_at(&self, id: NodeId, label: &str, view: ReadView) -> Option<bool> {
        self.node_labels_at(id, view)
            .map(|labels| labels.iter().any(|name| name.as_str() == label))
    }

    /// [`get_node_property_batch`](Self::get_node_property_batch) as `view`
    /// sees the nodes: `None` for a node it does not see.
    fn get_node_property_batch_at(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
        view: ReadView,
    ) -> Vec<Option<Value>> {
        let mut values = self.get_node_property_batch(ids, key);
        for (value, id) in values.iter_mut().zip(ids) {
            if value.is_some() && !self.is_node_visible_at(*id, view) {
                *value = None;
            }
        }
        values
    }

    /// [`get_nodes_properties_batch`](Self::get_nodes_properties_batch) as
    /// `view` sees the nodes: no values for a node it does not see.
    fn get_nodes_properties_batch_at(
        &self,
        ids: &[NodeId],
        view: ReadView,
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let mut values = self.get_nodes_properties_batch(ids);
        for (map, id) in values.iter_mut().zip(ids) {
            if !map.is_empty() && !self.is_node_visible_at(*id, view) {
                map.clear();
            }
        }
        values
    }

    /// [`get_nodes_properties_selective_batch`](Self::get_nodes_properties_selective_batch)
    /// as `view` sees the nodes: no values for a node it does not see.
    fn get_nodes_properties_selective_batch_at(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let mut values = self.get_nodes_properties_selective_batch(ids, keys);
        for (map, id) in values.iter_mut().zip(ids) {
            if !map.is_empty() && !self.is_node_visible_at(*id, view) {
                map.clear();
            }
        }
        values
    }

    /// [`get_edges_properties_selective_batch`](Self::get_edges_properties_selective_batch)
    /// as `view` sees the edges: no values for an edge it does not see.
    fn get_edges_properties_selective_batch_at(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
        view: ReadView,
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        let mut values = self.get_edges_properties_selective_batch(ids, keys);
        for (map, id) in values.iter_mut().zip(ids) {
            if !map.is_empty() && !self.is_edge_visible_at(*id, view) {
                map.clear();
            }
        }
        values
    }

    /// The edges of `node` in `direction` that `view` sees, as (other node,
    /// edge), in the order of [`edges_from`](Self::edges_from).
    fn edges_from_at(
        &self,
        node: NodeId,
        direction: Direction,
        view: ReadView,
    ) -> Vec<(NodeId, EdgeId)> {
        let mut edges = self.edges_from(node, direction);
        edges.retain(|&(_, edge)| self.is_edge_visible_at(edge, view));
        edges
    }

    /// The nodes at the other end of the edges of
    /// [`edges_from_at`](Self::edges_from_at).
    fn neighbors_at(&self, node: NodeId, direction: Direction, view: ReadView) -> Vec<NodeId> {
        self.edges_from_at(node, direction, view)
            .into_iter()
            .map(|(other, _)| other)
            .collect()
    }

    /// The nodes with `label` that `view` sees, in ID order.
    fn nodes_by_label_at(&self, label: &str, view: ReadView) -> Vec<NodeId> {
        self.filter_visible_node_ids_at(&self.nodes_by_label(label), view)
    }

    /// The nodes with at least one of `labels` that `view` sees, each once
    /// and in ID order (see [`nodes_with_any_label`](Self::nodes_with_any_label)).
    fn nodes_with_any_label_at(&self, labels: &[&str], view: ReadView) -> Vec<NodeId> {
        self.filter_visible_node_ids_at(&self.nodes_with_any_label(labels), view)
    }

    /// The nodes `view` sees whose `property` equals `value`
    /// (see [`find_nodes_by_property`](Self::find_nodes_by_property)).
    fn find_nodes_by_property_at(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Vec<NodeId> {
        self.filter_visible_node_ids_at(&self.find_nodes_by_property(property, value), view)
    }

    /// The nodes `view` sees that meet every equality of `conditions`
    /// (see [`find_nodes_by_properties`](Self::find_nodes_by_properties)).
    fn find_nodes_by_properties_at(
        &self,
        conditions: &[(&str, Value)],
        view: ReadView,
    ) -> Vec<NodeId> {
        self.filter_visible_node_ids_at(&self.find_nodes_by_properties(conditions), view)
    }

    /// [`find_nodes_maybe_equal`](Self::find_nodes_maybe_equal) for `view`:
    /// every node the view sees that `=` finds equal, maybe others the view
    /// sees, for a filter to decide; `None` without an index on `property`.
    fn find_nodes_maybe_equal_at(
        &self,
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Option<Vec<NodeId>> {
        self.find_nodes_maybe_equal(property, value)
            .map(|found| self.filter_visible_node_ids_at(&found, view))
    }

    /// The nodes `view` sees whose `property` lies within the bounds (see
    /// [`find_nodes_in_range`](Self::find_nodes_in_range)).
    fn find_nodes_in_range_at(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        view: ReadView,
    ) -> Vec<NodeId> {
        self.filter_visible_node_ids_at(
            &self.find_nodes_in_range(property, min, max, min_inclusive, max_inclusive),
            view,
        )
    }

    // --- History ---

    /// Returns all versions of a node with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Node)`. Properties and labels
    /// reflect the current state (they are not versioned per-epoch).
    ///
    /// Default returns empty (not all backends track version history).
    fn get_node_history(&self, _id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        Vec::new()
    }

    /// Returns all versions of an edge with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Edge)`. Properties reflect
    /// the current state (they are not versioned per-epoch).
    ///
    /// Default returns empty (not all backends track version history).
    fn get_edge_history(&self, _id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        Vec::new()
    }
}

/// Index-backed search capabilities that an LPG store may optionally provide.
///
/// Keeps the base [`GraphStore`] scoped to graph-structure operations. Stores
/// that back text or vector indexes implement these methods with real search
/// logic; stores that don't (columnar bases, projections, RDF adapters) accept
/// the no-op defaults and the executor falls through to per-row evaluation.
///
/// # Symmetric text and vector APIs
///
/// `text_search` and `vector_search` are peer operations returning owned
/// `Vec<(NodeId, f64)>`. The planner decides strategy based on `has_*_index`
/// and `vector_index_config` introspection; the store executes the chosen
/// plan, falling back to brute-force internally when the request is valid
/// but no matching index exists.
pub trait GraphStoreSearch: GraphStore {
    // --- Range scan (lazy) ---

    /// Returns a lazy iterator over node ids whose property value falls
    /// within `[min, max]` (with the given inclusivity).
    ///
    /// The default implementation eagerly materializes via
    /// [`find_nodes_in_range`](GraphStore::find_nodes_in_range) and chains
    /// `.into_iter()`. Stores with per-block zone maps (e.g. `CompactStore`)
    /// override this with a true lazy iterator that prunes blocks via
    /// zone-map skip checks before decoding any row, enabling Phase 4
    /// iterator bounds to deliver real work-skip on selective queries.
    fn find_nodes_in_range_iter<'a>(
        &'a self,
        property: &'a str,
        min: Option<&'a Value>,
        max: Option<&'a Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Box<dyn Iterator<Item = NodeId> + 'a> {
        Box::new(
            self.find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
                .into_iter(),
        )
    }

    /// [`find_nodes_in_range_iter`](Self::find_nodes_in_range_iter) for
    /// `view`: the nodes it sees whose value lies within the bounds. The
    /// default materializes
    /// [`find_nodes_in_range_at`](GraphStore::find_nodes_in_range_at), as
    /// the read without a view does; a store with a lazy range scan
    /// overrides both.
    fn find_nodes_in_range_iter_at<'a>(
        &'a self,
        property: &'a str,
        min: Option<&'a Value>,
        max: Option<&'a Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        view: ReadView,
    ) -> Box<dyn Iterator<Item = NodeId> + 'a> {
        Box::new(
            self.find_nodes_in_range_at(property, min, max, min_inclusive, max_inclusive, view)
                .into_iter(),
        )
    }

    // --- Text search (BM25) ---

    /// Returns true if a BM25 text index exists for the given label and property.
    #[cfg(feature = "text-index")]
    #[must_use]
    fn has_text_index(&self, _label: &str, _property: &str) -> bool {
        false
    }

    /// Scores a single document against a text query for per-row filter evaluation.
    ///
    /// Returns `None` when no text index exists for the (label, property) pair.
    /// The planner calls this when pushdown is unavailable, for example when
    /// the text predicate follows a traversal instead of a bare label scan.
    #[cfg(feature = "text-index")]
    fn score_text(
        &self,
        _node_id: NodeId,
        _label: &str,
        _property: &str,
        _query: &str,
    ) -> Option<f64> {
        None
    }

    /// [`score_text`](Self::score_text) for `view`: the score of the node's
    /// text as the view sees it. The default scores the text of now.
    #[cfg(feature = "text-index")]
    fn score_text_at(
        &self,
        node_id: NodeId,
        label: &str,
        property: &str,
        query: &str,
        _view: ReadView,
    ) -> Option<f64> {
        self.score_text(node_id, label, property, query)
    }

    /// Returns the top-`k` documents by BM25 score for a text query.
    ///
    /// Results are sorted by score descending. Returns an empty vec when no
    /// text index exists, so the caller can fall back to a slower path.
    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        _label: &str,
        _property: &str,
        _query: &str,
        _k: usize,
    ) -> Vec<(NodeId, f64)> {
        Vec::new()
    }

    /// Returns every document whose BM25 score meets or exceeds a threshold.
    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        _label: &str,
        _property: &str,
        _query: &str,
        _threshold: f64,
    ) -> Vec<(NodeId, f64)> {
        Vec::new()
    }

    /// [`text_search`](Self::text_search) for `view`: the documents it sees.
    /// The default drops the results of now that the view does not see, so
    /// it may return fewer than `k`.
    #[cfg(feature = "text-index")]
    fn text_search_at(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
        view: ReadView,
    ) -> Vec<(NodeId, f64)> {
        let mut found = self.text_search(label, property, query, k);
        found.retain(|&(id, _)| self.is_node_visible_at(id, view));
        found
    }

    /// [`text_search_with_threshold`](Self::text_search_with_threshold) for
    /// `view`: the documents it sees.
    #[cfg(feature = "text-index")]
    fn text_search_with_threshold_at(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
        view: ReadView,
    ) -> Vec<(NodeId, f64)> {
        let mut found = self.text_search_with_threshold(label, property, query, threshold);
        found.retain(|&(id, _)| self.is_node_visible_at(id, view));
        found
    }

    // --- Vector search (HNSW or brute force) ---

    /// Returns true if a vector index exists for the given label and property.
    #[cfg(feature = "vector-index")]
    #[must_use]
    fn has_vector_index(&self, _label: &str, _property: &str) -> bool {
        false
    }

    /// Returns the distance metric of the vector index at (label, property), if any.
    ///
    /// The planner uses this to decide between an HNSW-accelerated plan and a
    /// brute-force fallback: an index whose metric does not match the query's
    /// requested metric cannot serve the query directly, so the planner either
    /// routes to brute force or skips pushdown entirely.
    #[cfg(feature = "vector-index")]
    fn vector_index_config(
        &self,
        _label: &str,
        _property: &str,
    ) -> Option<crate::index::vector::HnswConfig> {
        None
    }

    /// Returns the top-`k` nearest neighbors for a vector similarity search.
    ///
    /// `label` is optional: `None` searches every node that has the named
    /// property. The store uses HNSW when an index exists for (label, property)
    /// whose metric matches `metric`; otherwise it falls back to brute force.
    ///
    /// Results are sorted by distance ascending (nearest first). Returns an
    /// empty vec when neither an index nor any indexable property is found.
    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        _label: Option<&str>,
        _property: &str,
        _query: &[f32],
        _k: usize,
        _metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        Vec::new()
    }

    /// Returns every node whose distance to the query vector is at or below a threshold.
    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        _label: Option<&str>,
        _property: &str,
        _query: &[f32],
        _threshold: f64,
        _metric: DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        Vec::new()
    }

    /// [`vector_search`](Self::vector_search) for `view`: the nodes it sees.
    /// The default drops the results of now that the view does not see, so
    /// it may return fewer than `k`.
    #[cfg(feature = "vector-index")]
    fn vector_search_at(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
        view: ReadView,
    ) -> Vec<(NodeId, f64)> {
        let mut found = self.vector_search(label, property, query, k, metric);
        found.retain(|&(id, _)| self.is_node_visible_at(id, view));
        found
    }

    /// [`vector_search_with_threshold`](Self::vector_search_with_threshold)
    /// for `view`: the nodes it sees.
    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold_at(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: DistanceMetric,
        view: ReadView,
    ) -> Vec<(NodeId, f64)> {
        let mut found =
            self.vector_search_with_threshold(label, property, query, threshold, metric);
        found.retain(|&(id, _)| self.is_node_visible_at(id, view));
        found
    }
}

/// Write operations for graph mutation.
///
/// Separated from [`GraphStore`] so read-only wrappers (snapshots, read
/// replicas) can implement only `GraphStore`. Any mutable store is also
/// readable via the supertrait bound.
pub trait GraphStoreMut: GraphStoreSearch {
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
    fn get_node(&self, _: NodeId) -> Option<Node> {
        None
    }
    fn get_edge(&self, _: EdgeId) -> Option<Edge> {
        None
    }
    fn get_node_versioned(&self, _: NodeId, _: EpochId, _: TransactionId) -> Option<Node> {
        None
    }
    fn get_edge_versioned(&self, _: EdgeId, _: EpochId, _: TransactionId) -> Option<Edge> {
        None
    }
    fn get_node_at_epoch(&self, _: NodeId, _: EpochId) -> Option<Node> {
        None
    }
    fn get_edge_at_epoch(&self, _: EdgeId, _: EpochId) -> Option<Edge> {
        None
    }
    fn get_node_property(&self, _: NodeId, _: &PropertyKey) -> Option<Value> {
        None
    }
    fn get_edge_property(&self, _: EdgeId, _: &PropertyKey) -> Option<Value> {
        None
    }
    fn get_node_property_batch(&self, ids: &[NodeId], _: &PropertyKey) -> Vec<Option<Value>> {
        vec![None; ids.len()]
    }
    fn try_get_node_property_batch(
        &self,
        ids: &[NodeId],
        _: &PropertyKey,
    ) -> Result<Vec<Option<Value>>> {
        Ok(vec![None; ids.len()])
    }
    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        vec![FxHashMap::default(); ids.len()]
    }
    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        _: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        vec![FxHashMap::default(); ids.len()]
    }
    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        _: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        vec![FxHashMap::default(); ids.len()]
    }
    fn neighbors(&self, _: NodeId, _: Direction) -> Vec<NodeId> {
        Vec::new()
    }
    fn edges_from(&self, _: NodeId, _: Direction) -> Vec<(NodeId, EdgeId)> {
        Vec::new()
    }
    fn out_degree(&self, _: NodeId) -> usize {
        0
    }
    fn in_degree(&self, _: NodeId) -> usize {
        0
    }
    fn has_backward_adjacency(&self) -> bool {
        false
    }
    fn node_ids(&self) -> Vec<NodeId> {
        Vec::new()
    }
    fn nodes_by_label(&self, _: &str) -> Vec<NodeId> {
        Vec::new()
    }
    fn node_count(&self) -> usize {
        0
    }
    fn edge_count(&self) -> usize {
        0
    }
    fn edge_type(&self, _: EdgeId) -> Option<ArcStr> {
        None
    }
    fn find_nodes_by_property(&self, _: &str, _: &Value) -> Vec<NodeId> {
        Vec::new()
    }
    fn find_nodes_by_properties(&self, _: &[(&str, Value)]) -> Vec<NodeId> {
        Vec::new()
    }
    fn find_nodes_in_range(
        &self,
        _: &str,
        _: Option<&Value>,
        _: Option<&Value>,
        _: bool,
        _: bool,
    ) -> Vec<NodeId> {
        Vec::new()
    }
    fn node_property_might_match(&self, _: &PropertyKey, _: CompareOp, _: &Value) -> bool {
        false
    }
    fn edge_property_might_match(&self, _: &PropertyKey, _: CompareOp, _: &Value) -> bool {
        false
    }
    fn statistics(&self) -> Arc<Statistics> {
        Arc::new(Statistics::default())
    }
    fn estimate_label_cardinality(&self, _: &str) -> f64 {
        0.0
    }
    fn estimate_avg_degree(&self, _: &str, _: bool) -> f64 {
        0.0
    }
    fn current_epoch(&self) -> EpochId {
        EpochId(0)
    }
}

impl GraphStoreSearch for NullGraphStore {}

impl GraphRead for NullGraphStore {
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
        assert!(GraphRead::node(&store, node, NOW).unwrap().is_none());
        assert!(GraphRead::edge(&store, edge, NOW).unwrap().is_none());
        assert!(!GraphRead::is_node_visible(&store, node, NOW).unwrap());
        assert!(!GraphRead::is_edge_visible(&store, edge, NOW).unwrap());
        assert_eq!(
            GraphRead::node_property_batch(&store, &[node, NodeId(2)], &key, NOW).unwrap(),
            vec![None, None]
        );
        let edges = GraphRead::edges(&store, node, Direction::Both, &[], NOW).unwrap();
        assert!(edges.is_empty(), "{edges:?}");
        assert_eq!(
            GraphRead::degree(&store, node, Direction::Outgoing, NOW).unwrap(),
            0
        );
        assert!(!GraphRead::has_backward_adjacency(&store));
        let ids = GraphRead::node_ids(&store, NOW).unwrap();
        assert!(ids.is_empty(), "{ids:?}");
        assert_eq!(GraphRead::node_count(&store, NOW).unwrap(), 0);
        assert_eq!(GraphRead::edge_count(&store, NOW).unwrap(), 0);
        for names in [
            GraphRead::all_labels(&store).unwrap(),
            GraphRead::all_edge_types(&store).unwrap(),
            GraphRead::all_property_keys(&store).unwrap(),
        ] {
            assert!(names.is_empty(), "{names:?}");
        }
        assert!(!GraphRead::has_property_index(&store, "name"));
        assert_eq!(
            GraphRead::find_nodes_maybe_equal(&store, "name", &Value::from("Gus"), NOW).unwrap(),
            None
        );
    }

    /// A store without history says so: an error, not an empty history.
    #[test]
    fn a_store_without_history_reports_it_unsupported() {
        let store = NullGraphStore;
        let error = GraphRead::node_history(&store, NodeId(1)).unwrap_err();
        assert!(error.to_string().contains("node history"), "{error}");
        assert!(GraphRead::edge_history(&store, EdgeId(1)).is_err());
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

    impl GraphRead for Required {
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
        fn get_node(&self, id: NodeId) -> Option<Node> {
            self.find_node(id)
        }
        fn get_edge(&self, id: EdgeId) -> Option<Edge> {
            self.find_edge(id)
        }
        fn get_node_versioned(&self, id: NodeId, _: EpochId, _: TransactionId) -> Option<Node> {
            self.find_node(id)
        }
        fn get_edge_versioned(&self, id: EdgeId, _: EpochId, _: TransactionId) -> Option<Edge> {
            self.find_edge(id)
        }
        fn get_node_at_epoch(&self, id: NodeId, _: EpochId) -> Option<Node> {
            self.find_node(id)
        }
        fn get_edge_at_epoch(&self, id: EdgeId, _: EpochId) -> Option<Edge> {
            self.find_edge(id)
        }
        fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
            self.find_node(id)
                .and_then(|n| n.properties.get(key).cloned())
        }
        fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
            self.find_edge(id)
                .and_then(|e| e.properties.get(key).cloned())
        }
        fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
            ids.iter()
                .map(|id| self.get_node_property(*id, key))
                .collect()
        }
        fn try_get_node_property_batch(
            &self,
            ids: &[NodeId],
            key: &PropertyKey,
        ) -> Result<Vec<Option<Value>>> {
            Ok(self.get_node_property_batch(ids, key))
        }
        fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
            ids.iter()
                .map(|id| {
                    let mut map = FxHashMap::default();
                    if let Some(n) = self.find_node(*id) {
                        for (k, v) in n.properties.iter() {
                            map.insert(k.clone(), v.clone());
                        }
                    }
                    map
                })
                .collect()
        }
        fn get_nodes_properties_selective_batch(
            &self,
            ids: &[NodeId],
            _: &[PropertyKey],
        ) -> Vec<FxHashMap<PropertyKey, Value>> {
            vec![FxHashMap::default(); ids.len()]
        }
        fn get_edges_properties_selective_batch(
            &self,
            ids: &[EdgeId],
            _: &[PropertyKey],
        ) -> Vec<FxHashMap<PropertyKey, Value>> {
            vec![FxHashMap::default(); ids.len()]
        }
        fn neighbors(&self, _: NodeId, _: Direction) -> Vec<NodeId> {
            Vec::new()
        }
        fn edges_from(&self, _: NodeId, _: Direction) -> Vec<(NodeId, EdgeId)> {
            Vec::new()
        }
        fn out_degree(&self, _: NodeId) -> usize {
            0
        }
        fn in_degree(&self, _: NodeId) -> usize {
            0
        }
        fn has_backward_adjacency(&self) -> bool {
            false
        }
        fn node_ids(&self) -> Vec<NodeId> {
            self.inner
                .lock()
                .unwrap()
                .nodes
                .iter()
                .map(|n| n.id)
                .collect()
        }
        fn nodes_by_label(&self, _: &str) -> Vec<NodeId> {
            Vec::new()
        }
        fn node_count(&self) -> usize {
            self.inner.lock().unwrap().nodes.len()
        }
        fn edge_count(&self) -> usize {
            self.inner.lock().unwrap().edges.len()
        }
        fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
            self.find_edge(id).map(|e| e.edge_type)
        }
        fn find_nodes_by_property(&self, _: &str, _: &Value) -> Vec<NodeId> {
            Vec::new()
        }
        fn find_nodes_by_properties(&self, _: &[(&str, Value)]) -> Vec<NodeId> {
            Vec::new()
        }
        fn find_nodes_in_range(
            &self,
            _: &str,
            _: Option<&Value>,
            _: Option<&Value>,
            _: bool,
            _: bool,
        ) -> Vec<NodeId> {
            Vec::new()
        }
        fn node_property_might_match(&self, _: &PropertyKey, _: CompareOp, _: &Value) -> bool {
            true
        }
        fn edge_property_might_match(&self, _: &PropertyKey, _: CompareOp, _: &Value) -> bool {
            true
        }
        fn statistics(&self) -> Arc<Statistics> {
            Arc::new(Statistics::default())
        }
        fn estimate_label_cardinality(&self, _: &str) -> f64 {
            0.0
        }
        fn estimate_avg_degree(&self, _: &str, _: bool) -> f64 {
            0.0
        }
        fn current_epoch(&self) -> EpochId {
            EpochId(0)
        }
    }

    impl GraphStoreSearch for TestMutStore {}

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
            store.get_node_property(id, &key),
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
            store.get_edge_property(edge_id, &since),
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
        assert!(
            store
                .get_node_property(node_id, &PropertyKey::from("city"))
                .is_none()
        );

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
        let node = store.get_node(id).expect("node should exist");
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
        let bare_node = store.get_node(bare).expect("bare node should exist");
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
        let edge = store.get_edge(edge_id).expect("edge should exist");
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
                .edge_type(edge_id)
                .as_ref()
                .map(arcstr::ArcStr::as_str),
            Some("LIVES_IN")
        );

        // With no properties, default still produces an edge.
        let bare = store.create_edge_with_props(src, dst, "VISITED", &[]);
        let bare_edge = store.get_edge(bare).expect("bare edge should exist");
        assert!(bare_edge.properties.is_empty());
    }

    #[test]
    fn test_mut_store_object_safe_dyn_dispatch() {
        // Exercise the object-safe contract: GraphStore methods through `dyn`.
        let store: Arc<dyn GraphStoreSearch> = Arc::new(TestMutStore::new());
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.edge_count(), 0);
        assert!(store.node_ids().is_empty(), "{:?}", store.node_ids());
        assert!(store.get_node(NodeId(1)).is_none());
        assert_eq!(store.current_epoch(), EpochId(0));
    }
}
