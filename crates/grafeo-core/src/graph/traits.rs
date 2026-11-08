//! Storage traits for the graph engine.
//!
//! These traits capture the minimal surface that query operators need from
//! the graph store. The split is intentional:
//!
//! - [`GraphStore`]: Read-only operations (scans, lookups, traversal, statistics)
//! - [`GraphBackend`](super::GraphBackend): the write side, a transaction's
//!   writes, commit and rollback
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
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, ReadView, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;

/// The labels of a node, as [`GraphStore::node_labels`] returns them.
pub type Labels = grafeo_common::change::Labels;

/// The most nodes [`GraphStore::sampled_degree`] reads: a label of at most this many
/// is read whole, so its 0 is known, where a larger label's may be a sample that missed.
pub const DEGREE_SAMPLE_NODES: usize = 256;

/// [`GraphStore::common_edges`] from the two lists: a store that cannot intersect
/// them where they lie.
pub(crate) fn common_edges_by_lists(
    near_edges: Vec<(NodeId, EdgeId)>,
    far_edges: Vec<(NodeId, EdgeId)>,
) -> Result<(Vec<CommonEdge>, usize)> {
    let read = near_edges.len() + far_edges.len();
    let hash_near = near_edges.len() < far_edges.len();
    let (hashed, walked) = if hash_near {
        (near_edges, far_edges)
    } else {
        (far_edges, near_edges)
    };
    let mut by_node: std::collections::HashMap<NodeId, Vec<EdgeId>> =
        std::collections::HashMap::with_capacity(hashed.len());
    for (node, edge) in hashed {
        by_node.entry(node).or_default().push(edge);
    }
    let mut common = Vec::new();
    for (node, walked_edge) in walked {
        for hashed_edge in by_node.get(&node).into_iter().flatten() {
            let (near_edge, far_edge) = if hash_near {
                (*hashed_edge, walked_edge)
            } else {
                (walked_edge, *hashed_edge)
            };
            common.push(CommonEdge {
                node,
                near_edge,
                far_edge,
            });
        }
    }
    common.sort_unstable();
    Ok((common, read))
}

/// One side of [`GraphStore::common_edges`]: a node's list of one hop.
#[derive(Debug, Clone, Copy)]
pub struct HopFrom<'a> {
    /// The node.
    pub node: NodeId,
    /// The way the list goes.
    pub direction: Direction,
    /// The edge types it holds (every type when empty).
    pub edge_types: &'a [ArcStr],
}

/// A node two lists reach, by one edge of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CommonEdge {
    /// The node.
    pub node: NodeId,
    /// The edge of the near list.
    pub near_edge: EdgeId,
    /// The edge of the far list.
    pub far_edge: EdgeId,
}

/// What a read of a key's column found in the rows of one call (see
/// [`GraphStore::key_column_group`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyColumnGroup {
    /// The nodes found, ascending.
    pub nodes: Vec<NodeId>,
    /// How many rows it looked at, for a statement's count of what it read.
    pub compared: usize,
    /// Whether each node found is `=` a value, not only maybe.
    pub exact: bool,
    /// The id the next read starts from; `None` when no group is left.
    pub end: Option<u64>,
    /// One past every node id the store had given out when it read.
    pub ids_given: u64,
}

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
/// - [`edges`](Self::edges) lists the edges the view sees, a self-loop once
///   with [`Direction::Both`], and the other end of each is a node the view
///   sees too. For a store the engine writes ([`GraphBackend`]) that holds by
///   construction: an edge's create claims its two ends against a concurrent
///   delete in the transaction manager (first writer wins, and at commit the
///   later one loses), and a delete of a node with an edge its writer sees
///   is refused, so no committed edge outlives an end and a store need not
///   check the other end on a read. A store with data of its own
///   (`GrafeoDB::with_read_store`) must not hold an edge without its ends,
///   or must leave it out of its reads.
///
/// [`GraphBackend`]: crate::graph::GraphBackend
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
///
/// # Stability: Experimental
///
/// Calling the trait is stable. Implementing it (for `GrafeoDB::with_store`
/// or `with_read_store`) is experimental through 0.6.x: any release may add
/// a required method.
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

    /// The labels of node `id` as `view` sees the node, as a bit mask: bit
    /// `i` for the label whose id is `i` in the store. `None` when the view
    /// does not see the node, or the store has no such mask for it (a label
    /// id of 64 or more, or a store without label ids): `labels(n)` then
    /// reads the names. A grouping by `labels(n)` keys its groups by the
    /// mask, one integer, and names each group's labels once (see
    /// [`label_mask_names`](Self::label_mask_names)).
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn node_label_mask(&self, _id: NodeId, _view: ReadView) -> Result<Option<u64>> {
        Ok(None)
    }

    /// The names of the labels of a mask [`node_label_mask`](Self::node_label_mask)
    /// gave, sorted by name as `labels(n)` lists them.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do, and an internal
    /// error from a store that gives no masks.
    fn label_mask_names(&self, _mask: u64) -> Result<Labels> {
        Err(grafeo_common::utils::error::Error::Internal(
            "a label mask from a store that gives none".to_string(),
        ))
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
    /// empty) that `view` sees, each as (other node, edge); the view sees
    /// each other end too (see "What the engine relies on").
    /// [`Direction::Both`] lists a self-loop once. Types match exactly.
    /// Empty for a node the view does not see.
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

    /// The nodes both `near`'s and `far`'s lists reach, each with every pair of an
    /// edge from either list, ordered by node and edges, and the entries read. The
    /// rows of following `near` and then looking its far ends up in `far`'s list.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn common_edges(
        &self,
        near: &HopFrom<'_>,
        far: &HopFrom<'_>,
        view: ReadView,
    ) -> Result<(Vec<CommonEdge>, usize)> {
        common_edges_by_lists(
            self.edges(near.node, near.direction, near.edge_types, view)?,
            self.edges(far.node, far.direction, far.edge_types, view)?,
        )
    }

    /// The edges [`edges`](Self::edges) lists for `node` whose other end is
    /// `other`, in the order it lists them: the closing hop of a cycle, which
    /// knows the node it comes back to and needs no list of the others.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn edges_between(
        &self,
        node: NodeId,
        other: NodeId,
        direction: Direction,
        types: &[ArcStr],
        view: ReadView,
    ) -> Result<Vec<EdgeId>> {
        let mut edges = self.edges(node, direction, types, view)?;
        edges.retain(|(end, _)| *end == other);
        Ok(edges.into_iter().map(|(_, edge)| edge).collect())
    }

    /// Whether `view` sees an edge of `node` in `direction` of one of
    /// `types` (any type when empty), as [`edges`](Self::edges) would list
    /// one: an `EXISTS` of one edge needs no more than the first.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn has_edge(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        view: ReadView,
    ) -> Result<bool> {
        Ok(!self.edges(node, direction, types, view)?.is_empty())
    }

    /// How many edges [`edges`](Self::edges) lists for `node` in `direction`
    /// of one of `types` (every type when empty) whose other end has every
    /// label of `end_labels` as `view` sees it: the count of
    /// `(node)-[:types]->(:end_labels)`, without the list.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn count_edges(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        end_labels: &[ArcStr],
        view: ReadView,
    ) -> Result<usize> {
        let mut count = 0;
        for (other, _) in self.edges(node, direction, types, view)? {
            let mut matches = true;
            for label in end_labels {
                if self.node_has_label(other, label, view)? != Some(true) {
                    matches = false;
                    break;
                }
            }
            count += usize::from(matches);
        }
        Ok(count)
    }

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

    /// Whether [`nodes_by_label_count`](Self::nodes_by_label_count) of
    /// `label` answers for `view` from a count the store keeps, without
    /// reading the label's nodes: what a query that counts reports as read.
    /// `false` unless the store keeps such counts.
    fn label_count_is_kept(&self, _label: &str, _view: ReadView) -> bool {
        false
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

    /// The labels of [`all_labels`](Self::all_labels) that a node `view`
    /// sees has: a label whose last node went is no longer one.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn labels_in_use(&self, view: ReadView) -> Result<Vec<String>> {
        let mut in_use = Vec::new();
        for label in self.all_labels()? {
            if self.nodes_by_label_count(&label, view)? > 0 {
                in_use.push(label);
            }
        }
        Ok(in_use)
    }

    /// The edge types of [`all_edge_types`](Self::all_edge_types) that an
    /// edge `view` sees has. The default reads every node's outgoing edges.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn edge_types_in_use(&self, view: ReadView) -> Result<Vec<String>> {
        let mut in_use = std::collections::BTreeSet::new();
        for node in self.node_ids(view)? {
            for (_, edge) in self.edges(node, Direction::Outgoing, &[], view)? {
                if let Some(edge_type) = self.edge_type(edge, view)? {
                    in_use.insert(edge_type.to_string());
                }
            }
        }
        Ok(in_use.into_iter().collect())
    }

    /// Every property key the store holds, of nodes and edges (see
    /// [`all_labels`](Self::all_labels)).
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn all_property_keys(&self) -> Result<Vec<String>>;

    // --- Lookups ---

    /// Whether the store keeps a property index on `property`, on every label or on one. Not
    /// an answer: the lookups answer alike either way. For what a lookup can read, see
    /// [`index_serves`](Self::index_serves).
    fn has_property_index(&self, _property: &str) -> bool {
        false
    }

    /// Whether a lookup of `property` whose results all have each of `labels` has an index to
    /// read: one on a label among `labels`, or one on every label. The planner then prefers a
    /// lookup to a scan. A store with no index on one label has only the second kind.
    fn index_serves(&self, _labels: &[&str], property: &str) -> bool {
        self.has_property_index(property)
    }

    /// Whether [`key_column_group`](Self::key_column_group) answers.
    fn reads_key_columns(&self) -> bool {
        false
    }

    /// The rows from node id `from` that the store reads at once, a row group or part of one,
    /// read for `query`: every node of them `view` sees, with the query's labels as `view` sees
    /// them, whose value of the key as `view` sees it `=` may find equal to one of the query's
    /// values, and of those labels' nodes maybe others, ascending. A store's rows are grouped as
    /// it likes; the answer says where the next read starts. `query` keeps what a next read of
    /// the same rows can use.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do, and of a store that does not read key
    /// columns (see [`reads_key_columns`](Self::reads_key_columns)).
    fn key_column_group(
        &self,
        _query: &mut super::rowgroup::KeyColumnQuery,
        _from: u64,
        _view: ReadView,
    ) -> Result<KeyColumnGroup> {
        Err(grafeo_common::utils::error::Error::Internal(
            "this store does not read key columns".to_string(),
        ))
    }

    /// What a walk back from a node of `far_label` reads over `types` in `back`, by the far side:
    /// a sample of the label's lists, read for their lengths alone, weighted by their edges, plus
    /// the edges of `types` the sample's mean does not account for, which may be one hub's. Edges
    /// to other labels count as such too, so it reads high. It spends the reads it takes from
    /// `budget`. `None` for a store that cannot read lengths, a label with no node, no types, or
    /// too little budget.
    fn far_walk_back_within(
        &self,
        _far_label: &str,
        _types: &[&str],
        _back: Direction,
        _budget: &mut u64,
    ) -> Option<f64> {
        None
    }

    /// How many distinct values the index a lookup of `property` among nodes with each of
    /// `labels` reads holds (see [`index_serves`](Self::index_serves)), for the planner to judge
    /// how many nodes a value finds. It may read high: values only old versions hold count until
    /// they are collected. `None` for a store that cannot tell, or without such an index.
    fn indexed_distinct_values(&self, _labels: &[&str], _property: &str) -> Option<u64> {
        None
    }

    /// The most edges [`edges`](Self::edges) can give for `node`, without going through them:
    /// removed and unseen ones may count, and under [`Direction::Both`] a self-loop twice. `None`
    /// for a store that cannot tell without reading them.
    fn edge_entries_at_most(
        &self,
        _node: NodeId,
        _direction: Direction,
        _types: &[ArcStr],
        _view: ReadView,
    ) -> Option<usize> {
        None
    }

    /// Whether `label` has a property index of its own on `property`.
    fn has_label_index(&self, _label: &str, _property: &str) -> bool {
        false
    }

    /// The property indexes the store keeps on one label, as `(label, property)`.
    fn label_index_entries(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// The mean count of `edge_type`'s edges in `direction` over a sample of
    /// `label`'s nodes, as committed, for the planner to judge a hop: no open
    /// transaction's writes count, and the same data gives the same value. `None`
    /// for a store that cannot sample, a label without a node or a read that fails.
    fn sampled_degree(&self, _label: &str, _edge_type: &str, _direction: Direction) -> Option<f64> {
        None
    }

    /// The largest count [`sampled_degree`](Self::sampled_degree)'s sample saw, from the same
    /// read: a node's list it may miss is longer. `None` as there.
    fn sampled_max_degree(
        &self,
        _label: &str,
        _edge_type: &str,
        _direction: Direction,
    ) -> Option<f64> {
        None
    }

    /// The mean count of `on_types`' edges in `on_direction` at the ends a hop reaches:
    /// over the edges of `via_types` in `via_direction` from a sample of `start_label`'s
    /// nodes (of all nodes for `None`), as committed, each end counted as often as an edge
    /// reaches it. So a hub many edges reach weighs what a path through them meets. A mean
    /// of 0 stands only when every edge was read. `None` for a store that cannot sample, a
    /// read that fails, or a 0 from part of the edges.
    fn sampled_reached_degree(
        &self,
        start_label: Option<&str>,
        via_types: &[&str],
        via_direction: Direction,
        on_types: &[&str],
        on_direction: Direction,
    ) -> Option<f64> {
        let mut unlimited = u64::MAX;
        self.sampled_reached_degree_within(
            start_label,
            via_types,
            via_direction,
            on_types,
            on_direction,
            &mut unlimited,
        )
    }

    /// [`sampled_reached_degree`](Self::sampled_reached_degree) that spends the reads it
    /// takes from `budget`, and reads nothing with none left: a value held costs nothing.
    fn sampled_reached_degree_within(
        &self,
        _start_label: Option<&str>,
        _via_types: &[&str],
        _via_direction: Direction,
        _on_types: &[&str],
        _on_direction: Direction,
        _budget: &mut u64,
    ) -> Option<f64> {
        None
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
    /// found through the property's index on every label: every node `view` sees that `=`
    /// finds equal, and maybe others it sees, for a filter to decide. `None`
    /// without one: the caller then scans.
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

    /// [`find_nodes_maybe_equal`](Self::find_nodes_maybe_equal), for a caller
    /// that would rather not read a common key: `None` also when the index
    /// holds more than `limit` candidates for `value`, known before any of them
    /// is read.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_few_nodes_maybe_equal(
        &self,
        _property: &str,
        _value: &Value,
        _limit: usize,
        _view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        Ok(None)
    }

    /// [`find_nodes_maybe_equal`](Self::find_nodes_maybe_equal) for a caller whose results
    /// all have each of `labels`, which it checks itself: through the index on the first of them
    /// that has one on `property`, else the one on every label. `None` without either.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_indexed_maybe_equal(
        &self,
        _labels: &[&str],
        property: &str,
        value: &Value,
        view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        self.find_nodes_maybe_equal(property, value, view)
    }

    /// The kinds of value (see
    /// [`ValueKinds`](crate::execution::operators::value_utils::ValueKinds)) the index that
    /// [`find_indexed_maybe_equal`](Self::find_indexed_maybe_equal) reads for `labels` and
    /// `property` has held since it was made, deleted values too; `None` without that index, or
    /// for a store that does not keep them, which a statement whose `=` raises across kinds then
    /// does not read.
    fn index_kinds(
        &self,
        _labels: &[&str],
        _property: &str,
    ) -> Option<crate::execution::operators::value_utils::ValueKinds> {
        None
    }

    /// [`find_few_nodes_maybe_equal`](Self::find_few_nodes_maybe_equal) through the index
    /// [`find_indexed_maybe_equal`](Self::find_indexed_maybe_equal) reads.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_few_indexed_maybe_equal(
        &self,
        _labels: &[&str],
        property: &str,
        value: &Value,
        limit: usize,
        view: ReadView,
    ) -> Result<Option<Vec<NodeId>>> {
        self.find_few_nodes_maybe_equal(property, value, limit, view)
    }

    /// [`find_nodes_by_properties`](Self::find_nodes_by_properties) for a caller whose results
    /// all have each of `labels`, which it checks itself: the indexes it reads are those
    /// [`find_indexed_maybe_equal`](Self::find_indexed_maybe_equal) reads.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_indexed_by_properties(
        &self,
        _labels: &[&str],
        conditions: &[(&str, Value)],
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        self.find_nodes_by_properties(conditions, view)
    }

    /// The nodes `view` sees whose `property` lies within `range`, by
    /// ascending id. Values compare as a filter of the range's rule compares
    /// them (see [`ValueRange`]).
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    fn find_nodes_in_range(
        &self,
        property: &str,
        range: &ValueRange,
        view: ReadView,
    ) -> Result<Vec<NodeId>> {
        let mut found = Vec::new();
        for id in self.node_ids(view)? {
            if let Some(value) = self.node_property(id, property, view)?
                && range.contains(&value)
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

    /// The statistics the cost-based optimizer plans with: estimates, not
    /// answers. A store may build them on each call (the row-group store
    /// walks every label and edge type), so a plan of one statement reads
    /// what it needs through [`label_node_estimate`](Self::label_node_estimate)
    /// and [`node_estimate`](Self::node_estimate) instead.
    fn statistics(&self) -> Arc<Statistics>;

    /// How many nodes have `label` by the [`statistics`](Self::statistics):
    /// 0 for a label they do not list. The physical planner chooses scan
    /// labels by it on every execution, so a store that keeps its counts
    /// answers without building the statistics.
    fn label_node_estimate(&self, label: &str) -> u64 {
        self.statistics()
            .get_label(label)
            .map_or(0, |label| label.node_count)
    }

    /// How many nodes the store holds by the [`statistics`](Self::statistics).
    fn node_estimate(&self) -> u64 {
        self.statistics().total_nodes
    }

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
    /// Returns the error of a read the store cannot do. The default, for a
    /// store without text search, is the unsupported error: `text_match` and
    /// `text_score` in a query then fail, where an absent score made them
    /// match nothing (#308).
    #[cfg(feature = "text-index")]
    fn score_text(
        &self,
        _id: NodeId,
        _label: &str,
        _property: &str,
        _query: &str,
        _view: ReadView,
    ) -> Result<Option<f64>> {
        Err(unsupported("text search"))
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

    /// Whether the store keeps a vector index on `property` for any label: a
    /// writer then reads the labels of a node it writes a list to, which the
    /// index may take as a vector. A store that keeps vector indexes
    /// implements it with [`vector_index_config`](Self::vector_index_config).
    #[cfg(feature = "vector-index")]
    fn has_vector_index_on(&self, property: &str) -> bool {
        let _ = property;
        false
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
    grafeo_common::utils::error::Error::from(grafeo_common::utils::error::QueryError::unsupported(
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

/// The bounds of a range read of a property, and the rule by which a value of another kind than a
/// bound compares with it: the rule of the statement's `<`, so a range read and a filter agree
/// (ISO GQL orders two kinds, openCypher does not; see [`AcrossKinds`]).
///
/// [`AcrossKinds`]: crate::execution::operators::value_utils::AcrossKinds
#[derive(Debug, Clone)]
pub struct ValueRange {
    min: Option<Value>,
    max: Option<Value>,
    min_inclusive: bool,
    max_inclusive: bool,
    across_kinds: crate::execution::operators::value_utils::AcrossKinds,
}

impl ValueRange {
    /// A range between the bounds (`None`: open), each inclusive or not, read by `across_kinds`.
    #[must_use]
    pub fn new(
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
        across_kinds: crate::execution::operators::value_utils::AcrossKinds,
    ) -> Self {
        Self {
            min: min.cloned(),
            max: max.cloned(),
            min_inclusive,
            max_inclusive,
            across_kinds,
        }
    }

    /// Whether `value` lies between the bounds: of one kind with a bound, as a filter compares
    /// them; of another, by the range's rule.
    #[must_use]
    pub fn contains(&self, value: &Value) -> bool {
        let order = |bound: &Value| self.order(value, bound);
        let above = self.min.as_ref().is_none_or(|min| match order(min) {
            Some(CmpOrdering::Greater) => true,
            Some(CmpOrdering::Equal) => self.min_inclusive,
            _ => false,
        });
        let below = self.max.as_ref().is_none_or(|max| match order(max) {
            Some(CmpOrdering::Less) => true,
            Some(CmpOrdering::Equal) => self.max_inclusive,
            _ => false,
        });
        above && below
    }

    /// The order of `value` against `bound` as the statement's `<` finds it; `None` when it is
    /// unknown or neither less nor greater. Two lists, or two paths, at their first pair of
    /// elements that are not equal, else by their lengths (see
    /// [`walk_elements`](crate::execution::operators::value_utils::walk_elements)).
    fn order(&self, value: &Value, bound: &Value) -> Option<CmpOrdering> {
        use crate::execution::operators::value_utils::{
            Walk, equals, order_beyond_kind, sequences, walk_elements,
        };
        if let Some((a, b)) = sequences(value, bound) {
            let temporal = grafeo_common::types::TemporalComparison::default();
            return match walk_elements(a, b, |x, y| {
                (equals(x, y, temporal) != Some(true)).then(|| self.order(x, y))
            }) {
                Walk::Decided(order) => order,
                Walk::Lengths(order) => Some(order),
            };
        }
        compare_for_range(value, bound)
            .or_else(|| order_beyond_kind(value, bound, self.across_kinds))
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
        // The edge counts: Mia's self-loop once both ways (and once each
        // way on its own), a label the far end must have, a type the edge
        // must have.
        let types = |names: &[&str]| -> Vec<ArcStr> {
            names.iter().map(|name| ArcStr::from(*name)).collect()
        };
        let count = |node, direction, edge_types: &[&str], end_labels: &[&str]| {
            store
                .count_edges(node, direction, &types(edge_types), &types(end_labels), NOW)
                .unwrap()
        };
        assert_eq!(count(mia, Direction::Both, &[], &[]), 2, "Gus, the loop");
        assert_eq!(count(mia, Direction::Outgoing, &[], &[]), 1, "the loop");
        assert_eq!(
            count(mia, Direction::Incoming, &[], &[]),
            2,
            "Gus, the loop"
        );
        assert_eq!(count(mia, Direction::Both, &[], &["Person"]), 2);
        assert_eq!(count(mia, Direction::Both, &[], &["City"]), 0);
        assert_eq!(count(alix, Direction::Outgoing, &[], &["City"]), 1);
        assert_eq!(count(alix, Direction::Outgoing, &[], &["Person"]), 1);
        assert_eq!(
            count(alix, Direction::Outgoing, &[], &["Person", "City"]),
            0,
            "every end label"
        );
        assert_eq!(count(alix, Direction::Outgoing, &["KNOWS"], &[]), 1);
        assert_eq!(count(alix, Direction::Outgoing, &["LIVES_IN"], &[]), 1);
        assert_eq!(
            count(alix, Direction::Outgoing, &["KNOWS", "LIVES_IN"], &[]),
            2,
            "any of the types"
        );
        assert_eq!(
            count(alix, Direction::Outgoing, &["LIVES_IN"], &["Person"]),
            0
        );
        assert_eq!(count(mia, Direction::Both, &["LIVES_IN"], &[]), 0);
        assert_eq!(count(alix, Direction::Incoming, &[], &[]), 0);
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
                    &crate::graph::ValueRange::new(
                        Some(&Value::Float64(3.0)),
                        Some(&Value::Int64(88)),
                        false,
                        true,
                        crate::execution::operators::value_utils::AcrossKinds::Apart
                    ),
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
}
