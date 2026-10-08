//! Storage traits for the graph engine.
//!
//! These traits capture the minimal surface that query operators need from
//! the graph store. The split is intentional:
//!
//! - [`GraphStore`]: Read-only operations (scans, lookups, traversal, statistics)
//! - [`GraphStoreMut`]: Write operations (create, delete, mutate)
//!
//! Admin operations (index management, MVCC internals, schema introspection,
//! statistics recomputation, WAL recovery) stay on the concrete [`LpgStore`]
//! and are not part of these traits.
//!
//! ## Design rationale
//!
//! The traits work with typed graph objects (`Node`, `Edge`, `Value`) rather
//! than raw bytes. This preserves zero-overhead access for in-memory storage
//! while allowing future backends (SpilloverStore, disk-backed) to implement
//! the same interface with transparent serialization where needed.
//!
//! [`LpgStore`]: crate::graph::lpg::LpgStore

use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
#[cfg(feature = "vector-index")]
use crate::index::vector::DistanceMetric;
use crate::statistics::Statistics;
use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashMap;
use std::sync::Arc;

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

    /// Returns the number of non-deleted nodes with a specific label.
    ///
    /// Default falls back to `self.nodes_by_label(label).len()`; stores that
    /// maintain a per-label index should override for an O(1) count without
    /// allocating the full ID list.
    fn nodes_by_label_count(&self, label: &str) -> usize {
        self.nodes_by_label(label).len()
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
    /// Returns an error, and deletes nothing, when the node's properties
    /// (which a rollback restores) cannot be read: a spilled value whose file
    /// cannot be read.
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

    /// Sets a node property within a transaction, recording the previous value
    /// so it can be restored on rollback.
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

    /// Sets an edge property within a transaction, recording the previous value
    /// so it can be restored on rollback.
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

    /// Removes a node property within a transaction, recording the previous value
    /// so it can be restored on rollback. Returns the previous value if it
    /// existed.
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

    /// Removes an edge property within a transaction, recording the previous value
    /// so it can be restored on rollback.
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

    /// Adds a label within a transaction, recording the change for rollback.
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

    /// Removes a label within a transaction, recording the change for rollback.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn null_graph_store_point_lookups() {
        let store = NullGraphStore;
        let nid = NodeId(1);
        let eid = EdgeId(1);
        let epoch = EpochId(0);
        let txn = TransactionId(1);

        assert!(store.get_node(nid).is_none());
        assert!(store.get_edge(eid).is_none());
        assert!(store.get_node_versioned(nid, epoch, txn).is_none());
        assert!(store.get_edge_versioned(eid, epoch, txn).is_none());
        assert!(store.get_node_at_epoch(nid, epoch).is_none());
        assert!(store.get_edge_at_epoch(eid, epoch).is_none());
    }

    #[test]
    fn null_graph_store_property_access() {
        let store = NullGraphStore;
        let nid = NodeId(1);
        let eid = EdgeId(1);
        let key = PropertyKey::from("name");

        assert!(store.get_node_property(nid, &key).is_none());
        assert!(store.get_edge_property(eid, &key).is_none());
        assert_eq!(
            store.get_node_property_batch(&[nid, NodeId(2)], &key),
            vec![None, None]
        );

        let node_props = store.get_nodes_properties_batch(&[nid]);
        assert_eq!(node_props.len(), 1);
        assert!(node_props[0].is_empty());

        let selective =
            store.get_nodes_properties_selective_batch(&[nid], std::slice::from_ref(&key));
        assert_eq!(selective.len(), 1);
        assert!(selective[0].is_empty());

        let edge_selective = store.get_edges_properties_selective_batch(&[eid], &[key]);
        assert_eq!(edge_selective.len(), 1);
        assert!(edge_selective[0].is_empty());
    }

    #[test]
    fn null_graph_store_traversal() {
        let store = NullGraphStore;
        let nid = NodeId(1);

        assert!(
            store.neighbors(nid, Direction::Outgoing).is_empty(),
            "expected empty"
        );
        assert!(
            store.edges_from(nid, Direction::Incoming).is_empty(),
            "expected empty"
        );
        assert_eq!(store.out_degree(nid), 0);
        assert_eq!(store.in_degree(nid), 0);
        assert!(!store.has_backward_adjacency());
    }

    #[test]
    fn null_graph_store_scans_and_counts() {
        let store = NullGraphStore;

        assert!(store.node_ids().is_empty(), "{:?}", store.node_ids());
        assert!(
            store.all_node_ids().is_empty(),
            "{:?}",
            store.all_node_ids()
        );
        assert!(store.nodes_by_label("Person").is_empty(), "expected empty");
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn null_graph_store_metadata_and_schema() {
        let store = NullGraphStore;
        let eid = EdgeId(1);
        let epoch = EpochId(0);
        let txn = TransactionId(1);

        assert!(store.edge_type(eid).is_none());
        assert!(store.edge_type_versioned(eid, epoch, txn).is_none());
        assert!(!store.has_property_index("name"));
        assert!(store.all_labels().is_empty(), "{:?}", store.all_labels());
        assert!(
            store.all_edge_types().is_empty(),
            "{:?}",
            store.all_edge_types()
        );
        assert!(
            store.all_property_keys().is_empty(),
            "{:?}",
            store.all_property_keys()
        );
    }

    #[test]
    fn null_graph_store_search() {
        let store = NullGraphStore;
        let key = PropertyKey::from("age");
        let val = Value::Int64(30);

        assert!(
            store.find_nodes_by_property("age", &val).is_empty(),
            "expected empty"
        );
        assert!(
            store
                .find_nodes_by_properties(&[("age", val.clone())])
                .is_empty(),
            "expected no nodes"
        );
        assert!(
            store
                .find_nodes_in_range("age", Some(&val), None, true, false)
                .is_empty(),
            "expected no nodes"
        );
        assert!(!store.node_property_might_match(&key, CompareOp::Eq, &val));
        assert!(!store.edge_property_might_match(&key, CompareOp::Eq, &val));
    }

    #[test]
    fn null_graph_store_statistics() {
        let store = NullGraphStore;

        let _stats = store.statistics();
        assert_eq!(store.estimate_label_cardinality("Person"), 0.0);
        assert_eq!(store.estimate_avg_degree("KNOWS", true), 0.0);
        assert_eq!(store.current_epoch(), EpochId(0));
    }

    #[test]
    fn null_graph_store_visibility() {
        let store = NullGraphStore;
        let nid = NodeId(1);
        let eid = EdgeId(1);
        let epoch = EpochId(0);
        let txn = TransactionId(1);

        assert!(!store.is_node_visible_at_epoch(nid, epoch));
        assert!(!store.is_node_visible_versioned(nid, epoch, txn));
        assert!(!store.is_edge_visible_at_epoch(eid, epoch));
        assert!(!store.is_edge_visible_versioned(eid, epoch, txn));

        assert!(
            store
                .filter_visible_node_ids(&[nid, NodeId(2)], epoch)
                .is_empty(),
            "expected no visible nodes"
        );
        assert!(
            store
                .filter_visible_node_ids_versioned(&[nid], epoch, txn)
                .is_empty(),
            "expected no visible nodes"
        );
    }

    #[test]
    fn null_graph_store_history() {
        let store = NullGraphStore;

        assert!(store.get_node_history(NodeId(1)).is_empty());
        assert!(store.get_edge_history(EdgeId(1)).is_empty());
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
