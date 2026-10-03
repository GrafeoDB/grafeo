//! CDC-aware graph store wrapper.
//!
//! Wraps a [`GraphStoreMut`] and buffers CDC events for every mutation.
//! Events are held in a transactional buffer (`pending_events`) that the
//! session flushes to the [`CdcLog`] on commit or discards on rollback.
//!
//! This mirrors the [`WalGraphStore`](super::wal_store::WalGraphStore)
//! decorator pattern but targets the CDC audit trail instead of WAL
//! durability.

use std::collections::HashMap;
use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{
    EdgeId, EpochId, HlcTimestamp, NodeId, PropertyKey, PropertyMap, TransactionId, Value,
};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::lpg::{CompareOp, Edge, Node};
use grafeo_core::graph::{Direction, GraphStore, GraphStoreMut, GraphStoreSearch};
use grafeo_core::statistics::Statistics;
use parking_lot::Mutex;

use crate::cdc::{CdcLog, ChangeEvent, ChangeKind, EntityId, PendingEvent};

/// A [`GraphStoreMut`] decorator that buffers CDC events for every mutation.
///
/// Read-only methods are forwarded to the inner store without CDC interaction.
///
/// Versioned (transactional) mutations buffer events into `pending_events`.
/// The owning session flushes this buffer to `CdcLog` on commit or clears it
/// on rollback.
///
/// Non-versioned mutations (used by the direct CRUD API) record directly to
/// `CdcLog` since they have no transaction context and are immediately visible.
pub(crate) struct CdcGraphStore {
    inner: Arc<dyn GraphStoreMut>,
    cdc_log: Arc<CdcLog>,
    /// Buffered events for the current transaction.
    pending_events: Arc<Mutex<Vec<PendingEvent>>>,
    /// Whether the events of non-versioned writes are buffered too, instead
    /// of being recorded as they happen.
    buffer_all: bool,
    /// The named graph this store writes to, `None` for the default graph:
    /// the events it buffers carry it.
    graph: Option<String>,
}

impl CdcGraphStore {
    /// Creates a new CDC-aware store with a fresh event buffer.
    pub fn new(inner: Arc<dyn GraphStoreMut>, cdc_log: Arc<CdcLog>) -> Self {
        Self {
            inner,
            cdc_log,
            pending_events: Arc::new(Mutex::new(Vec::new())),
            buffer_all: false,
            graph: None,
        }
    }

    /// Wraps a store sharing an existing event buffer.
    ///
    /// Used for named graphs so all mutations in a transaction (across
    /// default and named graphs) buffer to the same `Vec` for atomic
    /// flush/discard.
    pub fn wrap(
        inner: Arc<dyn GraphStoreMut>,
        cdc_log: Arc<CdcLog>,
        pending_events: Arc<Mutex<Vec<PendingEvent>>>,
    ) -> Self {
        Self {
            inner,
            cdc_log,
            pending_events,
            buffer_all: false,
            graph: None,
        }
    }

    /// The same store, for the named graph `graph`: the events it buffers
    /// say so, so the commit folds them per graph.
    #[must_use]
    pub fn for_graph(mut self, graph: String) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Wraps a store sharing an existing event buffer, and buffers the events
    /// of non-versioned writes there too: for a direct write outside any
    /// transaction, which records all of its events at once, merged and
    /// stamped with its epoch.
    pub fn wrap_buffered(
        inner: Arc<dyn GraphStoreMut>,
        cdc_log: Arc<CdcLog>,
        pending_events: Arc<Mutex<Vec<PendingEvent>>>,
    ) -> Self {
        Self {
            inner,
            cdc_log,
            pending_events,
            buffer_all: true,
            graph: None,
        }
    }

    /// Returns a handle to the pending events buffer.
    pub fn pending_events(&self) -> Arc<Mutex<Vec<PendingEvent>>> {
        Arc::clone(&self.pending_events)
    }

    /// Buffers a CDC event for later flush on commit.
    ///
    /// The epoch is always set to `PENDING`: the real commit epoch is assigned
    /// when the session flushes the buffer in `commit_inner()`. This ensures
    /// each transaction's events get the unique epoch from `fetch_add(1, SeqCst)`.
    fn buffer_event(&self, mut event: ChangeEvent) {
        event.epoch = EpochId::PENDING;
        self.pending_events.lock().push(PendingEvent {
            graph: self.graph.clone(),
            event,
        });
    }

    /// Records a CDC event directly (for non-versioned/auto-commit mutations),
    /// or buffers it when the store buffers every event.
    fn record_directly(&self, event: ChangeEvent) {
        if self.buffer_all {
            self.buffer_event(event);
        } else {
            self.cdc_log.record(event);
        }
    }

    /// The node as a change sees it: through `transaction_id` for a
    /// transactional change, so a node the transaction wrote itself is seen
    /// as it left it, and as committed otherwise.
    fn node_as_seen(&self, id: NodeId, transaction_id: Option<TransactionId>) -> Option<Node> {
        match transaction_id {
            Some(transaction_id) => {
                self.inner
                    .get_node_versioned(id, self.inner.current_epoch(), transaction_id)
            }
            None => self.inner.get_node(id),
        }
    }

    /// The edge as a change sees it, like [`node_as_seen`](Self::node_as_seen).
    fn edge_as_seen(&self, id: EdgeId, transaction_id: Option<TransactionId>) -> Option<Edge> {
        match transaction_id {
            Some(transaction_id) => {
                self.inner
                    .get_edge_versioned(id, self.inner.current_epoch(), transaction_id)
            }
            None => self.inner.get_edge(id),
        }
    }

    /// The labels of a node as a change sees it; none for a missing node.
    fn labels_as_seen(&self, id: NodeId, transaction_id: Option<TransactionId>) -> Vec<String> {
        self.node_as_seen(id, transaction_id)
            .map(|node| label_names(&node))
            .unwrap_or_default()
    }

    /// Returns the next HLC timestamp from the CDC log's clock.
    fn next_ts(&self) -> HlcTimestamp {
        self.cdc_log.next_timestamp()
    }
}

fn make_event(
    entity_id: EntityId,
    kind: ChangeKind,
    epoch: EpochId,
    timestamp: HlcTimestamp,
) -> ChangeEvent {
    ChangeEvent {
        entity_id,
        kind,
        epoch,
        timestamp,
        before: None,
        after: None,
        labels: None,
        before_labels: None,
        edge_type: None,
        src_id: None,
        dst_id: None,
        triple_subject: None,
        triple_predicate: None,
        triple_object: None,
        triple_graph: None,
    }
}

/// An entity's properties as an event snapshot; `None` when it has none.
fn property_snapshot(properties: &PropertyMap) -> Option<HashMap<String, Value>> {
    if properties.is_empty() {
        return None;
    }
    Some(
        properties
            .iter()
            .map(|(key, value)| (key.as_str().to_string(), value.clone()))
            .collect(),
    )
}

fn label_names(node: &Node) -> Vec<String> {
    node.labels.iter().map(ToString::to_string).collect()
}

/// Describes a deleted node: the labels and properties it had.
fn describe_deleted_node(event: &mut ChangeEvent, node: &Node) {
    event.labels = Some(label_names(node));
    event.before = property_snapshot(&node.properties);
}

/// Describes an edge: its type and endpoints.
fn describe_edge(event: &mut ChangeEvent, edge_type: &str, src: NodeId, dst: NodeId) {
    event.edge_type = Some(edge_type.to_string());
    event.src_id = Some(src.as_u64());
    event.dst_id = Some(dst.as_u64());
}

/// Describes a deleted edge: its type, endpoints and last properties.
fn describe_deleted_edge(event: &mut ChangeEvent, edge: &Edge) {
    describe_edge(event, &edge.edge_type, edge.src, edge.dst);
    event.before = property_snapshot(&edge.properties);
}

/// Describes a label change: the labels before and after it.
fn describe_label_change(event: &mut ChangeEvent, before: Vec<String>, after: Vec<String>) {
    event.before_labels = Some(before);
    event.labels = Some(after);
}

fn with_label(labels: &[String], label: &str) -> Vec<String> {
    let mut labels = labels.to_vec();
    labels.push(label.to_string());
    labels
}

fn without_label(labels: &[String], label: &str) -> Vec<String> {
    labels.iter().filter(|l| *l != label).cloned().collect()
}

// ---------------------------------------------------------------------------
// GraphStore (read-only): pure delegation
// ---------------------------------------------------------------------------

impl GraphStore for CdcGraphStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        self.inner.get_node(id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.inner.get_edge(id)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        self.inner.get_node_versioned(id, epoch, transaction_id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.inner.get_edge_versioned(id, epoch, transaction_id)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        self.inner.get_node_at_epoch(id, epoch)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        self.inner.get_edge_at_epoch(id, epoch)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.inner.get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.inner.get_edge_property(id, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        self.inner.get_node_property_batch(ids, key)
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_nodes_properties_batch(ids)
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_nodes_properties_selective_batch(ids, keys)
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.inner.get_edges_properties_selective_batch(ids, keys)
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        self.inner.neighbors(node, direction)
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        self.inner.edges_from(node, direction)
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.inner.out_degree(node)
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.inner.in_degree(node)
    }

    fn has_backward_adjacency(&self) -> bool {
        self.inner.has_backward_adjacency()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.node_ids()
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        self.inner.all_node_ids()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        self.inner.nodes_by_label(label)
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        self.inner.nodes_by_label_count(label)
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        self.inner.edge_type(id)
    }

    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        self.inner.edge_type_versioned(id, epoch, transaction_id)
    }

    fn has_property_index(&self, property: &str) -> bool {
        self.inner.has_property_index(property)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        self.inner.find_nodes_by_property(property, value)
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        self.inner.find_nodes_by_properties(conditions)
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        self.inner
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.node_property_might_match(property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.inner.edge_property_might_match(property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        self.inner.statistics()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.inner.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.inner.estimate_avg_degree(edge_type, outgoing)
    }

    fn current_epoch(&self) -> EpochId {
        self.inner.current_epoch()
    }

    fn all_labels(&self) -> Vec<String> {
        self.inner.all_labels()
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.inner.all_edge_types()
    }

    fn all_property_keys(&self) -> Vec<String> {
        self.inner.all_property_keys()
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        self.inner.is_node_visible_at_epoch(id, epoch)
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .is_node_visible_versioned(id, epoch, transaction_id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        self.inner.is_edge_visible_at_epoch(id, epoch)
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        self.inner
            .is_edge_visible_versioned(id, epoch, transaction_id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        self.inner.filter_visible_node_ids(ids, epoch)
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        self.inner
            .filter_visible_node_ids_versioned(ids, epoch, transaction_id)
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        self.inner.get_node_history(id)
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        self.inner.get_edge_history(id)
    }
}

// Pure delegation: CDC wraps the store to buffer mutation events but owns no
// index state, so every text/vector lookup has to fall through to the
// underlying `GraphStoreMut` (which is a `GraphStoreSearch` by bound). A stub
// impl silently turns into "no index exists" at every call site — the same
// regression that hit the WAL wrapper in issue #308.
impl GraphStoreSearch for CdcGraphStore {
    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.inner.has_text_index(label, property)
    }

    #[cfg(feature = "text-index")]
    fn score_text(&self, node_id: NodeId, label: &str, property: &str, query: &str) -> Option<f64> {
        self.inner.score_text(node_id, label, property, query)
    }

    #[cfg(feature = "text-index")]
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Vec<(NodeId, f64)> {
        self.inner.text_search(label, property, query, k)
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
    ) -> Vec<(NodeId, f64)> {
        self.inner
            .text_search_with_threshold(label, property, query, threshold)
    }

    #[cfg(feature = "vector-index")]
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        self.inner.has_vector_index(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_index_config(
        &self,
        label: &str,
        property: &str,
    ) -> Option<grafeo_core::index::vector::HnswConfig> {
        self.inner.vector_index_config(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        k: usize,
        metric: grafeo_core::index::vector::DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        self.inner.vector_search(label, property, query, k, metric)
    }

    #[cfg(feature = "vector-index")]
    fn vector_search_with_threshold(
        &self,
        label: Option<&str>,
        property: &str,
        query: &[f32],
        threshold: f64,
        metric: grafeo_core::index::vector::DistanceMetric,
    ) -> Vec<(NodeId, f64)> {
        self.inner
            .vector_search_with_threshold(label, property, query, threshold, metric)
    }
}

// ---------------------------------------------------------------------------
// GraphStoreMut: delegate + CDC buffer/record
// ---------------------------------------------------------------------------

impl GraphStoreMut for CdcGraphStore {
    // --- Node creation ---

    fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = self.inner.create_node(labels);
        let epoch = self.inner.current_epoch();
        let mut event = make_event(
            EntityId::Node(id),
            ChangeKind::Create,
            epoch,
            self.next_ts(),
        );
        event.labels = Some(labels.iter().map(|s| (*s).to_string()).collect());
        self.record_directly(event);
        id
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self
            .inner
            .create_node_versioned(labels, epoch, transaction_id);
        // Use PENDING epoch: the real commit epoch is assigned during flush.
        let mut event = make_event(
            EntityId::Node(id),
            ChangeKind::Create,
            EpochId::PENDING,
            self.next_ts(),
        );
        event.labels = Some(labels.iter().map(|s| (*s).to_string()).collect());
        self.buffer_event(event);
        id
    }

    // --- Edge creation ---

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let id = self.inner.create_edge(src, dst, edge_type);
        let epoch = self.inner.current_epoch();
        let mut event = make_event(
            EntityId::Edge(id),
            ChangeKind::Create,
            epoch,
            self.next_ts(),
        );
        describe_edge(&mut event, edge_type, src, dst);
        self.record_directly(event);
        id
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        let id = self
            .inner
            .create_edge_versioned(src, dst, edge_type, epoch, transaction_id);
        let mut event = make_event(
            EntityId::Edge(id),
            ChangeKind::Create,
            epoch,
            self.next_ts(),
        );
        describe_edge(&mut event, edge_type, src, dst);
        self.buffer_event(event);
        id
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        let ids = self.inner.batch_create_edges(edges);
        let epoch = self.inner.current_epoch();
        for (id, (src, dst, edge_type)) in ids.iter().zip(edges) {
            let mut event = make_event(
                EntityId::Edge(*id),
                ChangeKind::Create,
                epoch,
                self.next_ts(),
            );
            describe_edge(&mut event, edge_type, *src, *dst);
            self.record_directly(event);
        }
        ids
    }

    // --- Deletion ---

    fn delete_node(&self, id: NodeId) -> bool {
        let node = self.node_as_seen(id, None);
        let deleted = self.inner.delete_node(id);
        if deleted {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Node(id),
                ChangeKind::Delete,
                epoch,
                self.next_ts(),
            );
            if let Some(node) = &node {
                describe_deleted_node(&mut event, node);
            }
            self.record_directly(event);
        }
        deleted
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let node = self.node_as_seen(id, Some(transaction_id));
        let deleted = self.inner.delete_node_versioned(id, epoch, transaction_id);
        if deleted {
            let mut event = make_event(
                EntityId::Node(id),
                ChangeKind::Delete,
                epoch,
                self.next_ts(),
            );
            if let Some(node) = &node {
                describe_deleted_node(&mut event, node);
            }
            self.buffer_event(event);
        }
        deleted
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // Read the edges before they go; a self-loop is both outgoing and
        // incoming but deleted once.
        let mut edges: Vec<Edge> = self
            .inner
            .edges_from(node_id, Direction::Outgoing)
            .into_iter()
            .chain(self.inner.edges_from(node_id, Direction::Incoming))
            .filter_map(|(_, edge)| self.edge_as_seen(edge, None))
            .collect();
        edges.sort_unstable_by_key(|edge| edge.id);
        edges.dedup_by_key(|edge| edge.id);

        self.inner.delete_node_edges(node_id);

        let epoch = self.inner.current_epoch();
        for edge in edges {
            let mut event = make_event(
                EntityId::Edge(edge.id),
                ChangeKind::Delete,
                epoch,
                self.next_ts(),
            );
            describe_deleted_edge(&mut event, &edge);
            self.record_directly(event);
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        let edge = self.edge_as_seen(id, None);
        let deleted = self.inner.delete_edge(id);
        if deleted {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Edge(id),
                ChangeKind::Delete,
                epoch,
                self.next_ts(),
            );
            if let Some(edge) = &edge {
                describe_deleted_edge(&mut event, edge);
            }
            self.record_directly(event);
        }
        deleted
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let edge = self.edge_as_seen(id, Some(transaction_id));
        let deleted = self.inner.delete_edge_versioned(id, epoch, transaction_id);
        if deleted {
            let mut event = make_event(
                EntityId::Edge(id),
                ChangeKind::Delete,
                epoch,
                self.next_ts(),
            );
            if let Some(edge) = &edge {
                describe_deleted_edge(&mut event, edge);
            }
            self.buffer_event(event);
        }
        deleted
    }

    // --- Property mutation ---

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let old_value = self.inner.get_node_property(id, &PropertyKey::new(key));
        self.inner.set_node_property(id, key, value.clone());
        let epoch = self.inner.current_epoch();
        let mut event = make_event(
            EntityId::Node(id),
            ChangeKind::Update,
            epoch,
            self.next_ts(),
        );
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.record_directly(event);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        let old_value = self.inner.get_edge_property(id, &PropertyKey::new(key));
        self.inner.set_edge_property(id, key, value.clone());
        let epoch = self.inner.current_epoch();
        let mut event = make_event(
            EntityId::Edge(id),
            ChangeKind::Update,
            epoch,
            self.next_ts(),
        );
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.record_directly(event);
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let old_value = self.inner.get_node_property(id, &PropertyKey::new(key));
        self.inner
            .set_node_property_versioned(id, key, value.clone(), transaction_id);
        let epoch = self.inner.current_epoch();
        let mut event = make_event(
            EntityId::Node(id),
            ChangeKind::Update,
            epoch,
            self.next_ts(),
        );
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.buffer_event(event);
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let old_value = self.inner.get_edge_property(id, &PropertyKey::new(key));
        self.inner
            .set_edge_property_versioned(id, key, value.clone(), transaction_id);
        let epoch = self.inner.current_epoch();
        let mut event = make_event(
            EntityId::Edge(id),
            ChangeKind::Update,
            epoch,
            self.next_ts(),
        );
        event.before = old_value.map(|v| {
            let mut m = HashMap::new();
            m.insert(key.to_string(), v);
            m
        });
        let mut after = HashMap::new();
        after.insert(key.to_string(), value);
        event.after = Some(after);
        self.buffer_event(event);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let removed = self.inner.remove_node_property(id, key);
        if let Some(ref old_val) = removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Node(id),
                ChangeKind::Update,
                epoch,
                self.next_ts(),
            );
            let mut before = HashMap::new();
            before.insert(key.to_string(), old_val.clone());
            event.before = Some(before);
            self.record_directly(event);
        }
        removed
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        let removed = self.inner.remove_edge_property(id, key);
        if let Some(ref old_val) = removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Edge(id),
                ChangeKind::Update,
                epoch,
                self.next_ts(),
            );
            let mut before = HashMap::new();
            before.insert(key.to_string(), old_val.clone());
            event.before = Some(before);
            self.record_directly(event);
        }
        removed
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let removed = self
            .inner
            .remove_node_property_versioned(id, key, transaction_id);
        if let Some(ref old_val) = removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Node(id),
                ChangeKind::Update,
                epoch,
                self.next_ts(),
            );
            let mut before = HashMap::new();
            before.insert(key.to_string(), old_val.clone());
            event.before = Some(before);
            self.buffer_event(event);
        }
        removed
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        let removed = self
            .inner
            .remove_edge_property_versioned(id, key, transaction_id);
        if let Some(ref old_val) = removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Edge(id),
                ChangeKind::Update,
                epoch,
                self.next_ts(),
            );
            let mut before = HashMap::new();
            before.insert(key.to_string(), old_val.clone());
            event.before = Some(before);
            self.buffer_event(event);
        }
        removed
    }

    // --- Label mutation ---

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let before = self.labels_as_seen(node_id, None);
        let added = self.inner.add_label(node_id, label);
        if added {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Node(node_id),
                ChangeKind::Update,
                epoch,
                self.next_ts(),
            );
            let after = with_label(&before, label);
            describe_label_change(&mut event, before, after);
            self.record_directly(event);
        }
        added
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let before = self.labels_as_seen(node_id, None);
        let removed = self.inner.remove_label(node_id, label);
        if removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Node(node_id),
                ChangeKind::Update,
                epoch,
                self.next_ts(),
            );
            let after = without_label(&before, label);
            describe_label_change(&mut event, before, after);
            self.record_directly(event);
        }
        removed
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let before = self.labels_as_seen(node_id, Some(transaction_id));
        let added = self
            .inner
            .add_label_versioned(node_id, label, transaction_id);
        if added {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Node(node_id),
                ChangeKind::Update,
                epoch,
                self.next_ts(),
            );
            let after = with_label(&before, label);
            describe_label_change(&mut event, before, after);
            self.buffer_event(event);
        }
        added
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        let before = self.labels_as_seen(node_id, Some(transaction_id));
        let removed = self
            .inner
            .remove_label_versioned(node_id, label, transaction_id);
        if removed {
            let epoch = self.inner.current_epoch();
            let mut event = make_event(
                EntityId::Node(node_id),
                ChangeKind::Update,
                epoch,
                self.next_ts(),
            );
            let after = without_label(&before, label);
            describe_label_change(&mut event, before, after);
            self.buffer_event(event);
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_core::graph::lpg::LpgStore;

    /// Creates a `CdcGraphStore` wrapping a fresh `LpgStore`.
    fn sorted(labels: Option<&Vec<String>>) -> Vec<String> {
        let mut labels = labels.cloned().unwrap_or_default();
        labels.sort();
        labels
    }

    fn setup() -> (CdcGraphStore, Arc<CdcLog>) {
        let store = Arc::new(LpgStore::new().unwrap());
        let log = Arc::new(CdcLog::new());
        let cdc = CdcGraphStore::new(
            Arc::clone(&store) as Arc<dyn GraphStoreMut>,
            Arc::clone(&log),
        );
        (cdc, log)
    }

    // ---------------------------------------------------------------
    // Constructor and accessors
    // ---------------------------------------------------------------

    #[test]
    fn new_creates_empty_pending_buffer() {
        let (cdc, _log) = setup();
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn wrap_shares_event_buffer() {
        let store = Arc::new(LpgStore::new().unwrap());
        let log = Arc::new(CdcLog::new());
        let pending = Arc::new(Mutex::new(Vec::<PendingEvent>::new()));
        let cdc = CdcGraphStore::wrap(
            Arc::clone(&store) as Arc<dyn GraphStoreMut>,
            Arc::clone(&log),
            Arc::clone(&pending),
        );
        // Mutation through cdc should write to the shared buffer
        let id = cdc.create_node(&["Person"]);
        // create_node records directly, not into the buffer
        assert!(pending.lock().is_empty());
        // But the log should have the event
        assert_eq!(log.history(EntityId::Node(id)).len(), 1);
    }

    // ---------------------------------------------------------------
    // Read-only delegation (spot checks)
    // ---------------------------------------------------------------

    #[test]
    fn get_node_delegates_to_inner() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        let node = cdc.get_node(id);
        assert!(node.is_some());
        assert!(node.unwrap().labels.iter().any(|l| l.as_str() == "Person"));
    }

    #[test]
    fn get_edge_delegates_to_inner() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["A"]);
        let b = cdc.create_node(&["B"]);
        let eid = cdc.create_edge(a, b, "KNOWS");
        assert!(cdc.get_edge(eid).is_some());
    }

    #[test]
    fn node_count_and_edge_count_delegate() {
        let (cdc, _log) = setup();
        assert_eq!(cdc.node_count(), 0);
        assert_eq!(cdc.edge_count(), 0);
        let a = cdc.create_node(&["A"]);
        let b = cdc.create_node(&["B"]);
        cdc.create_edge(a, b, "E");
        assert_eq!(cdc.node_count(), 2);
        assert_eq!(cdc.edge_count(), 1);
    }

    #[test]
    fn node_ids_and_all_node_ids_delegate() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["X"]);
        assert!(cdc.node_ids().contains(&a));
        assert!(cdc.all_node_ids().contains(&a));
    }

    #[test]
    fn nodes_by_label_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["City"]);
        assert!(cdc.nodes_by_label("City").contains(&a));
        assert!(cdc.nodes_by_label("Unknown").is_empty());
    }

    #[test]
    fn edge_type_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let e = cdc.create_edge(a, b, "LIKES");
        assert_eq!(&*cdc.edge_type(e).unwrap(), "LIKES");
    }

    #[test]
    fn neighbors_and_edges_from_delegate() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        cdc.create_edge(a, b, "E");
        assert!(cdc.neighbors(a, Direction::Outgoing).contains(&b));
        assert!(!cdc.edges_from(a, Direction::Outgoing).is_empty());
    }

    #[test]
    fn degree_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        cdc.create_edge(a, b, "E");
        assert_eq!(cdc.out_degree(a), 1);
        assert_eq!(cdc.in_degree(b), 1);
    }

    #[test]
    fn property_access_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        cdc.set_node_property(a, "name", Value::from("Alix"));
        assert_eq!(
            cdc.get_node_property(a, &PropertyKey::new("name")),
            Some(Value::from("Alix"))
        );
    }

    #[test]
    fn all_labels_and_edge_types_delegate() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["Person"]);
        let b = cdc.create_node(&["City"]);
        cdc.create_edge(a, b, "LIVES_IN");
        let labels = cdc.all_labels();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"City".to_string()));
        let types = cdc.all_edge_types();
        assert!(types.contains(&"LIVES_IN".to_string()));
    }

    #[test]
    fn all_property_keys_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        cdc.set_node_property(a, "colour", Value::from("orange"));
        let keys = cdc.all_property_keys();
        assert!(keys.contains(&"colour".to_string()));
    }

    #[test]
    fn statistics_delegates() {
        let (cdc, _log) = setup();
        let _stats = cdc.statistics();
    }

    #[test]
    fn current_epoch_delegates() {
        let (cdc, _log) = setup();
        let _epoch = cdc.current_epoch();
    }

    #[test]
    fn has_backward_adjacency_delegates() {
        let (cdc, _log) = setup();
        let _ = cdc.has_backward_adjacency();
    }

    #[test]
    fn has_property_index_delegates() {
        let (cdc, _log) = setup();
        assert!(!cdc.has_property_index("nonexistent"));
    }

    #[test]
    fn find_nodes_by_property_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        cdc.set_node_property(a, "x", Value::Int64(42));
        // find_nodes_by_property may or may not use indexes, just verify no panic
        let _found = cdc.find_nodes_by_property("x", &Value::Int64(42));
    }

    #[test]
    fn find_nodes_by_properties_delegates() {
        let (cdc, _log) = setup();
        let _found = cdc.find_nodes_by_properties(&[("x", Value::Int64(1))]);
    }

    #[test]
    fn find_nodes_in_range_delegates() {
        let (cdc, _log) = setup();
        let _found = cdc.find_nodes_in_range(
            "x",
            Some(&Value::Int64(0)),
            Some(&Value::Int64(100)),
            true,
            true,
        );
    }

    #[test]
    fn estimate_label_cardinality_delegates() {
        let (cdc, _log) = setup();
        let _est = cdc.estimate_label_cardinality("Person");
    }

    #[test]
    fn estimate_avg_degree_delegates() {
        let (cdc, _log) = setup();
        let _est = cdc.estimate_avg_degree("KNOWS", true);
    }

    #[test]
    fn property_might_match_delegates() {
        let (cdc, _log) = setup();
        let pk = PropertyKey::new("x");
        let _ = cdc.node_property_might_match(&pk, CompareOp::Eq, &Value::Int64(1));
        let _ = cdc.edge_property_might_match(&pk, CompareOp::Eq, &Value::Int64(1));
    }

    #[test]
    fn visibility_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        let b = cdc.create_node(&[]);
        let e = cdc.create_edge(a, b, "E");
        let epoch = cdc.current_epoch();
        let _ = cdc.is_node_visible_at_epoch(a, epoch);
        let _ = cdc.is_edge_visible_at_epoch(e, epoch);
        let _ = cdc.filter_visible_node_ids(&[a], epoch);
    }

    #[test]
    fn history_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        let b = cdc.create_node(&[]);
        let e = cdc.create_edge(a, b, "E");
        let _ = cdc.get_node_history(a);
        let _ = cdc.get_edge_history(e);
    }

    #[test]
    fn batch_property_access_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["N"]);
        let b = cdc.create_node(&["N"]);
        cdc.set_node_property(a, "x", Value::Int64(1));
        cdc.set_node_property(b, "x", Value::Int64(2));
        let pk = PropertyKey::new("x");
        let batch = cdc.get_node_property_batch(&[a, b], &pk);
        assert_eq!(batch.len(), 2);
        let props = cdc.get_nodes_properties_batch(&[a, b]);
        assert_eq!(props.len(), 2);
        let selective =
            cdc.get_nodes_properties_selective_batch(&[a, b], std::slice::from_ref(&pk));
        assert_eq!(selective.len(), 2);

        let ea = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(ea, "w", Value::Int64(10));
        let edge_sel = cdc.get_edges_properties_selective_batch(&[ea], &[PropertyKey::new("w")]);
        assert_eq!(edge_sel.len(), 1);
    }

    // ---------------------------------------------------------------
    // Direct mutations (non-versioned): record to CdcLog immediately
    // ---------------------------------------------------------------

    #[test]
    fn create_node_records_directly() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person", "Employee"]);
        let events = log.history(EntityId::Node(id));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, ChangeKind::Create);
        assert_eq!(events[0].labels.as_ref().unwrap(), &["Person", "Employee"]);
        // pending buffer should be empty (direct recording)
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn create_edge_records_directly() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&["A"]);
        let b = cdc.create_node(&["B"]);
        let eid = cdc.create_edge(a, b, "KNOWS");
        let events = log.history(EntityId::Edge(eid));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, ChangeKind::Create);
        assert_eq!(events[0].edge_type.as_deref(), Some("KNOWS"));
        assert_eq!(events[0].src_id, Some(a.as_u64()));
        assert_eq!(events[0].dst_id, Some(b.as_u64()));
    }

    #[test]
    fn batch_create_edges_records_directly() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let c = cdc.create_node(&[]);
        let ids = cdc.batch_create_edges(&[(a, b, "X"), (b, c, "Y")]);
        assert_eq!(ids.len(), 2);
        for id in &ids {
            let events = log.history(EntityId::Edge(*id));
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].kind, ChangeKind::Create);
        }
    }

    #[test]
    fn delete_node_records_directly_with_before_props() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["P"]);
        cdc.set_node_property(id, "name", Value::from("Alix"));
        let deleted = cdc.delete_node(id);
        assert!(deleted);
        let events = log.history(EntityId::Node(id));
        // create + update(set) + delete
        let del_event = events
            .iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        let before = del_event.before.as_ref().unwrap();
        assert_eq!(before.get("name"), Some(&Value::from("Alix")));
        assert_eq!(sorted(del_event.labels.as_ref()), ["P"]);
    }

    #[test]
    fn delete_node_no_event_when_not_found() {
        let (cdc, log) = setup();
        let fake_id = NodeId::new(999);
        let deleted = cdc.delete_node(fake_id);
        assert!(!deleted);
        assert!(log.history(EntityId::Node(fake_id)).is_empty());
    }

    #[test]
    fn delete_edge_records_directly_with_before_props() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "weight", Value::Float64(1.5));
        let deleted = cdc.delete_edge(eid);
        assert!(deleted);
        let del_event = log
            .history(EntityId::Edge(eid))
            .into_iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        let before = del_event.before.as_ref().unwrap();
        assert_eq!(before.get("weight"), Some(&Value::Float64(1.5)));
        assert_eq!(del_event.edge_type.as_deref(), Some("E"));
        assert_eq!(del_event.src_id, Some(a.as_u64()));
        assert_eq!(del_event.dst_id, Some(b.as_u64()));
    }

    #[test]
    fn delete_edge_no_event_when_not_found() {
        let (cdc, log) = setup();
        let fake = EdgeId::new(999);
        assert!(!cdc.delete_edge(fake));
        assert!(log.history(EntityId::Edge(fake)).is_empty());
    }

    #[test]
    fn delete_node_edges_records_each_edge() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let c = cdc.create_node(&[]);
        let e1 = cdc.create_edge(a, b, "X");
        let e2 = cdc.create_edge(c, a, "Y");
        cdc.set_edge_property(e1, "p", Value::Int64(1));

        cdc.delete_node_edges(a);

        // Each edge's delete event says what the edge was.
        let deleted = |edge| {
            log.history(EntityId::Edge(edge))
                .into_iter()
                .find(|e| e.kind == ChangeKind::Delete)
                .map(|e| (e.edge_type, e.src_id, e.dst_id, e.before))
        };
        assert_eq!(
            deleted(e1),
            Some((
                Some("X".to_string()),
                Some(a.as_u64()),
                Some(b.as_u64()),
                Some(HashMap::from([("p".to_string(), Value::Int64(1))])),
            ))
        );
        assert_eq!(
            deleted(e2),
            Some((
                Some("Y".to_string()),
                Some(c.as_u64()),
                Some(a.as_u64()),
                None
            ))
        );
    }

    #[test]
    fn delete_node_edges_reports_a_self_loop_once() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let own = cdc.create_edge(a, a, "SELF");

        cdc.delete_node_edges(a);

        let deletes = log
            .history(EntityId::Edge(own))
            .into_iter()
            .filter(|e| e.kind == ChangeKind::Delete)
            .count();
        assert_eq!(deletes, 1);
    }

    #[test]
    fn set_node_property_records_old_and_new() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["N"]);
        cdc.set_node_property(id, "city", Value::from("Amsterdam"));
        cdc.set_node_property(id, "city", Value::from("Berlin"));

        let events = log.history(EntityId::Node(id));
        let updates: Vec<_> = events
            .iter()
            .filter(|e| e.kind == ChangeKind::Update)
            .collect();
        assert_eq!(updates.len(), 2);
        // First update: no before (new property), after = Amsterdam
        assert!(updates[0].before.is_none());
        assert_eq!(
            updates[0].after.as_ref().unwrap().get("city"),
            Some(&Value::from("Amsterdam"))
        );
        // Second update: before = Amsterdam, after = Berlin
        assert_eq!(
            updates[1].before.as_ref().unwrap().get("city"),
            Some(&Value::from("Amsterdam"))
        );
        assert_eq!(
            updates[1].after.as_ref().unwrap().get("city"),
            Some(&Value::from("Berlin"))
        );
    }

    #[test]
    fn set_edge_property_records_old_and_new() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Int64(1));
        cdc.set_edge_property(eid, "w", Value::Int64(2));

        let events = log.history(EntityId::Edge(eid));
        let updates: Vec<_> = events
            .iter()
            .filter(|e| e.kind == ChangeKind::Update)
            .collect();
        assert_eq!(updates.len(), 2);
        assert!(updates[0].before.is_none());
        assert_eq!(
            updates[1].before.as_ref().unwrap().get("w"),
            Some(&Value::Int64(1))
        );
        assert_eq!(
            updates[1].after.as_ref().unwrap().get("w"),
            Some(&Value::Int64(2))
        );
    }

    #[test]
    fn remove_node_property_records_before() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&[]);
        cdc.set_node_property(id, "x", Value::Int64(42));
        let removed = cdc.remove_node_property(id, "x");
        assert_eq!(removed, Some(Value::Int64(42)));

        let events = log.history(EntityId::Node(id));
        let last = events.last().unwrap();
        assert_eq!(last.kind, ChangeKind::Update);
        assert_eq!(
            last.before.as_ref().unwrap().get("x"),
            Some(&Value::Int64(42))
        );
        assert!(last.after.is_none());
    }

    #[test]
    fn remove_node_property_no_event_when_missing() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&[]);
        let removed = cdc.remove_node_property(id, "nope");
        assert!(removed.is_none());
        // Only the Create event, no Update
        let events = log.history(EntityId::Node(id));
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn remove_edge_property_records_before() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Float64(19.88));
        let removed = cdc.remove_edge_property(eid, "w");
        assert_eq!(removed, Some(Value::Float64(19.88)));

        let events = log.history(EntityId::Edge(eid));
        let last = events.last().unwrap();
        assert_eq!(last.kind, ChangeKind::Update);
        assert_eq!(
            last.before.as_ref().unwrap().get("w"),
            Some(&Value::Float64(19.88))
        );
    }

    #[test]
    fn remove_edge_property_no_event_when_missing() {
        let (cdc, log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        let removed = cdc.remove_edge_property(eid, "nope");
        assert!(removed.is_none());
        // Only Create event
        let events = log.history(EntityId::Edge(eid));
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn add_label_records_update() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person"]);
        let added = cdc.add_label(id, "Employee");
        assert!(added);

        let events = log.history(EntityId::Node(id));
        let update = events
            .iter()
            .find(|e| e.kind == ChangeKind::Update)
            .unwrap();
        assert_eq!(sorted(update.labels.as_ref()), ["Employee", "Person"]);
        assert_eq!(sorted(update.before_labels.as_ref()), ["Person"]);
    }

    #[test]
    fn add_label_no_event_when_already_present() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person"]);
        let added = cdc.add_label(id, "Person");
        assert!(!added);
        // Only the Create event
        assert_eq!(log.history(EntityId::Node(id)).len(), 1);
    }

    #[test]
    fn remove_label_records_labels_before_and_after() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person", "Employee"]);
        let removed = cdc.remove_label(id, "Employee");
        assert!(removed);

        let events = log.history(EntityId::Node(id));
        let update = events
            .iter()
            .find(|e| e.kind == ChangeKind::Update)
            .unwrap();
        assert_eq!(sorted(update.labels.as_ref()), ["Person"]);
        assert_eq!(
            sorted(update.before_labels.as_ref()),
            ["Employee", "Person"]
        );
    }

    #[test]
    fn remove_label_no_event_when_missing() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Person"]);
        let removed = cdc.remove_label(id, "Nonexistent");
        assert!(!removed);
        assert_eq!(log.history(EntityId::Node(id)).len(), 1);
    }

    // ---------------------------------------------------------------
    // Versioned mutations: buffer events for transactional flush
    // ---------------------------------------------------------------

    #[test]
    fn create_node_versioned_buffers_event() {
        let (cdc, log) = setup();
        let epoch = EpochId(1);
        let tx = TransactionId::new(1);
        let id = cdc.create_node_versioned(&["Person"], epoch, tx);

        // Event goes to buffer, not the log
        assert!(log.history(EntityId::Node(id)).is_empty());
        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].kind, ChangeKind::Create);
        assert_eq!(pending[0].epoch, EpochId::PENDING);
        assert_eq!(pending[0].labels.as_ref().unwrap(), &["Person"]);
    }

    #[test]
    fn create_edge_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&["A"]);
        let b = cdc.create_node(&["B"]);
        let epoch = EpochId(1);
        let tx = TransactionId::new(1);
        let eid = cdc.create_edge_versioned(a, b, "KNOWS", epoch, tx);

        let pending = cdc.pending_events().lock().clone();
        let edge_events: Vec<_> = pending
            .iter()
            .filter(|e| e.entity_id == EntityId::Edge(eid))
            .collect();
        assert_eq!(edge_events.len(), 1);
        assert_eq!(edge_events[0].kind, ChangeKind::Create);
        assert_eq!(edge_events[0].edge_type.as_deref(), Some("KNOWS"));
        assert_eq!(edge_events[0].src_id, Some(a.as_u64()));
        assert_eq!(edge_events[0].dst_id, Some(b.as_u64()));
    }

    #[test]
    fn delete_node_versioned_buffers_event_with_snapshot() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        cdc.set_node_property(id, "name", Value::from("Alix"));

        let epoch = EpochId(2);
        let tx = TransactionId::new(1);
        let deleted = cdc.delete_node_versioned(id, epoch, tx);
        assert!(deleted);

        let pending = cdc.pending_events().lock().clone();
        let del_event = pending
            .iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        assert_eq!(del_event.epoch, EpochId::PENDING);
        let before = del_event.before.as_ref().unwrap();
        assert_eq!(before.get("name"), Some(&Value::from("Alix")));
        // labels captured
        let labels = del_event.labels.as_ref().unwrap();
        assert!(labels.contains(&"Person".to_string()));
    }

    #[test]
    fn delete_node_versioned_no_buffer_when_not_found() {
        let (cdc, _log) = setup();
        let tx = TransactionId::new(1);
        let deleted = cdc.delete_node_versioned(NodeId::new(999), EpochId(1), tx);
        assert!(!deleted);
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn delete_edge_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Int64(5));

        let tx = TransactionId::new(1);
        let deleted = cdc.delete_edge_versioned(eid, EpochId(2), tx);
        assert!(deleted);

        let pending = cdc.pending_events().lock().clone();
        let del = pending
            .iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        assert_eq!(
            del.before.as_ref().unwrap().get("w"),
            Some(&Value::Int64(5))
        );
        assert_eq!(del.edge_type.as_deref(), Some("E"));
        assert_eq!(
            (del.src_id, del.dst_id),
            (Some(a.as_u64()), Some(b.as_u64()))
        );
    }

    #[test]
    fn delete_edge_versioned_no_buffer_when_not_found() {
        let (cdc, _log) = setup();
        let tx = TransactionId::new(1);
        assert!(!cdc.delete_edge_versioned(EdgeId::new(999), EpochId(1), tx));
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn set_node_property_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["N"]);
        cdc.set_node_property(id, "x", Value::Int64(1));

        let tx = TransactionId::new(1);
        cdc.set_node_property_versioned(id, "x", Value::Int64(2), tx);

        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        let event = &pending[0];
        assert_eq!(event.kind, ChangeKind::Update);
        assert_eq!(
            event.before.as_ref().unwrap().get("x"),
            Some(&Value::Int64(1))
        );
        assert_eq!(
            event.after.as_ref().unwrap().get("x"),
            Some(&Value::Int64(2))
        );
    }

    #[test]
    fn set_edge_property_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Float64(1.0));

        let tx = TransactionId::new(1);
        cdc.set_edge_property_versioned(eid, "w", Value::Float64(2.0), tx);

        let pending = cdc.pending_events().lock().clone();
        let edge_events: Vec<_> = pending
            .iter()
            .filter(|e| e.entity_id == EntityId::Edge(eid))
            .collect();
        assert_eq!(edge_events.len(), 1);
        assert_eq!(
            edge_events[0].before.as_ref().unwrap().get("w"),
            Some(&Value::Float64(1.0))
        );
        assert_eq!(
            edge_events[0].after.as_ref().unwrap().get("w"),
            Some(&Value::Float64(2.0))
        );
    }

    #[test]
    fn remove_node_property_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&[]);
        cdc.set_node_property(id, "x", Value::Int64(42));

        let tx = TransactionId::new(1);
        let removed = cdc.remove_node_property_versioned(id, "x", tx);
        assert_eq!(removed, Some(Value::Int64(42)));

        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].kind, ChangeKind::Update);
        assert_eq!(
            pending[0].before.as_ref().unwrap().get("x"),
            Some(&Value::Int64(42))
        );
        assert!(pending[0].after.is_none());
    }

    #[test]
    fn remove_node_property_versioned_no_event_when_missing() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&[]);
        let tx = TransactionId::new(1);
        let removed = cdc.remove_node_property_versioned(id, "nope", tx);
        assert!(removed.is_none());
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn remove_edge_property_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        cdc.set_edge_property(eid, "w", Value::Int64(7));

        let tx = TransactionId::new(1);
        let removed = cdc.remove_edge_property_versioned(eid, "w", tx);
        assert_eq!(removed, Some(Value::Int64(7)));

        let pending = cdc.pending_events().lock().clone();
        let edge_events: Vec<_> = pending
            .iter()
            .filter(|e| e.entity_id == EntityId::Edge(eid))
            .collect();
        assert_eq!(edge_events.len(), 1);
        assert_eq!(
            edge_events[0].before.as_ref().unwrap().get("w"),
            Some(&Value::Int64(7))
        );
    }

    #[test]
    fn remove_edge_property_versioned_no_event_when_missing() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let eid = cdc.create_edge(a, b, "E");
        let tx = TransactionId::new(1);
        let removed = cdc.remove_edge_property_versioned(eid, "nope", tx);
        assert!(removed.is_none());
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn add_label_versioned_buffers_event() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        let tx = TransactionId::new(1);
        let added = cdc.add_label_versioned(id, "Employee", tx);
        assert!(added);

        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].kind, ChangeKind::Update);
        assert_eq!(sorted(pending[0].labels.as_ref()), ["Employee", "Person"]);
        assert_eq!(sorted(pending[0].before_labels.as_ref()), ["Person"]);
    }

    #[test]
    fn add_label_versioned_no_buffer_when_duplicate() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        let tx = TransactionId::new(1);
        let added = cdc.add_label_versioned(id, "Person", tx);
        assert!(!added);
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn remove_label_versioned_buffers_labels_before_and_after() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person", "Employee"]);
        let tx = TransactionId::new(1);
        let removed = cdc.remove_label_versioned(id, "Employee", tx);
        assert!(removed);

        let pending = cdc.pending_events().lock().clone();
        assert_eq!(pending.len(), 1);
        assert_eq!(sorted(pending[0].labels.as_ref()), ["Person"]);
        assert_eq!(
            sorted(pending[0].before_labels.as_ref()),
            ["Employee", "Person"]
        );
    }

    #[test]
    fn remove_label_versioned_no_buffer_when_missing() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["Person"]);
        let tx = TransactionId::new(1);
        let removed = cdc.remove_label_versioned(id, "Nonexistent", tx);
        assert!(!removed);
        assert!(cdc.pending_events().lock().is_empty());
    }

    // ---------------------------------------------------------------
    // What events describe
    // ---------------------------------------------------------------

    #[test]
    fn a_deleted_node_without_properties_has_labels_and_no_before() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["N"]);
        assert!(cdc.delete_node(id));
        let delete = log
            .history(EntityId::Node(id))
            .into_iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        assert_eq!(delete.before, None);
        assert_eq!(sorted(delete.labels.as_ref()), ["N"]);
    }

    #[test]
    fn a_delete_in_the_creating_transaction_describes_the_node() {
        let (cdc, _log) = setup();
        // Not `new(1)`: that is `TransactionId::SYSTEM`.
        let tx = TransactionId::new(7);
        let epoch = cdc.current_epoch();
        let id = cdc.create_node_versioned(&["Person"], epoch, tx);
        cdc.set_node_property_versioned(id, "name", Value::from("Alix"), tx);
        assert!(cdc.delete_node_versioned(id, epoch, tx));

        let pending = cdc.pending_events().lock().clone();
        let delete = pending
            .iter()
            .find(|e| e.kind == ChangeKind::Delete)
            .unwrap();
        assert_eq!(sorted(delete.labels.as_ref()), ["Person"]);
        assert_eq!(
            delete.before,
            Some(HashMap::from([("name".to_string(), Value::from("Alix"))]))
        );
    }

    #[test]
    fn a_label_change_in_the_creating_transaction_lists_the_labels() {
        let (cdc, _log) = setup();
        // Not `new(1)`: that is `TransactionId::SYSTEM`.
        let tx = TransactionId::new(7);
        let id = cdc.create_node_versioned(&["Person"], cdc.current_epoch(), tx);
        assert!(cdc.add_label_versioned(id, "Admin", tx));

        let pending = cdc.pending_events().lock().clone();
        let change = pending
            .iter()
            .find(|e| e.kind == ChangeKind::Update)
            .unwrap();
        assert_eq!(sorted(change.before_labels.as_ref()), ["Person"]);
        assert_eq!(sorted(change.labels.as_ref()), ["Admin", "Person"]);
    }

    #[test]
    fn next_ts_returns_increasing_timestamps() {
        let (cdc, _log) = setup();
        let t1 = cdc.next_ts();
        let t2 = cdc.next_ts();
        assert!(t2 > t1);
    }

    #[test]
    fn make_event_creates_minimal_event() {
        let event = make_event(
            EntityId::Node(NodeId::new(1)),
            ChangeKind::Create,
            EpochId(5),
            HlcTimestamp::zero(),
        );
        assert_eq!(event.entity_id, EntityId::Node(NodeId::new(1)));
        assert_eq!(event.kind, ChangeKind::Create);
        assert_eq!(event.epoch, EpochId(5));
        assert!(event.before.is_none());
        assert!(event.after.is_none());
        assert!(event.labels.is_none());
        assert!(event.edge_type.is_none());
        assert!(event.src_id.is_none());
        assert!(event.dst_id.is_none());
    }

    // ---------------------------------------------------------------
    // Versioned read delegation (spot checks)
    // ---------------------------------------------------------------

    #[test]
    fn versioned_read_delegates() {
        let (cdc, _log) = setup();
        let id = cdc.create_node(&["N"]);
        let epoch = cdc.current_epoch();
        let tx = TransactionId::new(0);
        // These should delegate without panic
        let _ = cdc.get_node_versioned(id, epoch, tx);
        let _ = cdc.get_node_at_epoch(id, epoch);
        let _ = cdc.is_node_visible_versioned(id, epoch, tx);
        let _ = cdc.filter_visible_node_ids_versioned(&[id], epoch, tx);
    }

    #[test]
    fn versioned_edge_read_delegates() {
        let (cdc, _log) = setup();
        let a = cdc.create_node(&[]);
        let b = cdc.create_node(&[]);
        let e = cdc.create_edge(a, b, "E");
        let epoch = cdc.current_epoch();
        let tx = TransactionId::new(0);
        let _ = cdc.get_edge_versioned(e, epoch, tx);
        let _ = cdc.get_edge_at_epoch(e, epoch);
        let _ = cdc.is_edge_visible_versioned(e, epoch, tx);
    }

    // ---------------------------------------------------------------
    // Edge-direction-specific cases for delete_node_edges
    //
    // The loop walks `outgoing.chain(incoming)`. An isolated node or
    // a node with edges only in one direction must still deliver exactly
    // one Delete per edge, and zero Deletes when there are none.
    // ---------------------------------------------------------------

    #[test]
    fn delete_node_edges_on_isolated_node_emits_no_events() {
        let (cdc, log) = setup();
        let id = cdc.create_node(&["Solo"]);
        let create_count = log.history(EntityId::Node(id)).len();

        cdc.delete_node_edges(id);

        // No edges existed, so no Delete events anywhere in the log
        // should be tagged for a nonexistent edge id, and the node's
        // own history is unchanged.
        assert_eq!(
            log.history(EntityId::Node(id)).len(),
            create_count,
            "delete_node_edges on isolated node must not touch the node's history"
        );
    }

    #[test]
    fn delete_node_edges_with_only_outgoing_edges() {
        let (cdc, log) = setup();
        let src = cdc.create_node(&[]);
        let dst_a = cdc.create_node(&[]);
        let dst_b = cdc.create_node(&[]);
        let e1 = cdc.create_edge(src, dst_a, "OUT1");
        let e2 = cdc.create_edge(src, dst_b, "OUT2");

        cdc.delete_node_edges(src);

        let e1_deletes: Vec<_> = log
            .history(EntityId::Edge(e1))
            .into_iter()
            .filter(|ev| ev.kind == ChangeKind::Delete)
            .collect();
        let e2_deletes: Vec<_> = log
            .history(EntityId::Edge(e2))
            .into_iter()
            .filter(|ev| ev.kind == ChangeKind::Delete)
            .collect();
        assert_eq!(e1_deletes.len(), 1);
        assert_eq!(e2_deletes.len(), 1);
    }

    #[test]
    fn delete_node_edges_with_only_incoming_edges() {
        let (cdc, log) = setup();
        let target = cdc.create_node(&[]);
        let src_a = cdc.create_node(&[]);
        let src_b = cdc.create_node(&[]);
        let e1 = cdc.create_edge(src_a, target, "IN1");
        let e2 = cdc.create_edge(src_b, target, "IN2");

        cdc.delete_node_edges(target);

        for edge_id in [e1, e2] {
            let deletes: Vec<_> = log
                .history(EntityId::Edge(edge_id))
                .into_iter()
                .filter(|ev| ev.kind == ChangeKind::Delete)
                .collect();
            assert_eq!(
                deletes.len(),
                1,
                "incoming-only edge {edge_id:?} should have exactly one Delete event"
            );
        }
    }

    // ---------------------------------------------------------------
    // Batch and labels edge cases
    // ---------------------------------------------------------------

    #[test]
    fn batch_create_edges_empty_slice_records_nothing() {
        let (cdc, _log) = setup();
        let ids = cdc.batch_create_edges(&[]);
        assert!(ids.is_empty());
        // No transactional buffering either (batch uses direct recording).
        assert!(cdc.pending_events().lock().is_empty());
    }

    #[test]
    fn create_node_with_empty_labels_still_populates_labels_field() {
        // Documents a deliberate choice: the labels field on a Create
        // event is Some(empty vec), not None, even with no labels. This
        // matters for downstream consumers that distinguish
        // "missing field" vs "empty list of labels."
        let (cdc, log) = setup();
        let id = cdc.create_node(&[]);
        let events = log.history(EntityId::Node(id));
        assert_eq!(events.len(), 1);
        let labels = events[0]
            .labels
            .as_ref()
            .expect("labels must be Some even with empty input");
        assert!(labels.is_empty());
    }

    // ---------------------------------------------------------------
    // Shared-buffer semantics via wrap()
    //
    // Two CdcGraphStores wrapping the same pending_events Arc<Mutex<Vec>>
    // (e.g., default graph + named graph) must buffer versioned events
    // into the same Vec so the session can flush them atomically.
    // ---------------------------------------------------------------

    #[test]
    fn wrap_routes_versioned_events_to_shared_buffer() {
        let inner = Arc::new(LpgStore::new().unwrap());
        let log = Arc::new(CdcLog::new());
        let shared = Arc::new(Mutex::new(Vec::<PendingEvent>::new()));

        let cdc = CdcGraphStore::wrap(Arc::clone(&inner) as Arc<dyn GraphStoreMut>, log, shared);

        let id = cdc.create_node_versioned(&["P"], EpochId(1), TransactionId::new(1));
        let pending = cdc.pending_events();
        let buf = pending.lock();
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0].entity_id, EntityId::Node(id));
        assert_eq!(buf[0].kind, ChangeKind::Create);
    }

    #[test]
    fn two_wrapped_stores_share_buffer_for_versioned_mutations() {
        let inner_a = Arc::new(LpgStore::new().unwrap());
        let inner_b = Arc::new(LpgStore::new().unwrap());
        let log = Arc::new(CdcLog::new());
        let shared = Arc::new(Mutex::new(Vec::<PendingEvent>::new()));

        let cdc_a = CdcGraphStore::wrap(
            Arc::clone(&inner_a) as Arc<dyn GraphStoreMut>,
            Arc::clone(&log),
            Arc::clone(&shared),
        );
        let cdc_b = CdcGraphStore::wrap(
            Arc::clone(&inner_b) as Arc<dyn GraphStoreMut>,
            Arc::clone(&log),
            Arc::clone(&shared),
        );

        let tx = TransactionId::new(42);
        let _ = cdc_a.create_node_versioned(&["A"], EpochId(1), tx);
        let _ = cdc_b.create_node_versioned(&["B"], EpochId(1), tx);

        // Both events land in the single shared buffer, in order.
        let buf = shared.lock();
        assert_eq!(buf.len(), 2);
        let labels_a = buf[0].labels.as_ref().unwrap();
        let labels_b = buf[1].labels.as_ref().unwrap();
        assert_eq!(labels_a, &["A"]);
        assert_eq!(labels_b, &["B"]);
        // Both use PENDING since the commit epoch hasn't been assigned.
        assert_eq!(buf[0].epoch, EpochId::PENDING);
        assert_eq!(buf[1].epoch, EpochId::PENDING);
    }
}
